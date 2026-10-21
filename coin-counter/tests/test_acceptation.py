"""Portes d'acceptation : « le programme fait-il son travail ? »

Chacun de ces tests **échouait** avant la réparation. Ils sont écrits pour
casser si l'une des régressions d'origine revient, et ne demandent aucune
caméra.

Pourquoi un fichier à part : la suite historique était verte alors que
``--image`` annonçait ``0,00 DH`` sur une photographie de seize pièces. Elle
vérifiait des fonctions internes dans des configurations que la production
n'emprunte jamais (``detecter`` à la résolution native alors que le pipeline
réduit à 640 ; un total comparé à ``in (0, 100)`` ; un ``--image`` dont on
ne lisait que le code de sortie). Ces portes-ci passent par les **chemins
réellement exécutés** et comparent à des vérités connues.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pieces.classification import classifier_pieces, diametre_mesure
from pieces.config import CONFIG_DEFAUT, denominations_actives
from pieces.detection import detecter
from pieces.entites import (
    CERTITUDE_CONFIRME,
    CERTITUDE_TENTATIVE,
    Classification,
    PieceDetectee,
)
from pieces.etalonnage import estimer_echelle_catalogue
from pieces.etat import EtatApplication
from pieces.scenes import PiecePosee, rendre_scene, scene_etalon
from pieces.suivi import Suiveur
from pieces.traitement import inventorier_image, reduire_a_analyse, traiter_image

RACINE = Path(__file__).resolve().parent.parent
PHOTOS = json.loads((RACINE / "donnees" / "photos.json").read_text(encoding="utf-8"))
ANNOTATIONS = json.loads(
    (RACINE / "donnees" / "coins_annotations.json").read_text(encoding="utf-8")
)


def _etat(px_par_mm: float | None = None, centimes: bool = False):
    return EtatApplication(
        config=CONFIG_DEFAUT,
        suiveur=Suiveur(CONFIG_DEFAUT, compter_centimes=centimes),
        px_par_mm=px_par_mm,
    )


# ---------------------------------------------------------------------------
# Détection sur le CHEMIN DE PRODUCTION
# ---------------------------------------------------------------------------


def test_chemin_de_production_detecte_les_16_pieces(coins_jpg):
    """16/16 à l'échelle d'analyse réellement utilisée, pas en natif.

    La porte historique appelait ``detecter`` sur l'image 800×800 d'origine
    et annonçait 16/16, alors que le pipeline réduit d'abord à 640 — où
    seules 14 pièces survivaient. Elle mesurait un chemin que le programme
    n'exécute jamais.
    """
    image = cv2.imread(str(coins_jpg))
    analyse, _ = reduire_a_analyse(image, CONFIG_DEFAUT)
    resultat = detecter(analyse, CONFIG_DEFAUT)
    valides = [p for p in resultat.pieces if p.motif_incertitude is None]
    assert len(valides) == 16


def test_chemin_de_production_rappel_sur_photographies_reelles():
    """Rappel global sur six photographies réelles (46 pièces).

    Mesuré à 35 % avant réparation — deux images rendaient 0 pièce sur fond
    clair — et 87 % après. La porte est posée à 85 %, juste sous la mesure.

    Un état intermédiaire atteignait 91 %, mais par un raccourci qui
    acceptait aussi carrés, croissants et rondelles comme pièces comptables
    (cf. ``test_forme_non_monnayee_jamais_valide``). Le rappel a été ramené à
    87 % en refermant ce trou : sur-compter de l'argent est pire que d'en
    manquer.
    """
    trouvees = attendues = 0
    detail = []
    for scene in PHOTOS["scenes"]:
        image = cv2.imread(str(RACINE / "donnees" / scene["fichier"]))
        assert image is not None, scene["fichier"]
        analyse, _ = reduire_a_analyse(image, CONFIG_DEFAUT)
        resultat = detecter(analyse, CONFIG_DEFAUT)
        valides = sum(1 for p in resultat.pieces if p.motif_incertitude is None)
        trouvees += min(valides, scene["pieces"])
        attendues += scene["pieces"]
        detail.append(f"{scene['fichier']}: {valides}/{scene['pieces']}")
    assert trouvees / attendues >= 0.85, " ; ".join(detail)


@pytest.mark.parametrize("nom", ["noir", "blanc"])
def test_aucune_piece_sur_une_image_unie(nom):
    """Aucun faux positif : le rappel ne doit pas s'acheter en bruit."""
    import numpy as np

    valeur = 0 if nom == "noir" else 255
    image = np.full((480, 640, 3), valeur, dtype=np.uint8)
    resultat = detecter(image, CONFIG_DEFAUT)
    assert [p for p in resultat.pieces if p.motif_incertitude is None] == []


# ---------------------------------------------------------------------------
# Identification et total
# ---------------------------------------------------------------------------


def test_identification_exacte_sur_la_planche_reelle(coins_jpg):
    """Les seize dénominations de ``coins.jpg``, sans aucun étalonnage.

    L'échelle est retrouvée au catalogue ; la vérité vient de l'annotation
    indépendante (``donnees/coins_annotations.json``).
    """
    image = cv2.imread(str(coins_jpg))
    inventaire = inventorier_image(image, CONFIG_DEFAUT, centimes=True)
    assert inventaire.echelle_estimee

    correctes = 0
    attendues = 0
    for piece in inventaire.pieces:
        if piece.motif_incertitude is not None:
            continue
        attendues += 1
        proche = min(
            ANNOTATIONS["pieces"],
            key=lambda a: (a["centre"][0] - piece.centre[0] * inventaire.echelle) ** 2
            + (a["centre"][1] - piece.centre[1] * inventaire.echelle) ** 2,
        )
        obtenu = piece.classification.libelle if piece.classification else None
        correctes += obtenu == proche["denomination"]
    assert attendues == 16
    assert correctes == 16


def test_total_exact_sur_la_planche_reelle(coins_jpg):
    """37,00 DH hors centimes, 37,70 DH avec — deux exemplaires de chaque pièce."""
    image = cv2.imread(str(coins_jpg))
    assert inventorier_image(image, CONFIG_DEFAUT).total_centimes == 3700
    assert (
        inventorier_image(image, CONFIG_DEFAUT, centimes=True).total_centimes == 3770
    )


@pytest.mark.parametrize("px_par_mm", [3.5, 4.5, 5.5])
def test_total_exact_sur_scene_a_echelle_imposee(px_par_mm):
    """Scène synthétique : la vérité est connue **par construction**.

    C'est la seule famille de fixtures où une assertion de comptage est
    métrologiquement valide — une photographie de catalogue porte sa propre
    incertitude de l'ordre de 0,4 mm, du même ordre que la marge de décision.
    """
    scene = scene_etalon(px_par_mm)
    inventaire = inventorier_image(scene.image, CONFIG_DEFAUT)
    assert inventaire.total_centimes == scene.total_sans_centimes
    avec = inventorier_image(scene.image, CONFIG_DEFAUT, centimes=True)
    assert avec.total_centimes == scene.total_centimes


def test_echelle_fournie_est_honoree_et_non_estimee():
    scene = scene_etalon(4.5)
    inventaire = inventorier_image(scene.image, CONFIG_DEFAUT, px_par_mm=4.5)
    assert inventaire.echelle_px_mm == 4.5
    assert inventaire.echelle_estimee is False
    assert inventaire.total_centimes == scene.total_sans_centimes


def test_sans_echelle_et_sans_estimation_aucun_total_invente():
    """Interdire l'estimation doit donner zéro, pas un chiffre fabriqué."""
    scene = scene_etalon(4.5)
    inventaire = inventorier_image(scene.image, CONFIG_DEFAUT, auto_echelle=False)
    assert inventaire.echelle_px_mm is None
    assert inventaire.total_centimes == 0


def test_estimation_refuse_une_scene_degeneree():
    """Des disques tous identiques ne déterminent aucune échelle.

    Chaque dénomination du catalogue explique alors l'ensemble des
    observations : le « consensus » serait parfait et parfaitement arbitraire.
    """
    identiques = [(100.0, "argent", None)] * 6
    assert estimer_echelle_catalogue(identiques, CONFIG_DEFAUT) is None
    assert estimer_echelle_catalogue([], CONFIG_DEFAUT) is None
    assert estimer_echelle_catalogue([(100.0, "argent", None)], CONFIG_DEFAUT) is None


def test_estimation_proche_de_l_echelle_imposee():
    scene = scene_etalon(4.5)
    analyse, _ = reduire_a_analyse(scene.image, CONFIG_DEFAUT)
    resultat = detecter(analyse, CONFIG_DEFAUT)
    classifier_pieces(resultat.pieces, analyse, CONFIG_DEFAUT, None)
    observations = [
        (diametre_mesure(p), p.famille, p.polarite)
        for p in resultat.pieces
        if p.motif_incertitude is None
    ]
    estimation = estimer_echelle_catalogue(observations, CONFIG_DEFAUT)
    assert estimation is not None
    assert abs(estimation.px_par_mm - 4.5) / 4.5 < 0.03


# ---------------------------------------------------------------------------
# Comptage : ce qui ne doit JAMAIS arriver
# ---------------------------------------------------------------------------


def _piece(x: float, valeur: int, certitude: str, rayon: float = 50.0):
    piece = PieceDetectee(
        centre=(x, 100.0), rayon=rayon, aire=3.141592653589793 * rayon * rayon,
        source="contour",
    )
    piece.classification = Classification(valeur, "?", 1.0, 0.1, certitude)
    return piece


def test_une_identification_tentative_n_est_jamais_comptee():
    """Le défaut qui fabriquait 10,00 DH à partir d'une image non étalonnée.

    Le suivi ne lisait pas ``Classification.certitude`` : sa propre phase
    « confirmee » (présence temporelle) suffisait à compter, alors qu'elle ne
    dit rien de la confiance d'identification.
    """
    suiveur = Suiveur(CONFIG_DEFAUT)
    for frame in range(1, 13):
        suiveur.maj([_piece(100.0, 1000, CERTITUDE_TENTATIVE)], frame)
    assert suiveur.total_centimes == 0


def test_une_identification_confirmee_est_comptee_une_seule_fois():
    suiveur = Suiveur(CONFIG_DEFAUT)
    for frame in range(1, 13):
        suiveur.maj([_piece(100.0, 100, CERTITUDE_CONFIRME)], frame)
    assert suiveur.total_centimes == 100
    assert suiveur.par_valeur_total == {100: 1}


def test_une_piece_qui_sort_et_revient_n_est_pas_recomptee():
    suiveur = Suiveur(CONFIG_DEFAUT)
    frame = 1
    for _ in range(5):
        suiveur.maj([_piece(100.0, 100, CERTITUDE_CONFIRME)], frame)
        frame += 1
    for _ in range(8):  # au-delà de max_manques : la piste est purgée
        suiveur.maj([], frame)
        frame += 1
    for _ in range(5):
        suiveur.maj([_piece(100.0, 100, CERTITUDE_CONFIRME)], frame)
        frame += 1
    assert suiveur.total_centimes == 100


def test_une_piece_dont_le_rayon_derive_n_est_pas_recomptee():
    """L'utilisateur rapproche la pièce : l'association refuse, pas le total."""
    suiveur = Suiveur(CONFIG_DEFAUT)
    frame = 1
    for _ in range(5):
        suiveur.maj([_piece(100.0, 100, CERTITUDE_CONFIRME, rayon=50.0)], frame)
        frame += 1
    for _ in range(8):
        suiveur.maj([_piece(100.0, 100, CERTITUDE_CONFIRME, rayon=62.0)], frame)
        frame += 1
    assert suiveur.total_centimes == 100


def test_deux_detections_ne_peuvent_pas_prendre_la_meme_piste():
    suiveur = Suiveur(CONFIG_DEFAUT)
    suiveur.maj([_piece(100.0, 100, CERTITUDE_CONFIRME)], 1)
    etat = suiveur.maj(
        [
            _piece(105.0, 100, CERTITUDE_CONFIRME),
            _piece(108.0, 1000, CERTITUDE_CONFIRME),
        ],
        2,
    )
    assert len({p.id for p in etat.pistes}) == 2


def test_le_total_suivi_est_exact_sur_une_scene_connue():
    """Bout en bout : la scène synthétique, par le chemin webcam."""
    scene = scene_etalon(4.5)
    etat = _etat()
    resultat = None
    for _ in range(12):
        resultat = traiter_image(scene.image, etat)
    assert resultat is not None
    assert resultat.total_centimes == scene.total_sans_centimes
    assert etat.echelle_estimee is True


def test_le_total_suivi_honore_l_option_centimes():
    scene = scene_etalon(4.5)
    etat = _etat(centimes=True)
    for _ in range(12):
        resultat = traiter_image(scene.image, etat)
    assert resultat.total_centimes == scene.total_centimes


def test_deux_sessions_identiques_donnent_le_meme_total():
    scene = scene_etalon(4.5)
    totaux = []
    for _ in range(2):
        etat = _etat()
        for _ in range(10):
            resultat = traiter_image(scene.image, etat)
        totaux.append(resultat.total_centimes)
    assert totaux[0] == totaux[1]


# ---------------------------------------------------------------------------
# Polarité, géométrie des cadres, robustesse d'entrée
# ---------------------------------------------------------------------------


def test_fond_clair_pieces_sombres():
    """Polarité inverse (« negatif ») : le décompte ne doit pas changer."""
    scene = scene_etalon(4.5)
    inverse = 255 - scene.image
    analyse, _ = reduire_a_analyse(inverse, CONFIG_DEFAUT)
    resultat = detecter(analyse, CONFIG_DEFAUT)
    valides = [p for p in resultat.pieces if p.motif_incertitude is None]
    assert len(valides) == len(scene.inventaire)


def test_cadre_16_9_ne_perd_pas_de_piece():
    """Le plafond de rayon ne doit pas dépendre du format du capteur.

    Indexé sur la hauteur, il variait d'un facteur 3,2 entre un cadre carré
    et un 16:9 à champ horizontal identique.
    """
    import numpy as np

    scene = scene_etalon(4.5)
    h, w = scene.image.shape[:2]
    cible_h = round(w * 9 / 16)
    cadre = np.full((cible_h, w, 3), 30, dtype=np.uint8)
    hauteur = min(h, cible_h)
    cadre[:hauteur] = scene.image[:hauteur]
    analyse, _ = reduire_a_analyse(cadre, CONFIG_DEFAUT)
    resultat = detecter(analyse, CONFIG_DEFAUT)
    valides = [p for p in resultat.pieces if p.motif_incertitude is None]
    assert len(valides) >= 6


def test_image_en_niveaux_de_gris_ne_leve_pas():
    """``detecter`` annonçait accepter le gris ; le pipeline levait ensuite."""
    scene = scene_etalon(4.5)
    gris = cv2.cvtColor(scene.image, cv2.COLOR_BGR2GRAY)
    inventaire = inventorier_image(gris, CONFIG_DEFAUT)
    assert inventaire.pieces  # pas d'exception, et des pièces trouvées


def test_scene_vide_ne_compte_rien():
    import numpy as np

    vide = np.full((400, 640, 3), 30, dtype=np.uint8)
    inventaire = inventorier_image(vide, CONFIG_DEFAUT)
    assert inventaire.total_centimes == 0
    assert inventaire.par_valeur == {}


def test_scene_partielle_compte_ce_qu_elle_contient():
    """Une scène quelconque, pas seulement la scène étalon."""
    pieces = [
        PiecePosee("10 DH", 25, 25),
        PiecePosee("10 DH", 60, 25),
        PiecePosee("1 DH", 95, 25),
        PiecePosee("1/2 DH", 125, 25),
    ]
    scene = rendre_scene(pieces, 4.5, (260, 640))
    inventaire = inventorier_image(scene.image, CONFIG_DEFAUT)
    assert inventaire.total_centimes == scene.total_sans_centimes == 2150


# ---------------------------------------------------------------------------
# Cohérence de la configuration
# ---------------------------------------------------------------------------


def test_les_denominations_actives_excluent_les_centimes_par_defaut():
    valeurs = [d.valeur_centimes for d in denominations_actives(CONFIG_DEFAUT)]
    assert 5 not in valeurs and 10 not in valeurs and 20 not in valeurs
    assert 50 in valeurs


# ---------------------------------------------------------------------------
# Formes adverses : ce qui ressemble à une pièce sans en être une
# ---------------------------------------------------------------------------


def _forme(dessin) -> np.ndarray:
    """Scène sombre avec une seule forme métallique dessinée."""
    import numpy as _np

    rng = _np.random.default_rng(3)
    image = _np.full((480, 640, 3), 32, _np.uint8)
    image = _np.clip(
        image.astype(_np.int16) + rng.integers(-4, 5, image.shape), 0, 255
    ).astype(_np.uint8)
    dessin(image)
    return image


def _polygone(cotes: int, rayon: int = 60):
    import numpy as _np

    def dessin(image):
        points = _np.array(
            [
                [320 + rayon * _np.cos(2 * _np.pi * k / cotes),
                 240 + rayon * _np.sin(2 * _np.pi * k / cotes)]
                for k in range(cotes)
            ],
            _np.int32,
        )
        cv2.fillPoly(image, [points], METAL)
    return dessin


METAL = (185, 188, 192)


def _disque(image):
    cv2.circle(image, (320, 240), 60, METAL, -1, cv2.LINE_AA)


def _ellipse(image):
    cv2.ellipse(image, (320, 240), (72, 48), 0, 0, 360, METAL, -1, cv2.LINE_AA)


def _croissant(image):
    cv2.circle(image, (320, 240), 60, METAL, -1, cv2.LINE_AA)
    cv2.circle(image, (345, 240), 48, (32, 32, 32), -1, cv2.LINE_AA)


def _anneau(image):
    cv2.circle(image, (320, 240), 60, METAL, -1, cv2.LINE_AA)
    cv2.circle(image, (320, 240), 34, (32, 32, 32), -1, cv2.LINE_AA)


def _rectangle_arrondi(image):
    cv2.rectangle(image, (265, 185), (375, 295), METAL, -1)
    for cx, cy in ((265, 185), (375, 185), (265, 295), (375, 295)):
        cv2.circle(image, (cx, cy), 18, METAL, -1, cv2.LINE_AA)


def _goutte(image):
    import numpy as _np

    cv2.circle(image, (320, 250), 55, METAL, -1, cv2.LINE_AA)
    cv2.fillPoly(
        image, [_np.array([[285, 220], [355, 220], [320, 160]], _np.int32)], METAL
    )


@pytest.mark.parametrize(
    "nom, dessin",
    [
        ("carré", _polygone(4)),
        ("hexagone", _polygone(6)),
        ("ellipse 3:2", _ellipse),
        ("croissant", _croissant),
        ("anneau (rondelle)", _anneau),
        ("rectangle arrondi", _rectangle_arrondi),
        ("goutte", _goutte),
    ],
)
def test_forme_non_monnayee_jamais_valide(nom, dessin):
    """Aucune de ces formes ne doit ressortir comme pièce comptable.

    Chacune a réellement franchi la chaîne à un moment de la réparation :

    * carré, ellipse, croissant, goutte — le raccourci « cœur unique »
      renvoyait toute silhouette d'aire plausible comme pièce pleinement
      valide, court-circuitant d'un coup circularité, solidité, étendue et
      aspect ;
    * rondelle — son contour EXTERNE est un cercle parfait et ``contourArea``
      mesure la surface enfermée par le bord : le trou était invisible à
      toutes les portes, alors que ``docs/01-portee.md`` promettait une
      « porte d'absence de trou central » qui n'existait pas ;
    * rectangle arrondi — assez adouci par la fermeture morphologique pour
      franchir les portes de forme, il empruntait ensuite ce même raccourci,
      lequel sautait aussi les motifs d'incertitude : la porte d'occlusion ne
      voyait jamais qu'il ne remplit que 52 % de son cercle circonscrit.
    """
    resultat = detecter(_forme(dessin), CONFIG_DEFAUT)
    valides = [p for p in resultat.pieces if p.motif_incertitude is None]
    assert valides == [], f"{nom} accepté comme pièce : {valides}"


def test_disque_temoin_reste_detecte():
    """Contrôle : la batterie ci-dessus ne doit pas rejeter un vrai disque."""
    resultat = detecter(_forme(_disque), CONFIG_DEFAUT)
    assert len([p for p in resultat.pieces if p.motif_incertitude is None]) == 1


# ---------------------------------------------------------------------------
# Portes révélées manquantes par un test de mutation
# ---------------------------------------------------------------------------


def test_egalite_de_voix_ne_tranche_pas():
    """Une égalité ne doit pas geler une dénomination au hasard.

    Le comptage se déclenche dès que ``piste.valeur`` est renseignée : une
    égalité fortuite sur les deux premières frames suffisait donc à figer une
    valeur fausse avant que les voix ne s'accumulent. Mesuré : une pièce
    majoritairement lue 1000 était comptée 100.
    """
    suiveur = Suiveur(CONFIG_DEFAUT)
    # alternance stricte : aucune valeur n'est jamais majoritaire seule
    for frame, valeur in enumerate([100, 1000] * 6, start=1):
        suiveur.maj([_piece(100.0, valeur, CERTITUDE_CONFIRME)], frame)
    assert suiveur.total_centimes == 0, (
        "une alternance parfaite ne doit jamais produire de total"
    )


def test_majorite_l_emporte_malgre_une_egalite_initiale():
    """…mais une vraie majorité doit finir par être retenue."""
    suiveur = Suiveur(CONFIG_DEFAUT)
    sequence = [100, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    for frame, valeur in enumerate(sequence, start=1):
        suiveur.maj([_piece(100.0, valeur, CERTITUDE_CONFIRME)], frame)
    assert suiveur.total_centimes == 1000


def test_famille_illisible_ne_confirme_jamais():
    """Sans veto de couleur, la seule taille ne doit pas trancher.

    Le cas qui mord réellement avec le catalogue corrigé : un **vrai 5 DH**
    de 26,2 mm mesuré 26,7 mm (0,5 mm d'erreur, plausible à l'échelle
    d'analyse). La taille seule le confirmerait « 10 DH » — une valeur
    fausse. Avec le veto, on refuse de conclure.
    """
    from pieces.classification import _classer_taille
    from pieces.entites import (
        FAMILLE_BIMETALLIQUE,
        FAMILLE_INCONNUE,
        POLARITE_CENTRE_OR,
    )

    def piece_de(mm: float, px_par_mm: float = 4.0) -> PieceDetectee:
        diametre = mm * px_par_mm
        return PieceDetectee(
            centre=(80.0, 80.0), rayon=diametre / 2.0,
            aire=3.141592653589793 * (diametre / 2.0) ** 2, source="contour",
        )

    # couleur lisible : la bonne famille tranche correctement
    lisible = _classer_taille(
        piece_de(26.7), FAMILLE_BIMETALLIQUE, POLARITE_CENTRE_OR, CONFIG_DEFAUT, 4.0
    )
    assert lisible.libelle == "5 DH"

    # couleur illisible : aucune valeur, surtout pas « 10 DH »
    illisible = _classer_taille(
        piece_de(26.7), FAMILLE_INCONNUE, None, CONFIG_DEFAUT, 4.0
    )
    assert illisible.valeur_centimes is None
    assert illisible.certitude != CERTITUDE_CONFIRME


def test_une_frame_etroite_n_est_jamais_agrandie():
    """``echelle ≥ 1`` est un invariant, pas un vœu.

    Agrandir n'invente aucune information, et ``INTER_AREA`` — un filtre de
    réduction — y est de toute façon inadapté. L'ancienne version rendait
    0,5 pour une frame de 320 px, contredisant son propre contrat.
    """
    petite = np.full((240, 320, 3), 30, dtype=np.uint8)
    analyse, echelle = reduire_a_analyse(petite, CONFIG_DEFAUT)
    assert echelle >= 1.0
    assert analyse.shape[:2] == (240, 320)

    grande = np.full((720, 1280, 3), 30, dtype=np.uint8)
    analyse, echelle = reduire_a_analyse(grande, CONFIG_DEFAUT)
    assert echelle == pytest.approx(2.0)
    assert analyse.shape[1] == CONFIG_DEFAUT.largeur_analyse
