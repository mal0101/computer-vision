"""Tests de M2 : classification (couleur → veto, taille → valeur) et
étalonnage (échelle de session, contrôle de cohérence).

Les images synthétiques sont générées en **ondes HSV déterministes** puis
converties en BGR : la classification ne dépend d'aucune caméra réelle.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pieces.classification import classifier_pieces
from pieces.config import CONFIG_DEFAUT
from pieces.detection import detecter
from pieces.entites import (
    CERTITUDE_CONFIRME,
    CERTITUDE_INCONNU,
    CERTITUDE_TENTATIVE,
    Classification,
    PieceDetectee,
)
from pieces.erreurs import ErreurEtalonnage
from pieces.etalonnage import (
    ObservateurEtalonnage,
    controle_coherence,
    echelle_depuis_carte,
    etalonner_par_piece,
    valider_echelle,
)

SEUIL_SAT_METAL = 60.0  # reflet dans classification.py


def _hsv_or() -> tuple[int, int, int]:
    return 20, 180, 200  # hue OpenCV 20 → 40° réel : jaune laiton


def _hsv_argent() -> tuple[int, int, int]:
    return 120, 25, 190  # gris métallique : saturation faible


def _image_piece(
    centre_color: tuple[int, int, int] | None,
    couronne_color: tuple[int, int, int] | None,
    rayon: float = 60.0,
    centre: tuple[int, int] = (320, 240),
    taille: tuple[int, int] = (480, 640),
) -> np.ndarray:
    """Image BGR : fond sombre + pièce bimétallique/plate en HSV→BGR."""
    h, w = taille
    image = np.full((h, w, 3), (30, 30, 30), dtype=np.uint8)
    if couronne_color is not None:
        cour_bgr = cv2.cvtColor(
            np.array([[couronne_color]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        )[0, 0]
        cv2.circle(image, centre, int(rayon), tuple(cour_bgr.tolist()), -1, cv2.LINE_AA)
    if centre_color is not None:
        cen_bgr = cv2.cvtColor(
            np.array([[centre_color]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        )[0, 0]
        cv2.circle(
            image, centre, int(rayon * 0.45), tuple(cen_bgr.tolist()), -1, cv2.LINE_AA
        )
    return image


def _piece(rayon: float, centre: tuple[float, float] = (320.0, 240.0)) -> PieceDetectee:
    return PieceDetectee(
        centre=centre, rayon=rayon, aire=np.pi * rayon * rayon, source="contour"
    )


# ---------------------------------------------------------------------------
# Couleur : veto de famille
# ---------------------------------------------------------------------------


def test_piece_argent_famille_argent():
    image = _image_piece(_hsv_argent(), None)
    piece = _piece(60)
    classifier_pieces([piece], image, CONFIG_DEFAUT, px_par_mm=None)
    assert piece.famille == "argent"
    assert piece.polarite is None


def test_piece_or_famille_or():
    image = _image_piece(_hsv_or(), None)
    piece = _piece(60)
    classifier_pieces([piece], image, CONFIG_DEFAUT, px_par_mm=None)
    assert piece.famille == "or"
    assert piece.polarite is None


def test_piece_bimetallique_centre_or():
    image = _image_piece(_hsv_or(), _hsv_argent())
    piece = _piece(60)
    classifier_pieces([piece], image, CONFIG_DEFAUT, px_par_mm=None)
    assert piece.famille == "bimetallique"
    assert piece.polarite == "centre_or"


def test_piece_bimetallique_centre_argent():
    image = _image_piece(_hsv_argent(), _hsv_or())
    piece = _piece(60)
    classifier_pieces([piece], image, CONFIG_DEFAUT, px_par_mm=None)
    assert piece.famille == "bimetallique"
    assert piece.polarite == "centre_argent"


# ---------------------------------------------------------------------------
# Sans échelle : au plus « tentative »
# ---------------------------------------------------------------------------


def test_sans_echelle_or_ambigu_inconnu():
    image = _image_piece(_hsv_or(), None)
    piece = _piece(60)
    classifier_pieces([piece], image, CONFIG_DEFAUT, px_par_mm=None)
    assert piece.famille == "or"
    assert piece.classification.valeur_centimes is None
    # sans échelle la certitude plafonne à « tentative » (jamais confirme)
    assert piece.classification.certitude == CERTITUDE_TENTATIVE


def test_sans_echelle_bimetallique_5dh_tentative():
    image = _image_piece(_hsv_or(), _hsv_argent())
    piece = _piece(60)
    classifier_pieces([piece], image, CONFIG_DEFAUT, px_par_mm=None)
    assert piece.classification.valeur_centimes == 500
    assert piece.classification.certitude == CERTITUDE_TENTATIVE


def test_sans_echelle_bimetallique_10dh_tentative():
    image = _image_piece(_hsv_argent(), _hsv_or())
    piece = _piece(60)
    classifier_pieces([piece], image, CONFIG_DEFAUT, px_par_mm=None)
    assert piece.classification.valeur_centimes == 1000
    assert piece.classification.certitude == CERTITUDE_TENTATIVE


# ---------------------------------------------------------------------------
# Avec échelle : taille
# ---------------------------------------------------------------------------

PX_MM = 4.0  # 24 mm → 96 px de diamètre


def _classer(rayon: float, image: np.ndarray, px_par_mm: float = PX_MM) -> Classification:
    piece = _piece(rayon)
    classifier_pieces([piece], image, CONFIG_DEFAUT, px_par_mm=px_par_mm)
    assert piece.classification is not None
    return piece.classification


def test_taille_1dh_argent_confirme():
    # diamètre 96 px / 4 = 24,00 mm → 1 DH (variante 24.0, marge 2.0 vs 2 DH)
    image = _image_piece(_hsv_argent(), None)
    resultat = _classer(48.0, image)
    assert resultat.valeur_centimes == 100
    assert resultat.certitude == CERTITUDE_CONFIRME
    assert resultat.ecart_mm == pytest.approx(0.0, abs=0.01)


def test_taille_20c_or_identifie_meme_sans_centimes():
    # 23,00 mm → 20c (variante 23.0) : l'identification est exhaustive, le
    # filtre « centimes » n'agit qu'au comptage (M3)
    image = _image_piece(_hsv_or(), None)
    resultat = _classer((23.0 * PX_MM) / 2, image)
    assert resultat.valeur_centimes == 20
    assert resultat.certitude == CERTITUDE_CONFIRME


def test_taille_2dh_argent_contre_1dh():
    # 26,00 mm → 2 DH, à 2 mm du 1 DH : la marge (2.0 > marge_classif 0.20)
    image = _image_piece(_hsv_argent(), None)
    resultat = _classer((26.0 * PX_MM) / 2, image)
    assert resultat.valeur_centimes == 200
    assert resultat.certitude == CERTITUDE_CONFIRME


def test_taille_ambiguë_marge_insuffisante_inconnu():
    # 25,00 mm : à 1 mm de 1 DH (24) et de 2 DH (26) → marge nulle → UNKNOWN
    image = _image_piece(_hsv_argent(), None)
    resultat = _classer((25.0 * PX_MM) / 2, image)
    assert resultat.valeur_centimes is None
    assert resultat.certitude == CERTITUDE_INCONNU


def test_taille_hors_portee_motif():
    # 33 mm > portee_mm[30.5] → motif « hors_portee », non comptée
    image = _image_piece(_hsv_argent(), None)
    piece = _piece((33.0 * PX_MM) / 2)
    classifier_pieces([piece], image, CONFIG_DEFAUT, px_par_mm=PX_MM)
    assert piece.motif_incertitude == "hors_portee"
    assert not piece.est_comptable
    assert piece.classification.certitude == CERTITUDE_INCONNU


def test_taille_bimetallique_5dh_confirme():
    # 25,00 mm = variante 5 DH (25.0) : candidates bimétalliques 5/10 DH,
    # la polarité centre_or a déjà ciblé 5 DH → confirmé
    image = _image_piece(_hsv_or(), _hsv_argent())
    resultat = _classer((25.0 * PX_MM) / 2, image)
    assert resultat.valeur_centimes == 500
    assert resultat.certitude == CERTITUDE_CONFIRME


# ---------------------------------------------------------------------------
# Comportements transverses
# ---------------------------------------------------------------------------


def test_piece_incertaine_nonte_taillée():
    image = _image_piece(_hsv_or(), None, rayon=60)
    piece = _piece(60)
    piece.motif_incertitude = "eclat"
    classifier_pieces([piece], image, CONFIG_DEFAUT, px_par_mm=PX_MM)
    assert piece.famille is None
    assert piece.classification is None


def test_determinisme_classification():
    image = _image_piece(_hsv_argent(), None)
    premier = [PieceDetectee(**vars(_piece(48.0)))]
    second = [PieceDetectee(**vars(_piece(48.0)))]
    classifier_pieces(premier, image, CONFIG_DEFAUT, px_par_mm=PX_MM)
    classifier_pieces(second, image, CONFIG_DEFAUT, px_par_mm=PX_MM)
    assert vars(premier[0]) == vars(second[0])


def test_liste_vide_classifie_immediatement():
    assert classifier_pieces([], np.zeros((100, 100, 3), np.uint8), CONFIG_DEFAUT, None) == []


def test_piece_sombre_famille_inconnue():
    image = _image_piece((0, 0, 25), None)  # presque noire
    piece = _piece(60)
    classifier_pieces([piece], image, CONFIG_DEFAUT, px_par_mm=None)
    assert piece.famille == "inconnue"
    assert piece.classification.valeur_centimes is None


# ---------------------------------------------------------------------------
# Étalonnage : observateur de session
# ---------------------------------------------------------------------------


def test_observateur_stable_apres_huit_frames():
    obs = ObservateurEtalonnage()
    assert not obs.est_stable()
    for _ in range(7):
        obs.enregistrer(48.0)
    assert not obs.est_stable()  # 7 frames seulement
    obs.enregistrer(48.0)
    assert obs.est_stable()
    assert obs.nb_frames == 8
    assert obs.etendue == pytest.approx(0.0)


def test_observateur_tremblote_instable():
    obs = ObservateurEtalonnage()
    for i in range(8):
        obs.enregistrer(48.0 + (1.2 if i % 2 else 0.0))
    assert not obs.est_stable()  # étendue 1.2 px > tolérance 1 px
    obs.reinitialiser()
    assert obs.nb_frames == 0
    assert obs.rayon_median is None


def test_observateur_rejette_rayon_invalide():
    obs = ObservateurEtalonnage()
    with pytest.raises(ErreurEtalonnage):
        obs.enregistrer(-3.0)
    with pytest.raises(ErreurEtalonnage):
        obs.enregistrer(0.0)


def test_etalonner_par_piece():
    obs = ObservateurEtalonnage()
    for _ in range(8):
        obs.enregistrer(48.0)
    px_mm = etalonner_par_piece(obs, 24.0)  # pièce d'un dirham (connue)
    assert px_mm == pytest.approx(4.0)


def test_etalonner_tant_que_instable_retourne_none():
    obs = ObservateurEtalonnage()
    for _ in range(3):
        obs.enregistrer(48.0)
    assert etalonner_par_piece(obs, 24.0) is None


def test_echelle_depuis_carte_bancaire():
    # 342,4 px pour 85,60 mm → 4 px/mm
    px_mm = echelle_depuis_carte(342.4)
    assert px_mm == pytest.approx(4.0)


def test_echelle_depuis_carte_invalide():
    with pytest.raises(ErreurEtalonnage):
        echelle_depuis_carte(0.0)


def test_valider_echelle_plage():
    assert valider_echelle(CONFIG_DEFAUT, 4.0) == pytest.approx(4.0)
    with pytest.raises(ErreurEtalonnage):
        valider_echelle(CONFIG_DEFAUT, 0.5)  # < px_mm_min = 2.0
    with pytest.raises(ErreurEtalonnage):
        valider_echelle(CONFIG_DEFAUT, 1e6)
    with pytest.raises(ErreurEtalonnage):
        valider_echelle(CONFIG_DEFAUT, -2.0)


def test_controle_coherence_ok():
    assert controle_coherence(4.0, 96.0, 24.0) == pytest.approx(4.0)


def test_controle_coherence_echoue():
    with pytest.raises(ErreurEtalonnage):
        controle_coherence(4.0, 120.0, 24.0)  # 30 mm au lieu de 24 : écart 25 %


def test_controle_coherence_tolerance_personnalisee():
    # 1 % d'écart accepté avec une tolérance de 2 %
    assert controle_coherence(4.0, 95.04, 24.0, tolerance=0.02)
    with pytest.raises(ErreurEtalonnage):
        controle_coherence(4.0, 95.04, 24.0, tolerance=0.005)


# ---------------------------------------------------------------------------
# Intégration : détection (M1) → classification (M2) sur une scène complète
# ---------------------------------------------------------------------------

_ETALON = 4.0  # px/mm (§ échelle calibrée sur une pièce d'un dirham)


def test_chaine_detection_classification_quatre_pieces():
    """Une scène de 4 pièces (argent, or, bimétallique 5/10 DH) aboutit à 4
    classifications confirmées avec les bonnes valeurs."""
    rng = np.random.default_rng(20261006)
    image = np.full((480, 640, 3), 34, dtype=np.uint8)
    image = np.clip(
        image.astype(np.int16) + rng.integers(-4, 5, image.shape), 0, 255
    ).astype(np.uint8)
    bgr = {}
    for nom, hsv in (
        ("argent", _hsv_argent()),
        ("or", _hsv_or()),
    ):
        bgr[nom] = cv2.cvtColor(
            np.array([[hsv]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        )[0, 0].tolist()
    centres = [(110, 130), (420, 100), (160, 340), (500, 330)]
    # (diamètre mm, famile, couleur centre, couleur couronne)
    spec = [
        (24.0, "argent", bgr["argent"], bgr["argent"]),
        (23.0, "or", bgr["or"], bgr["or"]),
        (25.0, "bimetallique", bgr["or"], bgr["argent"]),
        (28.0, "bimetallique", bgr["argent"], bgr["or"]),
    ]
    attendus = {100, 20, 500, 1000}
    for (cx, cy), (diam, _, cen, cour) in zip(centres, spec, strict=True):
        cv2.circle(image, (cx, cy), int(round(_ETALON * diam / 2)), tuple(cour), -1, cv2.LINE_AA)
        cv2.circle(
            image, (cx, cy), int(round(_ETALON * diam / 2 * 0.45)), tuple(cen), -1, cv2.LINE_AA
        )
    resultat = detecter(image, CONFIG_DEFAUT)
    classifier_pieces(resultat.pieces, image, CONFIG_DEFAUT, px_par_mm=_ETALON)
    valides = [p for p in resultat.pieces if p.motif_incertitude is None]
    assert len(valides) == 4
    valeurs = {p.classification.valeur_centimes for p in valides}
    assert valeurs == attendus
    assert all(p.classification.certitude == CERTITUDE_CONFIRME for p in valides)
    familles = {p.famille for p in valides}
    assert familles == {"argent", "or", "bimetallique"}


def test_chain_deterministe_complete():
    rng = np.random.default_rng(20261006)
    image = np.full((480, 640, 3), 34, dtype=np.uint8)
    image = np.clip(
        image.astype(np.int16) + rng.integers(-4, 5, image.shape), 0, 255
    ).astype(np.uint8)
    bgr_silver = cv2.cvtColor(
        np.array([[_hsv_argent()]], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0, 0].tolist()
    cv2.circle(image, (200, 240), 48, tuple(bgr_silver), -1, cv2.LINE_AA)
    premier = detecter(image, CONFIG_DEFAUT)
    second = detecter(image, CONFIG_DEFAUT)
    classifier_pieces(premier.pieces, image, CONFIG_DEFAUT, px_par_mm=_ETALON)
    classifier_pieces(second.pieces, image, CONFIG_DEFAUT, px_par_mm=_ETALON)
    premiers = [vars(p) for p in premier.pieces]
    seconds = [vars(p) for p in second.pieces]
    assert premiers == seconds


def test_classification_sur_coins_jpg_ne_crashe_pas(coins_jpg):
    """Régression photographique : la chaîne complète démarre et est stable."""

    image = cv2.imread(str(coins_jpg))
    resultat = detecter(image, CONFIG_DEFAUT)
    classifier_pieces(resultat.pieces, image, CONFIG_DEFAUT, px_par_mm=None)
    valides = [p for p in resultat.pieces if p.motif_incertitude is None]
    assert len(valides) == 16
    assert all(p.classification is not None for p in valides)
    # hors portée inconnue reste sur les familles informées ou inconnues
    familles = {p.famille for p in valides}
    assert familles <= {"or", "argent", "bimetallique", "inconnue"}
