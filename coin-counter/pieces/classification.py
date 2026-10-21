"""Classification (M2) : couleur d'abord (veto), taille ensuite.

Ordre de décision, identifié dans ``docs/02-classification.md`` :

1. **Famille par la couleur** — décidée par la **saturation**, confirmée (et
   jamais décidée) par la teinte. Métal coloré (laiton, bronze
   d'aluminium) au-dessus de ``SEUIL_SAT_COLORE``, métal clair
   (cupronickel, acier nickelé) en dessous. Le bimétallique se reconnaît au
   **contraste de saturation** entre noyau et couronne
   (``SEUIL_CONTRASTE_BIMETAL``).

   Pourquoi pas la teinte : mesurée sur les pièces réelles, elle est
   dégénérée — or comme argent tombent entre 4 et 28 — parce que la teinte
   d'une surface peu chromatique est numériquement instable. La saturation,
   elle, sépare franchement (or 78–95, argent 3–15). L'ancienne règle, qui
   interrogeait la teinte en premier, obtenait 11/14 ; celle-ci 16/16.

2. **Polarité** — le bimétallique est désambiguïsé par celle des deux zones
   qui est le laiton (5 DH = noyau or / couronne argent, 10 DH = le
   contraire), une décision *pure couleur* utilisable même sans échelle.

3. **Taille** — avec ``px_par_mm``, le diamètre mesuré (``diametre_mesure``,
   équivalent en aire) est comparé aux variantes de chaque dénomination ; il
   faut ``marge ≥ config.marge_classif`` contre la deuxième meilleure, sinon
   UNKNOWN. Hors ``config.portee_mm`` → ``motif_incertitude = "hors_portee"``
   (jamais comptée).

Une famille **illisible** ne confirme jamais : le veto de couleur est
précisément ce qui sépare les diamètres voisins, et sans lui la seule taille
confirme une dénomination *fausse* 40 % du temps (balayage mesuré au pas de
0,1 mm). Une valeur fausse est pire qu'une absence de valeur.

Sans échelle (``px_par_mm is None``), aucune combinaison famille+polarité ne
peut trancher sauf le bimétallique qui n'a qu'un candidat ; le résultat reste
au plus **tentative** (jamais ``confirme``) — et le suivi refuse de compter
une ``tentative``.

Les pièces déjà marquées d'un ``motif_incertitude`` ne sont pas classifiées :
elles passent telles quelles (elles ne sont comptées nulle part).

Convention : les montants sont des **centimes entiers**.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .config import Config, Denomination
from .entites import (
    CERTITUDE_CONFIRME,
    CERTITUDE_INCONNU,
    CERTITUDE_TENTATIVE,
    FAMILLE_ARGENT,
    FAMILLE_BIMETALLIQUE,
    FAMILLE_INCONNUE,
    FAMILLE_OR,
    POLARITE_CENTRE_ARGENT,
    POLARITE_CENTRE_OR,
    Classification,
    PieceDetectee,
)

__all__ = ["classifier_pieces"]

#: rayon du disque central échantillonné (fraction du rayon de la pièce)
RAPPORT_CENTRE = 0.45
#: anneau « couronne » échantillonné (fractions interne / externe)
COURONNE_INT = 0.5
COURONNE_EXT = 0.9
#: contraste de SATURATION minimal entre noyau et couronne pour conclure
#: « bimétallique ». Mesuré sur les quatre pièces bimétalliques réelles de
#: coins.jpg : 31, 33, 56 et 58 — le plancher à 20 laisse une marge nette,
#: alors que les pièces monométalliques plafonnent à 4 (noyau contre
#: couronne du même métal). Remplace un écart de TEINTE de 14, qui ratait
#: un 10 DH sur deux (écart mesuré 9,6, la teinte d'une zone argentée étant
#: du bruit).
SEUIL_CONTRASTE_BIMETAL = 20.0
#: fraction minimale de pixels « de la couleur de la zone » ; en-dessous,
#: la zone est trop gravée / éclatée pour conclure
SEUIL_COUVERTURE = 0.30
#: largeur de fenêtre de couleur autour de la teinte médiane d'une zone
#: (extensibilité de la « couverture »)
ECART_HUE_PEUPLER = 20.0
#: saturation ≥ ce seuil : zone de métal coloré (laiton, bronze
#: d'aluminium) ; en dessous : métal clair (cupronickel, acier nickelé).
#: Mesuré sur coins.jpg : laiton 39 à 95, cupronickel 3 à 15. Le plancher
#: se pose à 30, au milieu d'un intervalle vide de 24 points. L'ancienne
#: valeur de 60 coupait l'intervalle du laiton en deux et faisait passer les
#: noyaux de 5 DH (39 et 47) pour de l'argent.
SEUIL_SAT_COLORE = 30.0
#: valeur V < ce seuil : pièce trop sombre pour une lecture de couleur
SEUIL_VALEUR_SOMBRE = 45.0
#: bande de teinte « or » (hue OpenCV : jaune-orangé des dinars marocains)
BANDE_OR_MIN = 12.0
BANDE_OR_MAX = 45.0
#: au-delà de cet écart (mm) à la variante la plus proche : inconnu
ECART_MM_MAX = 1.5


def _moyenne_circulaire_teinte(teintes: np.ndarray) -> float:
    """Moyenne circulaire d'une série de teintes (0..180), insensible au
    repliement grâce au doublement d'angle."""
    if teintes.size == 0:
        return 0.0
    theta = np.deg2rad(2.0 * teintes)
    return float(np.rad2deg(0.5 * np.arctan2(np.sin(theta).sum(), np.cos(theta).sum())) % 180.0)


def _distance_teinte(a: float, b: float | np.ndarray) -> float | np.ndarray:
    d = np.abs(np.asarray(a) - np.asarray(b)) % 180.0
    return np.minimum(d, 180.0 - d)


def _couleur_region(
    hsv: cv2.typing.MatLike,
    centre: tuple[float, float],
    rayon: float,
    rayon_int: float,
    rayon_ext: float,
) -> tuple[float, float, float, float]:
    """(teinte, saturation, valeur, couverture) de la zone annulaire.

    La couverture est la fraction des pixels de la zone à moins de
    ``ECART_HUE_PEUPLER`` de la teinte moyenne : elle mesure l'homogénéité
    de la couleur. Elle n'a de sens que sur une zone réellement colorée :
    la teinte d'une surface peu saturée est un bruit numérique (cf.
    ``_famille_piece``).

    Le masque est construit sur la **boîte englobante** de la zone, pas sur
    l'image entière. Deux ``np.ogrid`` pleine image par pièce coûtaient
    22,3 ms pour 14 pièces en 640×640 — soit 95 % du coût de la
    classification et quatre fois celui de la détection ; la version locale
    rend exactement les mêmes valeurs en 5,9 ms.
    """
    hauteur, largeur = hsv.shape[:2]
    portee = rayon_ext * rayon
    x0 = max(0, int(np.floor(centre[0] - portee)))
    x1 = min(largeur, int(np.ceil(centre[0] + portee)) + 1)
    y0 = max(0, int(np.floor(centre[1] - portee)))
    y1 = min(hauteur, int(np.ceil(centre[1] + portee)) + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0, 0.0, 0.0, 0.0
    vignette = hsv[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d2 = (yy - centre[1]) ** 2 + (xx - centre[0]) ** 2
    zone = d2 <= portee**2
    if rayon_int > 0.0:
        zone &= d2 >= (rayon_int * rayon) ** 2
    teintes = vignette[:, :, 0][zone].astype(np.float64)
    if teintes.size < 32:
        return 0.0, 0.0, 0.0, 0.0
    hue = _moyenne_circulaire_teinte(teintes)
    couv = float(
        np.mean(_distance_teinte(hue, np.asarray(teintes)) <= ECART_HUE_PEUPLER)
    )
    sat = float(np.median(vignette[:, :, 1][zone]))
    val = float(np.median(vignette[:, :, 2][zone]))
    return hue, sat, val, couv


def _dans_bande_or(teinte: float) -> bool:
    return BANDE_OR_MIN <= teinte <= BANDE_OR_MAX


def _est_laiton(hue: float, sat: float) -> bool:
    """Zone métallique colorée *et* dans la bande chaude du laiton.

    La teinte n'intervient qu'ici, en **confirmation** : elle empêche un
    objet franchement saturé mais bleu ou vert d'être pris pour de l'or. Elle
    ne peut jamais, à elle seule, conclure — voir ``_famille_piece``.
    """
    return sat >= SEUIL_SAT_COLORE and _dans_bande_or(hue)


def _famille_piece(
    hue_centre: float, sat_centre: float, val_centre: float, couv_centre: float,
    hue_couronne: float, sat_couronne: float, val_couronne: float,
    couv_couronne: float,
) -> tuple[str, str | None]:
    """Famille + polarité depuis les statistiques de couleur.

    **La saturation décide, la teinte confirme.** Mesuré sur les quatorze
    pièces réelles de ``coins.jpg`` : la teinte est dégénérée — toutes les
    pièces, or comme argent, tombent entre 4 et 28 (la seule exception, un
    ½ DH à 126, est du bruit pur : sa saturation vaut 3). La saturation, elle,
    sépare nettement :

    =================  ==================
    or (5c/10c/20c)    saturation 78 – 95
    argent (½/1/2 DH)  saturation  3 – 15
    bimétallique       une zone 39–66 face à l'autre 4–10
    =================  ==================

    C'est physique : la teinte d'une surface peu chromatique est le rapport
    de deux quantités quasi nulles, donc numériquement instable. L'ancienne
    règle interrogeait la teinte en premier et obtenait 11/14 ; elle ratait
    les deux 5 DH (noyau de laiton à saturation 47 et 39, sous un plancher
    fixé à 60) et un 10 DH (écart de teinte noyau/couronne de 9,6 seulement,
    puisque comparer la teinte d'une zone argentée à celle d'une zone dorée
    revient à comparer du bruit à un signal). La règle ci-dessous obtient
    14/14 sur les mêmes pièces.

    La couverture de teinte n'est consultée que sur une zone réellement
    colorée : exiger l'homogénéité d'une teinte qui n'existe pas condamnait
    les pièces argentées (mesuré : à saturation 2 et bruit σ = 5, 47 % des
    tirages passent sous le plancher de couverture).
    """
    if val_centre < SEUIL_VALEUR_SOMBRE:
        return FAMILLE_INCONNUE, None

    centre_colore = sat_centre >= SEUIL_SAT_COLORE
    couronne_reelle = val_couronne >= SEUIL_VALEUR_SOMBRE
    couronne_coloree = couronne_reelle and sat_couronne >= SEUIL_SAT_COLORE

    # Bimétallique : un CONTRASTE de saturation franc entre noyau et couronne,
    # dont au moins une des deux zones est un métal coloré. C'est le signal
    # robuste ; l'écart de teinte ne l'est pas (cf. docstring).
    contraste = abs(sat_centre - sat_couronne)
    if (
        couronne_reelle
        and contraste >= SEUIL_CONTRASTE_BIMETAL
        and max(sat_centre, sat_couronne) >= SEUIL_SAT_COLORE
    ):
        if sat_centre > sat_couronne:
            # noyau coloré : encore faut-il que ce soit du laiton
            if _est_laiton(hue_centre, sat_centre):
                return FAMILLE_BIMETALLIQUE, POLARITE_CENTRE_OR
        elif _est_laiton(hue_couronne, sat_couronne):
            return FAMILLE_BIMETALLIQUE, POLARITE_CENTRE_ARGENT
        return FAMILLE_BIMETALLIQUE, None

    if centre_colore:
        # une pièce monométallique colorée doit être du laiton ; sinon la
        # couleur est illisible (objet coloré, reflet) et on ne conclut pas
        if _est_laiton(hue_centre, sat_centre) and couv_centre >= SEUIL_COUVERTURE:
            return FAMILLE_OR, None
        return FAMILLE_INCONNUE, None

    # peu saturée au centre : métal clair. Si la couronne est, elle, un
    # laiton franc sans atteindre le contraste exigé plus haut, on préfère
    # ne pas trancher plutôt que d'annoncer « argent » à tort.
    if couronne_coloree:
        return FAMILLE_INCONNUE, None
    return FAMILLE_ARGENT, None


def _candidats(config: Config, fam: str, pol: str | None) -> list[Denomination]:
    """Dénominations compatibles famille + polarité.

    L'identification est **exhaustive** : le filtre ``centimes`` (10c/20c)
    n'intervient qu'au **comptage** (étape M3), pas ici — une pièce de 20c
    est toujours identifiable, elle n'est simplement pas ajoutée au total
    sans l'option ``--centimes``.
    """
    actifs = list(config.denominations)
    if fam == FAMILLE_INCONNUE:
        return actifs
    actifs = [d for d in actifs if d.famille == fam]
    if fam == FAMILLE_BIMETALLIQUE and pol is not None:
        actifs = [d for d in actifs if d.polarite == pol]
    return actifs


def diametre_mesure(piece: PieceDetectee) -> float:
    """Diamètre en pixels — estimateur **unique** de tout le projet.

    Diamètre équivalent en aire, ``2·√(A/π)`` : le diamètre du disque de même
    aire que la silhouette détectée. Il est invariant par rotation, moyenne
    sur toute la surface (donc insensible à quelques pixels de bord) et ne
    dépend d'aucun ajustement d'ellipse.

    Il est surtout le **même** des deux côtés de la chaîne. Auparavant la
    classification mesurait avec ``axe_majeur`` (ajustement d'ellipse) tandis
    que l'étalonnage mesurait avec ``2·rayon`` (cercle circonscrit) : mesuré
    sur les seize pièces réelles, les deux estimateurs diffèrent
    systématiquement de −0,59 % ± 0,15 %, soit ~0,15 mm sur une pièce de
    25 mm. C'est du même ordre que ``marge_classif`` (0,20 mm) : l'échelle
    était donc biaisée par rapport à ce qu'elle servait à mesurer.
    """
    if piece.aire > 0.0:
        return 2.0 * math.sqrt(piece.aire / math.pi)
    return piece.diametre


def _classer_taille(
    piece: PieceDetectee,
    fam: str,
    pol: str | None,
    config: Config,
    px_par_mm: float,
) -> Classification:
    """Décision dimensionnelle — exige un étalonnage (``px_par_mm``)."""
    mm = diametre_mesure(piece) / px_par_mm
    if not (config.portee_mm[0] <= mm <= config.portee_mm[1]):
        piece.motif_incertitude = "hors_portee"
        return Classification(None, "hors portée", -1.0, round(mm, 2), CERTITUDE_INCONNU)
    candidats = _candidats(config, fam, pol)
    if not candidats:
        return Classification(None, "inconnu", -1.0, round(mm, 2), CERTITUDE_INCONNU)
    # Famille illisible : on refuse de conclure sur la seule taille.
    #
    # Mesure honnête sur le catalogue **corrigé** : aux diamètres nominaux,
    # la taille seule suffit — aucune dénomination n'en recouvre une autre,
    # c'est le bénéfice de la table assainie (écart minimal 2,00 mm dans une
    # même famille, contre 0,20 mm avant correction). Le veto de couleur
    # protège donc contre l'**erreur de mesure**, pas contre le catalogue :
    # balayé à ±0,5 mm, un cas sur vingt bascule — un vrai 5 DH de 26,2 mm
    # lu 26,7 mm est confirmé « 10 DH » sans le veto. En deçà de 0,4 mm
    # d'erreur, aucun cas.
    #
    # Le garde-fou reste : une fausse valeur monétaire est strictement pire
    # qu'une absence de valeur (docs/02 §5, « pas de mauvaise valeur, juste
    # aucune »), et il ne coûte rien.
    if fam == FAMILLE_INCONNUE:
        return Classification(
            None, "couleur illisible", -1.0, round(mm, 2), CERTITUDE_INCONNU
        )
    notes = [
        (d, min(abs(mm - v) for v in d.variantes_mm), min(d.variantes_mm))
        for d in candidats
    ]
    notes.sort(key=lambda n: (n[1], n[2]))
    meilleur, ecart_best, _ = notes[0]
    if ecart_best > ECART_MM_MAX:
        return Classification(None, "inconnu", -1.0, round(mm, 2), CERTITUDE_INCONNU)
    if len(notes) > 1:
        second = notes[1][1]
        marge = second - ecart_best
    else:
        marge = math.inf
        second = ecart_best
    if marge < config.marge_classif:
        return Classification(None, "inconnu", -1.0, round(mm, 2), CERTITUDE_INCONNU)
    conclusion = Classification(
        meilleur.valeur_centimes, meilleur.libelle, round(marge, 3),
        round(ecart_best, 3), CERTITUDE_CONFIRME,
    )
    return conclusion


def _classer_sans_echelle(
    fam: str, pol: str | None, config: Config
) -> Classification:
    """Sans étalonnage : seule une famille+polarité à candidat unique peut
    fixer une valeur — et cela reste **tentative**."""
    candidats = _candidats(config, fam, pol)
    if len(candidats) == 1:
        seul = candidats[0]
        return Classification(
            seul.valeur_centimes, seul.libelle, 0.0, None, CERTITUDE_TENTATIVE
        )
    return Classification(None, "inconnu", -1.0, None, CERTITUDE_TENTATIVE)


def classifier_pieces(
    pieces: list[PieceDetectee],
    image_analyse: cv2.typing.MatLike,
    config: Config,
    px_par_mm: float | None,
) -> list[PieceDetectee]:
    """Identifie et classe en place la liste des pièces détectées.

    Les pièces incertaines (`motif_incertitude` non nul) ne sont pas
    classifiées. La fonction retourne la liste (convention de chaînage).
    """
    hsv = cv2.cvtColor(image_analyse, cv2.COLOR_BGR2HSV).astype(np.float64)
    for piece in pieces:
        if piece.motif_incertitude is not None:
            continue
        hue_c, sat_c, val_c, couv_c = _couleur_region(
            hsv, piece.centre, piece.rayon, 0.0, RAPPORT_CENTRE
        )
        hue_k, sat_k, val_k, couv_k = _couleur_region(
            hsv, piece.centre, piece.rayon, COURONNE_INT, COURONNE_EXT
        )
        fam, pol = _famille_piece(
            hue_c, sat_c, val_c, couv_c, hue_k, sat_k, val_k, couv_k
        )
        piece.famille = fam
        piece.polarite = pol
        if px_par_mm is None:
            piece.classification = _classer_sans_echelle(fam, pol, config)
        else:
            piece.classification = _classer_taille(piece, fam, pol, config, px_par_mm)
    return pieces
