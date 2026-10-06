"""Classification (M2) : couleur d'abord (veto), taille ensuite.

Ordre de décision, identifié dans ``docs/02-classification.md`` :

1. **Famille par la couleur** — la teinte « or » OU l'absence de saturation
   (« argent ») d'un disque est un veto absolu : une pièce or ne peut pas
   être argentée et inversement. Le bimétallique se reconnaît par l'écart de
   teinte entre le centre et la couronne (> ``SEUIL_DIFF_BIMETAL``), tant que
   chaque zone est *couverte* à plus de ``SEUIL_COUVERTURE`` par sa couleur
   (sinon la zone est gravée/marquée et le motif devient illisible).
2. **Polarité** — le bimétallique est désambiguïsé par la couleur du centre
   (5 DH = centre or / couronne argent, 10 DH = le contraire), une décision
   *pure couleur* utilisable même sans échelle.
3. **Taille** — avec ``px_par_mm`` (étalonnage), le diamètre mesuré est
   comparé aux variantes de chaque dénomination ; il faut ``marge ≥
   config.marge_classif`` contre la deuxième meilleure, sinon UNKNOWN.
   Hors ``config.portee_mm`` → ``motif_incertitude = "hors_portee"``
   (jamais comptée).

Sans échelle (``px_par_mm is None``), aucune combinaison famille+polarité ne
peut trancher sauf le bimétallique qui n'a qu'un candidat ; le résultat reste
au plus **tentative** (jamais ``confirme``).

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
#: écart de teinte minimal (hue OpenCV 0..180) entre centre et couronne
#: pour conclure « bimétallique » (or vs argent : ≈ 25-35° en hue)
SEUIL_DIFF_BIMETAL = 14.0
#: fraction minimale de pixels « de la couleur de la zone » ; en-dessous,
#: la zone est trop gravée / éclatée pour conclure
SEUIL_COUVERTURE = 0.30
#: largeur de fenêtre de couleur autour de la teinte médiane d'une zone
#: (extensibilité de la « couverture »)
ECART_HUE_PEUPLER = 20.0
#: saturation ≥ ce seuil : pièce colorée (donc candidate « or » ou
#: « bimétallique ») ; en-dessous : métal clair « argent »
SEUIL_SAT_METAL = 60.0
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
    de la couleur (une couronne gravée retombe sur une faible couverture).
    """
    yy, xx = np.ogrid[: hsv.shape[0], : hsv.shape[1]]
    d2 = (yy - centre[1]) ** 2 + (xx - centre[0]) ** 2
    zone = d2 <= (rayon_ext * rayon) ** 2
    if rayon_int > 0.0:
        zone &= d2 >= (rayon_int * rayon) ** 2
    teintes = hsv[:, :, 0][zone].astype(np.float64)
    if teintes.size < 32:
        return 0.0, 0.0, 0.0, 0.0
    hue = _moyenne_circulaire_teinte(teintes)
    couv = float(
        np.mean(_distance_teinte(hue, np.asarray(teintes)) <= ECART_HUE_PEUPLER)
    )
    sat = float(np.median(hsv[:, :, 1][zone]))
    val = float(np.median(hsv[:, :, 2][zone]))
    return hue, sat, val, couv


def _dans_bande_or(teinte: float) -> bool:
    return BANDE_OR_MIN <= teinte <= BANDE_OR_MAX


def _famille_piece(
    hue_centre: float, sat_centre: float, val_centre: float, couv_centre: float,
    hue_couronne: float, sat_couronne: float, val_couronne: float,
    couv_couronne: float,
) -> tuple[str, str | None]:
    """Famille + polarité depuis les statistiques de couleur.

    Retourne aussi ``FAMILLE_INCONNUE`` (pièce trop sombre, trop mottée…).
    """
    if val_centre < SEUIL_VALEUR_SOMBRE or couv_centre < SEUIL_COUVERTURE:
        return FAMILLE_INCONNUE, None
    ecart = _distance_teinte(hue_centre, hue_couronne)
    est_or_centre = _dans_bande_or(hue_centre) and sat_centre >= SEUIL_SAT_METAL
    est_or_couronne = _dans_bande_or(hue_couronne) and sat_couronne >= SEUIL_SAT_METAL
    # le bimétallique exige une couronne réellement métallique : un fond
    # sombre (< SEUIL_VALEUR_SOMBRE) ou une couronne trop mottée n'est pas
    # un second métal — une pièce « or » seule sur fond sombre reste « or ».
    couronne_reelle = (
        val_couronne >= SEUIL_VALEUR_SOMBRE and couv_couronne >= SEUIL_COUVERTURE
    )
    if (
        couronne_reelle
        and ecart >= SEUIL_DIFF_BIMETAL
        and (est_or_centre or est_or_couronne)
    ):
        if est_or_centre and not est_or_couronne:
            return FAMILLE_BIMETALLIQUE, POLARITE_CENTRE_OR
        if est_or_couronne and not est_or_centre:
            return FAMILLE_BIMETALLIQUE, POLARITE_CENTRE_ARGENT
        return FAMILLE_BIMETALLIQUE, None
    if est_or_centre:
        return FAMILLE_OR, None
    if sat_centre < SEUIL_SAT_METAL:
        return FAMILLE_ARGENT, None
    return FAMILLE_INCONNUE, None


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


def _classer_taille(
    piece: PieceDetectee,
    fam: str,
    pol: str | None,
    config: Config,
    px_par_mm: float,
) -> Classification:
    """Décision dimensionnelle — exige un étalonnage (``px_par_mm``)."""
    diam_px = piece.axe_majeur if piece.axe_majeur is not None else piece.diametre
    mm = diam_px / px_par_mm
    if not (config.portee_mm[0] <= mm <= config.portee_mm[1]):
        piece.motif_incertitude = "hors_portee"
        return Classification(None, "hors portée", -1.0, round(mm, 2), CERTITUDE_INCONNU)
    candidats = _candidats(config, fam, pol)
    if not candidats:
        return Classification(None, "inconnu", -1.0, round(mm, 2), CERTITUDE_INCONNU)
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
