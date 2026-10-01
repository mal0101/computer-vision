"""Fonctions enveloppantes de dessin (TP2 §2, TP3).

Le TP2 demandait d'écrire des fonctions enveloppantes autour de ``cv2.line``,
``cv2.rectangle``, ``cv2.circle``, ``cv2.ellipse``, ``cv2.polylines`` et
``cv2.putText`` avec des valeurs par défaut pour la couleur, l'épaisseur et le
type de ligne. Ce module reprend ces fonctions (mêmes noms français que dans
``TP2/codes TP2/fonctions/dessin.py``) et les réutilise partout dans
l'application : filtres d'annotation, tracé des histogrammes, incrustations de
l'interface OpenCV.

Différence avec le TP : ces fonctions **ne modifient pas** l'image reçue, elles
renvoient une copie. Dans une chaîne de traitement, muter l'image d'entrée
casserait l'affichage de l'étape précédente.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

__all__ = [
    "BLANC",
    "NOIR",
    "POLICES",
    "TYPES_LIGNE",
    "dess_ligne",
    "dess_rectangle",
    "dess_cercle",
    "dess_ellipse",
    "dess_polylignes",
    "dess_text",
    "taille_texte",
    "dess_texte_encadre",
]

BLANC = (255, 255, 255)
NOIR = (0, 0, 0)

#: Polices Hershey disponibles (seules polices embarquées dans OpenCV).
POLICES: dict[str, int] = {
    "Simplex": cv2.FONT_HERSHEY_SIMPLEX,
    "Plain": cv2.FONT_HERSHEY_PLAIN,
    "Duplex": cv2.FONT_HERSHEY_DUPLEX,
    "Complex": cv2.FONT_HERSHEY_COMPLEX,
    "Triplex": cv2.FONT_HERSHEY_TRIPLEX,
    "Complex Small": cv2.FONT_HERSHEY_COMPLEX_SMALL,
    "Script Simplex": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    "Script Complex": cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
}

#: Types de ligne (TP2 §2.1).
TYPES_LIGNE: dict[str, int] = {
    "4-connexe": cv2.LINE_4,
    "8-connexe": cv2.LINE_8,
    "Anticrénelage": cv2.LINE_AA,
}


def _copie(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image.copy())


def dess_ligne(
    image: np.ndarray,
    pt_debut: tuple[int, int],
    pt_fin: tuple[int, int],
    couleur: Sequence[int] = BLANC,
    epaisseur: int = 1,
    type_ligne: int = cv2.LINE_AA,
) -> np.ndarray:
    """Segment de ``pt_debut`` à ``pt_fin``. Enveloppe ``cv2.line``."""
    out = _copie(image)
    cv2.line(out, tuple(map(int, pt_debut)), tuple(map(int, pt_fin)),
             tuple(couleur), int(epaisseur), type_ligne)
    return out


def dess_rectangle(
    image: np.ndarray,
    haut_gauche: tuple[int, int],
    bas_droit: tuple[int, int],
    couleur: Sequence[int] = BLANC,
    epaisseur: int = 1,
    type_ligne: int = cv2.LINE_AA,
) -> np.ndarray:
    """Rectangle. ``epaisseur=cv2.FILLED`` (-1) pour un rectangle plein."""
    out = _copie(image)
    cv2.rectangle(out, tuple(map(int, haut_gauche)), tuple(map(int, bas_droit)),
                  tuple(couleur), int(epaisseur), type_ligne)
    return out


def dess_cercle(
    image: np.ndarray,
    centre: tuple[int, int],
    rayon: int,
    couleur: Sequence[int] = BLANC,
    epaisseur: int = 1,
    type_ligne: int = cv2.LINE_AA,
) -> np.ndarray:
    """Cercle de centre ``centre`` et de rayon ``rayon``."""
    out = _copie(image)
    cv2.circle(out, tuple(map(int, centre)), max(0, int(rayon)),
               tuple(couleur), int(epaisseur), type_ligne)
    return out


def dess_ellipse(
    image: np.ndarray,
    centre: tuple[int, int],
    axes: tuple[int, int],
    angle: float = 0.0,
    debut_angle: float = 0.0,
    fin_angle: float = 360.0,
    couleur: Sequence[int] = BLANC,
    epaisseur: int = 1,
    type_ligne: int = cv2.LINE_AA,
) -> np.ndarray:
    """Ellipse ou arc d'ellipse (``debut_angle`` / ``fin_angle`` en degrés)."""
    out = _copie(image)
    cv2.ellipse(out, tuple(map(int, centre)),
                (max(1, int(axes[0])), max(1, int(axes[1]))),
                float(angle), float(debut_angle), float(fin_angle),
                tuple(couleur), int(epaisseur), type_ligne)
    return out


def dess_polylignes(
    image: np.ndarray,
    points: Sequence[Sequence[tuple[int, int]]] | np.ndarray,
    est_fermer: bool = True,
    couleur: Sequence[int] = BLANC,
    epaisseur: int = 1,
    type_ligne: int = cv2.LINE_AA,
) -> np.ndarray:
    """Ligne brisée. ``points`` est une *liste de contours* (comme OpenCV)."""
    out = _copie(image)
    contours = [np.asarray(p, dtype=np.int32).reshape(-1, 1, 2) for p in points]
    cv2.polylines(out, contours, bool(est_fermer), tuple(couleur),
                  int(epaisseur), type_ligne)
    return out


def dess_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    taille: float = 1.0,
    couleur: Sequence[int] = BLANC,
    epaisseur: int = 1,
    police: int = cv2.FONT_HERSHEY_COMPLEX,
    type_ligne: int = cv2.LINE_AA,
) -> np.ndarray:
    """Texte dont ``org`` est le coin **bas-gauche** (convention OpenCV)."""
    out = _copie(image)
    cv2.putText(out, str(text), tuple(map(int, org)), police, float(taille),
                tuple(couleur), int(epaisseur), type_ligne)
    return out


def taille_texte(
    text: str,
    taille: float = 1.0,
    epaisseur: int = 1,
    police: int = cv2.FONT_HERSHEY_COMPLEX,
) -> tuple[int, int, int]:
    """``(largeur, hauteur, base)`` du rectangle englobant un texte."""
    (largeur, hauteur), base = cv2.getTextSize(str(text), police, float(taille),
                                               int(epaisseur))
    return int(largeur), int(hauteur), int(base)


def dess_texte_encadre(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    taille: float = 0.5,
    couleur: Sequence[int] = BLANC,
    fond: Sequence[int] = NOIR,
    epaisseur: int = 1,
    marge: int = 4,
    police: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> np.ndarray:
    """Texte posé sur un rectangle plein, pour rester lisible sur tout fond.

    Utilisé par l'application de bureau pour les messages d'aide incrustés
    (les textes blancs du TP3 disparaissaient sur les images claires).
    """
    largeur, hauteur, base = taille_texte(text, taille, epaisseur, police)
    x, y = int(org[0]), int(org[1])
    out = dess_rectangle(
        image,
        (x - marge, y - hauteur - marge),
        (x + largeur + marge, y + base + marge),
        fond,
        epaisseur=cv2.FILLED,
        type_ligne=cv2.LINE_8,
    )
    return dess_text(out, text, (x, y), taille, couleur, epaisseur, police)
