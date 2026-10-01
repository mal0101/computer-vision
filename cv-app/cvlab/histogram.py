"""Histogrammes (TP4 §8).

Un histogramme compte, pour chaque niveau d'intensité de 0 à 255, le nombre de
pixels qui portent ce niveau. Il ne dit rien de la *position* des pixels, mais
beaucoup de l'exposition et du contraste :

* masse groupée à gauche  -> image sous-exposée ;
* masse groupée à droite  -> image surexposée ;
* masse resserrée au milieu -> image terne, faible contraste ;
* deux bosses séparées   -> image à seuiller facilement (cf. méthode d'Otsu).

Le tracé est fait avec les fonctions de dessin du TP2 plutôt qu'avec
Matplotlib : le résultat est une image OpenCV, donc affichable aussi bien dans
Streamlit que dans une fenêtre ``cv2.imshow`` de l'application de bureau, sans
dépendance supplémentaire.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import drawing as ds
from . import images as im

__all__ = ["histogram_data", "draw_histogram", "statistics"]

#: Couleurs de tracé par canal (en BGR).
_COULEURS = {
    "Bleu": (255, 120, 60),
    "Vert": (90, 220, 120),
    "Rouge": (90, 110, 255),
    "Gris": (210, 210, 210),
    "Luminance": (210, 210, 210),
}


def histogram_data(
    image: np.ndarray,
    bins: int = 256,
    masque: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Histogramme par canal.

    Renvoie ``{"Bleu": ..., "Vert": ..., "Rouge": ...}`` pour une image
    couleur, ``{"Gris": ...}`` pour une image à un canal. Chaque valeur est un
    tableau de ``bins`` entiers (nombre de pixels par classe).
    """
    image = im.normalise(image)
    bins = max(2, min(256, int(bins)))
    if im.is_gray(image):
        canaux = {"Gris": image}
    else:
        bleu, vert, rouge = cv2.split(image)
        canaux = {"Bleu": bleu, "Vert": vert, "Rouge": rouge}

    out: dict[str, np.ndarray] = {}
    for nom, canal in canaux.items():
        hist = cv2.calcHist([canal], [0], masque, [bins], [0, 256])
        out[nom] = hist.flatten().astype(np.int64)
    return out


def statistics(image: np.ndarray) -> dict[str, dict[str, float]]:
    """Statistiques par canal : min, max, moyenne, écart-type, médiane."""
    image = im.normalise(image)
    if im.is_gray(image):
        canaux = {"Gris": image}
    else:
        bleu, vert, rouge = cv2.split(image)
        canaux = {"Bleu": bleu, "Vert": vert, "Rouge": rouge}
    return {
        nom: {
            "min": float(canal.min()),
            "max": float(canal.max()),
            "moyenne": round(float(canal.mean()), 2),
            "ecart_type": round(float(canal.std()), 2),
            "mediane": float(np.median(canal)),
        }
        for nom, canal in canaux.items()
    }


def draw_histogram(
    image: np.ndarray,
    largeur: int = 512,
    hauteur: int = 300,
    cumule: bool = False,
    echelle_log: bool = False,
    fond: tuple[int, int, int] = (28, 28, 30),
) -> np.ndarray:
    """Trace l'histogramme de ``image`` et renvoie une image BGR.

    Paramètres
    ----------
    cumule
        Trace l'histogramme cumulé (la fonction de répartition), utile pour
        comprendre l'égalisation d'histogramme : égaliser revient à rendre
        cette courbe aussi proche que possible d'une droite.
    echelle_log
        Axe vertical logarithmique ; indispensable quand un pic (fond uni,
        image seuillée) écrase tout le reste du tracé.
    """
    largeur = max(160, int(largeur))
    hauteur = max(120, int(hauteur))
    marge_g, marge_d, marge_h, marge_b = 44, 12, 14, 26
    aire_l = largeur - marge_g - marge_d
    aire_h = hauteur - marge_h - marge_b

    toile = np.zeros((hauteur, largeur, 3), np.uint8)
    toile[:] = fond

    donnees = histogram_data(image)
    if cumule:
        donnees = {nom: np.cumsum(h) for nom, h in donnees.items()}

    maximum = max((float(h.max()) for h in donnees.values()), default=1.0)
    maximum = max(maximum, 1.0)

    def vers_y(valeur: float) -> int:
        if echelle_log:
            ratio = np.log1p(valeur) / np.log1p(maximum)
        else:
            ratio = valeur / maximum
        return int(marge_h + aire_h - ratio * aire_h)

    # Grille horizontale (4 divisions) et graduations de l'axe des x.
    for i in range(5):
        y = marge_h + int(i * aire_h / 4)
        toile = ds.dess_ligne(toile, (marge_g, y), (marge_g + aire_l, y),
                              (62, 62, 66), 1, cv2.LINE_8)
    for niveau in (0, 64, 128, 192, 255):
        x = marge_g + int(niveau * aire_l / 255)
        toile = ds.dess_ligne(toile, (x, marge_h + aire_h),
                              (x, marge_h + aire_h + 4), (110, 110, 114), 1,
                              cv2.LINE_8)
        toile = ds.dess_text(toile, str(niveau), (x - 10, hauteur - 8), 0.38,
                             (170, 170, 175), 1, cv2.FONT_HERSHEY_SIMPLEX)

    # Cadre de l'aire de tracé.
    toile = ds.dess_rectangle(toile, (marge_g, marge_h),
                              (marge_g + aire_l, marge_h + aire_h),
                              (90, 90, 95), 1, cv2.LINE_8)

    # Une polyligne par canal (fonctions de dessin du TP2).
    for nom, hist in donnees.items():
        bins = len(hist)
        points = [
            (marge_g + int(i * aire_l / max(1, bins - 1)), vers_y(float(v)))
            for i, v in enumerate(hist)
        ]
        toile = ds.dess_polylignes(toile, [points], False,
                                   _COULEURS.get(nom, (220, 220, 220)), 1,
                                   cv2.LINE_AA)

    # Légende et libellé de l'axe vertical.
    titre = "Histogramme cumulé" if cumule else "Histogramme"
    if echelle_log:
        titre += " (log)"
    toile = ds.dess_text(toile, titre, (marge_g, 11), 0.4, (235, 235, 235), 1,
                         cv2.FONT_HERSHEY_SIMPLEX)
    for i, (nom, _) in enumerate(donnees.items()):
        x = marge_g + aire_l - 70
        y = marge_h + 14 + i * 14
        toile = ds.dess_ligne(toile, (x, y - 4), (x + 16, y - 4),
                              _COULEURS.get(nom, (220, 220, 220)), 2, cv2.LINE_8)
        toile = ds.dess_text(toile, nom, (x + 22, y), 0.36, (200, 200, 205), 1,
                             cv2.FONT_HERSHEY_SIMPLEX)
    toile = ds.dess_text(toile, f"{int(maximum)}", (4, marge_h + 8), 0.35,
                         (160, 160, 165), 1, cv2.FONT_HERSHEY_SIMPLEX)
    toile = ds.dess_text(toile, "0", (4, marge_h + aire_h), 0.35,
                         (160, 160, 165), 1, cv2.FONT_HERSHEY_SIMPLEX)
    return toile
