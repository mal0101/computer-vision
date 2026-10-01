"""Détection de contours et gradients (TP7, exercices 5 à 7).

Un contour est un endroit où l'intensité varie brusquement. On le détecte donc
en estimant une **dérivée** de l'image :

* dérivée première (Sobel, Scharr) — le gradient ; son module est grand sur les
  contours, et son signe indique le sens de la transition ;
* dérivée seconde (Laplacien) — elle s'annule au milieu du contour (passage par
  zéro) et réagit très fort au bruit, puisque dériver deux fois amplifie les
  hautes fréquences ;
* Canny — une chaîne complète : lissage, gradient, amincissement, puis double
  seuillage par hystérésis. C'est le seul des trois qui produit directement des
  contours fins et binaires.

Un point commun à tous : **il faut lisser avant de dériver**. Une dérivée est
un filtre passe-haut, le bruit aussi est une haute fréquence ; dériver une image
bruitée produit du bruit amplifié, pas des contours.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..params import BoolParam, ChoiceParam, FloatParam, IntParam
from ..registry import register

_DIRECTIONS = {
    "Horizontal (∂/∂x)": "x",
    "Vertical (∂/∂y)": "y",
    "Module du gradient": "magnitude",
    "Orientation du gradient": "orientation",
}


def _vers_uint8(tableau: np.ndarray) -> np.ndarray:
    """Valeur absolue puis borne à 255 (cv2.convertScaleAbs).

    Les dérivées sont signées et sortent de la plage 0-255 ; les stocker
    directement en uint8 écrêterait toutes les transitions descendantes à 0.
    C'est pourquoi on calcule en CV_64F puis on convertit (TP7 ex.5).
    """
    return cv2.convertScaleAbs(tableau)


@register(
    id="sobel",
    name="Sobel",
    family="Contours & gradients",
    summary="Dérivée première de l'image, horizontale, verticale ou son module.",
    theory=(
        "Noyau horizontal [[−1,0,1],[−2,0,2],[−1,0,1]] : une différence "
        "centrée en x, combinée à un lissage en y par les coefficients "
        "(1, 2, 1). Sobel est donc à la fois un dérivateur et un lisseur, ce "
        "qui le rend plus robuste au bruit qu'une simple différence de pixels. "
        "Le module √(Gx² + Gy²) est indépendant de l'orientation du contour ; "
        "Gx seul ne voit que les transitions verticales, Gy seul que les "
        "horizontales. ksize = 1 utilise un noyau 1×3 sans lissage."
    ),
    tp=("TP7 ex.5",),
    input_mode="gray",
    output_mode="gray",
    params=(
        ChoiceParam(name="direction", label="Composante", value="Module du gradient",
                    options=_DIRECTIONS),
        IntParam(name="taille_noyau", label="Taille du noyau (ksize)", value=3,
                 min=1, max=7, odd_only=True,
                 help="1, 3, 5 ou 7. Plus grand = contours plus épais et plus lissés."),
        FloatParam(name="echelle", label="Facteur d'échelle", value=1.0, min=0.1,
                   max=10.0, step=0.1,
                   help="Multiplie la dérivée avant conversion en 8 bits."),
        IntParam(name="decalage", label="Décalage (delta)", value=0, min=-128,
                 max=128, advanced=True),
        IntParam(name="flou_prealable", label="Flou gaussien préalable", value=0,
                 min=0, max=15, odd_only=False,
                 help="0 = aucun. Sinon taille du noyau gaussien appliqué avant."),
    ),
)
def sobel(
    image: np.ndarray,
    direction: str,
    taille_noyau: int,
    echelle: float,
    decalage: int,
    flou_prealable: int,
) -> tuple[np.ndarray, dict[str, object]]:
    source = _flouter(image, flou_prealable)
    ksize = int(taille_noyau)
    gx = cv2.Sobel(source, cv2.CV_64F, 1, 0, ksize=ksize,
                   scale=float(echelle), delta=float(decalage))
    gy = cv2.Sobel(source, cv2.CV_64F, 0, 1, ksize=ksize,
                   scale=float(echelle), delta=float(decalage))
    return _rendre_gradient(gx, gy, direction)


@register(
    id="scharr",
    name="Scharr",
    family="Contours & gradients",
    summary="Variante 3×3 de Sobel, plus précise en orientation.",
    theory=(
        "Noyau [[−3,0,3],[−10,0,10],[−3,0,3]] : les coefficients sont optimisés "
        "pour que la réponse soit la plus isotrope possible, c'est-à-dire que "
        "l'orientation estimée du gradient soit juste quel que que soit l'angle "
        "du contour. À taille 3×3, Scharr est strictement préférable à Sobel ; "
        "en revanche il n'existe qu'en 3×3."
    ),
    tp=("complément",),
    input_mode="gray",
    output_mode="gray",
    params=(
        ChoiceParam(name="direction", label="Composante", value="Module du gradient",
                    options=_DIRECTIONS),
        FloatParam(name="echelle", label="Facteur d'échelle", value=1.0, min=0.1,
                   max=10.0, step=0.1),
        IntParam(name="flou_prealable", label="Flou gaussien préalable", value=0,
                 min=0, max=15),
    ),
)
def scharr(
    image: np.ndarray, direction: str, echelle: float, flou_prealable: int
) -> tuple[np.ndarray, dict[str, object]]:
    source = _flouter(image, flou_prealable)
    gx = cv2.Scharr(source, cv2.CV_64F, 1, 0, scale=float(echelle))
    gy = cv2.Scharr(source, cv2.CV_64F, 0, 1, scale=float(echelle))
    return _rendre_gradient(gx, gy, direction)


def _rendre_gradient(
    gx: np.ndarray, gy: np.ndarray, direction: str
) -> tuple[np.ndarray, dict[str, object]]:
    """Fabrique l'image de sortie à partir des deux dérivées partielles."""
    if direction == "x":
        sortie = _vers_uint8(gx)
    elif direction == "y":
        sortie = _vers_uint8(gy)
    elif direction == "magnitude":
        module = cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))
        sortie = _vers_uint8(module)
    else:  # orientation
        angle = cv2.phase(
            gx.astype(np.float32), gy.astype(np.float32), angleInDegrees=True
        )
        # 0-360° ramené sur 0-255 ; à lire avec une palette cyclique (HSV).
        sortie = (angle * 255.0 / 360.0).astype(np.uint8)

    module = cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))
    return sortie, {
        "module max": f"{float(module.max()):.1f}",
        "module moyen": f"{float(module.mean()):.1f}",
    }


def _flouter(image: np.ndarray, taille: int) -> np.ndarray:
    """Flou gaussien optionnel appliqué avant dérivation."""
    taille = int(taille)
    if taille <= 0:
        return image
    if taille % 2 == 0:
        taille += 1
    return cv2.GaussianBlur(image, (taille, taille), 0)


@register(
    id="laplacien",
    name="Laplacien",
    family="Contours & gradients",
    summary="Dérivée seconde : ∂²f/∂x² + ∂²f/∂y².",
    theory=(
        "Le noyau 3×3 [[0,1,0],[1,−4,1],[0,1,0]] approche le laplacien. "
        "Contrairement à Sobel, il n'est pas directionnel : une seule "
        "convolution suffit pour détecter les contours dans toutes les "
        "orientations. En contrepartie il est très sensible au bruit (deux "
        "dérivations) et ne donne pas de sens de transition. L'usage classique "
        "est donc le Laplacien du gaussien (LoG) : flouter, puis appliquer le "
        "laplacien — ce que fait ici l'option de flou préalable."
    ),
    tp=("TP7 ex.6",),
    input_mode="gray",
    output_mode="gray",
    params=(
        IntParam(name="taille_noyau", label="Taille du noyau (ksize)", value=3,
                 min=1, max=31, odd_only=True,
                 help="1 utilise le noyau 3×3 classique ; au-delà, un noyau de Sobel."),
        FloatParam(name="echelle", label="Facteur d'échelle", value=1.0, min=0.1,
                   max=10.0, step=0.1),
        IntParam(name="decalage", label="Décalage (delta)", value=0, min=-128,
                 max=128, advanced=True),
        IntParam(name="flou_prealable", label="Flou gaussien préalable (LoG)",
                 value=3, min=0, max=15,
                 help="Fortement recommandé : le laplacien amplifie le bruit."),
        BoolParam(name="garder_signe", label="Afficher le signe (gris = zéro)",
                  value=False,
                  help=(
                      "Centre la sortie sur 128 au lieu de prendre la valeur "
                      "absolue : on distingue alors les deux côtés du contour."
                  )),
    ),
)
def laplacien(
    image: np.ndarray,
    taille_noyau: int,
    echelle: float,
    decalage: int,
    flou_prealable: int,
    garder_signe: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    source = _flouter(image, flou_prealable)
    brut = cv2.Laplacian(
        source, cv2.CV_64F, ksize=int(taille_noyau), scale=float(echelle),
        delta=float(decalage),
    )
    if garder_signe:
        sortie = np.clip(brut + 128.0, 0, 255).astype(np.uint8)
    else:
        sortie = _vers_uint8(brut)
    passages = int(np.count_nonzero(np.diff(np.sign(brut), axis=1)))
    return sortie, {
        "amplitude max": f"{float(np.abs(brut).max()):.1f}",
        "passages par zéro (lignes)": passages,
    }


@register(
    id="canny",
    name="Détecteur de Canny",
    family="Contours & gradients",
    summary="Contours fins et binaires par double seuillage à hystérésis.",
    theory=(
        "Quatre étapes : (1) lissage gaussien ; (2) gradient de Sobel ; "
        "(3) amincissement — on ne garde un pixel que s'il est maximal dans la "
        "direction du gradient ; (4) hystérésis — tout pixel de module > seuil "
        "haut est un contour, tout pixel < seuil bas est rejeté, et entre les "
        "deux un pixel n'est conservé que s'il touche un contour déjà accepté. "
        "C'est cette dernière étape qui donne des contours continus là où un "
        "seuil unique produirait des pointillés.\n"
        "Réglage : un rapport seuil_haut/seuil_bas de 2 à 3 est la règle "
        "usuelle. Monter les deux seuils ne garde que les contours francs ; les "
        "baisser fait apparaître du détail et du bruit (TP7 ex.7)."
    ),
    tp=("TP7 ex.7", "TP7 exo.py"),
    input_mode="gray",
    output_mode="gray",
    params=(
        IntParam(name="seuil_bas", label="Seuil bas", value=50, min=0, max=500,
                 help="Sous ce module de gradient, le pixel est rejeté."),
        IntParam(name="seuil_haut", label="Seuil haut", value=150, min=0, max=500,
                 help="Au-dessus, le pixel est un contour à coup sûr."),
        IntParam(name="ouverture", label="Ouverture de Sobel", value=3, min=3,
                 max=7, odd_only=True,
                 help="Taille du noyau de gradient interne : 3, 5 ou 7."),
        BoolParam(name="norme_l2", label="Norme L2 exacte", value=False,
                  help=(
                      "√(Gx²+Gy²) au lieu de |Gx|+|Gy| : un peu plus précis, "
                      "un peu plus lent."
                  )),
        IntParam(name="flou_prealable", label="Flou gaussien préalable", value=5,
                 min=0, max=15,
                 help="Canny lisse déjà en interne ; un flou supplémentaire réduit le bruit résiduel."),
    ),
)
def canny(
    image: np.ndarray,
    seuil_bas: int,
    seuil_haut: int,
    ouverture: int,
    norme_l2: bool,
    flou_prealable: int,
) -> tuple[np.ndarray, dict[str, object]]:
    bas, haut = sorted((int(seuil_bas), int(seuil_haut)))
    source = _flouter(image, flou_prealable)
    contours = cv2.Canny(
        source, bas, haut, apertureSize=int(ouverture), L2gradient=bool(norme_l2)
    )
    taux = float(np.count_nonzero(contours)) / contours.size
    infos: dict[str, object] = {
        "seuils utilisés": f"{bas} / {haut}",
        "rapport haut/bas": "∞" if bas == 0 else f"{haut / bas:.2f}",
        "pixels de contour": f"{taux * 100:.2f} %",
    }
    if (seuil_bas, seuil_haut) != (bas, haut):
        infos["note"] = "seuils réordonnés (le seuil bas doit être le plus petit)"
    return contours, infos
