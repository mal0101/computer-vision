"""Opérations fondées sur l'histogramme (TP4 §8).

L'histogramme décrit la répartition des intensités (voir
:mod:`cvlab.histogram`). On peut s'en servir non seulement pour diagnostiquer
une image, mais aussi pour la corriger : si les intensités sont concentrées sur
une plage étroite, redistribuer cette plage sur 0-255 augmente le contraste
sans toucher au contenu.

Point d'attention pour les images couleur : égaliser les trois canaux BGR
indépendamment **déforme les couleurs**, car les rapports entre canaux — qui
définissent la teinte — ne sont pas préservés. La bonne pratique est de
n'égaliser qu'un canal de luminance (Y de YCrCb, L de Lab, V de HSV) et de
laisser les canaux de chrominance intacts.
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import images as im
from ..params import ChoiceParam, FloatParam, IntParam
from ..registry import register

_ESPACES = {
    "Luminance Y (YCrCb)": "ycrcb",
    "Luminance L (Lab)": "lab",
    "Valeur V (HSV)": "hsv",
    "Trois canaux BGR (déforme les couleurs)": "bgr",
}


def _appliquer_sur_luminance(
    image: np.ndarray, espace: str, operation
) -> np.ndarray:
    """Applique ``operation`` (canal 8 bits -> canal 8 bits) sur la luminance."""
    if im.is_gray(image):
        return operation(image)

    if espace == "bgr":
        canaux = cv2.split(image)
        return cv2.merge([operation(c) for c in canaux])

    conversions = {
        "ycrcb": (cv2.COLOR_BGR2YCrCb, cv2.COLOR_YCrCb2BGR, 0),
        "lab": (cv2.COLOR_BGR2Lab, cv2.COLOR_Lab2BGR, 0),
        "hsv": (cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR, 2),
    }
    vers, retour, index = conversions[espace]
    converti = cv2.cvtColor(image, vers)
    plans = list(cv2.split(converti))
    plans[index] = operation(plans[index])
    return cv2.cvtColor(cv2.merge(plans), retour)


@register(
    id="egalisation_histogramme",
    name="Égalisation d'histogramme",
    family="Histogramme",
    summary="Redistribue les intensités pour occuper toute la plage 0-255.",
    theory=(
        "On calcule l'histogramme cumulé normalisé — la fonction de "
        "répartition — et on s'en sert comme table de correspondance : "
        "g = 255 · F(f). L'histogramme résultant est aussi plat que possible, "
        "ce qui maximise le contraste global. Deux limites : l'opération est "
        "**globale**, donc une petite zone mal exposée ne sera pas corrigée si "
        "le reste de l'image est correct ; et elle amplifie le bruit des zones "
        "uniformes. Le CLAHE répond à ces deux critiques."
    ),
    tp=("TP4 §8",),
    params=(
        ChoiceParam(name="espace", label="Canal égalisé",
                    value="Luminance Y (YCrCb)", options=_ESPACES,
                    help="Sans effet sur une image déjà en niveaux de gris."),
    ),
)
def egalisation_histogramme(image: np.ndarray, espace: str) -> np.ndarray:
    return _appliquer_sur_luminance(image, espace, cv2.equalizeHist)


@register(
    id="clahe",
    name="CLAHE (égalisation locale)",
    family="Histogramme",
    summary="Égalisation par tuiles, avec limitation du contraste.",
    theory=(
        "CLAHE découpe l'image en tuiles et égalise chaque tuile séparément, "
        "puis interpole entre les tuiles pour éviter les discontinuités. Le "
        "paramètre ``limite`` écrête l'histogramme avant égalisation et "
        "redistribue l'excédent : c'est ce qui empêche l'amplification "
        "explosive du bruit dans les zones uniformes, défaut principal de "
        "l'égalisation globale.\n"
        "Réglage : une grille 8×8 convient en général ; plus de tuiles = "
        "correction plus locale mais rendu moins naturel. Une limite de 2 à 4 "
        "est un bon compromis, au-delà on retrouve le bruit."
    ),
    tp=("TP4 §8", "complément"),
    params=(
        FloatParam(name="limite", label="Limite de contraste", value=2.0,
                   min=0.5, max=10.0, step=0.1,
                   help="Plus la limite est haute, plus le contraste (et le bruit) augmente."),
        IntParam(name="tuiles", label="Tuiles par côté", value=8, min=1, max=16,
                 help="Grille de découpage : 8 signifie une grille 8×8."),
        ChoiceParam(name="espace", label="Canal traité", value="Luminance L (Lab)",
                    options=_ESPACES),
    ),
)
def clahe(
    image: np.ndarray, limite: float, tuiles: int, espace: str
) -> np.ndarray:
    outil = cv2.createCLAHE(
        clipLimit=float(limite), tileGridSize=(int(tuiles), int(tuiles))
    )
    return _appliquer_sur_luminance(image, espace, outil.apply)


@register(
    id="etirement_histogramme",
    name="Étirement de contraste (normalisation)",
    family="Histogramme",
    summary="Étire linéairement la plage des intensités sur 0-255.",
    theory=(
        "g = 255 · (f − min) / (max − min). Contrairement à l'égalisation, "
        "cette transformation est **linéaire** : la forme de l'histogramme est "
        "conservée, seulement dilatée. Le rendu est donc plus naturel, mais la "
        "correction est moins énergique — et un seul pixel aberrant à 0 ou 255 "
        "suffit à l'annuler. D'où l'option de percentiles : on ignore les "
        "p % de pixels extrêmes avant de calculer min et max."
    ),
    tp=("TP4 §8", "complément"),
    params=(
        FloatParam(name="percentile_bas", label="Percentile bas ignoré",
                   value=1.0, min=0.0, max=20.0, step=0.5, unit="%"),
        FloatParam(name="percentile_haut", label="Percentile haut ignoré",
                   value=1.0, min=0.0, max=20.0, step=0.5, unit="%"),
    ),
)
def etirement_histogramme(
    image: np.ndarray, percentile_bas: float, percentile_haut: float
) -> tuple[np.ndarray, dict[str, object]]:
    bas = float(np.percentile(image, max(0.0, percentile_bas)))
    haut = float(np.percentile(image, min(100.0, 100.0 - percentile_haut)))
    if haut - bas < 1e-6:
        # Image uniforme : rien à étirer, on évite une division par zéro.
        return image.copy(), {"note": "plage d'intensité nulle, image inchangée"}
    echelle = 255.0 / (haut - bas)
    sortie = np.clip((image.astype(np.float32) - bas) * echelle, 0, 255)
    return sortie.astype(np.uint8), {
        "plage source": f"[{bas:.0f}, {haut:.0f}]",
        "gain appliqué": f"{echelle:.2f}",
    }


@register(
    id="trace_histogramme",
    name="Tracé de l'histogramme",
    family="Histogramme",
    summary="Remplace l'image par le graphique de son histogramme.",
    theory=(
        "Ce n'est pas un filtre au sens strict : la sortie est un graphique, "
        "pas une image traitée. Il est fourni comme étape de chaîne pour "
        "pouvoir comparer visuellement l'histogramme avant et après un "
        "traitement — par exemple pour voir l'effet d'une égalisation sur la "
        "courbe cumulée."
    ),
    tp=("TP4 §8",),
    output_mode="color",
    params=(
        ChoiceParam(name="mode", label="Mode", value="Histogramme",
                    options={"Histogramme": "simple", "Cumulé": "cumule"}),
        ChoiceParam(name="echelle", label="Échelle verticale", value="Linéaire",
                    options={"Linéaire": "lineaire", "Logarithmique": "log"}),
        IntParam(name="largeur", label="Largeur du graphique", value=640,
                 min=256, max=1600, step=32, unit="px"),
        IntParam(name="hauteur", label="Hauteur du graphique", value=400,
                 min=160, max=1200, step=20, unit="px"),
    ),
)
def trace_histogramme(
    image: np.ndarray, mode: str, echelle: str, largeur: int, hauteur: int
) -> np.ndarray:
    from ..histogram import draw_histogram

    return draw_histogram(
        image,
        largeur=int(largeur),
        hauteur=int(hauteur),
        cumule=(mode == "cumule"),
        echelle_log=(echelle == "log"),
    )
