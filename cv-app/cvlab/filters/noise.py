"""Générateurs de bruit (TP7, exercice 4).

Ajouter du bruit volontairement est la seule façon honnête de comparer des
filtres de débruitage : on connaît alors l'image propre de référence, donc on
peut mesurer l'erreur résiduelle après filtrage (cf. :mod:`cvlab.metrics`).

Chaque générateur accepte une **graine** : à graine fixée, le bruit est
reproductible, ce qui permet de comparer deux filtres sur exactement le même
bruit. Avec une graine à 0, un nouveau tirage est fait à chaque appel (utile
sur un flux webcam, où un bruit figé se verrait comme un calque immobile).
"""

from __future__ import annotations

import numpy as np

from ..params import FloatParam, IntParam
from ..registry import register


def _generateur(graine: int) -> np.random.Generator:
    """``Generator`` NumPy, déterministe si ``graine`` est non nulle."""
    return np.random.default_rng(int(graine) if graine else None)


@register(
    id="bruit_gaussien",
    name="Bruit gaussien",
    family="Bruit",
    summary="Ajoute un bruit additif suivant une loi normale.",
    theory=(
        "g(x, y) = f(x, y) + n(x, y) avec n ~ N(moyenne, σ²). C'est le modèle "
        "du bruit électronique d'un capteur : il affecte tous les pixels, "
        "faiblement et symétriquement. σ contrôle l'amplitude ; la moyenne "
        "décale en plus la luminosité. Le calcul passe par des flottants puis "
        "np.clip, car additionner directement sur du uint8 provoquerait un "
        "repliement (255 + 3 = 2)."
    ),
    tp=("TP7 ex.4",),
    params=(
        FloatParam(name="moyenne", label="Moyenne", value=0.0, min=-50.0,
                   max=50.0, step=1.0,
                   help="0 = bruit centré, sans effet sur la luminosité moyenne."),
        FloatParam(name="sigma", label="Écart-type (σ)", value=25.0, min=0.0,
                   max=100.0, step=1.0,
                   help="Amplitude du bruit. 25 est déjà bien visible."),
        IntParam(name="graine", label="Graine aléatoire", value=0, min=0,
                 max=9999, advanced=True,
                 help="0 = tirage différent à chaque calcul ; sinon reproductible."),
    ),
)
def bruit_gaussien(
    image: np.ndarray, moyenne: float, sigma: float, graine: int
) -> np.ndarray:
    rng = _generateur(graine)
    bruit = rng.normal(float(moyenne), max(0.0, float(sigma)), image.shape)
    bruitee = image.astype(np.float32) + bruit
    return np.clip(bruitee, 0, 255).astype(np.uint8)


@register(
    id="bruit_sel_poivre",
    name="Bruit sel et poivre",
    family="Bruit",
    summary="Force une fraction des pixels à 0 (poivre) ou 255 (sel).",
    theory=(
        "Bruit impulsionnel : quelques pixels sont totalement faux, les autres "
        "intacts. Modèle des pixels morts d'un capteur ou des erreurs de "
        "transmission. C'est le cas d'école du filtre médian : la moyenne est "
        "entraînée par les valeurs extrêmes, la médiane les ignore."
    ),
    tp=("TP7 ex.3", "TP7 ex.4"),
    notes=(
        "Le TP tirait les coordonnées dans une boucle Python ; ici le tirage "
        "est vectorisé (un masque NumPy), ce qui est équivalent mais des "
        "milliers de fois plus rapide — indispensable en temps réel."
    ),
    params=(
        FloatParam(name="proportion", label="Proportion de pixels touchés",
                   value=0.05, min=0.0, max=0.5, step=0.005, decimals=3,
                   help="0.05 = 5 % des pixels sont remplacés."),
        FloatParam(name="ratio_sel", label="Part de sel (blanc)", value=0.5,
                   min=0.0, max=1.0, step=0.05,
                   help="0 = uniquement du poivre (noir), 1 = uniquement du sel."),
        IntParam(name="graine", label="Graine aléatoire", value=0, min=0,
                 max=9999, advanced=True),
    ),
)
def bruit_sel_poivre(
    image: np.ndarray, proportion: float, ratio_sel: float, graine: int
) -> np.ndarray:
    rng = _generateur(graine)
    sortie = image.copy()
    proportion = min(max(0.0, float(proportion)), 1.0)
    if proportion == 0.0:
        return sortie

    # Un tirage par *pixel* (pas par canal) : un pixel sel est blanc sur les
    # trois canaux, comme un pixel mort de capteur.
    tirage = rng.random(image.shape[:2])
    part_sel = proportion * float(ratio_sel)
    part_poivre = proportion - part_sel
    sortie[tirage < part_sel] = 255
    sortie[(tirage >= part_sel) & (tirage < part_sel + part_poivre)] = 0
    return sortie


@register(
    id="bruit_uniforme",
    name="Bruit uniforme",
    family="Bruit",
    summary="Ajoute un bruit additif tiré uniformément dans un intervalle.",
    theory=(
        "n ~ U(min, max) : toutes les amplitudes de l'intervalle sont "
        "également probables, contrairement au bruit gaussien qui concentre "
        "ses valeurs près de la moyenne. Un intervalle asymétrique décale en "
        "plus la luminosité."
    ),
    tp=("TP7 ex.4",),
    params=(
        FloatParam(name="minimum", label="Borne inférieure", value=-30.0,
                   min=-128.0, max=0.0, step=1.0),
        FloatParam(name="maximum", label="Borne supérieure", value=30.0,
                   min=0.0, max=128.0, step=1.0),
        IntParam(name="graine", label="Graine aléatoire", value=0, min=0,
                 max=9999, advanced=True),
    ),
)
def bruit_uniforme(
    image: np.ndarray, minimum: float, maximum: float, graine: int
) -> np.ndarray:
    rng = _generateur(graine)
    bas, haut = sorted((float(minimum), float(maximum)))
    if haut - bas < 1e-9:  # intervalle vide : np.random refuserait low == high
        return image.copy()
    bruit = rng.uniform(bas, haut, image.shape)
    return np.clip(image.astype(np.float32) + bruit, 0, 255).astype(np.uint8)


@register(
    id="bruit_multiplicatif",
    name="Bruit multiplicatif (speckle)",
    family="Bruit",
    summary="Multiplie chaque pixel par un facteur aléatoire.",
    theory=(
        "g = f · (1 + n) avec n ~ N(0, σ²). Le bruit est proportionnel au "
        "signal : les zones claires sont plus perturbées que les zones "
        "sombres. C'est le modèle du bruit de speckle des images radar et "
        "échographiques."
    ),
    tp=("complément",),
    params=(
        FloatParam(name="sigma", label="Écart-type relatif (σ)", value=0.2,
                   min=0.0, max=1.0, step=0.02,
                   help="0.2 = ±20 % d'amplitude typique."),
        IntParam(name="graine", label="Graine aléatoire", value=0, min=0,
                 max=9999, advanced=True),
    ),
)
def bruit_multiplicatif(image: np.ndarray, sigma: float, graine: int) -> np.ndarray:
    rng = _generateur(graine)
    facteur = 1.0 + rng.normal(0.0, max(0.0, float(sigma)), image.shape)
    return np.clip(image.astype(np.float32) * facteur, 0, 255).astype(np.uint8)
