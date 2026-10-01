"""Mesures de comparaison entre deux images.

Le TP7 (exercice 4) demandait de comparer quantitativement l'effet des filtres
de lissage à l'aide de l'erreur quadratique moyenne. On généralise ici avec
quelques indicateurs classiques, tous calculés sur des flottants pour éviter
les débordements du type ``uint8``.

Convention : ces mesures comparent toujours le **résultat** à une
**référence**. Une MSE élevée signifie « le filtre a beaucoup modifié
l'image » — ce n'est pas en soi un défaut : sur une image bruitée, s'éloigner
de l'entrée bruitée est justement le but.
"""

from __future__ import annotations

import math

import numpy as np

from . import images as im

__all__ = ["mse", "rmse", "mae", "psnr", "difference", "compare"]


def _paire(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aligne taille, nombre de canaux et type de deux images."""
    a = im.normalise(a)
    b = im.match_shape(a, b)
    a, b = im.match_channels(a, b)
    return a.astype(np.float64), b.astype(np.float64)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Erreur quadratique moyenne : ``moyenne((a - b)^2)``. 0 = identiques."""
    fa, fb = _paire(a, b)
    return float(np.mean((fa - fb) ** 2))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Racine de la MSE, exprimée dans l'échelle des niveaux de gris (0-255)."""
    return math.sqrt(mse(a, b))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    """Erreur absolue moyenne : ``moyenne(|a - b|)``."""
    fa, fb = _paire(a, b)
    return float(np.mean(np.abs(fa - fb)))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Rapport signal/bruit de crête en décibels.

    ``PSNR = 10 * log10(255^2 / MSE)``. Renvoie ``inf`` si les images sont
    identiques. Au-delà de 40 dB la différence est généralement invisible.
    """
    erreur = mse(a, b)
    if erreur <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10((255.0**2) / erreur))


def difference(a: np.ndarray, b: np.ndarray, amplification: float = 1.0) -> np.ndarray:
    """Carte des écarts absolus, éventuellement amplifiée pour la visualiser."""
    fa, fb = _paire(a, b)
    ecart = np.abs(fa - fb) * max(0.0, float(amplification))
    return np.clip(ecart, 0, 255).astype(np.uint8)


def compare(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Toutes les mesures d'un coup, pour le tableau de l'interface."""
    erreur = mse(a, b)
    return {
        "MSE": erreur,
        "RMSE": math.sqrt(erreur),
        "MAE": mae(a, b),
        "PSNR (dB)": (
            float("inf") if erreur <= 1e-12 else 10.0 * math.log10((255.0**2) / erreur)
        ),
    }
