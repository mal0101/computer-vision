"""``cvlab`` — noyau de traitement d'images de l'atelier de vision.

Le paquet est organisé en couches, du plus bas au plus haut :

=========================  ====================================================
Module                     Rôle
=========================  ====================================================
:mod:`cvlab.params`        Déclaration typée des paramètres d'un filtre.
:mod:`cvlab.images`        Conventions d'image (BGR/uint8) et conversions.
:mod:`cvlab.drawing`       Fonctions de dessin du TP2, réutilisées partout.
:mod:`cvlab.registry`      Catalogue des filtres et application unitaire.
:mod:`cvlab.filters`       Les filtres eux-mêmes, un module par famille.
:mod:`cvlab.pipeline`      Enchaînement d'étapes et format de preset JSON.
:mod:`cvlab.histogram`     Calcul et tracé d'histogrammes.
:mod:`cvlab.metrics`       MSE, RMSE, MAE, PSNR entre deux images.
:mod:`cvlab.io_utils`      Fichiers, encodage mémoire, webcam.
:mod:`cvlab.cli`           Interface en ligne de commande.
=========================  ====================================================

Aucune couche ne dépend d'une interface graphique : les interfaces
(``ui/streamlit_app.py``, ``ui/desktop.py``) se construisent entièrement à
partir des :class:`~cvlab.params.ParamSpec` déclarés par les filtres. Ajouter
un filtre le rend donc immédiatement disponible dans les trois interfaces,
sans ligne de code supplémentaire.
"""

from __future__ import annotations

__version__ = "1.0.0"

from . import drawing, histogram, images, io_utils, metrics, params, registry
from .pipeline import Pipeline, Step
from .registry import (
    FilterDef,
    FilterError,
    all_filters,
    families,
    filters_by_family,
    get,
)

__all__ = [
    "__version__",
    "FilterDef",
    "FilterError",
    "Pipeline",
    "Step",
    "all_filters",
    "drawing",
    "families",
    "filters_by_family",
    "get",
    "histogram",
    "images",
    "io_utils",
    "metrics",
    "params",
    "registry",
]
