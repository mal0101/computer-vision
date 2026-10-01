"""Catalogue des filtres.

Importer ce paquet suffit à enregistrer tous les filtres : chaque module
applique le décorateur :func:`cvlab.registry.register` à ses fonctions au
moment de l'import. ``cvlab.registry`` importe ce paquet paresseusement, au
premier accès au catalogue.

Pour ajouter une famille de filtres, créer un module ici et l'ajouter à la
liste ci-dessous (voir ``docs/12-etendre.md``).
"""

from . import (
    annotation,  # noqa: F401
    blend,  # noqa: F401
    color,  # noqa: F401
    edges,  # noqa: F401
    geometry,  # noqa: F401
    histogram_ops,  # noqa: F401
    morphology,  # noqa: F401
    noise,  # noqa: F401
    smoothing,  # noqa: F401
    threshold,  # noqa: F401
)

__all__ = [
    "annotation",
    "blend",
    "color",
    "edges",
    "geometry",
    "histogram_ops",
    "morphology",
    "noise",
    "smoothing",
    "threshold",
]
