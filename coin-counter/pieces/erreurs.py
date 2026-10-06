"""Exceptions propres à coin-counter.

Toutes les erreurs de l'application dérivent d'ici (plus des deux erreurs
héritées de cv-app réexportées par ``pieces.compat``). Les messages sont en
français : ils sont affichés tels quels dans le HUD et en ligne de commande.
"""

from __future__ import annotations

__all__ = [
    "ErreurApplication",
    "ErreurConfiguration",
    "ErreurDetection",
    "ErreurEtalonnage",
]


class ErreurApplication(Exception):
    """Base de toutes les erreurs propres à coin-counter."""


class ErreurConfiguration(ErreurApplication, ValueError):
    """Fichier de configuration absent, mal formé ou invalide."""


class ErreurEtalonnage(ErreurApplication, RuntimeError):
    """Étalonnage impossible ou refusé (mesure instable, hors bornes…)."""


class ErreurDetection(ErreurApplication, RuntimeError):
    """Échec du pipeline de détection sur une image."""
