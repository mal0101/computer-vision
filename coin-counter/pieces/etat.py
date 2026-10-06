"""État de session (M3) : tout ce qui vit entre deux frames.

Un seul objet par instance de l'application ; ``traitement.traiter_image``
le fait évoluer (pièces → suivi → total). L'échelle d'étalonnage
(``px_par_mm``) est une propriété de session : elle n'est jamais écrite sur
disque (voir ```docs/02-classification.md`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .etalonnage import ObservateurEtalonnage
from .suivi import Suiveur

__all__ = ["EtatApplication"]


@dataclass
class EtatApplication:
    """État mutable d'une session de comptage."""

    config: Config
    suiveur: Suiveur
    #: étalonnage en pixels par millimètre, ou None tant que non calibré
    px_par_mm: float | None = None
    #: mode « tenir une pièce connue » : collecte de l'échelle
    mode_etalonnage: bool = False
    #: affichage du masque de profilage (touche D)
    afficher_masque: bool = False
    observateur: ObservateurEtalonnage | None = None
    #: diamètre réel (mm) de la pièce tenue pendant l'étalonnage
    diametre_etalon_mm: float = 24.0
    num_frame: int = 0
    #: dernière durée d'analyse (ms), pour le HUD
    analyse_ms: float = 0.0
    messages: list[str] = field(default_factory=list)

    def reinitialiser(self) -> None:
        """Remet le comptage à zéro (mais conserve l'échelle étalonnée)."""
        self.suiveur.reinitialiser()
        self.num_frame = 0
        self.messages = []
