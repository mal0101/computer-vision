"""Étalonnage (M2) : échelle planimétrique de la session, en mémoire.

Le pipeline mesure des diamètres en pixels à l'échelle d'analyse ; les
classifier en millimètres exige ``px_par_mm``. Deux voies, toujours **en
mémoire de session** (rien n'est persistant) :

* ``ObservateurEtalonnage`` — l'utilisateur tient une pièce **de valeur
  connue** face à la caméra ; l'observateur collecte le rayon détecté de la
  pièce la plus grande/médiane sur chaque frame et déclare l'échelle dès que
  la série est stable (≥ ``CONFIG_SUIVI`` frames, étendue ≤ 1 px) ;
* ``echelle_depuis_carte`` — l'utilisateur mesure une carte bancaire
  (85,60 mm de long) sur l'écran d'étalonnage.

``controle_coherence`` vérifie une deuxième mesure connue à ± 2 % et peut
rejeter une échelle aberrante (``ErreurEtalonnage``). ``valider_echelle``
borde ``px_par_mm`` dans une plage physiquement plausible pour l'échelle
d'analyse (caméra ≥ 40 cm, largeur d'analyse 640 px).
"""

from __future__ import annotations

import statistics

from .config import Config
from .erreurs import ErreurEtalonnage

__all__ = [
    "ObservateurEtalonnage",
    "etalonner_par_piece",
    "echelle_depuis_carte",
    "valider_echelle",
    "controle_coherence",
    "LONGUEUR_CARTE_BANCAIRE_MM",
    "TOLERANCE_COHERENCE",
]

#: longueur normalisée d'une carte bancaire ISO/IEC 7810 (mm)
LONGUEUR_CARTE_BANCAIRE_MM = 85.60
#: tolérance de cohérence du contrôle croisé (± %)
TOLERANCE_COHERENCE = 0.02
#: nombre de rayons nécessaires avant de déclarer l'étalonnage stable
#: (≥ 8 frames comme exigé par la porte M2)
MIN_FRAMES_STABLES = 8
#: étendue (max − min) des rayons tolérée sur la série stable (px)
ETENDUE_STABLE_PX = 1.0
#: échelle maximale plausible en px/mm à l'échelle d'analyse (≈ 640 px
#: pour 30 mm de pièce : caméra au plus près du bureau)
ECHELLE_PX_MM_MAX = 640.0 / 18.5


class ObservateurEtalonnage:
    """Collecteur de rayons pour l'étalonnage « pièce connue dans la main »."""

    def __init__(self, min_frames: int = MIN_FRAMES_STABLES) -> None:
        self._min_frames = min_frames
        self._rayons: list[float] = []

    def enregistrer(self, rayon_px: float) -> None:
        """Ajoute le rayon (px, échelle d'analyse) d'une frame d'étalonnage."""
        if not (rayon_px > 0.0):
            raise ErreurEtalonnage(
                f"rayon d'étalonnage invalide : {rayon_px!r}"
            )
        self._rayons.append(float(rayon_px))
        # on ne garde que la fenêtre utile à la stabilité
        if len(self._rayons) > 2 * self._min_frames:
            self._rayons = self._rayons[-self._min_frames:]

    def reinitialiser(self) -> None:
        self._rayons = []

    @property
    def rayon_median(self) -> float | None:
        return statistics.median(self._rayons) if self._rayons else None

    @property
    def nb_frames(self) -> int:
        return len(self._rayons)

    @property
    def etendue(self) -> float | None:
        if not self._rayons:
            return None
        return max(self._rayons) - min(self._rayons)

    def est_stable(self) -> bool:
        """Vrai si ≥ ``min_frames`` rayons cohérents à ± ``ETENDUE_STABLE_PX``."""
        if len(self._rayons) < self._min_frames:
            return False
        return self.etendue is not None and self.etendue <= ETENDUE_STABLE_PX


def valider_echelle(config: Config, px_par_mm: float) -> float:
    """Borne ``px_par_mm`` dans les plages plausibles, sinon erreur."""
    if not (isinstance(px_par_mm, (int, float)) and px_par_mm > 0.0):
        raise ErreurEtalonnage(
            f"échelle invalide : {px_par_mm!r} (px/mm doit être > 0)"
        )
    if px_par_mm < config.px_mm_min or px_par_mm > ECHELLE_PX_MM_MAX:
        raise ErreurEtalonnage(
            f"échelle aberrante : {px_par_mm:.2f} px/mm hors plage "
            f"[{config.px_mm_min:.1f}, {ECHELLE_PX_MM_MAX:.1f}]"
        )
    return float(px_par_mm)


def etalonner_par_piece(
    observateur: ObservateurEtalonnage, diametre_reel_mm: float
) -> float | None:
    """Échelle depuis une pièce de valeur connue tenue devant la caméra.

    Retourne ``None`` tant que l'observateur n'est pas stable ; sinon
    retourne ``px_par_mm`` (l'échelle est ensuite validée par l'appelant).
    """
    if not observateur.est_stable():
        return None
    median = observateur.rayon_median
    if median is None or not (diametre_reel_mm > 0.0):
        raise ErreurEtalonnage("étalonnage impossible (données insuffisantes)")
    return 2.0 * median / float(diametre_reel_mm)


def echelle_depuis_carte(
    longueur_px: float, longueur_reelle_mm: float = LONGUEUR_CARTE_BANCAIRE_MM
) -> float:
    """Échelle depuis la longueur mesurée d'un objet de taille connue."""
    if not (longueur_px > 0.0 and longueur_reelle_mm > 0.0):
        raise ErreurEtalonnage(
            f"mesure de carte invalide : {longueur_px!r} px / "
            f"{longueur_reelle_mm!r} mm"
        )
    return float(longueur_px) / float(longueur_reelle_mm)


def controle_coherence(
    px_par_mm: float,
    controle_diam_px: float,
    controle_diam_reel_mm: float,
    tolerance: float = TOLERANCE_COHERENCE,
) -> float:
    """Contrôle croisé : le diamètre mesuré doit coller à ± ``tolerance``.

    Lève ``ErreurEtalonnage`` si l'écart relatif dépasse la tolérance. Ce
    contrôle est *suggestif* : la valeur retournée reste l'échelle proposée.
    """
    mesure_mm = controle_diam_px / px_par_mm
    ecart_rel = abs(mesure_mm - controle_diam_reel_mm) / controle_diam_reel_mm
    if ecart_rel > tolerance:
        raise ErreurEtalonnage(
            f"contrôle de cohérence échoué : {mesure_mm:.2f} mm mesurés "
            f"pour {controle_diam_reel_mm:.2f} mm attendus (écart "
            f"{ecart_rel:.1%} > {tolerance:.1%})"
        )
    return float(px_par_mm)
