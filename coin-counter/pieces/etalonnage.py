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
from collections.abc import Sequence
from dataclasses import dataclass

from .config import Config
from .erreurs import ErreurEtalonnage

__all__ = [
    "ObservateurEtalonnage",
    "EchelleEstimee",
    "etalonner_par_piece",
    "echelle_depuis_carte",
    "estimer_echelle_catalogue",
    "valider_echelle",
    "echelle_max_plausible",
    "controle_coherence",
    "LONGUEUR_CARTE_BANCAIRE_MM",
    "TOLERANCE_COHERENCE",
    "INLIERS_MIN_CATALOGUE",
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
#: tolérance d'alignement d'une observation sur une variante du catalogue
#: (mm) lors de l'estimation par consensus.
#:
#: Fixée à 0,80 et non à la marge de décision (0,20) : il s'agit ici
#: d'*associer* une observation à une dénomination pour voter, pas de
#: trancher une identification. Mesuré à l'échelle d'analyse, le diamètre
#: équivalent en aire porte un biais additif d'environ −0,6 mm sur les
#: petites pièces ; une tolérance trop serrée les exclurait du vote et
#: déplacerait le consensus de ~4 %. L'identification, elle, reste soumise
#: à ``marge_classif``.
TOLERANCE_CATALOGUE_MM = 0.80
#: nombre minimal d'observations retombant sur le catalogue pour qu'une
#: échelle estimée soit jugée crédible
INLIERS_MIN_CATALOGUE = 3
#: nombre minimal de dénominations DISTINCTES parmi les inliers.
#:
#: Sans cette contrainte, une scène de disques tous identiques renvoie une
#: échelle parfaitement « consensuelle » et parfaitement arbitraire : chaque
#: dénomination du catalogue explique l'ensemble des observations, il suffit
#: de choisir l'échelle correspondante. Il faut au moins deux tailles
#: différentes pour que le rapport entre elles fixe réellement l'échelle.
DENOMINATIONS_MIN_CATALOGUE = 2


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
        # Fenêtre glissante stricte : on ne garde que les `min_frames`
        # dernières mesures, à chaque ajout.
        #
        # L'ancienne version laissait la liste croître jusqu'à `2·min_frames`
        # avant de la tronquer, tout en jugeant la stabilité sur la liste
        # ENTIÈRE. Deux conséquences mesurées : un transitoire de 4 frames
        # (l'utilisateur place encore sa pièce) repoussait la première
        # stabilité possible de la frame 8 annoncée à la frame 17 ; et une
        # perturbation récurrente faisait osciller `est_stable()` huit fois
        # en trente-six frames, chaque troncature « oubliant » la
        # perturbation que la frame suivante réintroduisait aussitôt.
        if len(self._rayons) > self._min_frames:
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


def echelle_max_plausible(config: Config, dimension_px: int | None = None) -> float:
    """Échelle au-delà de laquelle **aucune** pièce ne peut être détectée.

    Dérivée de la porte de détection plutôt que posée en constante : au-delà
    de cette échelle, même la plus petite pièce de ``portee_mm`` dépasse le
    plafond de rayon de ``detection`` et est rejetée comme « trop grosse ».
    Accepter une telle échelle reviendrait à déclarer l'étalonnage réussi
    puis à ne plus rien voir.

    L'ancienne constante valait ``640 / 18,5 ≈ 34,6`` px/mm, soit trois à six
    fois trop haut : sur toute la partie supérieure de cette plage, chaque
    pièce de la portée était systématiquement rejetée par la détection. La
    borne vaut ici ≈ 11,6 px/mm pour une analyse de 640 px de large.
    """
    dimension = config.largeur_analyse if dimension_px is None else dimension_px
    rayon_max_px = config.seuils.rayon_max_rel * dimension
    return 2.0 * rayon_max_px / config.portee_mm[0]


def valider_echelle(
    config: Config, px_par_mm: float, dimension_px: int | None = None
) -> float:
    """Borne ``px_par_mm`` dans les plages plausibles, sinon erreur."""
    if not (isinstance(px_par_mm, (int, float)) and px_par_mm > 0.0):
        raise ErreurEtalonnage(
            f"échelle invalide : {px_par_mm!r} (px/mm doit être > 0)"
        )
    maximum = echelle_max_plausible(config, dimension_px)
    if px_par_mm < config.px_mm_min or px_par_mm > maximum:
        raise ErreurEtalonnage(
            f"échelle aberrante : {px_par_mm:.2f} px/mm hors plage "
            f"[{config.px_mm_min:.1f}, {maximum:.1f}]"
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


@dataclass(frozen=True)
class EchelleEstimee:
    """Échelle **estimée** depuis le catalogue — jamais une mesure.

    Distincte d'un étalonnage : personne n'a présenté d'objet de taille
    connue, l'échelle est inférée de la constellation des diamètres observés.
    Le type la garde donc explicitement séparée, pour que l'interface puisse
    dire « estimée » là où un étalonnage dirait « mesurée ».
    """

    px_par_mm: float
    #: nombre d'observations retombant sur une dénomination compatible
    inliers: int
    #: nombre d'observations fournies
    total: int
    #: écart absolu médian des inliers à leur dénomination (mm)
    ecart_median_mm: float


def estimer_echelle_catalogue(
    observations: Sequence[tuple[float, str | None, str | None]],
    config: Config,
    tolerance_mm: float = TOLERANCE_CATALOGUE_MM,
) -> EchelleEstimee | None:
    """Estime ``px/mm`` à partir des seuls diamètres observés.

    ``observations`` : ``(diametre_px, famille, polarite)`` par pièce, la
    famille venant du veto de couleur (``None`` si illisible).

    Principe : les pièces marocaines ne prennent qu'un petit nombre de
    diamètres connus. L'échelle est donc l'unique paramètre d'un ajustement
    par consensus — on essaie chaque rapport ``diamètre / variante``
    compatible, et on retient celui qui fait retomber le plus d'observations
    sur le catalogue, puis on l'affine aux moindres carrés sur ces inliers.

    **La famille contraint le consensus** : une observation ne peut s'aligner
    que sur une dénomination de sa propre famille. C'est décisif — mesuré sur
    la sortie réelle du détecteur, le consensus libre se trompe de 3,97 %
    (il fait basculer les 2 DH sur des 10 DH), le consensus contraint par la
    couleur tombe à 0,69 %.

    Retourne ``None`` si le résultat n'est pas crédible : moins de
    ``INLIERS_MIN_CATALOGUE`` inliers, ou une échelle hors des bornes
    physiques. Un comptage ne doit jamais reposer sur une échelle devinée à
    partir de deux disques.
    """
    mesures = [
        (float(d), fam, pol) for d, fam, pol in observations if d and d > 0.0
    ]
    if len(mesures) < INLIERS_MIN_CATALOGUE:
        return None

    def compatibles(fam: str | None, pol: str | None) -> list[float]:
        retenues: list[float] = []
        for denom in config.denominations:
            if fam is not None and fam != "inconnue" and denom.famille != fam:
                continue
            if (
                fam == "bimetallique"
                and pol is not None
                and denom.polarite != pol
            ):
                continue
            retenues.extend(denom.variantes_mm)
        return retenues

    candidates = sorted(
        {
            diam / mm
            for diam, fam, pol in mesures
            for mm in compatibles(fam, pol)
            if mm > 0.0
        }
    )
    if not candidates:
        return None

    def evaluer(echelle: float) -> tuple[int, float, list[float]]:
        ecarts: list[float] = []
        for diam, fam, pol in mesures:
            variantes = compatibles(fam, pol)
            if not variantes:
                continue
            ecart = min(abs(diam / echelle - mm) for mm in variantes)
            if ecart <= tolerance_mm:
                ecarts.append(ecart)
        return len(ecarts), sum(ecarts), ecarts

    meilleure = max(
        candidates, key=lambda e: (evaluer(e)[0], -evaluer(e)[1])
    )
    inliers, _, ecarts = evaluer(meilleure)
    if inliers < INLIERS_MIN_CATALOGUE:
        return None
    if len(_denominations_retenues(mesures, meilleure, compatibles, tolerance_mm)) \
            < DENOMINATIONS_MIN_CATALOGUE:
        return None

    # affinage : moindres carrés sur les seules observations retenues
    numerateur = denominateur = 0.0
    for diam, fam, pol in mesures:
        variantes = compatibles(fam, pol)
        if not variantes:
            continue
        proche = min(variantes, key=lambda mm: abs(diam / meilleure - mm))
        if abs(diam / meilleure - proche) > tolerance_mm:
            continue
        numerateur += diam * proche
        denominateur += proche * proche
    if denominateur > 0.0:
        meilleure = numerateur / denominateur

    if meilleure < config.px_mm_min:
        return None
    return EchelleEstimee(
        px_par_mm=float(meilleure),
        inliers=inliers,
        total=len(mesures),
        ecart_median_mm=float(statistics.median(ecarts)) if ecarts else 0.0,
    )


def _denominations_retenues(mesures, echelle, compatibles, tolerance_mm) -> set[float]:
    """Diamètres nominaux distincts sur lesquels les inliers retombent."""
    retenues: set[float] = set()
    for diam, fam, pol in mesures:
        variantes = compatibles(fam, pol)
        if not variantes:
            continue
        proche = min(variantes, key=lambda mm: abs(diam / echelle - mm))
        if abs(diam / echelle - proche) <= tolerance_mm:
            retenues.add(proche)
    return retenues


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
