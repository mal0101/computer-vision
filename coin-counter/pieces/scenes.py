"""Scènes synthétiques **métrologiquement exactes** (bancs sans caméra).

Pourquoi ce module existe : on ne peut pas vérifier un *total* sur une
photographie dont on ignore l'échelle. `coins.jpg` est une planche de
catalogue : excellente pour compter des disques, insuffisante pour prouver
une somme (l'ajustement aux moindres carrés y laisse 0,27 mm de RMS et un
écart systématique d'environ 2 % entre ses deux rangées — assemblage d'au
moins deux prises). Or la porte de classification se joue à 0,20 mm.

Ici, au contraire, l'échelle est **imposée** : une pièce de `d` mm est
dessinée à exactement `d × px_par_mm` pixels. Le total attendu est donc
connu *par construction*, et une assertion de comptage devient un vrai test.

Les scènes sont **déterministes** (générateur à graine fixe) : deux appels
identiques rendent des octets identiques.

Fidélité colorimétrique : les teintes sont choisies pour que la saturation
HSV des zones rendues tombe dans les plages **mesurées sur les pièces
réelles** de `coins.jpg` — or 78–95, argent 3–15, couronne bimétallique
62–66. C'est la saturation, pas la teinte, qui sépare les métaux (la teinte
d'une surface peu saturée est un bruit numérique), donc c'est elle que les
scènes doivent reproduire fidèlement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CONFIG_DEFAUT, Config, Denomination

__all__ = [
    "PiecePosee",
    "SceneSynthetique",
    "rendre_scene",
    "scene_etalon",
    "LAITON_BGR",
    "CUPRONICKEL_BGR",
]

#: laiton / bronze d'aluminium (familles « or ») — saturation HSV ≈ 85,
#: soit le milieu de la plage mesurée sur les pièces réelles (78–95)
LAITON_BGR = (139, 194, 208)
#: cupronickel / acier nickelé (famille « argent ») — saturation HSV ≈ 10
CUPRONICKEL_BGR = (183, 186, 190)
#: niveau du fond sombre et mat (conditions d'acquisition, docs/01)
FOND_SOMBRE = 30
#: grain de gravure, composante de **luminance** (écart-type, niveaux) :
#: identique sur les trois canaux, elle module le relief sans créer de teinte
GRAIN_LUMINANCE = 13.0
#: composante de **chrominance** (écart-type, niveaux) : volontairement
#: faible. Un bruit par canal de même amplitude que la luminance ferait
#: monter la saturation d'une surface grise à ~28 (mesuré) alors que les
#: pièces argentées réelles tiennent entre 3 et 15 : le cupronickel
#: paraîtrait coloré et la famille « argent » ne serait plus représentative.
GRAIN_CHROMA = 3.0
#: rayon relatif du noyau d'une pièce bimétallique
RAPPORT_NOYAU = 0.60


@dataclass(frozen=True)
class PiecePosee:
    """Une pièce placée sur la scène, en millimètres réels."""

    libelle: str
    #: centre en millimètres depuis le coin haut-gauche
    x_mm: float
    y_mm: float


@dataclass(frozen=True)
class SceneSynthetique:
    """Image rendue + sa vérité terrain (connue par construction)."""

    image: np.ndarray
    #: échelle imposée au rendu (px/mm) — c'est la vérité, pas une estimation
    px_par_mm: float
    #: somme des pièces posées, en centimes entiers
    total_centimes: int
    #: total restreint aux pièces de 50 centimes et plus (défaut sans --centimes)
    total_sans_centimes: int
    #: libellé → nombre d'exemplaires posés
    inventaire: dict[str, int]


def _denomination(config: Config, libelle: str) -> Denomination:
    for d in config.denominations:
        if d.libelle == libelle:
            return d
    connus = ", ".join(d.libelle for d in config.denominations)
    raise KeyError(f"dénomination inconnue : {libelle!r} (connues : {connus})")


def _poser_disque(
    image: np.ndarray,
    cx: float,
    cy: float,
    rayon: float,
    bgr: tuple[int, int, int],
    rng: np.random.Generator,
) -> None:
    """Disque métallique : teinte de base, ombrage radial, grain de gravure.

    L'ombrage et le grain ne sont pas cosmétiques : sans eux la teinte est
    parfaitement plate, la couverture vaut 1,0 partout et les portes de
    couleur ne sont jamais réellement sollicitées.
    """
    hauteur, largeur = image.shape[:2]
    y0, y1 = max(0, int(cy - rayon) - 1), min(hauteur, int(cy + rayon) + 2)
    x0, x1 = max(0, int(cx - rayon) - 1), min(largeur, int(cx + rayon) + 2)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    distance = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    dedans = distance <= rayon
    if not dedans.any():
        return
    # centre légèrement plus clair que le bord : éclairage diffus à ~45°
    ombrage = (1.06 - 0.20 * (distance / max(rayon, 1e-9)) ** 2)[..., None]
    # Luminance (commune aux 3 canaux) + une pointe de chrominance (par
    # canal). Sans la seconde, la teinte serait rigoureusement constante et
    # la couverture vaudrait 1,00 partout, si bien que les portes de couleur
    # ne seraient jamais sollicitées ; avec trop de seconde, le gris vire au
    # coloré. Le dosage reproduit les pièces réelles (couverture 0,73–1,00).
    grain = (
        rng.normal(0.0, GRAIN_LUMINANCE, size=distance.shape)[..., None]
        + rng.normal(0.0, GRAIN_CHROMA, size=(*distance.shape, 3))
    )
    teinte = np.asarray(bgr, dtype=np.float32)[None, None, :] * ombrage + grain
    zone = image[y0:y1, x0:x1].astype(np.float32)
    zone[dedans] = np.clip(teinte, 0.0, 255.0)[dedans]
    image[y0:y1, x0:x1] = zone.astype(np.uint8)


def _poser_piece(
    image: np.ndarray,
    denomination: Denomination,
    cx: float,
    cy: float,
    px_par_mm: float,
    rng: np.random.Generator,
) -> None:
    rayon = denomination.variantes_mm[0] * px_par_mm / 2.0
    if denomination.famille == "bimetallique":
        noyau_or = denomination.polarite == "centre_or"
        anneau = CUPRONICKEL_BGR if noyau_or else LAITON_BGR
        noyau = LAITON_BGR if noyau_or else CUPRONICKEL_BGR
        _poser_disque(image, cx, cy, rayon, anneau, rng)
        _poser_disque(image, cx, cy, rayon * RAPPORT_NOYAU, noyau, rng)
    else:
        metal = LAITON_BGR if denomination.famille == "or" else CUPRONICKEL_BGR
        _poser_disque(image, cx, cy, rayon, metal, rng)


def rendre_scene(
    pieces: list[PiecePosee],
    px_par_mm: float,
    taille: tuple[int, int] = (720, 1280),
    config: Config = CONFIG_DEFAUT,
    graine: int = 20261020,
    fond: int = FOND_SOMBRE,
) -> SceneSynthetique:
    """Rend une scène à l'échelle **imposée** ``px_par_mm`` (hauteur, largeur).

    Lève ``KeyError`` si un libellé est inconnu, et ``ValueError`` si une
    pièce déborde du cadre (une vérité terrain partiellement hors champ ne
    serait pas une vérité terrain).
    """
    if px_par_mm <= 0.0:
        raise ValueError(f"px_par_mm doit être > 0, reçu {px_par_mm!r}")
    hauteur, largeur = taille
    rng = np.random.default_rng(graine)
    image = np.full((hauteur, largeur, 3), fond, dtype=np.uint8)
    image = np.clip(
        image.astype(np.int16) + rng.integers(-4, 5, image.shape), 0, 255
    ).astype(np.uint8)

    total = 0
    inventaire: dict[str, int] = {}
    for posee in pieces:
        denomination = _denomination(config, posee.libelle)
        rayon = denomination.variantes_mm[0] * px_par_mm / 2.0
        cx, cy = posee.x_mm * px_par_mm, posee.y_mm * px_par_mm
        if cx - rayon < 0 or cy - rayon < 0 or cx + rayon > largeur or cy + rayon > hauteur:
            raise ValueError(
                f"{posee.libelle} en ({posee.x_mm}, {posee.y_mm}) mm déborde "
                f"du cadre {largeur}×{hauteur} px à {px_par_mm} px/mm"
            )
        _poser_piece(image, denomination, cx, cy, px_par_mm, rng)
        total += denomination.valeur_centimes
        inventaire[denomination.libelle] = inventaire.get(denomination.libelle, 0) + 1

    sans_centimes = sum(
        _denomination(config, lab).valeur_centimes * n
        for lab, n in inventaire.items()
        if _denomination(config, lab).valeur_centimes >= 50
    )
    return SceneSynthetique(
        image=image,
        px_par_mm=px_par_mm,
        total_centimes=total,
        total_sans_centimes=sans_centimes,
        inventaire=inventaire,
    )


def scene_etalon(
    px_par_mm: float = 4.5,
    config: Config = CONFIG_DEFAUT,
    graine: int = 20261020,
) -> SceneSynthetique:
    """Scène de référence : une pièce de chaque dénomination, bien séparées.

    Sert d'autotest du pipeline (`--sans-fenetre`) et de socle aux portes de
    comptage.

    L'échelle par défaut (4,5 px/mm) est choisie pour que la scène fasse
    639 px de large, soit juste en dessous de ``largeur_analyse`` : l'image
    n'est alors pas réduite, et l'échelle **imposée** est exactement celle que
    voit le détecteur. À 5 px/mm la scène faisait 710 px et la réduction
    ramenait l'échelle effective à 4,51 — un écart sans conséquence pour le
    pipeline, mais qui rendait incompréhensible la comparaison « imposée »
    contre « estimée » affichée par l'autotest.

    La plus petite pièce (5c, 17,5 mm) y mesure 79 px de diamètre, bien
    au-dessus de ``diametre_min_px``.
    """
    libelles = [d.libelle for d in config.denominations]
    pas_mm = 34.0  # > plus grand diamètre (26,9) + marge de séparation
    marge_mm = 20.0
    posees = [
        PiecePosee(lab, marge_mm + pas_mm * (i % 4), marge_mm + pas_mm * (i // 4))
        for i, lab in enumerate(libelles)
    ]
    largeur = int(round((marge_mm * 2 + pas_mm * 3) * px_par_mm))
    hauteur = int(round((marge_mm * 2 + pas_mm * ((len(libelles) - 1) // 4)) * px_par_mm))
    return rendre_scene(
        posees, px_par_mm, (hauteur, largeur), config=config, graine=graine
    )
