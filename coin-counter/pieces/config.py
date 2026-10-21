"""Configuration : schéma, valeurs par défaut, chargement et validation.

Le fichier ``donnees/config.json`` est la source de vérité pour les seuils ;
le code définit les mêmes valeurs par défaut (testé par
``test_fichier_config_identique_aux_defauts``). Régénérer le fichier après
un changement de défaut :

    ../cv-app/.venv/bin/python -c "from pieces.config import CONFIG_DEFAUT, sauver_config; sauver_config(CONFIG_DEFAUT)"

Unités : angles en degrés, longueurs en millimètres ou en pixels à
l'échelle d'analyse, temps en millisecondes. Les montants sont des
**centimes entiers** — jamais de flottants monétaires.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .erreurs import ErreurConfiguration

__all__ = [
    "Denomination",
    "SeuilsDetection",
    "SeuilsProfil",
    "SuiviConfig",
    "Config",
    "DENOMINATIONS_DEFAUT",
    "CONFIG_DEFAUT",
    "VERSION_CONFIG",
    "chemin_config_defaut",
    "charger_config",
    "sauver_config",
    "valider_config",
    "config_vers_dict",
    "dict_vers_config",
    "denominations_actives",
    "formater_montant",
]

VERSION_CONFIG = 1


# ---------------------------------------------------------------------------
# Pièces marocaines — références Bank Al-Maghrib / Numista
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Denomination:
    """Une valeur faciale et ses variantes de diamètre (mm).

    ``variantes_mm`` contient les séries encore rencontrées : les diamètres
    ne sont pas monotones avec la valeur (2 DH 26 mm > 5 DH 25 mm > 1 DH
    24 mm), et 5 DH / 10 DH existent en plusieurs millésimes.

    Sources : Bank Al-Maghrib (série 2002) recoupées avec Numista/uCoin, puis
    **vérifiées par mesure** sur ``TP7/exercice 5/coins.jpg`` (ajustement aux
    moindres carrés sur les 7 dénominations connues : RMS 0,27 mm, écart
    maximal 0,39 mm — voir ``docs/02-classification.md``).
    """

    valeur_centimes: int
    libelle: str
    variantes_mm: tuple[float, ...]
    #: « or », « argent » ou « bimetallique » — veto absolu côté classification
    famille: str
    #: « centre_or » / « centre_argent » pour le bimétallique, sinon None
    polarite: str | None = None
    annees: str | None = None


#: Huit émissions ayant cours légal. L'ordre est celui de l'affichage HUD
#: (croissant de valeur). 5c/10c/20c ne sont comptés qu'avec --centimes.
#:
#: Trois corrections par rapport à la table d'origine, chacune vérifiée :
#:
#: * **5c (17,5 mm) ajouté** — la pièce existe, circule, et figure dans la
#:   fixture ``coins.jpg`` ; son absence la rendait définitivement
#:   « inconnue ». Mesure : 17,4–17,7 mm, spécification Numista : 17,5 mm.
#: * **5 DH : la variante 29,0 mm retirée** — le 5 DH « type 1980 » est
#:   *monométallique* cupronickel (argenté), pas bimétallique : l'inscrire
#:   sous une entrée ``bimetallique`` la rendait inidentifiable (le veto de
#:   couleur la classait « argent », famille sans variante à 29 mm).
#: * **10 DH : la variante 27,0 mm retirée** — aucune source (ni Bank
#:   Al-Maghrib ni Numista) ; à 0,1 mm de 26,9 mm elle n'était qu'un doublon
#:   élargissant silencieusement la fenêtre d'acceptation.
#:
#: Le 2 DH **reste « argent »**. La fiche web de Bank Al-Maghrib annonce à la
#: fois « Noyau Or / Bague Argent » et « Cuivre 75 % – Nickel 25 % » : ces
#: deux affirmations sont contradictoires, le cupronickel étant un alliage
#: unique de teinte argentée. Numista, uCoin et la mesure directe sur
#: ``coins.jpg`` (saturation 7–12 au centre *comme* sur la couronne, donc
#: aucune bimétallicité) concordent : la pièce est monométallique.
DENOMINATIONS_DEFAUT: tuple[Denomination, ...] = (
    Denomination(5, "5c", (17.5,), "or", None, "2002-"),
    Denomination(10, "10c", (20.0,), "or", None, "2002-"),
    Denomination(20, "20c", (23.0,), "or", None, "2002-"),
    Denomination(50, "1/2 DH", (21.0,), "argent", None, "2002-"),
    Denomination(100, "1 DH", (24.0,), "argent", None, "1965-"),
    Denomination(200, "2 DH", (26.0,), "argent", None, "2002-"),
    # 25,0 mm = série 2002/2011 ; 26,2 mm = série 1987 (toujours rencontrée)
    Denomination(500, "5 DH", (25.0, 26.2), "bimetallique", "centre_or", "1987-"),
    # 26,9 mm = série 2002/2011 ; 28,0 mm = série 1995
    Denomination(
        1000, "10 DH", (26.9, 28.0), "bimetallique", "centre_argent", "1995-"
    ),
)


# ---------------------------------------------------------------------------
# Seuils
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeuilsDetection:
    """Portes géométriques et réglages Hough (échelle d'analyse)."""

    #: circularité 4πA/P² — pièces réelles mesurées ≥ 0.86 (coins.jpg, la plus
    #: dégradée vaut 0.8596) ; capsules/carrés restent loin (≈ 0.78)
    circularite_min: float = 0.85
    #: Compacité aire/aire(convexe). Abaissée de 0,92 à 0,45 : elle mesure le
    #: *remplissage*, pas la rondeur, et une pièce dont la gravure creuse le
    #: masque y descend aussi bas qu'un croissant (mesuré : 0,527 et 0,582
    #: pour un nickel américain parfaitement rond). Elle ne sert plus que de
    #: garde-fou grossier ; la vraie séparation d'avec un croissant est faite
    #: par ``couverture_angulaire_min`` et ``excentricite_max`` ci-dessous.
    solidite_min: float = 0.45
    #: Fraction des secteurs angulaires (36 × 10°) autour du centre du cercle
    #: circonscrit où le contour est présent — « la forme entoure-t-elle son
    #: centre ? ». Mesuré : pièces réelles 0,889 à 1,000 ; croissant 0,722 ;
    #: croissant large 0,667 ; demi-disque 0,556.
    couverture_angulaire_min: float = 0.85
    #: Distance centroïde ↔ centre du cercle circonscrit, en fraction du
    #: rayon. Mesuré : pièces réelles ≤ 0,187 ; croissant 0,360 ;
    #: croissant large 0,334 ; demi-disque 0,350.
    excentricite_max: float = 0.25
    #: aire/aire(rectangle englobant) — un disque vaut π/4 ≈ 0.785 ; les petits
    #: disques pixellisés descendent vers ~0.66 ; en dessous de 0.62, la
    #: forme n'est plus un disque. L'étirement lui est surtout interdit par
    #: ``aspect_max``.
    etendue_min: float = 0.62
    #: rapport de la boîte — au-delà de 1.20, ce n'est pas un disque
    aspect_max: float = 1.20
    #: Rayon maxi / **largeur** d'analyse. Le dénominateur est la largeur et
    #: non la hauteur : c'est la seule dimension fixée par la réduction, donc
    #: la seule indépendante du format du capteur — avec la hauteur, le
    #: plafond variait d'un facteur 3,2 entre un cadre 1:1 et un 16:9 à champ
    #: horizontal identique, et rejetait en masse des pièces bien réelles.
    #:
    #: Ce plafond **n'est pas un rejet** : un disque isolé plus grand est
    #: toujours rendu comme détection (le faire disparaître coûtait de vraies
    #: pièces en gros plan). Il sert à décider si une silhouette mérite une
    #: tentative de séparation. C'est ``portee_mm``, exprimée en millimètres,
    #: qui empêche un objet trop grand d'être **compté** — une porte physique
    #: a un sens, un plafond en pixels n'en a que relativement au cadrage.
    #:
    #: L'ancien compagnon ``rayon_min_rel`` a été retiré : la largeur
    #: d'analyse étant fixe, un plancher relatif était rigoureusement
    #: équivalent au plancher absolu ``diametre_min_px``, et n'était lu nulle
    #: part — un utilisateur qui l'éditait ne changeait rien.
    rayon_max_rel: float = 0.15
    #: suppression des doublons : fusion si d < nms_diviseur × (r1+r2)/2
    nms_diviseur: float = 0.5
    #: |aire − πr²|/aire au-delà de quoi le disque est masqué/occlu
    occlusion_ecart_max: float = 0.15
    #: fraction de pixels saturés (> 250) dans la pièce → couleur illisible
    eclat_frac_max: float = 0.15
    #: fraction de FOND tolérée au centre d'une pièce. Au-delà, la silhouette
    #: est percée (rondelle, anneau, bouchon) : ``docs/01-portee.md`` promet
    #: cette porte, qui n'existait pas. Mesuré : vraies pièces ≤ 0,61 (une
    #: gravure sombre creuse le masque), anneau = 1,00 — le plancher à 0,85
    #: laisse 0,24 de marge aux pièces et 0,15 aux rondelles. La pièce percée
    #: est signalée, pas supprimée : si une pièce réelle y tombait un jour,
    #: elle serait affichée et non comptée — sous-compter vaut mieux que
    #: sur-compter de l'argent.
    creux_central_max: float = 0.85
    #: axe mineur / axe majeur de l'ellipse : < 0.90 = trop inclinée
    inclinaison_min: float = 0.90
    #: NOTE — l'ancien champ ``chevauchement_facteur`` (borne d'aire pour
    #: déclencher l'éclatement) a été retiré : la décision « cette silhouette
    #: est-elle un amas ? » se prend désormais en comparant son aire aux deux
    #: bornes contextuelles de ``detection`` (``_aire_eclatement_min`` sur deux
    #: pièces minimales, ``_aire_plausible_max`` sur ``AMAS_PIECES_MAX`` pièces
    #: maximales), et non plus à un facteur unique dépendant d'un rayon.


@dataclass(frozen=True)
class SeuilsProfil:
    """Déclencheurs des profils de prétraitement (ensemble fermé)."""

    #: fraction d'impulsions > seuil → profil « sel_poivre » : médian 3×3
    #: **avant** tout seuillage (le seuillage amplifie le bruit impulsif).
    #: Mesuré : images nettes ≤ 0,0004 ; sel & poivre 0,5 % ≥ 0,0026.
    impulsion_frac_max: float = 0.0015
    #: non-uniformité d'éclairage (quadrants) > seuil → profil « eclairage ».
    #: Mesuré : images saines ≤ 0,135 ; éclairage latéral marqué 0,402.
    #: L'ancienne valeur (1,5) était hors d'atteinte : le profil était mort.
    uniformite_max: float = 0.25
    #: variance du Laplacien < seuil → trop flou pour les contours stricts.
    #: Mesuré : scènes nettes ≥ 4 100 ; photographie molle 23,8 ; flou franc 8,6.
    laplacien_min: float = 30.0
    #: contraste RMS < seuil → profil « contraste » (CLAHE).
    #: Mesuré : images saines ≥ 0,083 ; scène à contraste écrasé 0,069.
    #: L'ancienne valeur (0,05) ne déclenchait sur aucune dégradation réelle.
    contraste_rms_min: float = 0.075


@dataclass(frozen=True)
class SuiviConfig:
    """Constantes du suivi de pistes — anti-scintillement des totaux."""

    #: confirmée après hits_confirmer présences sur fenetre_confirmer frames
    hits_confirmer: int = 3
    fenetre_confirmer: int = 5
    #: frames stables avant de figer le libellé d'une piste
    stabilite_requise: int = 2
    #: taille de la fenêtre de médiane sur les décisions de libellé
    mediane_libelle: int = 7
    #: piste perdue après max_manques manques dans fenetre_perdu frames
    fenetre_perdu: int = 10
    max_manques: int = 6
    #: lissage : v ← ema_alpha·v_nouv + (1−ema_alpha)·v_anc
    ema_alpha: float = 0.7
    #: association : d ≤ facteur_association·rayon + facteur_vitesse·‖v‖
    facteur_association: float = 0.35
    facteur_vitesse: float = 2.0
    #: rejet si |Δr|/r > portee_rayon_rel (changement d'identité)
    portee_rayon_rel: float = 0.15
    #: frames pendant lesquelles on se souvient d'une pièce DÉJÀ comptée,
    #: pour qu'un aller-retour hors champ ne la compte pas deux fois.
    #: 90 frames ≈ 3 s à 30 i/s : assez long pour couvrir une main qui passe,
    #: assez court pour qu'une pièce réellement reposée redevienne comptable.
    memoire_comptees: int = 90
    #: distance de reconnaissance d'une pièce mémorisée, en rayons
    rayon_memoire: float = 1.0


@dataclass(frozen=True)
class Config:
    """Configuration complète de l'application."""

    version: int = VERSION_CONFIG
    #: largeur de l'image d'analyse (px) : le flux est réduit à cette
    #: largeur avant détection ; le HUD est dessiné en plein cadre
    largeur_analyse: int = 640
    #: marge minimale entre la meilleure et la 2e note d'identification
    marge_classif: float = 0.20
    #: bornes du diamètre mesurable (mm) — au-delà : pas une pièce.
    #: La borne basse descend à 16,5 mm pour englober le 5c (17,5 mm) avec
    #: une marge ; l'ancienne valeur (18,5) excluait une pièce réelle.
    portee_mm: tuple[float, float] = (16.5, 30.5)
    #: échelle planimétrique minimale (px/mm à l'échelle d'analyse).
    #: Contrainte de cohérence (vérifiée par ``valider_config``) :
    #: ``px_mm_min × portee_mm[0] ≥ diametre_min_px``, sans quoi il existe une
    #: zone morte où une pièce passe l'étalonnage mais est rejetée par la
    #: porte « trop petit » de la détection. 2,5 × 16,5 = 41,25 ≥ 40.
    px_mm_min: float = 2.5
    #: diamètre minimal en pixels à l'échelle d'analyse
    diametre_min_px: float = 40.0
    #: 10c/20c ne sont comptés que si cette option est vraie
    centimes_par_defaut: bool = False
    seuils: SeuilsDetection = field(default_factory=SeuilsDetection)
    profils: SeuilsProfil = field(default_factory=SeuilsProfil)
    suivi: SuiviConfig = field(default_factory=SuiviConfig)
    denominations: tuple[Denomination, ...] = DENOMINATIONS_DEFAUT


CONFIG_DEFAUT = Config()


# ---------------------------------------------------------------------------
# Sérialisation
# ---------------------------------------------------------------------------


def config_vers_dict(config: Config) -> dict[str, object]:
    """Version JSON-compatible (tuples → listes)."""
    return {
        "version": config.version,
        "largeur_analyse": config.largeur_analyse,
        "marge_classif": config.marge_classif,
        "portee_mm": list(config.portee_mm),
        "px_mm_min": config.px_mm_min,
        "diametre_min_px": config.diametre_min_px,
        "centimes_par_defaut": config.centimes_par_defaut,
        "seuils": dict(vars(config.seuils)),
        "profils": dict(vars(config.profils)),
        "suivi": dict(vars(config.suivi)),
        "denominations": [
            {
                "valeur_centimes": d.valeur_centimes,
                "libelle": d.libelle,
                "variantes_mm": list(d.variantes_mm),
                "famille": d.famille,
                "polarite": d.polarite,
                "annees": d.annees,
            }
            for d in config.denominations
        ],
    }


def _champ(brut: dict[str, object], cle: str, attendu: type, fin: str) -> object:
    """Extrait un champ de type simple, avec message d'erreur localisé."""
    if cle not in brut:
        raise ErreurConfiguration(f"{fin} : champ « {cle} » manquant")
    valeur = brut[cle]
    if attendu is float:
        if isinstance(valeur, bool) or not isinstance(valeur, (int, float)):
            raise ErreurConfiguration(
                f"{fin}.{cle} doit être numérique, "
                f"reçu {type(valeur).__name__}"
            )
        return float(valeur)
    if attendu is int:
        if isinstance(valeur, bool) or not isinstance(valeur, int):
            raise ErreurConfiguration(
                f"{fin}.{cle} doit être un entier, reçu {type(valeur).__name__}"
            )
        return int(valeur)
    if not isinstance(valeur, attendu):
        raise ErreurConfiguration(
            f"{fin}.{cle} doit être de type {getattr(attendu, '__name__', attendu)}, "
            f"reçu {type(valeur).__name__}"
        )
    return valeur


def dict_vers_config(brut: object) -> Config:
    """Construit et **valide** une :class:`Config` depuis un objet JSON."""
    if not isinstance(brut, dict):
        raise ErreurConfiguration(
            f"configuration attendue comme objet JSON, reçu {type(brut).__name__}"
        )
    fin = "configuration"

    version = _champ(brut, "version", int, fin)
    largeur = _champ(brut, "largeur_analyse", int, fin)
    marge = _champ(brut, "marge_classif", float, fin)
    px_mm_min = _champ(brut, "px_mm_min", float, fin)
    diam_min = _champ(brut, "diametre_min_px", float, fin)
    centimes = _champ(brut, "centimes_par_defaut", bool, fin)

    portee = brut.get("portee_mm")
    if "portee_mm" not in brut:
        raise ErreurConfiguration("configuration : champ « portee_mm » manquant")
    if (
        not isinstance(portee, list)
        or len(portee) != 2
        or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in portee
        )
    ):
        raise ErreurConfiguration(
            "configuration.portee_mm doit être une liste de 2 nombres"
        )
    bornes = (float(portee[0]), float(portee[1]))

    def _section(cle: str, classe: type) -> object:
        section = brut.get(cle)
        if not isinstance(section, dict):
            raise ErreurConfiguration(
                f"configuration.{cle} doit être un objet, "
                f"reçu {type(section).__name__}"
            )
        # Le type attendu est déduit de la valeur par défaut : les compteurs
        # restent des entiers (3.0 ne doit pas entrer dans le suivi).
        ref = classe()
        valeurs = {
            nom_champ: _champ(
                section,
                nom_champ,
                float if isinstance(defaut, float) else int,
                f"configuration.{cle}",
            )
            for nom_champ, defaut in vars(ref).items()
        }
        return classe(**valeurs)

    seuils = _section("seuils", SeuilsDetection)
    profils = _section("profils", SeuilsProfil)
    suivi = _section("suivi", SuiviConfig)

    brut_denoms = brut.get("denominations")
    if not isinstance(brut_denoms, list) or not brut_denoms:
        raise ErreurConfiguration(
            "configuration.denominations doit être une liste non vide"
        )
    denomins: list[Denomination] = []
    for i, item in enumerate(brut_denoms):
        endroit = f"configuration.denominations[{i}]"
        if not isinstance(item, dict):
            raise ErreurConfiguration(f"{endroit} doit être un objet")
        variantes = item.get("variantes_mm")
        if (
            not isinstance(variantes, list)
            or not variantes
            or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool)
                for v in variantes
            )
        ):
            raise ErreurConfiguration(
                f"{endroit}.variantes_mm doit être une liste non vide de nombres"
            )
        polarite = item.get("polarite")
        if polarite is not None and not isinstance(polarite, str):
            raise ErreurConfiguration(f"{endroit}.polarite doit être une chaîne")
        annees = item.get("annees")
        if annees is not None and not isinstance(annees, str):
            raise ErreurConfiguration(f"{endroit}.annees doit être une chaîne")
        denomins.append(
            Denomination(
                valeur_centimes=_champ(item, "valeur_centimes", int, endroit),
                libelle=_champ(item, "libelle", str, endroit),
                variantes_mm=tuple(float(v) for v in variantes),
                famille=_champ(item, "famille", str, endroit),
                polarite=polarite,
                annees=annees,
            )
        )

    config = Config(
        version=version,  # type: ignore[arg-type]
        largeur_analyse=largeur,  # type: ignore[arg-type]
        marge_classif=marge,  # type: ignore[arg-type]
        portee_mm=bornes,
        px_mm_min=px_mm_min,  # type: ignore[arg-type]
        diametre_min_px=diam_min,  # type: ignore[arg-type]
        centimes_par_defaut=centimes,  # type: ignore[arg-type]
        seuils=seuils,  # type: ignore[arg-type]
        profils=profils,  # type: ignore[arg-type]
        suivi=suivi,  # type: ignore[arg-type]
        denominations=tuple(denomins),
    )
    valider_config(config)
    return config


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def valider_config(config: Config) -> None:
    """Lève :class:`ErreurConfiguration` si la configuration est incohérente."""
    if config.version != VERSION_CONFIG:
        raise ErreurConfiguration(
            f"version {config.version} non prise en charge "
            f"(attendu : {VERSION_CONFIG})"
        )
    if not 160 <= config.largeur_analyse <= 1920:
        raise ErreurConfiguration(
            f"largeur_analyse {config.largeur_analyse} hors bornes [160, 1920]"
        )
    if not 0.0 < config.marge_classif <= 1.0:
        raise ErreurConfiguration(
            f"marge_classif {config.marge_classif} hors bornes (0, 1]"
        )
    bas, haut = config.portee_mm
    if not 0.0 < bas < haut:
        raise ErreurConfiguration(
            f"portee_mm invalide ({bas}, {haut}) : il faut 0 < bas < haut"
        )
    if config.px_mm_min <= 0.0:
        raise ErreurConfiguration(f"px_mm_min {config.px_mm_min} doit être > 0")
    if config.diametre_min_px <= 0.0:
        raise ErreurConfiguration(
            f"diametre_min_px {config.diametre_min_px} doit être > 0"
        )
    # Zone morte : à l'échelle minimale acceptée, la plus petite pièce de la
    # portée doit encore franchir la porte « trop petit » de la détection.
    if config.px_mm_min * bas < config.diametre_min_px:
        raise ErreurConfiguration(
            f"incohérence : à px_mm_min = {config.px_mm_min} px/mm, une pièce "
            f"de {bas} mm mesure {config.px_mm_min * bas:.1f} px, en dessous "
            f"de diametre_min_px = {config.diametre_min_px} — elle serait "
            "étalonnable mais jamais détectable"
        )

    s = config.seuils
    bornes_seuils: list[tuple[str, float, tuple[float, float]]] = [
        ("circularite_min", s.circularite_min, (0.0, 1.0)),
        ("solidite_min", s.solidite_min, (0.0, 1.0)),
        ("couverture_angulaire_min", s.couverture_angulaire_min, (0.0, 1.0)),
        ("excentricite_max", s.excentricite_max, (0.0, 1.0)),
        ("etendue_min", s.etendue_min, (0.0, 1.0)),
        ("aspect_max", s.aspect_max, (1.0, 5.0)),
        ("rayon_max_rel", s.rayon_max_rel, (0.0, 1.0)),
        ("nms_diviseur", s.nms_diviseur, (0.0, 2.0)),
        ("occlusion_ecart_max", s.occlusion_ecart_max, (0.0, 1.0)),
        ("eclat_frac_max", s.eclat_frac_max, (0.0, 1.0)),
        ("creux_central_max", s.creux_central_max, (0.0, 1.0)),
        ("inclinaison_min", s.inclinaison_min, (0.0, 1.0)),
    ]
    for nom, valeur, (bas_b, haut_b) in bornes_seuils:
        if not bas_b < valeur <= haut_b:
            raise ErreurConfiguration(
                f"seuils.{nom} = {valeur} hors bornes ({bas_b}, {haut_b}]"
            )

    p = config.profils
    for nom, valeur in (
        ("impulsion_frac_max", p.impulsion_frac_max),
        ("uniformite_max", p.uniformite_max),
        ("laplacien_min", p.laplacien_min),
        ("contraste_rms_min", p.contraste_rms_min),
    ):
        if valeur <= 0.0:
            raise ErreurConfiguration(f"profils.{nom} = {valeur} doit être > 0")

    v = config.suivi
    if not 1 <= v.hits_confirmer <= v.fenetre_confirmer:
        raise ErreurConfiguration(
            f"suivi : hits_confirmer ({v.hits_confirmer}) doit être compris "
            f"entre 1 et fenetre_confirmer ({v.fenetre_confirmer})"
        )
    if not 1 <= v.stabilite_requise <= v.fenetre_confirmer:
        raise ErreurConfiguration(
            f"suivi : stabilite_requise ({v.stabilite_requise}) hors [1, "
            f"{v.fenetre_confirmer}]"
        )
    if v.mediane_libelle < 1:
        raise ErreurConfiguration("suivi : mediane_libelle doit être ≥ 1")
    if not 1 <= v.max_manques <= v.fenetre_perdu:
        raise ErreurConfiguration(
            f"suivi : max_manques ({v.max_manques}) doit être compris "
            f"entre 1 et fenetre_perdu ({v.fenetre_perdu})"
        )
    if not 0.0 < v.ema_alpha <= 1.0:
        raise ErreurConfiguration(f"suivi : ema_alpha {v.ema_alpha} hors (0, 1]")
    if v.facteur_association <= 0.0 or v.facteur_vitesse < 0.0:
        raise ErreurConfiguration("suivi : facteurs d'association invalides")
    if not 0.0 < v.portee_rayon_rel <= 1.0:
        raise ErreurConfiguration(
            f"suivi : portee_rayon_rel {v.portee_rayon_rel} hors (0, 1]"
        )
    if v.memoire_comptees < 0:
        raise ErreurConfiguration(
            f"suivi : memoire_comptees {v.memoire_comptees} doit être ≥ 0"
        )
    if v.rayon_memoire <= 0.0:
        raise ErreurConfiguration(
            f"suivi : rayon_memoire {v.rayon_memoire} doit être > 0"
        )

    if not config.denominations:
        raise ErreurConfiguration("aucune dénomination configurée")
    vues: set[int] = set()
    for d in config.denominations:
        if d.valeur_centimes <= 0:
            raise ErreurConfiguration(
                f"dénomination {d.libelle} : valeur {d.valeur_centimes} ≤ 0"
            )
        if d.valeur_centimes in vues:
            raise ErreurConfiguration(
                f"valeur {d.valeur_centimes} centimes configurée deux fois"
            )
        vues.add(d.valeur_centimes)
        if d.famille not in ("or", "argent", "bimetallique"):
            raise ErreurConfiguration(
                f"dénomination {d.libelle} : famille « {d.famille} » inconnue "
                "(attendu : or, argent, bimetallique)"
            )
        if d.famille == "bimetallique" and d.polarite not in (
            "centre_or",
            "centre_argent",
        ):
            raise ErreurConfiguration(
                f"dénomination {d.libelle} bimétallique : polarité obligatoire "
                "(centre_or ou centre_argent)"
            )
        if d.famille != "bimetallique" and d.polarite is not None:
            raise ErreurConfiguration(
                f"dénomination {d.libelle} : polarité réservée au bimétallique"
            )
        if any(mm <= 0.0 for mm in d.variantes_mm):
            raise ErreurConfiguration(
                f"dénomination {d.libelle} : les diamètres doivent être > 0"
            )
        if not all(bas <= mm <= haut for mm in d.variantes_mm):
            raise ErreurConfiguration(
                f"dénomination {d.libelle} : diamètre hors de portee_mm "
                f"({bas}, {haut}) — {d.variantes_mm}"
            )


# ---------------------------------------------------------------------------
# Chargement / sauvegarde
# ---------------------------------------------------------------------------


def chemin_config_defaut() -> Path:
    """``coin-counter/donnees/config.json`` (résolu depuis ce module)."""
    return Path(__file__).resolve().parents[1] / "donnees" / "config.json"


def charger_config(chemin: Path | str | None = None) -> Config:
    """Charge et valide la configuration. Lève :class:`ErreurConfiguration`."""
    cible = Path(chemin) if chemin is not None else chemin_config_defaut()
    if not cible.is_file():
        raise ErreurConfiguration(f"fichier de configuration introuvable : {cible}")
    try:
        brut = json.loads(cible.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ErreurConfiguration(f"lecture impossible : {cible} ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise ErreurConfiguration(
            f"JSON invalide dans {cible.name} (ligne {exc.lineno}, "
            f"colonne {exc.colno})"
        ) from exc
    return dict_vers_config(brut)


def sauver_config(config: Config, chemin: Path | str | None = None) -> Path:
    """Valide puis écrit ``config`` en JSON indenté (2 espaces)."""
    valider_config(config)
    cible = Path(chemin) if chemin is not None else chemin_config_defaut()
    cible.parent.mkdir(parents=True, exist_ok=True)
    cible.write_text(
        json.dumps(config_vers_dict(config), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return cible


# ---------------------------------------------------------------------------
# Aides métier
# ---------------------------------------------------------------------------


def denominations_actives(
    config: Config, centimes: bool | None = None
) -> tuple[Denomination, ...]:
    """Dénominations comptées selon l'option ``centimes``.

    ``centimes=None`` (défaut) utilise ``config.centimes_par_defaut`` ; les
    pièces de 5c/10c/20c ne sont comptées que si cette option est vraie.
    La valeur 50 (½ DH) est toujours comptée.

    L'*identification* reste exhaustive dans tous les cas : une pièce de 20c
    est toujours reconnue et affichée, elle n'est simplement pas ajoutée au
    total sans ``--centimes`` (cf. ``docs/02-classification.md``).
    """
    actif = config.centimes_par_defaut if centimes is None else bool(centimes)
    if actif:
        return config.denominations
    return tuple(d for d in config.denominations if d.valeur_centimes >= 50)


def formater_montant(centimes: int) -> str:
    """Affichage monétaire marocain : ``1250 → « 12,50 DH »`` (virgule FR)."""
    if not isinstance(centimes, int) or isinstance(centimes, bool):
        raise TypeError(
            f"montant attendu en centimes entiers, reçu {centimes!r}"
        )
    return f"{centimes // 100},{centimes % 100:02d} DH"
