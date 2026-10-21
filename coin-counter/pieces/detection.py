"""Détection des pièces — chaîne « quel que soit le filtre ».

Pipeline (décision de conception validée, cf. docs/01-portee.md) :

1. **Profil fermé** sélectionné par mesures objectives (ensemble de 5 profils,
   jamais de recherche de paramètres) ;
2. **seuillage d'Otsu global** en DOUBLE polarité (BINARY et BINARY_INV →
   image inversée/`negatif`, fond obstruant). Le seuillage adaptatif a été
   essayé puis écarté pour le défaut : mesuré, il noie les surfaces
   localement plates (fond mat uniforme) dans le blanc — seules ses arêtes
   persistent. Otsu + prétraitement de profil (CLAHE/médian) tient les
   éclairages dégradés tout en gardant les disques pleins ;
3. **morphologie** : ouverture 3×3 (mouchetures) puis fermeture 7×7 (la
   gravure fragmente le disque seuillé ; la fermeture le recoud) ;
4. **contours externes**, triés par (aire, centroïde) pour garantir le
   déterminisme quel que soit le chemin interne d'OpenCV 5 (chemin TRUCO) ;
5. **portes géométriques** — circularité mesurée sur l'**enveloppe convexe**
   (le périmètre brut est instable : la gravure crée des anses qui
   l'allongent sans toucher l'aire), solidité, étendue, aspect, plage de
   rayon — puis portes d'incertitude (occlusion, éclat, inclinaison,
   trop petit) ;
6. **éclatement des chevauchements** (distanceTransform + watershed) tenté
   dès qu'une silhouette pourrait contenir deux pièces ; en cas d'échec elle
   ressort marquée « chevauchement », jamais silencieusement perdue ;
7. **NMS** circulaire : fusion des doublons, l'incertitude étant contagieuse.

Le contre-vote de Hough a été retiré : il consommait ~15 % du budget de la
frame pour écrire une confiance que personne ne lisait.

Déterminisme total : aucun appel aléatoire, ordre fixe en sortie.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .entites import PieceDetectee

__all__ = ["ResultatDetection", "detecter"]

#: profils reconnus — l'ensemble est fermé (pas de recherche au vol)
PROFILS = ("defaut", "sel_poivre", "contraste", "eclairage", "flou")

# ---------------------------------------------------------------------------
# Constantes techniques du prétraitement (pas des seuils métier : ceux-là
# vivent dans config.seuils / config.profils).
# ---------------------------------------------------------------------------

#: écart à la médiane 3×3 (niveaux) au-delà duquel un pixel est une impulsion
#: Réglé à 120 par mesure : sur huit images nettes (planche TP7, six
#: photographies réelles, scène synthétique) la fraction plafonne à 0,0004,
#: tandis qu'un sel & poivre à 0,5 % donne déjà 0,0026 — un facteur 7.
IMPULSION_ECART_MIN = 120
#: découpage de l'image en 2×2 quadrants pour la mesure d'uniformité
QUADRANTS_N = 4
#: CLAHE : clipLimit et taille de grille (TP4 §5)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
#: noyau d'OUVERTURE (élimine les mouchetures isolées) — 3×3 suffit
MORPH_OUVERTURE = 3
#: noyau de FERMETURE (recoud le disque fragmenté par la gravure) — 7×7.
#: La gravure d'une pièce (portrait, armoiries) découpe le disque seuillé en
#: morceaux : la solidité s'effondre (mesuré 0,40 à 0,73 pour un plancher à
#: 0,92) alors que l'enveloppe reste parfaitement ronde. Une fermeture plus
#: large recoud ces morceaux. Réglé par balayage sur les sept fixtures :
#: 3 → 47/62, **7 → 51/62**, 11 → 47/62, 15 → 28/62. Au-delà de 7 le noyau
#: commence à souder les pièces voisines de la planche TP7 (16 → 9 à k=11).
MORPH_FERMETURE = 7
#: fitEllipse exige ≥ 5 points ; en dessous, le contour n'est pas fiable
ELLIPSE_MIN_POINTS = 5
#: tolérance d'aire lors de l'acceptation des parties d'un éclatement
WATERSHED_TOL_AIRE = 0.15
#: pas de rééchantillonnage du contour, en pixels (cf. _reechantillonner)
PAS_REECHANTILLONNAGE = 1.0
#: nombre de secteurs angulaires pour la mesure de couverture (10° chacun)
SECTEURS_ANGULAIRES = 36
#: rayon du disque central sondé pour détecter un trou, en fraction du rayon
RAPPORT_CREUX_CENTRAL = 0.35
#: nombre de pièces de taille maximale qu'un amas plausible peut contenir
AMAS_PIECES_MAX = 6.0
#: distance-transform : seuil relatif pour les marqueurs « cœur » des pièces.
#: Réglé à 0.4 : deux pièces collées (sans soudure profonde) donnent deux
#: cœurs distincts quand l'encoche de la transformée tombe en dessous ;
#: une soudure profonde reste un seul cœur → motif « chevauchement ».
WATERSHED_DIST_SEUIL = 0.4
#: Confiance par défaut d'une détection issue des contours.
#:
#: Le contre-vote de Hough qui modulait autrefois ce champ a été supprimé :
#: il coûtait ~15 % du budget de la frame pour écrire une valeur que
#: personne ne lisait (vérifié : `confiance` n'est consultée nulle part
#: ailleurs dans le paquet) et n'influençait donc aucune décision.
#: La confiance est désormais le taux de remplissage du cercle circonscrit,
#: déjà calculé pour la porte d'occlusion : une mesure réelle, et gratuite.
CONFIANCE_DEFAUT = 1.0


@dataclass
class ResultatDetection:
    """Sortie de :func:`detecter` — pièces + profil + masque pour le HUD."""

    pieces: list[PieceDetectee]
    #: profil de prétraitement actif (« defaut », « sel_poivre », …)
    profil: str
    #: masque binaire du profil retenu (même taille que l'entrée)
    masque: np.ndarray | None


# ---------------------------------------------------------------------------
# Diagnostics (sélection du profil) — tous déterministes
# ---------------------------------------------------------------------------


def _a_gris(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _fraction_impulsions(gris: np.ndarray) -> float:
    """Fraction de pixels s'écartant fortement de la médiane locale.

    Mesure l'**impulsion** (sel & poivre) et non le détail : une impulsion est
    par définition un pixel que ses voisins contredisent, c'est-à-dire
    exactement ce qu'un filtre médian répare. On compare donc chaque pixel à
    ``medianBlur(3×3)``.

    L'ancienne version comptait les pixels de |laplacien| > 100, ce qui mesure
    le **détail** : une arête nette y contribue autant qu'un point parasite.
    Mesuré, elle rendait 0,120 à 0,257 sur sept photographies nettes (dont la
    planche ``coins.jpg``) pour un seuil de 0,02 — six à treize fois au-dessus.
    Le profil « sel_poivre » gagnait donc sur *toute* image nette et les
    profils « defaut », « eclairage » et « contraste » étaient inatteignables.
    """
    median = cv2.medianBlur(gris, 3)
    ecart = cv2.absdiff(gris, median)
    return float(np.mean(ecart > IMPULSION_ECART_MIN))


def _uniformite(gris: np.ndarray) -> float:
    """Écart-type normalisé des moyennes lumineuses des 4 quadrants.

    Éclairage latéral → un quadrant nettement différent des autres →
    ``uniformite > config.profils.uniformite_max`` → profil « eclairage ».
    """
    h, w = gris.shape
    demie_h, demie_w = h // 2, w // 2
    quadrants = (
        gris[:demie_h, :demie_w],
        gris[:demie_h, demie_w:],
        gris[demie_h:, :demie_w],
        gris[demie_h:, demie_w:],
    )
    moyennes = np.array([float(q.mean()) for q in quadrants], dtype=np.float64)
    centre = float(gris.mean()) + 1e-6
    return float(moyennes.std() / centre)


def _variance_laplacien(gris: np.ndarray) -> float:
    """Énergie globale de contraste local (perte de focus / bluergé)."""
    lap = cv2.Laplacian(gris, cv2.CV_64F, ksize=3)
    return float(lap.var())


def _contraste_rms(gris: np.ndarray) -> float:
    """Contraste RMS normalisé [0, 1] (TP7 §2)."""
    donnees = gris.astype(np.float64)
    return float(np.sqrt(np.mean((donnees - donnees.mean()) ** 2)) / 255.0)


def _choisir_profil(gris: np.ndarray, config: Config) -> str:
    """Profil actif, dans l'ordre de priorité : poivre > éclairage > flou > contraste."""
    if _fraction_impulsions(gris) > config.profils.impulsion_frac_max:
        return "sel_poivre"
    if _uniformite(gris) > config.profils.uniformite_max:
        return "eclairage"
    if _variance_laplacien(gris) < config.profils.laplacien_min:
        return "flou"
    if _contraste_rms(gris) < config.profils.contraste_rms_min:
        return "contraste"
    return "defaut"


def _pretraiter(gris: np.ndarray, profil: str) -> np.ndarray:
    """Préparation de l'image de seuillage selon le profil actif.

    Le profil « flou » ne prétraite **rien**, volontairement. Il reste un
    *diagnostic* (affiché au HUD : l'image est molle, la mise au point mérite
    d'être revue), mais aucun prétraitement mesuré ne l'améliore. Comparé sur
    une photographie réelle molle et sur trois floutages gaussiens (σ = 2, 4,
    7) de la scène étalon, le décompte est identique — 3/3 et 8/8 — que l'on
    laisse l'image brute, qu'on lui applique un médian 3×3 ou un rehaussement
    de netteté. Le seuillage d'Otsu est insensible à l'adoucissement du bord
    d'un disque bien séparé : seul le niveau compte, et le flou le préserve.

    L'ancienne version appliquait ici un médian 3×3, strictement identique au
    profil « sel_poivre » — donc adoucissait encore une image déjà molle, sans
    bénéfice mesurable.
    """
    if profil == "sel_poivre":
        return cv2.medianBlur(gris, 3)
    if profil in ("eclairage", "contraste"):
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
        return clahe.apply(gris)
    return gris


def _seuiller(pre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deux polarités du seuillage d'Otsu (fond uniforme → disques pleins)."""
    _, normal = cv2.threshold(pre, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, inversee = cv2.threshold(pre, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return normal, inversee


def _morphologie(masque: np.ndarray) -> np.ndarray:
    """Ouverture 3×3 puis fermeture 7×7 (noyaux elliptiques).

    Deux tailles distinctes car les deux opérations ne combattent pas le même
    défaut : l'ouverture retire les mouchetures du fond (petites), la
    fermeture recoud le disque fragmenté par la gravure (large).
    """
    noyau_ouv = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_OUVERTURE, MORPH_OUVERTURE)
    )
    noyau_fer = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_FERMETURE, MORPH_FERMETURE)
    )
    ouvert = cv2.morphologyEx(masque, cv2.MORPH_OPEN, noyau_ouv)
    return cv2.morphologyEx(ouvert, cv2.MORPH_CLOSE, noyau_fer)


# ---------------------------------------------------------------------------
# Géométrie d'un contour
# ---------------------------------------------------------------------------


def _reechantillonner(contour: np.ndarray) -> np.ndarray:
    """Points du contour rééchantillonnés à pas constant le long du bord.

    Indispensable à la couverture angulaire : ``CHAIN_APPROX_SIMPLE``
    compresse les segments droits, si bien que les sommets s'agglutinent sur
    les portions horizontales et verticales. Mesuré sur un **cercle
    parfait**, la couverture brute n'atteint alors que 0,667 — quel que soit
    le rayon — contre 1,000 une fois le bord parcouru à pas constant. Mesurer
    la densité de sommets plutôt que la géométrie aurait fait rejeter toutes
    les petites pièces.
    """
    points = contour.reshape(-1, 2).astype(np.float64)
    if len(points) < 2:
        return points
    boucle = np.vstack([points, points[:1]])
    segments = np.diff(boucle, axis=0)
    longueurs = np.hypot(segments[:, 0], segments[:, 1])
    cumul = np.concatenate([[0.0], np.cumsum(longueurs)])
    perimetre = cumul[-1]
    if perimetre <= 0.0:
        return points
    abscisses = np.arange(0.0, perimetre, PAS_REECHANTILLONNAGE)
    return np.column_stack(
        (
            np.interp(abscisses, cumul, boucle[:, 0]),
            np.interp(abscisses, cumul, boucle[:, 1]),
        )
    )


def _geometrie(
    contour: np.ndarray, config: Config, dimension: int,
    respecte_rayon_max: bool = True,
) -> dict | None:
    """Mesures d'un contour, ou None s'il ne franchit pas les portes géométriques.

    ``dimension`` est la **largeur** de l'image d'analyse : les seuils relatifs
    de rayon (``rayon_max_rel``) s'expriment en fraction de cette dimension.
    La largeur, et non la hauteur : la porte n'a de sens métrologique que si le
    dénominateur grandit avec la caméra, or c'est la largeur qui est fixée par
    la réduction (``config.largeur_analyse``) ; sur un cadre large la hauteur
    sous-estimait la portée et rejetait des pièces isolées bien réelles.
    """
    aire = cv2.contourArea(contour)
    perimetre = cv2.arcLength(contour, True)
    if aire <= 0.0 or perimetre <= 0.0:
        return None
    coque = cv2.convexHull(contour)
    aire_coque = cv2.contourArea(coque)
    perimetre_coque = cv2.arcLength(coque, True)
    if aire_coque <= 0.0 or perimetre_coque <= 0.0:
        return None
    # Circularité mesurée sur l'ENVELOPPE CONVEXE, pas sur le contour brut.
    #
    # 4πA/P² au périmètre brut est numériquement instable : la gravure d'une
    # pièce laisse des amas sombres que le suivi de bord contourne par de
    # fines anses, ce qui allonge P sans toucher A — et la circularité chute
    # au carré. Mesuré sur une pièce de coins.jpg à 640 : 331 points pour
    # 723,9 px de périmètre, contre 166 points et 386,9 px pour sa voisine de
    # même taille, soit 0,245 contre 0,895 à aire quasi égale.
    #
    # L'enveloppe convexe ignore ces anses. Mesuré sur 62 contours (planche
    # TP7 + six photographies réelles) : circularité d'enveloppe médiane
    # 0,997 (5ᵉ centile 0,959) contre 0,888 (5ᵉ centile 0,174) au brut ; le
    # brut rate la porte 34 % du temps, l'enveloppe 5 %. Le décompte au brut
    # variait de 11 à 15 selon le noyau morphologique, celui d'enveloppe
    # valait 16 pour tous les noyaux essayés.
    #
    # La convexité perdue est reprise par ``solidite`` (aire/aire d'enveloppe)
    # juste en dessous, et la rondeur vraie par la porte d'occlusion
    # (aire contre πr²), qui écarte les polygones : un hexagone y vaut 0,83
    # pour un plancher à 0,87, un carré 0,64.
    circularite = 4.0 * np.pi * aire_coque / (perimetre_coque * perimetre_coque)
    if circularite < config.seuils.circularite_min:
        return None
    solidite = aire / aire_coque
    if solidite < config.seuils.solidite_min:
        return None
    # La silhouette doit **entourer son propre centre**.
    #
    # C'est la propriété qui sépare réellement une pièce d'un croissant ou
    # d'un demi-disque, et ``solidite`` ne la mesurait qu'indirectement : elle
    # dit à quel point la forme remplit son enveloppe, si bien qu'une pièce
    # dont la gravure creuse le masque y tombe aussi bas qu'un croissant
    # (mesuré : 0,527 et 0,582 pour un nickel américain parfaitement rond,
    # contre 0,593 pour un croissant). Punir la gravure était le même défaut
    # que celui de la circularité au périmètre brut.
    #
    # Deux mesures directes, prises autour du centre du cercle circonscrit :
    #
    # * **couverture angulaire** — fraction des 36 secteurs de 10° où le
    #   contour est présent. Pièces réelles 0,889 à 1,000 ; croissant 0,722 ;
    #   croissant large 0,667 ; demi-disque 0,556.
    # * **excentricité** — distance centroïde ↔ centre du cercle circonscrit,
    #   en fraction du rayon. Pièces réelles ≤ 0,187 ; croissant 0,360 ;
    #   demi-disque 0,350.
    (cx_cercle, cy_cercle), rayon_circ = cv2.minEnclosingCircle(contour)
    if rayon_circ <= 0.0:
        return None
    points = _reechantillonner(contour)
    angles = np.arctan2(points[:, 1] - cy_cercle, points[:, 0] - cx_cercle)
    secteurs, _ = np.histogram(angles, bins=SECTEURS_ANGULAIRES, range=(-np.pi, np.pi))
    if float(np.mean(secteurs > 0)) < config.seuils.couverture_angulaire_min:
        return None
    moments = cv2.moments(contour)
    if moments["m00"] > 0.0:
        gx = moments["m10"] / moments["m00"]
        gy = moments["m01"] / moments["m00"]
        excentricite = float(np.hypot(gx - cx_cercle, gy - cy_cercle) / rayon_circ)
        if excentricite > config.seuils.excentricite_max:
            return None
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return None
    etendue = aire / float(w * h)
    if etendue < config.seuils.etendue_min:
        return None
    aspect = max(w, h) / float(min(w, h))
    if aspect > config.seuils.aspect_max:
        return None
    (cx, cy), rayon = cv2.minEnclosingCircle(contour)
    if rayon <= 0.0:
        return None
    if respecte_rayon_max and rayon > config.seuils.rayon_max_rel * dimension:
        return None  # objet trop gros pour être une pièce → non retenu
    elliptiques = cv2.fitEllipse(contour) if len(contour) >= ELLIPSE_MIN_POINTS else None
    axe_majeur = axe_mineur = angle = None
    if elliptiques is not None:
        ((_, _), (d1, d2), angle) = elliptiques
        axe_majeur, axe_mineur = max(float(d1), float(d2)), min(float(d1), float(d2))
    return {
        "aire": float(aire),
        "centre": (float(cx), float(cy)),
        "rayon": float(rayon),
        "axe_majeur": axe_majeur,
        "axe_mineur": axe_mineur,
        "angle": float(angle) if angle is not None else 0.0,
    }


def _cle_deterministe(contour: np.ndarray) -> tuple[float, float, float]:
    """Tri stable quel que soit le chemin interne d'OpenCV (TRUCO 5.0)."""
    moments = cv2.moments(contour)
    aire = float(moments["m00"])
    if aire > 0:
        cx = moments["m10"] / aire
        cy = moments["m01"] / aire
    else:
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w / 2.0, y + h / 2.0
    return (-aire, cx, cy)


def _aire_eclatement_min(config: Config) -> float:
    """Aire minimale d'un contour pour mériter une tentative d'éclatement.

    Seuil dimensionné sur **deux plus petites pièces**, et non sur la plus
    grosse pièce imaginable comme auparavant.

    L'ancienne borne valait ``chevauchement_facteur × π × (rayon_max)²``, soit
    43 429 px² à 640 de large. Or deux pièces de 70 px de rayon qui se
    touchent forment une silhouette de ~30 000 px² : elle échouait aux portes
    géométriques (circularité 0,569) *et* restait sous la borne, donc aucun
    éclatement n'était même tenté et les **deux pièces disparaissaient sans
    laisser de trace** — ni motif d'incertitude, ni rejet comptabilisé.

    Une silhouette capable de contenir deux pièces mérite qu'on essaie de la
    séparer ; si l'éclatement échoue, elle ressort marquée « chevauchement »
    (visible au HUD, jamais comptée) plutôt que d'être silencieusement perdue.
    """
    rayon_min = config.diametre_min_px / 2.0
    return 2.0 * np.pi * rayon_min * rayon_min


def _aire_plausible_max(config: Config, dimension: int) -> float:
    """Aire au-delà de laquelle un contour n'est plus un amas de pièces.

    Borne haute de bon sens : un amas vraisemblable tient dans quelques
    pièces de taille maximale. Au-delà (mur, plan de travail, main), on ne
    signale rien du tout — signaler un « chevauchement » de la taille de
    l'image n'aiderait personne.
    """
    rayon_max = config.seuils.rayon_max_rel * dimension
    return AMAS_PIECES_MAX * np.pi * rayon_max * rayon_max


# ---------------------------------------------------------------------------
# Constructions des pièces
# ---------------------------------------------------------------------------


def _confiance_remplissage(aire: float, rayon: float) -> float:
    """Taux de remplissage du cercle circonscrit, borné à [0, 1].

    Un disque parfaitement capté vaut ≈ 1 ; une silhouette rognée par une
    gravure mal recousue, une occultation ou un éclat descend d'autant. C'est
    une mesure réellement calculée, contrairement à la confiance « Hough »
    qu'elle remplace.
    """
    circonscrit = np.pi * rayon * rayon
    if circonscrit <= 0.0:
        return 0.0
    return float(min(1.0, aire / circonscrit))


def _piece_depuis_geometrie(
    geometrie: dict,
    source: str,
    confiance: float | None,
    motif: str | None,
) -> PieceDetectee:
    if confiance is None:
        confiance = _confiance_remplissage(geometrie["aire"], geometrie["rayon"])
    return PieceDetectee(
        centre=geometrie["centre"],
        rayon=geometrie["rayon"],
        aire=geometrie["aire"],
        source=source,
        confiance=confiance,
        axe_majeur=geometrie["axe_majeur"],
        axe_mineur=geometrie["axe_mineur"],
        angle=geometrie["angle"],
        motif_incertitude=motif,
    )


def _masque_des_incs(masque: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Masque binaire réduit à UN contour donné (255 puis binaire)."""
    solo = np.zeros_like(masque)
    cv2.drawContours(solo, [contour], -1, 255, -1)
    return solo


def _coeurs_pieces(masque_contour: np.ndarray) -> tuple[int, np.ndarray]:
    """Marqueurs « cœur » (distanceTransform) d'une silhouette, et leur nombre.

    Un disque plein possède **un** maximum unique de distance (une bosse
    centrale) ; deux pièces soudées en possèdent **deux**, séparés par une
    selle. Ce décompte est donc l'arbitre objectif entre « un disque » et
    « un amas » — avant toute tentative de séparation.

    :returns: (nb de cœurs, carte binaire des cœurs).
    """
    distance = cv2.distanceTransform(masque_contour, cv2.DIST_L2, 5)
    _, coeurs = cv2.threshold(
        distance, WATERSHED_DIST_SEUIL * float(distance.max() + 1e-9), 255,
        cv2.THRESH_BINARY,
    )
    coeurs = coeurs.astype(np.uint8)
    nb, _ = cv2.connectedComponents(coeurs)
    return nb - 1, coeurs


def _scinder(
    contour: np.ndarray, pre_gris: np.ndarray,
    config: Config, dimension: int,
) -> list[PieceDetectee]:
    """distanceTransform + watershed pour séparer des pièces collées.

    Chaque partie n'est acceptée que si elle franchit toutes les portes ; il
    faut au moins deux parties valides, sans quoi on renvoie la pièce
    unique marquée « chevauchement ».
    """
    contour_mask = _masque_des_incs(np.zeros_like(pre_gris), contour)
    _, cœurs = _coeurs_pieces(contour_mask)
    inconnu = cv2.subtract(contour_mask, cœurs)
    nb, marqueurs = cv2.connectedComponents(cœurs)
    marqueurs = marqueurs + 1
    marqueurs[inconnu > 0] = 0
    support = cv2.cvtColor(pre_gris, cv2.COLOR_GRAY2BGR)
    cv2.watershed(support, marqueurs)

    valides: list[PieceDetectee] = []
    for etiquette in range(2, nb + 1):
        partie = np.where(marqueurs == etiquette, 255, 0).astype(np.uint8)
        sous_contours, _ = cv2.findContours(partie, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
        for sous in sous_contours:
            geometrie = _geometrie(sous, config, dimension)
            if geometrie is None:
                continue
            geometrie = _motif_incertitude(geometrie, sous, pre_gris, config)
            if geometrie.get("motif_incertitude") is not None:
                continue  # partie ambiguë : refus de l'éclatement
            valides.append(
                _piece_depuis_geometrie(geometrie, "contour", None, None)
            )
    if len(valides) >= 2 and _aires_equilibrees(valides, config):
        return valides
    return [_piece_chevauchante(contour, pre_gris)]


def _aires_equilibrees(
    valides: list[PieceDetectee], config: Config
) -> bool:
    """Les parties éclatées doivent être individuellement plausibles (pas un
    amas d'éclats) : chaque aire ~ πr² avec sa propre portée d'occlusion.
    """
    for piece in valides:
        aire_attendu = np.pi * piece.rayon * piece.rayon
        if abs(piece.aire - aire_attendu) / aire_attendu > config.seuils.occlusion_ecart_max:
            return False
    return True


def _piece_chevauchante(contour: np.ndarray, gris: np.ndarray) -> PieceDetectee:
    aire = float(cv2.contourArea(contour))
    centre_brut = cv2.moments(contour)
    if centre_brut["m00"] > 0:
        cx = centre_brut["m10"] / centre_brut["m00"]
        cy = centre_brut["m01"] / centre_brut["m00"]
    else:
        cx, cy = gris.shape[1] / 2.0, gris.shape[0] / 2.0
    rayon = float(np.sqrt(aire / np.pi))
    return PieceDetectee(
        centre=(float(cx), float(cy)),
        rayon=rayon,
        aire=aire,
        source="contour",
        confiance=CONFIANCE_DEFAUT,
        motif_incertitude="chevauchement",
    )


def _creux_central(masque: np.ndarray, geometrie: dict) -> float:
    """Fraction de **fond** dans le disque central de la silhouette.

    Une rondelle, un anneau ou un bouchon percé ont un trou au milieu ; leur
    contour EXTERNE reste pourtant un cercle parfait, et ``contourArea`` d'un
    contour externe mesure la surface enfermée par le bord — le trou est donc
    rigoureusement invisible à toutes les autres portes (circularité,
    solidité, étendue, aspect, occlusion). Mesuré : un anneau franchissait
    chacune d'elles et ressortait comme pièce valide.

    ``docs/01-portee.md`` promet justement une « porte d'absence de trou
    central » ; elle n'existait nulle part dans le code.

    Mesure sur fixtures : les vraies pièces plafonnent à 0,61 (une gravure
    sombre au centre creuse le masque), un anneau vaut 1,00.
    """
    centre, rayon = geometrie["centre"], geometrie["rayon"]
    hauteur, largeur = masque.shape[:2]
    r = max(1, int(RAPPORT_CREUX_CENTRAL * rayon))
    x0, x1 = max(0, int(centre[0]) - r), min(largeur, int(centre[0]) + r + 1)
    y0, y1 = max(0, int(centre[1]) - r), min(hauteur, int(centre[1]) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dedans = (yy - centre[1]) ** 2 + (xx - centre[0]) ** 2 <= r * r
    zone = masque[y0:y1, x0:x1][dedans]
    return float(np.mean(zone == 0)) if zone.size else 0.0


def _motif_incertitude(
    geometrie: dict, contour: np.ndarray, gris: np.ndarray, config: Config,
    masque: np.ndarray | None = None,
) -> dict:
    """Affecte un motif d'incertitude dans une copie de ``geometrie``.

    Ordre : occlusion (masqué/partiel) → trou central (rondelle) → éclat
    (couleur illisible) → inclinaison → trop petit. Une pièce incertaine est
    RENVOYÉE — visible au HUD — mais jamais comptée.
    """
    piece = dict(geometrie)
    aire, rayon = geometrie["aire"], geometrie["rayon"]
    if abs(aire - np.pi * rayon * rayon) / aire > config.seuils.occlusion_ecart_max:
        piece["motif_incertitude"] = "occlusion"
        return piece
    if masque is not None and _creux_central(masque, geometrie) > config.seuils.creux_central_max:
        piece["motif_incertitude"] = "trou_central"
        return piece
    cx, cy = geometrie["centre"]
    disque = np.zeros_like(gris, dtype=bool)
    cv2.circle(disque, (int(round(cx)), int(round(cy))), max(1, int(0.9 * rayon)), 1, -1)
    pixels = gris[disque]
    if pixels.size and (pixels > 250).mean() > config.seuils.eclat_frac_max:
        piece["motif_incertitude"] = "eclat"
        return piece
    majeur, mineur = geometrie["axe_majeur"], geometrie["axe_mineur"]
    if (
        majeur is not None
        and mineur is not None
        and majeur > 0.0
        and mineur / majeur < config.seuils.inclinaison_min
    ):
        piece["motif_incertitude"] = "inclinaison"
        return piece
    if 2.0 * rayon < config.diametre_min_px:
        piece["motif_incertitude"] = "trop_petit"
    return piece


def _traiter_contour(
    contour: np.ndarray, masque: np.ndarray, gris: np.ndarray,
    config: Config, dimension: int,
) -> list[PieceDetectee]:
    """Un contour → 0, 1 ou 2 pièces (chevauchement éclaté ou pas).

    Un disque isolé trop éloigné (Ø < ``diametre_min_px``) est signalé
    « trop petit » ; un objet trop gros pour être monnayé est soit refusé
    (disque unique près de la caméra) soit éclaté (plusieurs pièces soudées).
    """
    geometrie = _geometrie(contour, config, dimension, respecte_rayon_max=False)
    if geometrie is None:
        # La forme elle-même est refusée (circularité, solidité, étendue,
        # aspect) : ce n'est pas une pièce. Ce peut en revanche être un AMAS
        # de pièces, dont la silhouette commune n'a aucune raison d'être
        # ronde — on tente donc la séparation, mais sans droit au
        # raccourci « cœur unique » réservé aux formes déjà valides.
        return _tenter_eclatement(
            contour, gris, config, dimension, geometrie=None, masque=masque
        )
    if geometrie["rayon"] > config.seuils.rayon_max_rel * dimension:
        # La forme est bonne, seul le rayon dépasse le plafond : soit une
        # grosse pièce en gros plan, soit plusieurs pièces soudées.
        return _tenter_eclatement(
            contour, gris, config, dimension, geometrie=geometrie, masque=masque
        )
    geometrie = _motif_incertitude(geometrie, contour, gris, config, masque)
    return [
        _piece_depuis_geometrie(
            geometrie, "contour", None, geometrie.get("motif_incertitude")
        )
    ]


def _tenter_eclatement(
    contour: np.ndarray, gris: np.ndarray, config: Config, dimension: int,
    geometrie: dict | None, masque: np.ndarray | None = None,
) -> list[PieceDetectee]:
    """Éclatement watershed d'une silhouette qui n'est pas une pièce unique.

    Trois issues possibles, et aucune n'est un silence : les parties valides
    si la séparation réussit, une pièce « chevauchement » si elle échoue sur
    une silhouette de taille plausible, et rien du tout seulement pour les
    objets hors de toute échelle monétaire.

    **Un seul cœur ⇒ ce n'est pas un amas**, mais seulement si la forme est
    par ailleurs valide. La distanceTransform d'une silhouette pleine n'a
    qu'un maximum en son centre ; deux pièces soudées en ont deux, séparés
    par une selle. Un disque isolé plus gros que ``rayon_max_rel`` (pièce
    tenue près de la caméra) n'est donc PAS un chevauchement : le marquer
    comme tel le faisait disparaître du décompte (mesuré sur ``jouet_6.jpg``
    : trois disques de circularité 0,998, un seul cœur chacun, étiquetés
    « chevauchement » — trois pièces perdues).

``geometrie`` conditionne ce raccourci : c'est le résultat des portes de
    forme déjà calculé par l'appelant, ou ``None`` si la silhouette les a
    échouées. Sans cette condition, **toute** silhouette d'aire plausible à
    cœur unique ressortait comme pièce pleinement valide, court-circuitant
    d'un coup circularité, solidité, étendue et aspect — mesuré : carré,
    ellipse 3:2, croissant, anneau et rectangle arrondi étaient tous comptés
    comme des pièces. Le raccourci ne lève que le plafond de **rayon**, et
    les motifs d'incertitude s'appliquent ensuite normalement.
    """
    aire = float(cv2.contourArea(contour))
    if aire < _aire_eclatement_min(config):
        return []  # trop petit pour contenir deux pièces
    if aire > _aire_plausible_max(config, dimension):
        return []  # hors de toute échelle monétaire : ce n'est pas un amas
    if geometrie is None:
        # forme refusée : ce n'est pas une pièce unique, seulement peut-être
        # un amas — on ne tente que la séparation
        return _scinder(contour, gris, config, dimension)
    nb_coeurs, _ = _coeurs_pieces(_masque_des_incs(np.zeros_like(gris), contour))
    if nb_coeurs <= 1:
        # Le raccourci lève le plafond de RAYON, rien d'autre : la pièce
        # repasse par les motifs d'incertitude comme n'importe quelle autre.
        #
        # Les sauter revenait à désactiver l'occlusion, l'éclat, l'inclinaison
        # et le trou central pour toute silhouette un peu grande. Mesuré : un
        # rectangle arrondi, suffisamment adouci par la fermeture
        # morphologique pour franchir les portes de forme, ressortait comme
        # pièce pleinement valide alors qu'il ne remplit que 52 % de son
        # cercle circonscrit — la porte d'occlusion l'aurait écarté.
        geometrie = _motif_incertitude(contour=contour, geometrie=geometrie,
                                       gris=gris, config=config, masque=masque)
        return [
            _piece_depuis_geometrie(
                geometrie, "contour", None, geometrie.get("motif_incertitude")
            )
        ]
    return _scinder(contour, gris, config, dimension)


def _nms(pieces: list[PieceDetectee], config: Config) -> list[PieceDetectee]:
    """Fusion des doublons proches (fermeture morphologique, scissions…)."""
    resultat: list[PieceDetectee] = []
    for piece in sorted(pieces, key=lambda p: (p.centre[1], p.centre[0], p.rayon)):
        fusionnee = piece
        i = 0
        while i < len(resultat):
            autre = resultat[i]
            dx = fusionnee.centre[0] - autre.centre[0]
            dy = fusionnee.centre[1] - autre.centre[1]
            distance = (dx * dx + dy * dy) ** 0.5
            seuil = config.seuils.nms_diviseur * (fusionnee.rayon + autre.rayon) / 2.0
            if distance < seuil:
                # Deux détections du MÊME disque : on garde la mesure la plus
                # fiable (la plus grande aire), on n'additionne pas. Sommer
                # les aires d'un doublon doublait l'aire d'un seul disque et
                # rendait le couple (aire, rayon) incohérent : mesuré, une
                # fusion donnait aire 15 092 pour rayon 50, soit un écart à
                # πr² de 0,479 — plus de trois fois la porte d'occlusion, qui
                # n'est jamais re-vérifiée après fusion.
                garde, jete = (
                    (fusionnee, autre)
                    if fusionnee.aire >= autre.aire
                    else (autre, fusionnee)
                )
                # L'incertitude est CONTAGIEUSE : si l'une des deux détections
                # est douteuse, la fusion l'est aussi. L'ancienne expression
                # rendait None dès que *l'un* des deux motifs était None, si
                # bien qu'une pièce incertaine fusionnée avec une pièce sûre
                # ressortait comptable — un blanchiment d'incertitude, dans le
                # sens exactement opposé à la règle « une pièce incertaine
                # n'est jamais comptée ».
                motif = garde.motif_incertitude or jete.motif_incertitude
                fusionnee = PieceDetectee(
                    centre=(float(garde.centre[0]), float(garde.centre[1])),
                    rayon=garde.rayon,
                    aire=garde.aire,
                    source="contour",
                    confiance=max(fusionnee.confiance, autre.confiance),
                    axe_majeur=(
                        garde.axe_majeur
                        if garde.axe_majeur is not None else jete.axe_majeur
                    ),
                    axe_mineur=(
                        garde.axe_mineur
                        if garde.axe_mineur is not None else jete.axe_mineur
                    ),
                    angle=garde.angle,
                    motif_incertitude=motif,
                )
                del resultat[i]
            else:
                i += 1
        resultat.append(fusionnee)
    resultat.sort(key=lambda p: (-p.aire, p.centre[0], p.centre[1], p.rayon))
    return resultat


def _evaluer_polarite(
    masque: np.ndarray, gris: np.ndarray,
    config: Config, dimension: int,
) -> tuple[list[PieceDetectee], int, int]:
    """pièces produites, nb de valides, nb de rejets pour UNE polarité."""
    contours = list(cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    contours.sort(key=_cle_deterministe)
    pieces: list[PieceDetectee] = []
    for contour in contours:
        pieces.extend(_traiter_contour(contour, masque, gris, config, dimension))
    valides = sum(1 for p in pieces if p.motif_incertitude is None)
    rejets = len(pieces) - valides
    return pieces, valides, rejets


def detecter(image_analyse: np.ndarray, config: Config) -> ResultatDetection:
    """Détecte les pièces d'une image à l'échelle d'analyse.

    :param image_analyse: image BGR (H,W,3) ou grise (H,W), échelle d'analyse.
    :param config: configuration (seuils/profils).
    """
    gris = _a_gris(image_analyse)
    dimension = gris.shape[1]  # largeur : dénominateur des seuils relatifs de rayon
    profil = _choisir_profil(gris, config)
    pre = _pretraiter(gris, profil)
    normal, inversee = _seuiller(pre)
    norm_morph = _morphologie(normal)
    inv_morph = _morphologie(inversee)

    # Les DEUX polarités sont analysées, sans court-circuit.
    #
    # Une sonde (`connectedComponentsWithStats`) servait à sauter l'analyse
    # complète de la polarité perdante. Remesurée, elle coûte 0,39 ms pour
    # éviter 0,15 ms de travail : elle **perd** 0,24 ms par frame. La
    # documentation lui prêtait une économie de 11 ms, mesurée sur une
    # version antérieure de la chaîne de contours ; la fermeture
    # morphologique plus large a depuis rendu les masques nettement plus
    # simples à parcourir, et la sonde est devenue un coût net.
    #
    # Son invariant était par ailleurs faux : elle prétendait **borner** le
    # nombre de pièces d'une polarité, ce qui cesse d'être vrai dès que
    # plusieurs pièces se touchent ou touchent le bord et ne forment plus
    # qu'une seule composante connexe — la polarité gagnante pouvait alors
    # être écartée sans jamais être évaluée.
    pieces_n, val_n, rej_n = _evaluer_polarite(norm_morph, gris, config, dimension)
    pieces_i, val_i, rej_i = _evaluer_polarite(inv_morph, gris, config, dimension)
    if val_i > val_n or (val_i == val_n and rej_i < rej_n):
        masque, pieces = inv_morph, pieces_i
    else:
        masque, pieces = norm_morph, pieces_n

    pieces = _nms(pieces, config)
    return ResultatDetection(pieces=pieces, profil=profil, masque=masque)
