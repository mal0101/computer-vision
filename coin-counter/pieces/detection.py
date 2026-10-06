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
3. **morphologie** ouverture + fermeture 3×3 (noyau elliptique) ;
4. **contours externes**, triés par (aire, centroïde) pour garantir la
   déterminisme quel que soit le chemin interne d'OpenCV 5 (chemin TRUCO) ;
5. **portes géométriques** (circularité, solidité, étendue, aspect, plage de
   rayon) + portes d'incertitude (occlusion, éclat, inclinaison, trop petit) ;
6. **éclatement des chevauchements** (distanceTransform + watershed), refusé
   faute de parties valides ;
7. **Hough en contre-vote uniquement** : il confirme un candidat contour mais
   n'ajoute jamais de pièce et son rayon ne sert jamais (Hough échoue à
   trouver les bons rayons — mesuré) ;
8. **NMS** circulaire (fusion des doublons).

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

#: seuil |laplacien local| (pixel-à-pixel) pour compter une impulsion
LAPLACIEN_LOCAL_SEUIL = 100.0
#: découpage de l'image en 2×2 quadrants pour la mesure d'uniformité
QUADRANTS_N = 4
#: CLAHE : clipLimit et taille de grille (TP4 §5)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
#: noyau morphologique elliptique 3×3
MORPH_KERNEL_SIZE = 3
#: fitEllipse exige ≥ 5 points ; en dessous, le contour n'est pas fiable
ELLIPSE_MIN_POINTS = 5
#: tolérance d'aire lors de l'acceptation des parties d'un éclatement
WATERSHED_TOL_AIRE = 0.15
#: distance-transform : seuil relatif pour les marqueurs « cœur » des pièces.
    #: Réglé à 0.4 : deux pièces collées (sans soudure profonde) donnent deux
    #: cœurs distincts quand l'encoche de la transformée tombe en dessous ;
    #: une soudure profonde reste un seul cœur → motif « chevauchement ».
WATERSHED_DIST_SEUIL = 0.4
#: confiance des pièces selon la confirmation Hough (contre-vote)
CONF_AU_VOTE = 1.0
CONF_SANS_VOTE = 0.9
#: réduction de la grille de travail du contre-vote Hough (cf. _contre_vote_hough)
HOUGH_REPERE = 8


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
    """Fraction de pixels dont le laplacien local dépasse le seuil.

    Les impulsions (sel & poivre) produisent des valeurs extrêmes localisées
    que le seuillage adaptatif amplifie ensuite en taches. Au-delà de
    ``config.profils.impulsion_frac_max``, un médian 3×3 est appliqué
    AVANT tout seuillage.
    """
    lap = cv2.Laplacian(gris, cv2.CV_32F, ksize=3)
    return float(np.mean(np.abs(lap) > LAPLACIEN_LOCAL_SEUIL))


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
    """Préparation de l'image de seuillage selon le profil actif."""
    if profil == "sel_poivre":
        return cv2.medianBlur(gris, 3)
    if profil == "flou":
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
    """Ouverture puis fermeture 3×3 (noyau elliptique)."""
    noyau = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    ouvert = cv2.morphologyEx(masque, cv2.MORPH_OPEN, noyau)
    return cv2.morphologyEx(ouvert, cv2.MORPH_CLOSE, noyau)


# ---------------------------------------------------------------------------
# Géométrie d'un contour
# ---------------------------------------------------------------------------


def _geometrie(
    contour: np.ndarray, config: Config, hauteur: int,
    respecte_rayon_max: bool = True,
) -> dict | None:
    """Mesures d'un contour, ou None s'il ne franchit pas les portes géométriques."""
    aire = cv2.contourArea(contour)
    perimetre = cv2.arcLength(contour, True)
    if aire <= 0.0 or perimetre <= 0.0:
        return None
    circularite = 4.0 * np.pi * aire / (perimetre * perimetre)
    if circularite < config.seuils.circularite_min:
        return None
    coque = cv2.convexHull(contour)
    aire_coque = cv2.contourArea(coque)
    if aire_coque <= 0.0:
        return None
    solidite = aire / aire_coque
    if solidite < config.seuils.solidite_min:
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
    if respecte_rayon_max and rayon > config.seuils.rayon_max_rel * hauteur:
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


def _aire_chevauchement_max(config: Config, hauteur: int) -> float:
    """Aire à partir de laquelle un contour est candidat à l'éclatement."""
    rayon_max = config.seuils.rayon_max_rel * hauteur
    return config.seuils.chevauchement_facteur * np.pi * rayon_max * rayon_max


# ---------------------------------------------------------------------------
# Constructions des pièces
# ---------------------------------------------------------------------------


def _piece_depuis_geometrie(
    geometrie: dict,
    source: str,
    confiance: float,
    motif: str | None,
) -> PieceDetectee:
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


def _scinder(
    contour: np.ndarray, pre_gris: np.ndarray,
    config: Config, hauteur: int,
) -> list[PieceDetectee]:
    """distanceTransform + watershed pour séparer des pièces collées.

    Chaque partie n'est acceptée que si elle franchit toutes les portes ; il
    faut au moins deux parties valides, sans quoi on renvoie la pièce
    unique marquée « chevauchement ».
    """
    contour_mask = _masque_des_incs(np.zeros_like(pre_gris), contour)
    distance = cv2.distanceTransform(contour_mask, cv2.DIST_L2, 5)
    _, cœurs = cv2.threshold(
        distance, WATERSHED_DIST_SEUIL * float(distance.max() + 1e-9), 255,
        cv2.THRESH_BINARY,
    )
    cœurs = cœurs.astype(np.uint8)
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
            geometrie = _geometrie(sous, config, hauteur)
            if geometrie is None:
                continue
            geometrie = _motif_incertitude(geometrie, sous, pre_gris, config)
            if geometrie.get("motif_incertitude") is not None:
                continue  # partie ambiguë : refus de l'éclatement
            valides.append(
                _piece_depuis_geometrie(geometrie, "contour", CONF_SANS_VOTE, None)
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
        confiance=CONF_SANS_VOTE,
        motif_incertitude="chevauchement",
    )


def _motif_incertitude(
    geometrie: dict, contour: np.ndarray, gris: np.ndarray, config: Config
) -> dict:
    """Affecte un motif d'incertitude dans une copie de ``geometrie``.

    Ordre : occlusion (masqué/partiel) → éclat (couleur illisible) →
    inclinaison → trop petit. Une pièce incertaine est RENVOYÉE (visible au
    HUD) mais jamais comptée.
    """
    piece = dict(geometrie)
    aire, rayon = geometrie["aire"], geometrie["rayon"]
    if abs(aire - np.pi * rayon * rayon) / aire > config.seuils.occlusion_ecart_max:
        piece["motif_incertitude"] = "occlusion"
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
    config: Config, hauteur: int,
) -> list[PieceDetectee]:
    """Un contour → 0, 1 ou 2 pièces (chevauchement éclaté ou pas).

    Un disque isolé trop éloigné (Ø < ``diametre_min_px``) est signalé
    « trop petit » ; un objet trop gros pour être monnayé est soit refusé
    (disque unique près de la caméra) soit éclaté (plusieurs pièces soudées).
    """
    geometrie = _geometrie(contour, config, hauteur, respecte_rayon_max=False)
    if geometrie is None:
        return _tenter_eclatement(contour, gris, config, hauteur)
    if geometrie["rayon"] > config.seuils.rayon_max_rel * hauteur:
        return _tenter_eclatement(contour, gris, config, hauteur)
    geometrie = _motif_incertitude(geometrie, contour, gris, config)
    return [
        _piece_depuis_geometrie(
            geometrie, "contour", CONF_SANS_VOTE, geometrie.get("motif_incertitude")
        )
    ]


def _tenter_eclatement(
    contour: np.ndarray, gris: np.ndarray, config: Config, hauteur: int,
) -> list[PieceDetectee]:
    """Éclatement watershape si la forme dépasse la portée d'une pièce."""
    aire = float(cv2.contourArea(contour))
    if aire > _aire_chevauchement_max(config, hauteur):
        return _scinder(contour, gris, config, hauteur)
    return []


def _contre_vote_hough(
    pieces: list[PieceDetectee], gris: np.ndarray, config: Config, hauteur: int
) -> list[PieceDetectee]:
    """Hough en contre-vote : confirme un disjoint candidat, n'ajoute rien.

    Aucun cercle Hough sans correspondance n'est retenu, et le rayon Hough
    n'est jamais assigné (Hough échoue régulièrement sur les rayons, mesuré
    sur coins.jpg : param2 20 → 135 cercles fantômes).

    La transformée s'exécute sur une image réduite (``HOUGH_REPERE``) : elle
    ne sert qu'à confirmer la *position* d'un candidat déjà détecté par les
    contours. Perdu à pleine résolution à cause d'un coût > 1 s/frame
    (mesuré : 1 055 ms sur 640×640), le contre-vote sur grille ¼ concorde
    avec les 16 pièces de coins.jpg en ~10 ms tout en gardant la même
    sémantique (la confiance attribuée n'influence aucune porte).
    """
    valides = [p for p in pieces if p.motif_incertitude is None]
    if not valides:
        return pieces
    h, w = gris.shape[:2]
    k = HOUGH_REPERE
    grille = cv2.resize(gris, (max(1, w // k), max(1, h // k)),
                        interpolation=cv2.INTER_AREA) if k > 1 else gris
    rayons = np.array([p.rayon for p in valides])
    rayon_estime = float(np.median(rayons))
    min_dist = max(1.0, config.seuils.hough_min_dist_div * rayon_estime / k)
    try:
        cercles = cv2.HoughCircles(
            grille,
            cv2.HOUGH_GRADIENT,
            dp=config.seuils.hough_dp,
            minDist=min_dist,
            param1=config.seuils.hough_param1,
            param2=config.seuils.hough_param2,
        )
    except cv2.error:
        cercles = None
    centres_hough: list[tuple[float, float]] = []
    if cercles is not None:
        for (hx, hy, _rayon) in np.round(cercles[0]).astype(int):
            centres_hough.append((float(hx * k), float(hy * k)))

    for piece in valides:
        px, py = piece.centre
        confirme = any(
            (px - hx) ** 2 + (py - hy) ** 2 <= (piece.rayon * 0.5) ** 2
            for hx, hy in centres_hough
        )
        piece.confiance = CONF_AU_VOTE if confirme else CONF_SANS_VOTE
    return pieces


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
                aire_tot = fusionnee.aire + autre.aire
                cx = (fusionnee.centre[0] * fusionnee.aire
                      + autre.centre[0] * autre.aire) / max(aire_tot, 1e-9)
                cy = (fusionnee.centre[1] * fusionnee.aire
                      + autre.centre[1] * autre.aire) / max(aire_tot, 1e-9)
                motif = fusionnee.motif_incertitude if fusionnee.motif_incertitude is None else autre.motif_incertitude
                fusionnee = PieceDetectee(
                    centre=(float(cx), float(cy)),
                    rayon=max(fusionnee.rayon, autre.rayon),
                    aire=float(aire_tot),
                    source="contour",
                    confiance=max(fusionnee.confiance, autre.confiance),
                    axe_majeur=(
                        fusionnee.axe_majeur
                        if fusionnee.axe_majeur is not None else autre.axe_majeur
                    ),
                    axe_mineur=(
                        fusionnee.axe_mineur
                        if fusionnee.axe_mineur is not None else autre.axe_mineur
                    ),
                    angle=fusionnee.angle,
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
    config: Config, hauteur: int,
) -> tuple[list[PieceDetectee], int, int]:
    """pièces produites, nb de valides, nb de rejets pour UNE polarité."""
    contours = list(cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    contours.sort(key=_cle_deterministe)
    pieces: list[PieceDetectee] = []
    for contour in contours:
        pieces.extend(_traiter_contour(contour, masque, gris, config, hauteur))
    valides = sum(1 for p in pieces if p.motif_incertitude is None)
    rejets = len(pieces) - valides
    return pieces, valides, rejets


def _compter_grosses_taches(masque: np.ndarray, config: Config) -> int:
    """Composantes blanches d'aire plausible pour une pièce (polarité inverse).

    Borné inférieurement : toute pièce valide vit dans une composante d'aire
    ≥ π(r_min)², donc ce décompte borne le nombre de pièces d'une polarité.
    Sert à court-circuiter l'analyse complète de la polarité perdante (la
    moitié du coût de contour d'une frame, mesuré : −11 ms sur 640×640).
    """
    aire_min = np.pi * (config.diametre_min_px / 2.0) ** 2 * 0.95
    nb, _, stats, _ = cv2.connectedComponentsWithStats(masque, 8, cv2.CV_32S)
    if nb <= 1:
        return 0
    aires = stats[1:, cv2.CC_STAT_AREA]
    return int((aires >= aire_min).sum())


def detecter(image_analyse: np.ndarray, config: Config) -> ResultatDetection:
    """Détecte les pièces d'une image à l'échelle d'analyse.

    :param image_analyse: image BGR (H,W,3) ou grise (H,W), échelle d'analyse.
    :param config: configuration (seuils/profils).
    """
    gris = _a_gris(image_analyse)
    hauteur = gris.shape[0]
    profil = _choisir_profil(gris, config)
    pre = _pretraiter(gris, profil)
    normal, inversee = _seuiller(pre)
    norm_morph = _morphologie(normal)
    inv_morph = _morphologie(inversee)

    pieces_n, val_n, rej_n = _evaluer_polarite(norm_morph, gris, config, hauteur)
    masque, pieces = norm_morph, pieces_n

    # la polarité inverse ne mérite une analyse complète que si ses grosses
    # taches peuvent battre la normalité (borné : val_i ≤ grosses_inv)
    if _compter_grosses_taches(inv_morph, config) > val_n:
        pieces_i, val_i, rej_i = _evaluer_polarite(inv_morph, gris, config, hauteur)
        if val_i > val_n or (val_i == val_n and rej_i < rej_n):
            masque, pieces = inv_morph, pieces_i

    pieces = _contre_vote_hough(pieces, gris, config, hauteur)
    pieces = _nms(pieces, config)
    return ResultatDetection(pieces=pieces, profil=profil, masque=masque)
