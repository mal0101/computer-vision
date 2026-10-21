"""Rendu HUD (M3) : surcouche déterministe dessinée en plein cadre.

Le HUD montre ce qui a été **mesuré** : les cercles détectés (couleur par
certitude), le total (uniquement les pièces *comptées*), l'échelle
d'étalonnage ou « non calibré », le profil actif et la durée d'analyse.
Les libellés n'affichent que la valeur consolidée par le suivi — jamais
une hypothèse non stabilisée.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import formater_montant
from .entites import (
    CERTITUDE_CONFIRME,
    CERTITUDE_TENTATIVE,
    PieceDetectee,
    ResultatImage,
)
from .etat import EtatApplication

__all__ = ["dessiner_hud", "dessiner_hud_inventaire", "couleur_piece"]

#: épaisseur de contour des cercles au plein cadre
EPAISSEUR = 4
#: police (OpenCV 3+ ; la taille est en fraction de hauteur pour rester
#: lisible à toutes les résolutions)
POLICE = cv2.FONT_HERSHEY_DUPLEX


def _libelle_echelle(px_par_mm: float | None, estimee: bool) -> str:
    """Mention d'échelle : une estimation ne doit jamais passer pour une mesure."""
    if px_par_mm is None:
        return "non étalonné"
    return f"échelle : {px_par_mm:.2f} px/mm" + (" (estimée)" if estimee else "")


def dessiner_hud_inventaire(
    image_pleine: np.ndarray, inventaire, config
) -> np.ndarray:
    """HUD d'une image fixe : inventaire, sans notion de piste ni de session."""
    image = image_pleine.copy()
    libelles = {d.valeur_centimes: d.libelle for d in config.denominations}
    lignes = [
        (f"TOTAL : {formater_montant(inventaire.total_centimes)}", (60, 200, 80))
    ]
    if inventaire.par_valeur:
        lignes.append(
            (_detail_par_valeur(inventaire.par_valeur, libelles), (220, 220, 220))
        )
    lignes.append((
        f"profil : {inventaire.profil}   "
        f"{_libelle_echelle(inventaire.echelle_px_mm, inventaire.echelle_estimee)}",
        (0, 220, 220),
    ))
    hauteur_hud = _bandeau(image, lignes, inventaire.analyse_ms)
    _dessiner_pieces(image, inventaire.pieces, inventaire.echelle, hauteur_hud)
    return image


def couleur_piece(piece: PieceDetectee) -> tuple[int, int, int]:
    """Couleur de surcouche d'une pièce (BGR) selon sa certitude."""
    if piece.motif_incertitude is not None:
        return (110, 110, 110)  # gris : détectée mais non fiable
    if piece.classification is None:
        return (0, 165, 255)  # orange : identifiée mais non taggée
    if piece.classification.certitude == CERTITUDE_CONFIRME:
        return (60, 200, 80)  # vert
    if piece.classification.certitude == CERTITUDE_TENTATIVE:
        return (0, 220, 220)  # jaune
    return (0, 140, 255)  # orange foncé : UNKNOWN


def _echelle_texte(hauteur: int) -> float:
    """Facteur de police, borné pour rester lisible à toute résolution.

    Proportionnel à la hauteur (référence 720 px), mais encadré : en dessous
    de 0,45 le texte devient illisible sur une petite capture, au-dessus de
    1,6 il dévore une image 4K.
    """
    return float(min(1.6, max(0.45, hauteur / 720.0)))


def _texte(
    image: np.ndarray, s: str, org: tuple[int, int], couleur: tuple[int, int, int],
    hauteur: float, largeur_max: int | None = None,
) -> int:
    """Dessine une ligne et retourne la coordonnée y de la ligne suivante.

    Le texte est **tronqué** à la largeur disponible : ``cv2.putText`` ne sait
    ni couper ni replier, il écrit simplement dans le vide hors cadre, ce qui
    donnait des libellés coupés en plein mot.
    """
    echelle_txt = _echelle_texte(int(hauteur))
    limite = image.shape[1] if largeur_max is None else min(image.shape[1], largeur_max)
    disponible = max(0, limite - org[0] - 4)
    affiche = s
    while affiche:
        (w, _), _ = cv2.getTextSize(affiche, POLICE, echelle_txt, 2)
        if w <= disponible:
            break
        affiche = affiche[:-1]
    if affiche:
        cv2.putText(image, affiche, org, POLICE, echelle_txt, couleur, 2, cv2.LINE_AA)
    return org[1] + int(28 * echelle_txt)


def _dessiner_pieces(
    image: np.ndarray,
    pieces: list[PieceDetectee],
    echelle: float,
    hauteur_hud: float,
) -> None:
    """Cercle + libellé de chaque pièce, en restant dans le cadre."""
    h, w = image.shape[:2]
    for piece in pieces:
        cx = round(piece.centre[0] * echelle)
        cy = round(piece.centre[1] * echelle)
        r = round(piece.rayon * echelle)
        cv2.circle(image, (cx, cy), r, couleur_piece(piece), EPAISSEUR, cv2.LINE_AA)
        cv2.circle(
            image, (cx, cy), max(1, r - EPAISSEUR // 2), (255, 255, 255), 1,
            cv2.LINE_AA,
        )
        libelle = piece.classification.libelle if piece.classification else None
        if not libelle:
            continue
        if piece.piste_id is not None:
            libelle = f"{libelle}  #{piece.piste_id}"
        # Le libellé était posé en (cx + r/3, cy − r) sans aucune contrainte :
        # il passait sous le bandeau pour les pièces du haut et sortait du
        # cadre pour celles de droite. On le ramène dans la zone utile.
        ligne = int(24 * _echelle_texte(h))
        y = max(int(hauteur_hud) + ligne, cy - r)
        y = min(h - 4, y)
        x = max(4, min(cx + r // 3, w - 8))
        _texte(image, libelle, (x, y), (255, 255, 255), h)


def dessiner_hud(
    image_pleine: np.ndarray,
    resultat: ResultatImage,
    etat: EtatApplication,
) -> np.ndarray:
    """Dessine le HUD sur une copie et la retourne (n'entoure pas l'input)."""
    image = image_pleine.copy()
    h, w = image.shape[:2]
    libelles = {d.valeur_centimes: d.libelle for d in etat.config.denominations}
    total = resultat.total_centimes if resultat.total_centimes is not None else 0

    lignes: list[tuple[str, tuple[int, int, int]]] = [
        (f"TOTAL : {formater_montant(total)}", (60, 200, 80))
    ]
    detail = etat.suiveur.par_valeur_total  # décompte consolidé de la session
    if detail:
        lignes.append((_detail_par_valeur(detail, libelles), (220, 220, 220)))
    lignes.append((
        f"profil : {resultat.profil}   "
        f"{_libelle_echelle(etat.px_par_mm, etat.echelle_estimee)}",
        (0, 220, 220),
    ))
    if etat.mode_etalonnage:
        lignes.append((
            f"étalonnage : tenez une pièce de {etat.diametre_etalon_mm:.0f} mm "
            f"({etat.observateur.nb_frames if etat.observateur else 0}/8)",
            (0, 140, 255),
        ))
    # Les messages (étalonnage, remise à zéro…) étaient produits, transportés
    # jusqu'ici… et jamais dessinés : un étalonnage qui échouait ne disait
    # rien à l'écran.
    if resultat.messages:
        lignes.append((resultat.messages[-1], (0, 200, 255)))

    hauteur_hud = _bandeau(image, lignes, resultat.analyse_ms)
    _dessiner_pieces(image, resultat.pieces, resultat.echelle, hauteur_hud)
    _vignette_masque(image, resultat.masque, etat.afficher_masque, hauteur_hud)
    return image


def _detail_par_valeur(detail: dict[int, int], libelles: dict[int, str]) -> str:
    return "   ".join(
        f"{libelles.get(v, '?')} ×{n}" for v, n in sorted(detail.items())
    )


def _bandeau(
    image: np.ndarray,
    lignes: list[tuple[str, tuple[int, int, int]]],
    analyse_ms: float,
) -> float:
    """Dessine le bandeau et retourne sa hauteur réelle.

    La hauteur se déduit du **nombre de lignes** et de la taille de police,
    au lieu d'être posée à ``max(44, 8 % de la hauteur)`` : sur une image de
    800 px, trois lignes de ~31 px débordaient d'un bandeau de 64 px et le
    texte se superposait aux pièces. Le bloc de métrologie passe à la ligne
    sous le bloc principal quand l'image est trop étroite pour deux colonnes,
    au lieu d'être écrit à un ``w − 300`` fixe qui le faisait chevaucher le
    texte de gauche dès 640 px de large.
    """
    h, w = image.shape[:2]
    pas = int(30 * _echelle_texte(h))
    metrologie = [f"analyse : {analyse_ms:.1f} ms"]
    if analyse_ms > 0:
        metrologie.append(f"{1000.0 / analyse_ms:.0f} analyses/s")

    # Deux colonnes seulement si l'image est réellement assez large : à
    # 800 px avec une police proportionnelle, la colonne de droite recouvrait
    # encore la ligne de détail. En dessous, la métrologie passe simplement
    # sous le bloc principal.
    colonne_droite = w >= 1000
    nb_lignes = len(lignes) if colonne_droite else len(lignes) + len(metrologie)
    hauteur = max(44, 12 + nb_lignes * pas)
    cv2.rectangle(image, (0, 0), (w, int(hauteur)), (20, 20, 20), -1)

    marge = min(300, int(w * 0.42)) if colonne_droite else 0
    # la colonne de gauche s'arrête AVANT celle de droite, sinon les deux se
    # chevauchent dès qu'une ligne de détail est longue
    limite_gauche = (w - marge - 12) if colonne_droite else w
    y = 12 + int(pas * 0.75)
    for texte, couleur in lignes:
        _texte(image, texte, (14, y), couleur, h, largeur_max=limite_gauche)
        y += pas
    if colonne_droite:
        y_droite = 12 + int(pas * 0.75)
        for texte in metrologie:
            _texte(image, texte, (w - marge, y_droite), (255, 255, 255), h)
            y_droite += pas
    else:
        for texte in metrologie:
            _texte(image, texte, (14, y), (255, 255, 255), h)
            y += pas
    return float(hauteur)


def _vignette_masque(
    image: np.ndarray,
    masque: np.ndarray | None,
    afficher: bool,
    hauteur_hud: float,
) -> None:
    """Vignette du masque (touche D), toujours dans les limites du cadre.

    La taille se déduisait de celle du **masque** (échelle d'analyse) et était
    écrite telle quelle dans l'**image** (plein cadre) : dès que l'image était
    plus petite que la vignette calculée, l'affectation levait
    ``ValueError: could not broadcast input array`` — et comme rien n'attrape
    cette exception dans la boucle, une simple pression sur « d » sur une
    petite capture tuait toute la session.
    """
    if not afficher or masque is None:
        return
    h, w = image.shape[:2]
    disponible_h = int(h - hauteur_hud) - 20
    disponible_w = w - 28
    cote = min(int(masque.shape[0] * 0.4), disponible_h, disponible_w)
    if cote < 16:
        return  # pas la place : on n'affiche rien plutôt que de déborder
    vignette = cv2.cvtColor(
        np.where(masque > 0, 255, 0).astype(np.uint8), cv2.COLOR_GRAY2BGR
    )
    lx = 14
    ly = max(int(hauteur_hud) + 4, h - cote - 14)
    cv2.rectangle(image, (lx - 4, ly - 4), (lx + cote + 4, ly + cote + 4),
                  (60, 60, 60), -1)
    image[ly : ly + cote, lx : lx + cote] = cv2.resize(
        vignette, (cote, cote), interpolation=cv2.INTER_NEAREST
    )
