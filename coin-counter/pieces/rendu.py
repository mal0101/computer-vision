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

__all__ = ["dessiner_hud", "couleur_piece"]

#: épaisseur de contour des cercles au plein cadre
EPAISSEUR = 4
#: police (OpenCV 3+ ; la taille est en fraction de hauteur pour rester
#: lisible à toutes les résolutions)
POLICE = cv2.FONT_HERSHEY_DUPLEX


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


def _texte(
    image: np.ndarray, s: str, org: tuple[int, int], couleur: tuple[int, int, int],
    hauteur: float,
) -> int:
    """Dessine une ligne et retourne la coordonnée y de la ligne suivante."""
    echelle_txt = hauteur / 720.0
    cv2.putText(
        image, s, org, POLICE, echelle_txt, couleur, 2, cv2.LINE_AA
    )
    return org[1] + int(28 * echelle_txt)


def dessiner_hud(
    image_pleine: np.ndarray,
    resultat: ResultatImage,
    etat: EtatApplication,
) -> np.ndarray:
    """Dessine le HUD sur une copie et la retourne (n'entoure pas l'input)."""
    image = image_pleine.copy()
    h, w = image.shape[:2]
    hauteur_hud = max(44.0, h * 0.08)

    # noirceur pour la lisibilité du panneau
    cv2.rectangle(image, (0, 0), (w, int(hauteur_hud)), (20, 20, 20), -1)

    y = 14

    # --- pièces détectées ------------------------------------------------
    for piece in resultat.pieces:
        cx = round(piece.centre[0] * resultat.echelle)
        cy = round(piece.centre[1] * resultat.echelle)
        r = round(piece.rayon * resultat.echelle)
        cv2.circle(image, (cx, cy), r, couleur_piece(piece), EPAISSEUR // 1, cv2.LINE_AA)
        cv2.circle(image, (cx, cy), max(1, r - EPAISSEUR // 2), (255, 255, 255), 1, cv2.LINE_AA)
        libelle = piece.classification.libelle if piece.classification else None
        if libelle:
            _texte(image, f"{libelle}  #{piece.piste_id}", (cx + r // 3, cy - r), (255, 255, 255), h)

    # --- lignes du bandeau ------------------------------------------------
    total = resultat.total_centimes if resultat.total_centimes is not None else 0
    y = _texte(image, f"TOTAL : {formater_montant(total)}", (14, y), (60, 200, 80), h)

    detail = etat.suiveur.par_valeur_total  # décompte consolidé de la session
    if detail:
        libelles = {d.valeur_centimes: d.libelle for d in etat.config.denominations}
        morceaux = "   ".join(
            f"{libelles.get(v, '?')} ×{n}" for v, n in sorted(detail.items())
        )
        y = _texte(image, morceaux, (14, y), (220, 220, 220), h)

    if etat.px_par_mm is not None:
        txt_echelon = f"échelle : {etat.px_par_mm:.1f} px/mm"
    else:
        txt_echelon = "non calibré"
    y = _texte(image, f"profil : {resultat.profil}   {txt_echelon}", (14, y), (0, 220, 220), h)

    if etat.mode_etalonnage:
        y = _texte(
            image,
            f"étalonnage : tenez une pièce de {etat.diametre_etalon_mm:.0f} mm "
            f"({etat.observateur.nb_frames if etat.observateur else 0}/8)",
            (14, y),
            (0, 140, 255),
            h,
        )

    # coin supérieur droit : métrologie
    fps = 1000.0 / resultat.analyse_ms if resultat.analyse_ms > 0 else float("nan")
    y2 = 14
    y2 = _texte(image, f"analyse : {resultat.analyse_ms:.1f} ms", (w - 300, y2), (255, 255, 255), h)
    y2 = _texte(image, f"{(fps if fps < 999 else 0.0):.0f} fps", (w - 300, y2), (255, 255, 255), h)

    # masque de profilage (touche D)
    if etat.afficher_masque and resultat.masque is not None:
        masque = resultat.masque
        legend = int(masque.shape[0] * 0.4)
        vignette_bin = np.where(masque > 0, 255, 0).astype(np.uint8)
        vignette = cv2.cvtColor(vignette_bin, cv2.COLOR_GRAY2BGR)
        lx, ly = 14, max(int(h - masque.shape[0] * 0.4 - 14), int(hauteur_hud))
        cv2.rectangle(image, (lx - 4, ly - 4), (lx + legend + 4, ly + legend + 4), (60, 60, 60), -1)
        image[ly : ly + legend, lx : lx + legend] = cv2.resize(
            vignette, (legend, legend), interpolation=cv2.INTER_NEAREST
        )
    return image
