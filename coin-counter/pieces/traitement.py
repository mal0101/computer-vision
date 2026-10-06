"""Traitement d'une frame (M3) : détection → classification → suivi.

`traiter_image(frame, etat)` est le squelette du pipeline appelé à chaque
frame de la webcam (et par les tests de bout en bout). Il travaille à
l'**échelle d'analyse** (réduction à ``config.largeur_analyse``) puis
restitue les coordonnées à l'échelle plein cadre via ``ResultatImage.echelle``
(facteur multiplicateur : coordonnée d'analyse × echelle = coordonnée du flux).
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .classification import classifier_pieces
from .config import Config
from .detection import detecter
from .entites import ResultatImage
from .etalonnage import (
    etalonner_par_piece,
    valider_echelle,
)
from .etat import EtatApplication

__all__ = ["traiter_image", "reduire_a_analyse"]


def reduire_a_analyse(
    frame: np.ndarray, config: Config
) -> tuple[np.ndarray, float]:
    """Réduit la frame brute à l'échelle d'analyse, retourne (image, echelle).

    ``echelle = largeur_brute / largeur_analyse`` ≥ 1 : multiplier par elle
    toute coordonnée d'analyse pour revenir au plein cadre.
    """
    echelle = frame.shape[1] / max(1.0, float(config.largeur_analyse))
    hauteur = max(1, round(frame.shape[0] / echelle))
    analyse = cv2.resize(
        frame,
        (config.largeur_analyse, hauteur),
        interpolation=cv2.INTER_AREA,
    )
    return analyse, echelle


def _actualiser_etalonnage(etat: EtatApplication, pieces) -> None:
    """Mode étalonnage : collecte le rayon de la plus grande pièce valide."""
    if not etat.mode_etalonnage or etat.observateur is None:
        return
    valides = [p for p in pieces if p.motif_incertitude is None]
    if not valides:
        return
    piece = max(valides, key=lambda p: p.rayon)
    etat.observateur.enregistrer(piece.rayon)
    if etat.px_par_mm is not None:
        return
    echelle = etalonner_par_piece(etat.observateur, etat.diametre_etalon_mm)
    if echelle is None:
        return
    try:
        etat.px_par_mm = valider_echelle(etat.config, echelle)
        etat.messages.append(
            f"Échelle étalonnée : {etat.px_par_mm:.1f} px/mm "
            f"(diamètre de référence {etat.diametre_etalon_mm:.0f} mm)"
        )
        etat.mode_etalonnage = False
        etat.observateur.reinitialiser()
    except Exception as err:  # noqa: BLE001 — message à l'utilisateur
        etat.messages.append(f"Étalonnage impossible : {err}")
        etat.observateur.reinitialiser()


def traiter_image(
    frame: np.ndarray, etat: EtatApplication
) -> ResultatImage:
    """Exécute le pipeline complet sur une frame BGR et retourne le résultat."""
    config = etat.config
    etat.num_frame += 1
    analyse, echelle = reduire_a_analyse(frame, config)

    debut = time.perf_counter()
    resultat = detecter(analyse, config)
    etat.analyse_ms = (time.perf_counter() - debut) * 1000.0

    classifier_pieces(resultat.pieces, analyse, config, etat.px_par_mm)
    _actualiser_etalonnage(etat, resultat.pieces)
    comptage = etat.suiveur.maj(resultat.pieces, etat.num_frame)

    return ResultatImage(
        pieces=resultat.pieces,
        profil=resultat.profil,
        echelle_px_mm=etat.px_par_mm,
        analyse_ms=etat.analyse_ms,
        echelle=echelle,
        messages=list(etat.messages),
        masque=resultat.masque > 0 if resultat.masque is not None else None,
        total_centimes=comptage.total_centimes,
    )
