"""Traitement d'une frame (M3) : détection → classification → suivi.

`traiter_image(frame, etat)` est le squelette du pipeline appelé à chaque
frame de la webcam (et par les tests de bout en bout). Il travaille à
l'**échelle d'analyse** (réduction à ``config.largeur_analyse``) puis
restitue les coordonnées à l'échelle plein cadre via ``ResultatImage.echelle``
(facteur multiplicateur : coordonnée d'analyse × echelle = coordonnée du flux).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .classification import classifier_pieces, diametre_mesure
from .config import Config, denominations_actives
from .detection import detecter
from .entites import CERTITUDE_CONFIRME, PieceDetectee, ResultatImage
from .erreurs import ErreurEtalonnage
from .etalonnage import (
    EchelleEstimee,
    estimer_echelle_catalogue,
    etalonner_par_piece,
    valider_echelle,
)
from .etat import EtatApplication

#: nombre maximal de messages conservés dans l'état de session
MESSAGES_MAX = 20

__all__ = [
    "traiter_image",
    "reduire_a_analyse",
    "normaliser_frame",
    "inventorier_image",
    "estimer_echelle_frame",
    "Inventaire",
]


def normaliser_frame(frame: np.ndarray) -> np.ndarray:
    """Garantit une image BGR 3 canaux.

    ``detecter`` accepte une image en niveaux de gris, mais la classification
    y applique ensuite un ``cvtColor(BGR2HSV)`` inconditionnel et lève
    ``cv2.error: Bad number of channels``. La capacité annoncée par la
    détection était donc fausse au niveau du pipeline ; on convertit ici, une
    fois, plutôt que de promettre quelque chose qui casse plus loin.
    """
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def reduire_a_analyse(
    frame: np.ndarray, config: Config
) -> tuple[np.ndarray, float]:
    """Réduit la frame brute à l'échelle d'analyse, retourne (image, echelle).

    ``echelle = largeur_brute / largeur_analyse`` : multiplier par elle toute
    coordonnée d'analyse pour revenir au plein cadre.

    Une frame **plus étroite** que ``largeur_analyse`` n'est pas agrandie :
    interpoler ne crée aucune information, et ``INTER_AREA`` — un filtre de
    réduction — est de toute façon inadapté à l'agrandissement. L'ancienne
    version agrandissait systématiquement (une frame de 320 px donnait une
    échelle de 0,5), ce qui contredisait son propre contrat « echelle ≥ 1 ».
    L'invariant est désormais vrai : ``echelle ≥ 1`` toujours.
    """
    frame = normaliser_frame(frame)
    if frame.shape[1] <= config.largeur_analyse:
        return frame, 1.0
    echelle = frame.shape[1] / float(config.largeur_analyse)
    hauteur = max(1, round(frame.shape[0] / echelle))
    analyse = cv2.resize(
        frame,
        (config.largeur_analyse, hauteur),
        interpolation=cv2.INTER_AREA,
    )
    return analyse, echelle


@dataclass
class Inventaire:
    """Relevé d'**une** image : ce qui est visible maintenant.

    Distinct d'un total suivi. Le suivi est une machine temporelle : il exige
    ``hits_confirmer`` présences avant de compter, ce qui est indispensable
    en direct (une détection isolée est du bruit) mais rend un total
    structurellement impossible sur une image fixe — une seule frame ne peut
    pas produire trois présences. C'est l'une des causes du « 0,00 DH »
    historique sur ``--image``.

    Un inventaire répond donc à une autre question, légitime et vérifiable :
    « quelles pièces identifiez-vous *sur cette image*, et combien font-elles
    au total ? » Aucune confirmation temporelle n'y est requise, mais la
    confirmation d'**identification** l'est toujours : une pièce seulement
    « tentative » n'entre pas dans le total.
    """

    pieces: list[PieceDetectee]
    total_centimes: int
    par_valeur: dict[int, int]
    profil: str
    echelle_px_mm: float | None
    #: vrai si l'échelle vient du catalogue (estimée) et non d'un étalonnage
    echelle_estimee: bool
    analyse_ms: float
    echelle: float
    masque: np.ndarray | None = None
    messages: list[str] = field(default_factory=list)


def estimer_echelle_frame(
    pieces: list[PieceDetectee], config: Config
) -> EchelleEstimee | None:
    """Échelle déduite du catalogue pour les pièces d'une frame déjà classées."""
    observations = [
        (diametre_mesure(p), p.famille, p.polarite)
        for p in pieces
        if p.motif_incertitude is None
    ]
    return estimer_echelle_catalogue(observations, config)


def inventorier_image(
    frame: np.ndarray,
    config: Config,
    px_par_mm: float | None = None,
    auto_echelle: bool = True,
    centimes: bool | None = None,
) -> Inventaire:
    """Analyse **une** image et renvoie son inventaire (sans suivi).

    Si aucune échelle n'est fournie et que ``auto_echelle`` est vrai,
    l'échelle est estimée depuis le catalogue des diamètres marocains (voir
    ``etalonnage.estimer_echelle_catalogue``) : la classification peut alors
    confirmer, donc le total devient non nul — sans caméra, sans mire et sans
    intervention de l'utilisateur. Le drapeau ``echelle_estimee`` dit d'où
    vient le chiffre, pour que l'interface ne présente jamais une estimation
    comme une mesure.
    """
    analyse, echelle = reduire_a_analyse(frame, config)
    debut = time.perf_counter()
    resultat = detecter(analyse, config)
    analyse_ms = (time.perf_counter() - debut) * 1000.0

    messages: list[str] = []
    estimee = False
    if px_par_mm is None and auto_echelle:
        classifier_pieces(resultat.pieces, analyse, config, None)
        estimation = estimer_echelle_frame(resultat.pieces, config)
        if estimation is not None:
            px_par_mm = estimation.px_par_mm
            estimee = True
            messages.append(
                f"échelle estimée au catalogue : {px_par_mm:.2f} px/mm "
                f"({estimation.inliers}/{estimation.total} pièces concordantes)"
            )
        else:
            messages.append(
                "échelle non déterminée : trop peu de pièces reconnaissables "
                "— aucun total ne sera annoncé"
            )
    classifier_pieces(resultat.pieces, analyse, config, px_par_mm)

    actives = {d.valeur_centimes for d in denominations_actives(config, centimes)}
    par_valeur: dict[int, int] = {}
    total = 0
    for piece in resultat.pieces:
        valeur = _valeur_confirmee(piece)
        if valeur is None or valeur not in actives:
            continue
        par_valeur[valeur] = par_valeur.get(valeur, 0) + 1
        total += valeur
    return Inventaire(
        pieces=resultat.pieces,
        total_centimes=total,
        par_valeur=par_valeur,
        profil=resultat.profil,
        echelle_px_mm=px_par_mm,
        echelle_estimee=estimee,
        analyse_ms=analyse_ms,
        echelle=echelle,
        masque=resultat.masque > 0 if resultat.masque is not None else None,
        messages=messages,
    )


def _valeur_confirmee(piece: PieceDetectee) -> int | None:
    """Valeur d'une pièce dont l'identification est confirmée, sinon None."""
    if piece.motif_incertitude is not None:
        return None
    classification = piece.classification
    if classification is None or classification.certitude != CERTITUDE_CONFIRME:
        return None
    return classification.valeur_centimes


def _signaler(etat: EtatApplication, message: str) -> None:
    """Ajoute un message en évitant les répétitions et la croissance sans fin.

    ``etat.messages`` n'était jamais purgé et était recopié intégralement dans
    chaque ``ResultatImage``. Mesuré : un étalonnage qui échoue en boucle
    (l'observateur se réinitialise mais le mode reste actif) accumulait
    500 messages en 4 000 frames, tous recopiés à chaque frame.
    """
    if etat.messages and etat.messages[-1] == message:
        return
    etat.messages.append(message)
    if len(etat.messages) > MESSAGES_MAX:
        del etat.messages[:-MESSAGES_MAX]


def _actualiser_etalonnage(etat: EtatApplication, pieces) -> None:
    """Mode étalonnage : collecte le diamètre de la pièce de référence.

    Exige **une seule** pièce de taille plausible dans le champ. L'ancienne
    version prenait ``max(valides, key=rayon)`` : le plus gros objet visible
    devenait silencieusement la référence. Si l'utilisateur étalonnait avec
    un 1 DH (24 mm par défaut) alors qu'un objet plus grand traînait dans le
    cadre, l'échelle était faussée d'autant — un objet 21 % plus grand pris
    pour 24 mm dégonfle ensuite de 17 % toutes les mesures de la session, au
    point de faire lire 21,5 mm à une pièce de 2 DH (26 mm), soit un ½ DH.
    Mieux vaut refuser d'étalonner que d'étalonner sur la mauvaise pièce.
    """
    if not etat.mode_etalonnage or etat.observateur is None:
        return
    valides = [p for p in pieces if p.motif_incertitude is None]
    if not valides:
        return
    if len(valides) > 1:
        _signaler(
            etat,
            f"Étalonnage : {len(valides)} pièces dans le champ — n'en laissez "
            "qu'une seule, celle dont vous connaissez le diamètre",
        )
        etat.observateur.reinitialiser()
        return
    # même estimateur que la classification, sinon l'échelle est biaisée par
    # rapport à ce qu'elle sert à mesurer (cf. classification.diametre_mesure)
    etat.observateur.enregistrer(diametre_mesure(valides[0]) / 2.0)
    if etat.px_par_mm is not None:
        return
    try:
        echelle = etalonner_par_piece(etat.observateur, etat.diametre_etalon_mm)
        if echelle is None:
            return
        etat.px_par_mm = valider_echelle(etat.config, echelle)
    except ErreurEtalonnage as err:
        _signaler(etat, f"Étalonnage impossible : {err}")
        etat.observateur.reinitialiser()
        etat.mode_etalonnage = False  # ne pas boucler indéfiniment en échec
        return
    _signaler(
        etat,
        f"Échelle étalonnée : {etat.px_par_mm:.1f} px/mm "
        f"(diamètre de référence {etat.diametre_etalon_mm:.0f} mm)",
    )
    etat.mode_etalonnage = False
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

    # Sans échelle mesurée, on tente de l'estimer depuis le catalogue : c'est
    # ce qui permet à une session webcam de compter sans qu'on ait d'abord
    # présenté une pièce de référence. L'estimation est refaite tant qu'elle
    # n'a pas abouti, puis figée pour la session (elle ne doit pas osciller
    # d'une frame à l'autre, sans quoi les totaux deviendraient instables).
    if etat.px_par_mm is None and etat.auto_echelle and not etat.mode_etalonnage:
        estimation = estimer_echelle_frame(resultat.pieces, config)
        if estimation is not None:
            etat.px_par_mm = estimation.px_par_mm
            etat.echelle_estimee = True
            _signaler(
                etat,
                f"Échelle estimée au catalogue : {etat.px_par_mm:.2f} px/mm "
                f"({estimation.inliers}/{estimation.total} pièces concordantes)",
            )
            classifier_pieces(resultat.pieces, analyse, config, etat.px_par_mm)

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
