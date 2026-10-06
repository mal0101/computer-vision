"""Suivi de pistes (M3) : machine d'état anti-scintillement des totaux.

Une pièce devient visible (détectée + nommée), **tentative**, puis
**confirmée** après ``hits_confirmer`` présences sur une fenêtre de
``fenetre_confirmer`` frames. Elle n'entre dans le **total** qu'au passage
en ``comptee``, une seule fois, avec un libellé *consolidé* (médiane sur les
``mediane_libelle`` dernières décisions identiques pendant
``stabilite_requise`` frames). Une piste perdue n'est jamais dé-comptée : le
total est **monotone croissant** (aucun scintillement).

Règles de coopération avec la classification :

* une pièce ``motif_incertitude`` non nul (occultée, chevauchée, trop
  petite…) n'est ni associée ni comptée — le total reste figé ;
* une pièce identifiée ``inconnue`` (``valeur_centimes is None``) est
  *suivie* (elle reste affichée) mais n'ajoute rien au total ;
* le filtre ``centimes`` (10c/20c hors total sans ``--centimes``) agit ici,
  à l'étape de comptage, conformément à ``docs/02-classification.md``.

Association : distance centre prédite ≤ ``facteur_association·rayon +
facteur_vitesse·‖vitesse‖`` et ``|Δr|/r ≤ portee_rayon_rel``. Toutes les
opérations sont déterministes (parcours triés, tornade de choix greedy).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .entites import PieceDetectee

__all__ = [
    "Piste",
    "EtatComptage",
    "Suiveur",
    "PHASE_TENTATIVE",
    "PHASE_CONFIRMEE",
    "PHASE_COMPTEE",
    "PHASE_PERDUE",
]

PHASE_TENTATIVE = "tentative"
PHASE_CONFIRMEE = "confirmee"
PHASE_COMPTEE = "comptee"
PHASE_PERDUE = "perdue"


@dataclass
class Piste:
    """Mémoire temporelle d'une pièce vue dans le flux."""

    id: int
    centre: tuple[float, float]
    rayon: float
    vitesse: tuple[float, float] = (0.0, 0.0)
    phase: str = PHASE_TENTATIVE
    #: dernière valeur de classification observée (None si inconnu)
    derniere_valeur: int | None = None
    #: valeur du libellé consolidé (ou None tant qu'instable)
    valeur: int | None = None
    libelle: str | None = None
    manques: int = 0
    derniere_maj: int = 0
    comptee_a: int | None = None
    #: fenêtre glissante de présence sur fenetre_confirmer frames (bool)
    presence: deque = field(default_factory=deque, repr=False)
    #: décisions récentes (valeur ou None) pour la consolidation du libellé
    _recents: deque = field(default_factory=deque, repr=False)
    _candidat: int | None = None
    _stable: int = 0
    _fraiche: bool = True


@dataclass
class EtatComptage:
    """Recensement de la frame après exécution du suivi."""

    pistes: list[Piste]
    total_centimes: int
    par_valeur: dict[int, int]
    nb_comptees: int
    frame: int
    message: str | None = None


class Suiveur:
    """Machine à états des pistes ; instance unique par session de comptage."""

    def __init__(self, config: Config, compter_centimes: bool = False) -> None:
        self.config = config
        self.compter_centimes = bool(compter_centimes)
        self.pistes: list[Piste] = []
        self._prochain_id = 1
        self.total_centimes = 0
        self._par_valeur: dict[int, int] = {}
        self.frame = 0
        self._denoms = {
            d.valeur_centimes: d.libelle for d in config.denominations
        }

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def maj(self, pieces: list[PieceDetectee], num_frame: int) -> EtatComptage:
        """Met à jour les pistes depuis les pièces de la frame courante."""
        self.frame = num_frame
        # marque de fraîcheur : toute piste non associée recevra manques+1
        for piste in self.pistes:
            piste._fraiche = False
        for piece in pieces:
            if piece.motif_incertitude is not None:
                continue
            piste = self._associer(piece) or self._creer(piece)
            self._observer(piste, piece)
            piece.piste_id = piste.id
        self._transitionner()
        self._epurer()
        comptage = EtatComptage(
            pistes=list(self.pistes),
            total_centimes=self.total_centimes,
            par_valeur=dict(self._par_valeur),
            nb_comptees=len([p for p in self.pistes if p.phase == PHASE_COMPTEE]),
            frame=num_frame,
        )
        return comptage

    def reinitialiser(self) -> None:
        self.pistes = []
        self._prochain_id = 1
        self.total_centimes = 0
        self._par_valeur = {}
        self.frame = 0

    @property
    def par_valeur_total(self) -> dict[int, int]:
        """Décompte consolidé (par valeur) de la session, pour le HUD."""
        return dict(self._par_valeur)

    # ------------------------------------------------------------------
    # Association
    # ------------------------------------------------------------------

    def _associer(self, piece: PieceDetectee) -> Piste | None:
        meilleure = None
        meilleur_cout = math.inf
        for piste in self.pistes:
            if piste.phase == PHASE_PERDUE:
                continue
            pred = (
                piste.centre[0] + piste.vitesse[0] * (self.frame - piste.derniere_maj),
                piste.centre[1] + piste.vitesse[1] * (self.frame - piste.derniere_maj),
            )
            dx, dy = piece.centre[0] - pred[0], piece.centre[1] - pred[1]
            dist = math.hypot(dx, dy)
            vit = math.hypot(*piste.vitesse)
            seuil = (
                self.config.suivi.facteur_association * piste.rayon
                + self.config.suivi.facteur_vitesse * vit
            )
            if dist > seuil:
                continue
            if piste.rayon > 0.0:
                dr = abs(piece.rayon - piste.rayon) / piste.rayon
                if dr > self.config.suivi.portee_rayon_rel:
                    continue
            if dist < meilleur_cout:
                meilleur_cout = dist
                meilleure = piste
        return meilleure

    # ------------------------------------------------------------------
    # Construction / observation
    # ------------------------------------------------------------------

    def _creer(self, piece: PieceDetectee) -> Piste:
        piste = Piste(
            id=self._prochain_id,
            centre=(float(piece.centre[0]), float(piece.centre[1])),
            rayon=float(piece.rayon),
            derniere_maj=self.frame,
            presence=deque(maxlen=self.config.suivi.fenetre_confirmer),
            _recents=deque(maxlen=self.config.suivi.mediane_libelle),
        )
        self._prochain_id += 1
        self.pistes.append(piste)
        return piste

    def _observer(self, piste: Piste, piece: PieceDetectee) -> None:
        a = self.config.suivi.ema_alpha
        ancien_centre = piste.centre
        piste.vitesse = (
            piece.centre[0] - ancien_centre[0],
            piece.centre[1] - ancien_centre[1],
        )
        piste.centre = (
            a * piece.centre[0] + (1.0 - a) * ancien_centre[0],
            a * piece.centre[1] + (1.0 - a) * ancien_centre[1],
        )
        piste.rayon = a * piece.rayon + (1.0 - a) * piste.rayon
        piste.manques = 0
        piste.derniere_maj = self.frame
        piste._fraiche = True
        piste.presence.append(True)
        valeur = (
            piece.classification.valeur_centimes
            if piece.classification is not None
            else None
        )
        piste.derniere_valeur = valeur
        piste._recents.append(valeur)

    # ------------------------------------------------------------------
    # Transitions d'état
    # ------------------------------------------------------------------

    def _transitionner(self) -> None:
        cfg = self.config.suivi
        for piste in self.pistes:
            if piste.phase == PHASE_PERDUE:
                continue
            if not getattr(piste, "_fraiche", False):
                piste.manques += 1
                piste.presence.append(False)
            # confirmée : hits_confirmer présences sur fenetre_confirmer frames
            if piste.phase == PHASE_TENTATIVE and sum(piste.presence) >= cfg.hits_confirmer:
                piste.phase = PHASE_CONFIRMEE
            # libellé consolidé : médiane des valeurs côté candidates
            if piste.phase != PHASE_COMPTEE:
                self._consolider_libelle(piste)
            # comptage unique, au moment où la valeur est consolidée
            if (
                piste.phase == PHASE_CONFIRMEE
                and piste.valeur is not None
                and piste.manques == 0
                and self._comptable(piste.valeur)
                and piste.comptee_a is None
            ):
                piste.phase = PHASE_COMPTEE
                piste.comptee_a = self.frame
                self.total_centimes += piste.valeur
                self._par_valeur[piste.valeur] = (
                    self._par_valeur.get(piste.valeur, 0) + 1
                )
            # perdue
            if self._est_perdue(piste):
                piste.phase = PHASE_PERDUE

    def _consolider_libelle(self, piste: Piste) -> None:
        cfg = self.config.suivi
        valeurs = [v for v in piste._recents if v is not None]
        if not valeurs:
            piste._candidat = None
            piste._stable = 0
            return
        candidat = int(round(float(np.median(valeurs))))
        if candidat == piste._candidat:
            piste._stable += 1
        else:
            piste._candidat = candidat
            piste._stable = 1
        if (
            piste._stable >= cfg.stabilite_requise
            and piste._candidat is not None
            and piste._candidat in self._denoms
        ):
            piste.valeur = piste._candidat
            piste.libelle = self._denoms[piste._candidat]

    def _comptable(self, valeur: int) -> bool:
        return self.compter_centimes or valeur >= 50

    def _est_perdue(self, piste: Piste) -> bool:
        cfg = self.config.suivi
        if piste.manques >= cfg.max_manques:
            return True
        return (self.frame - piste.derniere_maj) > cfg.fenetre_perdu

    def _epurer(self) -> None:
        self.pistes = [p for p in self.pistes if p.phase != PHASE_PERDUE]
