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
from collections import Counter, deque
from dataclasses import dataclass, field

from .config import Config
from .entites import CERTITUDE_CONFIRME, PieceDetectee

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


def _valeur_comptable(piece: PieceDetectee) -> int | None:
    """Valeur d'une pièce, **uniquement** si son identification est confirmée.

    Deux axes portent ici des noms voisins et n'ont rien à voir :

    * ``Piste.phase`` ∈ {tentative, confirmee, …} décrit la **présence
      temporelle** — la pièce a-t-elle été revue assez souvent ?
    * ``Classification.certitude`` ∈ {inconnu, tentative, confirme} décrit la
      **confiance d'identification** — sait-on quelle pièce c'est ?

    Le comptage exigeait l'un et ignorait complètement l'autre. Une pièce
    simplement *tentative* (typiquement : aucune échelle, donc un
    bimétallique est le seul candidat de sa famille) était donc additionnée
    au total comme une pièce confirmée. C'est précisément ce qui fabriquait
    un total de 10,00 DH à partir d'une photographie non étalonnée, en
    contradiction avec ``docs/01-portee.md`` (« étalonnage obligatoire pour
    afficher un total confirmé ») et avec le README (« une pièce incertaine
    n'est jamais comptée »).
    """
    classification = piece.classification
    if classification is None:
        return None
    if classification.certitude != CERTITUDE_CONFIRME:
        return None
    return classification.valeur_centimes


@dataclass(frozen=True)
class _Empreinte:
    """Trace d'une piste déjà comptée, pour empêcher un second comptage."""

    centre: tuple[float, float]
    rayon: float
    valeur: int | None
    #: frame à laquelle la piste a été purgée (base de l'expiration)
    frame: int


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
        self._empreintes: list[_Empreinte] = []
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
        retenues = [p for p in pieces if p.motif_incertitude is None]
        appariement = self._apparier(retenues)
        for index, piece in enumerate(retenues):
            piste = appariement.get(index) or self._creer(piece)
            self._observer(piste, piece)
            piece.piste_id = piste.id
        self._transitionner()
        self._epurer()
        comptage = EtatComptage(
            pistes=list(self.pistes),
            total_centimes=self.total_centimes,
            par_valeur=dict(self._par_valeur),
            # Décompte CUMULÉ, cohérent avec total_centimes et par_valeur.
            # Il se lisait auparavant sur les pistes vivantes : dès qu'une
            # pièce comptée quittait le champ, nb_comptees retombait à 0 alors
            # que le total gardait sa valeur — deux chiffres irréconciliables
            # pour tout consommateur qui les croisait.
            nb_comptees=sum(self._par_valeur.values()),
            frame=num_frame,
        )
        return comptage

    def reinitialiser(self) -> None:
        self.pistes = []
        self._prochain_id = 1
        self.total_centimes = 0
        self._par_valeur = {}
        self.frame = 0
        self._empreintes = []

    @property
    def par_valeur_total(self) -> dict[int, int]:
        """Décompte consolidé (par valeur) de la session, pour le HUD."""
        return dict(self._par_valeur)

    # ------------------------------------------------------------------
    # Association
    # ------------------------------------------------------------------

    def _cout_association(
        self, piste: Piste, piece: PieceDetectee
    ) -> float | None:
        """Distance à la position prédite, ou None si l'appariement est exclu."""
        if piste.phase == PHASE_PERDUE:
            return None
        ecart = self.frame - piste.derniere_maj
        pred = (
            piste.centre[0] + piste.vitesse[0] * ecart,
            piste.centre[1] + piste.vitesse[1] * ecart,
        )
        dist = math.hypot(piece.centre[0] - pred[0], piece.centre[1] - pred[1])
        # Le rayon de recherche ne dépend QUE de la taille de la pièce.
        #
        # Il s'élargissait auparavant avec la vitesse (+2·‖v‖), ce qui le
        # portait de 17,5 px à l'arrêt à 77,5 px à 30 px/frame — plus large
        # que l'écart entre deux pièces voisines. Mesuré : deux pièces
        # distantes de 60 px se croisant échangeaient leurs identités à la
        # frame de croisement, et si l'une atteignait son comptage pendant
        # l'échange, la mauvaise dénomination était gelée définitivement.
        #
        # La vitesse appartient à la *prédiction* (ci-dessus), pas au rayon
        # de tolérance : une fois la position correctement prédite, le reste
        # n'est que du bruit de mesure, qui lui ne croît pas avec la vitesse.
        seuil = self.config.suivi.facteur_association * piste.rayon
        if dist > seuil:
            return None
        if piste.rayon > 0.0:
            dr = abs(piece.rayon - piste.rayon) / piste.rayon
            if dr > self.config.suivi.portee_rayon_rel:
                return None
        return dist

    def _apparier(
        self, pieces: list[PieceDetectee]
    ) -> dict[int, Piste]:
        """Appariement **exclusif** pièces → pistes, déterministe.

        Glouton sur la matrice des coûts triée : le meilleur couple d'abord,
        puis on retire la pièce *et* la piste du jeu. Rien n'empêchait
        jusqu'ici deux détections d'une même frame de réclamer la même
        piste — mesuré, `presence` recevait alors deux `True` pour une seule
        frame (la fenêtre de confirmation passait de 1 à 3 en deux frames au
        lieu de trois), le centre était lissé deux fois, et la seconde pièce
        écrasait silencieusement la classification de la première.
        """
        couples: list[tuple[float, int, int]] = []
        for index_piece, piece in enumerate(pieces):
            for index_piste, piste in enumerate(self.pistes):
                cout = self._cout_association(piste, piece)
                if cout is not None:
                    couples.append((cout, index_piece, index_piste))
        # tri total (coût, puis indices) : aucune ambiguïté d'ordre possible
        couples.sort()
        pieces_prises: set[int] = set()
        pistes_prises: set[int] = set()
        appariement: dict[int, Piste] = {}
        for _cout, index_piece, index_piste in couples:
            if index_piece in pieces_prises or index_piste in pistes_prises:
                continue
            pieces_prises.add(index_piece)
            pistes_prises.add(index_piste)
            appariement[index_piece] = self.pistes[index_piste]
        return appariement

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
        empreinte = self._deja_comptee(piece)
        if empreinte is not None:
            # même pièce physique qu'une piste déjà comptée : on reprend son
            # état « comptée » pour ne pas l'additionner une seconde fois
            piste.phase = PHASE_COMPTEE
            piste.comptee_a = self.frame
            piste.valeur = empreinte.valeur
            if empreinte.valeur is not None:
                piste.libelle = self._denoms.get(empreinte.valeur)
        self.pistes.append(piste)
        return piste

    def _observer(self, piste: Piste, piece: PieceDetectee) -> None:
        a = self.config.suivi.ema_alpha
        ancien_centre = piste.centre
        ecart = self.frame - piste.derniere_maj
        nouveau_centre = (
            a * piece.centre[0] + (1.0 - a) * ancien_centre[0],
            a * piece.centre[1] + (1.0 - a) * ancien_centre[1],
        )
        # Vitesse en px **par frame**, mesurée entre deux positions LISSÉES et
        # divisée par l'écart réel de frames.
        #
        # Auparavant : (détection brute − centre déjà lissé), sans division.
        # Deux défauts mesurés : à vitesse vraie constante de 10 px/frame la
        # valeur convergeait vers 14,3 (biais de rattrapage α/(1−α) ≈ 1,43),
        # et comme `_associer` multipliait ensuite par l'écart de frames, les
        # unités n'étaient homogènes que pour un écart de 1 — après quatre
        # frames d'absence l'extrapolation se trompait déjà de 3 px.
        if ecart > 0:
            piste.vitesse = (
                (nouveau_centre[0] - ancien_centre[0]) / ecart,
                (nouveau_centre[1] - ancien_centre[1]) / ecart,
            )
        piste.centre = nouveau_centre
        piste.rayon = a * piece.rayon + (1.0 - a) * piste.rayon
        piste.manques = 0
        piste.derniere_maj = self.frame
        piste._fraiche = True
        piste.presence.append(True)
        valeur = _valeur_comptable(piece)
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
        # Vote majoritaire, et non médiane.
        #
        # Une dénomination est une étiquette **catégorielle** : 100 et 1000
        # sont deux noms, pas deux points d'une échelle. La médiane y est un
        # non-sens arithmétique — `median([100, 1000])` vaut 550, qui n'est
        # aucune pièce et se faisait donc rejeter, si bien qu'une piste
        # hésitant entre deux lectures valides restait **éternellement** à
        # `valeur=None` (c'est l'origine des pistes bloquées en « confirmee »
        # sans valeur). Pire, sur un nombre impair la médiane peut élire une
        # minorité : sur les votes {100:1, 1000:2, 10:1, 20:1} elle désigne
        # 100, qui n'a qu'une voix, contre 1000 qui en a deux.
        #
        # Départage déterministe : à égalité de voix, la plus petite valeur
        # l'emporte — un choix arbitraire mais fixe, donc reproductible, et
        # prudent (il sous-estime plutôt qu'il ne surestime un total).
        comptes = Counter(valeurs)
        voix_max = max(comptes.values())
        tetes = [v for v, n in comptes.items() if n == voix_max]
        if len(tetes) > 1:
            # Égalité : on ne tranche PAS. Départager au hasard (ou par la
            # plus petite valeur) revient à décider à pile ou face, et comme
            # le comptage se déclenche dès que `valeur` est renseignée, une
            # égalité fortuite sur les deux premières frames gelait une
            # dénomination fausse avant même que les voix ne s'accumulent
            # (mesuré : une pièce majoritairement à 1000 comptée 100).
            # Une égalité signifie « on ne sait pas encore » : on attend.
            piste._stable = 0
            return
        candidat = tetes[0]
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
        """Retire les pistes perdues, en **mémorisant** celles déjà comptées.

        « Total monotone » ne veut pas dire « total juste » : il ne baissait
        jamais, mais rien n'empêchait la même pièce physique d'être comptée
        plusieurs fois. Deux chemins mesurés, menant chacun à 200 centimes
        pour une seule pièce :

        * elle sort du champ et revient — la piste était purgée après
          ``max_manques`` absences, la réapparition créait une piste neuve ;
        * son rayon dérive de plus de ``portee_rayon_rel`` (l'utilisateur la
          rapproche de la caméra) — l'association refuse, piste neuve.

        On garde donc une empreinte (centre, rayon, valeur) des pistes déjà
        comptées pendant ``memoire_comptees`` frames : une nouvelle piste qui
        lui ressemble reprend son identité au lieu d'ouvrir un second
        comptage. La protection est **bornée dans le temps** — au-delà de la
        fenêtre, une pièce réellement reposée sur la table est à nouveau
        comptable, ce qui est le comportement voulu pour un compteur en
        direct, et la limite est documentée plutôt que présentée comme une
        garantie absolue.
        """
        for piste in self.pistes:
            if piste.phase == PHASE_PERDUE and piste.comptee_a is not None:
                self._empreintes.append(
                    _Empreinte(
                        centre=piste.centre,
                        rayon=piste.rayon,
                        valeur=piste.valeur,
                        frame=self.frame,
                    )
                )
        self.pistes = [p for p in self.pistes if p.phase != PHASE_PERDUE]
        limite = self.frame - self.config.suivi.memoire_comptees
        self._empreintes = [e for e in self._empreintes if e.frame >= limite]

    def _deja_comptee(self, piece: PieceDetectee) -> _Empreinte | None:
        """Empreinte d'une pièce déjà comptée que ``piece`` vient ré-incarner.

        Deux sources, car une pièce peut se dédoubler de deux façons :

        * une piste comptée puis **purgée** (sortie du champ) — son empreinte
          survit ``memoire_comptees`` frames ;
        * une piste comptée **toujours vivante** dont l'association vient
          d'être refusée (dérive de rayon au-delà de ``portee_rayon_rel``,
          typiquement l'utilisateur rapproche la pièce de la caméra). Ne
          consulter que les empreintes purgées laissait ce second chemin
          ouvert : mesuré, une pièce dont le rayon grandit de 24 % était
          comptée deux fois.
        """
        cfg = self.config.suivi
        candidates = list(self._empreintes)
        candidates += [
            _Empreinte(p.centre, p.rayon, p.valeur, self.frame)
            for p in self.pistes
            if p.comptee_a is not None
        ]
        for empreinte in candidates:
            if empreinte.rayon <= 0.0:
                continue
            distance = math.hypot(
                piece.centre[0] - empreinte.centre[0],
                piece.centre[1] - empreinte.centre[1],
            )
            if distance > cfg.rayon_memoire * empreinte.rayon:
                continue
            return empreinte
        return None
