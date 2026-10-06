"""Tests de la machine d'état de comptage (M3, suivi).

Ces tests pilotent `Suiveur.maj` frame par frame avec des pièces
synthétiques : ils ne touchent à aucune caméra, et vérifient les
propriétés anti-scintillement du total (monotone, figé, jamais négatif).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from pieces.config import CONFIG_DEFAUT
from pieces.entites import (
    CERTITUDE_CONFIRME,
    Classification,
    PieceDetectee,
)
from pieces.suivi import (
    PHASE_COMPTEE,
    PHASE_CONFIRMEE,
    PHASE_PERDUE,
    PHASE_TENTATIVE,
    Suiveur,
)


def _piece(
    valeur: int | None,
    rayon: float = 48.0,
    centre: tuple[float, float] = (100.0, 100.0),
    motif: str | None = None,
) -> PieceDetectee:
    piece = PieceDetectee(
        centre=centre,
        rayon=rayon,
        aire=np.pi * rayon * rayon,
        source="contour",
        axe_majeur=2 * rayon,
        axe_mineur=2 * rayon,
    )
    if motif is not None:
        piece.motif_incertitude = motif
    else:
        piece.classification = Classification(
            valeur, "?", 1.0, 0.1, CERTITUDE_CONFIRME
        )
    return piece


def _jouer(
    suiveur: Suiveur, sequence: list[list[PieceDetectee]], depart: int = 1
) -> list:
    """Joue la séquence de frames et renvoie les EtatComptage successifs."""
    etats = []
    for i, pieces in enumerate(sequence, start=depart):
        copies = [copy.deepcopy(p) for p in pieces]
        etats.append(copy.deepcopy(suiveur.maj(copies, i)))
    return etats


# ---------------------------------------------------------------------------
# Confirmation et comptage
# ---------------------------------------------------------------------------


def test_une_piece_confirmee_et_comptee():
    suiveur = Suiveur(CONFIG_DEFAUT)
    sequence = [[_piece(100)] for _ in range(6)]
    # 3 présences sur 5 fenêtres → confirmée à la 3ᵉ frame ; la valeur étant
    # déjà consolidée (mediane stable), la pièce est comptée au même passage
    etats = _jouer(suiveur, sequence)
    assert etats[0].pistes[0].phase == PHASE_TENTATIVE
    assert etats[1].pistes[0].phase == PHASE_TENTATIVE
    assert etats[2].pistes[0].phase in (PHASE_CONFIRMEE, PHASE_COMPTEE)
    comptee = [e for e in etats if e.pistes and e.pistes[0].phase == PHASE_COMPTEE]
    assert len(comptee) > 0
    premiere = comptee[0]
    assert premiere.par_valeur.get(100, 0) == 1
    assert premiere.total_centimes == 100
    # puis le total est figé
    for e in etats[etats.index(premiere):]:
        assert e.total_centimes == 100


def test_les_pieces_identifiees_entrent_au_total():
    suiveur = Suiveur(CONFIG_DEFAUT)
    etats = _jouer(
        suiveur,
        [[_piece(100, centre=(100.0, 100.0)), _piece(200, centre=(300.0, 200.0))] for _ in range(6)],
    )
    dernier = etats[-1]
    assert len(dernier.pistes) == 2
    assert dernier.total_centimes == 300
    assert dernier.par_valeur == {100: 1, 200: 1}
    assert dernier.nb_comptees == 2


# ---------------------------------------------------------------------------
# Valeurs inconnues et pièces incertaines
# ---------------------------------------------------------------------------


def test_piece_inconnue_jamais_comptee():
    suiveur = Suiveur(CONFIG_DEFAUT)
    etats = _jouer(suiveur, [[_piece(None)] for _ in range(10)])
    assert etats[-1].pistes[0].phase == PHASE_CONFIRMEE
    assert etats[-1].total_centimes == 0
    assert etats[-1].par_valeur == {}


def test_piece_motif_incertitude_non_comptee_non_suivie():
    suiveur = Suiveur(CONFIG_DEFAUT)
    sequence = [[_piece(None, motif="chevauchement")] for _ in range(5)]
    etats = _jouer(suiveur, sequence)
    assert etats[-1].pistes == []
    assert etats[-1].total_centimes == 0


def test_une_piece_inconnue_apres_une_comptee_ne_vide_pas_le_total():
    suiveur = Suiveur(CONFIG_DEFAUT)
    seq = [[_piece(100)] for _ in range(5)] + [[_piece(None)] for _ in range(5)]
    etats = _jouer(suiveur, seq)
    assert etats[-1].total_centimes == 100  # déjà comptée, gardée


# ---------------------------------------------------------------------------
# Filtre « centimes »
# ---------------------------------------------------------------------------


def test_20c_hors_total_par_defaut():
    suiveur = Suiveur(CONFIG_DEFAUT, compter_centimes=False)
    etats = _jouer(suiveur, [[_piece(20)] for _ in range(6)])
    assert etats[-1].total_centimes == 0


def test_20c_comptee_avec_option_centimes():
    suiveur = Suiveur(CONFIG_DEFAUT, compter_centimes=True)
    etats = _jouer(suiveur, [[_piece(20)] for _ in range(6)])
    assert etats[-1].total_centimes == 20


def test_mixte_centimes_sans_option():
    suiveur = Suiveur(CONFIG_DEFAUT)
    etats = _jouer(suiveur, [[_piece(20), _piece(500)] for _ in range(6)])
    assert etats[-1].total_centimes == 500


# ---------------------------------------------------------------------------
# Scintillement : apparition/déapparition
# ---------------------------------------------------------------------------


def test_scintillement_lent_jamais_compte():
    # pièce présente 2 frames sur 6 : même dans la fenêtre glissante de 5,
    # jamais 3 présences → jamais confirmée
    suiveur = Suiveur(CONFIG_DEFAUT)
    seq = []
    for _ in range(4):
        seq.append([_piece(100)])
        seq.append([_piece(100)])
        for _ in range(4):
            seq.append([_piece(100, motif="occultation")])
    etats = _jouer(suiveur, seq)
    assert etats[-1].total_centimes == 0
    # la piste sporadique ne dépasse jamais « tentative »
    assert all(p.phase in (PHASE_TENTATIVE, PHASE_PERDUE) for e in etats for p in e.pistes)
    # et la valeur ne remonte jamais au-dessus du creux : total monotone
    totaux = [e.total_centimes for e in etats]
    assert all(b >= a for a, b in zip(totaux, totaux[1:]))  # noqa: B905 — fatigue volontaire : coupure de séq.


def test_piste_perdue_total_fige():
    suiveur = Suiveur(CONFIG_DEFAUT)
    etats = _jouer(suiveur, [[_piece(100)] for _ in range(6)])
    assert etats[-1].total_centimes == 100
    etats_apres = _jouer(suiveur, [[_piece(100, motif="occultation")]] * 12, depart=7)
    assert etats_apres[-1].total_centimes == 100  # jamais dé-comptée
    assert etats_apres[-1].pistes == []  # perdue puis épurée


# ---------------------------------------------------------------------------
# Association continue
# ---------------------------------------------------------------------------


def test_piece_qui_se_deplace_reste_une_seule_piste():
    suiveur = Suiveur(CONFIG_DEFAUT)
    seq = [
        [_piece(100, centre=(100.0 + 2.0 * i, 100.0))] for i in range(8)
    ]
    etats = _jouer(suiveur, seq)
    pistes = {e.pistes[0].id for e in etats if e.pistes}
    assert len(pistes) == 1
    assert etats[-1].total_centimes == 100


def test_bras_absent_court_puis_reassocie():
    suiveur = Suiveur(CONFIG_DEFAUT)
    seq = [[_piece(100)] for _ in range(4)]
    seq += [[_piece(100, motif="eclat")] for _ in range(3)]  # 3 misses
    seq += [[_piece(100)] for _ in range(3)]  # réapparition
    etats = _jouer(suiveur, seq)
    # la piste a survécu aux 3 misses et a été ré-associée
    derniers = etats[-1].pistes
    assert len(derniers) == 1
    assert derniers[0].manques == 0
    assert etats[-1].total_centimes == 100


def test_saut_de_rayon_cree_une_nouvelle_piste():
    suiveur = Suiveur(CONFIG_DEFAUT)
    seq = [[_piece(100, rayon=48.0)] for _ in range(3)]
    seq += [[_piece(500, rayon=90.0, centre=(101.0, 101.0))] for _ in range(4)]
    etats = _jouer(suiveur, seq)
    # Δr/r = 0.87 > portee_rayon_rel → nouvelle identité
    assert len({p.id for p in etats[-1].pistes}) == 2


# ---------------------------------------------------------------------------
# Déterminisme
# ---------------------------------------------------------------------------


def test_determinisme_suivi():
    seq = [[_piece(100, centre=(10.0 * i, 20.0)), _piece(200, centre=(300.0, 40.0))]
           for i in range(8)]
    a = Suiveur(CONFIG_DEFAUT)
    b = Suiveur(CONFIG_DEFAUT)
    ea = _jouer(a, copy.deepcopy(seq))
    eb = _jouer(b, copy.deepcopy(seq))
    for x, y in zip(ea, eb, strict=True):
        assert x.total_centimes == y.total_centimes
        assert len(x.pistes) == len(y.pistes)
        for px, py in zip(x.pistes, y.pistes, strict=True):
            assert px.id == py.id
            assert px.centre == pytest.approx(py.centre)
            assert px.rayon == pytest.approx(py.rayon)
            assert px.phase == py.phase


def test_reinitialiser_remet_a_zero():
    suiveur = Suiveur(CONFIG_DEFAUT)
    _jouer(suiveur, [[_piece(100)] for _ in range(6)])
    assert suiveur.total_centimes == 100
    suiveur.reinitialiser()
    assert suiveur.total_centimes == 0
    assert suiveur.pistes == []


def test_les_pieces_recoivent_leur_piste_id():
    suiveur = Suiveur(CONFIG_DEFAUT)
    piece = _piece(500)
    for i in range(3):
        suiveur.maj([copy.deepcopy(piece)], i + 1)
    # l'étiquetage a lieu sur l'objet passé à maj
    piece = _piece(500)
    suiveur.maj([piece], 100)
    assert piece.piste_id is not None
