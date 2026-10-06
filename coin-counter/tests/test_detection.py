"""Niveau L0 : tests de la chaîne de DÉTECTION (pieces.detection).

Portées évaluées ici (cf. docs/01-portee.md, matrice L0/L1/L2/L3) :
- décompte exact des pièces sur image synthétique propre (4 disques),
- image inversée (`negatif`, TP7) → même décompte,
- déterminisme strict d'un tour à l'autre (aucune course TRUCO),
- motifs d'incertitude : « trop petit », « chevauchement »,
- refus du bruit de fond (image noire / blanche / sans pièce).

Le décompte (>=1) passe par `ResultatDetection.pieces` filtré des pièces
dont le motif d'incertitude est None — seule façon de compter.
"""

from __future__ import annotations

import cv2
import numpy as np

from pieces.config import CONFIG_DEFAUT
from pieces.detection import detecter

CONFIG = CONFIG_DEFAUT


def _valides(resultat):
    return [p for p in resultat.pieces if p.motif_incertitude is None]


def _centres_rayons(pieces):
    return sorted(((p.centre[0], p.centre[1], p.rayon) for p in pieces), key=lambda c: c[0])


class TestImageSynthétiqueFournie:
    """Les 4 disques de reference (r = 20/28/34/45 px) vus fiables."""

    def test_decompte_les_4_disques(self, image_pieces):
        resultat = detecter(image_pieces, CONFIG)
        assert len(_valides(resultat)) == 4

    def test_les_rayons_sont_proches_des_references(self, image_pieces):
        resultat = detecter(image_pieces, CONFIG)
        rayons = sorted(p.rayon for p in _valides(resultat))
        assert len(rayons) == 4
        # tolérance ± 15 % (seuillage adaptatif + bruit léger)
        references = sorted((20.0, 28.0, 34.0, 45.0))
        for mesure, ref in zip(rayons, references, strict=True):
            assert abs(mesure - ref) / ref <= 0.15

    def test_toutes_sur_l_image_et_points_dans_l_image(self, image_pieces):
        resultat = detecter(image_pieces, CONFIG)
        h, w = image_pieces.shape[:2]
        for p in _valides(resultat):
            assert 0 <= p.centre[0] < w
            assert 0 <= p.centre[1] < h

    def test_image_inversee_donne_le_meme_decompte(self, image_pieces):
        # TP7 « negatif » : fond clair pièces sombres → polarité INVERSE
        inversee = (255 - image_pieces).astype(np.uint8)
        resultat = detecter(inversee, CONFIG)
        assert len(_valides(resultat)) == 4

    def test_seules_les_pieces_incertaines_sont_marquees(self, image_pieces):
        resultat = detecter(image_pieces, CONFIG)
        assert all(p.motif_incertitude is None for p in _valides(resultat))


class TestDeterminisme:
    """TRUCO (OpenCV 5) embarque un chemin parallèle : l'ordre des contours
    peut varier entre deux exécutions. Notre tri interne doit rendre la
    détection bit à bit identique."""

    def test_deux_executions_identiques(self, image_pieces):
        r1 = detecter(image_pieces, CONFIG)
        r2 = detecter(image_pieces, CONFIG)
        assert _centres_rayons(r1.pieces) == _centres_rayons(r2.pieces)
        assert r1.profil == r2.profil


class TestImagesSansPieces:
    def test_image_noire_aucune_piece(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        resultat = detecter(image, CONFIG)
        assert len(_valides(resultat)) == 0

    def test_image_blanche_aucune_piece(self):
        image = np.full((480, 640, 3), 255, dtype=np.uint8)
        resultat = detecter(image, CONFIG)
        assert len(_valides(resultat)) == 0


class TestMotifsDIncertitude:
    """Les pièces illisibles ne sont jamais comptées mais restent signalées."""

    def test_trop_petit_signal_mais_non_comptable(self, image_pieces):
        # petit disque isolé (diamètre 12 px < 40 px) sur fond sombre
        image = image_pieces.copy()
        cv2.circle(image, (90, 90), 6, (220, 220, 235), -1, cv2.LINE_AA)
        resultat = detecter(image, CONFIG)
        petits = [p for p in resultat.pieces if p.motif_incertitude == "trop_petit"]
        assert len(petits) == 1
        assert not petits[0].est_comptable

    def test_trop_gros_ignore(self, image_pieces):
        # cercle isolé de rayon 80 px ≫ portée 0.15·H → jamais compté
        # (posé loin des 4 disques pour ne pas les recouvrir)
        image = image_pieces.copy()
        cv2.circle(image, (30, 30), 80, (200, 210, 230), -1, cv2.LINE_AA)
        resultat = detecter(image, CONFIG)
        assert len(_valides(resultat)) == 4  # seuls les 4 disques survivent


class TestChevauchement:
    """Deux pièces soudées (figure en « 8 ») dont l'aire dépasse la portée :
    l'éclatement (watershed) doit les séparer en 2 pièces valides. Les
    constantes ci-dessous ont été fixées par mesure réelle (déterministe).
    """

    def test_deux_pieces_colles_separees(self):
        rng = np.random.default_rng(17)
        image = np.full((480, 640, 3), 40, dtype=np.uint8)
        image = np.clip(
            image.astype(np.int16) + rng.integers(-5, 6, image.shape), 0, 255
        ).astype(np.uint8)
        # deux disques en contact (chevauchement pixel mince, d = 140 px pour
        # r = 72) : la fermeture morphologique les soude en UNE silhouette,
        # l'éclatement (watershed) doit rendre les deux pièces complètes
        cv2.circle(image, (170, 240), 72, (200, 210, 235), -1, cv2.LINE_AA)
        cv2.circle(image, (310, 240), 72, (200, 210, 235), -1, cv2.LINE_AA)
        resultat = detecter(image, CONFIG)
        valides = _valides(resultat)
        assert len(valides) == 2
        assert abs(valides[0].centre[0] - valides[1].centre[0]) > 20
        assert all(abs(v.centre[1] - 240.0) < 2.0 for v in valides)


class TestProfil:
    """Le choix du profil est déterministe et documenté dans le masque."""

    def test_profil_par_defaut_sur_image_saine(self, image_pieces):
        resultat = detecter(image_pieces, CONFIG)
        assert resultat.profil == "defaut"

    def test_profil_sel_poivre_sur_image_bruitee(self, image_pieces):
        rng = np.random.default_rng(42)
        bruitee = image_pieces.copy()
        sel = rng.random(bruitee.shape[:2]) < 0.02
        bruitee[sel] = 255
        poivre = rng.random(bruitee.shape[:2]) < 0.02
        bruitee[poivre] = 0
        resultat = detecter(bruitee, CONFIG)
        assert resultat.profil == "sel_poivre"
        assert len(_valides(resultat)) >= 4

    def test_le_masque_a_la_taille_de_l_image(self, image_pieces):
        resultat = detecter(image_pieces, CONFIG)
        assert resultat.masque is not None
        assert resultat.masque.shape[:2] == image_pieces.shape[:2]
