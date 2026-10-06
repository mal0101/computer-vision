"""Tests de M3 : pipeline frame-à-frame (traitement), HUD (rendu), bureau,
CLI. Aucune webcam n'est ouverte : les captures sont des faux injectés.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pieces.config import CONFIG_DEFAUT
from pieces.entites import CERTITUDE_CONFIRME, CERTITUDE_TENTATIVE
from pieces.etalonnage import ObservateurEtalonnage
from pieces.etat import EtatApplication
from pieces.rendu import couleur_piece, dessiner_hud
from pieces.suivi import Suiveur
from pieces.traitement import reduire_a_analyse, traiter_image


def _coin_scene(rayon: float = 48.0, centre=(320, 240)) -> np.ndarray:
    """Scène BGR déterministe : pièce « 1 DH argent » sur fond sombre."""
    rng = np.random.default_rng(20261006)
    image = np.full((480, 640, 3), 34, dtype=np.uint8)
    image = np.clip(
        image.astype(np.int16) + rng.integers(-4, 5, image.shape), 0, 255
    ).astype(np.uint8)
    argent_hsv = np.array([[[120, 25, 200]]], dtype=np.uint8)
    argent_bgr = cv2.cvtColor(argent_hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    cv2.circle(image, centre, int(rayon), tuple(argent_bgr), -1, cv2.LINE_AA)
    return image


def _etat(px_par_mm: float | None = None, config=CONFIG_DEFAUT) -> EtatApplication:
    return EtatApplication(
        config=config,
        suiveur=Suiveur(config),
        px_par_mm=px_par_mm,
        observateur=ObservateurEtalonnage(),
    )


# ---------------------------------------------------------------------------
# Réduction à l'échelle d'analyse
# ---------------------------------------------------------------------------


def test_reduire_a_analyse_640x480():
    frame = np.zeros((480, 640, 3), np.uint8)
    analyse, echelle = reduire_a_analyse(frame, CONFIG_DEFAUT)
    assert analyse.shape[:2] == (480, 640)
    assert echelle == pytest.approx(1.0)


def test_reduire_a_analyse_1280x720():
    frame = np.zeros((720, 1280, 3), np.uint8)
    analyse, echelle = reduire_a_analyse(frame, CONFIG_DEFAUT)
    assert analyse.shape[:2] == (360, 640)
    assert echelle == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Pipeline de traitement
# ---------------------------------------------------------------------------


def test_traiter_image_total_apres_huit_frames():
    etat = _etat(px_par_mm=4.0)
    dernier = None
    for _ in range(8):
        dernier = traiter_image(_coin_scene(), etat)
        assert dernier is not None
    assert etat.suiveur.total_centimes == 100
    assert dernier.total_centimes == 100
    assert dernier.echelle == pytest.approx(1.0)
    assert dernier.analyse_ms >= 0.0
    assert dernier.echelle_px_mm == pytest.approx(4.0)
    assert len(dernier.pieces) >= 1


def test_traiter_image_masque_present():
    etat = _etat()
    resultat = traiter_image(_coin_scene(), etat)
    assert resultat.masque is not None
    assert resultat.masque.shape[:2] == (480, 640)
    assert resultat.masque.dtype == bool


def test_traiter_image_etalonnage_de_session():
    etat = _etat(px_par_mm=None)
    etat.mode_etalonnage = True
    for _ in range(40):
        traiter_image(_coin_scene(), etat)
    assert etat.px_par_mm is not None
    assert etat.px_par_mm == pytest.approx(4.0, abs=0.3)
    assert not etat.mode_etalonnage  # coupé automatiquement à la réussite
    assert "Échelle étalonnée" in " ".join(etat.messages)


def test_traiter_image_reinitialisation():
    etat = _etat(px_par_mm=4.0)
    for _ in range(8):
        traiter_image(_coin_scene(), etat)
    assert etat.suiveur.total_centimes == 100
    etat.reinitialiser()
    assert etat.suiveur.total_centimes == 0
    assert etat.num_frame == 0


def test_traiter_image_deterministe():
    a, b = _etat(px_par_mm=4.0), _etat(px_par_mm=4.0)
    for _ in range(6):
        ra = traiter_image(_coin_scene(), a)
        rb = traiter_image(_coin_scene(), b)
    assert ra.total_centimes == rb.total_centimes
    for pa, pb in zip(ra.pieces, rb.pieces, strict=True):
        assert pa.centre == pb.centre
        assert pa.rayon == pytest.approx(pb.rayon)


# ---------------------------------------------------------------------------
# Rendu HUD
# ---------------------------------------------------------------------------


def test_couleur_piece_par_certitude():
    from pieces.entites import Classification, PieceDetectee

    p = PieceDetectee(centre=(0, 0), rayon=1, aire=3.14, source="contour")
    assert couleur_piece(p) == (0, 165, 255)  # non classée
    p.motif_incertitude = "eclat"
    assert couleur_piece(p) == (110, 110, 110)
    p.motif_incertitude = None
    p.classification = Classification(100, "1 DH", 1.0, 0.1, CERTITUDE_CONFIRME)
    assert couleur_piece(p) == (60, 200, 80)
    p.classification = Classification(500, "5 DH", 0.0, None, CERTITUDE_TENTATIVE)
    assert couleur_piece(p) == (0, 220, 220)


def test_dessiner_hud_rend_une_copie_sans_modifier_l_entree():
    frame = _coin_scene()
    etat = _etat(px_par_mm=4.0)
    resultat = traiter_image(frame, etat)
    avant = frame.copy()
    hud = dessiner_hud(frame, resultat, etat)
    assert np.array_equal(frame, avant)  # entrée intacte
    assert hud.shape == frame.shape
    assert not np.array_equal(hud, avant)  # il y a bien un dessin


def test_dessiner_hud_deterministe():
    frame = _coin_scene()
    etat = _etat(px_par_mm=4.0)
    resultat = traiter_image(frame, etat)
    a = dessiner_hud(frame, resultat, etat)
    b = dessiner_hud(frame, resultat, etat)
    assert np.array_equal(a, b)


def test_dessiner_hud_sur_frame_vide():
    frame = np.zeros((480, 640, 3), np.uint8)
    etat = _etat()
    resultat = traiter_image(frame, etat)
    dessiner_hud(frame, resultat, etat)  # ne doit pas lever


# ---------------------------------------------------------------------------
# Bureau (avec faux capteur) et CLI
# ---------------------------------------------------------------------------


class FauxCapteur:
    """Capture simulée : rejoue une liste de frames puis ferme."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = list(frames)
        self._i = 0
        self.releas_ = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._i >= len(self._frames):
            return False, None
        frame = self._frames[self._i]
        self._i += 1
        return True, frame

    def isOpened(self) -> bool:
        return self._i < len(self._frames)

    def release(self) -> None:
        self.releas_ += 1


def test_bureau_loop_sans_fenetre_compte_une_piece():
    from pieces.ui.bureau import Bureau

    scenes = [_coin_scene() for _ in range(10)]
    capteur = FauxCapteur(scenes)
    bureau = Bureau(
        etat=_etat(px_par_mm=4.0), capture=capteur, afficher=False  # type: ignore[arg-type]
    )
    resultat = bureau.reel(limite=10)
    assert resultat is not None
    assert bureau.etat.suiveur.total_centimes == 100
    assert capteur.releas_ >= 1  # capture libérée


def test_bureau_total_monotone():
    from pieces.ui.bureau import Bureau

    bureau = Bureau(
        etat=_etat(px_par_mm=4.0),
        capture=FauxCapteur([_coin_scene()] * 12),  # type: ignore[arg-type]
        afficher=False,
    )
    bureau.reel(limite=12)
    # le total n'est ni négatif, ni remonté puis démonté
    assert bureau.etat.suiveur.total_centimes in (0, 100)
    assert bureau.etat.suiveur.total_centimes >= 0


def test_bureau_aucun_capteur_retourne_vite():
    from pieces.ui.bureau import Bureau

    bureau = Bureau(etat=_etat(), capture=None, afficher=False)
    assert bureau.reel(limite=100) is None


def test_touches_bureau():
    from pieces.ui.bureau import Bureau

    bureau = Bureau(etat=_etat(), capture=None, afficher=False)
    bureau._gerer_touche(ord("q"))
    assert not bureau._continuer
    bureau._gerer_touche(ord("c"))
    assert bureau.etat.mode_etalonnage
    bureau._gerer_touche(ord("c"))
    assert not bureau.etat.mode_etalonnage
    bureau._gerer_touche(ord("d"))
    assert bureau.etat.afficher_masque
    bureau._gerer_touche(ord("r"))
    assert bureau.etat.num_frame == 0


def test_cli_arguments_defauts():
    from pieces.cli import parcourir_arguments

    args = parcourir_arguments([])
    assert args.centimes is False
    assert args.camera == 0
    assert args.etalon_mm == 24.0
    assert args.limite is None
    assert args.sans_fenetre is False


def test_cli_arguments_plete():
    from pieces.cli import parcourir_arguments

    args = parcourir_arguments(
        ["--centimes", "--camera", "2", "--etalon-mm", "28", "--limite", "5"]
    )
    assert args.centimes
    assert args.camera == 2
    assert args.etalon_mm == 28.0
    assert args.limite == 5


def test_cli_version_exit_zero(capsys):
    from pieces.cli import parcourir_arguments

    with pytest.raises(SystemExit) as exc:
        parcourir_arguments(["--version"])
    assert exc.value.code == 0
    assert "coin-counter" in capsys.readouterr().out


def test_cli_sans_fenetre_code_zero(capsys):
    from pieces.cli import principal

    assert principal(["--sans-fenetre"]) == 0
    assert "sans fenêtre" in capsys.readouterr().out


def test_cli_config_invalide_code_deux(capsys):
    from pieces.cli import principal

    assert principal(["--config", "n'existe pas.json"]) == 2
    assert "configuration invalide" in capsys.readouterr().err


def test_cli_config_fichier_valide(tmp_path: Path):
    from pieces.cli import principal
    from pieces.config import config_vers_dict

    fichier = tmp_path / "config.json"
    fichier.write_text(
        json.dumps(config_vers_dict(CONFIG_DEFAUT)), encoding="utf-8"
    )
    assert principal(["--config", str(fichier), "--sans-fenetre"]) == 0


def test_cli_image_statique(tmp_path: Path):

    from pieces.cli import principal

    source = tmp_path / "scene.png"
    cv2.imwrite(str(source), _coin_scene())
    sortie = tmp_path / "hud.png"
    assert principal(["--image", str(source), "--sortie", str(sortie)]) == 0
    assert sortie.is_file()
    image = cv2.imread(str(sortie))
    assert image is not None and image.shape == (480, 640, 3)


def test_cli_image_illisible(capsys):
    from pieces.cli import principal

    assert principal(["--image", "n'existe pas.jpg"]) == 2
    assert "image illisible" in capsys.readouterr().err
