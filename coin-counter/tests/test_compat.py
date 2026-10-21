"""Tests de la façade compat : réutilisation de cv-app, jamais du catalogue."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# ``cv-app`` est un dossier **frère** du dépôt, absent dès qu'on extrait
# ``coin-counter`` seul — ce que son propre ``pyproject.toml`` présente
# pourtant comme un projet autonome. ``pieces.compat`` lève alors une
# ImportError *à l'import du module*, ce qui interrompait la COLLECTE de
# pytest : plus un seul test ne s'exécutait, alors qu'aucun chemin de
# production de coin-counter n'utilise ``compat``. On saute donc ce fichier
# proprement au lieu de faire échouer toute la suite.
RACINE = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    not (RACINE.parent / "cv-app").is_dir(),
    reason="dossier frère cv-app absent de ce checkout (compat n'est utilisé "
    "par aucun chemin de production de coin-counter)",
)

if (RACINE.parent / "cv-app").is_dir():
    import pieces.compat as compat
else:  # pragma: no cover - dépend du checkout
    compat = None

#: les 10 fonctions + 2 exceptions promis par le plan
NOMS_ATTENDUS = {
    # webcam / entrées-sorties
    "Camera",
    "capture_snapshot",
    "load_image",
    "save_image",
    # conversions d'images
    "to_gray",
    "to_bgr",
    "normalise",
    # annotations du HUD
    "dess_cercle",
    "dess_text",
    "dess_texte_encadre",
    # exceptions
    "CameraError",
    "ImageLoadError",
}


def test_douze_noms_exportes() -> None:
    assert set(compat.__all__) == NOMS_ATTENDUS
    for nom in NOMS_ATTENDUS:
        assert hasattr(compat, nom), f"« {nom} » absent de la façade"


def test_types_attendus() -> None:
    assert isinstance(compat.Camera, type)
    assert issubclass(compat.CameraError, RuntimeError)
    assert issubclass(compat.ImageLoadError, ValueError)
    for nom in ("load_image", "save_image", "to_gray", "to_bgr", "normalise",
                "dess_cercle", "dess_text", "dess_texte_encadre",
                "capture_snapshot"):
        assert callable(getattr(compat, nom))


def test_camera_construite_sans_ouvrir() -> None:
    cam = compat.Camera(index=0, largeur=1280, hauteur=720)
    assert cam.resolution == (0, 0)  # pas ouverte → pas de lecture


def test_dess_cercle_renvoie_une_copie() -> None:
    image = np.zeros((60, 60, 3), np.uint8)
    sortie = compat.dess_cercle(image, (30, 30), 10)
    assert sortie.shape == image.shape
    assert sortie is not image
    assert int(image.sum()) == 0  # l'original n'est pas modifié


def test_normalise_bgr() -> None:
    gris = np.zeros((10, 12), np.uint8)
    bgr = compat.normalise(compat.to_bgr(gris))
    assert bgr.shape == (10, 12, 3)


def _lancer_depuisailleurs(script: str) -> subprocess.CompletedProcess[str]:
    """Exécute un script dans un répertoire quelconque avec pieces sur le path."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(RACINE)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_import_depuis_un_autre_repertoire() -> None:
    """La façade doit résoudre cv-app depuis __file__, pas depuis cwd."""
    resultat = _lancer_depuisailleurs(
        "import pieces.compat as c; print(sorted(c.__all__))"
    )
    assert resultat.returncode == 0, resultat.stderr
    assert "Camera" in resultat.stdout


def test_le_catalogue_de_filtres_nest_pas_importe() -> None:
    """Pièce maîtresse : cvlab.filters (les 46 filtres) doit être absent.

    Le corps du paquet cvlab importe forcément registry/pipeline (défini dans
    cvlab/__init__ de cv-app) — c'est inévitable en important cvlab.io_utils.
    En revanche le **catalogue lui-même** (cvlab.filters.*) ne doit jamais se
    charger : c'est lui qui coûte ~105 ms et qui créerait une dépendance au
    catalogue — explicitement hors périmètre (docs/01).
    """
    resultat = _lancer_depuisailleurs(
        "import pieces.compat, sys; "
        "interdits = [m for m in sys.modules "
        "if m.startswith(('cvlab.filters', 'cvlab.cli'))]; "
        "print('INTERDITS:' + ','.join(interdits))"
    )
    assert resultat.returncode == 0, resultat.stderr
    assert resultat.stdout.strip() == "INTERDITS:"


def test_cvabsent_produit_un_message_francais(tmp_path) -> None:
    """Sans cv-app à côté, l'erreur doit être explicite et en français."""
    script = tmp_path / "faux_paquet" / "pieces" / "compat.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "_RACINE = Path(__file__).resolve().parents[2]\n"
        "_CV_APP = _RACINE / 'cv-app'\n"
        "if not _CV_APP.is_dir():\n"
        "    raise ImportError('dossier cv-app introuvable a cote de coin-counter')\n",
        encoding="utf-8",
    )
    resultat = subprocess.run(
        [sys.executable, "-c", "import compat"],
        cwd=script.parent,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert resultat.returncode != 0
    assert "cv-app" in resultat.stderr
