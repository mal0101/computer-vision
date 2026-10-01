"""Tests de l'interface en ligne de commande."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from cvlab import cli, io_utils


@pytest.fixture()
def image_disque(tmp_path: Path, image_couleur: np.ndarray) -> Path:
    return io_utils.save_image(image_couleur, tmp_path / "entree.png")


def test_parse_step_simple() -> None:
    etape = cli.parse_step("canny")
    assert etape.filter_id == "canny"
    assert etape.params["seuil_bas"] == 50  # défaut appliqué


def test_parse_step_avec_parametres() -> None:
    etape = cli.parse_step("canny:seuil_bas=10,seuil_haut=200,norme_l2=vrai")
    assert etape.params["seuil_bas"] == 10
    assert etape.params["seuil_haut"] == 200
    assert etape.params["norme_l2"] is True


def test_parse_step_corrige_les_valeurs_invalides() -> None:
    """Un noyau pair donné en ligne de commande est corrigé, pas refusé."""
    etape = cli.parse_step("filtre_median:taille_noyau=8")
    assert etape.params["taille_noyau"] == 9


@pytest.mark.parametrize(
    "expression",
    ["", "filtre_inexistant", "canny:seuil_zzz=3", "canny:seuil_bas",
     "canny:ouverture=abc"],
)
def test_parse_step_erreurs(expression: str) -> None:
    with pytest.raises((argparse.ArgumentTypeError, KeyError, ValueError)):
        cli.parse_step(expression)


def test_lister(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["lister"]) == 0
    sortie = capsys.readouterr().out
    assert "Contours & gradients" in sortie and "canny" in sortie
    assert "filtre(s)" in sortie


def test_lister_filtre_par_famille(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["lister", "--famille", "Seuillage", "-v"]) == 0
    sortie = capsys.readouterr().out
    assert "seuillage_simple" in sortie
    assert "Contours & gradients" not in sortie


def test_decrire(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["decrire", "filtre_gaussien"]) == 0
    assert "taille_noyau" in capsys.readouterr().out


def test_decrire_inconnu(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["decrire", "zzz"]) == 2
    assert "inconnu" in capsys.readouterr().err


def test_appliquer_ecrit_une_image(image_disque: Path, tmp_path: Path) -> None:
    sortie = tmp_path / "sortie.png"
    code = cli.main([
        "appliquer", str(image_disque),
        "-s", "filtre_gaussien:taille_noyau=5",
        "-s", "canny:seuil_bas=40,seuil_haut=120",
        "-o", str(sortie), "--etapes",
    ])
    assert code == 0
    assert sortie.is_file()
    resultat = io_utils.load_image(sortie)
    assert resultat.ndim == 2  # Canny produit une image à un canal


def test_appliquer_enregistre_histogramme_et_preset(
    image_disque: Path, tmp_path: Path
) -> None:
    preset = tmp_path / "chaine.json"
    trace = tmp_path / "histo.png"
    code = cli.main([
        "appliquer", str(image_disque), "-s", "negatif",
        "--histogramme", str(trace), "--sauver-preset", str(preset),
    ])
    assert code == 0
    assert trace.is_file() and preset.is_file()
    donnees = json.loads(preset.read_text(encoding="utf-8"))
    assert donnees["etapes"][0]["filtre"] == "negatif"


def test_appliquer_avec_preset(image_disque: Path, tmp_path: Path) -> None:
    preset = tmp_path / "p.json"
    preset.write_text(json.dumps({
        "version": 1, "nom": "T",
        "etapes": [{"filtre": "niveaux_de_gris", "parametres": {}, "actif": True}],
    }), encoding="utf-8")
    assert cli.main(["appliquer", str(image_disque), "--preset", str(preset)]) == 0


def test_appliquer_sans_etape(image_disque: Path) -> None:
    assert cli.main(["appliquer", str(image_disque)]) == 2


def test_appliquer_sans_source() -> None:
    assert cli.main(["appliquer", "-s", "negatif"]) == 1


def test_appliquer_reference_manquante(image_disque: Path) -> None:
    assert cli.main(["appliquer", str(image_disque), "-s", "fusion"]) == 2


def test_appliquer_avec_reference(image_disque: Path, tmp_path: Path,
                                 image_reference: np.ndarray) -> None:
    chemin_reference = io_utils.save_image(image_reference, tmp_path / "ref.png")
    code = cli.main([
        "appliquer", str(image_disque), "-s", "fusion:alpha=0.3",
        "--reference", str(chemin_reference), "-o", str(tmp_path / "f.png"),
    ])
    assert code == 0


def test_comparer(image_disque: Path, tmp_path: Path,
                  capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main([
        "comparer", str(image_disque),
        "-s", "filtre_moyenneur", "-s", "filtre_gaussien", "-s", "filtre_median",
        "--dossier-sortie", str(tmp_path / "comp"),
    ])
    assert code == 0
    sortie = capsys.readouterr().out
    assert "PSNR" in sortie and "Filtre médian" in sortie
    assert (tmp_path / "comp" / "filtre_median.png").is_file()


def test_comparer_sans_filtre(image_disque: Path) -> None:
    assert cli.main(["comparer", str(image_disque)]) == 2


def test_fichier_absent_code_1(tmp_path: Path) -> None:
    assert cli.main(["appliquer", str(tmp_path / "absent.png"), "-s", "negatif"]) == 1
