"""Tests de la chaîne de traitement et du format de preset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cvlab import registry
from cvlab.pipeline import PRESET_VERSION, Pipeline, Step


def test_ajout_et_ordre() -> None:
    chaine = Pipeline()
    chaine.add("niveaux_de_gris")
    chaine.add("canny")
    assert [s.filter_id for s in chaine] == ["niveaux_de_gris", "canny"]
    chaine.add("filtre_gaussien", index=1)
    assert [s.filter_id for s in chaine] == [
        "niveaux_de_gris", "filtre_gaussien", "canny"
    ]
    assert len(chaine) == 3


def test_deplacement_et_suppression() -> None:
    chaine = Pipeline()
    for identifiant in ("niveaux_de_gris", "filtre_median", "canny"):
        chaine.add(identifiant)
    assert chaine.move(2, -1) == 1
    assert [s.filter_id for s in chaine][1] == "canny"
    assert chaine.move(0, -5) == 0       # borné, pas d'erreur
    assert chaine.move(2, 9) == 2
    chaine.remove(0)
    assert len(chaine) == 2
    with pytest.raises(IndexError):
        chaine.remove(7)
    chaine.clear()
    assert len(chaine) == 0


def test_etape_desactivee_est_ignoree(image_couleur: np.ndarray) -> None:
    chaine = Pipeline()
    chaine.add("negatif")
    chaine.steps[0].enabled = False
    resultat = chaine.apply(image_couleur)
    assert np.array_equal(resultat.image, image_couleur)
    assert resultat.steps[0].skipped
    assert resultat.applied == []


def test_ordre_des_filtres_change_le_resultat(image_couleur: np.ndarray) -> None:
    """Flouter puis détecter ≠ détecter puis flouter : c'est tout le propos."""
    avant = Pipeline()
    avant.add("filtre_gaussien", {"taille_noyau": 9, "sigma_x": 3.0})
    avant.add("canny", {"flou_prealable": 0})

    apres = Pipeline()
    apres.add("canny", {"flou_prealable": 0})
    apres.add("filtre_gaussien", {"taille_noyau": 9, "sigma_x": 3.0})

    assert not np.array_equal(
        avant.apply(image_couleur).image, apres.apply(image_couleur).image
    )


def test_etape_en_erreur_est_signalee_sans_interrompre(
    image_couleur: np.ndarray,
) -> None:
    chaine = Pipeline()
    chaine.add("negatif")
    chaine.add("fusion")  # sans référence : échoue volontairement
    chaine.add("negatif")
    resultat = chaine.apply(image_couleur, reference=None)
    assert len(resultat.warnings) == 1
    assert "deuxième image" in resultat.warnings[0]
    # Les étapes 1 et 3 se sont appliquées : double négatif = image d'origine.
    assert np.array_equal(resultat.image, image_couleur)


def test_stop_on_error_propage(image_couleur: np.ndarray) -> None:
    chaine = Pipeline()
    chaine.add("fusion")
    with pytest.raises(registry.FilterError, match="Étape 1"):
        chaine.apply(image_couleur, reference=None, stop_on_error=True)


def test_collect_false_ne_garde_rien(image_couleur: np.ndarray) -> None:
    chaine = Pipeline()
    chaine.add("negatif")
    resultat = chaine.apply(image_couleur, collect=False)
    assert resultat.steps == []
    assert resultat.image.shape == image_couleur.shape


def test_needs_reference() -> None:
    chaine = Pipeline()
    chaine.add("negatif")
    assert chaine.needs_reference is False
    chaine.add("fusion")
    assert chaine.needs_reference is True
    chaine.steps[-1].enabled = False
    assert chaine.needs_reference is False


def test_serialisation_aller_retour() -> None:
    chaine = Pipeline(nom="Essai")
    chaine.add("filtre_gaussien", {"taille_noyau": 7, "sigma_x": 1.5})
    chaine.add("canny", {"seuil_bas": 30, "seuil_haut": 90})
    chaine.steps[1].enabled = False

    relu = Pipeline.from_json(chaine.to_json())
    assert relu.nom == "Essai"
    assert [s.filter_id for s in relu] == [s.filter_id for s in chaine]
    assert relu.steps[0].params["taille_noyau"] == 7
    assert relu.steps[1].enabled is False


def test_serialisation_json_lisible() -> None:
    chaine = Pipeline(nom="Accentué é")
    chaine.add("canny")
    donnees = json.loads(chaine.to_json())
    assert donnees["version"] == PRESET_VERSION
    assert donnees["nom"] == "Accentué é"
    assert donnees["etapes"][0]["filtre"] == "canny"


def test_import_cles_anglaises() -> None:
    """Tolérance de lecture : « filter »/« params »/« enabled » sont acceptés."""
    chaine = Pipeline.from_dict({
        "steps": [{"filter": "canny", "params": {"seuil_bas": 11}, "enabled": False}]
    })
    assert chaine.steps[0].filter_id == "canny"
    assert chaine.steps[0].params["seuil_bas"] == 11
    assert chaine.steps[0].enabled is False


@pytest.mark.parametrize(
    ("donnees", "motif"),
    [
        ({"version": 999, "etapes": []}, "version"),
        ({"etapes": [{"filtre": "inconnu"}]}, "inconnu"),
        ({"etapes": [{}]}, "sans identifiant"),
        ({"etapes": [{"filtre": "canny", "parametres": 5}]}, "parametres"),
        ({"etapes": 12}, "liste"),
        ("pas un objet", "objet JSON"),
    ],
)
def test_presets_invalides(donnees: object, motif: str) -> None:
    with pytest.raises((ValueError, KeyError), match=motif):
        Pipeline.from_dict(donnees)  # type: ignore[arg-type]


def test_json_illisible() -> None:
    with pytest.raises(ValueError, match="JSON invalide"):
        Pipeline.from_json("{ceci n'est pas du json")


def test_enregistrement_et_lecture(tmp_path: Path) -> None:
    chaine = Pipeline(nom="Disque")
    chaine.add("filtre_median", {"taille_noyau": 7})
    chemin = chaine.save(tmp_path / "sous" / "dossier" / "p.json")
    assert chemin.is_file()
    assert Pipeline.load(chemin).steps[0].params["taille_noyau"] == 7
    with pytest.raises(FileNotFoundError):
        Pipeline.load(tmp_path / "absent.json")


def test_presets_fournis_sont_valides(image_couleur: np.ndarray,
                                     image_reference: np.ndarray) -> None:
    """Tous les presets livrés doivent se charger et s'appliquer."""
    dossier = Path(__file__).resolve().parent.parent / "presets"
    fichiers = sorted(dossier.glob("*.json"))
    assert fichiers, "aucun preset fourni"
    for fichier in fichiers:
        chaine = Pipeline.load(fichier)
        assert len(chaine) > 0, fichier.name
        resultat = chaine.apply(image_couleur, image_reference)
        assert resultat.image.size > 0, fichier.name
        assert not resultat.warnings, f"{fichier.name}: {resultat.warnings}"


def test_resume_lisible() -> None:
    chaine = Pipeline()
    assert chaine.summary() == "(chaîne vide)"
    chaine.add("canny")
    chaine.steps[0].enabled = False
    assert chaine.summary().startswith("✗ 1. Détecteur de Canny")


def test_step_name_et_definition() -> None:
    step = Step(filter_id="canny", params={})
    assert step.name == "Détecteur de Canny"
    assert step.definition().id == "canny"


def test_duree_mesuree(image_couleur: np.ndarray) -> None:
    chaine = Pipeline()
    chaine.add("filtre_bilateral")
    resultat = chaine.apply(image_couleur)
    assert resultat.duree_totale_ms >= 0.0
    assert resultat.steps[0].duree_ms >= 0.0
