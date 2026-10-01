"""Tests du catalogue : chaque filtre doit être sain et robuste.

Ces tests sont le filet de sécurité principal du projet : ils parcourent le
catalogue et appliquent chaque filtre à des images de forme et de type variés,
puis à chacune des bornes de chacun de ses paramètres. Un filtre ajouté sans
précaution (noyau pair, division par zéro, mode de bord refusé par OpenCV,
sortie d'un type inattendu) échoue ici.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cvlab import registry
from cvlab.params import (
    BoolParam,
    ChoiceParam,
    ColorParam,
    FloatParam,
    IntParam,
    TextParam,
)
from cvlab.registry import FAMILY_ORDER, FilterDef

TOUS = registry.all_filters()
IDENTIFIANTS = [d.id for d in TOUS]


def test_catalogue_non_vide() -> None:
    assert len(TOUS) >= 40, "le catalogue devrait couvrir tous les TP"


def test_identifiants_uniques() -> None:
    assert len(IDENTIFIANTS) == len(set(IDENTIFIANTS))


def test_familles_connues() -> None:
    for definition in TOUS:
        assert definition.family in FAMILY_ORDER


def test_familles_ordonnees_et_non_vides() -> None:
    familles = registry.families()
    assert familles == [f for f in FAMILY_ORDER if f in set(familles)]
    for famille, definitions in registry.filters_by_family().items():
        assert definitions, f"famille vide: {famille}"


@pytest.mark.parametrize("definition", TOUS, ids=IDENTIFIANTS)
def test_metadonnees_renseignees(definition: FilterDef) -> None:
    """Chaque filtre doit être documenté : c'est ce qui alimente l'aide."""
    assert definition.name and definition.summary
    assert definition.theory, f"{definition.id}: principe non documenté"
    assert definition.tp, f"{definition.id}: origine (TP) non renseignée"
    assert definition.summary.endswith((".", "!", "?"))
    for spec in definition.params:
        assert spec.label, f"{definition.id}.{spec.name}: libellé manquant"


@pytest.mark.parametrize("definition", TOUS, ids=IDENTIFIANTS)
def test_defauts_valides(definition: FilterDef) -> None:
    """``coerce`` sur les valeurs par défaut doit être l'identité."""
    defauts = definition.defaults()
    assert definition.coerce(defauts) == defauts
    assert definition.coerce(None) == defauts  # valeurs manquantes complétées


@pytest.mark.parametrize("definition", TOUS, ids=IDENTIFIANTS)
def test_application_couleur_et_gris(
    definition: FilterDef,
    image_couleur: np.ndarray,
    image_gris: np.ndarray,
    image_reference: np.ndarray,
) -> None:
    for source in (image_couleur, image_gris):
        sortie, infos = definition.apply(source, None, image_reference)
        assert sortie.dtype == np.uint8
        assert sortie.size > 0
        assert sortie.ndim in (2, 3)
        if definition.output_mode == "gray":
            assert sortie.ndim == 2
        if definition.output_mode == "color":
            assert sortie.ndim == 3 and sortie.shape[2] == 3
        assert isinstance(infos, dict)
        # L'image d'entrée ne doit jamais être modifiée sur place.
        assert source.dtype == np.uint8


@pytest.mark.parametrize("definition", TOUS, ids=IDENTIFIANTS)
def test_entree_non_modifiee(
    definition: FilterDef, image_couleur: np.ndarray, image_reference: np.ndarray
) -> None:
    """Un filtre doit renvoyer une nouvelle image, pas muter la source.

    Sans cette garantie, l'affichage pas-à-pas montrerait partout la même
    image et la comparaison avant/après n'aurait aucun sens.
    """
    copie = image_couleur.copy()
    definition.apply(image_couleur, None, image_reference)
    assert np.array_equal(image_couleur, copie), f"{definition.id} modifie son entrée"


def _valeurs_limites(spec: Any) -> list[Any]:
    """Valeurs extrêmes à tester pour une spécification donnée."""
    if isinstance(spec, (IntParam, FloatParam)):
        return [spec.min, spec.max, spec.default]
    if isinstance(spec, BoolParam):
        return [True, False]
    if isinstance(spec, ChoiceParam):
        return list(spec.keys)
    if isinstance(spec, ColorParam):
        return [(0, 0, 0), (255, 255, 255)]
    if isinstance(spec, TextParam):
        return ["", "Éèàüñ 123 !"]
    return [spec.default]


@pytest.mark.parametrize("definition", TOUS, ids=IDENTIFIANTS)
def test_bornes_des_parametres(
    definition: FilterDef,
    image_couleur: np.ndarray,
    image_gris: np.ndarray,
    image_reference: np.ndarray,
) -> None:
    """Aucune borne de curseur ne doit provoquer d'erreur OpenCV."""
    for spec in definition.params:
        for valeur in _valeurs_limites(spec):
            for source in (image_couleur, image_gris):
                sortie, _ = definition.apply(
                    source, {spec.name: valeur}, image_reference
                )
                assert sortie.size > 0 and sortie.dtype == np.uint8, (
                    f"{definition.id}.{spec.name}={valeur!r}"
                )


@pytest.mark.parametrize(
    "forme", [(1, 1, 3), (1, 1), (2, 3, 3), (3, 3), (4, 7, 3)]
)
def test_images_minuscules(forme: tuple[int, ...], image_reference: np.ndarray) -> None:
    """Une image dégénérée ne doit pas produire de plantage inattendu.

    Seule exception admise : l'homographie, qui refuse explicitement quatre
    points confondus — c'est un message d'erreur voulu, pas un bogue.
    """
    source = np.full(forme, 120, np.uint8)
    for definition in TOUS:
        try:
            sortie, _ = definition.apply(source, None, image_reference)
        except registry.FilterError as exc:
            assert definition.id == "deformation_perspective", (
                f"{definition.id} échoue sur {forme}: {exc}"
            )
            continue
        assert sortie.size > 0


def test_parametre_inconnu_ignore() -> None:
    """Un preset ancien contenant un paramètre disparu reste utilisable."""
    definition = registry.get("canny")
    valeurs = definition.coerce({"seuil_bas": 10, "parametre_disparu": 42})
    assert valeurs["seuil_bas"] == 10
    assert "parametre_disparu" not in valeurs


def test_filtre_inconnu() -> None:
    with pytest.raises(registry.UnknownFilterError):
        registry.get("ce_filtre_nexiste_pas")


def test_spec_inconnue() -> None:
    with pytest.raises(KeyError):
        registry.get("canny").spec("inexistant")


def test_reference_manquante(image_couleur: np.ndarray) -> None:
    definition = registry.get("fusion")
    with pytest.raises(registry.FilterError, match="deuxième image"):
        definition.apply(image_couleur, None, None)


def test_describe_contient_les_parametres() -> None:
    texte = registry.get("filtre_gaussien").describe()
    assert "taille_noyau" in texte and "sigma_x" in texte and "TP7" in texte
