"""Tests des conversions d'image."""

from __future__ import annotations

import numpy as np
import pytest

from cvlab import images as im


def test_normalise_bgra_vers_bgr() -> None:
    bgra = np.zeros((4, 5, 4), np.uint8)
    assert im.normalise(bgra).shape == (4, 5, 3)


def test_normalise_canal_unique_aplati() -> None:
    assert im.normalise(np.zeros((4, 5, 1), np.uint8)).shape == (4, 5)


def test_normalise_16_bits() -> None:
    image = np.full((3, 3), 65535, np.uint16)
    sortie = im.normalise(image)
    assert sortie.dtype == np.uint8 and sortie.max() == 255


def test_normalise_flottant_mis_a_echelle() -> None:
    image = np.array([[-1.0, 0.0], [1.0, 2.0]], np.float32)
    sortie = im.normalise(image)
    assert sortie.dtype == np.uint8
    assert sortie.min() == 0 and sortie.max() == 255


def test_normalise_flottant_constant() -> None:
    sortie = im.normalise(np.full((3, 3), 7.0, np.float32))
    assert sortie.dtype == np.uint8 and sortie.max() == 0


@pytest.mark.parametrize(
    "mauvais",
    [None, np.zeros((0, 0), np.uint8), np.zeros((2, 2, 5), np.uint8),
     np.zeros((2, 2, 2, 2), np.uint8)],
)
def test_normalise_refuse(mauvais: object) -> None:
    with pytest.raises(im.InvalidImageError):
        im.normalise(mauvais)  # type: ignore[arg-type]


def test_gris_et_couleur_aller_retour(image_couleur: np.ndarray) -> None:
    gris = im.to_gray(image_couleur)
    assert gris.ndim == 2 and im.is_gray(gris)
    couleur = im.to_bgr(gris)
    assert couleur.shape == (*gris.shape, 3) and im.is_color(couleur)
    # Un gris promu en BGR a ses trois canaux identiques.
    assert np.array_equal(couleur[:, :, 0], couleur[:, :, 2])


def test_to_rgb_inverse_les_canaux() -> None:
    image = np.zeros((1, 1, 3), np.uint8)
    image[0, 0] = (255, 0, 0)  # bleu pur en BGR
    assert tuple(im.to_rgb(image)[0, 0]) == (0, 0, 255)


def test_match_shape_redimensionne(image_couleur, image_reference) -> None:
    aligne = im.match_shape(image_couleur, image_reference)
    assert aligne.shape[:2] == image_couleur.shape[:2]


def test_match_channels_promeut_le_gris(image_couleur) -> None:
    gris = im.to_gray(image_couleur)
    a, b = im.match_channels(gris, image_couleur)
    assert a.ndim == 3 and b.ndim == 3
    a, b = im.match_channels(gris, gris)
    assert a.ndim == 2 and b.ndim == 2


def test_describe_image(image_couleur) -> None:
    infos = im.describe_image(image_couleur)
    assert infos["largeur"] == image_couleur.shape[1]
    assert infos["canaux"] == 3
    assert 0 <= infos["moyenne"] <= 255
