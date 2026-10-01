"""Tests des mesures et des histogrammes."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from cvlab import histogram, metrics


def test_mesures_identiques(image_couleur: np.ndarray) -> None:
    assert metrics.mse(image_couleur, image_couleur) == pytest.approx(0.0)
    assert metrics.mae(image_couleur, image_couleur) == pytest.approx(0.0)
    assert metrics.psnr(image_couleur, image_couleur) == float("inf")


def test_mse_connue() -> None:
    a = np.zeros((2, 2), np.uint8)
    b = np.full((2, 2), 10, np.uint8)
    assert metrics.mse(a, b) == pytest.approx(100.0)
    assert metrics.rmse(a, b) == pytest.approx(10.0)
    assert metrics.mae(a, b) == pytest.approx(10.0)


def test_psnr_connue() -> None:
    a = np.zeros((4, 4), np.uint8)
    b = np.full((4, 4), 255, np.uint8)
    # MSE = 255² donc PSNR = 0 dB : le pire cas possible.
    assert metrics.psnr(a, b) == pytest.approx(0.0, abs=1e-6)


def test_mesures_tolerent_tailles_differentes(image_couleur, image_reference) -> None:
    valeurs = metrics.compare(image_couleur, image_reference)
    assert set(valeurs) == {"MSE", "RMSE", "MAE", "PSNR (dB)"}
    assert all(v >= 0 for v in valeurs.values())


def test_mesures_tolerent_canaux_differents(image_couleur) -> None:
    gris = cv2.cvtColor(image_couleur, cv2.COLOR_BGR2GRAY)
    assert metrics.mse(image_couleur, gris) >= 0.0


def test_difference_amplifiee(image_couleur) -> None:
    flou = cv2.GaussianBlur(image_couleur, (7, 7), 0)
    faible = metrics.difference(image_couleur, flou, 1.0)
    fort = metrics.difference(image_couleur, flou, 4.0)
    assert fort.mean() >= faible.mean()
    assert fort.dtype == np.uint8


def test_histogramme_couleur_et_gris(image_couleur) -> None:
    donnees = histogram.histogram_data(image_couleur)
    assert set(donnees) == {"Bleu", "Vert", "Rouge"}
    for valeurs in donnees.values():
        assert len(valeurs) == 256
        # La somme des classes vaut le nombre de pixels du canal.
        assert valeurs.sum() == image_couleur.shape[0] * image_couleur.shape[1]

    gris = cv2.cvtColor(image_couleur, cv2.COLOR_BGR2GRAY)
    assert set(histogram.histogram_data(gris)) == {"Gris"}


def test_histogramme_bins_bornes(image_couleur) -> None:
    assert len(histogram.histogram_data(image_couleur, bins=1)["Bleu"]) == 2
    assert len(histogram.histogram_data(image_couleur, bins=9999)["Bleu"]) == 256


def test_trace_histogramme_dimensions(image_couleur) -> None:
    trace = histogram.draw_histogram(image_couleur, largeur=400, hauteur=200)
    assert trace.shape == (200, 400, 3) and trace.dtype == np.uint8
    # Taille minimale imposée pour que le tracé reste lisible.
    petit = histogram.draw_histogram(image_couleur, largeur=10, hauteur=10)
    assert petit.shape[0] >= 120 and petit.shape[1] >= 160


def test_trace_sur_image_uniforme() -> None:
    """Un pic unique ne doit pas provoquer de division par zéro."""
    uniforme = np.full((20, 20), 128, np.uint8)
    for cumule in (False, True):
        for log in (False, True):
            trace = histogram.draw_histogram(uniforme, cumule=cumule,
                                             echelle_log=log)
            assert trace.size > 0


def test_statistiques(image_couleur) -> None:
    stats = histogram.statistics(image_couleur)
    assert set(stats) == {"Bleu", "Vert", "Rouge"}
    for valeurs in stats.values():
        assert valeurs["min"] <= valeurs["mediane"] <= valeurs["max"]
        assert valeurs["ecart_type"] >= 0
