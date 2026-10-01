"""Fixtures communes aux tests."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

RACINE = Path(__file__).resolve().parent.parent
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))


@pytest.fixture(scope="session")
def image_couleur() -> np.ndarray:
    """Petite image BGR contenant contours francs, dégradé, bruit et aplats.

    Volontairement minuscule (90×140) : les tests appliquent chaque filtre des
    dizaines de fois, et un filtre bilatéral sur une grande image rendrait la
    suite trop lente pour être lancée à chaque modification.
    """
    rng = np.random.default_rng(20240101)
    image = np.zeros((90, 140, 3), np.uint8)
    image[:] = (40, 80, 120)
    image[:, :70, 2] = np.linspace(0, 255, 70, dtype=np.uint8)
    cv2.circle(image, (100, 45), 26, (230, 190, 70), -1)
    cv2.rectangle(image, (6, 60), (50, 84), (255, 255, 255), -1)
    bruit = rng.integers(-18, 19, image.shape)
    return np.clip(image.astype(np.int16) + bruit, 0, 255).astype(np.uint8)


@pytest.fixture(scope="session")
def image_gris(image_couleur: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_couleur, cv2.COLOR_BGR2GRAY)


@pytest.fixture(scope="session")
def image_reference() -> np.ndarray:
    """Deuxième image, de dimensions différentes de la première.

    La différence de taille est intentionnelle : elle vérifie que les filtres à
    deux images redimensionnent bien la référence, comme l'exige
    ``cv2.addWeighted``.
    """
    image = np.full((60, 100, 3), 90, np.uint8)
    cv2.circle(image, (50, 30), 24, (250, 250, 250), -1)
    return image
