"""Fixtures communes aux tests de coin-counter.

Toutes les images synthétiques sont **déterministes** (générateur à graine
fixe) : la suite doit donner exactement les mêmes résultats d'un tour à
l'autre, sur n'importe quelle machine.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

#: racine du dossier coin-counter (les tests importent `pieces` via pythonpath)
RACINE = Path(__file__).resolve().parent.parent
#: racine du dépôt complet (contient cv-app/, TP7/, …)
RACINE_DEPOT = RACINE.parent
#: fixture photographique du TP7 (identique dans les exercices 5, 6 et 7)
COINS_JPG = RACINE_DEPOT / "TP7" / "exercice 5" / "coins.jpg"

#: les 4 cercles de référence du plan de test : rayons en pixels, tous
#: dans la plage [0.02·H, 0.15·H] d'une image de 480 de haut
RAYONS_REFERENCE = (20.0, 28.0, 34.0, 45.0)


@pytest.fixture(scope="session")
def image_pieces() -> np.ndarray:
    """Image BGR déterministe : 4 disques clairs sur fond sombre bruité.

    Dimensions 640×480 (échelle d'analyse par défaut). Les disques ont une
    teinte métallique légèrement différente (deux familles or/argent) et un
    dégradé radial pour rester réalistes sans sortir du déterminisme.
    """
    rng = np.random.default_rng(20261006)
    image = np.full((480, 640, 3), 42, dtype=np.uint8)
    image = np.clip(
        image.astype(np.int16) + rng.integers(-6, 7, image.shape), 0, 255
    ).astype(np.uint8)
    centres = [(120, 130), (430, 110), (180, 350), (470, 330)]
    teintes = [(190, 205, 225), (130, 185, 235), (205, 215, 230), (140, 190, 240)]
    for (cx, cy), rayon, teinte in zip(centres, RAYONS_REFERENCE, teintes, strict=True):
        # dégradé radial discret : disque plat + petit disque plus clair
        cv2.circle(image, (cx, cy), int(rayon), teinte, -1, cv2.LINE_AA)
        clair = tuple(min(255, int(c) + 35) for c in teinte)
        cv2.circle(image, (cx, cy), max(2, int(rayon * 0.35)), clair, -1, cv2.LINE_AA)
    return image


@pytest.fixture(scope="session")
def image_grise(image_pieces: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_pieces, cv2.COLOR_BGR2GRAY)


@pytest.fixture(scope="session")
def coins_jpg() -> Path:
    """Chemin vers la fixture photographique du TP7 (vérifie qu'elle existe)."""
    if not COINS_JPG.is_file():
        pytest.skip(f"fixture absente : {COINS_JPG}")
    return COINS_JPG
