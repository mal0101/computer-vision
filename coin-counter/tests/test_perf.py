"""Porte de performance (M4) — marqueur ``perf``, hors tour par défaut.

Mesure à l'échelle d'analyse réelle (640 px de large, comme
``pieces.traitement.reduire_a_analyse``) après échauffement. La porte :
**médiane ≤ 33 ms** sur coins.jpg. Voir docs/05-robustesse.md pour les
détails et `make perf` pour l'exécuter.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import cv2
import pytest

from pieces.config import CONFIG_DEFAUT
from pieces.detection import detecter

COINS_JPG = (
    Path(__file__).resolve().parent.parent.parent / "TP7" / "exercice 5" / "coins.jpg"
)

SEUIL_MEDIANE_MS = 33.0
LARGEUR_ANALYSE = 640
ITERATIONS = 40


@pytest.mark.perf
def test_mediane_detection_sous_33_ms():
    if not COINS_JPG.is_file():
        pytest.skip(f"fixture absente : {COINS_JPG}")
    image = cv2.imread(str(COINS_JPG))
    assert image is not None
    hauteur = round(image.shape[0] * LARGEUR_ANALYSE / image.shape[1])
    analyse = cv2.resize(image, (LARGEUR_ANALYSE, hauteur), interpolation=cv2.INTER_AREA)

    detecter(analyse, CONFIG_DEFAUT)  # échauffement (imports one-shot)

    durees: list[float] = []
    for _ in range(ITERATIONS):
        debut = time.perf_counter()
        detecter(analyse, CONFIG_DEFAUT)
        durees.append((time.perf_counter() - debut) * 1000.0)

    med = statistics.median(durees)
    assert med <= SEUIL_MEDIANE_MS, (
        f"médiane {med:.1f} ms > {SEUIL_MEDIANE_MS:.0f} ms "
        f"(min {min(durees):.1f} ms, p95 {sorted(durees)[int(ITERATIONS * 0.95) - 1]:.1f} ms)"
    )


@pytest.mark.perf
def test_detection_deterministe_sur_analyse():
    if not COINS_JPG.is_file():
        pytest.skip(f"fixture absente : {COINS_JPG}")
    image = cv2.imread(str(COINS_JPG))
    analyse = cv2.resize(image, (LARGEUR_ANALYSE, 640), interpolation=cv2.INTER_AREA)
    a = detecter(analyse, CONFIG_DEFAUT)
    b = detecter(analyse, CONFIG_DEFAUT)
    assert [(p.centre, p.rayon) for p in a.pieces] == [
        (p.centre, p.rayon) for p in b.pieces
    ]
