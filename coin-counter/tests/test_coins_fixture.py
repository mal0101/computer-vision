"""Porte M1 (niveau L0) : fixture photographique real coins.jpg.

Le contrat de porte est : **16 pièces détectées, chacune à IoU ≥ 0.60** avec
l'annotation de référence (`donnees/coins_annotations.json`, générée par la
chaîne elle-même et revue). Cette porte est détection UNIQUEMENT : les
annotations ne contiennent aucune dénomination (les familles de couleurs y
sont absentes), parce que la photo est « mélange de tailles » nativement.

coins.jpg est identique dans `TP7/exercice {5,6,7}/coins.jpg`, il est utilisé
par les exercices du dépôt — la fixture est donc partagée, pas propriétaire.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from pieces.config import CONFIG_DEFAUT
from pieces.detection import detecter

RACINE = Path(__file__).resolve().parent.parent
CHEMIN_ANNOTATIONS = RACINE / "donnees" / "coins_annotations.json"
IOU_MIN = 0.60


def _iou_cercles(groupe: np.ndarray, a: tuple, b: tuple) -> float:
    """IoU de deux disques sur une grille binaire 800×800."""
    h, w = groupe.shape
    ya, xa, ra = a
    yb, xb, rb = b
    yy, xx = np.mgrid[0:h, 0:w]
    d2_rayon_a = (yy - ya) ** 2 + (xx - xa) ** 2 <= ra * ra
    d2_rayon_b = (yy - yb) ** 2 + (xx - xb) ** 2 <= rb * rb
    inter = np.logical_and(d2_rayon_a, d2_rayon_b).sum()
    uni = np.logical_or(d2_rayon_a, d2_rayon_b).sum()
    return float(inter) / float(uni) if uni else 0.0


def test_coins_jpg_16_pieces_detectees(coins_jpg):
    image = cv2.imread(str(coins_jpg))
    assert image is not None
    resultat = detecter(image, CONFIG_DEFAUT)
    valides = [p for p in resultat.pieces if p.motif_incertitude is None]
    assert len(valides) == 16


def test_coins_jpg_chaque_piece_io_ge_060(coins_jpg):
    image = cv2.imread(str(coins_jpg))
    annotation = json.loads(CHEMIN_ANNOTATIONS.read_text(encoding="utf-8"))
    attendues = annotation["pieces"]
    assert len(attendues) == 16
    resultat = detecter(image, CONFIG_DEFAUT)
    valides = [p for p in resultat.pieces if p.motif_incertitude is None]
    # appariement au plus proche (distance euclidienne des centres)
    reste = list(valides)
    scores = []
    for attendue in attendues:
        (xa, ya), ra = attendue["centre"], attendue["rayon"]
        if not reste:
            break
        plus_proche = min(
            reste, key=lambda p: (p.centre[0] - xa) ** 2 + (p.centre[1] - ya) ** 2
        )
        reste.remove(plus_proche)
        score = _iou_cercles(
            np.zeros((800, 800), dtype=bool),
            (ya, xa, ra),
            (plus_proche.centre[1], plus_proche.centre[0], plus_proche.rayon),
        )
        scores.append(score)
    assert len(scores) == 16
    echecs = [(a["rayon"], round(s, 2)) for a, s in zip(attendues, scores, strict=True) if s < IOU_MIN]
    assert not echecs, f"pièces sous le seuil IoU : {echecs}"


def test_coins_jpg_profil_encode_coherent(coins_jpg):
    annotation = json.loads(CHEMIN_ANNOTATIONS.read_text(encoding="utf-8"))
    image = cv2.imread(str(coins_jpg))
    resultat = detecter(image, CONFIG_DEFAUT)
    assert annotation["profil_encode"] == resultat.profil


def test_annotations_rayons_dans_la_plage_62_99(coins_jpg):
    annotation = json.loads(CHEMIN_ANNOTATIONS.read_text(encoding="utf-8"))
    rayons = [p["rayon"] for p in annotation["pieces"]]
    assert min(rayons) >= 62.0
    assert max(rayons) <= 99.5
