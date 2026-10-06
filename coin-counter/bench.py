#!/usr/bin/env python3
"""Table de temps d'exécution du pipeline de détection (M4).

Usage :
    python bench.py [--image chemin] [--largeur 640] [--iterations 40]
                    [--seuil-ms 33]

Mesure à l'échelle d'analyse réelle (réduction à ``--largeur``, la même que
``pieces.traitement.reduire_a_analyse``), après échauffement. Le code de
sortie est 0 si la médiane respecte ``--seuil-ms``, sinon 1 (porte `perf`).

Chiffres de référence sur le matériel cible (M1 M4) : médiane ≈ 6 ms sur
coins.jpg à 640 de large — la porte de 33 ms a une marge de ~5×.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import cv2

from pieces.config import CONFIG_DEFAUT
from pieces.detection import detecter


def _arguments() -> argparse.Namespace:
    analyseur = argparse.ArgumentParser(description=__doc__)
    analyseur.add_argument("--image", default="../TP7/exercice 5/coins.jpg")
    analyseur.add_argument("--largeur", type=int, default=640)
    analyseur.add_argument("--iterations", type=int, default=40)
    analyseur.add_argument("--seuil-ms", type=float, default=33.0)
    return analyseur.parse_args()


def mesurer(
    chemin: str, largeur: int, iterations: int
) -> tuple[float, float, float, float, int]:
    """(min, médiane, p95, moyenne ms, nb de pièces) à l'échelle d'analyse."""
    image = cv2.imread(str(chemin))
    if image is None:
        raise FileNotFoundError(f"image introuvable ou illisible : {chemin}")
    hauteur = max(1, round(image.shape[0] * largeur / image.shape[1]))
    analyse = cv2.resize(image, (largeur, hauteur), interpolation=cv2.INTER_AREA)

    detecter(analyse, CONFIG_DEFAUT)  # échauffement (imports, tables LUT…)

    durees: list[float] = []
    pieces = 0
    for _ in range(iterations):
        debut = time.perf_counter()
        resultat = detecter(analyse, CONFIG_DEFAUT)
        durees.append((time.perf_counter() - debut) * 1000.0)
        pieces = max(pieces, len(resultat.pieces))

    med = statistics.median(durees)
    p95 = sorted(durees)[max(0, int(iterations * 0.95) - 1)]
    return min(durees), med, p95, statistics.mean(durees), pieces


def principal() -> int:
    args = _arguments()
    try:
        mn, med, p95, moy, pieces = mesurer(
            args.image, args.largeur, args.iterations
        )
    except FileNotFoundError as err:
        print(f"bench : {err}", file=sys.stderr)
        return 1
    print(f"image            : {args.image}")
    print(f"échelle d'analyse: {args.largeur} px de large")
    print(f"pièces détectées : {pieces}")
    print(f"min / médiane/p95 / moyenne : {mn:.1f} / {med:.1f} / {p95:.1f} / {moy:.1f} ms")
    ok = med <= args.seuil_ms
    print(f"porte ({args.seuil_ms:.0f} ms)        : {'OK' if ok else 'DÉPASSÉE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(principal())
