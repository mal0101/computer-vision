# coin-counter

Compteur de pièces marocaines **en direct** (webcam), traitement d'image
**classique OpenCV** — aucun apprentissage automatique. Détecte des pièces
de {½, 1, 2, 5, 10} DH (+10c/20c avec `--centimes`) sur fond sombre,
identifie, étalonne et **additionne** en centimes entiers (`12,50 DH`).

Code source en français ; tests pytest — le marqueur `perf` est exclu du
tour par défaut et réactivé par `make perf`.

## Prétendances (ce que la version 1 garantit)

Niveaux de portée : **vérifié** = passé par la suite de tests sur disque ;
**cible (non vérifié)** = promis par le code mais pas validé en session.

| Niveau | Type | Statut | Comment |
|---|---|---|---|
| Tier 1 | image fixe (TP7 `coins.jpg`, 800×800, 16 pièces) | **vérifié** | 152 tests verts − porte IoU ≥ 0,60 (16/16) |
| Tier 2 | données synthétiques (étalonnage, motifs, survêtements) | **vérifié** | `tests/test_*.py`, `make check` |
| Tier 3 | **webcam en direct** (retour fluide, fond réel) | **cible (non vérifié)** | pas de caméra connectée pendant le développement |
| – | packaging / binaires | **hors périmètre** | programme lancé via `python -m pieces.cli` |

Contexte d'acquisition documenté : résolution ≥ 720p, distance 20–25 cm,
lumière diffuse ~45°, fond sombre et mat, pièces à plat. Voir
`docs/01-portee.md`.

## Démarrage rapide

```bash
# environnement
make venv            # crée cv-app/.venv s'il n'existe pas
make check           # ruff + suite complète (152 tests, 2 désélectionnés)
make perf            # portes de performance (médiane détection ≤ 33 ms)
make bench           # tableau des temps d'exécution (bench.py)
make image IMAGE=../TP7/exercice\ 5/coins.jpg
make webcam          # directement sur la webcam (indice 0)
```

## Utilisation

```bash
python -m pieces.cli --help
python -m pieces.cli --camera 0 --centimes          # direct, avec 10c/20c
python -m pieces.cli --image photo.jpg --sortie sorties/hud.png
python -m pieces.cli --camera 0 --limite 1 --sortie sorties/capture.png
python -m pieces.cli --sans-fenetre                # autotest du pipeline
```

Fenêtré : `Échap`/`q` quitter, `c` étalonnage, `d` vignette du masque,
`r` remise à zéro.

## Architecture

```
pieces/
  config.py          Config (classes @dataclass) + publication json
  entites.py         PieceDetectee, EtatApplication, …
  portee.py          règles de morphologie (circularité, trou central, rejets)
  compat.py          diagnostic du type OpenCV/Fermé
  erreurs.py         erreurs typées (et chaine validation)
  detection.py       détecter (profilage RLE bandes + Hough réduit)
  classification.py  couleur → famille/polarité → taille (score & seuils)
  etalonnage.py      étalonnage de session (jamais persisté)
  suivi.py           Suiveur (machine d'états) + etalonnage vue
  traitement.py      traiter_image (échelle d'analyse → … → RésultatImage)
  rendu.py           dessiner_hud
  ui/bureau.py       Bureau (boucle webcam, touches, creer_capture)
  cli.py             argparse + modes (webcam/image/sans-fenetre)
tests/
  conftest.py        fixtures suivi/bureau, config publique
  test_config.py     socle (config, entités, portée, compat, erreurs)
  test_detection.py  16/16 coins.jpg (IoU ≥ 0,60)…
bench.py             tableau des temps (médiane ≤ 33 ms)
```

## Portes (mesures, pas des objectifs)

- Détection : 16/16 sur `coins.jpg` (IoU ≥ 0,60) — `tests/test_detection.py`.
- Robustesse L1/L2/L3 + échelle d'analyse 640 (14/16, porte ≥ 13) —
  `tests/test_robustesse.py`, détails `docs/05-robustesse.md`.
- Performance : médiane 5,5 ms par frame à 640 px de large (porte 33 ms) —
  `tests/test_perf.py`, `make perf`, `bench.py`.
- Une pièce incertaine n'est **jamais** comptée ; le total ne baisse
  jamais (machine d'états, `docs/03-suivi-comptage.md`).

## Validation

```bash
make check     # ruff clean + make check → 152 passed, 2 deselected
make perf      # 2 passed (médiane ≤ 33 ms + déterminisme)
```

## Documentation

- `docs/01-portee.md` — périmètre contractuel, conditions d'acquisition.
- `docs/02-classification.md` — couleur/polarité/taille, étalonnage.
- `docs/03-suivi-comptage.md` — machine d'états, total monotone.
- `docs/04-interface-utilisateur.md` — bureau, CLI, Makefile.
- `docs/05-robustesse.md` — matrice L1/L2/L3, performance, limites.