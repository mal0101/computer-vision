# 05 — Robustesse et performance (M4)

Mesures **réelles**, pas des valeurs rêvées : chaque porte est gravée dans
`tests/test_robustesse.py` et `tests/test_perf.py` (marqueur `perf`,
hors tour par défaut, réactivable par `make perf`). Outil :
`python bench.py [--image …] [--largeur 640] [--seuil-ms 33]`.

## 1. Stratégie temps réel

`detecter` doit tenir ≤ 33 ms par frame à l'échelle d'analyse. Deux
optimisations structurelles (au lieu de micro-refactors) :

1. **Contre-vote Hough sur grille réduite** — le contre-vote ne sert qu'à
   *confirmer la position* d'un candidat déjà détecté par les contours
   (aucun cercle Hough sans correspondance n'est retenu, le rayon Hough
   n'est jamais assigné). À pleine résolution il coûtait > 1 s/frame
   (mesuré : **1 055 ms** sur 640×640). `_contre_vote_hough` redimensionne
   la grille par `HOUGH_REPERE = 8` (INTER_AREA) : même sémantique, pour
   **~10 ms** mesurés sur `coins.jpg` (16/16 confirmés), et la confiance
   attribuée n'influence aucune porte.
2. **Polarité en court-circuit** — `_compter_grosses_taches`
   (`connectedComponentsWithStats`) compte les composantes d'aire ≥
   π·(r_min)²·0,95 de la polarité inverse : elle borne le nombre de pièces
   que cette polarité pourrait produire. Si elle ne peut pas battre la
   polarité normale (`val_i ≤ val_n`), l'analyse complète de l'inverse
   (~la moitié du coût de contour d'une frame) est sautée — mesuré
   **−11 ms** en 640×640.

Conséquences mesurées : **médiane 5,5 ms** en 640×640 (min 5,1, p95 6,1,
moyenne 5,6, 40 itérations, après échauffement), **~8,5 ms** en 800×800
(native), pour `coins.jpg`. Détection **déterministe** (aucun appel
aléatoire, tris déterministes) : deux sessions → mêmes centres (testé).
La matrice de robustesse ci-dessous s'exécute en quelques secondes.

## 2. Matrice de robustesse (coins.jpg, 16 pièces, seed gravée)

Dégradations appliquées avant détection, à la **résolution native**
(800×800). Score = pièces dont la grande bordure (IoU ≥ 0,60) est trouvée
devant la référence. Seuil de classe = borne basse mesurée dans
`tests/test_robustesse.py`.

| Classe | Dégradation | Score mesuré | Porte |
|---|---|---|---|
| base | aucun dégradation, native | **16/16** | — |
| L1 | flou gaussien σ = 6 | **15** | ≥ 15 |
| L1 | sel & poivre 1 % | **16** | ≥ 16 |
| L1 | rotation 1,5° | **16** | ≥ 16 |
| L1 | échelle ×0,98 | **16** | ≥ 16 |
| L1 | luminosité +10 (α = 0,92) | **16** | ≥ 16 |
| L2 | flou gaussien σ = 1,5 | **16** | ≥ 12 |
| L2 | sel & poivre 4 % | **15** | ≥ 12 |
| L2 | rotation 4° | **15** | ≥ 12 |
| L2 | contraste 0,7 −15 | **15** | ≥ 12 |
| L3 | flou gaussien σ = 3 | **16** | ≥ 3 |
| L3 | ombrage latéral | **15** | ≥ 3 |
| L3 | perspective légère | **7** | ≥ 3 |
| L3 | rotation 12° | **3** | ≥ 3 |

> L'échelle d'analyse réelle (640×640, `largeur_analyse`) recale la même
> opération : **14/16** — c'est la mesure qui compte pour le direct, et
> elle est gravée en porte dédiée `test_l0_reduction_echelle_analyse`
> (≥ 13). La matrice au format décroissant L1 ≫ L2 ≫ L3 reflète l'échelle
> de dégradation, pas une hiérarchie d'objectifs : chaque seuil de porte
> est la **mesure**, pas une exigence flottante.

## 3. Portes exhaustives

`tests/test_robustesse.py` :

- `test_cas_degradations[...]` : 13 cas paramétrés L1/L2/L3 (ids
  `L1-gaussien`, `L1-sel_poivre`, …, `L3-rotation`, …). Chaque cas est
  **un** test dont la porte est la valeur du tableau ci-dessus ; le
  remplacer « pour faire passer » est un changement parallèle du paramètre
  de dégradation (le cas « rotation 12° → 3 pièces » est un vrai point de
  porte : au-delà de ~15° il n'y a plus rien à trouver).
- `test_l0_reduction_echelle_analyse` : **≥ 13** pièces (14 mesurées) à
  l'échelle réellement utilisée en direct.
- `test_tir_deterministe` : deux détections sur 3 images → mêmes
  (centre, diamètre).

`tests/test_perf.py` (marqueur `perf`) : `test_mediane_detection_sous_33_ms`
(≤ 33 ms, 40 itérations) et `test_detection_deterministe_sur_analyse`
(skip si `coins.jpg` absent).

## 4. Limites connues (assumées)

- Les rotations fortes (≥ ~8°) effacent peu à peu les pièces du masque
  (rééchantillonnage) : à 12° il n'en reste que 3 ; la porte L3
  « rotation 12° » est un **plancher de survie**, pas un objectif. En
  direct, une pièce penchée de > 8° est plus sûrement « tentative » que
  « inexistante ».
- Le masque de profilage est fait pour un fond sombre mat : les cas
  « ombrage/perspective » dégradent un masque *brut* (le pipeline
  n'applique jamais de filtre au passage webcam — défini en
  `docs/01-portee.md` § 2).
- La porte Hough réduite gagne le temps au prix d'une granularité de vote
  moindre (grille `HOUGH_REPERE=8`) ; elle ne sert qu'à confirmer une
  position déjà détectée par les contours, jamais à ajouter ou dimensionner
  une pièce.