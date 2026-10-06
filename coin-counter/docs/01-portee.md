# 01 — Portée du projet (périmètre contractuel)

> Document de **périmètre** : il définit ce que le logiciel promet, ce qu'il
> refuse de promettre, et dans quelles conditions. Toute affirmation qui
> dépasse ce document est considérée comme non vérifiée.

## 1. Objet

`coin-counter` détecte, identifie et **additionne** des pièces marocaines
visibles par une webcam, en direct. Le traitement est purement classique
(OpenCV) : aucun apprentissage automatique, aucun modèle entraîné.

## 2. Ce que fait la version 1

- Détection des pièces entièrement visibles sur un fond sombre mat,
  quel que soit le filtre éventuellement appliqué par l'environnement
  (le flux webcam **n'applique aucun filtre** — c'est le cœur de l'exigence).
- Identification : {10c, 20c, ½, 1, 2, 5, 10} DH — les pièces de
  **10c et 20c ne sont comptées qu'avec l'option `--centimes`** (défaut :
  {½, 1, 2, 5, 10} DH).
- Sommation exacte en **centimes entiers**, affichée au format
  marocain : `12,50 DH` (virgule française).
- Étalonnage explicite (pièce de référence ou carte bancaire 85,6 mm),
  obligatoire pour afficher un total confirmé.
- Suivi de pistes : anti-scintillement (médiane sur 7 décisions, seuil de
  confirmation 3/5 frames), total recalculé par frame uniquement sur les
  pièces confirmées.
- HUD dans une fenêtre OpenCV : total, comptes par dénomination, pièces
  incertaines (jamais comptées), profil actif, échelle, temps/frame.

## 3. Ce que la version 1 ne fait PAS

- **Billets de banque** (rectangle → message « billet non gere »).
- **Pièces empilées ou chevauchantes non résolues** : si l'éclatement
  (watershed) échoue, la pièce est déclarée incertaine et le dernier total
  confirmé est **gelé** — jamais de devinette sur de l'argent.
- Pièces en main, pièces partiellement hors champ, pièces fortement
  inclinées (gate d'inclinaison), pièces éblouies (gate d'éclat).
- **Garantie sur les monnaies étrangères** : les euros (2 € 25,75 mm,
  50c 24,25 mm…), capsules de bouteilles (26 mm), capsules PET (28 mm) et
  rondelles (20/24/28 mm) **passent la porte de taille** — ils sont écartés
  par les portes de circularité, d'absence de trou central et de couleur
  métallique, jamais par le seul diamètre.
- Texte arabe à l'écran (OpenCV `putText` produirait du mojibake).
- OCR, apprentissage automatique, interface Streamlit, détection de
  fréquences (filtres FFT).

## 4. Conditions d'acquisition (documentées, hors code)

| Condition | Valeur |
|---|---|
| Résolution de capture | 1280 × 720 minimum |
| Distance caméra ↔ pièces | 20 – 25 cm |
| Lumière | diffuse, incidente à ~45° (pas de reflet spéculaire) |
| Fond | sombre et mat (contraste avec les deux familles de teintes) |
| Disposition | pièces non superposées, entièrement visibles |

## 5. Références monétaires (Bank Al-Maghrib / Numista)

| Dénomination | Ø mm | Famille | Bimétallique |
|---|---|---|---|
| 10c | 20 | or | non |
| 20c | 23 | or | non |
| ½ DH | 21 | argent | non |
| 1 DH | 24 | argent | non |
| 2 DH | 26 | argent | non |
| 5 DH | 25 (variantes 26,2 / 29) | bimétallique | centre or |
| 10 DH | 27 (variantes 26,9 / 28) | bimétallique | centre argent |

Les diamètres ne sont **pas monotones** avec la valeur et l'écart entre
classes voisines n'est que de 1 mm : la classification repose sur
**taille + famille de couleur + polarité bimétallique** (les deux derniers
sont des **vetos**, jamais des scores pondérés), avec une marge minimale
au-delà de laquelle la pièce est déclarée `inconnue` et **non comptée**.

## 6. Affirmation officielle (modèle — à ne pas dépasser)

> Le détecteur n'applique **aucun** filtre cv-app au flux réel. La
> robustesse est mesurée par la matrice de `tests/test_robustesse.py`
> (L1/L2/L3 sur `coins.jpg` + cas de l'échelle d'analyse) et les portes
> de performance de `tests/test_perf.py` — voir `docs/05-robustesse.md`.
> Exceptions documentées : `negatif` (polarité — gérée par seuillage en
> double polarité), filtres géométriques (échelle/orientation —
> tolérances d'ellipse + recalibrage), rotations fortes (≥ ~8°) et
> ombrage/perspective marqués (masque de profilage conçu pour un fond
> mat sombre). Le flux webcam n'applique aucun filtre. Rien n'est garanti
> hors des conditions documentées (fond mat sombre, lumière diffuse,
> pièces non superposées, étalonnage effectué, 20–25 cm).

Toute métrique issue de `tests/` est libellée **cible**, jamais **garanti**,
tant que `donnees/veriteTerrain.json` ne contient pas de scénario réel
rempli (étape 6 de la recette humaine).

## 7. Porosité aux filtres : pourquoi c'est gérable

Le détecteur n'utilise **aucun** filtre de cv-app. À l'entrée, il choisit
dans un ensemble **fermé** de profils de prétraitement
(`defaut`, `sel_poivre`, `contraste`, `eclairage`, `flou`) par des
mesures objectives (fraction d'impulsions, uniformité par quadrants,
variance du Laplacien, contraste RMS) sur des seuils versionnés dans
`donnees/config.json` — puis seuille en **double polarité** (Otsu global,
BINARY et BINARY_INV pour absorber les fonds obstruants). La matrice de
robustesse (`tests/test_robustesse.py`) dégrade `coins.jpg` par
classes L1/L2/L3 (flou, sel & poivre, rotation, échelle, luminosité,
contraste, ombrage, perspective) et exige sur chaque cas le **compte des
portes gravées** (mesures, pas des objectifs) ; le cas « échelle
d'analyse » (réduction 800→640) est une porte dédiée. Toute exception est
documentée dans `docs/05-robustesse.md`.

## 8. Pièces de preuve

| Fichier | Rôle |
|---|---|
| `../TP7/exercice 5/coins.jpg` + `donnees/coins_annotations.json` | fixture (800×800, 16 blobs) — **détection uniquement** |
| `donnees/config.json` | seuils versionnés (identiques aux défauts du code) |
| `donnees/veriteTerrain.json` | captures réelles — vide tant que la recette humaine n'est pas faite |
| `docs/05-robustesse.md` | matrice de robustesse remplie + tableau de performances |
