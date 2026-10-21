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
- Identification : {5c, 10c, 20c, ½, 1, 2, 5, 10} DH — les pièces de
  **5c, 10c et 20c ne sont comptées qu'avec l'option `--centimes`** (défaut :
  {½, 1, 2, 5, 10} DH). L'identification, elle, reste exhaustive.
- Sommation exacte en **centimes entiers**, affichée au format
  marocain : `12,50 DH` (virgule française).
- Échelle (px/mm) par l'une de trois voies : `--echelle-px-mm` (mesurée),
  étalonnage interactif sur une pièce de référence, ou **estimation au
  catalogue** des diamètres. Une estimation est toujours annoncée comme
  telle. Sans aucune des trois, aucun total n'est affiché — jamais un
  chiffre inventé.
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
| 5c | 17,5 | or (bronze d'aluminium) | non |
| 10c | 20,0 | or | non |
| 20c | 23,0 | or | non |
| ½ DH | 21,0 | argent | non |
| 1 DH | 24,0 | argent | non |
| 2 DH | 26,0 | argent (cupronickel) | non |
| 5 DH | 25,0 (variante 1987 : 26,2) | bimétallique | noyau or |
| 10 DH | 26,9 (variante 1995 : 28,0) | bimétallique | noyau argent |

Table recoupée Bank Al-Maghrib / Numista **et vérifiée par mesure** sur
`TP7/exercice 5/coins.jpg` : ajustement aux moindres carrés sur les sept
dénominations de diamètre connu, RMS 0,27 mm, écart maximal 0,39 mm. Le 5c,
exclu de l'ajustement, y retombe à 17,4 mm pour 17,5 mm au catalogue.

Trois corrections par rapport à la table d'origine :

* **5c ajouté.** La pièce circule et figure dans la fixture ; son absence la
  rendait définitivement « inconnue ». `portee_mm` descend à 16,5 mm.
* **5 DH : variante 29,0 mm retirée.** Le 5 DH « type 1980 » est
  *monométallique* cupronickel : inscrit sous une entrée bimétallique, il
  était inidentifiable, le veto de couleur le classant « argent », famille
  qui ne possède aucune variante à 29 mm.
* **10 DH : variante 27,0 mm retirée** — sans source, et à 0,1 mm de
  26,9 mm elle ne faisait qu'élargir la fenêtre d'acceptation.

Le **2 DH reste « argent »**. La fiche web de Bank Al-Maghrib annonce
simultanément « Noyau Or / Bague Argent » et « Cuivre 75 % – Nickel 25 % » :
ces deux mentions sont contradictoires, le cupronickel étant un alliage
unique de teinte argentée. Numista, uCoin et la mesure directe (saturation
7–12 au centre *comme* sur la couronne) concordent : la pièce est
monométallique.

Écart minimal **à l'intérieur** d'une même famille : 2,00 mm (1 DH / 2 DH),
soit dix fois la marge de décision. Les collisions entre familles — 2 DH
(26,0) contre 10 DH (26,9), par exemple — sont tranchées par la couleur.

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
