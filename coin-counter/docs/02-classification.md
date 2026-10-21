# 02 — Classification et étalonnage (M2)

Ce document contient les décisions d'ingénierie de l'identification (M2).
Il est **factuel** : chaque règle y est implémentée et testée dans
`pieces/classification.py`, `pieces/etalonnage.py` et
`tests/test_classification.py`.

## 1. Ordre de décision : couleur d'abord, taille ensuite

Toute pièce est un **veto couleur** avant tout calcul de taille :

1. **Famille** — décidée par la **saturation**, la teinte ne servant que de
   confirmation. Au-dessus de `SEUIL_SAT_COLORE` (30) : métal coloré
   (laiton, bronze d'aluminium) ; en dessous : métal clair (cupronickel,
   acier nickelé). Une pièce trop sombre (valeur < 45) reste `inconnue`.

   **Pourquoi pas la teinte.** Mesurée sur les seize pièces réelles de
   `coins.jpg`, elle est dégénérée : or comme argent tombent toutes entre 4
   et 28 (la seule valeur à 126 appartient à un ½ DH de saturation 3, c'est
   du bruit). La teinte d'une surface peu chromatique est le rapport de deux
   quantités presque nulles, donc instable par construction. La saturation,
   elle, sépare franchement :

   | famille | saturation mesurée |
   |---|---|
   | or (5c, 10c, 20c) | 78 – 95 |
   | argent (½, 1, 2 DH) | 3 – 15 |
   | bimétallique | une zone 39–66 face à l'autre 4–10 |

   L'ancienne règle, qui interrogeait la teinte d'abord et plaçait son
   plancher de saturation à 60, obtenait **11/14** : elle manquait les deux
   5 DH (noyaux de laiton à 47 et 39, sous le plancher) et un 10 DH. La
   règle par saturation obtient **16/16**.

2. **Polarité bimétallique** — déduite du **contraste de saturation** entre
   noyau et couronne (≥ 20), la zone la plus saturée désignant le laiton :
   `centre_or` (5 DH) ou `centre_argent` (10 DH). Comparer les *teintes* des
   deux zones revenait à comparer un signal à du bruit, puisque l'une des
   deux est argentée — mesuré, l'écart tombait à 9,6 pour un seuil de 14 et
   la pièce basculait en « argent ». La polarité reste une décision
   *purement colorimétrique*, utilisable sans échelle.
3. **Taille** — le diamètre mesuré (fit ellipsoïdal prioritaire, sinon
   `2·rayon`) est converti en millimètres par `px_par_mm` et classé contre
   les variantes de chaque dénomination.

L'identification est **exhaustive** : le filtre `centimes` (10c/20c)
n'agit **qu'au comptage** (étape M3). Une pièce de 20c est toujours
identifiable ; elle n'est simplement pas ajoutée au total sans `--centimes`.

## 2. Score de taille et seuils

- Le diamètre est mesuré par `classification.diametre_mesure`, le diamètre
  **équivalent en aire** `2·√(A/π)` — le même estimateur des deux côtés de
  la chaîne. Auparavant la classification employait l'axe majeur de
  l'ellipse ajustée et l'étalonnage le cercle circonscrit : mesuré sur les
  seize pièces, ces deux estimateurs diffèrent systématiquement de
  −0,59 % ± 0,15 %, soit ~0,15 mm à 25 mm — l'ordre de grandeur de
  `marge_classif` (0,20 mm). L'échelle était donc biaisée par rapport à ce
  qu'elle servait à mesurer.
- Pour chaque dénomination candidate, `écart = min |mm − variante|`.
- Il faut `meilleur_écart ≤ 1,5 mm` (une mesure trop loin de toutes les
  variantes est `inconnue`).
- Il faut `marge = 2ᵉ écart − meilleur écart ≥ marge_classif (0,20 mm)`
  sinon `inconnue` : un 25,0 mm pile entre 1 DH (24,0) et 2 DH (26,0) n'est
  pas identifiable.
- Hors `portee_mm = [16,5 ; 30,5]` la pièce reçoit le motif
  `hors_portee` : elle reste détectée mais n'est **jamais** comptée.
- La certitude `confirme` n'est accordée **qu'avec un étalonnage**
  (`px_par_mm` non nul). Sans échelle, la certitude plafonne à
  `tentative`, et seule une famille+polarité à candidat unique (le
  bimétallique) fixe une valeur.

## 3. Étalonnage : échelle de session, jamais persistée

`ObservateurEtalonnage` collecte le rayon détecté d'une pièce de **valeur
connue** tenue devant la caméra. L'échelle est déclarée stable quand :

```
nb rayons ≥ 8   ET   (max − min) ≤ 1 px
```

`etalonner_par_piece(obs, diametre_reel_mm)` retourne alors
`2 · rayon_médian / diamètre` sinon `None`. `echelle_depuis_carte`
propose la même échelle depuis la longueur d'une carte bancaire
(85,60 mm ISO/IEC 7810). `valider_echelle` borde `px_par_mm` dans une plage
physiquement plausible à l'échelle d'analyse (`≥ px_mm_min = 2,5` et
`≤ 640/16,5 ≈ 38,8` px/mm). `controle_coherence` réalise un contrôle croisé
sur une deuxième mesure connue à ± 2 %.

Rien n'est écrit sur disque : l'échelle existe pour la session seulement,
et l'interface (étape M3) doit afficher « non calibré » tant qu'aucune
échelle n'est validée.

## 4. Chiffres de référence (mesures, pas des valeurs rêvées)

| règle | valeur | mesure |
|---|---|---|
| différences minuscule de teinte centre/couronne | 14° hue | or (≈ 20°) vs argent (hue erratique, sat < 60) |
| seuil de saturation « métal coloré » | 60 | pièces synthétiques HSV→BGR |
| marge de séparation | 0,20 mm | 25,0 mm ambigus ; 24,0 mm non |
| plage de diamètre | 16,5–30,5 mm | 5c (17,5) … 10 DH (28,0) |
| contrôle croisé | ± 2 % | tolérance ISO |

## 5. Limites assumées

- La couleur est échantillonnée par *médiane/moyenne circulaire* sur les
  zones centre (0–0,45·r) et couronne (0,5–0,9·r) ; une pièce réelle
  usée ou à patine forte peut basculer en `inconnue` — c'est un
  comportement sûr (pas de mauvaise valeur, juste aucune).
- Le classement par taille suppose des pièces posées à plat.
- `coins.jpg` (TP7) n'a pas de vérité de dénomination : les tests M2 sont
  synthétiques couleur/diamètre ; la porte photographique reste la
  **détection** (IoU ≥ 0,60).