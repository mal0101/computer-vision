# 02 — Classification et étalonnage (M2)

Ce document contient les décisions d'ingénierie de l'identification (M2).
Il est **factuel** : chaque règle y est implémentée et testée dans
`pieces/classification.py`, `pieces/etalonnage.py` et
`tests/test_classification.py`.

## 1. Ordre de décision : couleur d'abord, taille ensuite

Toute pièce est un **veto couleur** avant tout calcul de taille :

1. **Famille** — la teinte « or » (bande hue ouverte 12–45°, saturation
   ≥ 60) sans disque central sombre conclut à `or` ; une saturation faible
   (< 60) à `argent` ; une pièce trop sombre (valeur < 45) ou trop mottée
   (couverture < 30 %) à `inconnue`. Une famille `or` n'est jamais
   identifiée comme `argent` et réciproquement.
2. **Polarité bimétallique** — si le centre et la couronne divergent en
   teinte (≥ 14° en hue OpenCV) **et que la couronne est réellement un
   second métal** (ni fond sombre, ni gravure trop dense), la pièce est
   `bimétallique` : `centre_or` (5 DH) ou `centre_argent` (10 DH). La
   polarité est une décision *purement colorimétrique*, utilisable sans
   échelle.
3. **Taille** — le diamètre mesuré (fit ellipsoïdal prioritaire, sinon
   `2·rayon`) est converti en millimètres par `px_par_mm` et classé contre
   les variantes de chaque dénomination.

L'identification est **exhaustive** : le filtre `centimes` (10c/20c)
n'agit **qu'au comptage** (étape M3). Une pièce de 20c est toujours
identifiable ; elle n'est simplement pas ajoutée au total sans `--centimes`.

## 2. Score de taille et seuils

- Pour chaque dénomination candidat, `écart = min |mm − variant|`.
- Il faut `meilleur_écart ≤ 1,5 mm` (une mesure trop loin de toutes les
  variantes est `inconnue`).
- Il faut `marge = 2ᵉ écart − meilleur écart ≥ marge_classif (0,20 mm)`
  sinon `inconnue` : un 25,0 mm pile entre 1 DH (24,0) et 2 DH (26,0) n'est
  pas identifiable.
- Hors `portee_mm = [18,5 ; 30,5]` la pièce reçoit le motif
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
physiquement plausible à l'échelle d'analyse (`≥ px_mm_min = 2,0` et
`≤ 640/18,5 ≈ 34,6` px/mm). `controle_coherence` réalise un contrôle croisé
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
| plage de diamètre | 18,5–30,5 mm | 10c (20,0) … 10 DH (28,0) |
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