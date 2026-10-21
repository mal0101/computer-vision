# 05 — Robustesse et performance (M4)

Mesures **réelles**, pas des valeurs rêvées : chaque porte est gravée dans
`tests/test_robustesse.py` et `tests/test_perf.py` (marqueur `perf`,
hors tour par défaut, réactivable par `make perf`). Outil :
`python bench.py [--image …] [--largeur 640] [--seuil-ms 33]`.

## 1. Stratégie temps réel

`detecter` doit tenir ≤ 33 ms par frame à l'échelle d'analyse. La chaîne
est purement **contour-based** : le contre-vote de Hough, qui ne servait
qu'à réécrire une confiance que personne ne lisait, a été **retiré**
(~15 % du budget de la frame pour rien). Deux optimisations structurelles
restent (au lieu de micro-refactors) :

1. **Les deux polarités sont analysées**, sans court-circuit. Une sonde
   (`connectedComponentsWithStats`) sautait auparavant l'analyse complète de
   la polarité perdante, créditée de **−11 ms**. Remesurée sur la chaîne
   actuelle, elle coûte 0,39 ms pour éviter 0,15 ms de travail : une perte
   nette de 0,24 ms par frame, la fermeture morphologique plus large ayant
   entre-temps rendu les masques bien plus simples à parcourir.

   Son invariant était de surcroît faux — elle prétendait *borner* le nombre
   de pièces d'une polarité, ce qui cesse d'être vrai dès que des pièces se
   touchent ou touchent le bord. Sa suppression a fait passer le rappel sur
   photographies réelles de **76 % à 91 %** : sur fond clair, elle écartait
   la bonne polarité sans jamais l'évaluer.
2. **Contours externes uniquement** — chaque silhouette est traitée une
   fois (`RETR_EXTERNAL`), sans hiérarchie ni parcours d'arbres.

Conséquences mesurées sur `coins.jpg` : **médiane 5,6 ms** en 640×640
(min 5,5, p95 5,8, 40 itérations après échauffement) et **8,6 ms** en
800×800 natif pour la détection ; **7,2 ms** pour la classification de
16 pièces (22,3 ms auparavant, le masque couleur étant construit sur la
boîte englobante de chaque pièce et non sur l'image entière). Détection
**déterministe** : 50 passes bit à bit identiques (testé).

## 2. Matrice de robustesse (coins.jpg, 16 pièces, seed gravée)

Dégradations appliquées avant détection, à la **résolution native**
(800×800). Score = pièces dont la grande bordure (IoU ≥ 0,60) est trouvée
devant la référence. Seuil de classe = borne basse mesurée dans
`tests/test_robustesse.py`.

| Classe | Dégradation | Score mesuré | Porte |
|---|---|---|---|
| base | aucun dégradation, native | **16/16** | — |
| L1 | flou gaussien σ = 6 | **16** | ≥ 15 |
| L1 | sel & poivre 1 % | **16** | ≥ 16 |
| L1 | rotation 1,5° | **16** | ≥ 16 |
| L1 | échelle ×0,98 | **16** | ≥ 16 |
| L1 | luminosité +10 (α = 0,92) | **16** | ≥ 16 |
| L2 | flou gaussien σ = 1,5 | **16** | ≥ 12 |
| L2 | sel & poivre 4 % | **16** | ≥ 12 |
| L2 | rotation 4° | **15** | ≥ 12 |
| L2 | contraste 0,7 −15 | **16** | ≥ 12 |
| L3 | flou gaussien σ = 3 | **16** | ≥ 3 |
| L3 | ombrage latéral | **16** | ≥ 3 |
| L3 | perspective légère | **8** | ≥ 3 |
| L3 | rotation 12° | **3** | ≥ 3 |

> L'échelle d'analyse réelle (640×640, `largeur_analyse`) est celle qui
> compte, puisque c'est celle que le pipeline emploie : **16/16** depuis la
> réparation, contre 14/16 auparavant. Gravée en porte dédiée
> `test_l0_reduction_echelle_analyse` (== 16), et doublée d'une porte de
> bout en bout dans `tests/test_acceptation.py`.
>
> Le gain vient de la circularité mesurée sur l'**enveloppe convexe** du
> contour plutôt que sur le contour brut : la gravure d'une pièce creuse
> dans la silhouette seuillée des anses qui allongent le périmètre sans
> toucher l'aire, et `4πA/P²` s'effondre au carré (mesuré sur une pièce :
> 0,245 contre 0,895 pour sa voisine de même taille). L'enveloppe les
> ignore ; la convexité perdue est reprise par `solidite`, la rondeur vraie
> par la porte d'occlusion.

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
- **Rappel photos mesuré : 42/46 (91,3 %), surplus 0** sur
  `donnees/photos.json` (banc de détection, 6 scènes ; `surplus` =
  détections en trop, qui doit rester 0). Les 4 pièces manquées sont des
  **artefacts de masque**, pas des disques à rattraper : arcs fins
  sous-contrastés (`jouet_6`, `uae_grille` : circ. 0,38–0,52) et bords de
  pièces rognés par un fond clair (`us_grille` : solidité 0,90–0,92).
  Vérifié par balayage : abaisser `solidite_min` (0,92 → 0,90) *réduit* le
  rappel à 41/46 (les blobs raclés entrent en compétition NMS avec les
  vraies pièces) et `circularite_min` (0,85 → 0,50) ne change rien — les
  artefacts sont filtrés ailleurs. **Aucune porte globale ne récupère ces
  pièces** : les chasser coûterait de la précision.
- **Un seul cœur ⇒ un seul disque.** `_tenter_eclatement` ne tente le
  watershed que si la distanceTransform révèle ≥ 2 cœurs ; un disque isolé
  plus gros que `rayon_max_rel` (pièce tenue près de la caméra, grosse
  pièce sur le banc) est rendu comme pièce à part entière au lieu d'être
  marqué « chevauchement » à tort (mesuré : 3 disques de circ. 0,998 et un
  seul cœur sur `jouet_6` récupérés ainsi).