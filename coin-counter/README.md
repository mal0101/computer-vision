# coin-counter

Compteur de **dirhams marocains** par traitement d'image classique (OpenCV),
sans aucun apprentissage automatique. Détecte les pièces de la série 2002
— {5c, 10c, 20c, ½, 1, 2, 5, 10} DH — les identifie, et **additionne** en
centimes entiers (`37,00 DH`).

Deux régimes, volontairement distincts :

* **image fixe** → un *inventaire* : « quelles pièces vois-tu ici, et combien
  font-elles ? » ;
* **webcam** → un *total suivi* : une pièce doit être revue plusieurs frames
  avant d'être comptée, et ne peut l'être qu'une seule fois.

Code et commentaires en français. Tests pytest ; le marqueur `perf` est hors
du tour par défaut et réactivé par `make perf`.

## Démarrage rapide

```bash
make venv                                    # crée .venv depuis requirements.lock
make autotest                                # autotest complet, sans caméra
make check                                   # ruff + suite de tests
make image IMAGE=../TP7/exercice\ 5/coins.jpg
make webcam                                  # comptage en direct
```

```
$ python -m pieces.cli --image "../TP7/exercice 5/coins.jpg"
échelle estimée au catalogue : 5.61 px/mm (16/16 pièces concordantes)
pièces détectées : 16 (profil defaut)
échelle : 5.61 px/mm (estimée au catalogue)
TOTAL : 37,00 DH   1/2 DH×2  1 DH×2  2 DH×2  5 DH×2  10 DH×2
```

## Mesures

Toutes obtenues sur le **chemin réellement exécuté** par le programme
(réduction à `largeur_analyse = 640` comprise), reproductibles par
`make check`, `make perf` et `make bench`.

| Mesure | Valeur | Porte |
|---|---|---|
| Détection, planche TP7 `coins.jpg` (16 pièces) | **16/16** | `test_chemin_de_production_detecte_les_16_pieces` |
| Identification, mêmes 16 pièces, sans étalonnage | **16/16** | `test_identification_exacte_sur_la_planche_reelle` |
| Total, mêmes 16 pièces | **37,00 DH** (exact) | `test_total_exact_sur_la_planche_reelle` |
| Détection, 6 photographies réelles (46 pièces) | **87 %** (40/46) | porte ≥ 85 % |
| Faux positifs sur image unie | **0** | porte = 0 |
| Détection | **7,2 ms** médiane à 640 px | porte ≤ 33 ms |
| Classification | **7,2 ms** pour 16 pièces | — |

Les six pièces manquantes du banc photographique se touchent, ou leur
gravure creuse tant le masque seuillé que la silhouette cesse d'être un
disque. Aucune porte n'a été relâchée pour les rattraper : un passage
intermédiaire atteignait 91 %, mais seulement parce qu'un raccourci laissait
alors passer carrés, ellipses, croissants et rondelles comme pièces
comptables. 87 % avec des portes qui tiennent vaut mieux que 91 % avec un
trou par lequel une rondelle devient 2 DH.

## Échelle : trois sources, jamais confondues

Convertir des pixels en millimètres demande une échelle. Par ordre de
préférence :

1. `--echelle-px-mm N` — échelle **mesurée**, fournie par l'utilisateur ;
2. étalonnage interactif (touche `c`) — on présente une pièce de diamètre
   connu, seule dans le champ ;
3. **estimation au catalogue** — les diamètres marocains forment un ensemble
   discret connu, l'échelle est donc l'unique paramètre d'un ajustement par
   consensus, contraint par le veto de couleur.

La troisième est ce qui rend le programme utilisable sans mire ni caméra.
Elle est **toujours annoncée comme estimée** (HUD, sortie texte, JSON) et
jamais présentée comme une mesure. Précision constatée : −0,9 % sur scène
synthétique à échelle imposée, −2,3 % sur la planche photographique —
suffisant pour identifier correctement les seize pièces, parce que la couleur
sépare déjà les familles et qu'il ne reste à trancher que des écarts d'au
moins 2 mm. `--sans-auto-echelle` la désactive : le total vaut alors 0 plutôt
qu'un chiffre inventé.

## Ce que le programme refuse de faire

- **Compter une pièce qu'il n'a pas identifiée avec certitude.** Une
  classification seulement « tentative » n'entre jamais dans un total.
- **Compter deux fois la même pièce** : ni après une sortie du champ, ni
  après une dérive de rayon. La protection dure `memoire_comptees` frames
  (90, soit ~3 s) — au-delà, une pièce réellement reposée redevient
  comptable. C'est une limite assumée, pas une garantie absolue.
- **Deviner une échelle** à partir de disques tous identiques : ils ne
  déterminent rien, et l'estimation renvoie « indéterminé ».
- **Prendre une rondelle, un croissant ou un carré pour une pièce** : sept
  formes adverses sont gravées en test (`test_forme_non_monnayee_jamais_valide`).
- Billets, pièces empilées non séparables, pièces fortement inclinées ou
  éblouies : signalées, jamais comptées.

## Utilisation

```bash
python -m pieces.cli --help
python -m pieces.cli --sans-fenetre                     # autotest, aucun matériel
python -m pieces.cli --image photo.jpg --centimes       # inventaire, avec 5c/10c/20c
python -m pieces.cli --image photo.jpg --json rapport.json --lister-pieces
python -m pieces.cli --image photo.jpg --echelle-px-mm 5.6
python -m pieces.cli --camera 0 --limite 300 --sortie sorties/hud.png
```

Fenêtré : `Échap`/`q` quitter, `c` étalonnage, `d` vignette du masque,
`r` remise à zéro (conserve l'échelle).

Codes de sortie : `0` succès, `1` erreur d'exécution (caméra, pipeline),
`2` argument ou configuration invalide.

## Architecture

```
pieces/
  config.py          Config (@dataclass) + catalogue des dénominations + json
  entites.py         PieceDetectee, Classification, ResultatImage
  erreurs.py         erreurs typées
  detection.py       detecter : profils, Otsu double polarité, portes, NMS
  classification.py  couleur (saturation) → famille/polarité → taille
  etalonnage.py      étalonnage de session + estimation au catalogue
  scenes.py          scènes synthétiques à échelle imposée (bancs sans caméra)
  suivi.py           Suiveur : machine d'états, comptage non répété
  traitement.py      traiter_image (flux) et inventorier_image (image fixe)
  rendu.py           dessiner_hud, dessiner_hud_inventaire
  ui/bureau.py       Bureau (boucle webcam, touches)
  cli.py             argparse + modes (image / webcam / autotest)
  compat.py          façade vers cv-app ; facultative, hors chemin de production
tests/
  test_acceptation.py  portes de bout en bout (« le programme fait-il son travail ? »)
  test_detection.py / test_classification.py / test_suivi.py / test_m3.py …
donnees/
  config.json          seuils publiés
  coins_annotations.json  annotation INDÉPENDANTE de coins.jpg (16 pièces)
  photos/ + photos.json   banc de 6 photographies réelles (46 pièces)
bench.py             tableau des temps d'exécution
```

## Documentation

- `docs/01-portee.md` — périmètre, conditions d'acquisition, références.
- `docs/02-classification.md` — couleur, polarité, taille, échelle.
- `docs/03-suivi-comptage.md` — machine d'états, non-répétition du comptage.
- `docs/04-interface-utilisateur.md` — CLI, HUD, Makefile.
- `docs/05-robustesse.md` — matrice de dégradations, performance, limites.
