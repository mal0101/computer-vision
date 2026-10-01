# 06 — Ligne de commande

```bash
python -m cvlab.cli --help
```

Utile pour traiter un lot d'images, reproduire exactement une mesure, ou
consulter le catalogue sans ouvrir d'interface graphique.

## Syntaxe d'une étape

```
FILTRE[:clé=valeur,clé=valeur,...]
```

Les clés sont les noms techniques des paramètres (colonne `Paramètre` de la
[référence](08-reference-filtres.md), ou `python -m cvlab.cli decrire FILTRE`).
Les paramètres non mentionnés prennent leur valeur par défaut. Les valeurs hors
bornes sont **corrigées** (un noyau de 8 devient 9), un nom de paramètre
inconnu est **refusé** avec la liste des noms valides.

Les booléens acceptent `vrai`/`faux`, `true`/`false`, `oui`/`non`, `1`/`0`.

## Les cinq commandes

### `lister` — le catalogue

```bash
python -m cvlab.cli lister
python -m cvlab.cli lister --famille "Contours"
python -m cvlab.cli lister -v                 # avec résumés et paramètres
```

### `decrire` — la fiche d'un filtre

```bash
python -m cvlab.cli decrire filtre_gaussien
```

```
filtre_gaussien  —  Filtre gaussien
  famille  : Lissage & débruitage
  résumé   : Moyenne pondérée par une cloche gaussienne.
  origine  : TP7 ex.2
  entrée   : any   sortie : same
  principe : Le noyau vaut G(i, j) = exp(−(i² + j²) / 2σ²), normalisé...
  paramètres :
    - taille_noyau (entier, impair) [1..31] défaut=5 px
    - sigma_x (réel) [0.0..20.0] défaut=0.0
    ...
```

### `appliquer` — traiter une image

```bash
# Une chaîne donnée en ligne de commande
python -m cvlab.cli appliquer photo.jpg \
    -s "filtre_gaussien:taille_noyau=7,sigma_x=1.5" \
    -s "canny:seuil_bas=60,seuil_haut=180,flou_prealable=0" \
    -o contours.png --etapes

# Une chaîne enregistrée depuis l'interface web
python -m cvlab.cli appliquer photo.jpg --preset presets/01_contours_canny.json -o sortie.png

# Un instantané de webcam comme source
python -m cvlab.cli appliquer --webcam 0 -s negatif -o inverse.png

# Un filtre à deux images
python -m cvlab.cli appliquer a.jpg -s "fusion:alpha=0.35" --reference b.jpg -o melange.jpg
```

| Option | Rôle |
| --- | --- |
| `-s`, `--etape` | une étape ; répétable, appliquée dans l'ordre |
| `--preset FICHIER` | chaîne JSON ; les `-s` éventuels sont ajoutés après |
| `--webcam INDEX` | instantané de webcam au lieu d'un fichier |
| `--reference IMAGE` | deuxième image (fusion, opérations binaires) |
| `-o`, `--sortie` | fichier image de sortie |
| `--histogramme FICHIER` | enregistre aussi le tracé de l'histogramme du résultat |
| `--sauver-preset FICHIER` | enregistre la chaîne utilisée en JSON |
| `--qualite N` | qualité JPEG/WebP (95 par défaut) |
| `--etapes` | détaille chaque étape : durée et valeurs calculées |

La commande affiche les caractéristiques de la source et du résultat, le
détail de la chaîne, la durée totale, et l'écart entre source et résultat.
Contrairement aux interfaces graphiques, une étape en erreur **interrompt** le
traitement et renvoie un code de retour non nul : un échec doit être visible
dans un script.

### `comparer` — évaluer plusieurs filtres (TP7 ex.4)

```bash
python -m cvlab.cli comparer "../TP7/exercice 1/image_bruitee_1.jpg" \
    -s filtre_moyenneur -s filtre_gaussien -s filtre_median -s filtre_bilateral
```

```
filtre                                     MSE     RMSE      MAE   PSNR dB       ms
-----------------------------------------------------------------------------------
Filtre moyenneur (boîte)                478.48    21.87    17.28     21.33      1.5
Filtre gaussien                         358.62    18.94    15.05     22.58      0.3
Filtre médian                           455.90    21.35    16.50     21.54      1.1
Filtre bilatéral                        174.28    13.20    10.57     25.72      2.2
```

> **Lecture.** Sans `--reference`, la référence est l'image d'entrée
> elle-même : la MSE mesure alors *à quel point le filtre a modifié l'image*,
> pas sa qualité de débruitage. Pour une évaluation honnête, partez d'une image
> propre, bruitez-la, puis comparez les résultats à l'originale :

```bash
# 1. bruiter une image propre, de façon reproductible
python -m cvlab.cli appliquer propre.png \
    -s "bruit_sel_poivre:proportion=0.08,graine=42" -o bruitee.png

# 2. comparer les filtres à l'image PROPRE
python -m cvlab.cli comparer bruitee.png --reference propre.png \
    -s "filtre_moyenneur:largeur_noyau=5,hauteur_noyau=5" \
    -s "filtre_gaussien:taille_noyau=5" \
    -s "filtre_median:taille_noyau=5" \
    -s "filtre_bilateral:diametre=9" \
    --dossier-sortie resultats/
```

Le médian l'emporte alors nettement, conformément à ce qu'annonce la théorie.

### `webcam` — tester la caméra

```bash
python -m cvlab.cli webcam --index 0 -o instantane.png
python -m cvlab.cli webcam --index 1 --largeur 1280 --hauteur 720
```

Affiche la résolution obtenue et les caractéristiques de l'image lue, ou un
message d'erreur explicite si la caméra est inaccessible.

## Traitement par lot

```bash
# Toutes les images d'un dossier, même chaîne
for f in photos/*.jpg; do
  python -m cvlab.cli appliquer "$f" \
      --preset presets/06_correction_exposition.json \
      -o "corrigees/$(basename "$f")"
done
```

## Codes de retour

| Code | Signification |
| --- | --- |
| `0` | succès |
| `1` | erreur d'exécution (fichier illisible, filtre en échec, webcam inaccessible) |
| `2` | erreur d'usage (étape manquante, paramètre inconnu, filtre inexistant) |
| `130` | interruption au clavier |
