# 10 — Chaînes et presets

## Pourquoi l'ordre compte

Une chaîne applique les filtres **dans l'ordre**, l'image de l'étape *n*
devenant l'entrée de l'étape *n+1*. Ce n'est pas commutatif :

| Chaîne | Résultat |
| --- | --- |
| flou gaussien → Canny | contours nets, peu de bruit — **la bonne séquence** |
| Canny → flou gaussien | des contours bruités, ensuite floutés : inutilisable |
| seuillage → ouverture | le bruit clair isolé disparaît — **la bonne séquence** |
| ouverture → seuillage | l'ouverture agit sur des niveaux de gris, l'effet est différent et moins net |
| bruit → médian | on peut évaluer le débruitage |
| médian → bruit | on ajoute du bruit à une image lissée : aucun intérêt |

L'onglet *Étapes* de l'interface web montre l'image après chaque étape : c'est
la façon la plus directe de s'en convaincre.

## Comportement en cas d'erreur

Dans les interfaces graphiques, une étape qui échoue est **signalée et
ignorée** : la chaîne continue avec l'image de l'étape précédente. Sans cela,
un réglage momentanément invalide rendrait l'interface inutilisable le temps de
le corriger.

En ligne de commande, c'est l'inverse : la première erreur **interrompt** le
traitement et renvoie un code de retour non nul, car un échec doit être visible
dans un script.

## Format de preset

Fichier JSON, encodage UTF-8 :

```json
{
  "version": 1,
  "nom": "Contours de Canny (TP7 ex.7)",
  "etapes": [
    {
      "filtre": "niveaux_de_gris",
      "actif": true,
      "parametres": {}
    },
    {
      "filtre": "filtre_gaussien",
      "actif": true,
      "parametres": {
        "taille_noyau": 5,
        "sigma_x": 0.0,
        "sigma_y": 0.0,
        "bord": "Miroir (défaut)"
      }
    },
    {
      "filtre": "canny",
      "actif": true,
      "parametres": {
        "seuil_bas": 50,
        "seuil_haut": 150,
        "ouverture": 3,
        "norme_l2": false,
        "flou_prealable": 0
      }
    }
  ]
}
```

Points à retenir :

- `filtre` est l'**identifiant stable** du filtre (colonne `id` de la
  [référence](08-reference-filtres.md)), jamais son nom affiché.
- Les paramètres de type *choix* sont écrits en clair (`"Miroir (défaut)"`),
  pas sous forme de constante numérique : le fichier reste lisible et survit à
  un changement de valeur d'une constante OpenCV.
- Un paramètre absent prend sa valeur par défaut ; un paramètre **inconnu** est
  ignoré silencieusement, pour qu'un preset ancien reste utilisable après la
  disparition d'un réglage.
- Un `filtre` inconnu, en revanche, provoque une erreur explicite au
  chargement : mieux vaut échouer que d'appliquer une chaîne silencieusement
  incomplète.
- Les clés anglaises `steps` / `filter` / `params` / `enabled` sont acceptées
  en lecture, par tolérance.

## Utiliser les presets

**Interface web** — barre latérale → *3 · Chaînes enregistrées* : exporter,
importer, ou charger une des chaînes d'exemple.

**Ligne de commande** :

```bash
python -m cvlab.cli appliquer photo.jpg --preset presets/01_contours_canny.json -o sortie.png
python -m cvlab.cli appliquer photo.jpg -s negatif --sauver-preset ma_chaine.json
```

**Application de bureau** — `--preset` applique la chaîne **avant** le filtre
réglé par les trackbars, ce qui permet de figer un prétraitement et de n'ajuster
que la dernière étape :

```bash
python ui/desktop.py --webcam --preset presets/01_contours_canny.json --filtre morphologie
```

**Depuis vos scripts** :

```python
from cvlab import io_utils
from cvlab.pipeline import Pipeline

chaine = Pipeline.load("presets/01_contours_canny.json")
image = io_utils.load_image("photo.jpg")
resultat = chaine.apply(image)
io_utils.save_image(resultat.image, "sortie.png")
```

## Les huit chaînes fournies

| Fichier | Contenu | Illustre |
| --- | --- | --- |
| `01_contours_canny.json` | gris → gaussien → Canny | TP7 ex.7 : la séquence correcte pour détecter des contours |
| `02_debruitage_sel_poivre.json` | bruit sel-poivre (graine 42) → médian | TP7 ex.3 : le médian face au bruit impulsionnel |
| `03_gradient_sobel.json` | gaussien → module du gradient → fausses couleurs | TP7 ex.5 : lire une carte de gradient |
| `04_segmentation_cellules.json` | gris → gaussien → Otsu → ouverture → gradient morphologique | segmentation complète d'objets |
| `05_document_redresse.json` | perspective → gris → seuillage adaptatif | TP4 §3 : redresser et binariser une page |
| `06_correction_exposition.json` | CLAHE → gamma → accentuation | TP4 §8 : récupérer une photo terne |
| `07_canaux_hsv.json` | teinte décalée, saturation poussée → histogramme | TP2 §1.3 : manipuler l'espace HSV |
| `08_effet_dessin.json` | bilatéral → posterisation → accentuation | effet stylisé par aplatissement des couleurs |

Chaque preset est vérifié par la suite de tests : il doit se charger, s'appliquer
sans avertissement et produire une image non vide.
