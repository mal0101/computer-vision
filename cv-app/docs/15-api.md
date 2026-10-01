# 15 — Référence de l'API

Le paquet `cvlab` s'utilise directement depuis un script ou un notebook. Il ne
dépend que d'OpenCV et de NumPy.

## `cvlab.images` — conventions d'image

Toute image manipulée par `cvlab` est un tableau NumPy `uint8`, soit `(H, W)`
en niveaux de gris, soit `(H, W, 3)` en **BGR**.

| Fonction | Rôle |
| --- | --- |
| `normalise(image)` | ramène à la convention : BGRA → BGR, `(H,W,1)` → `(H,W)`, 16 bits et flottants → `uint8` |
| `is_gray(image)`, `is_color(image)` | nature de l'image |
| `to_gray(image)`, `to_bgr(image)` | conversions |
| `to_rgb(image)` | **pour l'affichage** (Streamlit, Matplotlib, PIL) |
| `match_shape(image, reference)` | redimensionne `reference` à la taille d'`image` |
| `match_channels(a, b)` | aligne le nombre de canaux (gris promu en BGR) |
| `describe_image(image)` | dictionnaire de métadonnées (taille, canaux, min, max, moyenne…) |

`InvalidImageError` est levée pour une image absente, vide ou de forme non
prise en charge.

## `cvlab.params` — déclaration des paramètres

`ParamSpec` et ses six sous-classes : `IntParam`, `FloatParam`, `BoolParam`,
`ChoiceParam`, `ColorParam`, `TextParam`. Voir
[09 — Paramètres](09-parametres.md).

Utilitaires : `force_odd(valeur, minimum)`, `bgr_to_hex(couleur)`.

## `cvlab.registry` — catalogue

| Fonction | Rôle |
| --- | --- |
| `get(filter_id)` | le `FilterDef` d'identifiant donné ; lève `UnknownFilterError` |
| `all_filters()` | tous les filtres, triés par famille puis par nom |
| `families()` | les familles non vides, dans l'ordre d'affichage |
| `filters_by_family()` | dictionnaire `famille → filtres` |
| `register(**kwargs)` | décorateur d'enregistrement (voir [12](12-etendre.md)) |

### `FilterDef`

| Membre | Rôle |
| --- | --- |
| `id`, `name`, `family`, `summary`, `theory`, `tp`, `notes` | métadonnées |
| `params` | tuple de `ParamSpec` |
| `input_mode`, `output_mode` | `any`/`gray`/`color`, `same`/`gray`/`color` |
| `needs_reference`, `realtime_safe` | contraintes d'usage |
| `defaults()` | dictionnaire des valeurs par défaut |
| `coerce(valeurs)` | valide et complète |
| `resolve(valeurs)` | valide puis traduit en arguments OpenCV |
| `apply(image, valeurs, reference)` | **renvoie `(image, infos)`** ; lève `FilterError` |
| `describe()` | fiche texte multi-lignes |

## `cvlab.pipeline` — chaînes

| Objet | Rôle |
| --- | --- |
| `Step(filter_id, params, enabled)` | une étape |
| `Pipeline(steps, nom)` | la chaîne |
| `Pipeline.add(id, params, index)` | ajoute une étape |
| `Pipeline.remove(i)`, `.move(i, delta)`, `.clear()` | édition |
| `Pipeline.apply(image, reference, collect, stop_on_error)` | → `PipelineResult` |
| `Pipeline.to_json()`, `.from_json()`, `.save()`, `.load()` | presets |
| `Pipeline.needs_reference` | la chaîne réclame-t-elle une deuxième image |
| `Pipeline.summary()` | résumé texte |

`PipelineResult` porte `image` (le résultat), `steps` (une `StepResult` par
étape : image intermédiaire, paramètres, `infos`, `duree_ms`, `error`,
`skipped`), `warnings` et `duree_totale_ms`.

## `cvlab.io_utils` — entrées/sorties

| Fonction | Rôle |
| --- | --- |
| `load_image(chemin, flag)` | charge ; lève `ImageLoadError` (gère les chemins accentués) |
| `decode_image(octets)` | décode depuis la mémoire |
| `encode_image(image, extension, qualite)` | encode en mémoire |
| `save_image(image, chemin, qualite)` | écrit sur disque, crée les dossiers |
| `sample_images(racine, limite)` | images d'exemple des dossiers `TP*` et `assets/` |
| `Camera(index, largeur, hauteur, fps)` | webcam ; gestionnaire de contexte |
| `capture_snapshot(index, echauffement)` | une seule photo |

`Camera` expose `open()`, `read()`, `read_optional()`, `set_property()`,
`resolution`, `release()`. Lève `CameraError`.

## `cvlab.metrics` — mesures

`mse`, `rmse`, `mae`, `psnr`, `difference(a, b, amplification)`,
`compare(a, b)`. Les images de tailles ou de canaux différents sont alignées
automatiquement.

## `cvlab.histogram` — histogrammes

| Fonction | Rôle |
| --- | --- |
| `histogram_data(image, bins, masque)` | `{canal: tableau de comptes}` |
| `statistics(image)` | min, max, moyenne, écart-type, médiane par canal |
| `draw_histogram(image, largeur, hauteur, cumule, echelle_log)` | **une image BGR** du tracé |

Le tracé est une image OpenCV : affichable aussi bien dans Streamlit que dans
une fenêtre `cv2.imshow`, sans dépendance de tracé supplémentaire.

## `cvlab.drawing` — dessin (TP2)

`dess_ligne`, `dess_rectangle`, `dess_cercle`, `dess_ellipse`,
`dess_polylignes`, `dess_text`, `dess_texte_encadre`, `taille_texte`.
Dictionnaires `POLICES` et `TYPES_LIGNE`.

Mêmes noms et mêmes signatures que `TP2/codes TP2/fonctions/dessin.py`, à une
différence près : **elles renvoient une copie** au lieu de modifier l'image sur
place.

## Exemple complet

```python
from cvlab import histogram, io_utils, metrics, registry
from cvlab.pipeline import Pipeline

image = io_utils.load_image("photo.jpg")

# 1. Un filtre isolé, avec ses valeurs calculées
contours, infos = registry.get("canny").apply(
    image, {"seuil_bas": 60, "seuil_haut": 180}
)
print(infos)          # {'seuils utilisés': '60 / 180', 'rapport haut/bas': '3.00', ...}

# 2. Une chaîne, étape par étape
chaine = Pipeline(nom="Segmentation")
chaine.add("niveaux_de_gris")
chaine.add("filtre_gaussien", {"taille_noyau": 5})
chaine.add("seuillage_automatique", {"methode": "Otsu"})
chaine.add("morphologie", {"operation": "Ouverture", "taille": 5})

resultat = chaine.apply(image)
for etape in resultat.steps:
    print(f"{etape.index + 1}. {etape.name:<32} {etape.duree_ms:6.1f} ms  {etape.infos}")

# 3. Mesurer, tracer, enregistrer
print(metrics.compare(image, resultat.image))
io_utils.save_image(histogram.draw_histogram(resultat.image), "histogramme.png")
io_utils.save_image(resultat.image, "segmente.png")
chaine.save("presets/ma_segmentation.json")

# 4. Explorer le catalogue
for definition in registry.all_filters():
    if "TP7" in " ".join(definition.tp):
        print(definition.id, "—", definition.summary)
```
