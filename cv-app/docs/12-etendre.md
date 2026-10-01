# 12 — Étendre l'application

## Ajouter un filtre

Un filtre, c'est **une fonction et un décorateur**. Il apparaît ensuite tout
seul dans l'interface web, dans l'application de bureau, dans la ligne de
commande, dans la documentation générée et dans la suite de tests.

### 1. Écrire la fonction

Dans le module de la famille appropriée (`cvlab/filters/smoothing.py`,
`edges.py`, `morphology.py`…) :

```python
@register(
    id="flou_de_mouvement",                    # identifiant STABLE
    name="Flou de mouvement",                  # nom affiché
    family="Lissage & débruitage",             # doit figurer dans FAMILY_ORDER
    summary="Simule un filé directionnel.",    # une phrase, terminée par un point
    theory=(
        "Convolution par un segment de droite : tous les coefficients valent "
        "1/n le long d'une direction, 0 ailleurs. C'est le modèle du flou "
        "produit par un déplacement de l'appareil pendant la pose."
    ),
    tp=("complément",),                        # TP d'origine, obligatoire
    params=(
        IntParam(name="longueur", label="Longueur du filé", value=15,
                 min=3, max=61, odd_only=True, unit="px",
                 help="Distance parcourue pendant la pose."),
        FloatParam(name="angle", label="Direction", value=0.0,
                   min=-90.0, max=90.0, step=1.0, unit="°"),
    ),
)
def flou_de_mouvement(image, longueur, angle):
    noyau = np.zeros((longueur, longueur), np.float32)
    noyau[longueur // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D(
        (longueur / 2 - 0.5, longueur / 2 - 0.5), float(angle), 1.0
    )
    noyau = cv2.warpAffine(noyau, rotation, (longueur, longueur))
    noyau /= max(noyau.sum(), 1e-6)            # préserver la luminosité
    return cv2.filter2D(image, -1, noyau)
```

### 2. Les règles à respecter

| Règle | Pourquoi |
| --- | --- |
| **Ne jamais modifier l'image reçue** — renvoyer une nouvelle image | sinon l'affichage pas-à-pas et la comparaison avant/après n'ont plus de sens ; un test le vérifie |
| Renvoyer un tableau `uint8`, forme `(H, W)` ou `(H, W, 3)` BGR | convention unique du projet ; `apply()` normalise, mais mieux vaut être propre |
| Déclarer **toutes** les bornes dans les `ParamSpec` | l'interface ne peut alors produire aucune valeur refusée par OpenCV |
| Renseigner `theory` et `tp` | ce sont des champs obligatoires, vérifiés par un test ; ils alimentent l'aide et la documentation |
| Terminer `summary` par un point | vérifié par un test, pour l'homogénéité de l'affichage |
| Utiliser `input_mode="gray"` si le filtre exige un canal unique | l'image est convertie automatiquement, la fonction n'a plus à s'en soucier |
| Marquer `realtime_safe=False` si le filtre est lent | l'interface prévient l'utilisateur |
| Lever `FilterError` pour un cas réellement impossible | le message s'affiche proprement au lieu d'une trace d'exception |

### 3. Renvoyer des valeurs calculées (facultatif)

Pour afficher une valeur obtenue pendant le calcul, renvoyer un couple :

```python
    return sortie, {"σ effectif": f"{sigma:.2f}", "taille": f"{l}×{h} px"}
```

Ces valeurs apparaissent sous les curseurs, dans l'incrustation de
l'application de bureau et avec `--etapes` en ligne de commande.

### 4. Vérifier

```bash
python -m pytest                     # les tests génériques couvrent le nouveau filtre
python -m cvlab.cli decrire flou_de_mouvement
python outils/generer_doc.py         # régénérer la référence (sinon un test échoue)
```

Les tests génériques appliquent automatiquement le nouveau filtre à des images
couleur, en niveaux de gris et minuscules, à **chaque borne de chacun de ses
paramètres**, et vérifient qu'il ne modifie pas son entrée. Il est donc
inutile — et déconseillé — d'écrire un test par filtre pour ces aspects.
Écrivez en revanche un test de **comportement** dans
`tests/test_filters_behaviour.py` : qu'est-ce que votre filtre doit *faire*
qu'une simple absence d'exception ne prouve pas ?

## Ajouter une famille

1. Ajouter le nom à `FAMILY_ORDER` dans `cvlab/registry.py`, à sa place dans la
   progression des TP (l'ordre de ce tuple est l'ordre d'affichage).
2. Créer `cvlab/filters/ma_famille.py`.
3. L'importer dans `cvlab/filters/__init__.py`.

Un filtre déclarant une famille absente de `FAMILY_ORDER` échoue à
l'enregistrement, avec un message qui indique quoi faire.

## Ajouter un type de paramètre

Rare, mais prévu :

1. Créer la classe dans `cvlab/params.py` (hériter de `ParamSpec`, implémenter
   `default` et `coerce`, fixer `kind`).
2. Ajouter sa branche dans `ui/widgets.py` → `widget_pour_parametre()`.
3. Ajouter sa branche dans `ui/desktop.py` → `EchelleTrackbar.pour()`, ou le
   laisser renvoyer `None` s'il n'est pas représentable par une trackbar — il
   faut alors l'ajouter à la liste des types sans trackbar dans le test
   `test_echelle_couvre_toutes_les_positions`.
4. Ajouter son rendu dans `outils/generer_doc.py` → `_type_lisible()`.

## Utiliser le noyau dans vos propres scripts

```python
from cvlab import io_utils, metrics, registry
from cvlab.pipeline import Pipeline

# Un filtre isolé
image = io_utils.load_image("photo.jpg")
contours, infos = registry.get("canny").apply(
    image, {"seuil_bas": 60, "seuil_haut": 180}
)
print(infos["pixels de contour"])

# Une chaîne
chaine = Pipeline(nom="Mon traitement")
chaine.add("filtre_gaussien", {"taille_noyau": 7})
chaine.add("canny", {"seuil_bas": 50, "seuil_haut": 150, "flou_prealable": 0})
resultat = chaine.apply(image)

for etape in resultat.steps:
    print(f"{etape.name}: {etape.duree_ms:.1f} ms", etape.infos)

io_utils.save_image(resultat.image, "sortie.png")
chaine.save("ma_chaine.json")

# Comparer deux traitements
print(metrics.compare(image, resultat.image))
```

Voir [15 — Référence de l'API](15-api.md) pour le détail des modules.

## Conventions de code du projet

- **Français** pour les noms publics, les libellés, les messages et les
  commentaires : le projet prolonge des TP rédigés en français, et les noms des
  fonctions de dessin reprennent exactement ceux du TP2.
- **Annotations de type** partout, `from __future__ import annotations` en tête.
- **Les commentaires expliquent le pourquoi**, pas le comment : les contraintes
  d'OpenCV, les pièges, les choix de conception. Un commentaire qui paraphrase
  le code est à supprimer.
- **Les docstrings de module** situent le fichier dans l'ensemble et citent le
  TP d'origine.
