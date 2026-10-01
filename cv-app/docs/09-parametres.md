# 09 — Paramètres : types, bornes et validation

Chaque filtre déclare ses paramètres sous forme d'objets `ParamSpec`
(`cvlab/params.py`). Cette déclaration est l'unique source de vérité : elle
alimente les curseurs de l'interface web, les trackbars de l'application de
bureau, la validation de la ligne de commande et la
[référence générée](08-reference-filtres.md).

## Les six types

| Classe | Widget web | Trackbar | Exemple |
| --- | --- | --- | --- |
| `IntParam` | curseur entier | oui | taille de noyau, seuil |
| `FloatParam` | curseur réel | oui (200 crans) | σ, α, gamma |
| `BoolParam` | interrupteur | oui (0/1) | norme L2 de Canny |
| `ChoiceParam` | liste déroulante | oui (une position par option) | type de seuillage, interpolation |
| `ColorParam` | sélecteur de couleur | **non** | couleur d'annotation |
| `TextParam` | champ de saisie | **non** | texte d'annotation |

Les deux derniers n'ont pas d'équivalent en trackbar : l'application de bureau
utilise alors la valeur par défaut et le signale dans le terminal.

## Attributs communs

| Attribut | Rôle |
| --- | --- |
| `name` | nom de l'argument passé à la fonction, et clé dans les presets et la ligne de commande |
| `label` | libellé affiché |
| `help` | infobulle expliquant le rôle du paramètre |
| `group` | regroupe des paramètres liés dans un dépliant (les 4 coins d'une homographie) |
| `advanced` | masque le paramètre sous *Réglages avancés* |
| `unit` | unité ajoutée au libellé (`px`, `%`, `°`) — `IntParam` et `FloatParam` |

## Attributs spécifiques

```python
IntParam(
    name="taille_noyau", label="Taille du noyau",
    value=5,            # valeur par défaut
    min=3, max=31,      # bornes
    step=1,             # pas du curseur
    odd_only=True,      # contraint à rester impair
    unit="px",
)

FloatParam(name="sigma", label="σ", value=0.0, min=0.0, max=20.0,
           step=0.1, decimals=2)

ChoiceParam(name="forme", label="Forme", value="Ellipse",
            options={"Rectangle": cv2.MORPH_RECT,
                     "Ellipse": cv2.MORPH_ELLIPSE,
                     "Croix": cv2.MORPH_CROSS})

ColorParam(name="couleur", label="Couleur", value=(0, 255, 0))   # BGR
TextParam(name="texte", label="Texte", value="", max_length=120)
BoolParam(name="inverser", label="Inverser", value=False)
```

## Validation : `coerce`

`coerce(valeur)` **ramène dans les bornes** plutôt que de rejeter.

| Entrée | `IntParam(value=5, min=3, max=31, odd_only=True)` |
| --- | --- |
| `8` | `9` — arrondi à l'impair supérieur |
| `-10` | `3` — borné en bas |
| `1000` | `31` — borné en haut |
| `"7"` | `7` — chaîne acceptée (ligne de commande) |
| `"beaucoup"` | `ValueError` explicite |

Ce choix est délibéré : un preset enregistré avec une version antérieure, ou
une valeur tapée un peu vite en ligne de commande, ne doivent pas bloquer le
travail. Seules les valeurs **dont le sens est indéterminable** (texte
non numérique, option inexistante, `NaN`) sont refusées.

`ColorParam` accepte en plus les chaînes hexadécimales `#rrggbb` et `#rgb`
(format des sélecteurs web) et les convertit en triplet BGR.
`BoolParam` accepte `vrai`/`faux`, `true`/`false`, `oui`/`non`, `1`/`0`.

## Traduction : `resolve`

`resolve(valeur)` traduit en argument utilisable par OpenCV. Seul
`ChoiceParam` fait une vraie traduction :

```
utilisateur et preset :  "Ellipse"
fonction du filtre    :  cv2.MORPH_ELLIPSE   (entier 2)
```

Les presets stockent donc la **chaîne lisible**, jamais la constante : un
preset reste lisible à l'œil et survit à un changement de valeur numérique
d'une constante OpenCV.

## Garantie sur les tailles de noyau impaires

OpenCV refuse un noyau pair pour `GaussianBlur`, `medianBlur`, `Laplacian` ou
`adaptiveThreshold`. L'application rend l'erreur **impossible**, à trois
niveaux :

1. `coerce` corrige toute valeur paire reçue de l'extérieur ;
2. l'interface web construit un curseur de pas 2 partant d'une borne impaire ;
3. l'application de bureau construit une trackbar dont chaque cran correspond à
   un impair.

Un test automatique parcourt **toutes** les positions de **toutes** les
trackbars de **tous** les filtres et vérifie que la valeur obtenue est
canonique (`coerce(v) == v`).

## Valeurs calculées en retour

Symétriquement, un filtre peut renvoyer des valeurs *calculées* à montrer à
l'utilisateur, sous forme d'un dictionnaire `infos` :

| Filtre | Valeurs renvoyées |
| --- | --- |
| Seuillage automatique | le seuil trouvé, le pourcentage de pixels non nuls |
| Filtre gaussien | le σ effectivement utilisé (utile quand on laisse σ = 0) |
| Détecteur de Canny | seuils utilisés, rapport haut/bas, pourcentage de contours |
| Noyau personnalisé | le noyau complet et la somme de ses coefficients |
| Redimensionner, recadrer, perspective | la taille du résultat |
| Masque de couleur | le pourcentage de pixels sélectionnés |

Elles s'affichent sous les curseurs dans l'interface web, dans l'incrustation
de l'application de bureau, et avec `--etapes` en ligne de commande.
