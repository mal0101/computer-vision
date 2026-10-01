# 03 — Architecture

## Le principe directeur

**Un filtre déclare ses paramètres ; les interfaces se construisent à partir de
cette déclaration.**

Aucune interface ne connaît le nom d'un filtre en particulier. L'interface web
ne sait pas ce qu'est un « seuil de Canny » : elle lit un objet
`IntParam(name="seuil_bas", min=0, max=500, value=50)` et en déduit qu'il lui
faut un curseur entier de 0 à 500 démarrant à 50. L'application de bureau lit
la *même* déclaration et en déduit une trackbar. La ligne de commande s'en sert
pour valider `seuil_bas=60`.

Conséquence pratique : **ajouter un filtre, c'est écrire une fonction et un
décorateur**. Il devient aussitôt disponible dans les trois interfaces, dans la
documentation générée et dans la suite de tests, sans toucher une ligne de code
d'interface. Voir [12 — Étendre l'application](12-etendre.md).

## Les couches

```
        ┌──────────────────┬──────────────────┬──────────────────┐
        │ ui/streamlit_app │   ui/desktop     │    cvlab/cli     │   interfaces
        │   + ui/widgets   │  (trackbars)     │  (arguments)     │
        └────────┬─────────┴────────┬─────────┴────────┬─────────┘
                 └──────────────────┼──────────────────┘
                                    ▼
                        ┌───────────────────────┐
                        │   cvlab/pipeline.py   │   chaînes d'étapes, presets
                        └───────────┬───────────┘
                                    ▼
                        ┌───────────────────────┐
                        │   cvlab/registry.py   │   catalogue, validation,
                        │   cvlab/params.py     │   application unitaire
                        └───────────┬───────────┘
                                    ▼
                        ┌───────────────────────┐
                        │    cvlab/filters/     │   46 fonctions OpenCV
                        └───────────┬───────────┘
                                    ▼
        ┌────────────┬──────────────┼──────────────┬────────────┐
        │  images.py │  drawing.py  │ histogram.py │ metrics.py │   socle
        │ (BGR/uint8)│   (TP2)      │   (TP4 §8)   │ (TP7 ex.4) │
        └────────────┴──────────────┴──────────────┴────────────┘
                                    ▼
                           OpenCV  +  NumPy
```

**Règle de dépendance** : les flèches ne remontent jamais. `cvlab` ne dépend
d'aucune interface ; `ui` dépend de `cvlab`. On peut donc importer `cvlab` dans
un notebook ou un script sans installer Streamlit.

## Les objets du modèle

### `ParamSpec` — un paramètre réglable

Un objet immuable (`frozen dataclass`) portant tout ce qu'il faut savoir sur un
paramètre : nom technique, libellé affiché, texte d'aide, type, bornes, valeur
par défaut, groupe d'affichage, caractère avancé ou non.

Deux méthodes font tout le travail :

- `coerce(valeur)` — valide et **ramène dans les bornes**. Un noyau de 8 fourni
  par un preset devient 9 ; un seuil de 900 devient 500. L'application ne
  rejette presque jamais : elle corrige et continue.
- `resolve(valeur)` — traduit en argument OpenCV. Seul `ChoiceParam` fait une
  vraie traduction : l'utilisateur et les presets manipulent la chaîne lisible
  `"Ellipse"`, la fonction reçoit l'entier `cv2.MORPH_ELLIPSE`.

Voir [09 — Paramètres](09-parametres.md) pour les six types disponibles.

### `FilterDef` — un filtre du catalogue

Produit par le décorateur `@register(...)`. Il porte les métadonnées
(identifiant stable, nom, famille, résumé, principe, TP d'origine), la liste
des `ParamSpec`, et la fonction de traitement.

Sa méthode `apply(image, valeurs, reference)` orchestre tout :

1. `resolve` les valeurs → arguments OpenCV ;
2. `prepare` l'image selon `input_mode` (conversion en gris ou promotion en
   BGR, pour que la fonction n'ait jamais à s'en soucier) ;
3. redimensionne la deuxième image si le filtre en réclame une ;
4. appelle la fonction, en enveloppant toute erreur OpenCV dans une
   `FilterError` porteuse d'un message lisible ;
5. normalise la sortie selon `output_mode` et renvoie `(image, infos)`.

Le dictionnaire `infos` sert aux valeurs *calculées* à montrer à
l'utilisateur : le seuil trouvé par Otsu, le σ effectif d'un gaussien, le
pourcentage de pixels de contour, le noyau de convolution utilisé.

### `Pipeline` — une suite d'étapes

Une liste de `Step` (filtre + valeurs + actif/inactif). `apply()` passe l'image
d'une étape à la suivante et renvoie un `PipelineResult` qui conserve, pour
chaque étape, l'image intermédiaire, les infos, la durée et l'éventuelle
erreur. C'est ce qui alimente l'affichage pas-à-pas et le graphique des durées.

Une étape en erreur n'interrompt pas la chaîne : elle est signalée et ignorée,
la suite continue avec l'image précédente. Sinon, un mauvais réglage
momentané rendrait l'interface inutilisable le temps de le corriger. La ligne
de commande, elle, utilise `stop_on_error=True` : un échec doit être visible
dans un script.

### Conventions d'image

Toute image qui circule dans `cvlab` est un tableau NumPy `uint8`, soit
`(H, W)` en niveaux de gris, soit `(H, W, 3)` en **BGR**. Les images à quatre
canaux, en 16 bits ou en flottant sont ramenées à cette forme dès le
chargement, par `cvlab.images.normalise`. Un seul endroit convertit en RGB pour
l'affichage : `cvlab.images.to_rgb`, appelé par les interfaces.

## Choix d'implémentation et leurs raisons

**Positions et tailles en pourcentage, jamais en pixels.** Les filtres
géométriques (recadrage, translation, points de perspective) et les
annotations expriment leurs paramètres en pourcentage des dimensions de
l'image. Un preset réglé sur une photo 4000×3000 reste donc valable sur une
capture webcam 640×480, et les bornes des curseurs n'ont pas à être recalculées
à chaque changement d'image.

**Les filtres ne modifient jamais leur entrée.** Sans cette garantie,
l'affichage pas-à-pas montrerait partout la même image. Un test automatique le
vérifie pour les 46 filtres.

**Le bruit est reproductible sur demande.** Chaque générateur de bruit accepte
une graine : à graine fixée, deux filtres peuvent être comparés sur exactement
le même bruit. Graine à 0 = tirage différent à chaque calcul, ce qui est le bon
comportement sur un flux webcam (un bruit figé se verrait comme un calque
immobile).

**Les contraintes vivent dans la déclaration, pas dans les fonctions.** Un
noyau de flou médian doit être impair et ≥ 3 : c'est écrit une fois, dans
`IntParam(min=3, odd_only=True)`. L'interface web en déduit un curseur qui
avance de 2 en 2, l'application de bureau une trackbar dont chaque cran est
impair, la ligne de commande corrige la valeur saisie. Il est impossible
d'envoyer un noyau pair à OpenCV.

**Une résolution de travail réglable.** L'interface web réduit par défaut
l'image à 900 px de côté. Une photo de 12 Mpx rendrait chaque mouvement de
curseur poussif. Le fait est signalé à l'utilisateur, car la taille change
l'effet *apparent* d'un noyau : un flou 5×5 est bien plus visible sur une image
de 400 px que sur une de 4000 px.

## Flux d'une interaction (interface web)

1. L'utilisateur bouge un curseur → Streamlit relance le script.
2. `construire_chaine()` reconstruit un `Pipeline` depuis `st.session_state`.
3. `pipeline.apply()` recalcule la chaîne et conserve chaque étape.
4. L'éditeur affiche les curseurs ; leurs nouvelles valeurs sont écrites dans
   `session_state`.
5. La chaîne est recalculée une seconde fois avec ces valeurs à jour, puis
   affichée. (Ce deuxième calcul est le prix à payer pour que l'éditeur montre
   les durées et les valeurs calculées de l'étape en cours ; sur une image à la
   résolution de travail, il reste de l'ordre de quelques dizaines de
   millisecondes.)

L'état de session ne contient que des structures sérialisables — les étapes
sont des dictionnaires, pas des objets `Pipeline`. Elles s'exportent donc
directement en preset JSON, et l'état survit aux rechargements de code.
