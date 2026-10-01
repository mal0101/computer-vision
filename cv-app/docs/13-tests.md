# 13 — Tests et qualité

```bash
python -m pytest                      # toute la suite
python -m pytest -q --no-header       # sortie courte
python -m pytest tests/test_filters_behaviour.py -v
python -m pytest -k "median or canny" # sélection par motif
python -m pytest --durations=10       # les tests les plus lents
```

La suite compte **plus de 600 tests** et s'exécute en moins de 30 secondes.

## Organisation

| Fichier | Portée | Ce qu'il attrape |
| --- | --- | --- |
| `test_params.py` | validation des paramètres | une borne mal posée, une coercition qui laisse passer une valeur invalide |
| `test_images.py` | conventions d'image | une conversion BGR/RGB oubliée, une image 16 bits ou BGRA mal normalisée |
| `test_registry.py` | **tous les filtres, génériquement** | un filtre qui plante à une borne, qui modifie son entrée, qui n'est pas documenté |
| `test_filters_behaviour.py` | **sémantique de chaque filtre** | une inversion d'argument qui ne lève aucune exception |
| `test_pipeline.py` | chaînes et presets | une sérialisation non réversible, un preset corrompu accepté |
| `test_metrics_histogram.py` | mesures et histogrammes | une formule fausse, une division par zéro sur image uniforme |
| `test_io_drawing.py` | fichiers, encodage, dessin | un chemin accentué illisible, une primitive de dessin qui mute son entrée |
| `test_cli.py` | ligne de commande | un code de retour incorrect, un message d'erreur absent |
| `test_ui_desktop.py` | logique de l'application de bureau | une trackbar qui peut produire une valeur invalide |
| `test_ui_streamlit.py` | interface web, via `AppTest` | une erreur dans n'importe quel onglet, pour n'importe quel filtre |

## Les trois tests qui comptent le plus

### 1. Toutes les bornes de tous les paramètres

`test_registry.py::test_bornes_des_parametres` applique chaque filtre à une
image couleur **et** en niveaux de gris, pour **chaque borne** de **chacun** de
ses paramètres. C'est ce test qui a révélé que le moteur de filtrage d'OpenCV
refuse `BORDER_WRAP` (assertion `columnBorderType != BORDER_WRAP`), alors que
les fonctions géométriques l'acceptent.

### 2. Toutes les positions de toutes les trackbars

`test_ui_desktop.py::test_echelle_couvre_toutes_les_positions` parcourt
**chaque position** de **chaque trackbar** de **chaque filtre** et vérifie que
la valeur obtenue est déjà canonique (`coerce(v) == v`). C'est la garantie
formelle qu'aucun réglage de l'application de bureau ne peut envoyer une valeur
invalide à OpenCV — un noyau pair, par exemple.

### 3. Chaque filtre dans l'interface web

`test_ui_streamlit.py::test_chaque_filtre_s_affiche` lance l'application
Streamlit (sans navigateur, via `streamlit.testing.v1.AppTest`) une fois par
filtre et vérifie qu'aucune exception ni erreur affichée ne survient. Comme
Streamlit rend les onglets côté serveur, **tout le code de tous les onglets**
s'exécute à chaque fois.

## Tests de sémantique

Ne pas planter ne suffit pas. `test_filters_behaviour.py` vérifie que chaque
filtre fait bien ce qu'il annonce. Quelques exemples :

- le **médian** obtient une MSE plus faible que le moyenneur *et* que le
  gaussien sur du bruit sel et poivre — le résultat central du TP7 ex.3,
  vérifié chiffres en main ;
- le **bilatéral** préserve mieux une marche d'escalier qu'un gaussien de force
  comparable ;
- **Sobel X** réagit à une transition verticale et **pas** à une horizontale ;
- le **module du gradient** donne le même maximum sur une transition
  horizontale et sur la même transition tournée de 90° (isotropie) ;
- le **négatif** est involutif, le **miroir** aussi ;
- une **correction gamma de 1.0** laisse l'image inchangée au bit près ;
- le noyau moyenneur **construit à la main** donne le même résultat que
  `cv2.blur` (TP7 ex.1) ;
- le **seuillage adaptatif** retrouve des traits sur une image à éclairage
  inégal, là où un seuil global coupe l'image en deux ;
- l'**ouverture** supprime les points isolés en préservant le grand objet ;
- égaliser la **luminance** altère moins la teinte qu'égaliser les trois canaux
  BGR séparément ;
- le bruit **sel et poivre** touche un pixel entier (les trois canaux), pas un
  canal isolé ;
- la **proportion** de pixels bruités est respectée à 2 % près.

## La documentation est testée aussi

`outils/generer_doc.py --verifier` échoue si `docs/08-reference-filtres.md` ne
correspond plus au catalogue. Un filtre ajouté, un paramètre renommé ou une
borne modifiée sans régénération fait échouer la suite : la référence **ne peut
pas** diverger du code.

Les huit presets fournis sont également vérifiés : chacun doit se charger,
s'appliquer sans avertissement et produire une image non vide.

## Ce qui n'est pas testé automatiquement

La **boucle d'affichage** de l'application de bureau (`cv2.imshow`,
`cv2.waitKey`) et la **capture webcam** réelle : elles demandent un serveur
graphique, un périphérique et un œil humain. Tout ce qui les entoure l'est :
conversion des trackbars, traduction des gestes souris, traitement, gestion des
erreurs, enregistrement.

Pour la vérification manuelle :

```bash
python -m cvlab.cli webcam --index 0 -o /tmp/essai.png   # accès à la caméra
python ui/desktop.py --webcam                            # boucle temps réel
```

## Style

Le projet est configuré pour `ruff` (`pyproject.toml`) :

```bash
python -m pip install ruff
ruff check .
```
