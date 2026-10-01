# 14 — Dépannage

## Installation et lancement

**`ModuleNotFoundError: No module named 'cv2'`**
L'environnement virtuel n'est pas activé, ou les dépendances ne sont pas
installées.
```bash
source .venv/bin/activate      # Windows : .venv\Scripts\activate
python -m pip install -r requirements.txt
```

**`ModuleNotFoundError: No module named 'cvlab'`**
Les commandes se lancent depuis le dossier `cv-app`, pas depuis la racine du
dépôt ni depuis `ui/`.
```bash
cd cv-app && streamlit run ui/streamlit_app.py
```

**`TypeError: dataclass() got an unexpected keyword argument 'kw_only'`**
Python antérieur à 3.10. Vérifier avec `python --version` et utiliser un
interpréteur plus récent.

**`streamlit : command not found`**
```bash
python -m streamlit run ui/streamlit_app.py
```

## Webcam

**« webcam 0 inaccessible »**

| Cause | Vérification |
| --- | --- |
| Autorisation refusée (macOS) | *Réglages Système → Confidentialité et sécurité → Caméra*, autoriser le terminal |
| Caméra occupée | fermer les visioconférences et les autres applications qui l'utilisent |
| Mauvais index | essayer `--index 1`, `--index 2` |
| Pas de caméra | `python -m cvlab.cli webcam --index 0` donne un diagnostic |

**L'image de la webcam est noire ou très sombre.**
Les premières images sortent souvent mal exposées, le temps que l'exposition
automatique se stabilise. `capture_snapshot()` jette déjà cinq images ;
augmenter avec `echauffement=15` si nécessaire. Dans l'application de bureau,
attendre une seconde avant de juger.

**La résolution demandée n'est pas appliquée.**
`capture.set` renvoie silencieusement `False` quand le pilote refuse la valeur.
La résolution réellement obtenue est affichée au lancement. Essayer des valeurs
standard : 640×480, 1280×720, 1920×1080.

**Le navigateur ne propose pas la caméra (interface web).**
`st.camera_input` exige un contexte sécurisé : `localhost` ou HTTPS. En accès
distant par `http://`, utiliser le bouton *Capturer (OpenCV)*, qui passe par la
caméra de la machine qui exécute l'application.

## Images

**« format non reconnu par OpenCV »**
Format non géré (HEIC d'iPhone, RAW d'appareil photo, SVG). Convertir en JPEG
ou PNG au préalable.

**L'image s'affiche avec les rouges et les bleus inversés.**
Dans l'application, non : la conversion est centralisée. Si cela arrive dans
**votre** code utilisant `cvlab`, c'est l'oubli classique de `to_rgb()` avant
un affichage Matplotlib ou Streamlit.

**L'image téléversée est refusée (trop grande).**
Limite fixée à 50 Mo dans `.streamlit/config.toml`
(`server.maxUploadSize`), modifiable.

**Un chemin avec des accents ne se charge pas.**
Pris en charge : `load_image` lit les octets en Python puis décode avec
`cv2.imdecode`, ce qui contourne la limitation de `cv2.imread` sur les chemins
non ASCII.

## Résultats inattendus

**Le filtre ne semble rien faire.**

| Vérifier | |
| --- | --- |
| l'étape est-elle **active** ? | l'interrupteur *Actif* |
| le paramètre est-il à sa **valeur neutre** ? | gamma = 1, α = 1, noyau = 1×1, intensité = 0 |
| la **résolution de travail** n'est-elle pas trop grande ? | un flou 3×3 est invisible sur une image de 4000 px |
| une étape **ultérieure** n'efface-t-elle pas l'effet ? | onglet *Étapes* |

**L'image devient toute noire ou toute blanche.**
Typiquement un seuillage mal réglé, ou un gradient dont l'échelle est trop
faible. Regarder l'**histogramme** (onglet *Analyse*, échelle logarithmique) :
il dira immédiatement si tout s'est accumulé sur une seule valeur.

**Canny ne donne presque aucun contour.**
Seuils trop hauts, ou image trop floutée en amont. Baisser les seuils en
conservant un rapport haut/bas de 2 à 3, et réduire le flou préalable. Le
pourcentage de pixels de contour affiché sous les curseurs aide à viser.

**Canny donne un fouillis de contours.**
L'inverse : monter les seuils, ou augmenter le flou préalable. Sur une image
bruitée, c'est le flou qui règle le problème, pas les seuils.

**Le seuillage global échoue sur une photo de document.**
Éclairage inégal. Utiliser le **seuillage adaptatif** : taille de bloc
supérieure à l'épaisseur des traits, constante C entre 2 et 10.
Voir [07 §7](07-theorie-filtrage.md#7-seuillage).

**Le filtre médian ne supprime pas mon bruit.**
Le médian traite le bruit **impulsionnel**. Pour un bruit gaussien, utiliser le
filtre gaussien ou le bilatéral. Identifier le bruit d'abord :
[07 §4](07-theorie-filtrage.md#4-le-bruit).

**L'égalisation d'histogramme dénature les couleurs.**
Le paramètre *Canal égalisé* est sans doute sur « Trois canaux BGR ». Choisir
un canal de luminance (Y de YCrCb, L de Lab) ; c'est le réglage par défaut.

**La déformation de perspective produit une image vide ou étirée.**
L'ordre des points est imposé : haut-gauche, haut-droit, bas-droit, bas-gauche.
Quatre points alignés déclenchent un message explicite. Dans l'application de
bureau, les désigner à la souris évite l'erreur.

## Performance

**L'interface web est lente à chaque mouvement de curseur.**

- réduire la **résolution de travail** (barre latérale) ;
- désactiver les étapes coûteuses pendant le réglage ;
- le **filtre bilatéral** est de loin le plus coûteux : grand diamètre + grande
  image = plusieurs secondes.

**L'application de bureau saccade sur la webcam.**
```bash
python ui/desktop.py --webcam --largeur 640 --hauteur 480
```
Ou figer l'image avec `c` pendant le réglage.

## Presets

**« preset en version N, cette application lit au plus la version 1 »**
Le fichier vient d'une version plus récente de l'application.

**« filtre inconnu »**
L'identifiant du filtre n'existe pas (ou plus). `python -m cvlab.cli lister`
donne les identifiants valides.

**Un paramètre de mon preset semble ignoré.**
Un paramètre inconnu est ignoré volontairement, pour qu'un preset ancien reste
utilisable. Vérifier l'orthographe du nom avec
`python -m cvlab.cli decrire FILTRE`.

## Tests

**`test_reference_a_jour` échoue.**
La référence des filtres est périmée :
```bash
python outils/generer_doc.py
```

**Les tests Streamlit sont lents.**
Ils lancent l'application une fois par filtre (une vingtaine de secondes au
total). Les ignorer pendant le développement :
```bash
python -m pytest --ignore=tests/test_ui_streamlit.py
```
