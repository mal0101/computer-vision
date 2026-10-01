# 04 — Interface web (Streamlit)

```bash
streamlit run ui/streamlit_app.py
```

L'application s'ouvre sur `http://localhost:8501`. Pour y accéder depuis une
autre machine du réseau : `streamlit run ui/streamlit_app.py --server.address 0.0.0.0`.

## Barre latérale

### 1 · Image source

Quatre origines, au choix :

| Origine | Détail |
| --- | --- |
| **Fichier** | téléversement depuis votre ordinateur — `jpg`, `jpeg`, `png`, `bmp`, `tif`, `tiff`, `webp`, `ppm`, `pgm`, 50 Mo au plus |
| **Webcam (instantané)** | deux voies : le **capteur du navigateur** (`st.camera_input`, fonctionne même à distance) ou la **caméra locale via OpenCV** (bouton *Capturer*, utile si le navigateur ne voit pas la caméra) |
| **Exemple des TP** | les images des dossiers `TP1/` à `TP7/` du dépôt, plus celles que vous déposez dans `cv-app/assets/` |
| **Mire de test** | une image générée : damier, dégradé, cercles, traits fins et texte — faite pour mettre les filtres en difficulté |

> **Flux vidéo continu.** L'interface web ne propose que des *instantanés* de
> webcam : un navigateur ne peut pas fournir à Streamlit un flux image par
> image sans dépendance supplémentaire. Le flux continu est dans l'application
> de bureau : `python ui/desktop.py --webcam`.

**Résolution de travail** (plus grand côté : 400, 600, 900, 1200, 1600 px, ou
taille réelle). L'image est réduite avant traitement pour que l'aperçu reste
réactif. Attention : la taille change l'effet *apparent* d'un noyau. Un flou
5×5 se voit beaucoup plus sur une image de 400 px que sur une de 4000 px. Pour
un export final, repassez en *taille réelle*.

### 2 · Deuxième image

Nécessaire pour la *Fusion pondérée*, les *Opérations binaires* et le
*Masquage par une seconde image*. Elle est automatiquement redimensionnée à la
taille de l'image courante, comme le faisait `TraitementImage.fusion()` du
TP4. La section s'ouvre d'elle-même et affiche un avertissement si la chaîne
contient un filtre qui en a besoin.

### 3 · Chaînes enregistrées

- **Exporter en JSON** — enregistre la chaîne courante (filtres, ordre, toutes
  les valeurs). Voir [10 — Chaînes et presets](10-chaines-et-presets.md).
- **Importer un JSON** — recharge une chaîne.
- **Chaînes d'exemple** — les huit chaînes fournies dans `presets/`.

## Colonne de gauche : l'éditeur de chaîne

Choisissez une **famille**, puis un **filtre**, puis *Ajouter*. Le bouton
*Détails du filtre sélectionné* ouvre sa fiche complète (principe, TP
d'origine, liste des paramètres) avant même de l'ajouter.

Chaque étape de la chaîne affiche :

| Contrôle | Effet |
| --- | --- |
| **Actif** | désactive l'étape sans la supprimer — la meilleure façon d'isoler l'effet d'un filtre |
| **↑ ↓** | déplace l'étape ; l'ordre change le résultat |
| **⟲** | remet les paramètres par défaut |
| **✕** | supprime l'étape |

Sous les contrôles, les **curseurs des paramètres**, construits à partir de la
déclaration du filtre : curseurs entiers (avançant de 2 en 2 pour les tailles
de noyau, qui doivent rester impaires), curseurs réels, interrupteurs, listes
déroulantes, sélecteurs de couleur, champs de texte. Les paramètres liés (les
quatre coins d'une homographie, les bornes d'un masque HSV) sont regroupés dans
un dépliant ; les paramètres rares sont sous *Réglages avancés*.

Chaque étape affiche enfin sa **durée** et ses **valeurs calculées** : le seuil
trouvé par Otsu, le σ effectif d'un gaussien, le pourcentage de pixels de
contour, le noyau de convolution utilisé, la taille du résultat.

## Colonne de droite : les sept onglets

### Résultat

Quatre modes d'affichage : *Côte à côte*, *Résultat seul*, *Source seule*, et
**Superposition réglable** — un curseur fait passer progressivement de la
source au résultat, ce qui révèle des différences subtiles qu'une comparaison
côte à côte laisse passer.

En bas : export en `png` (sans perte), `jpg`, `webp` ou `bmp`, avec réglage de
qualité pour les formats avec perte.

### Étapes

L'image telle qu'elle sort de **chaque** étape, en vignettes. C'est la façon la
plus directe de comprendre pourquoi l'ordre des filtres compte. Un graphique
donne la durée de chaque étape.

### Analyse

- **Histogrammes** de la source et du résultat, côte à côte, en mode normal ou
  **cumulé** (la courbe cumulée est la clé pour comprendre l'égalisation :
  égaliser revient à redresser cette courbe en droite), avec échelle
  **logarithmique** optionnelle — indispensable quand un pic unique écrase tout
  le tracé, par exemple sur une image seuillée.
- **Statistiques** par canal : min, max, moyenne, écart-type, médiane.
- **Mesures d'écart** entre source et résultat : MSE, RMSE, MAE, PSNR.
- **Carte des écarts** : `|source − résultat|`, amplifiable, pour voir
  *où* le filtre a agi.

> **Attention à la lecture des mesures.** Elles quantifient une *différence*,
> pas une *qualité*. Si la source est bruitée, un bon débruitage s'en écarte
> nécessairement, donc sa MSE est élevée. Pour évaluer un débruitage, bruitez
> vous-même une image propre (famille *Bruit*) et comparez le résultat à
> l'originale — c'est le protocole de l'exercice 4 du TP7.

### Pixels

L'inspecteur de pixel, équivalent de l'exercice « La souris » du TP3 : donnez
des coordonnées (x, y) et lisez la valeur du pixel **avant et après**
traitement, en BGR, en HSV et en hexadécimal, avec un aperçu de la couleur, le
tableau des valeurs du voisinage et son agrandissement au plus proche voisin.

Rappel du TP2 : l'origine (0, 0) est en **haut à gauche**, x désigne la colonne
et y la ligne — l'inverse du repère mathématique habituel.

### Comparateur

Applique plusieurs filtres **indépendamment** à la même image et les compare,
images et chiffres côte à côte. C'est le protocole des exercices 1 à 4 du TP7.

Le choix *Référence pour les mesures* permet de comparer soit à l'image source,
soit à la sortie de la première étape de la chaîne principale — ce qui donne la
comparaison honnête : mettez un filtre de bruit en première étape, et la
référence devient l'image propre.

### Catalogue

Les 46 filtres, avec recherche sur le nom, le principe ou le TP d'origine, et
un bouton pour ajouter directement à la chaîne.

### Aide

Mode d'emploi court, tableau des trois interfaces, pièges fréquents.

## Raccourcis et comportements utiles

- La **mire de test** se régénère au bouton ; elle est déterministe.
- Les réglages d'un filtre sont **conservés** quand vous désactivez puis
  réactivez l'étape.
- Une étape en erreur (filtre de fusion sans deuxième image, par exemple) est
  signalée en rouge et **ignorée** : le reste de la chaîne continue de
  s'appliquer, l'interface reste utilisable.
- Le thème sombre est volontaire : c'est la convention des logiciels de
  traitement d'image, car un entourage clair fausse la perception du contraste.
  Il se change dans `.streamlit/config.toml`.
