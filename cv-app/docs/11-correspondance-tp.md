# 11 — Correspondance avec les TP 1 à 7

Ce chapitre dit, notion par notion, **où chaque élément des travaux pratiques
se retrouve dans l'application**. Il sert de preuve de couverture et de guide
de lecture pour qui connaît les TP.

## Remarque sur les sources disponibles

Le dépôt contient les énoncés et les codes de **TP1, TP2, TP3, TP4 et TP7**.
Il n'y a dans le dépôt ni dossier `TP5/` ni dossier `TP6/`.

Les notions couvertes par l'application et qui ne proviennent pas directement
d'un de ces cinq TP sont donc marquées **`complément`** dans la
[référence des filtres](08-reference-filtres.md). Il s'agit du seuillage, de la
morphologie mathématique et de quelques variantes (Scharr, CLAHE, masque flou,
posterisation, bruit multiplicatif). Elles sont incluses parce qu'elles
complètent naturellement les familles traitées en TP — l'introduction du TP4
annonce d'ailleurs des sections sur la luminosité, le contraste, les opérations
bit à bit et l'histogramme — et parce qu'une chaîne de traitement réaliste
(segmenter des cellules, redresser un document) en a besoin. Elles sont
clairement identifiées comme telles, et aucun filtre issu d'un TP n'a été
omis.

---

## TP1 — Premiers pas avec Python et OpenCV

| Notion du TP | Source du TP | Où dans l'application |
| --- | --- | --- |
| `cv2.imread`, chargement d'une image | `Charger une image.py` (§3.1) | `cvlab/io_utils.py` → `load_image()` — avec une **erreur explicite** quand le fichier est absent, là où `cv2.imread` renvoie silencieusement `None` |
| Indicateurs `IMREAD_COLOR`, `GRAYSCALE`, `UNCHANGED` | §3.1 | `load_image(flag=...)`, et normalisation des images BGRA / 16 bits dans `cvlab/images.py` |
| `cv2.cvtColor`, conversion en niveaux de gris | `Convertir une image en niveau de gris .py` (§3.2) | filtre `niveaux_de_gris`, et `cvlab.images.to_gray()` |
| `cv2.imwrite`, enregistrement | §3.2 | `cvlab/io_utils.py` → `save_image()`, bouton de téléchargement de l'interface web, touche `s` de l'application de bureau |
| `cv2.VideoCapture`, lecture d'une vidéo | `Charger video.py` (§3.3) | `cvlab/io_utils.py` → classe `Camera` |
| **Webcam**, `VideoCapture(0)`, boucle d'affichage, sortie par ESC | `Afficher Webcam.py` (§3.4) | **`ui/desktop.py --webcam`** : boucle temps réel, sortie par ESC ; `capture_snapshot()` pour l'instantané de l'interface web |
| `capture.set(CAP_PROP_*)` : largeur, hauteur, luminosité, contraste | §3.4 | `Camera(largeur=, hauteur=, fps=)` et `Camera.set_property()`, options `--largeur` / `--hauteur` |
| `cv2.putText`, premier script | `Premier essai de script Python.py` (§2) | `cvlab/drawing.py` → `dess_text()`, filtre `annoter_texte` |
| `cv2.waitKey`, `destroyAllWindows` | §3.1 | boucle de `ui/desktop.py` |

## TP2 — Dessiner avec OpenCV

| Notion du TP | Source du TP | Où dans l'application |
| --- | --- | --- |
| Pixels, origine en haut à gauche, résolution | §1.1 | filtre `grille_reperes`, onglet *Pixels* de l'interface web, sonde de pixel de l'application de bureau |
| Quantification sur 8 bits, 16,7 millions de couleurs | §1.1 | filtre `posterisation` (illustre directement la perte de niveaux) |
| Espace BGR, canaux séparés | §1.2, `Les fondamentales de l'image.py` | filtre `canal_bgr`, avec les deux rendus demandés par le TP (couleur primitive et niveaux de gris) |
| Espace HSV, teinte 0-179, saturation et valeur 0-255 | §1.3 | filtres `canal_hsv` et `ajuster_hsv` ; plages documentées dans [07 §1](07-theorie-filtrage.md#lespace-hsv) |
| `cv2.split` | `Les fondamentales de l'image.py` | `canal_hsv`, `cvlab/histogram.py` |
| **Fonctions enveloppantes** `dess_ligne`, `dess_rectangle`, `dess_cercle`, `dess_ellipse`, `dess_polylignes`, `dess_text` | §2.1 à §2.3, `fonctions/dessin.py` | **`cvlab/drawing.py`** — mêmes noms, mêmes signatures ; une seule différence : elles renvoient une copie au lieu de modifier l'image sur place (indispensable dans une chaîne de traitement) |
| Types de ligne `LINE_4`, `LINE_8`, `LINE_AA`, `FILLED` | §2.1 | exposés en paramètre du filtre `annoter_forme` |
| Polices Hershey | §2.3 | les 8 polices dans `cvlab.drawing.POLICES`, réglables dans `annoter_texte` |
| Dessin d'une icône composite | §2.4, `dessiner_logo.py` | les primitives sont réutilisées par le tracé d'histogramme (`cvlab/histogram.py`), la mire de test et les incrustations de l'application de bureau |

## TP3 — Fonctions d'interaction utilisateur

| Notion du TP | Source du TP | Où dans l'application |
| --- | --- | --- |
| `cv2.setMouseCallback`, événements souris | §1, `La souris.py` | `ui/desktop.py` → `AtelierBureau._souris()` |
| Affichage de (x, y) et de la valeur BGR sous le curseur | `La souris.py` | **sonde de pixel** incrustée de l'application de bureau ; onglet *Pixels* de l'interface web (avec BGR, HSV, hexadécimal et voisinage) |
| Dessiner un cercle à la souris | §2, `Dessiner cercle avec souris.py` | filtre `annoter_forme` (cercle paramétrable) |
| Dessiner un polygone à la souris | §3, `Dessiner un polygone avec souris.py` | sélection des 4 points de perspective au clic, dans l'application de bureau |
| **Recadrer à la souris** par glisser-déposer | §4, `Recadrer une image avec la souris.py` | **glisser à la souris sur le filtre `recadrer`** dans l'application de bureau : les paramètres se mettent à jour |
| `cv2.createTrackbar` / `getTrackbarPos` | §5, `Exemple Trackbar.py` | **toute la fenêtre *Reglages*** de l'application de bureau, avec la classe `EchelleTrackbar` qui gère entiers, réels, impairs, booléens et choix |
| Réglage de la webcam par trackbars (luminosité, contraste, saturation, teinte) | `Entrée utilisateur avec Trackbar.py` | filtres `luminosite_contraste` et `ajuster_hsv` — appliqués en traitement d'image plutôt qu'en réglage de capteur, donc indépendants du pilote et applicables aussi aux fichiers ; `Camera.set_property()` reste disponible pour piloter le capteur |
| Redimensionnement en pourcentage | `redimension()` | filtre `redimensionner` |

## TP4 — Transformations d'image

| Notion du TP | Source du TP | Où dans l'application |
| --- | --- | --- |
| Classe `TraitementImage`, approche objet | §1, `fonctions/TraitementImage.py` | le `FilterDef` du registre joue ce rôle : il encapsule métadonnées, paramètres et traitement |
| `redimention()` en pourcentage, `cv2.resize` | §1 | filtre `redimensionner`, avec le choix de l'interpolation en plus |
| `crop()` par deux points, avec échange si inversés | §1 | filtre `recadrer` — en pourcentage, donc indépendant de la résolution |
| `rotation()`, `getRotationMatrix2D` + `warpAffine` | §1 | filtre `rotation`, avec option de conservation du cadre complet |
| **`fusion()`**, `cv2.addWeighted`, `g = α·f₁ + (1−α)·f₂` | §2, `ImageFusion.py` | filtre `fusion` — la deuxième image est redimensionnée automatiquement, comme dans le TP |
| Réglage de α par trackbar | `ImageFusion.py` | curseur α dans l'interface web, trackbar dans l'application de bureau |
| **Déformation de perspective**, 4 points, `getPerspectiveTransform` + `warpPerspective` | §3, `DéformatiomImage.py` | filtre `deformation_perspective` ; les 4 points se désignent **à la souris** dans l'application de bureau, exactement comme dans le TP |
| Luminosité et contraste (annoncé §2 de l'introduction) | introduction | filtres `luminosite_contraste`, `correction_gamma` |
| Teinte, saturation, valeur (annoncé §3) | introduction | filtre `ajuster_hsv` |
| Opérations bit à bit (annoncé §5) | introduction | filtres `operation_binaire`, `masque_par_reference` |
| Flou gaussien et médian (annoncé §7) | introduction | famille *Lissage & débruitage* (développée au TP7) |
| **Histogramme** (annoncé §8) | introduction | `cvlab/histogram.py`, filtres `egalisation_histogramme`, `clahe`, `etirement_histogramme`, `trace_histogramme`, onglet *Analyse* |

## TP7 — Filtrage, bruit et contours

| Exercice | Source | Où dans l'application |
| --- | --- | --- |
| **ex.1** — filtre moyenneur, noyau construit à la main (`np.ones((3,3))/9` + `filter2D`), comparaison 3×3 à 15×15 | `exercice 1/exo1.py` | filtres `filtre_moyenneur` et `noyau_personnalise` (9 noyaux prédéfinis, normalisation et décalage réglables) ; planche `lissage-taille-noyau.png` |
| **ex.2** — flou gaussien, effet de la taille **et** de σ | `exercice 2/exo2.py` | filtre `filtre_gaussien` ; les deux réglages sont distincts et le **σ effectif** est affiché |
| **ex.3** — bruit sel-poivre, comparaison moyenneur / gaussien / médian | `exercice 3/exo3.py` | filtres `bruit_sel_poivre` et `filtre_median` ; onglet *Comparateur* ; preset `02_debruitage_sel_poivre.json` ; un test automatique vérifie que le médian obtient bien la meilleure MSE |
| **ex.4** — générateurs de bruit (gaussien, sel-poivre, uniforme) et comparaison par **MSE** | `exercice 4/bruit.py`, `exercice 4/exo4.py` | famille *Bruit* (4 générateurs, avec graine reproductible) ; `cvlab/metrics.py` (MSE, RMSE, MAE, PSNR) ; commande `cvlab.cli comparer` ; onglet *Analyse* |
| **ex.5** — Sobel X, Sobel Y, module du gradient, `CV_64F` puis `convertScaleAbs` | `exercice 5/exo5.py` | filtre `sobel` : les trois sorties du TP, plus l'orientation du gradient ; `scharr` en complément |
| **ex.6** — Laplacien, `convertScaleAbs` | `exercice 6/exo6.py` | filtre `laplacien`, avec flou préalable (LoG) et affichage signé en option |
| **ex.7** — Canny, comparaison de trois paires de seuils | `exercice 7/exo7.py` | filtre `canny` : les deux seuils, l'ouverture de Sobel, la norme L2, le flou préalable ; le **rapport haut/bas** et le **pourcentage de pixels de contour** sont affichés ; planche `canny-seuils.png` |
| **exo.py** — webcam → gris → flou → Canny en temps réel | `exo.py` | **`python ui/desktop.py --webcam --filtre canny`**, ou le preset `01_contours_canny.json` chargé dans l'interface web |

## Ce que l'application ajoute aux TP

Sans sortir du périmètre des notions étudiées :

- **l'enchaînement** des filtres, réordonnable et désactivable étape par étape,
  avec affichage pas-à-pas ;
- **la mesure systématique** (MSE, RMSE, MAE, PSNR) et la carte des écarts, là
  où le TP7 ne calculait la MSE que dans un exercice ;
- **la reproductibilité** du bruit par graine, qui rend les comparaisons
  honnêtes ;
- **l'enregistrement des réglages** en presets JSON rejouables ;
- **trois interfaces** sur le même noyau, dont une en temps réel sur la webcam ;
- **la validation systématique des paramètres**, qui rend impossibles les
  erreurs classiques d'OpenCV (noyau pair, seuils inversés, image couleur
  passée à un seuillage).
