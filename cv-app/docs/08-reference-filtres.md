<!--
  FICHIER GÉNÉRÉ — NE PAS MODIFIER À LA MAIN.
  Source : le catalogue de cvlab.registry.
  Régénérer avec :  python outils/generer_doc.py
-->

# Référence des filtres

Fiche de chaque filtre : ce qu'il fait, le principe sous-jacent, et le rôle
exact de chacun de ses paramètres avec ses bornes.

Les bornes indiquées sont celles que l'application impose : un curseur ne peut
pas produire de valeur hors de ces limites, et les tailles de noyau marquées
« impair » n'avancent que de deux en deux. Toute valeur hors bornes fournie par
un preset ou par la ligne de commande est ramenée dans l'intervalle plutôt que
rejetée.

Conventions des colonnes :

- **entrée** — `any` : le filtre accepte couleur et niveaux de gris ;
  `gray` : l'image est convertie en niveaux de gris avant traitement ;
  `color` : l'image est promue en BGR.
- **sortie** — `same` : même nature que l'entrée ; `gray` / `color` : imposée.
- **origine** — le TP dont provient la notion ; `complément` signale un filtre
  ajouté pour compléter une famille, non traité tel quel en TP.


**46 filtres** répartis en 11 familles.

## Sommaire

- [Couleur & espaces colorimétriques](#couleur--espaces-colorimetriques) — 6 filtres
  - [Canal BGR](#canal-bgr) `canal_bgr`
  - [Canal HSV](#canal-hsv) `canal_hsv`
  - [Fausses couleurs (colormap)](#fausses-couleurs-colormap) `fausses_couleurs`
  - [Masque de couleur (HSV)](#masque-de-couleur-hsv) `masque_couleur_hsv`
  - [Niveaux de gris](#niveaux-de-gris) `niveaux_de_gris`
  - [Teinte / saturation / valeur](#teinte--saturation--valeur) `ajuster_hsv`
- [Luminosité & contraste](#luminosite--contraste) — 4 filtres
  - [Correction gamma](#correction-gamma) `correction_gamma`
  - [Luminosité & contraste](#luminosite--contraste) `luminosite_contraste`
  - [Négatif](#negatif) `negatif`
  - [Posterisation (quantification)](#posterisation-quantification) `posterisation`
- [Géométrie](#geometrie) — 6 filtres
  - [Déformation de perspective](#deformation-de-perspective) `deformation_perspective`
  - [Recadrer (crop)](#recadrer-crop) `recadrer`
  - [Redimensionner](#redimensionner) `redimensionner`
  - [Retourner (miroir)](#retourner-miroir) `retourner`
  - [Rotation](#rotation) `rotation`
  - [Translation](#translation) `translation`
- [Bruit](#bruit) — 4 filtres
  - [Bruit gaussien](#bruit-gaussien) `bruit_gaussien`
  - [Bruit multiplicatif (speckle)](#bruit-multiplicatif-speckle) `bruit_multiplicatif`
  - [Bruit sel et poivre](#bruit-sel-et-poivre) `bruit_sel_poivre`
  - [Bruit uniforme](#bruit-uniforme) `bruit_uniforme`
- [Lissage & débruitage](#lissage--debruitage) — 6 filtres
  - [Accentuation (masque flou)](#accentuation-masque-flou) `nettete_unsharp`
  - [Filtre bilatéral](#filtre-bilateral) `filtre_bilateral`
  - [Filtre gaussien](#filtre-gaussien) `filtre_gaussien`
  - [Filtre moyenneur (boîte)](#filtre-moyenneur-boite) `filtre_moyenneur`
  - [Filtre médian](#filtre-median) `filtre_median`
  - [Noyau de convolution personnalisé](#noyau-de-convolution-personnalise) `noyau_personnalise`
- [Contours & gradients](#contours--gradients) — 4 filtres
  - [Détecteur de Canny](#detecteur-de-canny) `canny`
  - [Laplacien](#laplacien) `laplacien`
  - [Scharr](#scharr) `scharr`
  - [Sobel](#sobel) `sobel`
- [Seuillage](#seuillage) — 4 filtres
  - [Seuillage adaptatif](#seuillage-adaptatif) `seuillage_adaptatif`
  - [Seuillage automatique (Otsu / triangle)](#seuillage-automatique-otsu--triangle) `seuillage_automatique`
  - [Seuillage global](#seuillage-global) `seuillage_simple`
  - [Seuillage par bande d'intensité](#seuillage-par-bande-dintensite) `seuillage_par_bande`
- [Morphologie](#morphologie) — 2 filtres
  - [Contour par gradient morphologique](#contour-par-gradient-morphologique) `squelette_contour_morpho`
  - [Opération morphologique](#operation-morphologique) `morphologie`
- [Histogramme](#histogramme) — 4 filtres
  - [CLAHE (égalisation locale)](#clahe-egalisation-locale) `clahe`
  - [Tracé de l'histogramme](#trace-de-lhistogramme) `trace_histogramme`
  - [Égalisation d'histogramme](#egalisation-dhistogramme) `egalisation_histogramme`
  - [Étirement de contraste (normalisation)](#etirement-de-contraste-normalisation) `etirement_histogramme`
- [Fusion & opérations binaires](#fusion--operations-binaires) — 3 filtres
  - [Fusion pondérée](#fusion-ponderee) `fusion`
  - [Masquage par une seconde image](#masquage-par-une-seconde-image) `masque_par_reference`
  - [Opération binaire](#operation-binaire) `operation_binaire`
- [Annotation](#annotation) — 3 filtres
  - [Forme géométrique](#forme-geometrique) `annoter_forme`
  - [Grille de repères](#grille-de-reperes) `grille_reperes`
  - [Texte](#texte) `annoter_texte`


## Couleur & espaces colorimétriques

### Canal BGR

`canal_bgr`

Isole un canal bleu, vert ou rouge.

**Origine** : TP2 §1.2 · **Entrée** : `color` · **Sortie** : `same`

**Principe**

Une image couleur est un empilement de trois plans d'intensité. En mode « couleur primitive » on met à zéro les deux autres plans (l'image reste BGR) ; en mode « niveaux de gris » on affiche le plan seul, ce qui montre que, pris isolément, un canal ne porte aucune couleur (TP2 §1.2).

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `canal` | Canal | choix : `Bleu`, `Vert`, `Rouge` | `Rouge` | Indice du plan à conserver dans le tableau BGR. |
| `mode` | Rendu | choix : `Couleur primitive`, `Niveaux de gris` | `Couleur primitive` | Garder les trois canaux (deux mis à zéro) ou n'afficher que le plan. |

```bash
python -m cvlab.cli decrire canal_bgr
```

### Canal HSV

`canal_hsv`

Affiche le canal teinte, saturation ou valeur.

**Origine** : TP2 §1.3 · **Entrée** : `color` · **Sortie** : `gray`

**Principe**

L'espace HSV décrit une couleur par sa teinte (angle sur la roue chromatique), sa saturation (pureté) et sa valeur (luminosité). OpenCV stocke ces trois canaux sur 8 bits : la teinte est donc divisée par deux et vaut 0-179, tandis que saturation et valeur occupent toute la plage 0-255 (TP2 §1.3).

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `canal` | Canal | choix : `Teinte (H)`, `Saturation (S)`, `Valeur (V)` | `Teinte (H)` | — |
| `etirer` | Étirer le contraste | booléen | False | Ramène le canal sur 0-255. Utile pour la teinte, dont la plage réelle 0-179 rend l'affichage brut très sombre. |

```bash
python -m cvlab.cli decrire canal_hsv
```

### Fausses couleurs (colormap)

`fausses_couleurs`

Applique une palette de couleurs à une image d'intensité.

**Origine** : complément · **Entrée** : `gray` · **Sortie** : `color`

**Principe**

cv2.applyColorMap remplace chaque niveau de gris par une couleur d'une table de 256 entrées. Cela n'ajoute aucune information : c'est un outil de lecture, pratique pour repérer des variations faibles qu'un dégradé de gris masque (cartes de gradient, de profondeur, de chaleur).

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `palette` | Palette | choix : `Viridis`, `Turbo`, `Jet`, `Inferno`, `Magma`, `Plasma`, `Cividis`, `Chaleur (Hot)`, `Os (Bone)`, `Océan`, `Arc-en-ciel` | `Viridis` | — |

```bash
python -m cvlab.cli decrire fausses_couleurs
```

### Masque de couleur (HSV)

`masque_couleur_hsv`

Sélectionne les pixels dont la couleur est dans un intervalle HSV.

**Origine** : TP2 §1.3, complément · **Entrée** : `color` · **Sortie** : `same`

**Principe**

cv2.inRange construit un masque binaire : 255 là où les trois canaux sont dans l'intervalle, 0 ailleurs. On travaille en HSV et non en BGR car la teinte varie peu avec l'éclairage, alors que les trois composantes BGR varient toutes ensemble. Si la teinte minimale est supérieure à la maximale, l'intervalle traverse le rouge (0/179) : deux masques sont alors combinés.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `teinte_min` | Teinte min | entier, 0 à 179 | 0 | — |
| `teinte_max` | Teinte max | entier, 0 à 179 | 179 | — |
| `saturation_min` | Saturation min | entier, 0 à 255 | 60 | — |
| `saturation_max` | Saturation max | entier, 0 à 255 | 255 | — |
| `valeur_min` | Valeur min | entier, 0 à 255 | 60 | — |
| `valeur_max` | Valeur max | entier, 0 à 255 | 255 | — |
| `rendu` | Rendu | choix : `Segmentation`, `Masque binaire`, `Fond en gris` | `Segmentation` | Garder la zone en couleur, afficher le masque, ou désaturer le reste. |

```bash
python -m cvlab.cli decrire masque_couleur_hsv
```

### Niveaux de gris

`niveaux_de_gris`

Convertit l'image couleur en une image à un seul canal.

**Origine** : TP1 §3.2 · **Entrée** : `any` · **Sortie** : `gray`

**Principe**

cv2.cvtColor(..., COLOR_BGR2GRAY) applique la pondération de la luminance perçue : Y = 0.299·R + 0.587·V + 0.114·B. Le vert pèse le plus car l'œil y est le plus sensible ; une simple moyenne des trois canaux donnerait un résultat plus plat.

*Aucun paramètre.*

```bash
python -m cvlab.cli decrire niveaux_de_gris
```

### Teinte / saturation / valeur

`ajuster_hsv`

Décale la teinte et multiplie saturation et valeur.

**Origine** : TP3 §5, TP4 §3 · **Entrée** : `color` · **Sortie** : `color`

**Principe**

On passe en HSV, on modifie chaque canal séparément puis on revient en BGR. Le décalage de teinte est circulaire (modulo 180 chez OpenCV) : au-delà de 179 on revient au rouge. Saturation et valeur sont des gains multiplicatifs, bornés à 255 pour éviter le repliement du type uint8.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `decalage_teinte` | Décalage de teinte | entier, -179 à 179 | 0 | Rotation sur la roue chromatique : change la famille de couleurs. |
| `gain_saturation` | Gain de saturation | réel, 0 à 3, pas 0.05 | 1.0 | 0 = image grise, 1 = inchangé, >1 = couleurs plus vives. |
| `gain_valeur` | Gain de valeur (luminosité) | réel, 0 à 3, pas 0.05 | 1.0 | 0 = image noire, 1 = inchangé, >1 = image plus claire. |

```bash
python -m cvlab.cli decrire ajuster_hsv
```


## Luminosité & contraste

### Correction gamma

`correction_gamma`

Transformation non linéaire : g = 255·(f/255)^(1/γ).

**Origine** : TP4 §2, complément · **Entrée** : `any` · **Sortie** : `same`

**Principe**

Contrairement au réglage linéaire, la correction gamma modifie surtout les tons moyens et préserve le noir et le blanc. γ > 1 éclaircit les ombres, γ < 1 les assombrit. Le calcul se fait par table de correspondance (cv2.LUT) : 256 valeurs calculées une fois, puis un simple accès mémoire par pixel — bien plus rapide qu'une puissance par pixel.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `gamma` | Gamma (γ) | réel, 0.1 à 5, pas 0.05 | 1.0 | 1 = inchangé. >1 éclaircit les tons sombres, <1 les assombrit. |

```bash
python -m cvlab.cli decrire correction_gamma
```

### Luminosité & contraste

`luminosite_contraste`

Transformation linéaire des intensités : g = α·f + β.

**Origine** : TP3 §5, TP4 §2 · **Entrée** : `any` · **Sortie** : `same`

**Principe**

α (gain) multiplie l'écart des intensités, donc le contraste ; β (offset) décale toutes les intensités, donc la luminosité. cv2.convertScaleAbs calcule |α·f + β| puis borne à 255, ce qui évite le repliement du uint8 (sans cela, 250 + 10 donnerait 4).

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `alpha` | Contraste (α) | réel, 0 à 3, pas 0.05 | 1.0 | 1 = inchangé. <1 écrase le contraste, >1 l'accentue. |
| `beta` | Luminosité (β) | entier, -128 à 128 | 0 | Ajouté à chaque pixel après multiplication par α. |

```bash
python -m cvlab.cli decrire luminosite_contraste
```

### Négatif

`negatif`

Inverse les intensités : g = 255 − f.

**Origine** : TP4 §5 · **Entrée** : `any` · **Sortie** : `same`

**Principe**

Opération ponctuelle la plus simple, réalisée par cv2.bitwise_not (complément à un sur 8 bits). Elle rend lisibles des détails situés dans les hautes lumières, d'où son usage courant en imagerie médicale.

*Aucun paramètre.*

```bash
python -m cvlab.cli decrire negatif
```

### Posterisation (quantification)

`posterisation`

Réduit le nombre de niveaux d'intensité par canal.

**Origine** : TP2 §1.1, complément · **Entrée** : `any` · **Sortie** : `same`

**Principe**

On divise la plage 0-255 en N paliers : g = round(f/pas)·pas. C'est une illustration directe de la quantification d'une image numérique (TP2 §1.1) : avec 2 niveaux par canal il ne reste que 8 couleurs possibles au lieu de 16,7 millions.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `niveaux` | Niveaux par canal | entier, 2 à 64 | 4 | 2 niveaux par canal = 8 couleurs ; 256 = image inchangée. |

```bash
python -m cvlab.cli decrire posterisation
```


## Géométrie

### Déformation de perspective

`deformation_perspective`

Redresse un quadrilatère en rectangle (4 points source).

**Origine** : TP4 §3 · **Entrée** : `any` · **Sortie** : `same`

**Principe**

On donne quatre points de l'image d'origine et les quatre coins du rectangle de sortie ; cv2.getPerspectiveTransform résout le système et renvoie l'homographie 3×3, que cv2.warpPerspective applique. C'est la correction utilisée pour redresser la photo d'une page prise de biais (TP4 §3). L'ordre des points compte : haut-gauche, haut-droit, bas-droit, bas-gauche.

> **Note** — Dans l'application de bureau, les quatre points peuvent être désignés directement à la souris (clic gauche).

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `p1x` | 1 · haut-gauche X | réel, 0 à 100, pas 0.5 | 10.0 | — |
| `p1y` | 1 · haut-gauche Y | réel, 0 à 100, pas 0.5 | 10.0 | — |
| `p2x` | 2 · haut-droit X | réel, 0 à 100, pas 0.5 | 90.0 | — |
| `p2y` | 2 · haut-droit Y | réel, 0 à 100, pas 0.5 | 10.0 | — |
| `p3x` | 3 · bas-droit X | réel, 0 à 100, pas 0.5 | 90.0 | — |
| `p3y` | 3 · bas-droit Y | réel, 0 à 100, pas 0.5 | 90.0 | — |
| `p4x` | 4 · bas-gauche X | réel, 0 à 100, pas 0.5 | 10.0 | — |
| `p4y` | 4 · bas-gauche Y | réel, 0 à 100, pas 0.5 | 90.0 | — |
| `largeur_sortie` | Largeur de sortie | entier, 32 à 2048, pas 16 | 480 | — |
| `hauteur_sortie` | Hauteur de sortie | entier, 32 à 2048, pas 16 | 640 | — |
| `interpolation` | Interpolation | choix : `Plus proche voisin`, `Bilinéaire`, `Bicubique`, `Aire (réduction)`, `Lanczos4` | `Bilinéaire` | — |

```bash
python -m cvlab.cli decrire deformation_perspective
```

### Recadrer (crop)

`recadrer`

Extrait une région rectangulaire de l'image.

**Origine** : TP3 §4, TP4 §1 · **Entrée** : `any` · **Sortie** : `same`

**Principe**

Le recadrage est un simple découpage de tableau NumPy, ``image[y1:y2, x1:x2]`` : aucun pixel n'est recalculé. Attention à l'ordre des indices — NumPy indexe en (ligne, colonne), donc (y, x), alors que les fonctions de dessin d'OpenCV prennent des points (x, y).

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `x` | Bord gauche | réel, 0 à 99, pas 0.5 | 10.0 | — |
| `y` | Bord haut | réel, 0 à 99, pas 0.5 | 10.0 | — |
| `largeur` | Largeur | réel, 1 à 100, pas 0.5 | 80.0 | — |
| `hauteur` | Hauteur | réel, 1 à 100, pas 0.5 | 80.0 | — |

```bash
python -m cvlab.cli decrire recadrer
```

### Redimensionner

`redimensionner`

Change la taille de l'image par un pourcentage.

**Origine** : TP4 §1 · **Entrée** : `any` · **Sortie** : `same`

**Principe**

cv2.resize recalcule la grille de pixels. Le choix de l'interpolation compte : « plus proche voisin » recopie le pixel le plus proche (rapide, crénelé, seul choix correct pour une image de labels), « aire » moyenne les pixels source et évite le repliement de spectre en réduction, « bicubique » et « Lanczos » donnent les meilleurs agrandissements.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `pourcentage` | Échelle | réel, 5 à 400, pas 1 | 100.0 | 100 % = taille d'origine. |
| `interpolation` | Interpolation | choix : `Plus proche voisin`, `Bilinéaire`, `Bicubique`, `Aire (réduction)`, `Lanczos4` | `Bilinéaire` | — |

```bash
python -m cvlab.cli decrire redimensionner
```

### Retourner (miroir)

`retourner`

Symétrie horizontale, verticale ou les deux.

**Origine** : complément · **Entrée** : `any` · **Sortie** : `same`

**Principe**

cv2.flip(image, code) : code = 1 inverse les colonnes (effet miroir), 0 inverse les lignes, −1 fait les deux (équivalent à une rotation de 180°). C'est l'opération à appliquer à une image de webcam pour obtenir un rendu « miroir » naturel.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `sens` | Sens | choix : `Horizontal`, `Vertical`, `Les deux` | `Horizontal` | — |

```bash
python -m cvlab.cli decrire retourner
```

### Rotation

`rotation`

Fait pivoter l'image autour de son centre, avec mise à l'échelle.

**Origine** : TP4 §1 · **Entrée** : `any` · **Sortie** : `same`

**Principe**

cv2.getRotationMatrix2D construit la matrice affine 2×3 [[s·cosθ, s·sinθ, tx], [−s·sinθ, s·cosθ, ty]], que cv2.warpAffine applique à chaque pixel. Avec la taille d'origine, les coins sortent du cadre et sont perdus ; l'option « conserver tout le contenu » agrandit le cadre et corrige la translation pour que rien ne soit coupé.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `angle` | Angle | réel, -180 à 180, pas 1 | 0.0 | Positif = sens antihoraire (convention OpenCV). |
| `echelle` | Échelle | réel, 0.1 à 3, pas 0.05 | 1.0 | — |
| `conserver_cadre` | Conserver tout le contenu | booléen | False | Agrandit le cadre pour qu'aucun coin ne soit coupé. |
| `bord` | Traitement des bords | choix : `Réplication`, `Constante (noir)`, `Miroir`, `Enroulement` | `Constante (noir)` | — |

```bash
python -m cvlab.cli decrire rotation
```

### Translation

`translation`

Décale l'image horizontalement et verticalement.

**Origine** : TP4 §1, complément · **Entrée** : `any` · **Sortie** : `same`

**Principe**

Transformation affine la plus simple : matrice [[1, 0, tx], [0, 1, ty]] passée à cv2.warpAffine. Le traitement des bords décide de ce qui remplit la zone découverte.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `dx` | Décalage horizontal | réel, -100 à 100, pas 1 | 0.0 | — |
| `dy` | Décalage vertical | réel, -100 à 100, pas 1 | 0.0 | — |
| `bord` | Traitement des bords | choix : `Réplication`, `Constante (noir)`, `Miroir`, `Enroulement` | `Constante (noir)` | — |

```bash
python -m cvlab.cli decrire translation
```


## Bruit

### Bruit gaussien

`bruit_gaussien`

Ajoute un bruit additif suivant une loi normale.

**Origine** : TP7 ex.4 · **Entrée** : `any` · **Sortie** : `same`

**Principe**

g(x, y) = f(x, y) + n(x, y) avec n ~ N(moyenne, σ²). C'est le modèle du bruit électronique d'un capteur : il affecte tous les pixels, faiblement et symétriquement. σ contrôle l'amplitude ; la moyenne décale en plus la luminosité. Le calcul passe par des flottants puis np.clip, car additionner directement sur du uint8 provoquerait un repliement (255 + 3 = 2).

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `moyenne` | Moyenne | réel, -50 à 50, pas 1 | 0.0 | 0 = bruit centré, sans effet sur la luminosité moyenne. |
| `sigma` | Écart-type (σ) | réel, 0 à 100, pas 1 | 25.0 | Amplitude du bruit. 25 est déjà bien visible. |
| `graine` | Graine aléatoire | entier, 0 à 9999 | 0 | 0 = tirage différent à chaque calcul ; sinon reproductible. |

```bash
python -m cvlab.cli decrire bruit_gaussien
```

### Bruit multiplicatif (speckle)

`bruit_multiplicatif`

Multiplie chaque pixel par un facteur aléatoire.

**Origine** : complément · **Entrée** : `any` · **Sortie** : `same`

**Principe**

g = f · (1 + n) avec n ~ N(0, σ²). Le bruit est proportionnel au signal : les zones claires sont plus perturbées que les zones sombres. C'est le modèle du bruit de speckle des images radar et échographiques.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `sigma` | Écart-type relatif (σ) | réel, 0 à 1, pas 0.02 | 0.2 | 0.2 = ±20 % d'amplitude typique. |
| `graine` | Graine aléatoire | entier, 0 à 9999 | 0 | — |

```bash
python -m cvlab.cli decrire bruit_multiplicatif
```

### Bruit sel et poivre

`bruit_sel_poivre`

Force une fraction des pixels à 0 (poivre) ou 255 (sel).

**Origine** : TP7 ex.3, TP7 ex.4 · **Entrée** : `any` · **Sortie** : `same`

**Principe**

Bruit impulsionnel : quelques pixels sont totalement faux, les autres intacts. Modèle des pixels morts d'un capteur ou des erreurs de transmission. C'est le cas d'école du filtre médian : la moyenne est entraînée par les valeurs extrêmes, la médiane les ignore.

> **Note** — Le TP tirait les coordonnées dans une boucle Python ; ici le tirage est vectorisé (un masque NumPy), ce qui est équivalent mais des milliers de fois plus rapide — indispensable en temps réel.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `proportion` | Proportion de pixels touchés | réel, 0 à 0.5, pas 0.005 | 0.05 | 0.05 = 5 % des pixels sont remplacés. |
| `ratio_sel` | Part de sel (blanc) | réel, 0 à 1, pas 0.05 | 0.5 | 0 = uniquement du poivre (noir), 1 = uniquement du sel. |
| `graine` | Graine aléatoire | entier, 0 à 9999 | 0 | — |

```bash
python -m cvlab.cli decrire bruit_sel_poivre
```

### Bruit uniforme

`bruit_uniforme`

Ajoute un bruit additif tiré uniformément dans un intervalle.

**Origine** : TP7 ex.4 · **Entrée** : `any` · **Sortie** : `same`

**Principe**

n ~ U(min, max) : toutes les amplitudes de l'intervalle sont également probables, contrairement au bruit gaussien qui concentre ses valeurs près de la moyenne. Un intervalle asymétrique décale en plus la luminosité.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `minimum` | Borne inférieure | réel, -128 à 0, pas 1 | -30.0 | — |
| `maximum` | Borne supérieure | réel, 0 à 128, pas 1 | 30.0 | — |
| `graine` | Graine aléatoire | entier, 0 à 9999 | 0 | — |

```bash
python -m cvlab.cli decrire bruit_uniforme
```


## Lissage & débruitage

### Accentuation (masque flou)

`nettete_unsharp`

Renforce les détails en soustrayant une version floue de l'image.

**Origine** : complément · **Entrée** : `any` · **Sortie** : `same`

**Principe**

g = f + k·(f − flou(f)). La différence f − flou(f) isole les hautes fréquences, c'est-à-dire les détails et les contours ; on les réinjecte amplifiés. Le rayon du flou fixe l'échelle des détails accentués, k leur intensité. Trop de k crée des halos clairs autour des contours et amplifie le bruit — le masque flou est un filtre passe-haut, et le bruit est une haute fréquence.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `taille_noyau` | Rayon du flou (noyau) | entier, 3 à 31, impair | 5 | — |
| `sigma` | σ du flou | réel, 0 à 20, pas 0.1 | 1.0 | 0 = déduit de la taille du noyau. |
| `intensite` | Intensité (k) | réel, 0 à 3, pas 0.05 | 1.0 | — |
| `seuil` | Seuil de protection | entier, 0 à 64 | 0 | Les zones dont l'écart au flou est inférieur au seuil ne sont pas accentuées : évite d'amplifier le bruit des aplats. |

```bash
python -m cvlab.cli decrire nettete_unsharp
```

### Filtre bilatéral

`filtre_bilateral`

Lisse les zones homogènes en préservant les contours.

**Origine** : TP7 ex.4 · **Entrée** : `any` · **Sortie** : `same` · **Coûteux** (éviter en temps réel)

**Principe**

Deux gaussiennes multipliées : une sur la **distance spatiale** (σ_espace) comme un flou gaussien ordinaire, et une sur la **différence d'intensité** (σ_couleur). Un voisin très différent du pixel central reçoit un poids quasi nul, donc le filtre ne mélange pas les deux côtés d'un contour. Conséquences pratiques : σ_couleur élevé (> 150) rapproche le résultat d'un flou gaussien ; d = 0 fait déduire le diamètre de σ_espace ; et le coût est bien supérieur à celui d'un gaussien, car le noyau doit être recalculé pour chaque pixel.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `diametre` | Diamètre du voisinage (d) | entier, 0 à 25 | 9 | 0 = déduit de σ_espace. Au-delà de 9, c'est très lent. |
| `sigma_couleur` | σ couleur | réel, 1 à 250, pas 1 | 75.0 | Tolérance sur l'écart d'intensité : grand = contours moins protégés. |
| `sigma_espace` | σ espace | réel, 1 à 250, pas 1 | 75.0 | Portée géométrique du filtre, en pixels. |

```bash
python -m cvlab.cli decrire filtre_bilateral
```

### Filtre gaussien

`filtre_gaussien`

Moyenne pondérée par une cloche gaussienne.

**Origine** : TP7 ex.2 · **Entrée** : `any` · **Sortie** : `same`

**Principe**

Le noyau vaut G(i, j) = exp(−(i² + j²) / 2σ²), normalisé. Les pixels proches du centre pèsent plus que les pixels éloignés : le lissage est plus doux et plus naturel que celui du moyenneur, sans les artefacts rectangulaires de ce dernier. Deux réglages interviennent : la **taille du noyau** (étendue du voisinage) et **σ** (largeur de la cloche). Avec σ = 0, OpenCV le déduit de la taille (σ ≈ 0,3·((n−1)/2 − 1) + 0,8) ; c'est ce que faisait le TP7 ex.2 dans sa première version. Si σ est grand par rapport au noyau, la cloche est tronquée et le filtre se rapproche d'un moyenneur.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `taille_noyau` | Taille du noyau | entier, 1 à 31, impair | 5 | — |
| `sigma_x` | σ horizontal | réel, 0 à 20, pas 0.1 | 0.0 | 0 = calculé automatiquement depuis la taille du noyau. |
| `sigma_y` | σ vertical | réel, 0 à 20, pas 0.1 | 0.0 | 0 = identique à σ horizontal. |
| `bord` | Traitement des bords | choix : `Miroir (défaut)`, `Miroir avec duplication`, `Réplication`, `Constante (noir)` | `Miroir (défaut)` | — |

```bash
python -m cvlab.cli decrire filtre_gaussien
```

### Filtre moyenneur (boîte)

`filtre_moyenneur`

Remplace chaque pixel par la moyenne de son voisinage.

**Origine** : TP7 ex.1 · **Entrée** : `any` · **Sortie** : `same`

**Principe**

Noyau de coefficients tous égaux à 1/(l·h) — pour 3×3, neuf fois 1/9. C'est le filtre passe-bas le plus simple. Plus le noyau est grand, plus le bruit disparaît, mais plus l'image devient floue : le noyau ne distingue pas un contour d'une fluctuation de bruit. Un noyau rectangulaire (largeur ≠ hauteur) floute davantage dans une seule direction, ce qui se voit comme un effet de filé.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `largeur_noyau` | Largeur du noyau | entier, 1 à 31, impair | 5 | — |
| `hauteur_noyau` | Hauteur du noyau | entier, 1 à 31, impair | 5 | — |
| `bord` | Traitement des bords | choix : `Miroir (défaut)`, `Miroir avec duplication`, `Réplication`, `Constante (noir)` | `Miroir (défaut)` | Comment compléter le voisinage au bord de l'image. |

```bash
python -m cvlab.cli decrire filtre_moyenneur
```

### Filtre médian

`filtre_median`

Remplace chaque pixel par la médiane de son voisinage.

**Origine** : TP7 ex.3, TP7 ex.4 · **Entrée** : `any` · **Sortie** : `same`

**Principe**

Filtre de rang, non linéaire : les n² valeurs du voisinage sont triées et c'est la valeur centrale qui est retenue. Un pixel sel (255) ou poivre (0) se retrouve en bout de tri et n'influence donc pas le résultat — d'où son efficacité remarquable sur le bruit impulsionnel, là où moyenneur et gaussien étalent la tache. La valeur sortie existe toujours dans l'image d'origine, ce qui préserve mieux les contours francs.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `taille_noyau` | Taille du noyau | entier, 3 à 31, impair | 5 | Doit être impair et ≥ 3. Au-delà de 5, OpenCV exige du 8 bits. |

```bash
python -m cvlab.cli decrire filtre_median
```

### Noyau de convolution personnalisé

`noyau_personnalise`

Applique un noyau choisi via cv2.filter2D.

**Origine** : TP7 ex.1 · **Entrée** : `any` · **Sortie** : `same`

**Principe**

cv2.filter2D effectue la corrélation du noyau avec l'image (une convolution au retournement du noyau près, sans conséquence pour les noyaux symétriques). C'est l'outil qui permet de construire un filtre « à la main », comme dans la première version du TP7 ex.1 (``np.ones((3, 3)) / 9``). Un noyau dont la somme vaut 1 préserve la luminosité moyenne ; une somme nulle (détecteurs de contours) donne une image presque noire à laquelle on ajoute un décalage pour la rendre lisible.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `modele` | Noyau | choix : `Identité`, `Moyenneur`, `Gaussien approché`, `Netteté (sharpen)`, `Laplacien 4-voisins`, `Laplacien 8-voisins`, `Relief (emboss)`, `Sobel horizontal`, `Sobel vertical` | `Moyenneur` | — |
| `taille` | Taille (noyaux redimensionnables) | entier, 3 à 15, impair | 3 | Ne s'applique qu'aux noyaux moyenneur et gaussien approché. |
| `normaliser` | Normaliser le noyau | booléen | True | Divise par la somme des coefficients (préserve la luminosité). |
| `decalage` | Décalage ajouté (delta) | entier, -128 à 128 | 0 | Utile pour visualiser un noyau de somme nulle (essayez 128). |

```bash
python -m cvlab.cli decrire noyau_personnalise
```


## Contours & gradients

### Détecteur de Canny

`canny`

Contours fins et binaires par double seuillage à hystérésis.

**Origine** : TP7 ex.7, TP7 exo.py · **Entrée** : `gray` · **Sortie** : `gray`

**Principe**

Quatre étapes : (1) lissage gaussien ; (2) gradient de Sobel ; (3) amincissement — on ne garde un pixel que s'il est maximal dans la direction du gradient ; (4) hystérésis — tout pixel de module > seuil haut est un contour, tout pixel < seuil bas est rejeté, et entre les deux un pixel n'est conservé que s'il touche un contour déjà accepté. C'est cette dernière étape qui donne des contours continus là où un seuil unique produirait des pointillés.
Réglage : un rapport seuil_haut/seuil_bas de 2 à 3 est la règle usuelle. Monter les deux seuils ne garde que les contours francs ; les baisser fait apparaître du détail et du bruit (TP7 ex.7).

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `seuil_bas` | Seuil bas | entier, 0 à 500 | 50 | Sous ce module de gradient, le pixel est rejeté. |
| `seuil_haut` | Seuil haut | entier, 0 à 500 | 150 | Au-dessus, le pixel est un contour à coup sûr. |
| `ouverture` | Ouverture de Sobel | entier, 3 à 7, impair | 3 | Taille du noyau de gradient interne : 3, 5 ou 7. |
| `norme_l2` | Norme L2 exacte | booléen | False | √(Gx²+Gy²) au lieu de \|Gx\|+\|Gy\| : un peu plus précis, un peu plus lent. |
| `flou_prealable` | Flou gaussien préalable | entier, 0 à 15 | 5 | Canny lisse déjà en interne ; un flou supplémentaire réduit le bruit résiduel. |

```bash
python -m cvlab.cli decrire canny
```

### Laplacien

`laplacien`

Dérivée seconde : ∂²f/∂x² + ∂²f/∂y².

**Origine** : TP7 ex.6 · **Entrée** : `gray` · **Sortie** : `gray`

**Principe**

Le noyau 3×3 [[0,1,0],[1,−4,1],[0,1,0]] approche le laplacien. Contrairement à Sobel, il n'est pas directionnel : une seule convolution suffit pour détecter les contours dans toutes les orientations. En contrepartie il est très sensible au bruit (deux dérivations) et ne donne pas de sens de transition. L'usage classique est donc le Laplacien du gaussien (LoG) : flouter, puis appliquer le laplacien — ce que fait ici l'option de flou préalable.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `taille_noyau` | Taille du noyau (ksize) | entier, 1 à 31, impair | 3 | 1 utilise le noyau 3×3 classique ; au-delà, un noyau de Sobel. |
| `echelle` | Facteur d'échelle | réel, 0.1 à 10, pas 0.1 | 1.0 | — |
| `decalage` | Décalage (delta) | entier, -128 à 128 | 0 | — |
| `flou_prealable` | Flou gaussien préalable (LoG) | entier, 0 à 15 | 3 | Fortement recommandé : le laplacien amplifie le bruit. |
| `garder_signe` | Afficher le signe (gris = zéro) | booléen | False | Centre la sortie sur 128 au lieu de prendre la valeur absolue : on distingue alors les deux côtés du contour. |

```bash
python -m cvlab.cli decrire laplacien
```

### Scharr

`scharr`

Variante 3×3 de Sobel, plus précise en orientation.

**Origine** : complément · **Entrée** : `gray` · **Sortie** : `gray`

**Principe**

Noyau [[−3,0,3],[−10,0,10],[−3,0,3]] : les coefficients sont optimisés pour que la réponse soit la plus isotrope possible, c'est-à-dire que l'orientation estimée du gradient soit juste quel que que soit l'angle du contour. À taille 3×3, Scharr est strictement préférable à Sobel ; en revanche il n'existe qu'en 3×3.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `direction` | Composante | choix : `Horizontal (∂/∂x)`, `Vertical (∂/∂y)`, `Module du gradient`, `Orientation du gradient` | `Module du gradient` | — |
| `echelle` | Facteur d'échelle | réel, 0.1 à 10, pas 0.1 | 1.0 | — |
| `flou_prealable` | Flou gaussien préalable | entier, 0 à 15 | 0 | — |

```bash
python -m cvlab.cli decrire scharr
```

### Sobel

`sobel`

Dérivée première de l'image, horizontale, verticale ou son module.

**Origine** : TP7 ex.5 · **Entrée** : `gray` · **Sortie** : `gray`

**Principe**

Noyau horizontal [[−1,0,1],[−2,0,2],[−1,0,1]] : une différence centrée en x, combinée à un lissage en y par les coefficients (1, 2, 1). Sobel est donc à la fois un dérivateur et un lisseur, ce qui le rend plus robuste au bruit qu'une simple différence de pixels. Le module √(Gx² + Gy²) est indépendant de l'orientation du contour ; Gx seul ne voit que les transitions verticales, Gy seul que les horizontales. ksize = 1 utilise un noyau 1×3 sans lissage.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `direction` | Composante | choix : `Horizontal (∂/∂x)`, `Vertical (∂/∂y)`, `Module du gradient`, `Orientation du gradient` | `Module du gradient` | — |
| `taille_noyau` | Taille du noyau (ksize) | entier, 1 à 7, impair | 3 | 1, 3, 5 ou 7. Plus grand = contours plus épais et plus lissés. |
| `echelle` | Facteur d'échelle | réel, 0.1 à 10, pas 0.1 | 1.0 | Multiplie la dérivée avant conversion en 8 bits. |
| `decalage` | Décalage (delta) | entier, -128 à 128 | 0 | — |
| `flou_prealable` | Flou gaussien préalable | entier, 0 à 15 | 0 | 0 = aucun. Sinon taille du noyau gaussien appliqué avant. |

```bash
python -m cvlab.cli decrire sobel
```


## Seuillage

### Seuillage adaptatif

`seuillage_adaptatif`

Un seuil local par voisinage, robuste à l'éclairage inégal.

**Origine** : complément · **Entrée** : `gray` · **Sortie** : `gray`

**Principe**

Le seuil du pixel (x, y) est calculé sur son voisinage de taille ``taille_bloc`` : moyenne arithmétique (MEAN_C) ou moyenne pondérée par une gaussienne (GAUSSIAN_C), moins une constante C. Comparer un pixel à la moyenne de ses voisins revient à détecter s'il est plus sombre que son entourage immédiat — d'où l'excellent résultat sur du texte photographié avec une ombre.
Réglage : ``taille_bloc`` doit être plus grand que les détails à isoler (sinon l'intérieur des traits épais se vide) ; C élimine le bruit des zones uniformes, où la moyenne locale est presque égale au pixel lui-même.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `methode` | Méthode de moyenne | choix : `Moyenne`, `Gaussienne` | `Gaussienne` | — |
| `type_seuil` | Type | choix : `Binaire`, `Binaire inversé` | `Binaire` | — |
| `taille_bloc` | Taille du voisinage | entier, 3 à 99, impair | 11 | Impair et ≥ 3. Doit excéder la taille des détails à isoler. |
| `constante` | Constante soustraite (C) | entier, -30 à 30 | 2 | Augmenter C nettoie les zones uniformes, mais efface les détails ténus. |
| `valeur_max` | Valeur maximale | entier, 1 à 255 | 255 | — |

```bash
python -m cvlab.cli decrire seuillage_adaptatif
```

### Seuillage automatique (Otsu / triangle)

`seuillage_automatique`

Calcule le seuil optimal depuis l'histogramme.

**Origine** : complément · **Entrée** : `gray` · **Sortie** : `gray`

**Principe**

La méthode d'Otsu essaie tous les seuils possibles et retient celui qui minimise la variance intra-classe (de façon équivalente : qui maximise la variance inter-classe). Elle suppose un histogramme **bimodal** — une bosse pour le fond, une pour les objets ; sur une image à éclairage inégal, le seuil trouvé est mauvais et c'est le seuillage adaptatif qu'il faut utiliser. La méthode du triangle convient mieux quand une seule classe domine largement l'image.
Un lissage gaussien préalable est recommandé : il resserre les deux bosses de l'histogramme et stabilise le seuil trouvé.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `methode` | Méthode | choix : `Otsu`, `Triangle` | `Otsu` | — |
| `type_seuil` | Type | choix : `Binaire`, `Binaire inversé` | `Binaire` | — |
| `flou_prealable` | Flou gaussien préalable | entier, 0 à 15 | 5 | 0 = aucun. Un léger flou rend le seuil trouvé plus stable. |

```bash
python -m cvlab.cli decrire seuillage_automatique
```

### Seuillage global

`seuillage_simple`

Compare chaque pixel à un seuil fixé par l'utilisateur.

**Origine** : TP7, complément · **Entrée** : `gray` · **Sortie** : `gray`

**Principe**

cv2.threshold applique la règle choisie :
• Binaire        : f > s ? maxval : 0
• Binaire inversé : f > s ? 0 : maxval
• Troncature     : f > s ? s : f (plafonnement)
• Vers zéro      : f > s ? f : 0
Les deux premières donnent une image binaire ; les autres gardent une partie de l'information d'intensité, ce qui sert à préparer un autre traitement plutôt qu'à segmenter.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `seuil` | Seuil | entier, 0 à 255 | 127 | — |
| `valeur_max` | Valeur maximale | entier, 1 à 255 | 255 | Valeur attribuée aux pixels retenus (255 = blanc). |
| `type_seuil` | Type | choix : `Binaire`, `Binaire inversé`, `Troncature`, `Vers zéro`, `Vers zéro inversé` | `Binaire` | — |

```bash
python -m cvlab.cli decrire seuillage_simple
```

### Seuillage par bande d'intensité

`seuillage_par_bande`

Garde les pixels dont l'intensité est dans un intervalle.

**Origine** : complément · **Entrée** : `gray` · **Sortie** : `gray`

**Principe**

cv2.inRange sur un seul canal : le masque vaut 255 si ``min ≤ f ≤ max``. Utile quand l'objet n'est ni le plus clair ni le plus sombre de l'image — cas qu'un seuil unique ne peut pas traiter.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `minimum` | Intensité minimale | entier, 0 à 255 | 80 | — |
| `maximum` | Intensité maximale | entier, 0 à 255 | 200 | — |
| `inverser` | Inverser le masque | booléen | False | — |

```bash
python -m cvlab.cli decrire seuillage_par_bande
```


## Morphologie

### Contour par gradient morphologique

`squelette_contour_morpho`

Trace le bord des régions claires (dilatation − érosion).

**Origine** : complément · **Entrée** : `any` · **Sortie** : `same`

**Principe**

Avec un élément 3×3, la dilatation moins l'érosion ne laisse qu'un liseré d'un pixel de part et d'autre de chaque frontière. Comparé à Canny, ce détecteur n'a pas de seuil à régler et donne des contours fermés, mais il est beaucoup plus sensible au bruit : on l'applique donc en général sur une image déjà seuillée.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `forme` | Forme de l'élément | choix : `Rectangle`, `Ellipse`, `Croix` | `Rectangle` | — |
| `taille` | Épaisseur du contour | entier, 3 à 15, impair | 3 | Taille de l'élément structurant : fixe l'épaisseur du liseré. |

```bash
python -m cvlab.cli decrire squelette_contour_morpho
```

### Opération morphologique

`morphologie`

Érosion, dilatation, ouverture, fermeture, gradient, chapeaux.

**Origine** : complément · **Entrée** : `any` · **Sortie** : `same`

**Principe**

cv2.morphologyEx applique l'opération choisie avec l'élément structurant fourni par cv2.getStructuringElement. La **forme** de cet élément compte : un rectangle produit des coins carrés, une ellipse respecte mieux les objets ronds, une croix ne touche que les 4 voisins directs. La **taille** fixe l'échelle des détails affectés — c'est le réglage le plus déterminant. Les **itérations** répètent l'opération : n itérations d'un élément 3×3 équivalent approximativement à un élément (2n+1)×(2n+1), en plus rapide.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `operation` | Opération | choix : `Érosion`, `Dilatation`, `Ouverture`, `Fermeture`, `Gradient morphologique`, `Chapeau haut de forme (top-hat)`, `Chapeau noir (black-hat)` | `Ouverture` | — |
| `forme` | Forme de l'élément structurant | choix : `Rectangle`, `Ellipse`, `Croix` | `Ellipse` | — |
| `taille` | Taille de l'élément | entier, 1 à 31, impair | 5 | — |
| `iterations` | Itérations | entier, 1 à 10 | 1 | — |

```bash
python -m cvlab.cli decrire morphologie
```


## Histogramme

### CLAHE (égalisation locale)

`clahe`

Égalisation par tuiles, avec limitation du contraste.

**Origine** : TP4 §8, complément · **Entrée** : `any` · **Sortie** : `same`

**Principe**

CLAHE découpe l'image en tuiles et égalise chaque tuile séparément, puis interpole entre les tuiles pour éviter les discontinuités. Le paramètre ``limite`` écrête l'histogramme avant égalisation et redistribue l'excédent : c'est ce qui empêche l'amplification explosive du bruit dans les zones uniformes, défaut principal de l'égalisation globale.
Réglage : une grille 8×8 convient en général ; plus de tuiles = correction plus locale mais rendu moins naturel. Une limite de 2 à 4 est un bon compromis, au-delà on retrouve le bruit.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `limite` | Limite de contraste | réel, 0.5 à 10, pas 0.1 | 2.0 | Plus la limite est haute, plus le contraste (et le bruit) augmente. |
| `tuiles` | Tuiles par côté | entier, 1 à 16 | 8 | Grille de découpage : 8 signifie une grille 8×8. |
| `espace` | Canal traité | choix : `Luminance Y (YCrCb)`, `Luminance L (Lab)`, `Valeur V (HSV)`, `Trois canaux BGR (déforme les couleurs)` | `Luminance L (Lab)` | — |

```bash
python -m cvlab.cli decrire clahe
```

### Tracé de l'histogramme

`trace_histogramme`

Remplace l'image par le graphique de son histogramme.

**Origine** : TP4 §8 · **Entrée** : `any` · **Sortie** : `color`

**Principe**

Ce n'est pas un filtre au sens strict : la sortie est un graphique, pas une image traitée. Il est fourni comme étape de chaîne pour pouvoir comparer visuellement l'histogramme avant et après un traitement — par exemple pour voir l'effet d'une égalisation sur la courbe cumulée.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `mode` | Mode | choix : `Histogramme`, `Cumulé` | `Histogramme` | — |
| `echelle` | Échelle verticale | choix : `Linéaire`, `Logarithmique` | `Linéaire` | — |
| `largeur` | Largeur du graphique | entier, 256 à 1600, pas 32 | 640 | — |
| `hauteur` | Hauteur du graphique | entier, 160 à 1200, pas 20 | 400 | — |

```bash
python -m cvlab.cli decrire trace_histogramme
```

### Égalisation d'histogramme

`egalisation_histogramme`

Redistribue les intensités pour occuper toute la plage 0-255.

**Origine** : TP4 §8 · **Entrée** : `any` · **Sortie** : `same`

**Principe**

On calcule l'histogramme cumulé normalisé — la fonction de répartition — et on s'en sert comme table de correspondance : g = 255 · F(f). L'histogramme résultant est aussi plat que possible, ce qui maximise le contraste global. Deux limites : l'opération est **globale**, donc une petite zone mal exposée ne sera pas corrigée si le reste de l'image est correct ; et elle amplifie le bruit des zones uniformes. Le CLAHE répond à ces deux critiques.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `espace` | Canal égalisé | choix : `Luminance Y (YCrCb)`, `Luminance L (Lab)`, `Valeur V (HSV)`, `Trois canaux BGR (déforme les couleurs)` | `Luminance Y (YCrCb)` | Sans effet sur une image déjà en niveaux de gris. |

```bash
python -m cvlab.cli decrire egalisation_histogramme
```

### Étirement de contraste (normalisation)

`etirement_histogramme`

Étire linéairement la plage des intensités sur 0-255.

**Origine** : TP4 §8, complément · **Entrée** : `any` · **Sortie** : `same`

**Principe**

g = 255 · (f − min) / (max − min). Contrairement à l'égalisation, cette transformation est **linéaire** : la forme de l'histogramme est conservée, seulement dilatée. Le rendu est donc plus naturel, mais la correction est moins énergique — et un seul pixel aberrant à 0 ou 255 suffit à l'annuler. D'où l'option de percentiles : on ignore les p % de pixels extrêmes avant de calculer min et max.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `percentile_bas` | Percentile bas ignoré | réel, 0 à 20, pas 0.5 | 1.0 | — |
| `percentile_haut` | Percentile haut ignoré | réel, 0 à 20, pas 0.5 | 1.0 | — |

```bash
python -m cvlab.cli decrire etirement_histogramme
```


## Fusion & opérations binaires

### Fusion pondérée

`fusion`

Mélange deux images : g = α·f₁ + (1−α)·f₂ + γ.

**Origine** : TP4 §2 · **Entrée** : `any` · **Sortie** : `same` · **Deuxième image requise**

**Principe**

cv2.addWeighted réalise exactement la formule du TP4 §2. α est le poids de l'image courante : α = 1 ne garde qu'elle, α = 0 ne garde que l'image de référence, α = 0,5 donne une surimpression équilibrée. La somme des poids vaut 1, donc la luminosité moyenne est préservée ; γ permet d'ajouter un décalage constant si le résultat est trop sombre.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `alpha` | Poids de l'image courante (α) | réel, 0 à 1, pas 0.01 | 0.5 | 1 = image courante seule, 0 = image de référence seule. |
| `gamma` | Décalage (γ) | entier, -128 à 128 | 0 | — |

```bash
python -m cvlab.cli decrire fusion
```

### Masquage par une seconde image

`masque_par_reference`

Garde l'image courante là où la référence est claire.

**Origine** : TP4 §5 · **Entrée** : `any` · **Sortie** : `same` · **Deuxième image requise**

**Principe**

La deuxième image est convertie en niveaux de gris puis seuillée pour produire un masque binaire ; ``cv2.bitwise_and(..., mask=m)`` ne conserve l'image courante que là où le masque est non nul. C'est la brique de base de l'incrustation : on combine ensuite la zone gardée avec un autre fond (TP4 §5).

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `seuil` | Seuil du masque | entier, 0 à 255 | 127 | La référence est binarisée à ce seuil pour servir de masque. |
| `sens` | Zone conservée | choix : `Zones claires`, `Zones sombres` | `Zones claires` | — |

```bash
python -m cvlab.cli decrire masque_par_reference
```

### Opération binaire

`operation_binaire`

ET, OU, OU exclusif ou NON, bit à bit, entre deux images.

**Origine** : TP4 §5 · **Entrée** : `any` · **Sortie** : `same` · **Deuxième image requise**

**Principe**

Les opérations s'appliquent bit à bit sur les octets des deux images :
• ET  : ne garde que ce qui est clair dans les deux (intersection) ;
• OU  : garde ce qui est clair dans au moins une (union) ;
• OU exclusif : met en évidence les différences, et vaut 0 là où les deux images sont identiques ;
• NON : complément de l'image courante, la référence est ignorée.
Elles servent surtout avec une image binaire en guise de masque : ``bitwise_and(image, image, mask=m)`` est la façon canonique de ne garder qu'une région (cf. le filtre « Masque de couleur »).

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `operation` | Opération | choix : `ET (AND)`, `OU (OR)`, `OU exclusif (XOR)`, `NON (NOT, image courante)`, `Différence absolue` | `ET (AND)` | — |

```bash
python -m cvlab.cli decrire operation_binaire
```


## Annotation

### Forme géométrique

`annoter_forme`

Trace une ligne, un rectangle, un cercle ou une ellipse.

**Origine** : TP2 §2, TP3 §2-3 · **Entrée** : `any` · **Sortie** : `same`

**Principe**

Les primitives cv2.line, cv2.rectangle, cv2.circle et cv2.ellipse partagent la même signature de fin : couleur, épaisseur, type de ligne. Une épaisseur de −1 (``cv2.FILLED``) remplit la forme. Le type de ligne ``LINE_AA`` active l'anticrénelage : les bords obliques sont lissés, ce qui est bien plus lisible à l'écran que le tracé 8-connexe (TP2 §2.1).

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `forme` | Forme | choix : `Ligne`, `Rectangle`, `Cercle`, `Ellipse` | `Rectangle` | — |
| `x1` | Point 1 · X | réel, 0 à 100, pas 0.5 | 20.0 | — |
| `y1` | Point 1 · Y | réel, 0 à 100, pas 0.5 | 20.0 | — |
| `x2` | Point 2 · X | réel, 0 à 100, pas 0.5 | 80.0 | Pour un cercle ou une ellipse : définit les rayons depuis le point 1. |
| `y2` | Point 2 · Y | réel, 0 à 100, pas 0.5 | 80.0 | — |
| `angle` | Angle (ellipse) | réel, -180 à 180, pas 1 | 0.0 | — |
| `arc_debut` | Début d'arc | réel, 0 à 360, pas 5 | 0.0 | — |
| `arc_fin` | Fin d'arc | réel, 0 à 360, pas 5 | 360.0 | — |
| `couleur` | Couleur | couleur BGR | (0, 0, 255) | — |
| `epaisseur` | Épaisseur | entier, 1 à 40 | 2 | — |
| `rempli` | Forme pleine | booléen | False | — |
| `type_ligne` | Type de ligne | choix : `4-connexe`, `8-connexe`, `Anticrénelage` | `Anticrénelage` | — |

```bash
python -m cvlab.cli decrire annoter_forme
```

### Grille de repères

`grille_reperes`

Superpose une grille graduée en pixels.

**Origine** : TP2 §1.1, TP3 §1 · **Entrée** : `any` · **Sortie** : `color`

**Principe**

Aide de lecture pour retrouver des coordonnées sur l'image. Elle rappelle la convention du TP2 §1.1 : l'origine (0, 0) est en haut à gauche, l'axe des x va vers la droite (colonnes) et l'axe des y vers le bas (lignes) — l'inverse du repère mathématique habituel.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `pas` | Pas de la grille | entier, 10 à 500, pas 10 | 50 | — |
| `couleur` | Couleur | couleur BGR | (80, 200, 255) | — |
| `epaisseur` | Épaisseur | entier, 1 à 4 | 1 | — |
| `etiquettes` | Afficher les coordonnées | booléen | True | — |

```bash
python -m cvlab.cli decrire grille_reperes
```

### Texte

`annoter_texte`

Écrit un texte sur l'image.

**Origine** : TP1 §2, TP2 §2.3 · **Entrée** : `any` · **Sortie** : `same`

**Principe**

cv2.putText ne sait dessiner que les polices vectorielles Hershey embarquées dans OpenCV — pas de police système, et pas de caractères hors Latin-1 (les accents passent, pas les alphabets non latins). Le point ``org`` donné est le coin **bas-gauche** de la première ligne, pas le coin haut-gauche : c'est la cause classique du texte qui « disparaît » quand on lui donne y = 0.

**Paramètres**

| Paramètre | Libellé | Type et bornes | Défaut | Rôle |
| --- | --- | --- | --- | --- |
| `texte` | Texte | texte, 120 caractères au plus | `Vision par ordinateur` | — |
| `x` | Position X | réel, 0 à 100, pas 0.5 | 5.0 | — |
| `y` | Position Y | réel, 0 à 100, pas 0.5 | 10.0 | — |
| `taille` | Échelle de police | réel, 0.2 à 6, pas 0.1 | 1.0 | — |
| `epaisseur` | Épaisseur | entier, 1 à 20 | 2 | — |
| `couleur` | Couleur | couleur BGR | (0, 255, 0) | — |
| `police` | Police | choix : `Simplex`, `Plain`, `Duplex`, `Complex`, `Triplex`, `Complex Small`, `Script Simplex`, `Script Complex` | `Complex` | — |
| `fond` | Fond opaque derrière le texte | booléen | False | Garantit la lisibilité sur une image claire ou chargée. |

```bash
python -m cvlab.cli decrire annoter_texte
```


---

*Page générée par `outils/generer_doc.py` depuis le catalogue des filtres. Pour la modifier, changer les métadonnées du filtre dans `cvlab/filters/` puis relancer le générateur.*
