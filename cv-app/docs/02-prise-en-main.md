# 02 — Prise en main

Trois sessions guidées, à faire dans l'ordre. Comptez cinq minutes chacune.

---

## Session 1 — Détecter les contours de votre propre photo (interface web)

```bash
streamlit run ui/streamlit_app.py
```

1. **Barre latérale → 1 · Image source → Fichier** : téléversez une photo.
   Pas de photo sous la main ? Choisissez *Mire de test* : une image fabriquée
   exprès avec des contours francs, un dégradé, des traits fins et des aplats.
2. **Colonne de gauche** → famille *Contours & gradients*, filtre
   *Détecteur de Canny* → **Ajouter**.
3. Déplacez **Seuil bas** et **Seuil haut**. Observez :
   - seuils bas (20 / 60) → beaucoup de contours, mais aussi du bruit ;
   - seuils hauts (150 / 300) → seulement les contours francs, des trous
     apparaissent dans les contours.
   Le panneau sous les curseurs affiche le **rapport haut/bas** et le
   **pourcentage de pixels de contour** : visez un rapport de 2 à 3.
4. **Ajoutez une étape avant** : famille *Lissage & débruitage* →
   *Filtre gaussien*, puis remontez-la avec le bouton ↑. Mettez
   « Flou gaussien préalable » de Canny à 0 pour ne pas flouter deux fois.
   Augmentez la taille du noyau du gaussien : les contours parasites
   disparaissent, les vrais contours restent.
5. **Onglet Étapes** : l'image après chaque filtre, côte à côte. C'est là qu'on
   voit pourquoi l'ordre compte.
6. **Onglet Résultat → Télécharger** pour récupérer l'image.

> **À retenir.** Dériver une image bruitée amplifie le bruit. On lisse toujours
> avant de détecter des contours — c'est l'étape 1 de l'algorithme de Canny,
> et le paramètre « flou préalable » de tous les filtres de gradient.

---

## Session 2 — Voir le filtre médian battre la moyenne (comparateur)

L'exercice 3 du TP7, en version interactive.

1. Chargez une image (ou la mire de test).
2. Dans la chaîne : famille *Bruit* → **Bruit sel et poivre**, proportion
   `0.08`, graine `42` (une graine non nulle fige le tirage : indispensable
   pour comparer).
3. **Onglet Comparateur** → famille *Lissage & débruitage* → cochez
   *Filtre moyenneur*, *Filtre gaussien*, *Filtre médian*, *Filtre bilatéral*.
4. Lisez le tableau en bas : le **médian** obtient la MSE la plus faible, et
   visuellement il efface les points sans étaler de tache grise.

Pourquoi : le moyenneur et le gaussien calculent une **moyenne**, qu'un pixel à
0 ou à 255 déplace fortement ; le médian **trie** et prend la valeur centrale,
les valeurs aberrantes se retrouvent en bout de tri et n'ont aucun poids. Le
bilatéral, lui, protège les forts écarts d'intensité — or un pixel sel *est* un
fort écart, donc il le protège aussi : c'est le mauvais outil pour ce bruit.

Refaites l'essai en remplaçant le bruit sel-poivre par un **bruit gaussien**
(σ = 25) : le classement s'inverse, le gaussien et le bilatéral prennent
l'avantage. **Le bon filtre dépend du bruit**, c'est tout l'enjeu.

![Comparaison des filtres de lissage](images/lissage-comparaison.png)

---

## Session 3 — Filtrer la webcam en temps réel (interface de bureau)

```bash
python ui/desktop.py --webcam
```

Deux fenêtres s'ouvrent : l'image filtrée et la fenêtre **Reglages** avec une
trackbar par paramètre.

1. Tapez `n` plusieurs fois pour parcourir les filtres ; `f` liste les filtres
   dans le terminal.
2. Arrivez sur *Détecteur de Canny* et bougez les trackbars pendant que vous
   vous déplacez devant la caméra.
3. `o` affiche l'image d'origine dans une seconde fenêtre, `h` l'histogramme,
   `m` les mesures MSE/PSNR.
4. Passez sur *Recadrer* et **tracez un rectangle à la souris** : les
   paramètres du filtre se mettent à jour tout seuls — c'est l'exercice
   « Recadrer une image avec la souris » du TP3.
5. `c` fige l'image (pratique pour régler tranquillement), `c` à nouveau pour
   reprendre le flux, `s` enregistre l'image affichée, `ESC` quitte.

---

## Et ensuite

- Chargez une **chaîne d'exemple** : barre latérale → *3 · Chaînes
  enregistrées* → l'une des huit chaînes fournies (contours de Canny,
  débruitage, segmentation de cellules, redressement de document, correction
  d'exposition…).
- Lisez [07 — Théorie du filtrage](07-theorie-filtrage.md) pour comprendre ce
  que chaque réglage change réellement.
- Consultez [08 — Référence des filtres](08-reference-filtres.md) pour la fiche
  exacte d'un filtre et les bornes de ses paramètres.
