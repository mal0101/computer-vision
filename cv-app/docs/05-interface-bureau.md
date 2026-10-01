# 05 — Interface de bureau (OpenCV)

```bash
python ui/desktop.py                       # dernière image d'exemple trouvée
python ui/desktop.py image.jpg             # un fichier précis
python ui/desktop.py --webcam              # flux webcam continu
python ui/desktop.py --webcam 1 --largeur 1280 --hauteur 720
python ui/desktop.py image.jpg --filtre canny
python ui/desktop.py --webcam --famille "Contours"
python ui/desktop.py image.jpg --preset presets/01_contours_canny.json
```

Cette interface est la continuation directe du **TP3** : réglages par
trackbars (`cv2.createTrackbar`), interactions à la souris
(`cv2.setMouseCallback`), et surtout **flux webcam en temps réel** comme au
TP1 §3.4 — ce que l'interface web ne peut pas offrir.

## Les fenêtres

| Fenêtre | Contenu |
| --- | --- |
| **Atelier de filtres (OpenCV)** | l'image filtrée, avec les informations incrustées |
| **Reglages** | une trackbar par paramètre réglable du filtre courant |
| **Original** | l'image d'entrée (touche `o`) |
| **Histogramme** | l'histogramme du résultat (touche `h`) |

## Touches

| Touche | Effet |
| --- | --- |
| `n` / `p` | filtre suivant / précédent |
| `f` | liste des filtres dans le terminal |
| `o` | afficher / masquer l'image d'origine |
| `h` | afficher / masquer l'histogramme |
| `i` | afficher / masquer l'aide incrustée |
| `m` | afficher / masquer les mesures MSE et PSNR |
| `r` | réinitialiser les paramètres du filtre courant |
| `c` | figer / reprendre le flux webcam |
| `s` | enregistrer l'image affichée (dans `sorties/`, ou `--sorties DOSSIER`) |
| `ESC` ou `q` | quitter |

## Souris

| Geste | Effet |
| --- | --- |
| déplacement | **sonde de pixel** : coordonnées et valeur BGR affichées en bas de l'incrustation (TP3 §1) |
| glisser (bouton gauche) sur le filtre *Recadrer* | définit la région de recadrage ; les trackbars se mettent à jour (TP3 §4) |
| quatre clics gauche sur *Déformation de perspective* | définit les quatre points source, dans l'ordre haut-gauche, haut-droit, bas-droit, bas-gauche (TP4 §3) |
| clic droit | efface les points de perspective en cours |

## Les trackbars : ce qu'il faut savoir

`cv2.createTrackbar` ne manipule que des **entiers positifs**. La classe
`EchelleTrackbar` fait la traduction dans les deux sens :

| Type de paramètre | Correspondance |
| --- | --- |
| entier | position = valeur − minimum (les bornes négatives sont donc gérées : le paramètre β de −128 à 128 devient une trackbar de 0 à 256) |
| entier impair | pas de 2 depuis la borne impaire : **aucune position ne peut produire un noyau pair** |
| réel | 200 crans entre le minimum et le maximum |
| booléen | 0 / 1 |
| choix | une position par option |
| **texte, couleur** | *pas de trackbar possible* — la valeur par défaut est utilisée, et le terminal le signale au changement de filtre. Utilisez l'interface web pour ces paramètres. |

Les valeurs sont relues par `cv2.getTrackbarPos` à chaque image, pas dans le
callback : il n'y a donc aucun problème de concurrence entre le réglage et
l'affichage. OpenCV ne sachant pas supprimer une trackbar, changer de filtre
détruit et reconstruit la fenêtre *Reglages*.

**Les réglages sont conservés** par filtre : revenir sur un filtre déjà réglé
retrouve ses valeurs.

## Options de ligne de commande

| Option | Rôle |
| --- | --- |
| `image` | fichier à traiter (facultatif) |
| `--webcam [INDEX]` | flux webcam continu ; index 0 par défaut |
| `--largeur`, `--hauteur` | résolution demandée à la webcam (le pilote peut la refuser) |
| `--filtre ID` | filtre affiché au démarrage |
| `--famille NOM` | restreindre la navigation `n`/`p` à une famille |
| `--preset FICHIER` | chaîne JSON appliquée **avant** le filtre réglé — pratique pour régler une dernière étape au-dessus d'un prétraitement figé |
| `--reference IMAGE` | deuxième image ; sans elle, les filtres de fusion sont retirés de la liste |
| `--max-cote PX` | réduit une image fixe trop grande (1000 px par défaut) |
| `--sorties DOSSIER` | destination de la touche `s` |

## Sur la webcam

- La résolution demandée n'est pas garantie : `capture.set` renvoie
  silencieusement `False` si le pilote la refuse. La résolution réellement
  obtenue est affichée au lancement.
- Sur macOS, l'application essaie le backend `AVFOUNDATION` puis `CAP_ANY`, et
  donne un message explicite si la caméra reste inaccessible (occupée par une
  autre application, ou autorisation refusée).
- La touche `c` **fige** l'image : utile pour régler finement un filtre coûteux
  sans que la scène bouge.
- Les filtres marqués coûteux (le bilatéral) restent fluides sur une petite
  résolution ; sinon, réduisez avec `--largeur 640 --hauteur 480`.

## Ce qui n'est pas testé automatiquement

La boucle d'affichage a besoin d'un serveur graphique et d'un œil humain. En
revanche, **toute la logique qui l'entoure** l'est : conversion
trackbar ↔ paramètre pour les 46 filtres et chacune de leurs positions,
traduction des gestes souris en paramètres, traitement d'une image, gestion des
erreurs, enregistrement. Voir [13 — Tests et qualité](13-tests.md).
