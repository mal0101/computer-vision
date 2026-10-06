# 04 — Interface utilisateur (M3/M4)

`Bureau` (boucle webcam) et `pieces.cli` (interface ligne de commande).
Testés sans matériel : la caméra n'est touchée que dans
`pieces/ui/bureau.creer_capture` et `pieces.cli.principal` (mode webcam) ;
tout le reste reçoit une capture **injectée**.

## 1. Bureau — boucle directe

`Bureau(config, etat, capture=None, afficher=True)` :

- lit une frame (`capture.read()`) → `traiter_image` → `dessiner_hud` →
  `cv2.imshow` (si `afficher`) → `waitKey(1)` ;
- termine quand `capture` fournit `False/None`, sur touche, ou après
  `limite` frames (mode `--limite`, pratique pour CI sans fenêtre) ;
- `release()` la capture en `finally`, `destroyAllWindows` si affichage.

### Touches (session fenêtrée)

| Touche | Action |
|---|---|
| `Échap` / `q` | quitter |
| `c` | lancer/arrêter l'étalonnage « tenir la pièce connue » |
| `d` | afficher/masquer la vignette du masque de profilage |
| `r` | remettre le comptage à zéro |

## 2. Rendu HUD (`pieces/rendu.dessiner_hud`)

- **Total** `TOTAL : 12,50 DH` + décompte `5 DH ×1  100c ×3  …`
  (répartition par valeur consolidée de la session) ;
- bandeau de synthèse : profil actif, échelle `4,0 px/mm` ou
  « non calibré », temps/frame (`xx ms`) ;
- chaque pièce : anneau coloré selon sa certitude
  `confirme` (vert) / `tentative` (jaune) / `inconnue` (orange) /
  motif d'incertitude (gris) ;
- détail : le dessin part d'une **copie** (la frame d'entrée n'est jamais
  modifiée), et la taille de police de `putText` suit la hauteur de la
  frame (lisible à 720p).

## 3. CLI (`python -m pieces.cli`)

```
--config JSON       fichier de configuration (défaut donnees/config.json)
--centimes          compter aussi les 10c/20c
--camera N          indice de la webcam (défaut 0)
--etalon-mm MM      diamètre réel de la pièce tenue (défaut 24,0 = 1 DH)
--limite N          borner le nombre de frames (CI)
--sortie PNG        écrire le HUD de la dernière frame puis arrêter
--sans-fenetre      autotest du pipeline, aucune webcam ni fenêtre
--image JPG         traiter une image statique (sorties/hud.png par défaut)
--version
```

Codes de sortie : `0` succès, `1` erreur caméra/runtime, `2` argument ou
configuration invalide.

### Raccourcis Makefile

```
make check     lint + suite de tests (porte principale)
make perf      tests de performance (-m perf)
make bench     tableau des temps d'exécution
make webcam    session en direct              (python -m pieces.cli --camera 0)
make image     IMAGE=… traite une image        (python -m pieces.cli --image …)
make capture   une frame enregistrée           (--limite 1 --sortie sorties/capture.png)
```

## 4. Tests

`tests/test_m3.py` (bureau) : faux capteur (`FauxCapteur`) qui rejoue une
liste de frames ; `test_bureau_loop_sans_fenetre_compte_une_piece` vérifie
qu'un total « 1 DH » émerge après 10 frames sans aucune fenêtre ;
`test_bureau_total_monotone` ; `test_touches_bureau` (q/c/d/r) ;
capture `None` → retour immédiat. CLI : arguments, `--version` (exit 0),
`--sans-fenetre` (0), config invalide (2), `--image` produit un PNG.

> Aucun test de la suite par défaut n'ouvre de webcam. La porte webcam
> réelle (Tier 3) reste **non vérifiée** en CI : pas de caméra connectée
> pendant le développement (cf. `docs/01-portee.md` § 8 et README § 4).