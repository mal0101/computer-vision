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

- **Total** `TOTAL : 37,00 DH` + décompte `1 DH ×2  2 DH ×2  …`
  (répartition par valeur consolidée de la session) ;
- bandeau de synthèse : profil actif, échelle `5,61 px/mm` — suivie de
  **« (estimée) »** quand elle vient du catalogue et non d'une mesure, la
  distinction ne devant jamais s'effacer — ou « non étalonné » ; durée
  d'analyse et cadence correspondante ;
- dernier message de session (étalonnage réussi ou refusé, remise à zéro).
  Ces messages étaient produits et transportés jusqu'au rendu, mais jamais
  dessinés : un étalonnage qui échouait ne disait rien à l'écran ;
- chaque pièce : anneau coloré selon sa certitude
  `confirme` (vert) / `tentative` (jaune) / `inconnue` (orange) /
  motif d'incertitude (gris) ;
- détail : le dessin part d'une **copie** (la frame d'entrée n'est jamais
  modifiée) ; la taille de police suit la hauteur de la frame, bornée pour
  rester lisible d'une petite capture à la 4K ; la hauteur du bandeau se
  déduit du nombre de lignes à écrire, et le bloc de métrologie passe sous
  le bloc principal quand le cadre est trop étroit pour deux colonnes.
  Les libellés de pièce sont ramenés dans le cadre et tronqués au besoin —
  `cv2.putText` ne sait ni couper ni replier, il écrivait simplement dans le
  vide.

`dessiner_hud_inventaire` rend la même surcouche pour une image fixe, sans
notion de piste ni de session.

La vignette du masque (touche `d`) est bornée aux dimensions réelles du
cadre : sa taille se déduisait de celle du **masque** (échelle d'analyse) et
était écrite telle quelle dans l'**image** (plein cadre), si bien qu'une
simple pression sur `d` sur une petite capture levait
`ValueError: could not broadcast input array` et tuait toute la session.

## 3. CLI (`python -m pieces.cli`)

```
--config JSON          fichier de configuration (défaut donnees/config.json)
--image JPG            analyser une image fixe → inventaire, puis s'arrêter
--camera N             session en direct sur la webcam d'indice N
--sans-fenetre         autotest du pipeline (aucune caméra, vérité connue)
--centimes             compter aussi les 5c/10c/20c
--echelle-px-mm N      échelle MESURÉE, fournie (désactive l'estimation)
--sans-auto-echelle    interdire l'estimation de l'échelle au catalogue
--etalon-mm MM         diamètre réel de la pièce d'étalonnage (défaut 24,0)
--limite N             borner le nombre de frames
--sortie PNG           écrire le HUD
--json FICHIER         rapport lisible par une machine
--lister-pieces        détailler chaque pièce sur la sortie standard
--version
```

**Trois modes exclusifs**, et aucun par défaut. Lancer la commande nue
affiche désormais un message et rend 2 : auparavant elle ouvrait la webcam 0
et bouclait indéfiniment, sans la moindre sortie ni moyen d'arrêt autre
qu'une interruption — une commande non interactive restait bloquée à pleine
charge CPU.

`--image` produit un **inventaire** (ce que l'image contient), `--camera` un
**total suivi** (une pièce comptée une seule fois sur la durée). La
distinction n'est pas cosmétique : le suivi exige plusieurs présences avant
de compter, ce qu'une image unique ne peut structurellement pas fournir —
`--image` rendait donc fatalement `0,00 DH`.

`--sans-fenetre` **exécute réellement** la chaîne sur une scène de référence
dont le total est connu par construction, et son code de sortie reflète
l'état du pipeline. Il se contentait auparavant d'imprimer une phrase et de
rendre 0, sans rien exécuter : il ne pouvait signaler aucune régression,
alors qu'il était documenté comme « autotest du pipeline ».

`--etalon-mm` est validé dès l'analyse des arguments (bornes 10–40 mm). Une
valeur ≤ 0 faisait auparavant remonter une exception qui **terminait la
session** — et perdait le total accumulé — à la première tentative
d'étalonnage ; une valeur énorme bouclait en silence, l'échec n'étant écrit
que dans un journal que le HUD n'affichait pas.

Codes de sortie : `0` succès, `1` erreur caméra/runtime, `2` argument ou
configuration invalide.

### Raccourcis Makefile

```
make check     lint + suite de tests (porte principale)
make perf      tests de performance (-m perf)
make bench     tableau des temps d'exécution
make autotest  autotest sans matériel         (python -m pieces.cli --sans-fenetre)
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