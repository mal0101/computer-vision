# Documentation de l'atelier de filtres

Application interactive permettant d'appliquer les filtres étudiés en TP1 à
TP7 à **votre** image ou à **votre** webcam, en réglant **vous-même** chaque
paramètre.

## Par où commencer

| Vous voulez… | Lisez |
| --- | --- |
| installer et lancer | [01 — Installation](01-installation.md) |
| une première session en 5 minutes | [02 — Prise en main](02-prise-en-main.md) |
| comprendre comment c'est construit | [03 — Architecture](03-architecture.md) |
| l'interface web en détail | [04 — Interface web](04-interface-web.md) |
| la fenêtre OpenCV et la webcam | [05 — Interface de bureau](05-interface-bureau.md) |
| automatiser, traiter un lot | [06 — Ligne de commande](06-ligne-de-commande.md) |
| comprendre *pourquoi* un filtre fait ça | [07 — Théorie du filtrage](07-theorie-filtrage.md) |
| la fiche exacte d'un filtre | [08 — Référence des filtres](08-reference-filtres.md) |
| les types de paramètres et leurs bornes | [09 — Paramètres](09-parametres.md) |
| enregistrer et rejouer une chaîne | [10 — Chaînes et presets](10-chaines-et-presets.md) |
| savoir d'où vient chaque fonctionnalité | [11 — Correspondance avec les TP](11-correspondance-tp.md) |
| ajouter votre propre filtre | [12 — Étendre l'application](12-etendre.md) |
| lancer et comprendre les tests | [13 — Tests et qualité](13-tests.md) |
| réparer un problème | [14 — Dépannage](14-depannage.md) |
| appeler le code depuis vos scripts | [15 — Référence de l'API](15-api.md) |

## Ce que fait l'application

- **46 filtres** répartis en 11 familles, couvrant les notions des TP1 à TP7 :
  espaces colorimétriques, géométrie, bruit, lissage, contours, seuillage,
  morphologie, histogramme, fusion, annotation.
- **Tous les paramètres sont saisis par l'utilisateur** : chaque filtre déclare
  ses paramètres avec leur type, leurs bornes et leur rôle ; les interfaces
  construisent les curseurs à partir de cette déclaration, rien n'est figé dans
  le code d'interface.
- **L'image vient de l'utilisateur** : fichier téléversé, instantané de webcam,
  flux webcam continu, image des TP, ou mire de test générée.
- **Les filtres s'enchaînent** : on empile des étapes, on les réordonne, on les
  désactive une par une pour isoler l'effet de chacune.
- **Trois interfaces** sur le même noyau : web, bureau (OpenCV), ligne de
  commande.

## Les trois interfaces en un coup d'œil

| | Web (Streamlit) | Bureau (OpenCV) | Ligne de commande |
| --- | --- | --- | --- |
| Lancement | `streamlit run ui/streamlit_app.py` | `python ui/desktop.py` | `python -m cvlab.cli` |
| Réglage | curseurs, listes, sélecteur de couleur, champs texte | trackbars | arguments `clé=valeur` |
| Image | fichier, instantané webcam, exemples, mire | fichier, **flux webcam continu** | fichier, instantané webcam |
| Chaîne de filtres | oui, éditable | un filtre + chaîne préalable | oui, via `-s` ou un preset |
| Souris | inspecteur de pixel par coordonnées | sonde de pixel, recadrage au glisser, 4 points de perspective | — |
| Analyse | histogrammes, MSE/PSNR, carte des écarts, comparateur | histogramme, MSE/PSNR incrustés | tableau comparatif |
| Pour quoi | régler, comparer, exporter | temps réel, démonstration | lot, mesures reproductibles |

## Arborescence

```
cv-app/
├── cvlab/              noyau, sans aucune dépendance d'interface
│   ├── params.py       déclaration typée des paramètres
│   ├── images.py       conventions BGR / uint8 et conversions
│   ├── drawing.py      fonctions de dessin du TP2
│   ├── registry.py     catalogue des filtres
│   ├── filters/        les filtres, un module par famille
│   ├── pipeline.py     chaînes d'étapes et format de preset
│   ├── histogram.py    calcul et tracé d'histogrammes
│   ├── metrics.py      MSE, RMSE, MAE, PSNR
│   ├── io_utils.py     fichiers, encodage mémoire, webcam
│   └── cli.py          interface en ligne de commande
├── ui/
│   ├── streamlit_app.py  interface web
│   ├── widgets.py        fabrique de widgets depuis les ParamSpec
│   └── desktop.py        fenêtre OpenCV, trackbars, webcam
├── outils/
│   ├── demo.py           planches d'illustration de la documentation
│   └── generer_doc.py    génération de la référence des filtres
├── presets/            chaînes d'exemple au format JSON
├── tests/              plus de 600 tests (pytest)
├── docs/               cette documentation
└── assets/             vos images
```
