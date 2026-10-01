# 01 — Installation

## Prérequis

- **Python 3.10 ou plus récent.** Le code utilise les `dataclass` à champs
  nommés obligatoires (`kw_only`), apparues en 3.10.
- Une **webcam** si vous voulez utiliser la capture vidéo (facultatif : tout le
  reste fonctionne sur fichier).

Vérifier la version de Python :

```bash
python3 --version      # doit afficher 3.10 ou plus
```

## Installation

Depuis le dossier `cv-app` :

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Ou, avec le `Makefile` fourni :

```bash
make venv
```

### Vérifier l'installation

```bash
python -c "import cv2, numpy, streamlit; print(cv2.__version__, numpy.__version__, streamlit.__version__)"
python -m cvlab.cli lister | tail -3        # doit afficher « 46 filtre(s). »
python -m pytest -q                         # plus de 600 tests, aucun échec
```

## Lancement

```bash
# Interface web : s'ouvre dans le navigateur sur http://localhost:8501
streamlit run ui/streamlit_app.py

# Interface de bureau, flux webcam temps réel
python ui/desktop.py --webcam

# Interface de bureau sur un fichier
python ui/desktop.py ../TP7/exercice\ 5/cells.jpg

# Ligne de commande
python -m cvlab.cli lister
```

Les mêmes commandes via `make` : `make web`, `make webcam`, `make bureau`,
`make test`.

## Dépendances et rôle de chacune

| Paquet | Rôle | Obligatoire |
| --- | --- | --- |
| `opencv-python` | tout le traitement d'image, la webcam, les fenêtres et les trackbars | oui |
| `numpy` | représentation des images (tableaux `uint8`) | oui |
| `streamlit` | interface web | pour l'interface web seulement |
| `pillow` | décodage des images téléversées, utilisé par Streamlit | avec Streamlit |
| `pytest` | suite de tests | développement seulement |

Le noyau `cvlab` ne dépend que d'OpenCV et de NumPy : vous pouvez l'utiliser
dans vos propres scripts sans installer Streamlit.

## Versions d'OpenCV

Le code est vérifié avec **OpenCV 4.9+ et 5.0**. Les rares différences entre
ces versions sont gérées ou évitées :

- toutes les constantes utilisées (`COLORMAP_*`, `MORPH_*`, `THRESH_*`,
  `BORDER_*`, `CAP_PROP_*`) existent dans les deux versions ;
- `cv2.BORDER_WRAP` n'est pas proposé pour les filtres de convolution, car le
  moteur de filtrage d'OpenCV le refuse (il reste disponible pour les
  transformations géométriques, où il est valide).

## Autorisations de la webcam

- **macOS** : la première ouverture déclenche une demande d'autorisation
  « Caméra » pour le terminal ou l'application qui lance Python. Si elle a été
  refusée, la réautoriser dans *Réglages Système → Confidentialité et sécurité
  → Caméra*. L'application essaie d'abord le backend `AVFOUNDATION`, puis
  `CAP_ANY`.
- **Windows** : *Paramètres → Confidentialité → Caméra*, autoriser les
  applications de bureau.
- **Linux** : l'utilisateur doit appartenir au groupe `video` ; vérifier que
  `/dev/video0` existe.

Pour tester l'accès sans lancer d'interface :

```bash
python -m cvlab.cli webcam --index 0 -o /tmp/essai.png
```

## Installation en paquet (facultatif)

```bash
python -m pip install -e .
cvlab lister          # la commande « cvlab » devient disponible
```

## Désinstallation

Supprimer le dossier `.venv`. Aucune donnée n'est écrite ailleurs, sauf les
images que vous enregistrez vous-même (par défaut dans `cv-app/sorties/`).
