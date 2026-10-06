"""Façade unique de réutilisation de cv-app/ (dossier frère du dépôt).

Règle du projet : **tout** ce qui est réutilisé de cv-app passe par ce
module. Ailleurs dans ``pieces``, on importe ``cv2``/``numpy`` directement
et les modules internes de coin-counter — jamais ``cvlab``.

Ce qui est réutilisé (10 fonctions + 2 exceptions) :

* ``Camera`` / ``capture_snapshot``      — webcam (fallback AVFOUNDATION,
  chaude de 5 images pour un instantané stable) ;
* ``load_image`` / ``save_image``        — entrées/sorties fichier, erreurs
  explicites au lieu de ``None`` ;
* ``to_gray`` / ``to_bgr`` / ``normalise`` — conventions d'images (BGR uint8) ;
* ``dess_cercle`` / ``dess_text`` / ``dess_texte_encadre`` — annotations du HUD ;
* ``CameraError`` / ``ImageLoadError``   — exceptions d'entrée/sortie.

Ce qui **n'est pas** réutilisé : le catalogue de filtres ``cvlab.filters``
(les 46 modules, ≈105 ms — jamais chargés, cf. test) et toute fonction du
registre ``cvlab.registry`` : coin-counter doit fonctionner quel que soit
le filtre, y compris sans filtre. Le corps du paquet ``cvlab`` importe
pourtant ``registry``/``pipeline`` au chargement (défini dans
``cvlab/__init__`` de cv-app) — c'est inévitable et sans effet ici.

``cv-app`` n'est pas installé dans le venv : ``cvlab`` n'est importable que
via ``sys.path``. Le chemin est donc résolu **à partir de ce fichier**
(``compat.py``), ce qui fonctionne quel que soit le répertoire courant.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: racine du dépôt (parent de coin-counter/)
_RACINE = Path(__file__).resolve().parents[2]
#: dossier cv-app à injecter dans sys.path
_CV_APP = _RACINE / "cv-app"

if not _CV_APP.is_dir():
    raise ImportError(
        "dossier cv-app introuvable à côté de coin-counter "
        f"(attendu : {_CV_APP}). Ce projet réutilise sa webcam et ses "
        "outils d'entrée/sortie : cloner le dépôt complet."
    )
if str(_CV_APP) not in sys.path:
    sys.path.insert(0, str(_CV_APP))

# Importé *après* l'injection de sys.path (cf. per-file-ignores ruff E402).
from cvlab.drawing import dess_cercle, dess_text, dess_texte_encadre  # noqa: E402
from cvlab.images import normalise, to_bgr, to_gray  # noqa: E402
from cvlab.io_utils import (  # noqa: E402
    Camera,
    CameraError,
    ImageLoadError,
    capture_snapshot,
    load_image,
    save_image,
)

__all__ = [
    "Camera",
    "CameraError",
    "ImageLoadError",
    "capture_snapshot",
    "load_image",
    "save_image",
    "normalise",
    "to_gray",
    "to_bgr",
    "dess_cercle",
    "dess_text",
    "dess_texte_encadre",
]
