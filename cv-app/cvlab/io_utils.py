"""Entrées / sorties : fichiers, encodage mémoire, webcam.

Trois sources d'image sont prévues, conformément à l'énoncé du projet :

1. un fichier choisi par l'utilisateur (téléversé dans l'interface web, ou
   passé en argument à la ligne de commande) ;
2. la **webcam** — instantané dans l'interface web, flux continu dans
   l'application de bureau ;
3. des images d'exemple, récupérées dans les dossiers ``TP*`` du dépôt.

Points d'attention regroupés ici une fois pour toutes :

* ``cv2.imread`` renvoie ``None`` sans lever d'exception quand le chemin est
  faux — c'est l'erreur de débutant la plus fréquente, donc on la transforme en
  exception explicite ;
* ``cv2.imread`` ne gère pas les chemins non ASCII sur toutes les plateformes ;
  on lit donc les octets en Python puis on décode avec ``cv2.imdecode`` ;
* sur macOS, l'ouverture de la webcam nécessite l'autorisation « Caméra » du
  terminal, et le premier appel peut prendre une seconde ou deux.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import images as im

__all__ = [
    "CameraError",
    "ImageLoadError",
    "EXTENSIONS_IMAGE",
    "load_image",
    "decode_image",
    "encode_image",
    "save_image",
    "sample_images",
    "Camera",
    "capture_snapshot",
]


class ImageLoadError(ValueError):
    """Le fichier ou le flux d'octets n'a pas pu être décodé en image."""


class CameraError(RuntimeError):
    """La webcam n'a pas pu être ouverte ou lue."""


#: Extensions reconnues par OpenCV et proposées dans les sélecteurs de fichier.
EXTENSIONS_IMAGE: tuple[str, ...] = (
    "jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp", "ppm", "pgm",
)


def load_image(path: str | Path, flag: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """Charge une image depuis un fichier (TP1 §3.1).

    Lève :class:`ImageLoadError` si le fichier est absent ou illisible, au lieu
    de renvoyer ``None`` comme ``cv2.imread``.
    """
    chemin = Path(path).expanduser()
    if not chemin.is_file():
        raise ImageLoadError(f"fichier introuvable : {chemin}")
    try:
        donnees = np.frombuffer(chemin.read_bytes(), dtype=np.uint8)
    except OSError as exc:
        raise ImageLoadError(f"lecture impossible : {chemin} ({exc})") from exc
    image = cv2.imdecode(donnees, flag)
    if image is None:
        raise ImageLoadError(
            f"format non reconnu par OpenCV : {chemin.name} "
            f"(extensions gérées : {', '.join(EXTENSIONS_IMAGE)})"
        )
    return im.normalise(image)


def decode_image(data: bytes, flag: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """Décode une image depuis des octets (fichier téléversé dans Streamlit)."""
    if not data:
        raise ImageLoadError("flux d'octets vide")
    tableau = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(tableau, flag)
    if image is None:
        raise ImageLoadError(
            "contenu non reconnu comme image "
            f"(extensions gérées : {', '.join(EXTENSIONS_IMAGE)})"
        )
    return im.normalise(image)


def encode_image(
    image: np.ndarray, extension: str = ".png", qualite: int = 95
) -> bytes:
    """Encode une image en mémoire (pour un bouton de téléchargement)."""
    image = im.normalise(image)
    extension = extension if extension.startswith(".") else f".{extension}"
    parametres: list[int] = []
    if extension.lower() in (".jpg", ".jpeg"):
        parametres = [cv2.IMWRITE_JPEG_QUALITY, int(max(1, min(100, qualite)))]
    elif extension.lower() == ".webp":
        parametres = [cv2.IMWRITE_WEBP_QUALITY, int(max(1, min(100, qualite)))]
    try:
        succes, tampon = cv2.imencode(extension, image, parametres)
    except cv2.error as exc:
        # Extension inconnue : OpenCV lève, on uniformise avec le cas « échec ».
        raise ImageLoadError(
            f"extension non prise en charge pour l'encodage : {extension}"
        ) from exc
    if not succes:
        raise ImageLoadError(f"encodage impossible en {extension}")
    return tampon.tobytes()


def save_image(image: np.ndarray, path: str | Path, qualite: int = 95) -> Path:
    """Écrit une image sur disque (TP1 §3.2), en créant les dossiers manquants."""
    chemin = Path(path).expanduser()
    chemin.parent.mkdir(parents=True, exist_ok=True)
    donnees = encode_image(image, chemin.suffix or ".png", qualite)
    chemin.write_bytes(donnees)
    return chemin


def sample_images(racine: str | Path | None = None, limite: int = 40) -> list[Path]:
    """Images d'exemple trouvées dans les dossiers ``TP*`` du dépôt.

    Permet d'essayer l'application sans rien téléverser, avec les images qui
    ont servi pendant les travaux pratiques (images bruitées du TP7, cellules,
    pièces de monnaie, photographies du TP1-TP4).
    """
    base = Path(racine) if racine else Path(__file__).resolve().parent.parent.parent
    trouvees: list[Path] = []
    if not base.is_dir():
        return trouvees
    for dossier in sorted(base.glob("TP*")):
        if not dossier.is_dir():
            continue
        for extension in ("jpg", "jpeg", "png", "bmp"):
            trouvees.extend(sorted(dossier.rglob(f"*.{extension}")))
    # Dossier d'exemples propre à l'application, s'il est rempli.
    local = Path(__file__).resolve().parent.parent / "assets"
    if local.is_dir():
        for extension in ("jpg", "jpeg", "png", "bmp"):
            trouvees.extend(sorted(local.rglob(f"*.{extension}")))
    # Dédoublonnage en conservant l'ordre.
    vues: set[Path] = set()
    uniques: list[Path] = []
    for chemin in trouvees:
        resolu = chemin.resolve()
        if resolu not in vues and "__pycache__" not in resolu.parts:
            vues.add(resolu)
            uniques.append(chemin)
    return uniques[:limite]


@dataclass
class Camera:
    """Webcam, utilisable comme gestionnaire de contexte (TP1 §3.4).

    Exemple ::

        with Camera(index=0, largeur=1280, hauteur=720) as cam:
            image = cam.read()

    ``cv2.VideoCapture`` ne lève pas d'exception quand le périphérique est
    absent : il renvoie un objet dont ``isOpened()`` est faux. On teste donc
    explicitement, et on essaie plusieurs *backends* — sur macOS,
    ``CAP_AVFOUNDATION`` est souvent nécessaire.
    """

    index: int = 0
    largeur: int | None = None
    hauteur: int | None = None
    fps: int | None = None

    def __post_init__(self) -> None:
        self._capture: cv2.VideoCapture | None = None

    # -- cycle de vie ---------------------------------------------------
    def open(self) -> Camera:
        """Ouvre le périphérique. Lève :class:`CameraError` en cas d'échec."""
        if self._capture is not None and self._capture.isOpened():
            return self

        # Sur macOS, le backend générique échoue souvent là où AVFoundation
        # fonctionne : on l'essaie en premier, puis on retombe sur CAP_ANY.
        backends: list[int] = []
        if sys.platform == "darwin":
            backends.append(cv2.CAP_AVFOUNDATION)
        backends.append(cv2.CAP_ANY)

        derniere_erreur = ""
        for backend in backends:
            capture = cv2.VideoCapture(int(self.index), backend)
            if capture.isOpened():
                self._capture = capture
                self._configurer()
                return self
            capture.release()
            derniere_erreur = f"backend {backend} refusé"

        raise CameraError(
            f"webcam {self.index} inaccessible ({derniere_erreur}). "
            "Vérifier qu'aucune autre application ne l'utilise et que "
            "l'autorisation « Caméra » est accordée au terminal."
        )

    def _configurer(self) -> None:
        """Applique les propriétés demandées (TP1 §3.4, TP3 §5).

        ``capture.set`` renvoie ``False`` quand le pilote refuse la valeur ; ce
        n'est pas une erreur fatale, la webcam garde simplement son réglage.
        """
        assert self._capture is not None
        if self.largeur:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.largeur))
        if self.hauteur:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.hauteur))
        if self.fps:
            self._capture.set(cv2.CAP_PROP_FPS, float(self.fps))

    def set_property(self, propriete: int, valeur: float) -> bool:
        """Règle une propriété de capture (luminosité, contraste, teinte...)."""
        if self._capture is None:
            raise CameraError("webcam non ouverte")
        return bool(self._capture.set(int(propriete), float(valeur)))

    def read(self) -> np.ndarray:
        """Lit une image. Lève :class:`CameraError` si la lecture échoue."""
        if self._capture is None:
            self.open()
        assert self._capture is not None
        succes, image = self._capture.read()
        if not succes or image is None:
            raise CameraError(
                "lecture de la webcam impossible (périphérique déconnecté ?)"
            )
        return im.normalise(image)

    def read_optional(self) -> np.ndarray | None:
        """Comme :meth:`read` mais renvoie ``None`` au lieu de lever.

        Pratique dans une boucle d'affichage, où une image perdue ne doit pas
        interrompre le programme.
        """
        try:
            return self.read()
        except CameraError:
            return None

    @property
    def resolution(self) -> tuple[int, int]:
        """Résolution réellement obtenue (le pilote peut ignorer la demande)."""
        if self._capture is None:
            return (0, 0)
        return (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def release(self) -> None:
        """Libère le périphérique. Toujours appeler, sinon la caméra reste prise."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> Camera:
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.release()


def capture_snapshot(
    index: int = 0,
    echauffement: int = 5,
    largeur: int | None = None,
    hauteur: int | None = None,
) -> np.ndarray:
    """Prend une seule photo avec la webcam.

    ``echauffement`` images sont lues et jetées avant la bonne : les premières
    images d'une webcam sortent souvent noires ou mal exposées, le temps que
    l'exposition automatique se stabilise.
    """
    with Camera(index=index, largeur=largeur, hauteur=hauteur) as camera:
        image = camera.read()
        for _ in range(max(0, int(echauffement))):
            suivante = camera.read_optional()
            if suivante is not None:
                image = suivante
        return image
