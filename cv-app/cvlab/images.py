"""Conventions et conversions d'images.

Toute l'application manipule des tableaux NumPy ``uint8`` :

* image couleur  -> forme ``(H, W, 3)`` dans l'ordre **BGR** (ordre natif
  d'OpenCV, cf. TP2 §1.2) ;
* image en niveaux de gris -> forme ``(H, W)``, un seul canal.

Les images à 4 canaux (BGRA, PNG transparents) sont ramenées en BGR dès le
chargement, et les images 16 bits sont ramenées en 8 bits : cela évite des
erreurs opaques d'OpenCV loin en aval dans la chaîne de traitement.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "InvalidImageError",
    "is_gray",
    "is_color",
    "normalise",
    "to_gray",
    "to_bgr",
    "to_rgb",
    "match_shape",
    "match_channels",
    "describe_image",
]


class InvalidImageError(ValueError):
    """Image absente, vide ou de forme non prise en charge."""


def _check(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise InvalidImageError("image absente (None)")
    if not isinstance(image, np.ndarray):
        raise InvalidImageError(f"tableau NumPy attendu, reçu {type(image).__name__}")
    if image.size == 0:
        raise InvalidImageError("image vide (0 pixel)")
    if image.ndim not in (2, 3):
        raise InvalidImageError(
            f"forme non prise en charge: {image.shape} "
            "(attendu (H, W) ou (H, W, C))"
        )
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise InvalidImageError(
            f"nombre de canaux non pris en charge: {image.shape[2]} (attendu 1, 3 ou 4)"
        )
    return image


def is_gray(image: np.ndarray) -> bool:
    """Vrai si l'image a un seul canal."""
    _check(image)
    return image.ndim == 2 or image.shape[2] == 1


def is_color(image: np.ndarray) -> bool:
    """Vrai si l'image a trois canaux (BGR)."""
    return not is_gray(image)


def normalise(image: np.ndarray) -> np.ndarray:
    """Ramène n'importe quelle image lisible à la convention interne.

    * 4 canaux (BGRA) -> 3 canaux BGR ;
    * 1 canal de forme ``(H, W, 1)`` -> ``(H, W)`` ;
    * profondeur 16/32 bits ou flottante -> ``uint8`` (mise à l'échelle
      min-max pour le flottant, division par 257 pour le 16 bits).
    """
    _check(image)
    out = image

    if out.ndim == 3 and out.shape[2] == 4:
        out = cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)
    elif out.ndim == 3 and out.shape[2] == 1:
        out = out[:, :, 0]

    if out.dtype == np.uint8:
        return np.ascontiguousarray(out)
    if out.dtype == np.uint16:
        return np.ascontiguousarray((out / 257.0).round().clip(0, 255).astype(np.uint8))
    if np.issubdtype(out.dtype, np.floating):
        finite = out[np.isfinite(out)]
        if finite.size == 0:
            return np.zeros(out.shape, np.uint8)
        low, high = float(finite.min()), float(finite.max())
        if high - low < 1e-12:
            scaled = np.zeros(out.shape, np.float32)
        else:
            scaled = (out.astype(np.float32) - low) * (255.0 / (high - low))
        return np.ascontiguousarray(np.clip(scaled, 0, 255).astype(np.uint8))
    # entiers signés et autres : on borne puis on caste
    return np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))


def to_gray(image: np.ndarray) -> np.ndarray:
    """Image en niveaux de gris de forme ``(H, W)`` (cf. TP1 §3.2)."""
    image = normalise(image)
    if is_gray(image):
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Image couleur de forme ``(H, W, 3)``, en dupliquant le canal si besoin."""
    image = normalise(image)
    if is_color(image):
        return image
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Image RVB, pour l'affichage par Streamlit / Matplotlib / PIL.

    OpenCV lit et écrit en BGR, les bibliothèques d'affichage attendent du
    RGB : oublier cette conversion est l'erreur classique du TP7 (les images
    s'affichent avec le rouge et le bleu échangés).
    """
    image = normalise(image)
    if is_gray(image):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def match_shape(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Redimensionne ``reference`` à la taille de ``image``.

    C'est exactement ce que fait ``TraitementImage.fusion()`` du TP4 avant
    d'appeler ``cv2.addWeighted`` : les deux opérandes doivent avoir la même
    taille, sinon OpenCV lève une erreur.
    """
    image = normalise(image)
    reference = normalise(reference)
    height, width = image.shape[:2]
    if reference.shape[:2] == (height, width):
        return reference
    # INTER_AREA pour une réduction, INTER_LINEAR pour un agrandissement.
    shrinking = reference.shape[0] > height or reference.shape[1] > width
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(reference, (width, height), interpolation=interpolation)


def match_channels(
    first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Aligne le nombre de canaux de deux images (gris promu en BGR)."""
    first = normalise(first)
    second = normalise(second)
    if is_gray(first) and is_gray(second):
        return first, second
    return to_bgr(first), to_bgr(second)


def describe_image(image: np.ndarray) -> dict[str, object]:
    """Métadonnées d'une image, affichées dans le panneau d'information."""
    image = normalise(image)
    height, width = image.shape[:2]
    channels = 1 if is_gray(image) else image.shape[2]
    return {
        "largeur": int(width),
        "hauteur": int(height),
        "canaux": int(channels),
        "pixels": int(width * height),
        "type": str(image.dtype),
        "octets": int(image.nbytes),
        "min": int(image.min()),
        "max": int(image.max()),
        "moyenne": round(float(image.mean()), 2),
        "ecart_type": round(float(image.std()), 2),
    }
