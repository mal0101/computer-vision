"""Morphologie mathématique.

Ces opérations raisonnent sur la **forme** des régions claires d'une image, pas
sur leurs intensités. Un petit motif appelé **élément structurant** est promené
sur l'image ; selon qu'on demande le minimum ou le maximum sous ce motif, les
régions claires se rétractent ou s'étendent :

* **érosion**   : minimum local — les régions claires maigrissent, les petits
  points blancs isolés disparaissent ;
* **dilatation**: maximum local — les régions claires grossissent, les trous et
  les fissures se bouchent ;
* **ouverture** = érosion puis dilatation — supprime le bruit clair sans
  changer notablement la taille des grands objets ;
* **fermeture** = dilatation puis érosion — bouche les trous sombres à
  l'intérieur des objets ;
* **gradient**  = dilatation − érosion — ne laisse que le contour des objets ;
* **chapeau haut de forme** = image − ouverture — isole les petits détails
  clairs, donc corrige un fond inégal ;
* **chapeau noir** = fermeture − image — isole les petits détails sombres.

Ces opérations s'appliquent surtout à une image **binaire** (issue d'un
seuillage), mais OpenCV les accepte aussi en niveaux de gris, où elles agissent
comme des filtres de rang (min / max au lieu de la médiane).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..params import ChoiceParam, IntParam
from ..registry import register

_FORMES = {
    "Rectangle": cv2.MORPH_RECT,
    "Ellipse": cv2.MORPH_ELLIPSE,
    "Croix": cv2.MORPH_CROSS,
}

_OPERATIONS = {
    "Érosion": cv2.MORPH_ERODE,
    "Dilatation": cv2.MORPH_DILATE,
    "Ouverture": cv2.MORPH_OPEN,
    "Fermeture": cv2.MORPH_CLOSE,
    "Gradient morphologique": cv2.MORPH_GRADIENT,
    "Chapeau haut de forme (top-hat)": cv2.MORPH_TOPHAT,
    "Chapeau noir (black-hat)": cv2.MORPH_BLACKHAT,
}


@register(
    id="morphologie",
    name="Opération morphologique",
    family="Morphologie",
    summary="Érosion, dilatation, ouverture, fermeture, gradient, chapeaux.",
    theory=(
        "cv2.morphologyEx applique l'opération choisie avec l'élément "
        "structurant fourni par cv2.getStructuringElement. La **forme** de cet "
        "élément compte : un rectangle produit des coins carrés, une ellipse "
        "respecte mieux les objets ronds, une croix ne touche que les 4 "
        "voisins directs. La **taille** fixe l'échelle des détails affectés — "
        "c'est le réglage le plus déterminant. Les **itérations** répètent "
        "l'opération : n itérations d'un élément 3×3 équivalent "
        "approximativement à un élément (2n+1)×(2n+1), en plus rapide."
    ),
    tp=("complément",),
    params=(
        ChoiceParam(name="operation", label="Opération", value="Ouverture",
                    options=_OPERATIONS),
        ChoiceParam(name="forme", label="Forme de l'élément structurant",
                    value="Ellipse", options=_FORMES),
        IntParam(name="taille", label="Taille de l'élément", value=5, min=1,
                 max=31, odd_only=True, unit="px"),
        IntParam(name="iterations", label="Itérations", value=1, min=1, max=10),
    ),
)
def morphologie(
    image: np.ndarray, operation: int, forme: int, taille: int, iterations: int
) -> tuple[np.ndarray, dict[str, object]]:
    noyau = cv2.getStructuringElement(forme, (int(taille), int(taille)))
    sortie = cv2.morphologyEx(
        image, operation, noyau, iterations=int(iterations)
    )
    return sortie, {
        "élément structurant": f"{int(taille)}×{int(taille)}",
        "pixels actifs dans l'élément": int(np.count_nonzero(noyau)),
    }


@register(
    id="squelette_contour_morpho",
    name="Contour par gradient morphologique",
    family="Morphologie",
    summary="Trace le bord des régions claires (dilatation − érosion).",
    theory=(
        "Avec un élément 3×3, la dilatation moins l'érosion ne laisse qu'un "
        "liseré d'un pixel de part et d'autre de chaque frontière. Comparé à "
        "Canny, ce détecteur n'a pas de seuil à régler et donne des contours "
        "fermés, mais il est beaucoup plus sensible au bruit : on l'applique "
        "donc en général sur une image déjà seuillée."
    ),
    tp=("complément",),
    params=(
        ChoiceParam(name="forme", label="Forme de l'élément", value="Rectangle",
                    options=_FORMES),
        IntParam(name="taille", label="Épaisseur du contour", value=3, min=3,
                 max=15, odd_only=True,
                 help="Taille de l'élément structurant : fixe l'épaisseur du liseré."),
    ),
)
def squelette_contour_morpho(
    image: np.ndarray, forme: int, taille: int
) -> np.ndarray:
    noyau = cv2.getStructuringElement(forme, (int(taille), int(taille)))
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, noyau)
