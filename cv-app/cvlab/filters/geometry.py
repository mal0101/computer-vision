"""Transformations géométriques.

Origine : TP3 §4 (recadrage à la souris), TP4 §1 (redimensionnement, recadrage,
rotation) et TP4 §3 (déformation de perspective).

Toutes les positions et tailles sont exprimées en **pourcentage** des
dimensions de l'image, jamais en pixels. Cela a deux conséquences utiles :
un preset enregistré sur une photo 4000×3000 reste valable sur une capture
webcam 640×480, et les bornes des curseurs n'ont pas besoin d'être recalculées
à chaque changement d'image.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..params import BoolParam, ChoiceParam, FloatParam, IntParam
from ..registry import FilterError, register

#: Méthodes d'interpolation. INTER_AREA est le bon choix en réduction
#: (moyenne des pixels source) ; INTER_CUBIC / LANCZOS4 en agrandissement.
_INTERPOLATIONS = {
    "Plus proche voisin": cv2.INTER_NEAREST,
    "Bilinéaire": cv2.INTER_LINEAR,
    "Bicubique": cv2.INTER_CUBIC,
    "Aire (réduction)": cv2.INTER_AREA,
    "Lanczos4": cv2.INTER_LANCZOS4,
}

_BORDS = {
    "Réplication": cv2.BORDER_REPLICATE,
    "Constante (noir)": cv2.BORDER_CONSTANT,
    "Miroir": cv2.BORDER_REFLECT_101,
    "Enroulement": cv2.BORDER_WRAP,
}


@register(
    id="redimensionner",
    name="Redimensionner",
    family="Géométrie",
    summary="Change la taille de l'image par un pourcentage.",
    theory=(
        "cv2.resize recalcule la grille de pixels. Le choix de "
        "l'interpolation compte : « plus proche voisin » recopie le pixel le "
        "plus proche (rapide, crénelé, seul choix correct pour une image de "
        "labels), « aire » moyenne les pixels source et évite le repliement de "
        "spectre en réduction, « bicubique » et « Lanczos » donnent les "
        "meilleurs agrandissements."
    ),
    tp=("TP4 §1",),
    params=(
        FloatParam(
            name="pourcentage",
            label="Échelle",
            value=100.0,
            min=5.0,
            max=400.0,
            step=1.0,
            unit="%",
            help="100 % = taille d'origine.",
        ),
        ChoiceParam(
            name="interpolation",
            label="Interpolation",
            value="Bilinéaire",
            options=_INTERPOLATIONS,
        ),
    ),
)
def redimensionner(
    image: np.ndarray, pourcentage: float, interpolation: int
) -> tuple[np.ndarray, dict[str, object]]:
    hauteur, largeur = image.shape[:2]
    # max(1, ...) : une échelle de 5 % sur une image de 10 px donnerait 0.
    nouvelle_largeur = max(1, int(round(largeur * pourcentage / 100.0)))
    nouvelle_hauteur = max(1, int(round(hauteur * pourcentage / 100.0)))
    sortie = cv2.resize(
        image, (nouvelle_largeur, nouvelle_hauteur), interpolation=interpolation
    )
    return sortie, {"taille": f"{nouvelle_largeur}×{nouvelle_hauteur} px"}


@register(
    id="recadrer",
    name="Recadrer (crop)",
    family="Géométrie",
    summary="Extrait une région rectangulaire de l'image.",
    theory=(
        "Le recadrage est un simple découpage de tableau NumPy, "
        "``image[y1:y2, x1:x2]`` : aucun pixel n'est recalculé. Attention à "
        "l'ordre des indices — NumPy indexe en (ligne, colonne), donc (y, x), "
        "alors que les fonctions de dessin d'OpenCV prennent des points "
        "(x, y)."
    ),
    tp=("TP3 §4", "TP4 §1"),
    params=(
        FloatParam(name="x", label="Bord gauche", value=10.0, min=0.0, max=99.0,
                   step=0.5, unit="%", group="Position"),
        FloatParam(name="y", label="Bord haut", value=10.0, min=0.0, max=99.0,
                   step=0.5, unit="%", group="Position"),
        FloatParam(name="largeur", label="Largeur", value=80.0, min=1.0, max=100.0,
                   step=0.5, unit="%", group="Taille"),
        FloatParam(name="hauteur", label="Hauteur", value=80.0, min=1.0, max=100.0,
                   step=0.5, unit="%", group="Taille"),
    ),
)
def recadrer(
    image: np.ndarray, x: float, y: float, largeur: float, hauteur: float
) -> tuple[np.ndarray, dict[str, object]]:
    img_h, img_l = image.shape[:2]
    x1 = int(round(img_l * x / 100.0))
    y1 = int(round(img_h * y / 100.0))
    x2 = x1 + int(round(img_l * largeur / 100.0))
    y2 = y1 + int(round(img_h * hauteur / 100.0))

    # On borne à l'image et on garantit au moins 1 pixel : une région vide
    # ferait planter tous les filtres en aval.
    x1 = max(0, min(img_l - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(x1 + 1, min(img_l, x2))
    y2 = max(y1 + 1, min(img_h, y2))

    decoupe = image[y1:y2, x1:x2]
    return np.ascontiguousarray(decoupe), {
        "région": f"({x1}, {y1}) → ({x2}, {y2})",
        "taille": f"{x2 - x1}×{y2 - y1} px",
    }


@register(
    id="rotation",
    name="Rotation",
    family="Géométrie",
    summary="Fait pivoter l'image autour de son centre, avec mise à l'échelle.",
    theory=(
        "cv2.getRotationMatrix2D construit la matrice affine 2×3 "
        "[[s·cosθ, s·sinθ, tx], [−s·sinθ, s·cosθ, ty]], que cv2.warpAffine "
        "applique à chaque pixel. Avec la taille d'origine, les coins sortent "
        "du cadre et sont perdus ; l'option « conserver tout le contenu » "
        "agrandit le cadre et corrige la translation pour que rien ne soit "
        "coupé."
    ),
    tp=("TP4 §1",),
    params=(
        FloatParam(name="angle", label="Angle", value=0.0, min=-180.0, max=180.0,
                   step=1.0, unit="°",
                   help="Positif = sens antihoraire (convention OpenCV)."),
        FloatParam(name="echelle", label="Échelle", value=1.0, min=0.1, max=3.0,
                   step=0.05),
        BoolParam(name="conserver_cadre", label="Conserver tout le contenu",
                  value=False,
                  help="Agrandit le cadre pour qu'aucun coin ne soit coupé."),
        ChoiceParam(name="bord", label="Traitement des bords", value="Constante (noir)",
                    options=_BORDS, advanced=True),
    ),
)
def rotation(
    image: np.ndarray,
    angle: float,
    echelle: float,
    conserver_cadre: bool,
    bord: int,
) -> tuple[np.ndarray, dict[str, object]]:
    hauteur, largeur = image.shape[:2]
    centre = (largeur / 2.0, hauteur / 2.0)
    matrice = cv2.getRotationMatrix2D(centre, float(angle), float(echelle))

    sortie_l, sortie_h = largeur, hauteur
    if conserver_cadre:
        # Dimensions de la boîte englobante de l'image tournée.
        cos = abs(matrice[0, 0])
        sin = abs(matrice[0, 1])
        sortie_l = max(1, int(round(hauteur * sin + largeur * cos)))
        sortie_h = max(1, int(round(hauteur * cos + largeur * sin)))
        # On recentre : la translation compense le décalage du nouveau cadre.
        matrice[0, 2] += sortie_l / 2.0 - centre[0]
        matrice[1, 2] += sortie_h / 2.0 - centre[1]

    sortie = cv2.warpAffine(
        image, matrice, (sortie_l, sortie_h), borderMode=bord
    )
    return sortie, {"taille": f"{sortie_l}×{sortie_h} px"}


@register(
    id="retourner",
    name="Retourner (miroir)",
    family="Géométrie",
    summary="Symétrie horizontale, verticale ou les deux.",
    theory=(
        "cv2.flip(image, code) : code = 1 inverse les colonnes (effet miroir), "
        "0 inverse les lignes, −1 fait les deux (équivalent à une rotation de "
        "180°). C'est l'opération à appliquer à une image de webcam pour "
        "obtenir un rendu « miroir » naturel."
    ),
    tp=("complément",),
    params=(
        ChoiceParam(
            name="sens",
            label="Sens",
            value="Horizontal",
            options={"Horizontal": 1, "Vertical": 0, "Les deux": -1},
        ),
    ),
)
def retourner(image: np.ndarray, sens: int) -> np.ndarray:
    return cv2.flip(image, sens)


@register(
    id="translation",
    name="Translation",
    family="Géométrie",
    summary="Décale l'image horizontalement et verticalement.",
    theory=(
        "Transformation affine la plus simple : matrice [[1, 0, tx], "
        "[0, 1, ty]] passée à cv2.warpAffine. Le traitement des bords décide "
        "de ce qui remplit la zone découverte."
    ),
    tp=("TP4 §1", "complément"),
    params=(
        FloatParam(name="dx", label="Décalage horizontal", value=0.0, min=-100.0,
                   max=100.0, step=1.0, unit="%"),
        FloatParam(name="dy", label="Décalage vertical", value=0.0, min=-100.0,
                   max=100.0, step=1.0, unit="%"),
        ChoiceParam(name="bord", label="Traitement des bords",
                    value="Constante (noir)", options=_BORDS),
    ),
)
def translation(image: np.ndarray, dx: float, dy: float, bord: int) -> np.ndarray:
    hauteur, largeur = image.shape[:2]
    tx = largeur * float(dx) / 100.0
    ty = hauteur * float(dy) / 100.0
    matrice = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, matrice, (largeur, hauteur), borderMode=bord)


@register(
    id="deformation_perspective",
    name="Déformation de perspective",
    family="Géométrie",
    summary="Redresse un quadrilatère en rectangle (4 points source).",
    theory=(
        "On donne quatre points de l'image d'origine et les quatre coins du "
        "rectangle de sortie ; cv2.getPerspectiveTransform résout le système "
        "et renvoie l'homographie 3×3, que cv2.warpPerspective applique. "
        "C'est la correction utilisée pour redresser la photo d'une page prise "
        "de biais (TP4 §3). L'ordre des points compte : haut-gauche, "
        "haut-droit, bas-droit, bas-gauche."
    ),
    tp=("TP4 §3",),
    notes=(
        "Dans l'application de bureau, les quatre points peuvent être "
        "désignés directement à la souris (clic gauche)."
    ),
    params=(
        FloatParam(name="p1x", label="1 · haut-gauche X", value=10.0, min=0.0,
                   max=100.0, step=0.5, unit="%", group="Points source"),
        FloatParam(name="p1y", label="1 · haut-gauche Y", value=10.0, min=0.0,
                   max=100.0, step=0.5, unit="%", group="Points source"),
        FloatParam(name="p2x", label="2 · haut-droit X", value=90.0, min=0.0,
                   max=100.0, step=0.5, unit="%", group="Points source"),
        FloatParam(name="p2y", label="2 · haut-droit Y", value=10.0, min=0.0,
                   max=100.0, step=0.5, unit="%", group="Points source"),
        FloatParam(name="p3x", label="3 · bas-droit X", value=90.0, min=0.0,
                   max=100.0, step=0.5, unit="%", group="Points source"),
        FloatParam(name="p3y", label="3 · bas-droit Y", value=90.0, min=0.0,
                   max=100.0, step=0.5, unit="%", group="Points source"),
        FloatParam(name="p4x", label="4 · bas-gauche X", value=10.0, min=0.0,
                   max=100.0, step=0.5, unit="%", group="Points source"),
        FloatParam(name="p4y", label="4 · bas-gauche Y", value=90.0, min=0.0,
                   max=100.0, step=0.5, unit="%", group="Points source"),
        IntParam(name="largeur_sortie", label="Largeur de sortie", value=480,
                 min=32, max=2048, step=16, unit="px", group="Sortie"),
        IntParam(name="hauteur_sortie", label="Hauteur de sortie", value=640,
                 min=32, max=2048, step=16, unit="px", group="Sortie"),
        ChoiceParam(name="interpolation", label="Interpolation", value="Bilinéaire",
                    options=_INTERPOLATIONS, advanced=True),
    ),
)
def deformation_perspective(
    image: np.ndarray,
    p1x: float, p1y: float,
    p2x: float, p2y: float,
    p3x: float, p3y: float,
    p4x: float, p4y: float,
    largeur_sortie: int,
    hauteur_sortie: int,
    interpolation: int,
) -> tuple[np.ndarray, dict[str, object]]:
    hauteur, largeur = image.shape[:2]
    source = np.float32([
        [largeur * p1x / 100.0, hauteur * p1y / 100.0],
        [largeur * p2x / 100.0, hauteur * p2y / 100.0],
        [largeur * p3x / 100.0, hauteur * p3y / 100.0],
        [largeur * p4x / 100.0, hauteur * p4y / 100.0],
    ])

    # Quatre points alignés (ou confondus) rendent le système singulier.
    aire = cv2.contourArea(source.astype(np.float32))
    if aire < 1.0:
        raise FilterError(
            "Les quatre points source sont alignés ou confondus : "
            "l'homographie est indéterminée. Écartez-les davantage."
        )

    destination = np.float32([
        [0, 0],
        [largeur_sortie - 1, 0],
        [largeur_sortie - 1, hauteur_sortie - 1],
        [0, hauteur_sortie - 1],
    ])
    matrice = cv2.getPerspectiveTransform(source, destination)
    sortie = cv2.warpPerspective(
        image, matrice, (int(largeur_sortie), int(hauteur_sortie)),
        flags=interpolation,
    )
    return sortie, {
        "aire du quadrilatère": f"{aire:.0f} px²",
        "taille": f"{int(largeur_sortie)}×{int(hauteur_sortie)} px",
    }
