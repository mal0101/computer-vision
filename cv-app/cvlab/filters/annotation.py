"""Annotations : formes et texte (TP2 §2, TP3).

Ces filtres n'analysent rien, ils **dessinent** par-dessus l'image. Ils
réutilisent les fonctions enveloppantes de :mod:`cvlab.drawing`, écrites au
TP2, et servent à produire une image légendée à partager ou à enregistrer.

Les positions sont en pourcentage des dimensions de l'image, pour qu'une
annotation reste au même endroit si la source change de résolution.
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import drawing as ds
from .. import images as im
from ..params import BoolParam, ChoiceParam, ColorParam, FloatParam, IntParam, TextParam
from ..registry import register


@register(
    id="annoter_texte",
    name="Texte",
    family="Annotation",
    summary="Écrit un texte sur l'image.",
    theory=(
        "cv2.putText ne sait dessiner que les polices vectorielles Hershey "
        "embarquées dans OpenCV — pas de police système, et pas de caractères "
        "hors Latin-1 (les accents passent, pas les alphabets non latins). Le "
        "point ``org`` donné est le coin **bas-gauche** de la première ligne, "
        "pas le coin haut-gauche : c'est la cause classique du texte qui "
        "« disparaît » quand on lui donne y = 0."
    ),
    tp=("TP1 §2", "TP2 §2.3"),
    params=(
        TextParam(name="texte", label="Texte", value="Vision par ordinateur"),
        FloatParam(name="x", label="Position X", value=5.0, min=0.0, max=100.0,
                   step=0.5, unit="%", group="Position"),
        FloatParam(name="y", label="Position Y", value=10.0, min=0.0, max=100.0,
                   step=0.5, unit="%", group="Position"),
        FloatParam(name="taille", label="Échelle de police", value=1.0, min=0.2,
                   max=6.0, step=0.1),
        IntParam(name="epaisseur", label="Épaisseur", value=2, min=1, max=20),
        ColorParam(name="couleur", label="Couleur", value=(0, 255, 0)),
        ChoiceParam(name="police", label="Police", value="Complex",
                    options=ds.POLICES),
        BoolParam(name="fond", label="Fond opaque derrière le texte", value=False,
                  help="Garantit la lisibilité sur une image claire ou chargée."),
    ),
)
def annoter_texte(
    image: np.ndarray,
    texte: str,
    x: float,
    y: float,
    taille: float,
    epaisseur: int,
    couleur: tuple[int, int, int],
    police: int,
    fond: bool,
) -> np.ndarray:
    sortie = im.to_bgr(image)
    hauteur, largeur = sortie.shape[:2]
    position = (int(largeur * x / 100.0), int(hauteur * y / 100.0))
    if fond:
        return ds.dess_texte_encadre(
            sortie, texte, position, taille, couleur, (0, 0, 0), epaisseur,
            police=police,
        )
    return ds.dess_text(sortie, texte, position, taille, couleur, epaisseur, police)


@register(
    id="annoter_forme",
    name="Forme géométrique",
    family="Annotation",
    summary="Trace une ligne, un rectangle, un cercle ou une ellipse.",
    theory=(
        "Les primitives cv2.line, cv2.rectangle, cv2.circle et cv2.ellipse "
        "partagent la même signature de fin : couleur, épaisseur, type de "
        "ligne. Une épaisseur de −1 (``cv2.FILLED``) remplit la forme. Le type "
        "de ligne ``LINE_AA`` active l'anticrénelage : les bords obliques sont "
        "lissés, ce qui est bien plus lisible à l'écran que le tracé 8-connexe "
        "(TP2 §2.1)."
    ),
    tp=("TP2 §2", "TP3 §2-3"),
    params=(
        ChoiceParam(
            name="forme",
            label="Forme",
            value="Rectangle",
            options={"Ligne": "ligne", "Rectangle": "rectangle",
                     "Cercle": "cercle", "Ellipse": "ellipse"},
        ),
        FloatParam(name="x1", label="Point 1 · X", value=20.0, min=0.0, max=100.0,
                   step=0.5, unit="%", group="Géométrie"),
        FloatParam(name="y1", label="Point 1 · Y", value=20.0, min=0.0, max=100.0,
                   step=0.5, unit="%", group="Géométrie"),
        FloatParam(name="x2", label="Point 2 · X", value=80.0, min=0.0, max=100.0,
                   step=0.5, unit="%", group="Géométrie",
                   help="Pour un cercle ou une ellipse : définit les rayons depuis le point 1."),
        FloatParam(name="y2", label="Point 2 · Y", value=80.0, min=0.0, max=100.0,
                   step=0.5, unit="%", group="Géométrie"),
        FloatParam(name="angle", label="Angle (ellipse)", value=0.0, min=-180.0,
                   max=180.0, step=1.0, unit="°", group="Ellipse"),
        FloatParam(name="arc_debut", label="Début d'arc", value=0.0, min=0.0,
                   max=360.0, step=5.0, unit="°", group="Ellipse"),
        FloatParam(name="arc_fin", label="Fin d'arc", value=360.0, min=0.0,
                   max=360.0, step=5.0, unit="°", group="Ellipse"),
        ColorParam(name="couleur", label="Couleur", value=(0, 0, 255)),
        IntParam(name="epaisseur", label="Épaisseur", value=2, min=1, max=40),
        BoolParam(name="rempli", label="Forme pleine", value=False),
        ChoiceParam(name="type_ligne", label="Type de ligne", value="Anticrénelage",
                    options=ds.TYPES_LIGNE, advanced=True),
    ),
)
def annoter_forme(
    image: np.ndarray,
    forme: str,
    x1: float, y1: float, x2: float, y2: float,
    angle: float, arc_debut: float, arc_fin: float,
    couleur: tuple[int, int, int],
    epaisseur: int,
    rempli: bool,
    type_ligne: int,
) -> np.ndarray:
    sortie = im.to_bgr(image)
    hauteur, largeur = sortie.shape[:2]
    pt1 = (int(largeur * x1 / 100.0), int(hauteur * y1 / 100.0))
    pt2 = (int(largeur * x2 / 100.0), int(hauteur * y2 / 100.0))
    trait = cv2.FILLED if rempli else int(epaisseur)

    if forme == "ligne":
        return ds.dess_ligne(sortie, pt1, pt2, couleur, max(1, int(epaisseur)),
                             type_ligne)
    if forme == "rectangle":
        return ds.dess_rectangle(sortie, pt1, pt2, couleur, trait, type_ligne)
    if forme == "cercle":
        rayon = int(round(((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2) ** 0.5))
        return ds.dess_cercle(sortie, pt1, rayon, couleur, trait, type_ligne)
    axes = (abs(pt2[0] - pt1[0]), abs(pt2[1] - pt1[1]))
    return ds.dess_ellipse(sortie, pt1, axes, angle, arc_debut, arc_fin,
                           couleur, trait, type_ligne)


@register(
    id="grille_reperes",
    name="Grille de repères",
    family="Annotation",
    summary="Superpose une grille graduée en pixels.",
    theory=(
        "Aide de lecture pour retrouver des coordonnées sur l'image. Elle "
        "rappelle la convention du TP2 §1.1 : l'origine (0, 0) est en haut à "
        "gauche, l'axe des x va vers la droite (colonnes) et l'axe des y vers "
        "le bas (lignes) — l'inverse du repère mathématique habituel."
    ),
    tp=("TP2 §1.1", "TP3 §1"),
    output_mode="color",
    params=(
        IntParam(name="pas", label="Pas de la grille", value=50, min=10, max=500,
                 step=10, unit="px"),
        ColorParam(name="couleur", label="Couleur", value=(80, 200, 255)),
        IntParam(name="epaisseur", label="Épaisseur", value=1, min=1, max=4),
        BoolParam(name="etiquettes", label="Afficher les coordonnées", value=True),
    ),
)
def grille_reperes(
    image: np.ndarray,
    pas: int,
    couleur: tuple[int, int, int],
    epaisseur: int,
    etiquettes: bool,
) -> np.ndarray:
    sortie = im.to_bgr(image)
    hauteur, largeur = sortie.shape[:2]
    pas = max(10, int(pas))

    for x in range(0, largeur, pas):
        sortie = ds.dess_ligne(sortie, (x, 0), (x, hauteur), couleur,
                               int(epaisseur), cv2.LINE_8)
        if etiquettes and x > 0:
            sortie = ds.dess_text(sortie, str(x), (x + 2, 12), 0.35, couleur, 1,
                                  cv2.FONT_HERSHEY_SIMPLEX)
    for y in range(0, hauteur, pas):
        sortie = ds.dess_ligne(sortie, (0, y), (largeur, y), couleur,
                               int(epaisseur), cv2.LINE_8)
        if etiquettes and y > 0:
            sortie = ds.dess_text(sortie, str(y), (2, y - 3), 0.35, couleur, 1,
                                  cv2.FONT_HERSHEY_SIMPLEX)
    return sortie
