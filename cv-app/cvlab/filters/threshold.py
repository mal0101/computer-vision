"""Seuillage (binarisation).

Seuiller, c'est décider pour chaque pixel s'il appartient à l'objet ou au fond.
C'est l'étape qui transforme une image d'intensités en image de régions, et
donc le préalable de toute analyse de formes (morphologie, comptage d'objets,
extraction de contours fermés).

Trois stratégies, par ordre de robustesse croissante :

* **seuil global fixé à la main** — simple, mais il faut le régler pour chaque
  image, et un éclairage inégal le met en échec ;
* **seuil global automatique (Otsu)** — le seuil est calculé depuis
  l'histogramme ; suppose un histogramme à deux bosses ;
* **seuil adaptatif** — un seuil différent par voisinage ; seule méthode qui
  résiste à un éclairage inégal (page photographiée avec une ombre, par ex.).

Le seuillage exige une image à **un seul canal** : « que vaut un pixel
supérieur au seuil » n'a pas de sens pour un triplet BGR. Tous les filtres de
ce module convertissent donc l'entrée en niveaux de gris.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..params import BoolParam, ChoiceParam, IntParam
from ..registry import register

_TYPES = {
    "Binaire": cv2.THRESH_BINARY,
    "Binaire inversé": cv2.THRESH_BINARY_INV,
    "Troncature": cv2.THRESH_TRUNC,
    "Vers zéro": cv2.THRESH_TOZERO,
    "Vers zéro inversé": cv2.THRESH_TOZERO_INV,
}


@register(
    id="seuillage_simple",
    name="Seuillage global",
    family="Seuillage",
    summary="Compare chaque pixel à un seuil fixé par l'utilisateur.",
    theory=(
        "cv2.threshold applique la règle choisie :\n"
        "• Binaire        : f > s ? maxval : 0\n"
        "• Binaire inversé : f > s ? 0 : maxval\n"
        "• Troncature     : f > s ? s : f (plafonnement)\n"
        "• Vers zéro      : f > s ? f : 0\n"
        "Les deux premières donnent une image binaire ; les autres gardent une "
        "partie de l'information d'intensité, ce qui sert à préparer un autre "
        "traitement plutôt qu'à segmenter."
    ),
    tp=("TP7", "complément"),
    input_mode="gray",
    output_mode="gray",
    params=(
        IntParam(name="seuil", label="Seuil", value=127, min=0, max=255),
        IntParam(name="valeur_max", label="Valeur maximale", value=255, min=1,
                 max=255,
                 help="Valeur attribuée aux pixels retenus (255 = blanc)."),
        ChoiceParam(name="type_seuil", label="Type", value="Binaire",
                    options=_TYPES),
    ),
)
def seuillage_simple(
    image: np.ndarray, seuil: int, valeur_max: int, type_seuil: int
) -> tuple[np.ndarray, dict[str, object]]:
    _, sortie = cv2.threshold(image, int(seuil), int(valeur_max), type_seuil)
    taux = float(np.count_nonzero(sortie)) / sortie.size
    return sortie, {"pixels non nuls": f"{taux * 100:.1f} %"}


@register(
    id="seuillage_automatique",
    name="Seuillage automatique (Otsu / triangle)",
    family="Seuillage",
    summary="Calcule le seuil optimal depuis l'histogramme.",
    theory=(
        "La méthode d'Otsu essaie tous les seuils possibles et retient celui "
        "qui minimise la variance intra-classe (de façon équivalente : qui "
        "maximise la variance inter-classe). Elle suppose un histogramme "
        "**bimodal** — une bosse pour le fond, une pour les objets ; sur une "
        "image à éclairage inégal, le seuil trouvé est mauvais et c'est le "
        "seuillage adaptatif qu'il faut utiliser. La méthode du triangle "
        "convient mieux quand une seule classe domine largement l'image.\n"
        "Un lissage gaussien préalable est recommandé : il resserre les deux "
        "bosses de l'histogramme et stabilise le seuil trouvé."
    ),
    tp=("complément",),
    input_mode="gray",
    output_mode="gray",
    params=(
        ChoiceParam(name="methode", label="Méthode", value="Otsu",
                    options={"Otsu": cv2.THRESH_OTSU,
                             "Triangle": cv2.THRESH_TRIANGLE}),
        ChoiceParam(name="type_seuil", label="Type", value="Binaire",
                    options={"Binaire": cv2.THRESH_BINARY,
                             "Binaire inversé": cv2.THRESH_BINARY_INV}),
        IntParam(name="flou_prealable", label="Flou gaussien préalable", value=5,
                 min=0, max=15,
                 help="0 = aucun. Un léger flou rend le seuil trouvé plus stable."),
    ),
)
def seuillage_automatique(
    image: np.ndarray, methode: int, type_seuil: int, flou_prealable: int
) -> tuple[np.ndarray, dict[str, object]]:
    source = image
    taille = int(flou_prealable)
    if taille > 0:
        if taille % 2 == 0:
            taille += 1
        source = cv2.GaussianBlur(image, (taille, taille), 0)
    seuil, sortie = cv2.threshold(source, 0, 255, type_seuil | methode)
    taux = float(np.count_nonzero(sortie)) / sortie.size
    return sortie, {
        "seuil calculé": int(seuil),
        "pixels non nuls": f"{taux * 100:.1f} %",
    }


@register(
    id="seuillage_adaptatif",
    name="Seuillage adaptatif",
    family="Seuillage",
    summary="Un seuil local par voisinage, robuste à l'éclairage inégal.",
    theory=(
        "Le seuil du pixel (x, y) est calculé sur son voisinage de taille "
        "``taille_bloc`` : moyenne arithmétique (MEAN_C) ou moyenne pondérée "
        "par une gaussienne (GAUSSIAN_C), moins une constante C. Comparer un "
        "pixel à la moyenne de ses voisins revient à détecter s'il est plus "
        "sombre que son entourage immédiat — d'où l'excellent résultat sur du "
        "texte photographié avec une ombre.\n"
        "Réglage : ``taille_bloc`` doit être plus grand que les détails à "
        "isoler (sinon l'intérieur des traits épais se vide) ; C élimine le "
        "bruit des zones uniformes, où la moyenne locale est presque égale au "
        "pixel lui-même."
    ),
    tp=("complément",),
    input_mode="gray",
    output_mode="gray",
    params=(
        ChoiceParam(name="methode", label="Méthode de moyenne", value="Gaussienne",
                    options={"Moyenne": cv2.ADAPTIVE_THRESH_MEAN_C,
                             "Gaussienne": cv2.ADAPTIVE_THRESH_GAUSSIAN_C}),
        ChoiceParam(name="type_seuil", label="Type", value="Binaire",
                    options={"Binaire": cv2.THRESH_BINARY,
                             "Binaire inversé": cv2.THRESH_BINARY_INV}),
        IntParam(name="taille_bloc", label="Taille du voisinage", value=11,
                 min=3, max=99, odd_only=True,
                 help="Impair et ≥ 3. Doit excéder la taille des détails à isoler."),
        IntParam(name="constante", label="Constante soustraite (C)", value=2,
                 min=-30, max=30,
                 help="Augmenter C nettoie les zones uniformes, mais efface les détails ténus."),
        IntParam(name="valeur_max", label="Valeur maximale", value=255, min=1,
                 max=255, advanced=True),
    ),
)
def seuillage_adaptatif(
    image: np.ndarray,
    methode: int,
    type_seuil: int,
    taille_bloc: int,
    constante: int,
    valeur_max: int,
) -> tuple[np.ndarray, dict[str, object]]:
    sortie = cv2.adaptiveThreshold(
        image, int(valeur_max), methode, type_seuil,
        int(taille_bloc), float(constante),
    )
    taux = float(np.count_nonzero(sortie)) / sortie.size
    return sortie, {"pixels non nuls": f"{taux * 100:.1f} %"}


@register(
    id="seuillage_par_bande",
    name="Seuillage par bande d'intensité",
    family="Seuillage",
    summary="Garde les pixels dont l'intensité est dans un intervalle.",
    theory=(
        "cv2.inRange sur un seul canal : le masque vaut 255 si "
        "``min ≤ f ≤ max``. Utile quand l'objet n'est ni le plus clair ni le "
        "plus sombre de l'image — cas qu'un seuil unique ne peut pas traiter."
    ),
    tp=("complément",),
    input_mode="gray",
    output_mode="gray",
    params=(
        IntParam(name="minimum", label="Intensité minimale", value=80, min=0,
                 max=255),
        IntParam(name="maximum", label="Intensité maximale", value=200, min=0,
                 max=255),
        BoolParam(name="inverser", label="Inverser le masque", value=False),
    ),
)
def seuillage_par_bande(
    image: np.ndarray, minimum: int, maximum: int, inverser: bool
) -> tuple[np.ndarray, dict[str, object]]:
    bas, haut = sorted((int(minimum), int(maximum)))
    masque = cv2.inRange(image, bas, haut)
    if inverser:
        masque = cv2.bitwise_not(masque)
    taux = float(np.count_nonzero(masque)) / masque.size
    return masque, {
        "bande": f"[{bas}, {haut}]",
        "pixels retenus": f"{taux * 100:.1f} %",
    }
