"""Fusion de deux images et opérations binaires (TP4 §2 et §5).

Ces filtres ont besoin d'une **deuxième image** (``needs_reference=True``).
L'application la redimensionne automatiquement à la taille de l'image courante
avant l'appel — c'est ce que faisait ``TraitementImage.fusion()`` du TP4, et
c'est obligatoire : ``cv2.addWeighted`` et les opérations binaires exigent deux
opérandes de même taille et de même nombre de canaux.
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import images as im
from ..params import ChoiceParam, FloatParam, IntParam
from ..registry import register


@register(
    id="fusion",
    name="Fusion pondérée",
    family="Fusion & opérations binaires",
    summary="Mélange deux images : g = α·f₁ + (1−α)·f₂ + γ.",
    theory=(
        "cv2.addWeighted réalise exactement la formule du TP4 §2. α est le "
        "poids de l'image courante : α = 1 ne garde qu'elle, α = 0 ne garde "
        "que l'image de référence, α = 0,5 donne une surimpression équilibrée. "
        "La somme des poids vaut 1, donc la luminosité moyenne est préservée ; "
        "γ permet d'ajouter un décalage constant si le résultat est trop sombre."
    ),
    tp=("TP4 §2",),
    needs_reference=True,
    params=(
        FloatParam(name="alpha", label="Poids de l'image courante (α)",
                   value=0.5, min=0.0, max=1.0, step=0.01,
                   help="1 = image courante seule, 0 = image de référence seule."),
        IntParam(name="gamma", label="Décalage (γ)", value=0, min=-128, max=128,
                 advanced=True),
    ),
)
def fusion(
    image: np.ndarray, reference: np.ndarray, alpha: float, gamma: int
) -> np.ndarray:
    premiere, seconde = im.match_channels(image, reference)
    return cv2.addWeighted(
        premiere, float(alpha), seconde, 1.0 - float(alpha), float(gamma)
    )


@register(
    id="operation_binaire",
    name="Opération binaire",
    family="Fusion & opérations binaires",
    summary="ET, OU, OU exclusif ou NON, bit à bit, entre deux images.",
    theory=(
        "Les opérations s'appliquent bit à bit sur les octets des deux images :\n"
        "• ET  : ne garde que ce qui est clair dans les deux (intersection) ;\n"
        "• OU  : garde ce qui est clair dans au moins une (union) ;\n"
        "• OU exclusif : met en évidence les différences, et vaut 0 là où les "
        "deux images sont identiques ;\n"
        "• NON : complément de l'image courante, la référence est ignorée.\n"
        "Elles servent surtout avec une image binaire en guise de masque : "
        "``bitwise_and(image, image, mask=m)`` est la façon canonique de ne "
        "garder qu'une région (cf. le filtre « Masque de couleur »)."
    ),
    tp=("TP4 §5",),
    needs_reference=True,
    params=(
        ChoiceParam(
            name="operation",
            label="Opération",
            value="ET (AND)",
            options={
                "ET (AND)": "and",
                "OU (OR)": "or",
                "OU exclusif (XOR)": "xor",
                "NON (NOT, image courante)": "not",
                "Différence absolue": "absdiff",
            },
        ),
    ),
)
def operation_binaire(
    image: np.ndarray, reference: np.ndarray, operation: str
) -> np.ndarray:
    if operation == "not":
        return cv2.bitwise_not(image)
    premiere, seconde = im.match_channels(image, reference)
    if operation == "and":
        return cv2.bitwise_and(premiere, seconde)
    if operation == "or":
        return cv2.bitwise_or(premiere, seconde)
    if operation == "xor":
        return cv2.bitwise_xor(premiere, seconde)
    return cv2.absdiff(premiere, seconde)


@register(
    id="masque_par_reference",
    name="Masquage par une seconde image",
    family="Fusion & opérations binaires",
    summary="Garde l'image courante là où la référence est claire.",
    theory=(
        "La deuxième image est convertie en niveaux de gris puis seuillée pour "
        "produire un masque binaire ; ``cv2.bitwise_and(..., mask=m)`` ne "
        "conserve l'image courante que là où le masque est non nul. C'est la "
        "brique de base de l'incrustation : on combine ensuite la zone gardée "
        "avec un autre fond (TP4 §5)."
    ),
    tp=("TP4 §5",),
    needs_reference=True,
    params=(
        IntParam(name="seuil", label="Seuil du masque", value=127, min=0,
                 max=255,
                 help="La référence est binarisée à ce seuil pour servir de masque."),
        ChoiceParam(name="sens", label="Zone conservée", value="Zones claires",
                    options={"Zones claires": "clair", "Zones sombres": "sombre"}),
    ),
)
def masque_par_reference(
    image: np.ndarray, reference: np.ndarray, seuil: int, sens: str
) -> tuple[np.ndarray, dict[str, object]]:
    gris = im.to_gray(reference)
    mode = cv2.THRESH_BINARY if sens == "clair" else cv2.THRESH_BINARY_INV
    _, masque = cv2.threshold(gris, int(seuil), 255, mode)
    sortie = cv2.bitwise_and(image, image, mask=masque)
    taux = float(np.count_nonzero(masque)) / masque.size
    return sortie, {"surface conservée": f"{taux * 100:.1f} %"}
