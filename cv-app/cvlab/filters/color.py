"""Couleur, espaces colorimétriques, luminosité et contraste.

Origine : TP1 §3.2 (conversion en niveaux de gris), TP2 §1 (pixels, canaux BGR
et HSV), TP3 §5 (réglages de la webcam par trackbars), TP4 §2-3 (luminosité,
contraste, teinte, saturation, valeur).
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import images as im
from ..params import BoolParam, ChoiceParam, FloatParam, IntParam
from ..registry import register

# ---------------------------------------------------------------------------
# Conversions et canaux
# ---------------------------------------------------------------------------


@register(
    id="niveaux_de_gris",
    name="Niveaux de gris",
    family="Couleur & espaces colorimétriques",
    summary="Convertit l'image couleur en une image à un seul canal.",
    theory=(
        "cv2.cvtColor(..., COLOR_BGR2GRAY) applique la pondération de la "
        "luminance perçue : Y = 0.299·R + 0.587·V + 0.114·B. Le vert pèse le "
        "plus car l'œil y est le plus sensible ; une simple moyenne des trois "
        "canaux donnerait un résultat plus plat."
    ),
    tp=("TP1 §3.2",),
    output_mode="gray",
)
def niveaux_de_gris(image: np.ndarray) -> np.ndarray:
    return im.to_gray(image)


@register(
    id="canal_bgr",
    name="Canal BGR",
    family="Couleur & espaces colorimétriques",
    summary="Isole un canal bleu, vert ou rouge.",
    theory=(
        "Une image couleur est un empilement de trois plans d'intensité. En "
        "mode « couleur primitive » on met à zéro les deux autres plans "
        "(l'image reste BGR) ; en mode « niveaux de gris » on affiche le plan "
        "seul, ce qui montre que, pris isolément, un canal ne porte aucune "
        "couleur (TP2 §1.2)."
    ),
    tp=("TP2 §1.2",),
    input_mode="color",
    params=(
        ChoiceParam(
            name="canal",
            label="Canal",
            value="Rouge",
            options={"Bleu": 0, "Vert": 1, "Rouge": 2},
            help="Indice du plan à conserver dans le tableau BGR.",
        ),
        ChoiceParam(
            name="mode",
            label="Rendu",
            value="Couleur primitive",
            options={"Couleur primitive": "couleur", "Niveaux de gris": "gris"},
            help="Garder les trois canaux (deux mis à zéro) ou n'afficher que le plan.",
        ),
    ),
)
def canal_bgr(image: np.ndarray, canal: int, mode: str) -> np.ndarray:
    plan = image[:, :, canal]
    if mode == "gris":
        return plan
    sortie = np.zeros_like(image)
    sortie[:, :, canal] = plan
    return sortie


@register(
    id="canal_hsv",
    name="Canal HSV",
    family="Couleur & espaces colorimétriques",
    summary="Affiche le canal teinte, saturation ou valeur.",
    theory=(
        "L'espace HSV décrit une couleur par sa teinte (angle sur la roue "
        "chromatique), sa saturation (pureté) et sa valeur (luminosité). "
        "OpenCV stocke ces trois canaux sur 8 bits : la teinte est donc "
        "divisée par deux et vaut 0-179, tandis que saturation et valeur "
        "occupent toute la plage 0-255 (TP2 §1.3)."
    ),
    tp=("TP2 §1.3",),
    input_mode="color",
    output_mode="gray",
    params=(
        ChoiceParam(
            name="canal",
            label="Canal",
            value="Teinte (H)",
            options={"Teinte (H)": 0, "Saturation (S)": 1, "Valeur (V)": 2},
        ),
        BoolParam(
            name="etirer",
            label="Étirer le contraste",
            value=False,
            help=(
                "Ramène le canal sur 0-255. Utile pour la teinte, dont la "
                "plage réelle 0-179 rend l'affichage brut très sombre."
            ),
        ),
    ),
)
def canal_hsv(image: np.ndarray, canal: int, etirer: bool) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    plan = hsv[:, :, canal]
    if etirer:
        plan = cv2.normalize(plan, None, 0, 255, cv2.NORM_MINMAX)
    return plan


@register(
    id="ajuster_hsv",
    name="Teinte / saturation / valeur",
    family="Couleur & espaces colorimétriques",
    summary="Décale la teinte et multiplie saturation et valeur.",
    theory=(
        "On passe en HSV, on modifie chaque canal séparément puis on revient "
        "en BGR. Le décalage de teinte est circulaire (modulo 180 chez "
        "OpenCV) : au-delà de 179 on revient au rouge. Saturation et valeur "
        "sont des gains multiplicatifs, bornés à 255 pour éviter le "
        "repliement du type uint8."
    ),
    tp=("TP3 §5", "TP4 §3"),
    input_mode="color",
    output_mode="color",
    params=(
        IntParam(
            name="decalage_teinte",
            label="Décalage de teinte",
            value=0,
            min=-179,
            max=179,
            unit="(0-179 chez OpenCV)",
            help="Rotation sur la roue chromatique : change la famille de couleurs.",
        ),
        FloatParam(
            name="gain_saturation",
            label="Gain de saturation",
            value=1.0,
            min=0.0,
            max=3.0,
            step=0.05,
            help="0 = image grise, 1 = inchangé, >1 = couleurs plus vives.",
        ),
        FloatParam(
            name="gain_valeur",
            label="Gain de valeur (luminosité)",
            value=1.0,
            min=0.0,
            max=3.0,
            step=0.05,
            help="0 = image noire, 1 = inchangé, >1 = image plus claire.",
        ),
    ),
)
def ajuster_hsv(
    image: np.ndarray,
    decalage_teinte: int,
    gain_saturation: float,
    gain_valeur: float,
) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    teinte, saturation, valeur = cv2.split(hsv)

    # La teinte est cyclique : l'arithmétique se fait modulo 180.
    teinte = ((teinte.astype(np.int16) + int(decalage_teinte)) % 180).astype(np.uint8)
    saturation = np.clip(
        saturation.astype(np.float32) * float(gain_saturation), 0, 255
    ).astype(np.uint8)
    valeur = np.clip(
        valeur.astype(np.float32) * float(gain_valeur), 0, 255
    ).astype(np.uint8)

    return cv2.cvtColor(cv2.merge((teinte, saturation, valeur)), cv2.COLOR_HSV2BGR)


@register(
    id="masque_couleur_hsv",
    name="Masque de couleur (HSV)",
    family="Couleur & espaces colorimétriques",
    summary="Sélectionne les pixels dont la couleur est dans un intervalle HSV.",
    theory=(
        "cv2.inRange construit un masque binaire : 255 là où les trois canaux "
        "sont dans l'intervalle, 0 ailleurs. On travaille en HSV et non en BGR "
        "car la teinte varie peu avec l'éclairage, alors que les trois "
        "composantes BGR varient toutes ensemble. Si la teinte minimale est "
        "supérieure à la maximale, l'intervalle traverse le rouge (0/179) : "
        "deux masques sont alors combinés."
    ),
    tp=("TP2 §1.3", "complément"),
    input_mode="color",
    params=(
        IntParam(name="teinte_min", label="Teinte min", value=0, min=0, max=179,
                 group="Teinte"),
        IntParam(name="teinte_max", label="Teinte max", value=179, min=0, max=179,
                 group="Teinte"),
        IntParam(name="saturation_min", label="Saturation min", value=60, min=0,
                 max=255, group="Saturation"),
        IntParam(name="saturation_max", label="Saturation max", value=255, min=0,
                 max=255, group="Saturation"),
        IntParam(name="valeur_min", label="Valeur min", value=60, min=0, max=255,
                 group="Valeur"),
        IntParam(name="valeur_max", label="Valeur max", value=255, min=0, max=255,
                 group="Valeur"),
        ChoiceParam(
            name="rendu",
            label="Rendu",
            value="Segmentation",
            options={
                "Segmentation": "segment",
                "Masque binaire": "masque",
                "Fond en gris": "fond_gris",
            },
            help="Garder la zone en couleur, afficher le masque, ou désaturer le reste.",
        ),
    ),
)
def masque_couleur_hsv(
    image: np.ndarray,
    teinte_min: int,
    teinte_max: int,
    saturation_min: int,
    saturation_max: int,
    valeur_min: int,
    valeur_max: int,
    rendu: str,
) -> tuple[np.ndarray, dict[str, object]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s_lo, s_hi = sorted((saturation_min, saturation_max))
    v_lo, v_hi = sorted((valeur_min, valeur_max))

    if teinte_min <= teinte_max:
        masque = cv2.inRange(
            hsv, (teinte_min, s_lo, v_lo), (teinte_max, s_hi, v_hi)
        )
    else:
        # Intervalle à cheval sur 0 (rouges) : union de deux intervalles.
        bas = cv2.inRange(hsv, (0, s_lo, v_lo), (teinte_max, s_hi, v_hi))
        haut = cv2.inRange(hsv, (teinte_min, s_lo, v_lo), (179, s_hi, v_hi))
        masque = cv2.bitwise_or(bas, haut)

    taux = float(np.count_nonzero(masque)) / masque.size
    infos = {"pixels sélectionnés": f"{taux * 100:.1f} %"}

    if rendu == "masque":
        return masque, infos
    if rendu == "segment":
        return cv2.bitwise_and(image, image, mask=masque), infos
    fond = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    inverse = cv2.bitwise_not(masque)
    return cv2.add(
        cv2.bitwise_and(image, image, mask=masque),
        cv2.bitwise_and(fond, fond, mask=inverse),
    ), infos


@register(
    id="fausses_couleurs",
    name="Fausses couleurs (colormap)",
    family="Couleur & espaces colorimétriques",
    summary="Applique une palette de couleurs à une image d'intensité.",
    theory=(
        "cv2.applyColorMap remplace chaque niveau de gris par une couleur "
        "d'une table de 256 entrées. Cela n'ajoute aucune information : c'est "
        "un outil de lecture, pratique pour repérer des variations faibles "
        "qu'un dégradé de gris masque (cartes de gradient, de profondeur, de "
        "chaleur)."
    ),
    tp=("complément",),
    input_mode="gray",
    output_mode="color",
    params=(
        ChoiceParam(
            name="palette",
            label="Palette",
            value="Viridis",
            options={
                "Viridis": cv2.COLORMAP_VIRIDIS,
                "Turbo": cv2.COLORMAP_TURBO,
                "Jet": cv2.COLORMAP_JET,
                "Inferno": cv2.COLORMAP_INFERNO,
                "Magma": cv2.COLORMAP_MAGMA,
                "Plasma": cv2.COLORMAP_PLASMA,
                "Cividis": cv2.COLORMAP_CIVIDIS,
                "Chaleur (Hot)": cv2.COLORMAP_HOT,
                "Os (Bone)": cv2.COLORMAP_BONE,
                "Océan": cv2.COLORMAP_OCEAN,
                "Arc-en-ciel": cv2.COLORMAP_RAINBOW,
            },
        ),
    ),
)
def fausses_couleurs(image: np.ndarray, palette: int) -> np.ndarray:
    return cv2.applyColorMap(image, palette)


# ---------------------------------------------------------------------------
# Luminosité et contraste
# ---------------------------------------------------------------------------


@register(
    id="luminosite_contraste",
    name="Luminosité & contraste",
    family="Luminosité & contraste",
    summary="Transformation linéaire des intensités : g = α·f + β.",
    theory=(
        "α (gain) multiplie l'écart des intensités, donc le contraste ; "
        "β (offset) décale toutes les intensités, donc la luminosité. "
        "cv2.convertScaleAbs calcule |α·f + β| puis borne à 255, ce qui évite "
        "le repliement du uint8 (sans cela, 250 + 10 donnerait 4)."
    ),
    tp=("TP3 §5", "TP4 §2"),
    params=(
        FloatParam(
            name="alpha",
            label="Contraste (α)",
            value=1.0,
            min=0.0,
            max=3.0,
            step=0.05,
            help="1 = inchangé. <1 écrase le contraste, >1 l'accentue.",
        ),
        IntParam(
            name="beta",
            label="Luminosité (β)",
            value=0,
            min=-128,
            max=128,
            help="Ajouté à chaque pixel après multiplication par α.",
        ),
    ),
)
def luminosite_contraste(image: np.ndarray, alpha: float, beta: int) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=float(alpha), beta=int(beta))


@register(
    id="correction_gamma",
    name="Correction gamma",
    family="Luminosité & contraste",
    summary="Transformation non linéaire : g = 255·(f/255)^(1/γ).",
    theory=(
        "Contrairement au réglage linéaire, la correction gamma modifie "
        "surtout les tons moyens et préserve le noir et le blanc. γ > 1 "
        "éclaircit les ombres, γ < 1 les assombrit. Le calcul se fait par "
        "table de correspondance (cv2.LUT) : 256 valeurs calculées une fois, "
        "puis un simple accès mémoire par pixel — bien plus rapide qu'une "
        "puissance par pixel."
    ),
    tp=("TP4 §2", "complément"),
    params=(
        FloatParam(
            name="gamma",
            label="Gamma (γ)",
            value=1.0,
            min=0.1,
            max=5.0,
            step=0.05,
            help="1 = inchangé. >1 éclaircit les tons sombres, <1 les assombrit.",
        ),
    ),
)
def correction_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(1e-3, float(gamma))
    table = np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], np.float32
    )
    return cv2.LUT(image, np.clip(table, 0, 255).astype(np.uint8))


@register(
    id="negatif",
    name="Négatif",
    family="Luminosité & contraste",
    summary="Inverse les intensités : g = 255 − f.",
    theory=(
        "Opération ponctuelle la plus simple, réalisée par cv2.bitwise_not "
        "(complément à un sur 8 bits). Elle rend lisibles des détails situés "
        "dans les hautes lumières, d'où son usage courant en imagerie "
        "médicale."
    ),
    tp=("TP4 §5",),
)
def negatif(image: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(image)


@register(
    id="posterisation",
    name="Posterisation (quantification)",
    family="Luminosité & contraste",
    summary="Réduit le nombre de niveaux d'intensité par canal.",
    theory=(
        "On divise la plage 0-255 en N paliers : g = round(f/pas)·pas. C'est "
        "une illustration directe de la quantification d'une image numérique "
        "(TP2 §1.1) : avec 2 niveaux par canal il ne reste que 8 couleurs "
        "possibles au lieu de 16,7 millions."
    ),
    tp=("TP2 §1.1", "complément"),
    params=(
        IntParam(
            name="niveaux",
            label="Niveaux par canal",
            value=4,
            min=2,
            max=64,
            help="2 niveaux par canal = 8 couleurs ; 256 = image inchangée.",
        ),
    ),
)
def posterisation(image: np.ndarray, niveaux: int) -> np.ndarray:
    niveaux = max(2, int(niveaux))
    pas = 255.0 / (niveaux - 1)
    table = np.array(
        [round(round(i / pas) * pas) for i in range(256)], np.float32
    )
    return cv2.LUT(image, np.clip(table, 0, 255).astype(np.uint8))
