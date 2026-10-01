"""Lissage et débruitage (TP7, exercices 1 à 4).

Rappel de convolution. Filtrer une image, c'est faire glisser un petit tableau
de coefficients (le **noyau**) sur l'image et remplacer chaque pixel par la
somme pondérée de son voisinage :

    g(x, y) = Σ_i Σ_j  k(i, j) · f(x − i, y − j)

Trois familles apparaissent dans ce module :

* **filtres linéaires** (moyenneur, gaussien, noyau personnalisé) : une vraie
  convolution, donc rapides et prévisibles, mais ils lissent les contours
  autant que le bruit ;
* **filtres de rang** (médian) : on trie les valeurs du voisinage, ce qui
  n'est pas une convolution ; très efficace sur le bruit impulsionnel ;
* **filtres non linéaires guidés par l'intensité** (bilatéral) : la pondération
  dépend du contenu, ce qui préserve les contours au prix d'un coût de calcul
  bien supérieur.

Toutes les tailles de noyau sont impaires : un noyau pair n'a pas de pixel
central, donc pas de centre de symétrie, et OpenCV le refuse.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..params import BoolParam, ChoiceParam, FloatParam, IntParam
from ..registry import register

#: Modes de bord acceptés par le moteur de filtrage d'OpenCV.
#: ``BORDER_WRAP`` en est volontairement absent : les fonctions de convolution
#: le refusent (assertion ``columnBorderType != BORDER_WRAP``), contrairement
#: aux fonctions de déformation géométrique où il est valide.
_BORDS = {
    "Miroir (défaut)": cv2.BORDER_DEFAULT,
    "Miroir avec duplication": cv2.BORDER_REFLECT,
    "Réplication": cv2.BORDER_REPLICATE,
    "Constante (noir)": cv2.BORDER_CONSTANT,
}


@register(
    id="filtre_moyenneur",
    name="Filtre moyenneur (boîte)",
    family="Lissage & débruitage",
    summary="Remplace chaque pixel par la moyenne de son voisinage.",
    theory=(
        "Noyau de coefficients tous égaux à 1/(l·h) — pour 3×3, neuf fois 1/9. "
        "C'est le filtre passe-bas le plus simple. Plus le noyau est grand, "
        "plus le bruit disparaît, mais plus l'image devient floue : le noyau "
        "ne distingue pas un contour d'une fluctuation de bruit. Un noyau "
        "rectangulaire (largeur ≠ hauteur) floute davantage dans une seule "
        "direction, ce qui se voit comme un effet de filé."
    ),
    tp=("TP7 ex.1",),
    params=(
        IntParam(name="largeur_noyau", label="Largeur du noyau", value=5, min=1,
                 max=31, odd_only=True, unit="px"),
        IntParam(name="hauteur_noyau", label="Hauteur du noyau", value=5, min=1,
                 max=31, odd_only=True, unit="px"),
        ChoiceParam(name="bord", label="Traitement des bords",
                    value="Miroir (défaut)", options=_BORDS, advanced=True,
                    help="Comment compléter le voisinage au bord de l'image."),
    ),
)
def filtre_moyenneur(
    image: np.ndarray, largeur_noyau: int, hauteur_noyau: int, bord: int
) -> tuple[np.ndarray, dict[str, object]]:
    sortie = cv2.blur(
        image, (int(largeur_noyau), int(hauteur_noyau)), borderType=bord
    )
    return sortie, {
        "coefficient du noyau": f"1/{largeur_noyau * hauteur_noyau}",
    }


@register(
    id="filtre_gaussien",
    name="Filtre gaussien",
    family="Lissage & débruitage",
    summary="Moyenne pondérée par une cloche gaussienne.",
    theory=(
        "Le noyau vaut G(i, j) = exp(−(i² + j²) / 2σ²), normalisé. Les pixels "
        "proches du centre pèsent plus que les pixels éloignés : le lissage "
        "est plus doux et plus naturel que celui du moyenneur, sans les "
        "artefacts rectangulaires de ce dernier. Deux réglages interviennent : "
        "la **taille du noyau** (étendue du voisinage) et **σ** (largeur de la "
        "cloche). Avec σ = 0, OpenCV le déduit de la taille "
        "(σ ≈ 0,3·((n−1)/2 − 1) + 0,8) ; c'est ce que faisait le TP7 ex.2 dans "
        "sa première version. Si σ est grand par rapport au noyau, la cloche "
        "est tronquée et le filtre se rapproche d'un moyenneur."
    ),
    tp=("TP7 ex.2",),
    params=(
        IntParam(name="taille_noyau", label="Taille du noyau", value=5, min=1,
                 max=31, odd_only=True, unit="px"),
        FloatParam(name="sigma_x", label="σ horizontal", value=0.0, min=0.0,
                   max=20.0, step=0.1,
                   help="0 = calculé automatiquement depuis la taille du noyau."),
        FloatParam(name="sigma_y", label="σ vertical", value=0.0, min=0.0,
                   max=20.0, step=0.1,
                   help="0 = identique à σ horizontal.", advanced=True),
        ChoiceParam(name="bord", label="Traitement des bords",
                    value="Miroir (défaut)", options=_BORDS, advanced=True),
    ),
)
def filtre_gaussien(
    image: np.ndarray,
    taille_noyau: int,
    sigma_x: float,
    sigma_y: float,
    bord: int,
) -> tuple[np.ndarray, dict[str, object]]:
    taille = int(taille_noyau)
    sortie = cv2.GaussianBlur(
        image, (taille, taille), sigmaX=float(sigma_x), sigmaY=float(sigma_y),
        borderType=bord,
    )
    # σ effectivement utilisé par OpenCV lorsque l'on passe 0.
    sigma_effectif = (
        float(sigma_x) if sigma_x > 0 else 0.3 * ((taille - 1) * 0.5 - 1) + 0.8
    )
    return sortie, {"σ effectif": f"{sigma_effectif:.2f}"}


@register(
    id="filtre_median",
    name="Filtre médian",
    family="Lissage & débruitage",
    summary="Remplace chaque pixel par la médiane de son voisinage.",
    theory=(
        "Filtre de rang, non linéaire : les n² valeurs du voisinage sont "
        "triées et c'est la valeur centrale qui est retenue. Un pixel sel "
        "(255) ou poivre (0) se retrouve en bout de tri et n'influence donc "
        "pas le résultat — d'où son efficacité remarquable sur le bruit "
        "impulsionnel, là où moyenneur et gaussien étalent la tache. "
        "La valeur sortie existe toujours dans l'image d'origine, ce qui "
        "préserve mieux les contours francs."
    ),
    tp=("TP7 ex.3", "TP7 ex.4"),
    params=(
        IntParam(name="taille_noyau", label="Taille du noyau", value=5, min=3,
                 max=31, odd_only=True, unit="px",
                 help="Doit être impair et ≥ 3. Au-delà de 5, OpenCV exige du 8 bits."),
    ),
)
def filtre_median(image: np.ndarray, taille_noyau: int) -> np.ndarray:
    return cv2.medianBlur(image, int(taille_noyau))


@register(
    id="filtre_bilateral",
    name="Filtre bilatéral",
    family="Lissage & débruitage",
    summary="Lisse les zones homogènes en préservant les contours.",
    theory=(
        "Deux gaussiennes multipliées : une sur la **distance spatiale** "
        "(σ_espace) comme un flou gaussien ordinaire, et une sur la "
        "**différence d'intensité** (σ_couleur). Un voisin très différent du "
        "pixel central reçoit un poids quasi nul, donc le filtre ne mélange "
        "pas les deux côtés d'un contour. Conséquences pratiques : σ_couleur "
        "élevé (> 150) rapproche le résultat d'un flou gaussien ; d = 0 fait "
        "déduire le diamètre de σ_espace ; et le coût est bien supérieur à "
        "celui d'un gaussien, car le noyau doit être recalculé pour chaque "
        "pixel."
    ),
    tp=("TP7 ex.4",),
    realtime_safe=False,
    params=(
        IntParam(name="diametre", label="Diamètre du voisinage (d)", value=9,
                 min=0, max=25, unit="px",
                 help="0 = déduit de σ_espace. Au-delà de 9, c'est très lent."),
        FloatParam(name="sigma_couleur", label="σ couleur", value=75.0, min=1.0,
                   max=250.0, step=1.0,
                   help="Tolérance sur l'écart d'intensité : grand = contours moins protégés."),
        FloatParam(name="sigma_espace", label="σ espace", value=75.0, min=1.0,
                   max=250.0, step=1.0,
                   help="Portée géométrique du filtre, en pixels."),
    ),
)
def filtre_bilateral(
    image: np.ndarray, diametre: int, sigma_couleur: float, sigma_espace: float
) -> np.ndarray:
    return cv2.bilateralFilter(
        image, int(diametre), float(sigma_couleur), float(sigma_espace)
    )


@register(
    id="noyau_personnalise",
    name="Noyau de convolution personnalisé",
    family="Lissage & débruitage",
    summary="Applique un noyau choisi via cv2.filter2D.",
    theory=(
        "cv2.filter2D effectue la corrélation du noyau avec l'image (une "
        "convolution au retournement du noyau près, sans conséquence pour les "
        "noyaux symétriques). C'est l'outil qui permet de construire un filtre "
        "« à la main », comme dans la première version du TP7 ex.1 "
        "(``np.ones((3, 3)) / 9``). Un noyau dont la somme vaut 1 préserve la "
        "luminosité moyenne ; une somme nulle (détecteurs de contours) donne "
        "une image presque noire à laquelle on ajoute un décalage pour la "
        "rendre lisible."
    ),
    tp=("TP7 ex.1",),
    params=(
        ChoiceParam(
            name="modele",
            label="Noyau",
            value="Moyenneur",
            options={
                "Identité": "identite",
                "Moyenneur": "moyenne",
                "Gaussien approché": "gaussien",
                "Netteté (sharpen)": "nettete",
                "Laplacien 4-voisins": "laplacien4",
                "Laplacien 8-voisins": "laplacien8",
                "Relief (emboss)": "relief",
                "Sobel horizontal": "sobel_x",
                "Sobel vertical": "sobel_y",
            },
        ),
        IntParam(name="taille", label="Taille (noyaux redimensionnables)",
                 value=3, min=3, max=15, odd_only=True,
                 help="Ne s'applique qu'aux noyaux moyenneur et gaussien approché."),
        BoolParam(name="normaliser", label="Normaliser le noyau", value=True,
                  help="Divise par la somme des coefficients (préserve la luminosité)."),
        IntParam(name="decalage", label="Décalage ajouté (delta)", value=0,
                 min=-128, max=128,
                 help="Utile pour visualiser un noyau de somme nulle (essayez 128)."),
    ),
)
def noyau_personnalise(
    image: np.ndarray, modele: str, taille: int, normaliser: bool, decalage: int
) -> tuple[np.ndarray, dict[str, object]]:
    taille = int(taille)
    noyau = _construire_noyau(modele, taille)

    somme = float(noyau.sum())
    if normaliser and abs(somme) > 1e-9:
        noyau = noyau / somme

    sortie = cv2.filter2D(image, -1, noyau, delta=float(decalage))
    return sortie, {
        "noyau": _formater_noyau(noyau),
        "somme des coefficients": f"{float(noyau.sum()):.3f}",
    }


def _construire_noyau(modele: str, taille: int) -> np.ndarray:
    """Noyaux prédéfinis. Seuls « moyenne » et « gaussien » dépendent de la taille."""
    if modele == "moyenne":
        return np.ones((taille, taille), np.float32)
    if modele == "gaussien":
        colonne = cv2.getGaussianKernel(taille, -1)
        return (colonne @ colonne.T).astype(np.float32)
    if modele == "identite":
        noyau = np.zeros((taille, taille), np.float32)
        noyau[taille // 2, taille // 2] = 1.0
        return noyau
    fixes = {
        "nettete": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32),
        "laplacien4": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32),
        "laplacien8": np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float32),
        "relief": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], np.float32),
        "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32),
        "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32),
    }
    return fixes[modele]


def _formater_noyau(noyau: np.ndarray) -> str:
    """Rendu texte compact du noyau, affiché dans le panneau d'informations."""
    if noyau.shape[0] > 7:
        return f"{noyau.shape[0]}×{noyau.shape[1]} (trop grand pour l'affichage)"
    lignes = [
        " ".join(f"{v:6.3f}" for v in ligne) for ligne in np.asarray(noyau)
    ]
    return "\n".join(lignes)


@register(
    id="nettete_unsharp",
    name="Accentuation (masque flou)",
    family="Lissage & débruitage",
    summary="Renforce les détails en soustrayant une version floue de l'image.",
    theory=(
        "g = f + k·(f − flou(f)). La différence f − flou(f) isole les hautes "
        "fréquences, c'est-à-dire les détails et les contours ; on les "
        "réinjecte amplifiés. Le rayon du flou fixe l'échelle des détails "
        "accentués, k leur intensité. Trop de k crée des halos clairs autour "
        "des contours et amplifie le bruit — le masque flou est un filtre "
        "passe-haut, et le bruit est une haute fréquence."
    ),
    tp=("complément",),
    params=(
        IntParam(name="taille_noyau", label="Rayon du flou (noyau)", value=5,
                 min=3, max=31, odd_only=True),
        FloatParam(name="sigma", label="σ du flou", value=1.0, min=0.0, max=20.0,
                   step=0.1, help="0 = déduit de la taille du noyau."),
        FloatParam(name="intensite", label="Intensité (k)", value=1.0, min=0.0,
                   max=3.0, step=0.05),
        IntParam(name="seuil", label="Seuil de protection", value=0, min=0,
                 max=64,
                 help=(
                     "Les zones dont l'écart au flou est inférieur au seuil ne "
                     "sont pas accentuées : évite d'amplifier le bruit des "
                     "aplats."
                 )),
    ),
)
def nettete_unsharp(
    image: np.ndarray,
    taille_noyau: int,
    sigma: float,
    intensite: float,
    seuil: int,
) -> np.ndarray:
    taille = int(taille_noyau)
    flou = cv2.GaussianBlur(image, (taille, taille), float(sigma))
    aiguise = cv2.addWeighted(
        image, 1.0 + float(intensite), flou, -float(intensite), 0
    )
    if seuil <= 0:
        return aiguise
    # On n'accentue que là où l'écart au flou dépasse le seuil.
    ecart = cv2.absdiff(image, flou)
    masque = (ecart >= int(seuil)).astype(np.uint8)
    return np.where(masque.astype(bool), aiguise, image).astype(np.uint8)
