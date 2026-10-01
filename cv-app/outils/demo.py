"""Génère les planches d'illustration de la documentation.

    python outils/demo.py [--sortie docs/images] [--image chemin]

Chaque planche est une mosaïque fabriquée uniquement avec OpenCV et les
fonctions de dessin du TP2 : aucune dépendance supplémentaire, et les images
produites sont reproductibles (les générateurs de bruit sont lancés avec une
graine fixe).

Ce script sert aussi de démonstration de l'API : il montre comment appliquer un
filtre par programme, sans passer par une interface.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cvlab import drawing as ds  # noqa: E402
from cvlab import histogram as hist  # noqa: E402
from cvlab import images as im  # noqa: E402
from cvlab import io_utils, metrics, registry  # noqa: E402


def mosaique(
    vignettes: list[tuple[str, np.ndarray]],
    colonnes: int = 3,
    largeur_vignette: int = 300,
    marge: int = 8,
    hauteur_titre: int = 26,
) -> np.ndarray:
    """Assemble des images légendées en une grille.

    Les vignettes sont mises à la même largeur en conservant leur rapport
    d'aspect, puis centrées dans une cellule de hauteur commune.
    """
    if not vignettes:
        raise ValueError("aucune vignette à assembler")

    redimensionnees: list[tuple[str, np.ndarray]] = []
    for titre, image in vignettes:
        couleur = im.to_bgr(image)
        hauteur, largeur = couleur.shape[:2]
        facteur = largeur_vignette / largeur
        redimensionnees.append((
            titre,
            cv2.resize(
                couleur,
                (largeur_vignette, max(1, int(round(hauteur * facteur)))),
                interpolation=cv2.INTER_AREA if facteur < 1 else cv2.INTER_LINEAR,
            ),
        ))

    hauteur_cellule = max(img.shape[0] for _, img in redimensionnees)
    lignes = (len(redimensionnees) + colonnes - 1) // colonnes
    largeur_totale = colonnes * largeur_vignette + (colonnes + 1) * marge
    hauteur_totale = lignes * (hauteur_cellule + hauteur_titre) + (lignes + 1) * marge

    planche = np.zeros((hauteur_totale, largeur_totale, 3), np.uint8)
    planche[:] = (24, 26, 30)

    for index, (titre, image) in enumerate(redimensionnees):
        ligne, colonne = divmod(index, colonnes)
        x = marge + colonne * (largeur_vignette + marge)
        y = marge + ligne * (hauteur_cellule + hauteur_titre + marge)
        planche = ds.dess_text(planche, titre, (x + 2, y + 17), 0.46,
                               (235, 235, 240), 1, cv2.FONT_HERSHEY_SIMPLEX)
        decalage = (hauteur_cellule - image.shape[0]) // 2
        haut = y + hauteur_titre + decalage
        planche[haut:haut + image.shape[0], x:x + image.shape[1]] = image
        planche = ds.dess_rectangle(
            planche, (x - 1, haut - 1),
            (x + image.shape[1], haut + image.shape[0]),
            (70, 74, 82), 1, cv2.LINE_8,
        )
    return planche


def _appliquer(identifiant: str, image: np.ndarray, **params) -> np.ndarray:
    sortie, _ = registry.get(identifiant).apply(image, params)
    return sortie


def image_de_travail(chemin: str | None) -> np.ndarray:
    """Image de départ : celle demandée, sinon une image des TP, sinon une mire."""
    if chemin:
        return io_utils.load_image(chemin)
    exemples = [
        c for c in io_utils.sample_images(limite=200)
        if "cells" in c.name or "coins" in c.name or "image1" in c.name
    ]
    if exemples:
        image = io_utils.load_image(exemples[0])
    else:
        from ui.streamlit_app import image_de_demonstration

        image = image_de_demonstration()
    cote = max(image.shape[:2])
    if cote > 600:
        facteur = 600 / cote
        image = cv2.resize(
            image,
            (int(image.shape[1] * facteur), int(image.shape[0] * facteur)),
            interpolation=cv2.INTER_AREA,
        )
    return image


# ---------------------------------------------------------------------------
# Planches
# ---------------------------------------------------------------------------
def planche_lissage(source: np.ndarray) -> np.ndarray:
    """TP7 ex.1 à 4 : comparaison des quatre filtres de lissage."""
    bruitee = _appliquer("bruit_sel_poivre", source, proportion=0.06,
                         ratio_sel=0.5, graine=1)
    vignettes = [
        ("Originale", source),
        ("Bruit sel-poivre 6 %", bruitee),
        ("Moyenneur 5x5", _appliquer("filtre_moyenneur", bruitee,
                                     largeur_noyau=5, hauteur_noyau=5)),
        ("Gaussien 5x5 s=0", _appliquer("filtre_gaussien", bruitee,
                                        taille_noyau=5, sigma_x=0.0)),
        ("Median 5x5", _appliquer("filtre_median", bruitee, taille_noyau=5)),
        ("Bilateral d=9", _appliquer("filtre_bilateral", bruitee, diametre=9)),
    ]
    # On annote chaque vignette de son écart à l'image propre : c'est la mesure
    # qui justifie le choix du médian sur ce type de bruit.
    legendees = [vignettes[0], vignettes[1]]
    for titre, image in vignettes[2:]:
        erreur = metrics.mse(source, image)
        legendees.append((f"{titre} — MSE {erreur:.0f}", image))
    return mosaique(legendees, colonnes=3)


def planche_taille_noyau(source: np.ndarray) -> np.ndarray:
    """TP7 ex.1 : l'effet de la taille du noyau d'un moyenneur."""
    vignettes = [("Originale", source)]
    for taille in (3, 5, 9, 15, 21):
        vignettes.append((
            f"Moyenneur {taille}x{taille}",
            _appliquer("filtre_moyenneur", source, largeur_noyau=taille,
                       hauteur_noyau=taille),
        ))
    return mosaique(vignettes, colonnes=3)


def planche_contours(source: np.ndarray) -> np.ndarray:
    """TP7 ex.5 à 7 : gradients, laplacien et Canny."""
    gris = im.to_gray(source)
    vignettes = [
        ("Originale (gris)", gris),
        ("Sobel X", _appliquer("sobel", gris, direction="Horizontal (∂/∂x)",
                               flou_prealable=3)),
        ("Sobel Y", _appliquer("sobel", gris, direction="Vertical (∂/∂y)",
                               flou_prealable=3)),
        ("Module du gradient", _appliquer("sobel", gris,
                                          direction="Module du gradient",
                                          flou_prealable=3)),
        ("Laplacien (LoG)", _appliquer("laplacien", gris, flou_prealable=5)),
        ("Canny 50/150", _appliquer("canny", gris, seuil_bas=50, seuil_haut=150)),
    ]
    return mosaique(vignettes, colonnes=3)


def planche_seuils_canny(source: np.ndarray) -> np.ndarray:
    """TP7 ex.7 : l'effet des deux seuils de l'hystérésis."""
    gris = im.to_gray(source)
    vignettes = [("Originale (gris)", gris)]
    for bas, haut in ((20, 60), (50, 150), (100, 200), (150, 250), (200, 400)):
        contours = _appliquer("canny", gris, seuil_bas=bas, seuil_haut=haut)
        taux = 100.0 * np.count_nonzero(contours) / contours.size
        vignettes.append((f"Canny {bas}/{haut} — {taux:.1f} %", contours))
    return mosaique(vignettes, colonnes=3)


def planche_seuillage(source: np.ndarray) -> np.ndarray:
    """Seuillage global, Otsu et adaptatif face à un éclairage inégal."""
    gris = im.to_gray(source)
    # On simule un éclairage inégal : un dégradé multiplicatif.
    hauteur, largeur = gris.shape
    rampe = np.tile(np.linspace(0.45, 1.25, largeur, dtype=np.float32),
                    (hauteur, 1))
    inegal = np.clip(gris.astype(np.float32) * rampe, 0, 255).astype(np.uint8)
    vignettes = [
        ("Eclairage inegal", inegal),
        ("Seuil global 127", _appliquer("seuillage_simple", inegal, seuil=127)),
        ("Otsu", _appliquer("seuillage_automatique", inegal, methode="Otsu")),
        ("Adaptatif 21, C=5", _appliquer("seuillage_adaptatif", inegal,
                                         taille_bloc=21, constante=5)),
    ]
    return mosaique(vignettes, colonnes=2)


def planche_morphologie(source: np.ndarray) -> np.ndarray:
    """Les sept opérations morphologiques sur une image binarisée."""
    binaire = _appliquer("seuillage_automatique", source, methode="Otsu")
    vignettes = [("Binarisee (Otsu)", binaire)]
    for operation in (
        "Érosion", "Dilatation", "Ouverture", "Fermeture",
        "Gradient morphologique", "Chapeau haut de forme (top-hat)",
        "Chapeau noir (black-hat)",
    ):
        vignettes.append((
            operation,
            _appliquer("morphologie", binaire, operation=operation,
                       forme="Ellipse", taille=5, iterations=1),
        ))
    return mosaique(vignettes, colonnes=4, largeur_vignette=250)


def planche_histogramme(source: np.ndarray) -> np.ndarray:
    """TP4 §8 : égalisation, CLAHE, étirement, et les histogrammes associés."""
    sombre = _appliquer("luminosite_contraste", source, alpha=0.45, beta=20)
    vignettes = [
        ("Image terne", sombre),
        ("Histogramme", hist.draw_histogram(sombre, 520, 300)),
        ("Egalisation", _appliquer("egalisation_histogramme", sombre)),
        ("Histogramme egalise",
         hist.draw_histogram(_appliquer("egalisation_histogramme", sombre),
                             520, 300)),
        ("CLAHE 2.0, 8x8", _appliquer("clahe", sombre, limite=2.0, tuiles=8)),
        ("Histogramme CLAHE",
         hist.draw_histogram(_appliquer("clahe", sombre, limite=2.0, tuiles=8),
                             520, 300)),
    ]
    return mosaique(vignettes, colonnes=2, largeur_vignette=340)


def planche_couleur(source: np.ndarray) -> np.ndarray:
    """TP2 §1 : canaux BGR et HSV."""
    couleur = im.to_bgr(source)
    vignettes = [
        ("Originale", couleur),
        ("Canal bleu", _appliquer("canal_bgr", couleur, canal="Bleu",
                                  mode="Couleur primitive")),
        ("Canal vert", _appliquer("canal_bgr", couleur, canal="Vert",
                                  mode="Couleur primitive")),
        ("Canal rouge", _appliquer("canal_bgr", couleur, canal="Rouge",
                                   mode="Couleur primitive")),
        ("Teinte (H)", _appliquer("canal_hsv", couleur, canal="Teinte (H)",
                                  etirer=True)),
        ("Saturation (S)", _appliquer("canal_hsv", couleur,
                                      canal="Saturation (S)")),
        ("Valeur (V)", _appliquer("canal_hsv", couleur, canal="Valeur (V)")),
        ("Niveaux de gris", _appliquer("niveaux_de_gris", couleur)),
    ]
    return mosaique(vignettes, colonnes=4, largeur_vignette=250)


def planche_bruits(source: np.ndarray) -> np.ndarray:
    """TP7 ex.4 : les quatre modèles de bruit."""
    vignettes = [
        ("Originale", source),
        ("Gaussien s=25", _appliquer("bruit_gaussien", source, sigma=25.0,
                                     graine=1)),
        ("Sel-poivre 5 %", _appliquer("bruit_sel_poivre", source,
                                      proportion=0.05, graine=1)),
        ("Uniforme +/-30", _appliquer("bruit_uniforme", source, minimum=-30.0,
                                      maximum=30.0, graine=1)),
        ("Speckle s=0.3", _appliquer("bruit_multiplicatif", source, sigma=0.3,
                                     graine=1)),
    ]
    return mosaique(vignettes, colonnes=3)


def planche_geometrie(source: np.ndarray) -> np.ndarray:
    """TP4 §1 et §3 : redimensionnement, rotation, recadrage, perspective."""
    vignettes = [
        ("Originale", source),
        ("Rotation 30 deg", _appliquer("rotation", source, angle=30.0,
                                       conserver_cadre=True)),
        ("Recadrage central", _appliquer("recadrer", source, x=25.0, y=25.0,
                                         largeur=50.0, hauteur=50.0)),
        ("Miroir horizontal", _appliquer("retourner", source, sens="Horizontal")),
        ("Perspective", _appliquer("deformation_perspective", source,
                                   p1x=18.0, p1y=8.0, p2x=88.0, p2y=18.0,
                                   p3x=94.0, p3y=92.0, p4x=8.0, p4y=82.0)),
        ("Translation 20 %", _appliquer("translation", source, dx=20.0, dy=10.0,
                                        bord="Constante (noir)")),
    ]
    return mosaique(vignettes, colonnes=3)


PLANCHES = {
    "couleur-canaux": planche_couleur,
    "geometrie": planche_geometrie,
    "bruits": planche_bruits,
    "lissage-comparaison": planche_lissage,
    "lissage-taille-noyau": planche_taille_noyau,
    "contours": planche_contours,
    "canny-seuils": planche_seuils_canny,
    "seuillage": planche_seuillage,
    "morphologie": planche_morphologie,
    "histogramme": planche_histogramme,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sortie", default=str(RACINE / "docs" / "images"),
                        help="dossier de destination des planches")
    parser.add_argument("--image", help="image de départ (sinon : une image des TP)")
    parser.add_argument("--planche", action="append", choices=sorted(PLANCHES),
                        help="ne produire que ces planches (répétable)")
    args = parser.parse_args(argv)

    source = image_de_travail(args.image)
    dossier = Path(args.sortie)
    dossier.mkdir(parents=True, exist_ok=True)
    print(f"Image de départ : {im.describe_image(source)}")

    noms = args.planche or sorted(PLANCHES)
    for nom in noms:
        planche = PLANCHES[nom](source)
        chemin = io_utils.save_image(planche, dossier / f"{nom}.png")
        print(f"  {chemin.relative_to(RACINE)}  ({planche.shape[1]}×{planche.shape[0]})")
    print(f"{len(noms)} planche(s) produite(s) dans {dossier}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
