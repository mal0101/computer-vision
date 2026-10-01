"""Tests de l'application de bureau (parties sans interface graphique).

La boucle d'affichage elle-même n'est pas testée automatiquement — elle a
besoin d'un serveur graphique et d'un œil humain. Ce qui est testé ici, c'est
toute la logique qui l'entoure : conversion trackbar ↔ valeur de paramètre,
traduction des gestes souris en paramètres, et traitement d'une image.
"""

from __future__ import annotations

import numpy as np
import pytest

from cvlab import registry
from cvlab.params import (
    BoolParam,
    ChoiceParam,
    ColorParam,
    FloatParam,
    IntParam,
    TextParam,
)
from cvlab.pipeline import Pipeline
from ui.desktop import AtelierBureau, EchelleTrackbar, _couper, _nom_trackbar

TOUS = registry.all_filters()


@pytest.mark.parametrize("definition", TOUS, ids=[d.id for d in TOUS])
def test_echelle_couvre_toutes_les_positions(definition) -> None:
    """Toute position de trackbar doit donner une valeur de paramètre valide."""
    for spec in definition.params:
        echelle = EchelleTrackbar.pour(spec)
        if echelle is None:
            assert isinstance(spec, (TextParam, ColorParam)), (
                f"{definition.id}.{spec.name}: type sans trackbar inattendu"
            )
            continue
        assert echelle.maximum_trackbar >= 1
        for position in range(echelle.maximum_trackbar + 1):
            valeur = echelle.vers_valeur(position)
            # coerce ne doit rien changer : la valeur est déjà canonique.
            assert spec.coerce(valeur) == valeur, (
                f"{definition.id}.{spec.name} position {position}"
            )


@pytest.mark.parametrize("definition", TOUS, ids=[d.id for d in TOUS])
def test_echelle_aller_retour(definition) -> None:
    """valeur -> position -> valeur doit être stable (hors arrondi des réels)."""
    for spec in definition.params:
        echelle = EchelleTrackbar.pour(spec)
        if echelle is None:
            continue
        for position in range(echelle.maximum_trackbar + 1):
            valeur = echelle.vers_valeur(position)
            if isinstance(spec, FloatParam):
                # Un réel repasse par un cran : on tolère un pas de quantification.
                pas = (spec.max - spec.min) / echelle.maximum_trackbar
                assert abs(
                    float(echelle.vers_valeur(echelle.vers_position(valeur)))
                    - float(valeur)
                ) <= pas + 1e-9
            else:
                assert echelle.vers_position(valeur) == position


def test_echelle_impair_ne_produit_que_des_impairs() -> None:
    spec = IntParam(name="k", label="k", value=5, min=3, max=31, odd_only=True)
    echelle = EchelleTrackbar.pour(spec)
    assert echelle is not None
    valeurs = [echelle.vers_valeur(p) for p in range(echelle.maximum_trackbar + 1)]
    assert all(v % 2 == 1 for v in valeurs)
    assert min(valeurs) == 3 and max(valeurs) == 31


def test_echelle_entier_negatif() -> None:
    spec = IntParam(name="b", label="b", value=0, min=-128, max=128)
    echelle = EchelleTrackbar.pour(spec)
    assert echelle is not None
    assert echelle.vers_valeur(0) == -128
    assert echelle.vers_valeur(echelle.maximum_trackbar) == 128
    assert echelle.vers_valeur(echelle.vers_position(0)) == 0


def test_echelle_booleen_et_choix() -> None:
    booleen = EchelleTrackbar.pour(BoolParam(name="b", label="b", value=False))
    assert booleen is not None
    assert booleen.vers_valeur(0) is False and booleen.vers_valeur(1) is True

    choix = EchelleTrackbar.pour(
        ChoiceParam(name="c", label="c", value="A", options={"A": 1, "B": 2, "C": 3})
    )
    assert choix is not None
    assert choix.vers_valeur(2) == "C"
    assert choix.vers_position("B") == 1


def test_echelle_absente_pour_texte_et_couleur() -> None:
    assert EchelleTrackbar.pour(TextParam(name="t", label="t")) is None
    assert EchelleTrackbar.pour(
        ColorParam(name="c", label="c", value=(0, 0, 0))
    ) is None


def test_noms_de_trackbar_uniques_par_filtre() -> None:
    """Le nom de trackbar sert de clé de lecture : il doit être unique."""
    for definition in TOUS:
        noms = [_nom_trackbar(s) for s in definition.params]
        assert len(noms) == len(set(noms)), definition.id
        for nom in noms:
            assert nom.isascii(), f"{definition.id}: « {nom} » n'est pas ASCII"
            assert nom, definition.id


def _atelier(image: np.ndarray, **kwargs) -> AtelierBureau:
    return AtelierBureau(
        source_image=image, camera=None, filtres=TOUS,
        index_initial=[d.id for d in TOUS].index(kwargs.pop("filtre", "canny")),
        **kwargs,
    )


def test_traitement_sans_fenetre(image_couleur: np.ndarray) -> None:
    """``traiter`` ne doit pas dépendre de l'existence des fenêtres OpenCV.

    ``cv2.getTrackbarPos`` lève ``cv2.error`` si la fenêtre n'existe pas ; le
    code doit retomber sur les valeurs mémorisées.
    """
    atelier = _atelier(image_couleur)
    sortie = atelier.traiter(image_couleur)
    assert sortie.dtype == np.uint8 and sortie.size > 0
    assert atelier.duree_ms >= 0.0


def test_traitement_avec_chaine_prealable(image_couleur: np.ndarray) -> None:
    prefixe = Pipeline()
    prefixe.add("negatif")
    atelier = _atelier(image_couleur, chaine_prefixe=prefixe)
    atelier.traiter(image_couleur)
    assert atelier.derniere_entree is not None
    # Le préfixe a bien été appliqué avant le filtre réglé.
    assert not np.array_equal(atelier.derniere_entree, image_couleur)


def test_erreur_de_filtre_affichee_sur_l_image(image_couleur: np.ndarray) -> None:
    """Un filtre en échec produit une image porteuse du message, pas une exception."""
    atelier = AtelierBureau(
        source_image=image_couleur, camera=None,
        filtres=[registry.get("fusion")], index_initial=0, reference=None,
    )
    sortie = atelier.traiter(image_couleur)
    assert sortie.size > 0
    assert atelier.duree_ms == 0.0


def test_rectangle_souris_vers_parametres(image_couleur: np.ndarray) -> None:
    atelier = _atelier(image_couleur, filtre="recadrer")
    atelier.derniere_entree = image_couleur
    hauteur, largeur = image_couleur.shape[:2]
    # On simule un glisser du quart au trois-quarts de l'image, sans toucher
    # aux fenêtres OpenCV (construire_trackbars est neutralisé).
    atelier.construire_trackbars = lambda: None  # type: ignore[method-assign]
    atelier._appliquer_rectangle(
        (largeur // 4, hauteur // 4), (3 * largeur // 4, 3 * hauteur // 4)
    )
    valeurs = atelier.valeurs["recadrer"]
    assert valeurs["x"] == pytest.approx(25.0, abs=1.5)
    assert valeurs["largeur"] == pytest.approx(50.0, abs=2.0)


def test_rectangle_trop_petit_ignore(image_couleur: np.ndarray) -> None:
    atelier = _atelier(image_couleur, filtre="recadrer")
    atelier.derniere_entree = image_couleur
    atelier.construire_trackbars = lambda: None  # type: ignore[method-assign]
    avant = dict(atelier.valeurs["recadrer"])
    atelier._appliquer_rectangle((10, 10), (12, 12))
    assert atelier.valeurs["recadrer"] == avant


def test_points_perspective_vers_parametres(image_couleur: np.ndarray) -> None:
    atelier = _atelier(image_couleur, filtre="deformation_perspective")
    atelier.derniere_entree = image_couleur
    atelier.construire_trackbars = lambda: None  # type: ignore[method-assign]
    hauteur, largeur = image_couleur.shape[:2]
    atelier.points_perspective = [
        (0, 0), (largeur - 1, 0), (largeur - 1, hauteur - 1), (0, hauteur - 1)
    ]
    atelier._appliquer_points_perspective()
    valeurs = atelier.valeurs["deformation_perspective"]
    assert valeurs["p1x"] == pytest.approx(0.0, abs=1.0)
    assert valeurs["p3x"] == pytest.approx(100.0, abs=2.0)
    assert atelier.points_perspective == []


def test_changement_de_filtre_conserve_les_reglages(image_couleur: np.ndarray) -> None:
    atelier = _atelier(image_couleur)
    atelier.construire_trackbars = lambda: None  # type: ignore[method-assign]
    atelier.valeurs["canny"]["seuil_bas"] = 77
    atelier.changer_filtre(1)
    atelier.changer_filtre(-1)
    assert atelier.valeurs["canny"]["seuil_bas"] == 77


def test_reinitialisation(image_couleur: np.ndarray) -> None:
    atelier = _atelier(image_couleur)
    atelier.construire_trackbars = lambda: None  # type: ignore[method-assign]
    atelier.valeurs["canny"]["seuil_bas"] = 7
    atelier.reinitialiser()
    assert atelier.valeurs["canny"]["seuil_bas"] == 50


def test_changement_de_filtre_boucle(image_couleur: np.ndarray) -> None:
    atelier = _atelier(image_couleur)
    atelier.construire_trackbars = lambda: None  # type: ignore[method-assign]
    atelier.index = len(TOUS) - 1
    atelier.changer_filtre(1)
    assert atelier.index == 0
    atelier.changer_filtre(-1)
    assert atelier.index == len(TOUS) - 1


def test_enregistrement(tmp_path, image_couleur: np.ndarray) -> None:
    atelier = _atelier(image_couleur, dossier_sortie=tmp_path)
    atelier.derniere_sortie = image_couleur
    atelier.enregistrer()
    assert list(tmp_path.glob("*.png"))


def test_incrustation_lisible(image_couleur: np.ndarray) -> None:
    atelier = _atelier(image_couleur)
    atelier.derniere_sortie = image_couleur
    atelier.position_souris = (10, 10)
    atelier.notifier("message de test")
    rendu = atelier.incruster(image_couleur, image_couleur)
    assert rendu.shape == image_couleur.shape
    assert not np.array_equal(rendu, image_couleur)  # du texte a été ajouté


def test_incrustation_sur_image_grise(image_gris: np.ndarray) -> None:
    atelier = _atelier(image_gris)
    atelier.position_souris = (5, 5)
    rendu = atelier.incruster(image_gris, image_gris)
    assert rendu.ndim == 3  # l'incrustation est en couleur


def test_decoupe_de_ligne() -> None:
    assert _couper("a=1, b=2", 100) == ["a=1, b=2"]
    assert len(_couper(", ".join(f"p{i}=1" for i in range(40)), 20)) <= 4
