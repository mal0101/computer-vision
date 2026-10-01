"""Tests des entrées/sorties et des fonctions de dessin."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from cvlab import drawing as ds
from cvlab import io_utils


def test_encodage_decodage_aller_retour(image_couleur: np.ndarray) -> None:
    donnees = io_utils.encode_image(image_couleur, ".png")
    relu = io_utils.decode_image(donnees)
    # Le PNG est sans perte : l'image doit être identique au bit près.
    assert np.array_equal(relu, image_couleur)


def test_encodage_jpeg_avec_perte(image_couleur: np.ndarray) -> None:
    petit = io_utils.encode_image(image_couleur, ".jpg", qualite=50)
    grand = io_utils.encode_image(image_couleur, ".jpg", qualite=100)
    assert len(petit) < len(grand)


def test_encodage_extension_sans_point(image_couleur: np.ndarray) -> None:
    assert io_utils.encode_image(image_couleur, "png")


def test_encodage_format_inconnu(image_couleur: np.ndarray) -> None:
    """Une extension inconnue donne une erreur applicative, pas une cv2.error."""
    with pytest.raises(io_utils.ImageLoadError, match="non prise en charge"):
        io_utils.encode_image(image_couleur, ".inconnu")


def test_enregistrement_et_chargement(tmp_path: Path, image_couleur) -> None:
    chemin = io_utils.save_image(image_couleur, tmp_path / "a" / "b" / "img.png")
    assert chemin.is_file()
    assert np.array_equal(io_utils.load_image(chemin), image_couleur)


def test_chargement_fichier_absent(tmp_path: Path) -> None:
    with pytest.raises(io_utils.ImageLoadError, match="introuvable"):
        io_utils.load_image(tmp_path / "absent.png")


def test_chargement_fichier_invalide(tmp_path: Path) -> None:
    mauvais = tmp_path / "faux.png"
    mauvais.write_bytes(b"ceci n'est pas une image")
    with pytest.raises(io_utils.ImageLoadError, match="format non reconnu"):
        io_utils.load_image(mauvais)


def test_decodage_vide() -> None:
    with pytest.raises(io_utils.ImageLoadError, match="vide"):
        io_utils.decode_image(b"")


def test_chargement_chemin_accentue(tmp_path: Path, image_couleur) -> None:
    """Les chemins non ASCII passent, grâce au couple read_bytes + imdecode."""
    chemin = tmp_path / "éléphant_çà.png"
    io_utils.save_image(image_couleur, chemin)
    assert io_utils.load_image(chemin).shape == image_couleur.shape


def test_images_exemple_ne_leve_pas() -> None:
    """La recherche d'exemples doit être tolérante à l'absence de dossier TP*."""
    assert isinstance(io_utils.sample_images(), list)
    assert io_utils.sample_images(racine="/dossier/qui/nexiste/pas") == []


def test_limite_du_nombre_d_exemples() -> None:
    assert len(io_utils.sample_images(limite=3)) <= 3


def test_camera_non_ouverte_leve() -> None:
    camera = io_utils.Camera(index=0)
    with pytest.raises(io_utils.CameraError, match="non ouverte"):
        camera.set_property(cv2.CAP_PROP_BRIGHTNESS, 10)
    assert camera.resolution == (0, 0)


# ---------------------------------------------------------------------------
# Dessin (TP2)
# ---------------------------------------------------------------------------
def test_dessin_ne_modifie_pas_l_original() -> None:
    """Les enveloppes du TP2 renvoient une copie : indispensable en chaîne."""
    toile = np.zeros((40, 60, 3), np.uint8)
    reference = toile.copy()
    ds.dess_ligne(toile, (0, 0), (59, 39), (255, 255, 255), 2)
    assert np.array_equal(toile, reference)


def test_toutes_les_primitives_dessinent() -> None:
    toile = np.zeros((60, 80, 3), np.uint8)
    sorties = [
        ds.dess_ligne(toile, (2, 2), (70, 50), (255, 0, 0), 2),
        ds.dess_rectangle(toile, (4, 4), (40, 30), (0, 255, 0), cv2.FILLED),
        ds.dess_cercle(toile, (40, 30), 15, (0, 0, 255), 2),
        ds.dess_ellipse(toile, (40, 30), (20, 10), 30, 0, 300, (255, 255, 0), 2),
        ds.dess_polylignes(toile, [[(5, 5), (30, 40), (60, 10)]], True,
                           (255, 0, 255), 2),
        ds.dess_text(toile, "Test", (5, 40), 0.6, (255, 255, 255), 1),
        ds.dess_texte_encadre(toile, "Encadré", (10, 30), 0.5),
    ]
    for sortie in sorties:
        assert sortie.shape == toile.shape
        assert sortie.any(), "la primitive n'a rien dessiné"


def test_rayon_negatif_tolere() -> None:
    toile = np.zeros((20, 20, 3), np.uint8)
    assert ds.dess_cercle(toile, (10, 10), -5, (255, 255, 255)).shape == toile.shape


def test_axes_nuls_toleres() -> None:
    toile = np.zeros((20, 20, 3), np.uint8)
    assert ds.dess_ellipse(toile, (10, 10), (0, 0), 0, 0, 360).shape == toile.shape


def test_taille_texte_croissante() -> None:
    petite = ds.taille_texte("abc", 0.5)
    grande = ds.taille_texte("abc", 2.0)
    assert grande[0] > petite[0] and grande[1] > petite[1]


def test_polices_et_types_de_ligne_valides() -> None:
    toile = np.zeros((40, 200, 3), np.uint8)
    for police in ds.POLICES.values():
        assert ds.dess_text(toile, "Ok", (5, 25), 0.5, police=police).any()
    for type_ligne in ds.TYPES_LIGNE.values():
        assert ds.dess_ligne(toile, (0, 0), (199, 39), (255, 255, 255), 2,
                             type_ligne).any()
