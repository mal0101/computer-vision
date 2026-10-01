"""Tests de comportement : vérifier que chaque filtre fait bien *ce* qu'il dit.

Les tests de :mod:`tests.test_registry` garantissent qu'aucun filtre ne plante.
Ceux-ci vérifient la sémantique : qu'un filtre médian supprime effectivement le
bruit impulsionnel, qu'un seuillage produit bien une image binaire, que la
teinte est traitée de façon circulaire, etc. C'est ce qui attrape une inversion
d'argument qui ne lèverait aucune exception.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from cvlab import images as im
from cvlab import metrics, registry


def appliquer(identifiant: str, image: np.ndarray, **params):
    """Raccourci : applique un filtre et renvoie l'image seule."""
    sortie, _ = registry.get(identifiant).apply(image, params)
    return sortie


def infos_de(identifiant: str, image: np.ndarray, **params) -> dict:
    _, infos = registry.get(identifiant).apply(image, params)
    return infos


# ---------------------------------------------------------------------------
# Couleur
# ---------------------------------------------------------------------------
def test_niveaux_de_gris_un_canal(image_couleur: np.ndarray) -> None:
    assert appliquer("niveaux_de_gris", image_couleur).ndim == 2


def test_canal_bgr_isole_le_bon_plan() -> None:
    image = np.zeros((4, 4, 3), np.uint8)
    image[:, :] = (10, 20, 30)  # B=10, G=20, R=30
    rouge = appliquer("canal_bgr", image, canal="Rouge", mode="Couleur primitive")
    assert rouge[0, 0, 2] == 30 and rouge[0, 0, 0] == 0 and rouge[0, 0, 1] == 0
    plan = appliquer("canal_bgr", image, canal="Bleu", mode="Niveaux de gris")
    assert plan.ndim == 2 and plan[0, 0] == 10


def test_canal_hsv_teinte_bornee_a_179() -> None:
    """Rappel du TP2 : la teinte d'OpenCV vaut 0-179, pas 0-359."""
    arc = np.zeros((1, 180, 3), np.uint8)
    for teinte in range(180):
        arc[0, teinte] = (teinte, 255, 255)
    rgb = cv2.cvtColor(arc, cv2.COLOR_HSV2BGR)
    plan = appliquer("canal_hsv", rgb, canal="Teinte (H)", etirer=False)
    assert plan.max() <= 179


def test_ajuster_hsv_decalage_circulaire() -> None:
    """Un décalage de 180 ramène exactement à la teinte de départ."""
    image = np.full((4, 4, 3), 0, np.uint8)
    image[:, :] = cv2.cvtColor(np.uint8([[[100, 200, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
    inchangee = appliquer("ajuster_hsv", image, decalage_teinte=0)
    tour_complet = appliquer("ajuster_hsv", image, decalage_teinte=179)
    assert np.array_equal(inchangee, image)
    assert not np.array_equal(tour_complet, image)


def test_ajuster_hsv_saturation_nulle_donne_du_gris(image_couleur: np.ndarray) -> None:
    sortie = appliquer("ajuster_hsv", image_couleur, gain_saturation=0.0)
    bleu, vert, rouge = cv2.split(sortie)
    assert np.array_equal(bleu, vert) and np.array_equal(vert, rouge)


def test_masque_couleur_intervalle_a_cheval() -> None:
    """Teinte min > max : l'intervalle traverse le rouge, union de deux plages."""
    rouge = np.zeros((8, 8, 3), np.uint8)
    rouge[:, :] = (0, 0, 255)
    masque = appliquer("masque_couleur_hsv", rouge, teinte_min=170, teinte_max=10,
                       rendu="Masque binaire")
    assert masque.ndim == 2
    assert np.count_nonzero(masque) == masque.size


def test_masque_couleur_rendu_segmentation(image_couleur: np.ndarray) -> None:
    sortie = appliquer("masque_couleur_hsv", image_couleur, rendu="Segmentation")
    assert sortie.ndim == 3
    infos = infos_de("masque_couleur_hsv", image_couleur)
    assert "%" in str(infos["pixels sélectionnés"])


# ---------------------------------------------------------------------------
# Luminosité, contraste
# ---------------------------------------------------------------------------
def test_luminosite_augmente_la_moyenne(image_couleur: np.ndarray) -> None:
    clair = appliquer("luminosite_contraste", image_couleur, alpha=1.0, beta=40)
    assert clair.mean() > image_couleur.mean()


def test_contraste_nul_donne_une_image_noire(image_couleur: np.ndarray) -> None:
    assert appliquer("luminosite_contraste", image_couleur, alpha=0.0,
                     beta=0).max() == 0


def test_luminosite_ne_deborde_pas() -> None:
    """255 + 100 doit donner 255, pas un repliement à 100."""
    blanc = np.full((4, 4), 255, np.uint8)
    assert appliquer("luminosite_contraste", blanc, alpha=1.0, beta=100).max() == 255


def test_gamma_neutre_et_sens(image_couleur: np.ndarray) -> None:
    assert np.array_equal(
        appliquer("correction_gamma", image_couleur, gamma=1.0), image_couleur
    )
    clair = appliquer("correction_gamma", image_couleur, gamma=2.2)
    sombre = appliquer("correction_gamma", image_couleur, gamma=0.5)
    assert clair.mean() > image_couleur.mean() > sombre.mean()


def test_gamma_preserve_les_extremes() -> None:
    bornes = np.array([[0, 255]], np.uint8)
    sortie = appliquer("correction_gamma", bornes, gamma=3.0)
    assert sortie[0, 0] == 0 and sortie[0, 1] == 255


def test_negatif_est_involutif(image_couleur: np.ndarray) -> None:
    double = appliquer("negatif", appliquer("negatif", image_couleur))
    assert np.array_equal(double, image_couleur)


def test_posterisation_reduit_les_niveaux(image_couleur: np.ndarray) -> None:
    sortie = appliquer("posterisation", image_couleur, niveaux=3)
    # Au plus 3 niveaux distincts par canal.
    assert len(np.unique(sortie[:, :, 0])) <= 3


# ---------------------------------------------------------------------------
# Géométrie
# ---------------------------------------------------------------------------
def test_redimensionner_taille_exacte(image_couleur: np.ndarray) -> None:
    sortie = appliquer("redimensionner", image_couleur, pourcentage=50.0)
    assert sortie.shape[0] == round(image_couleur.shape[0] * 0.5)
    assert sortie.shape[1] == round(image_couleur.shape[1] * 0.5)


def test_redimensionner_jamais_vide() -> None:
    minuscule = np.full((3, 3, 3), 100, np.uint8)
    assert appliquer("redimensionner", minuscule, pourcentage=5.0).size > 0


def test_recadrer_region(image_couleur: np.ndarray) -> None:
    sortie = appliquer("recadrer", image_couleur, x=25.0, y=25.0,
                       largeur=50.0, hauteur=50.0)
    assert sortie.shape[0] == pytest.approx(image_couleur.shape[0] // 2, abs=2)
    assert sortie.shape[1] == pytest.approx(image_couleur.shape[1] // 2, abs=2)


def test_recadrer_borne_hors_cadre(image_couleur: np.ndarray) -> None:
    sortie = appliquer("recadrer", image_couleur, x=99.0, y=99.0,
                       largeur=100.0, hauteur=100.0)
    assert sortie.size > 0  # au moins un pixel, jamais une image vide


def test_rotation_identite(image_couleur: np.ndarray) -> None:
    assert np.array_equal(
        appliquer("rotation", image_couleur, angle=0.0, echelle=1.0), image_couleur
    )


def test_rotation_cadre_agrandi(image_couleur: np.ndarray) -> None:
    serre = appliquer("rotation", image_couleur, angle=45.0, conserver_cadre=False)
    large = appliquer("rotation", image_couleur, angle=45.0, conserver_cadre=True)
    assert serre.shape[:2] == image_couleur.shape[:2]
    assert large.shape[0] > image_couleur.shape[0]


def test_retourner_est_involutif(image_couleur: np.ndarray) -> None:
    for sens in ("Horizontal", "Vertical", "Les deux"):
        double = appliquer("retourner", appliquer("retourner", image_couleur,
                                                 sens=sens), sens=sens)
        assert np.array_equal(double, image_couleur)


def test_retourner_horizontal_inverse_les_colonnes() -> None:
    image = np.zeros((2, 4, 3), np.uint8)
    image[:, 0] = 255
    sortie = appliquer("retourner", image, sens="Horizontal")
    assert sortie[0, -1, 0] == 255 and sortie[0, 0, 0] == 0


def test_translation_decale(image_couleur: np.ndarray) -> None:
    sortie = appliquer("translation", image_couleur, dx=25.0, dy=0.0,
                       bord="Constante (noir)")
    assert sortie.shape == image_couleur.shape
    # La bande découverte à gauche est noire.
    assert sortie[:, :5].max() == 0


def test_perspective_taille_de_sortie(image_couleur: np.ndarray) -> None:
    sortie = appliquer("deformation_perspective", image_couleur,
                       largeur_sortie=128, hauteur_sortie=256)
    assert sortie.shape[:2] == (256, 128)


def test_perspective_points_degeneres(image_couleur: np.ndarray) -> None:
    with pytest.raises(registry.FilterError, match="alignés ou confondus"):
        appliquer("deformation_perspective", image_couleur,
                  p1x=10.0, p1y=10.0, p2x=10.0, p2y=10.0,
                  p3x=10.0, p3y=10.0, p4x=10.0, p4y=10.0)


# ---------------------------------------------------------------------------
# Bruit
# ---------------------------------------------------------------------------
def test_bruit_gaussien_reproductible(image_couleur: np.ndarray) -> None:
    a = appliquer("bruit_gaussien", image_couleur, sigma=20.0, graine=7)
    b = appliquer("bruit_gaussien", image_couleur, sigma=20.0, graine=7)
    c = appliquer("bruit_gaussien", image_couleur, sigma=20.0, graine=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_bruit_gaussien_sigma_nul_neutre(image_couleur: np.ndarray) -> None:
    assert np.array_equal(
        appliquer("bruit_gaussien", image_couleur, moyenne=0.0, sigma=0.0),
        image_couleur,
    )


def test_bruit_gaussien_augmente_l_ecart_type() -> None:
    uni = np.full((60, 60), 128, np.uint8)
    assert appliquer("bruit_gaussien", uni, sigma=25.0, graine=1).std() > 15


def test_sel_poivre_proportion_respectee() -> None:
    uni = np.full((200, 200), 128, np.uint8)
    sortie = appliquer("bruit_sel_poivre", uni, proportion=0.1, ratio_sel=0.5,
                       graine=3)
    touches = np.count_nonzero((sortie == 0) | (sortie == 255))
    assert touches / sortie.size == pytest.approx(0.1, abs=0.02)


def test_sel_poivre_que_du_sel() -> None:
    uni = np.full((100, 100), 128, np.uint8)
    sortie = appliquer("bruit_sel_poivre", uni, proportion=0.2, ratio_sel=1.0,
                       graine=5)
    assert np.count_nonzero(sortie == 0) == 0
    assert np.count_nonzero(sortie == 255) > 0


def test_sel_poivre_proportion_nulle(image_couleur: np.ndarray) -> None:
    assert np.array_equal(
        appliquer("bruit_sel_poivre", image_couleur, proportion=0.0), image_couleur
    )


def test_sel_poivre_touche_tout_le_pixel() -> None:
    """Un pixel « sel » doit être blanc sur les trois canaux, pas sur un seul."""
    uni = np.full((80, 80, 3), 100, np.uint8)
    sortie = appliquer("bruit_sel_poivre", uni, proportion=0.3, ratio_sel=1.0,
                       graine=11)
    blancs = np.all(sortie == 255, axis=2)
    partiels = np.any(sortie == 255, axis=2)
    assert np.array_equal(blancs, partiels)


def test_bruit_uniforme_intervalle_vide(image_couleur: np.ndarray) -> None:
    assert np.array_equal(
        appliquer("bruit_uniforme", image_couleur, minimum=0.0, maximum=0.0),
        image_couleur,
    )


def test_bruit_multiplicatif_depend_du_signal() -> None:
    """Le bruit de speckle est proportionnel : le noir reste noir."""
    image = np.zeros((40, 40), np.uint8)
    image[:, 20:] = 200
    sortie = appliquer("bruit_multiplicatif", image, sigma=0.5, graine=4)
    assert sortie[:, :20].max() == 0
    assert sortie[:, 20:].std() > 10


# ---------------------------------------------------------------------------
# Lissage
# ---------------------------------------------------------------------------
def test_lissages_reduisent_l_ecart_type(image_couleur: np.ndarray) -> None:
    depart = image_couleur.std()
    for identifiant, params in [
        ("filtre_moyenneur", {"largeur_noyau": 7, "hauteur_noyau": 7}),
        ("filtre_gaussien", {"taille_noyau": 7, "sigma_x": 2.0}),
        ("filtre_median", {"taille_noyau": 7}),
        ("filtre_bilateral", {"diametre": 7}),
    ]:
        assert appliquer(identifiant, image_couleur, **params).std() < depart, identifiant


def test_noyau_1x1_est_neutre(image_couleur: np.ndarray) -> None:
    assert np.array_equal(
        appliquer("filtre_moyenneur", image_couleur, largeur_noyau=1,
                  hauteur_noyau=1),
        image_couleur,
    )


def test_median_bat_la_moyenne_sur_sel_et_poivre() -> None:
    """Le résultat clé du TP7 ex.3, vérifié chiffres en main."""
    propre = np.full((120, 120), 128, np.uint8)
    propre[40:80, 40:80] = 200
    bruitee = appliquer("bruit_sel_poivre", propre, proportion=0.1,
                        ratio_sel=0.5, graine=2)

    median = appliquer("filtre_median", bruitee, taille_noyau=5)
    moyenne = appliquer("filtre_moyenneur", bruitee, largeur_noyau=5,
                        hauteur_noyau=5)
    gaussien = appliquer("filtre_gaussien", bruitee, taille_noyau=5, sigma_x=0.0)

    assert metrics.mse(propre, median) < metrics.mse(propre, moyenne)
    assert metrics.mse(propre, median) < metrics.mse(propre, gaussien)


def test_gaussien_bat_la_moyenne_sur_bruit_gaussien() -> None:
    propre = np.full((120, 120), 128, np.uint8)
    propre[30:90, 30:90] = 190
    bruitee = appliquer("bruit_gaussien", propre, sigma=22.0, graine=6)
    gaussien = appliquer("filtre_gaussien", bruitee, taille_noyau=5, sigma_x=1.4)
    assert metrics.mse(propre, gaussien) < metrics.mse(propre, bruitee)


def test_bilateral_preserve_mieux_le_contour_que_le_gaussien() -> None:
    """Propriété annoncée du filtre bilatéral, mesurée sur une marche d'escalier."""
    image = np.full((80, 80), 60, np.uint8)
    image[:, 40:] = 200
    bilateral = appliquer("filtre_bilateral", image, diametre=9,
                          sigma_couleur=30.0, sigma_espace=30.0)
    gaussien = appliquer("filtre_gaussien", image, taille_noyau=9, sigma_x=3.0)
    assert metrics.mse(image, bilateral) < metrics.mse(image, gaussien)


def test_noyau_personnalise_identite(image_couleur: np.ndarray) -> None:
    sortie = appliquer("noyau_personnalise", image_couleur, modele="Identité",
                       taille=3, normaliser=True, decalage=0)
    assert np.array_equal(sortie, image_couleur)


def test_noyau_personnalise_moyenne_equivaut_au_moyenneur(
    image_couleur: np.ndarray,
) -> None:
    """Le noyau construit à la main doit égaler cv2.blur (TP7 ex.1)."""
    manuel = appliquer("noyau_personnalise", image_couleur, modele="Moyenneur",
                       taille=3, normaliser=True, decalage=0)
    opencv = appliquer("filtre_moyenneur", image_couleur, largeur_noyau=3,
                       hauteur_noyau=3)
    assert metrics.mse(manuel, opencv) < 1.0


def test_noyau_personnalise_informe_du_noyau(image_couleur: np.ndarray) -> None:
    infos = infos_de("noyau_personnalise", image_couleur, modele="Moyenneur",
                     taille=3, normaliser=True)
    assert "0.111" in infos["noyau"]
    assert float(infos["somme des coefficients"]) == pytest.approx(1.0, abs=1e-3)


def test_unsharp_accentue_les_contours() -> None:
    image = np.full((60, 60), 100, np.uint8)
    image[:, 30:] = 150
    doux = cv2.GaussianBlur(image, (5, 5), 1.5)
    accentue = appliquer("nettete_unsharp", doux, taille_noyau=5, sigma=1.5,
                         intensite=1.5, seuil=0)
    # L'écart entre les deux plateaux redevient plus net.
    assert accentue.std() > doux.std()


def test_unsharp_intensite_nulle_neutre(image_couleur: np.ndarray) -> None:
    sortie = appliquer("nettete_unsharp", image_couleur, intensite=0.0, seuil=0)
    assert metrics.mse(sortie, image_couleur) < 1.0


# ---------------------------------------------------------------------------
# Contours
# ---------------------------------------------------------------------------
def test_sobel_x_voit_les_transitions_verticales() -> None:
    """Gx réagit à une marche verticale, Gy n'y voit rien."""
    image = np.zeros((40, 40), np.uint8)
    image[:, 20:] = 255
    gx = appliquer("sobel", image, direction="Horizontal (∂/∂x)", flou_prealable=0)
    gy = appliquer("sobel", image, direction="Vertical (∂/∂y)", flou_prealable=0)
    assert gx.max() > 200
    assert gy.max() < 30


def test_sobel_module_isotrope() -> None:
    horizontale = np.zeros((40, 40), np.uint8)
    horizontale[20:, :] = 255
    verticale = horizontale.T.copy()
    module_h = appliquer("sobel", horizontale, direction="Module du gradient",
                         flou_prealable=0)
    module_v = appliquer("sobel", verticale, direction="Module du gradient",
                         flou_prealable=0)
    assert module_h.max() == module_v.max()


def test_sobel_orientation_dans_les_bornes(image_gris: np.ndarray) -> None:
    sortie = appliquer("sobel", image_gris, direction="Orientation du gradient")
    assert sortie.dtype == np.uint8 and sortie.max() <= 255


def test_scharr_module_non_nul(image_gris: np.ndarray) -> None:
    assert appliquer("scharr", image_gris).max() > 0


def test_laplacien_signe_centre_sur_128() -> None:
    image = np.zeros((40, 40), np.uint8)
    image[:, 20:] = 255
    signe = appliquer("laplacien", image, garder_signe=True, flou_prealable=0)
    absolu = appliquer("laplacien", image, garder_signe=False, flou_prealable=0)
    assert signe.min() < 128 < signe.max()   # les deux côtés du contour
    assert absolu.min() == 0


def test_canny_est_binaire(image_gris: np.ndarray) -> None:
    contours = appliquer("canny", image_gris)
    assert set(np.unique(contours)).issubset({0, 255})


def test_canny_seuils_hauts_gardent_moins_de_contours(image_gris: np.ndarray) -> None:
    bas = appliquer("canny", image_gris, seuil_bas=20, seuil_haut=60)
    haut = appliquer("canny", image_gris, seuil_bas=150, seuil_haut=300)
    assert np.count_nonzero(haut) < np.count_nonzero(bas)


def test_canny_reordonne_les_seuils(image_gris: np.ndarray) -> None:
    """Seuils inversés : l'application corrige et le signale, sans échouer."""
    infos = infos_de("canny", image_gris, seuil_bas=200, seuil_haut=50)
    assert infos["seuils utilisés"] == "50 / 200"
    assert "réordonnés" in str(infos["note"])


def test_flou_prealable_reduit_le_bruit_des_contours() -> None:
    rng = np.random.default_rng(3)
    bruit = rng.integers(0, 256, (120, 120), dtype=np.uint8)
    sans = appliquer("canny", bruit, seuil_bas=50, seuil_haut=150,
                     flou_prealable=0)
    avec = appliquer("canny", bruit, seuil_bas=50, seuil_haut=150,
                     flou_prealable=7)
    assert np.count_nonzero(avec) < np.count_nonzero(sans)


# ---------------------------------------------------------------------------
# Seuillage
# ---------------------------------------------------------------------------
def test_seuillage_binaire() -> None:
    degrade = np.tile(np.arange(256, dtype=np.uint8), (4, 1))
    sortie = appliquer("seuillage_simple", degrade, seuil=127, valeur_max=255,
                       type_seuil="Binaire")
    assert sortie[0, 100] == 0 and sortie[0, 200] == 255
    assert set(np.unique(sortie)).issubset({0, 255})


def test_seuillage_inverse() -> None:
    degrade = np.tile(np.arange(256, dtype=np.uint8), (4, 1))
    sortie = appliquer("seuillage_simple", degrade, seuil=127,
                       type_seuil="Binaire inversé")
    assert sortie[0, 100] == 255 and sortie[0, 200] == 0


def test_seuillage_troncature() -> None:
    degrade = np.tile(np.arange(256, dtype=np.uint8), (4, 1))
    sortie = appliquer("seuillage_simple", degrade, seuil=100,
                       type_seuil="Troncature")
    assert sortie.max() == 100 and sortie[0, 50] == 50


def test_otsu_trouve_le_seuil_entre_deux_modes() -> None:
    """Histogramme bimodal : Otsu doit trancher entre les deux bosses.

    Les deux modes sont bruités : sur une image à exactement deux valeurs,
    tout seuil compris entre elles sépare aussi bien, et OpenCV renvoie alors
    la plus petite — ce qui rendrait le test arbitraire.
    """
    rng = np.random.default_rng(31)
    image = np.clip(rng.normal(60, 10, (100, 100)), 0, 255).astype(np.uint8)
    image[:, 50:] = np.clip(
        rng.normal(190, 10, (100, 50)), 0, 255
    ).astype(np.uint8)

    seuillee, infos = registry.get("seuillage_automatique").apply(
        image, {"methode": "Otsu", "flou_prealable": 0, "type_seuil": "Binaire"}
    )
    assert 80 < infos["seuil calculé"] < 170
    # La séparation retrouve bien les deux moitiés de l'image.
    assert seuillee[:, :50].mean() < 20
    assert seuillee[:, 50:].mean() > 235


def test_otsu_sur_image_a_deux_valeurs_exactes() -> None:
    """Cas dégénéré documenté : tout seuil entre les deux valeurs est optimal."""
    image = np.full((60, 60), 50, np.uint8)
    image[:, 30:] = 200
    seuillee, infos = registry.get("seuillage_automatique").apply(
        image, {"methode": "Otsu", "flou_prealable": 0}
    )
    assert 50 <= infos["seuil calculé"] < 200
    assert seuillee[:, :30].max() == 0 and seuillee[:, 30:].min() == 255


def test_seuillage_adaptatif_resiste_a_un_eclairage_inegal() -> None:
    """Un dégradé de fond met en échec le seuil global, pas l'adaptatif."""
    hauteur, largeur = 80, 160
    fond = np.tile(np.linspace(20, 230, largeur, dtype=np.uint8), (hauteur, 1))
    image = fond.copy()
    # Des traits sombres, de contraste constant par rapport au fond local.
    for colonne in range(10, largeur, 20):
        image[:, colonne:colonne + 4] = np.clip(
            fond[:, colonne:colonne + 4].astype(np.int16) - 40, 0, 255
        ).astype(np.uint8)

    adaptatif = appliquer("seuillage_adaptatif", image, methode="Moyenne",
                          taille_bloc=15, constante=5,
                          type_seuil="Binaire inversé")
    global_ = appliquer("seuillage_simple", image, seuil=127,
                        type_seuil="Binaire inversé")
    # Le seuil global coupe l'image en deux moitiés ; l'adaptatif retrouve les traits.
    assert np.count_nonzero(adaptatif) < np.count_nonzero(global_)
    # Les traits sont retrouvés sur toute la largeur, y compris côté clair.
    assert adaptatif[:, largeur // 2:].any()


def test_seuillage_par_bande() -> None:
    degrade = np.tile(np.arange(256, dtype=np.uint8), (4, 1))
    masque = appliquer("seuillage_par_bande", degrade, minimum=100, maximum=150)
    assert masque[0, 120] == 255 and masque[0, 50] == 0 and masque[0, 200] == 0
    inverse = appliquer("seuillage_par_bande", degrade, minimum=100, maximum=150,
                        inverser=True)
    assert inverse[0, 120] == 0


def test_seuillage_bande_reordonne() -> None:
    degrade = np.tile(np.arange(256, dtype=np.uint8), (4, 1))
    infos = infos_de("seuillage_par_bande", degrade, minimum=200, maximum=50)
    assert infos["bande"] == "[50, 200]"


# ---------------------------------------------------------------------------
# Morphologie
# ---------------------------------------------------------------------------
def test_erosion_retrecit_dilatation_grossit() -> None:
    image = np.zeros((60, 60), np.uint8)
    image[20:40, 20:40] = 255
    erode = appliquer("morphologie", image, operation="Érosion", forme="Rectangle",
                      taille=5, iterations=1)
    dilate = appliquer("morphologie", image, operation="Dilatation",
                       forme="Rectangle", taille=5, iterations=1)
    assert np.count_nonzero(erode) < np.count_nonzero(image)
    assert np.count_nonzero(dilate) > np.count_nonzero(image)


def test_ouverture_supprime_le_bruit_clair_isole() -> None:
    image = np.zeros((80, 80), np.uint8)
    image[20:60, 20:60] = 255        # grand objet
    image[5, 5] = 255                # point parasite
    image[70, 70] = 255
    ouverte = appliquer("morphologie", image, operation="Ouverture",
                        forme="Rectangle", taille=3, iterations=1)
    assert ouverte[5, 5] == 0 and ouverte[70, 70] == 0
    assert np.count_nonzero(ouverte) > 1000  # l'objet principal est préservé


def test_fermeture_bouche_les_trous() -> None:
    image = np.zeros((60, 60), np.uint8)
    image[15:45, 15:45] = 255
    image[28:32, 28:32] = 0          # trou au centre
    fermee = appliquer("morphologie", image, operation="Fermeture",
                       forme="Rectangle", taille=5, iterations=1)
    assert fermee[30, 30] == 255


def test_gradient_morphologique_est_un_contour() -> None:
    image = np.zeros((60, 60), np.uint8)
    image[20:40, 20:40] = 255
    contour = appliquer("squelette_contour_morpho", image, forme="Rectangle",
                        taille=3)
    assert contour[30, 30] == 0      # intérieur vide
    assert contour[20, 30] > 0       # bord marqué


def test_iterations_amplifient_l_effet() -> None:
    image = np.zeros((60, 60), np.uint8)
    image[20:40, 20:40] = 255
    une = appliquer("morphologie", image, operation="Érosion", taille=3,
                    iterations=1, forme="Rectangle")
    trois = appliquer("morphologie", image, operation="Érosion", taille=3,
                      iterations=3, forme="Rectangle")
    assert np.count_nonzero(trois) < np.count_nonzero(une)


# ---------------------------------------------------------------------------
# Histogramme
# ---------------------------------------------------------------------------
def test_egalisation_etale_l_histogramme() -> None:
    """Une image à faible contraste doit couvrir plus de niveaux après égalisation."""
    rng = np.random.default_rng(12)
    terne = (rng.normal(128, 8, (120, 120))).clip(100, 160).astype(np.uint8)
    egalisee = appliquer("egalisation_histogramme", terne)
    assert egalisee.std() > terne.std()
    assert (egalisee.max() - egalisee.min()) > (terne.max() - terne.min())


def test_egalisation_preserve_les_couleurs_via_la_luminance() -> None:
    """Égaliser Y ne doit pas tordre les couleurs autant qu'égaliser B, G et R."""
    rng = np.random.default_rng(13)
    image = np.zeros((80, 80, 3), np.uint8)
    image[:, :, 0] = rng.integers(90, 130, (80, 80))
    image[:, :, 1] = rng.integers(60, 100, (80, 80))
    image[:, :, 2] = rng.integers(120, 160, (80, 80))

    def teinte_moyenne(img: np.ndarray) -> float:
        return float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0].mean())

    par_luminance = appliquer("egalisation_histogramme", image,
                              espace="Luminance Y (YCrCb)")
    par_canaux = appliquer("egalisation_histogramme", image,
                           espace="Trois canaux BGR (déforme les couleurs)")
    depart = teinte_moyenne(image)
    assert abs(teinte_moyenne(par_luminance) - depart) < abs(
        teinte_moyenne(par_canaux) - depart
    )


def test_clahe_augmente_le_contraste_local() -> None:
    rng = np.random.default_rng(14)
    terne = (rng.normal(128, 6, (120, 120))).clip(110, 146).astype(np.uint8)
    assert appliquer("clahe", terne, limite=4.0, tuiles=8).std() > terne.std()


def test_etirement_couvre_toute_la_plage() -> None:
    terne = np.tile(np.linspace(80, 170, 100, dtype=np.uint8), (40, 1))
    etiree = appliquer("etirement_histogramme", terne, percentile_bas=0.0,
                       percentile_haut=0.0)
    assert etiree.min() == 0 and etiree.max() == 255


def test_etirement_image_uniforme() -> None:
    uni = np.full((20, 20), 128, np.uint8)
    sortie, infos = registry.get("etirement_histogramme").apply(uni, {})
    assert np.array_equal(sortie, uni)
    assert "inchangée" in str(infos["note"])


def test_trace_histogramme_dimensions(image_couleur: np.ndarray) -> None:
    sortie = appliquer("trace_histogramme", image_couleur, largeur=320,
                       hauteur=200)
    assert sortie.shape == (200, 320, 3)


# ---------------------------------------------------------------------------
# Fusion et opérations binaires
# ---------------------------------------------------------------------------
def test_fusion_extremes(image_couleur: np.ndarray,
                        image_reference: np.ndarray) -> None:
    definition = registry.get("fusion")
    seule, _ = definition.apply(image_couleur, {"alpha": 1.0}, image_reference)
    autre, _ = definition.apply(image_couleur, {"alpha": 0.0}, image_reference)
    assert metrics.mse(seule, image_couleur) < 1.0
    assert metrics.mse(autre, im.match_shape(image_couleur, image_reference)) < 1.0


def test_fusion_milieu_entre_les_deux(image_couleur, image_reference) -> None:
    definition = registry.get("fusion")
    milieu, _ = definition.apply(image_couleur, {"alpha": 0.5}, image_reference)
    assert metrics.mse(milieu, image_couleur) > 0
    assert metrics.mse(milieu, image_couleur) < metrics.mse(
        im.match_shape(image_couleur, image_reference), image_couleur
    )


def test_fusion_redimensionne_la_reference(image_couleur, image_reference) -> None:
    definition = registry.get("fusion")
    sortie, _ = definition.apply(image_couleur, {"alpha": 0.5}, image_reference)
    assert sortie.shape[:2] == image_couleur.shape[:2]


def test_operations_binaires() -> None:
    definition = registry.get("operation_binaire")
    a = np.array([[0b1100]], np.uint8)
    b = np.array([[0b1010]], np.uint8)
    assert definition.apply(a, {"operation": "ET (AND)"}, b)[0][0, 0] == 0b1000
    assert definition.apply(a, {"operation": "OU (OR)"}, b)[0][0, 0] == 0b1110
    assert definition.apply(a, {"operation": "OU exclusif (XOR)"}, b)[0][0, 0] == 0b0110
    assert definition.apply(a, {"operation": "NON (NOT, image courante)"}, b)[0][0, 0] == 255 - 0b1100
    assert definition.apply(a, {"operation": "Différence absolue"}, b)[0][0, 0] == 2


def test_masque_par_reference() -> None:
    definition = registry.get("masque_par_reference")
    image = np.full((20, 20), 200, np.uint8)
    masque = np.zeros((20, 20), np.uint8)
    masque[:, 10:] = 255
    sortie, infos = definition.apply(image, {"seuil": 127, "sens": "Zones claires"},
                                     masque)
    assert sortie[:, :10].max() == 0 and sortie[:, 10:].min() == 200
    assert "50" in str(infos["surface conservée"])

    inverse, _ = definition.apply(image, {"sens": "Zones sombres"}, masque)
    assert inverse[:, 10:].max() == 0


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------
def test_annoter_texte_modifie_l_image(image_couleur: np.ndarray) -> None:
    sortie = appliquer("annoter_texte", image_couleur, texte="Essai", x=5.0,
                       y=50.0, taille=1.0, epaisseur=2)
    assert not np.array_equal(sortie, image_couleur)


def test_annoter_texte_vide_neutre(image_couleur: np.ndarray) -> None:
    sortie = appliquer("annoter_texte", image_couleur, texte="")
    assert np.array_equal(sortie, im.to_bgr(image_couleur))


def test_annoter_texte_sur_image_grise(image_gris: np.ndarray) -> None:
    sortie = appliquer("annoter_texte", image_gris, texte="Gris")
    assert sortie.ndim == 3  # l'annotation force la couleur


@pytest.mark.parametrize("forme", ["Ligne", "Rectangle", "Cercle", "Ellipse"])
def test_annoter_formes(forme: str, image_couleur: np.ndarray) -> None:
    sortie = appliquer("annoter_forme", forme=forme, image=image_couleur,
                       couleur=(0, 0, 255), epaisseur=3)
    assert not np.array_equal(sortie, image_couleur)


def test_grille_reperes(image_couleur: np.ndarray) -> None:
    sortie = appliquer("grille_reperes", image_couleur, pas=20, etiquettes=True)
    assert sortie.shape[:2] == image_couleur.shape[:2]
    assert not np.array_equal(sortie, image_couleur)
