"""Tests de l'interface web, pilotée par ``streamlit.testing``.

``AppTest`` exécute le script de l'application comme le ferait le serveur
Streamlit, sans navigateur. Tout le code des onglets s'exécute (Streamlit rend
les onglets côté serveur), donc une erreur dans n'importe quel panneau est
détectée ici.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cvlab import registry
from cvlab.pipeline import Pipeline

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

# Chemin absolu : AppTest.from_file résout un chemin relatif par rapport au
# fichier appelant (ici tests/), pas par rapport au dossier de travail.
RACINE = Path(__file__).resolve().parent.parent
CHEMIN_APP = str(RACINE / "ui" / "streamlit_app.py")
DELAI = 300


def _etapes(identifiants: list[str]) -> list[dict]:
    return [
        {
            "uid": index + 1,
            "filtre": identifiant,
            "parametres": registry.get(identifiant).defaults(),
            "actif": True,
        }
        for index, identifiant in enumerate(identifiants)
    ]


@pytest.fixture()
def app() -> AppTest:
    essai = AppTest.from_file(CHEMIN_APP, default_timeout=DELAI)
    essai.run()
    assert not essai.exception, essai.exception
    return essai


def test_demarrage_sans_image(app: AppTest) -> None:
    """Sans image, l'application invite à en choisir une et reste utilisable."""
    assert app.title[0].value.startswith("Atelier de filtres")
    assert any("barre latérale" in info.value for info in app.info)


def test_mire_de_test_charge_une_image(app: AppTest) -> None:
    app.sidebar.radio[0].set_value("Mire de test").run()
    assert not app.exception, app.exception
    assert app.session_state["source_image"] is not None
    assert app.session_state["source_image"].ndim == 3


def test_tous_les_onglets_avec_une_chaine_complete(app: AppTest) -> None:
    """Une chaîne traversant toutes les familles ne doit rien casser."""
    app.sidebar.radio[0].set_value("Mire de test").run()
    app.session_state["reference_image"] = np.full((40, 60, 3), 128, np.uint8)
    app.session_state["reference_nom"] = "référence de test"
    identifiants = [
        "canal_bgr", "luminosite_contraste", "redimensionner",
        "bruit_sel_poivre", "filtre_median", "sobel", "seuillage_automatique",
        "morphologie", "clahe", "fusion", "annoter_texte", "grille_reperes",
        "trace_histogramme",
    ]
    app.session_state["etapes"] = _etapes(identifiants)
    app.session_state["compteur_etapes"] = len(identifiants)
    app.run()
    assert not app.exception, app.exception
    assert not app.error, [e.value for e in app.error]


@pytest.mark.parametrize("identifiant", [d.id for d in registry.all_filters()])
def test_chaque_filtre_s_affiche(identifiant: str) -> None:
    """Chaque filtre du catalogue doit pouvoir être réglé dans l'interface.

    C'est ce test qui attrape un paramètre dont le type n'aurait pas de widget,
    ou dont les bornes empêcheraient la création d'un curseur.
    """
    app = AppTest.from_file(CHEMIN_APP, default_timeout=DELAI)
    app.run()
    app.sidebar.radio[0].set_value("Mire de test").run()
    app.session_state["reference_image"] = np.full((40, 60, 3), 128, np.uint8)
    app.session_state["etapes"] = _etapes([identifiant])
    app.session_state["compteur_etapes"] = 1
    app.run()
    assert not app.exception, f"{identifiant}: {app.exception}"
    assert not app.error, f"{identifiant}: {[e.value for e in app.error]}"


def test_etape_desactivee(app: AppTest) -> None:
    app.sidebar.radio[0].set_value("Mire de test").run()
    app.session_state["etapes"] = _etapes(["negatif"])
    app.session_state["etapes"][0]["actif"] = False
    app.run()
    assert not app.exception, app.exception


def test_filtre_a_deux_images_sans_reference_avertit(app: AppTest) -> None:
    app.sidebar.radio[0].set_value("Mire de test").run()
    app.session_state["etapes"] = _etapes(["fusion"])
    app.session_state["reference_image"] = None
    app.run()
    assert not app.exception, app.exception
    textes = [w.value for w in app.warning]
    assert any("deuxième image" in t for t in textes), textes


def test_presets_fournis_se_chargent(app: AppTest) -> None:
    app.sidebar.radio[0].set_value("Mire de test").run()
    dossier = RACINE / "presets"
    for fichier in sorted(dossier.glob("*.json")):
        chaine = Pipeline.load(fichier)
        app.session_state["etapes"] = [
            {
                "uid": index + 1,
                "filtre": step.filter_id,
                "parametres": step.normalised_params(),
                "actif": step.enabled,
            }
            for index, step in enumerate(chaine.steps)
        ]
        app.session_state["compteur_etapes"] = len(chaine.steps)
        app.session_state["nom_chaine"] = chaine.nom
        app.run()
        assert not app.exception, f"{fichier.name}: {app.exception}"


def test_resolution_de_travail(app: AppTest) -> None:
    app.sidebar.radio[0].set_value("Mire de test").run()
    app.session_state["taille_travail"] = 400
    app.run()
    assert not app.exception, app.exception
    # La mire mesure 640×420 : à 400 px de côté maximum, elle est réduite.
    assert max(app.session_state["source_image"].shape[:2]) == 640


def test_ajout_de_filtre_par_bouton(app: AppTest) -> None:
    app.sidebar.radio[0].set_value("Mire de test").run()
    avant = len(app.session_state["etapes"])
    boutons = [b for b in app.button if b.label.startswith("Ajouter «")]
    assert boutons, "bouton d'ajout introuvable"
    boutons[0].click().run()
    assert not app.exception, app.exception
    assert len(app.session_state["etapes"]) == avant + 1


def test_mire_de_demonstration_contient_du_contraste() -> None:
    from ui.streamlit_app import image_de_demonstration, reduire_pour_travail

    mire = image_de_demonstration(320, 200)
    assert mire.shape == (200, 320, 3)
    assert mire.std() > 20  # du contraste, donc des contours à détecter

    reduite, modifie = reduire_pour_travail(mire, 100)
    assert modifie and max(reduite.shape[:2]) == 100
    inchangee, modifie = reduire_pour_travail(mire, 0)
    assert not modifie and inchangee.shape == mire.shape


def test_coordonnees_de_pixel_bornees_apres_retrecissement(app: AppTest) -> None:
    """Changer pour une image plus petite ne doit pas casser l'inspecteur.

    Streamlit refuse une valeur mémorisée supérieure au maximum du widget :
    sans écrêtage préalable, passer d'une grande image à une petite après avoir
    choisi des coordonnées élevées ferait échouer la page.
    """
    app.sidebar.radio[0].set_value("Mire de test").run()
    app.session_state["pixel_x"] = 600
    app.session_state["pixel_y"] = 400
    # La chaîne réduit fortement l'image : les coordonnées deviennent invalides.
    app.session_state["etapes"] = _etapes(["redimensionner"])
    app.session_state["etapes"][0]["parametres"]["pourcentage"] = 10.0
    app.session_state["taille_travail"] = 400
    app.run()
    assert not app.exception, app.exception
    assert app.session_state["pixel_x"] <= 400


def test_decodage_non_repete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un fichier téléversé n'est décodé qu'une fois.

    Streamlit renvoie le même objet à chaque réexécution du script ; sans la
    signature mémorisée, l'image serait redécodée à chaque mouvement de
    curseur.
    """
    import streamlit as st

    from cvlab import io_utils as io_module
    from ui import streamlit_app

    donnees = io_module.encode_image(np.full((40, 60, 3), 70, np.uint8), ".png")

    class FauxFichier:
        name = "essai.png"
        size = len(donnees)
        file_id = "abc"

        @staticmethod
        def getvalue() -> bytes:
            return donnees

    appels = {"n": 0}
    vrai_decode = io_module.decode_image

    def compter(*args, **kwargs):
        appels["n"] += 1
        return vrai_decode(*args, **kwargs)

    monkeypatch.setattr(streamlit_app.io_utils, "decode_image", compter)
    st.session_state["source_signature"] = None
    st.session_state["source_image"] = None
    st.session_state["source_nom"] = ""

    streamlit_app._charger_si_nouveau(FauxFichier(), "fichier")
    streamlit_app._charger_si_nouveau(FauxFichier(), "fichier")

    assert appels["n"] == 1
    assert st.session_state["source_image"].shape == (40, 60, 3)
    assert st.session_state["source_nom"] == "essai.png"
