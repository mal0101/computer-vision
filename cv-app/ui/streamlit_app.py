"""Atelier de filtres — interface web (Streamlit).

Lancement depuis le dossier ``cv-app`` ::

    streamlit run ui/streamlit_app.py

Ce que propose l'interface, point par point avec l'énoncé du projet :

* **l'image vient de l'utilisateur** : fichier téléversé, instantané de webcam
  (capteur du navigateur ou caméra locale via OpenCV), ou image d'exemple
  issue des dossiers ``TP*`` du dépôt ;
* **tous les paramètres qui jouent un rôle sont saisis par l'utilisateur** :
  chaque filtre déclare ses paramètres, et l'interface construit les curseurs
  correspondants automatiquement — aucune valeur n'est codée en dur ;
* **les filtres s'enchaînent** : on empile les étapes, on les réordonne, on les
  désactive une par une pour isoler l'effet de chacune.

Le flux webcam **continu** est dans l'application de bureau
(``python ui/desktop.py --webcam``) : un navigateur ne peut pas donner à
Streamlit un flux vidéo image par image sans dépendance supplémentaire, alors
qu'OpenCV le fait nativement, comme au TP1 §3.4.
"""

from __future__ import annotations

import sys
from pathlib import Path

# L'exécution par « streamlit run » place le dossier du script sur sys.path,
# pas la racine du projet : on l'ajoute pour pouvoir importer cvlab et ui.
RACINE = Path(__file__).resolve().parent.parent
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from cvlab import __version__, histogram, io_utils, metrics, registry  # noqa: E402
from cvlab import images as im
from cvlab.pipeline import Pipeline, PipelineResult, Step  # noqa: E402
from ui import widgets  # noqa: E402

TITRE = "Atelier de filtres — Vision par ordinateur"
DOSSIER_PRESETS = RACINE / "presets"

# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=TITRE,
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      /* Densité : un atelier affiche beaucoup de réglages, on resserre. */
      .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1700px; }
      h1 { font-size: 1.55rem !important; letter-spacing: -0.015em; margin-bottom: .1rem; }
      h2 { font-size: 1.12rem !important; margin-top: .4rem; }
      h3 { font-size: .98rem !important; }
      /* Les images gardent un fond neutre : indispensable pour juger un contraste. */
      [data-testid="stImage"] img {
        border-radius: 6px;
        background: repeating-conic-gradient(#2a2d34 0% 25%, #22252b 0% 50%) 50%/16px 16px;
      }
      [data-testid="stSidebar"] { border-right: 1px solid #272b33; }
      .etape-carte {
        border: 1px solid #2a2e36; border-left: 3px solid #4cc2ff;
        border-radius: 6px; padding: .45rem .7rem; margin-bottom: .35rem;
        background: #171a1f;
      }
      .etape-carte.inactive { border-left-color: #4a4f59; opacity: .62; }
      .etape-carte.erreur { border-left-color: #ff6b6b; }
      .etiquette {
        display: inline-block; font-size: .68rem; letter-spacing: .04em;
        text-transform: uppercase; color: #8b93a1; border: 1px solid #333842;
        border-radius: 10px; padding: 0 .45rem; margin-right: .3rem;
      }
      code { font-size: .78rem !important; }
      [data-testid="stMetricValue"] { font-size: 1.15rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# État de session
# ---------------------------------------------------------------------------
def init_etat() -> None:
    """Valeurs initiales de ``st.session_state``.

    Les étapes sont stockées sous forme de dictionnaires sérialisables (et non
    d'objets :class:`Pipeline`) : l'état survit ainsi aux rechargements de code
    de Streamlit, et il s'exporte directement en preset JSON.
    """
    st.session_state.setdefault("etapes", [])
    st.session_state.setdefault("compteur_etapes", 0)
    st.session_state.setdefault("nom_chaine", "Ma chaîne")
    st.session_state.setdefault("source_image", None)
    st.session_state.setdefault("source_nom", "")
    st.session_state.setdefault("reference_image", None)
    st.session_state.setdefault("reference_nom", "")
    st.session_state.setdefault("taille_travail", 900)
    st.session_state.setdefault("source_signature", None)


def _charger_si_nouveau(fichier, etiquette: str) -> None:
    """Décode un fichier téléversé, mais seulement s'il a changé.

    Streamlit renvoie le même objet à chaque réexécution du script : sans ce
    garde-fou, l'image serait redécodée à chaque mouvement de curseur, ce qui
    rend l'interface poussive sur une photo de plusieurs mégapixels.
    """
    signature = (etiquette, getattr(fichier, "file_id", None) or fichier.name,
                 fichier.size)
    if st.session_state["source_signature"] == signature:
        return
    try:
        st.session_state["source_image"] = io_utils.decode_image(fichier.getvalue())
    except io_utils.ImageLoadError as exc:
        st.sidebar.error(str(exc))
        return
    st.session_state["source_nom"] = (
        fichier.name if etiquette == "fichier" else "webcam (navigateur)"
    )
    st.session_state["source_signature"] = signature


def nouvel_identifiant() -> int:
    st.session_state["compteur_etapes"] += 1
    return st.session_state["compteur_etapes"]


def ajouter_etape(filter_id: str, params: dict | None = None) -> None:
    definition = registry.get(filter_id)
    st.session_state["etapes"].append({
        "uid": nouvel_identifiant(),
        "filtre": definition.id,
        "parametres": definition.coerce(params),
        "actif": True,
    })


def construire_chaine() -> Pipeline:
    """Reconstruit un :class:`Pipeline` depuis l'état de session."""
    chaine = Pipeline(nom=st.session_state["nom_chaine"])
    for brute in st.session_state["etapes"]:
        chaine.steps.append(Step(
            filter_id=brute["filtre"],
            params=brute["parametres"],
            enabled=brute["actif"],
        ))
    return chaine


def charger_chaine(chaine: Pipeline) -> None:
    """Remplace la chaîne courante par ``chaine`` (import de preset)."""
    st.session_state["nom_chaine"] = chaine.nom
    st.session_state["etapes"] = [
        {
            "uid": nouvel_identifiant(),
            "filtre": step.filter_id,
            "parametres": step.normalised_params(),
            "actif": step.enabled,
        }
        for step in chaine.steps
    ]


# ---------------------------------------------------------------------------
# Sources d'image
# ---------------------------------------------------------------------------
def image_de_demonstration(largeur: int = 640, hauteur: int = 420) -> np.ndarray:
    """Mire de test fabriquée avec les fonctions de dessin du TP2.

    Elle réunit volontairement ce qui met les filtres en difficulté : des
    contours francs, un dégradé continu, des détails fins, des aplats de
    couleur et du texte.
    """
    from cvlab import drawing as ds

    toile = np.zeros((hauteur, largeur, 3), np.uint8)
    # Dégradé horizontal : utile pour juger une correction gamma ou un seuillage.
    degrade = np.linspace(0, 255, largeur, dtype=np.float32)
    toile[:, :, 0] = np.tile(degrade, (hauteur, 1)).astype(np.uint8)
    toile[:, :, 1] = 60
    toile[:, :, 2] = np.tile(degrade[::-1], (hauteur, 1)).astype(np.uint8)

    # Damier : contours francs dans les deux directions (test de Sobel).
    pas = 28
    for ligne in range(0, hauteur // 2, pas):
        for colonne in range(0, largeur // 2, pas):
            if ((ligne // pas) + (colonne // pas)) % 2 == 0:
                toile[ligne:ligne + pas, colonne:colonne + pas] = (245, 245, 245)

    # Formes pleines et traits fins (test de lissage et de morphologie).
    toile = ds.dess_cercle(toile, (int(largeur * 0.72), int(hauteur * 0.34)), 62,
                           (70, 200, 255), cv2.FILLED)
    toile = ds.dess_rectangle(toile, (int(largeur * 0.08), int(hauteur * 0.62)),
                              (int(largeur * 0.42), int(hauteur * 0.92)),
                              (120, 255, 140), cv2.FILLED)
    for index in range(10):
        x = int(largeur * 0.52) + index * 9
        toile = ds.dess_ligne(toile, (x, int(hauteur * 0.62)),
                              (x, int(hauteur * 0.95)), (255, 255, 255),
                              max(1, index % 4))
    toile = ds.dess_ellipse(toile, (int(largeur * 0.30), int(hauteur * 0.26)),
                            (74, 40), 25, 0, 360, (255, 120, 220), 3)
    toile = ds.dess_text(toile, "Mire de test", (int(largeur * 0.05), 30), 0.8,
                         (20, 20, 20), 2)
    return toile


def reduire_pour_travail(image: np.ndarray, maximum: int) -> tuple[np.ndarray, bool]:
    """Réduit l'image si son plus grand côté dépasse ``maximum``.

    Une photo de 12 Mpx rend l'interface poussive : chaque mouvement de curseur
    relance toute la chaîne. On travaille donc sur une version réduite, et on
    le signale clairement à l'utilisateur.
    """
    hauteur, largeur = image.shape[:2]
    cote = max(hauteur, largeur)
    if maximum <= 0 or cote <= maximum:
        return image, False
    facteur = maximum / float(cote)
    nouvelle = cv2.resize(
        image,
        (max(1, int(largeur * facteur)), max(1, int(hauteur * facteur))),
        interpolation=cv2.INTER_AREA,
    )
    return nouvelle, True


def panneau_source() -> None:
    """Section « Image source » de la barre latérale."""
    st.sidebar.header("1 · Image source")
    mode = st.sidebar.radio(
        "Origine de l'image",
        ("Fichier", "Webcam (instantané)", "Exemple des TP", "Mire de test"),
        key="mode_source",
        help=(
            "Le flux webcam continu est dans l'application de bureau : "
            "python ui/desktop.py --webcam"
        ),
    )

    if mode == "Fichier":
        fichier = st.sidebar.file_uploader(
            "Choisir une image",
            type=list(io_utils.EXTENSIONS_IMAGE),
            key="televersement",
        )
        if fichier is not None:
            _charger_si_nouveau(fichier, "fichier")

    elif mode == "Webcam (instantané)":
        st.sidebar.caption(
            "Capteur du navigateur : fonctionne aussi à distance, et demande "
            "l'autorisation « Caméra »."
        )
        photo = st.sidebar.camera_input("Prendre une photo", key="camera_navigateur")
        if photo is not None:
            _charger_si_nouveau(photo, "camera")

        st.sidebar.divider()
        st.sidebar.caption("Ou caméra locale, via OpenCV (TP1 §3.4) :")
        colonne_index, colonne_bouton = st.sidebar.columns([1, 2])
        index = colonne_index.number_input(
            "N°", min_value=0, max_value=8, value=0, key="index_camera_locale",
            label_visibility="collapsed",
        )
        if colonne_bouton.button("Capturer (OpenCV)", width="stretch"):
            try:
                st.session_state["source_image"] = io_utils.capture_snapshot(
                    index=int(index)
                )
                st.session_state["source_nom"] = f"webcam OpenCV {int(index)}"
                st.session_state["source_signature"] = None
            except io_utils.CameraError as exc:
                st.sidebar.error(str(exc))

    elif mode == "Exemple des TP":
        exemples = io_utils.sample_images()
        if not exemples:
            st.sidebar.info(
                "Aucune image trouvée dans les dossiers TP*. "
                "Déposer des images dans cv-app/assets/ pour les voir ici."
            )
        else:
            libelles = [f"{c.parent.name}/{c.name}" for c in exemples]
            choix = st.sidebar.selectbox("Image d'exemple", libelles,
                                         key="choix_exemple")
            if st.sidebar.button("Charger l'exemple", width="stretch"):
                chemin = exemples[libelles.index(choix)]
                try:
                    st.session_state["source_image"] = io_utils.load_image(chemin)
                    st.session_state["source_nom"] = choix
                    st.session_state["source_signature"] = None
                except io_utils.ImageLoadError as exc:
                    st.sidebar.error(str(exc))

    else:  # Mire de test
        if st.sidebar.button("Générer la mire", width="stretch") or (
            st.session_state["source_image"] is None
        ):
            st.session_state["source_image"] = image_de_demonstration()
            st.session_state["source_nom"] = "mire de test (générée)"
            st.session_state["source_signature"] = None

    st.session_state["taille_travail"] = st.sidebar.select_slider(
        "Résolution de travail (plus grand côté)",
        options=(400, 600, 900, 1200, 1600, 0),
        value=st.session_state["taille_travail"],
        format_func=lambda v: "taille réelle" if v == 0 else f"{v} px",
        help=(
            "Réduire accélère l'aperçu. Attention : la taille change l'effet "
            "apparent d'un noyau — un flou 5×5 est bien plus visible sur une "
            "image de 400 px que sur une de 4000 px."
        ),
    )


def panneau_reference() -> None:
    """Section « Deuxième image », nécessaire aux filtres de fusion."""
    chaine = construire_chaine()
    requis = chaine.needs_reference
    with st.sidebar.expander(
        "2 · Deuxième image" + (" ⚠ requise" if requis else " (fusion)"),
        expanded=requis and st.session_state["reference_image"] is None,
    ):
        st.caption(
            "Utilisée par la fusion pondérée, les opérations binaires et le "
            "masquage. Elle est automatiquement redimensionnée à la taille de "
            "l'image courante (TP4 §2)."
        )
        fichier = st.file_uploader(
            "Image de référence", type=list(io_utils.EXTENSIONS_IMAGE),
            key="televersement_reference",
        )
        if fichier is not None:
            try:
                st.session_state["reference_image"] = io_utils.decode_image(
                    fichier.getvalue()
                )
                st.session_state["reference_nom"] = fichier.name
            except io_utils.ImageLoadError as exc:
                st.error(str(exc))

        exemples = io_utils.sample_images()
        if exemples:
            libelles = ["—"] + [f"{c.parent.name}/{c.name}" for c in exemples]
            choix = st.selectbox("…ou un exemple", libelles, key="reference_exemple")
            if choix != "—" and st.button("Charger comme référence",
                                         width="stretch"):
                chemin = exemples[libelles.index(choix) - 1]
                st.session_state["reference_image"] = io_utils.load_image(chemin)
                st.session_state["reference_nom"] = choix

        if st.session_state["reference_image"] is not None:
            widgets.afficher_image(
                st.session_state["reference_image"],
                f"Référence : {st.session_state['reference_nom']}",
            )
            if st.button("Retirer la référence", width="stretch"):
                st.session_state["reference_image"] = None
                st.session_state["reference_nom"] = ""
                st.rerun()


def panneau_presets() -> None:
    """Section « Chaîne enregistrée » : import, export, exemples fournis."""
    with st.sidebar.expander("3 · Chaînes enregistrées", expanded=False):
        st.text_input("Nom de la chaîne", key="nom_chaine")
        chaine = construire_chaine()
        st.download_button(
            "Exporter en JSON",
            data=chaine.to_json(),
            file_name=f"{_nom_fichier(chaine.nom)}.json",
            mime="application/json",
            width="stretch",
            disabled=len(chaine) == 0,
        )
        importe = st.file_uploader("Importer un JSON", type=["json"],
                                   key="import_preset")
        if importe is not None and st.button("Appliquer l'import", width="stretch"):
            try:
                charger_chaine(Pipeline.from_json(importe.getvalue()))
                st.success("Chaîne importée.")
                st.rerun()
            except ValueError as exc:
                st.error(f"Import impossible : {exc}")

        fournis = sorted(DOSSIER_PRESETS.glob("*.json")) if DOSSIER_PRESETS.is_dir() else []
        if fournis:
            st.divider()
            st.caption("Chaînes d'exemple fournies :")
            libelles = [c.stem.replace("_", " ") for c in fournis]
            choix = st.selectbox("Exemple", libelles, key="preset_fourni")
            if st.button("Charger cet exemple", width="stretch"):
                try:
                    charger_chaine(Pipeline.load(fournis[libelles.index(choix)]))
                    st.rerun()
                except ValueError as exc:
                    st.error(f"Preset invalide : {exc}")


def _nom_fichier(texte: str) -> str:
    """Nom de fichier sûr à partir d'un titre libre."""
    propre = "".join(c if c.isalnum() or c in "-_ " else "_" for c in texte)
    return propre.strip().replace(" ", "_").lower() or "chaine"


# ---------------------------------------------------------------------------
# Éditeur de chaîne
# ---------------------------------------------------------------------------
def editeur_chaine(resultat: PipelineResult | None) -> None:
    """Colonne de gauche : ajout, réglage, réordonnancement des étapes."""
    st.subheader("Chaîne de traitement")

    groupes = registry.filters_by_family()
    colonne_famille, colonne_filtre = st.columns([1, 1])
    famille = colonne_famille.selectbox("Famille", list(groupes.keys()),
                                        key="famille_ajout")
    definitions = groupes[famille]
    noms = [d.name for d in definitions]
    nom_choisi = colonne_filtre.selectbox("Filtre", noms, key="filtre_ajout")
    choisi = definitions[noms.index(nom_choisi)]

    st.caption(choisi.summary)
    colonne_ajout, colonne_vider = st.columns([2, 1])
    if colonne_ajout.button(f"Ajouter « {choisi.name} »", type="primary",
                            width="stretch"):
        ajouter_etape(choisi.id)
        st.rerun()
    if colonne_vider.button("Tout vider", width="stretch",
                            disabled=not st.session_state["etapes"]):
        st.session_state["etapes"] = []
        st.rerun()

    with st.popover("Détails du filtre sélectionné", width="stretch"):
        _fiche_filtre(choisi)

    st.divider()

    etapes = st.session_state["etapes"]
    if not etapes:
        st.info(
            "La chaîne est vide : l'image affichée est l'image source. "
            "Ajoutez un filtre ci-dessus, ou chargez une chaîne d'exemple "
            "dans la barre latérale."
        )
        return

    traces = {t.index: t for t in (resultat.steps if resultat else [])}

    for position, brute in enumerate(list(etapes)):
        definition = registry.get(brute["filtre"])
        trace = traces.get(position)
        classe = "etape-carte"
        if not brute["actif"]:
            classe += " inactive"
        elif trace is not None and trace.error:
            classe += " erreur"

        st.markdown(
            f'<div class="{classe}">'
            f'<span class="etiquette">étape {position + 1}</span>'
            f'<b>{definition.name}</b>'
            + (f'<span style="color:#8b93a1"> · {trace.duree_ms:.1f} ms</span>'
               if trace is not None and not trace.skipped and not trace.error else "")
            + "</div>",
            unsafe_allow_html=True,
        )

        with st.container():
            boutons = st.columns([1.3, 1, 1, 1, 1])
            brute["actif"] = boutons[0].toggle(
                "Actif", value=brute["actif"], key=f"actif_{brute['uid']}"
            )
            if boutons[1].button("↑", key=f"haut_{brute['uid']}",
                                 disabled=position == 0, help="Monter"):
                etapes.insert(position - 1, etapes.pop(position))
                st.rerun()
            if boutons[2].button("↓", key=f"bas_{brute['uid']}",
                                 disabled=position == len(etapes) - 1,
                                 help="Descendre"):
                etapes.insert(position + 1, etapes.pop(position))
                st.rerun()
            if boutons[3].button("⟲", key=f"reset_{brute['uid']}",
                                 help="Valeurs par défaut"):
                for spec in definition.params:
                    st.session_state.pop(f"p{brute['uid']}_{spec.name}", None)
                brute["parametres"] = definition.defaults()
                st.rerun()
            if boutons[4].button("✕", key=f"suppr_{brute['uid']}",
                                 help="Supprimer l'étape"):
                del etapes[position]
                st.rerun()

            if definition.needs_reference and st.session_state["reference_image"] is None:
                st.warning(
                    "Ce filtre a besoin d'une deuxième image "
                    "(barre latérale, section 2)."
                )
            if not definition.realtime_safe:
                st.caption("⏱ Filtre coûteux : préférer une petite résolution de travail.")

            brute["parametres"] = widgets.formulaire_parametres(
                definition, brute["parametres"], prefixe=f"p{brute['uid']}"
            )

            if trace is not None and trace.error:
                st.error(trace.error)
            elif trace is not None and trace.infos:
                widgets.tableau_informations(trace.infos)


def _fiche_filtre(definition) -> None:
    """Fiche descriptive d'un filtre (résumé, principe, origine, paramètres)."""
    st.markdown(f"### {definition.name}")
    st.caption(f"`{definition.id}` · {definition.family}")
    st.write(definition.summary)
    if definition.theory:
        st.markdown("**Principe**")
        st.write(definition.theory)
    if definition.notes:
        st.info(definition.notes)
    details = []
    if definition.tp:
        details.append(f"Origine : {', '.join(definition.tp)}")
    details.append(f"Entrée : {definition.input_mode} · sortie : {definition.output_mode}")
    if definition.needs_reference:
        details.append("Nécessite une deuxième image")
    if not definition.realtime_safe:
        details.append("Coûteux en calcul")
    st.caption(" — ".join(details))
    if definition.params:
        st.markdown("**Paramètres**")
        for spec in definition.params:
            st.markdown(f"- `{spec.name}` — {spec.label} : {spec.describe()}")
            if spec.help:
                st.caption(f"  {spec.help}")


# ---------------------------------------------------------------------------
# Onglets de résultat
# ---------------------------------------------------------------------------
def onglet_resultat(
    source: np.ndarray, resultat: PipelineResult, reduite: bool
) -> None:
    """Comparaison avant / après et téléchargement."""
    affichage = st.radio(
        "Affichage",
        ("Côte à côte", "Résultat seul", "Source seule", "Superposition réglable"),
        horizontal=True,
        key="mode_affichage",
        label_visibility="collapsed",
    )

    if affichage == "Côte à côte":
        gauche, droite = st.columns(2)
        with gauche:
            widgets.afficher_image(source, f"Source — {st.session_state['source_nom']}")
        with droite:
            widgets.afficher_image(resultat.image, "Résultat")
    elif affichage == "Résultat seul":
        widgets.afficher_image(resultat.image, "Résultat")
    elif affichage == "Source seule":
        widgets.afficher_image(source, f"Source — {st.session_state['source_nom']}")
    else:
        melange = st.slider("Part du résultat", 0.0, 1.0, 0.5, 0.01,
                            key="melange_apercu")
        premiere, seconde = im.match_channels(
            source, im.match_shape(source, resultat.image)
        )
        widgets.afficher_image(
            cv2.addWeighted(seconde, melange, premiere, 1.0 - melange, 0),
            f"Superposition — {melange * 100:.0f} % de résultat",
        )

    if reduite:
        st.caption(
            "ℹ L'image a été réduite à la résolution de travail choisie dans la "
            "barre latérale. Le téléchargement exporte cette version réduite."
        )
    for avertissement in resultat.warnings:
        st.warning(avertissement)

    st.divider()
    colonnes = st.columns([1, 1, 2])
    format_choisi = colonnes[0].selectbox("Format", ("png", "jpg", "webp", "bmp"),
                                          key="format_export")
    qualite = colonnes[1].slider("Qualité", 50, 100, 95, key="qualite_export",
                                 disabled=format_choisi == "png" or format_choisi == "bmp")
    try:
        donnees = io_utils.encode_image(resultat.image, f".{format_choisi}", qualite)
        colonnes[2].download_button(
            f"Télécharger le résultat ({len(donnees) // 1024} Ko)",
            data=donnees,
            file_name=f"{_nom_fichier(st.session_state['nom_chaine'])}.{format_choisi}",
            mime=f"image/{format_choisi}",
            type="primary",
            width="stretch",
        )
    except io_utils.ImageLoadError as exc:
        colonnes[2].error(str(exc))


def onglet_etapes(source: np.ndarray, resultat: PipelineResult) -> None:
    """Galerie pas-à-pas : l'image après chaque étape."""
    st.caption(
        "L'image telle qu'elle sort de chaque étape. C'est la façon la plus "
        "directe de voir pourquoi l'ordre des filtres change le résultat."
    )
    if not resultat.steps:
        st.info("Aucune étape dans la chaîne.")
        return

    vignettes = [("Source", source, None)]
    for trace in resultat.steps:
        etiquette = f"{trace.index + 1}. {trace.name}"
        if trace.skipped:
            etiquette += " (désactivée)"
        elif trace.error:
            etiquette += " (en erreur)"
        vignettes.append((etiquette, trace.image, trace))

    par_ligne = st.slider("Vignettes par ligne", 2, 5, 3, key="vignettes_par_ligne")
    for debut in range(0, len(vignettes), par_ligne):
        colonnes = st.columns(par_ligne)
        # strict=False : la dernière ligne a souvent moins de vignettes
        # que de colonnes.
        for colonne, (etiquette, image, trace) in zip(
            colonnes, vignettes[debut:debut + par_ligne], strict=False
        ):
            with colonne:
                widgets.afficher_image(image, etiquette)
                if trace is not None and not trace.skipped and not trace.error:
                    st.caption(f"{trace.duree_ms:.1f} ms")
                    widgets.tableau_informations(trace.infos)
                elif trace is not None and trace.error:
                    st.error(trace.error)

    st.divider()
    st.caption("Durée par étape (hors affichage)")
    mesurees = [t for t in resultat.steps if not t.skipped and not t.error]
    if mesurees:
        # Libellé numéroté : deux étapes peuvent utiliser le même filtre, le
        # seul nom ne suffirait pas à les distinguer sur le graphique.
        st.bar_chart(
            [
                {"étape": f"{t.index + 1}. {t.name}", "durée (ms)": round(t.duree_ms, 2)}
                for t in mesurees
            ],
            x="étape",
            y="durée (ms)",
            horizontal=True,
            sort=False,
            height=max(140, 42 * len(mesurees)),
        )
        st.caption(f"Total : {resultat.duree_totale_ms:.1f} ms")


def onglet_analyse(source: np.ndarray, resultat: PipelineResult) -> None:
    """Histogrammes, statistiques et mesures d'écart."""
    st.caption(
        "L'histogramme compte les pixels par niveau d'intensité ; il ne dit "
        "rien de leur position, mais tout de l'exposition et du contraste "
        "(TP4 §8)."
    )
    reglages = st.columns(3)
    mode = reglages[0].radio("Mode", ("Histogramme", "Cumulé"), horizontal=True,
                             key="mode_histogramme")
    echelle_log = reglages[1].toggle("Échelle logarithmique", key="log_histogramme")
    carte = reglages[2].toggle("Carte des écarts", value=False, key="carte_ecarts",
                               help="Visualise |source − résultat| pixel par pixel.")

    gauche, droite = st.columns(2)
    with gauche:
        st.markdown("**Source**")
        widgets.afficher_image(histogram.draw_histogram(
            source, largeur=760, hauteur=330, cumule=(mode == "Cumulé"),
            echelle_log=echelle_log,
        ))
        st.dataframe(histogram.statistics(source), width="stretch")
    with droite:
        st.markdown("**Résultat**")
        widgets.afficher_image(histogram.draw_histogram(
            resultat.image, largeur=760, hauteur=330, cumule=(mode == "Cumulé"),
            echelle_log=echelle_log,
        ))
        st.dataframe(histogram.statistics(resultat.image), width="stretch")

    st.divider()
    st.markdown("**Écart entre la source et le résultat** (TP7 ex.4)")
    mesures = metrics.compare(source, resultat.image)
    colonnes = st.columns(len(mesures))
    explications = {
        "MSE": "erreur quadratique moyenne ; 0 = images identiques",
        "RMSE": "racine de la MSE, en niveaux de gris",
        "MAE": "écart absolu moyen, moins sensible aux valeurs extrêmes",
        "PSNR (dB)": "> 40 dB : différence généralement invisible",
    }
    for colonne, (nom, valeur) in zip(colonnes, mesures.items(), strict=True):
        texte = "∞" if valeur == float("inf") else f"{valeur:.2f}"
        colonne.metric(nom, texte, help=explications.get(nom))
    st.caption(
        "Ces chiffres mesurent une **différence**, pas une qualité : si la "
        "source est bruitée, un bon débruitage s'en écarte nécessairement. "
        "Pour évaluer un débruitage, bruitez vous-même une image propre "
        "(famille « Bruit ») et comparez le résultat à l'originale."
    )

    if carte:
        amplification = st.slider("Amplification de la carte", 1.0, 10.0, 2.0, 0.5,
                                  key="amplification_ecarts")
        widgets.afficher_image(
            metrics.difference(source, resultat.image, amplification),
            f"|source − résultat| × {amplification:g}",
        )


def onglet_pixels(source: np.ndarray, resultat: PipelineResult) -> None:
    """Inspecteur de pixel — l'équivalent du TP3 §1 (« La souris »)."""
    st.caption(
        "Valeur d'un pixel avant et après traitement. Rappel du TP2 §1.1 : "
        "l'origine (0, 0) est en haut à gauche, x désigne la colonne et y la "
        "ligne."
    )
    hauteur, largeur = source.shape[:2]
    # Une valeur mémorisée sur une image plus grande dépasserait le maximum du
    # widget, ce que Streamlit refuse : on la ramène dans les bornes d'abord.
    _borner_etat("pixel_x", largeur - 1)
    _borner_etat("pixel_y", hauteur - 1)
    colonnes = st.columns([1, 1, 2])
    x = colonnes[0].number_input("x (colonne)", 0, largeur - 1, largeur // 2,
                                 key="pixel_x")
    y = colonnes[1].number_input("y (ligne)", 0, hauteur - 1, hauteur // 2,
                                 key="pixel_y")
    rayon = colonnes[2].slider("Taille du voisinage affiché", 1, 15, 5, 2,
                               key="pixel_voisinage")

    gauche, droite = st.columns(2)
    for colonne, (titre, image) in zip(
        (gauche, droite),
        (("Source", source), ("Résultat", resultat.image)),
        strict=True,
    ):
        with colonne:
            st.markdown(f"**{titre}**")
            _fiche_pixel(image, int(x), int(y), int(rayon))


def _borner_etat(cle: str, maximum: int) -> None:
    """Ramène une valeur mémorisée dans [0, maximum] avant de créer le widget."""
    valeur = st.session_state.get(cle)
    if isinstance(valeur, (int, float)):
        st.session_state[cle] = int(max(0, min(maximum, valeur)))


def _fiche_pixel(image: np.ndarray, x: int, y: int, rayon: int) -> None:
    """Affiche la valeur d'un pixel, sa couleur et son voisinage."""
    hauteur, largeur = image.shape[:2]
    if not (0 <= x < largeur and 0 <= y < hauteur):
        st.info(
            f"Hors de cette image ({largeur}×{hauteur}) : un filtre géométrique "
            "a changé les dimensions."
        )
        return

    if im.is_gray(image):
        valeur = int(image[y, x])
        st.code(f"intensité = {valeur}")
        apercu = np.full((60, 160), valeur, np.uint8)
    else:
        bleu, vert, rouge = (int(c) for c in image[y, x])
        hsv = cv2.cvtColor(np.uint8([[[bleu, vert, rouge]]]), cv2.COLOR_BGR2HSV)[0, 0]
        st.code(
            f"(B, G, R) = ({bleu}, {vert}, {rouge})\n"
            f"(H, S, V) = ({int(hsv[0])}, {int(hsv[1])}, {int(hsv[2])})\n"
            f"hexadécimal = #{rouge:02x}{vert:02x}{bleu:02x}"
        )
        apercu = np.zeros((60, 160, 3), np.uint8)
        apercu[:] = (bleu, vert, rouge)
    widgets.afficher_image(apercu, "couleur du pixel", largeur=160)

    demi = rayon // 2
    x1, y1 = max(0, x - demi), max(0, y - demi)
    x2, y2 = min(largeur, x + demi + 1), min(hauteur, y + demi + 1)
    voisinage = image[y1:y2, x1:x2]
    st.caption(f"Voisinage {voisinage.shape[1]}×{voisinage.shape[0]} (valeurs brutes)")
    if im.is_gray(voisinage):
        st.dataframe(voisinage, width="stretch")
    else:
        st.dataframe(cv2.cvtColor(voisinage, cv2.COLOR_BGR2GRAY), width="stretch")
        st.caption("Intensités (niveaux de gris) du voisinage.")
    agrandi = cv2.resize(voisinage, (240, 240), interpolation=cv2.INTER_NEAREST)
    widgets.afficher_image(agrandi, "voisinage agrandi (plus proche voisin)",
                           largeur=240)


def onglet_comparateur(source: np.ndarray) -> None:
    """Comparaison de plusieurs filtres sur la même image (TP7 ex.1-4)."""
    st.caption(
        "Applique plusieurs filtres **indépendamment** à la même image et les "
        "compare chiffres en main. C'est le protocole des exercices 1 à 4 du "
        "TP7 : faire varier un seul réglage et regarder ce qui change."
    )

    groupes = registry.filters_by_family()
    colonnes = st.columns([1, 2])
    famille = colonnes[0].selectbox("Famille à comparer", list(groupes.keys()),
                                    index=list(groupes.keys()).index("Lissage & débruitage")
                                    if "Lissage & débruitage" in groupes else 0,
                                    key="comparateur_famille")
    definitions = groupes[famille]
    noms = [d.name for d in definitions]
    choisis = colonnes[1].multiselect(
        "Filtres", noms, default=noms[: min(4, len(noms))],
        key="comparateur_filtres",
    )
    if not choisis:
        st.info("Sélectionner au moins un filtre.")
        return

    reference_mode = st.radio(
        "Référence pour les mesures",
        ("Image source", "Première étape de la chaîne courante"),
        horizontal=True,
        key="comparateur_reference",
        help=(
            "Pour évaluer un débruitage, mettez d'abord un filtre de bruit "
            "dans la chaîne principale : la source propre devient alors la "
            "référence honnête."
        ),
    )
    reference = source
    if reference_mode != "Image source" and st.session_state["etapes"]:
        chaine = construire_chaine()
        chaine.steps = chaine.steps[:1]
        reference = chaine.apply(source, collect=False).image

    par_ligne = st.slider("Vignettes par ligne", 2, 4, 3,
                          key="comparateur_colonnes")
    lignes: list[dict[str, object]] = []
    selectionnes = [definitions[noms.index(nom)] for nom in choisis]

    for debut in range(0, len(selectionnes), par_ligne):
        cols = st.columns(par_ligne)
        for colonne, definition in zip(
            cols, selectionnes[debut:debut + par_ligne], strict=False
        ):
            with colonne:
                st.markdown(f"**{definition.name}**")
                with st.expander("Réglages", expanded=False):
                    valeurs = widgets.formulaire_parametres(
                        definition, definition.defaults(),
                        prefixe=f"cmp_{definition.id}",
                    )
                try:
                    sortie, infos = definition.apply(
                        source, valeurs, st.session_state["reference_image"]
                    )
                except Exception as exc:  # noqa: BLE001 - message à l'écran
                    st.error(str(exc))
                    continue
                widgets.afficher_image(sortie)
                mesures = metrics.compare(reference, sortie)
                st.caption(
                    f"MSE {mesures['MSE']:.1f} · PSNR "
                    + ("∞" if mesures["PSNR (dB)"] == float("inf")
                       else f"{mesures['PSNR (dB)']:.1f} dB")
                )
                widgets.tableau_informations(infos)
                lignes.append({
                    "Filtre": definition.name,
                    "MSE": round(mesures["MSE"], 2),
                    "RMSE": round(mesures["RMSE"], 2),
                    "MAE": round(mesures["MAE"], 2),
                    "PSNR (dB)": (
                        None if mesures["PSNR (dB)"] == float("inf")
                        else round(mesures["PSNR (dB)"], 2)
                    ),
                })

    if lignes:
        st.divider()
        st.dataframe(lignes, width="stretch", hide_index=True)
        st.caption("PSNR vide = images identiques (MSE nulle).")


def onglet_catalogue() -> None:
    """Référence complète du catalogue, consultable sans quitter l'atelier."""
    st.caption(
        f"{len(registry.all_filters())} filtres, regroupés par famille. "
        "La documentation détaillée est dans le dossier `docs/`."
    )
    recherche = st.text_input(
        "Rechercher", placeholder="noyau, seuil, contour, TP7…",
        key="recherche_catalogue",
    )
    terme = recherche.strip().lower()

    for famille, definitions in registry.filters_by_family().items():
        gardees = [
            d for d in definitions
            if not terme
            or terme in d.name.lower()
            or terme in d.id.lower()
            or terme in d.summary.lower()
            or terme in d.theory.lower()
            or any(terme in t.lower() for t in d.tp)
        ]
        if not gardees:
            continue
        st.markdown(f"## {famille}")
        for definition in gardees:
            with st.expander(f"{definition.name} — `{definition.id}`"):
                _fiche_filtre(definition)
                if st.button("Ajouter à la chaîne", key=f"cat_{definition.id}"):
                    ajouter_etape(definition.id)
                    st.rerun()


def onglet_aide() -> None:
    """Mode d'emploi court et renvois vers la documentation."""
    st.markdown(
        """
        ## Comment utiliser l'atelier

        1. **Choisir une image** dans la barre latérale : un fichier, un
           instantané de webcam, une image des TP, ou la mire de test.
        2. **Empiler des filtres** dans la colonne de gauche. Chaque étape est
           réglable, désactivable et déplaçable ; l'ordre compte.
        3. **Lire le résultat** : onglet *Résultat* pour l'avant/après,
           *Étapes* pour voir l'effet de chaque filtre isolément, *Analyse*
           pour les histogrammes et les mesures d'écart.
        4. **Exporter** l'image, ou la chaîne en JSON pour la retrouver plus tard.

        ## Les trois interfaces

        | Interface | Lancement | À utiliser pour |
        |---|---|---|
        | Web (celle-ci) | `streamlit run ui/streamlit_app.py` | régler, comparer, documenter |
        | Bureau (OpenCV) | `python ui/desktop.py --webcam` | **flux webcam temps réel**, trackbars, souris |
        | Ligne de commande | `python -m cvlab.cli --help` | traitement par lot, mesures reproductibles |

        ## Pièges fréquents

        * **Taille de noyau paire** : refusée par OpenCV. Les curseurs de
          l'application n'avancent que de 2 en 2 sur ces paramètres, le
          problème ne peut donc pas se produire ici.
        * **Rouge et bleu inversés** : OpenCV travaille en BGR, les
          bibliothèques d'affichage en RGB. La conversion est centralisée dans
          `cvlab.images.to_rgb`.
        * **Dériver une image bruitée** : Sobel, le Laplacien et Canny
          amplifient le bruit. Mettre un flou gaussien avant — c'est le rôle du
          paramètre « flou préalable ».
        * **Seuiller une image couleur** : sans objet ; les filtres de
          seuillage convertissent donc l'entrée en niveaux de gris.
        * **Comparer avec la MSE** : elle mesure un écart, pas une qualité.
          Voir la note de l'onglet *Analyse*.

        ## Pour aller plus loin

        Le dossier `docs/` contient l'installation, l'architecture, la
        référence de chaque filtre avec ses formules, la correspondance avec
        les TP 1 à 7, et le guide d'extension.
        """
    )
    st.caption(f"cvlab {__version__} · OpenCV {cv2.__version__} · NumPy {np.__version__}")


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------
def main() -> None:
    init_etat()

    st.title(TITRE)
    st.caption(
        "Appliquez les filtres étudiés en TP1 à TP7 à votre propre image ou à "
        "votre webcam, en réglant vous-même chaque paramètre."
    )

    panneau_source()
    panneau_reference()
    panneau_presets()

    source_brute = st.session_state["source_image"]
    if source_brute is None:
        st.info(
            "Choisissez une image dans la barre latérale pour commencer "
            "— ou générez la mire de test."
        )
        onglet_catalogue()
        return

    source, reduite = reduire_pour_travail(
        source_brute, int(st.session_state["taille_travail"])
    )

    st.sidebar.divider()
    st.sidebar.caption("Image de travail")
    st.sidebar.json(im.describe_image(source), expanded=False)

    colonne_gauche, colonne_droite = st.columns([5, 7], gap="large")

    # Première passe : on applique la chaîne telle qu'elle était au chargement,
    # pour disposer des durées et des informations à afficher dans l'éditeur.
    chaine = construire_chaine()
    resultat = chaine.apply(source, st.session_state["reference_image"],
                            collect=True)

    with colonne_gauche:
        editeur_chaine(resultat)

    # Deuxième passe : les curseurs viennent peut-être d'être déplacés pendant
    # le rendu de l'éditeur, on recalcule pour afficher un résultat à jour.
    chaine = construire_chaine()
    resultat = chaine.apply(source, st.session_state["reference_image"],
                            collect=True)

    with colonne_droite:
        onglets = st.tabs([
            "Résultat", "Étapes", "Analyse", "Pixels", "Comparateur",
            "Catalogue", "Aide",
        ])
        with onglets[0]:
            onglet_resultat(source, resultat, reduite)
        with onglets[1]:
            onglet_etapes(source, resultat)
        with onglets[2]:
            onglet_analyse(source, resultat)
        with onglets[3]:
            onglet_pixels(source, resultat)
        with onglets[4]:
            onglet_comparateur(source)
        with onglets[5]:
            onglet_catalogue()
        with onglets[6]:
            onglet_aide()


main()
