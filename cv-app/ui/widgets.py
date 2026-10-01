"""Fabrique de widgets Streamlit à partir des spécifications de paramètres.

C'est la pièce qui rend l'interface indépendante du catalogue : aucune fonction
ici ne connaît le nom d'un filtre. On lit un :class:`~cvlab.params.ParamSpec`,
on en déduit le widget approprié, et on renvoie la valeur saisie.

Ajouter un filtre au catalogue le rend donc immédiatement réglable dans
l'interface, sans toucher à ce fichier.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import streamlit as st

from cvlab import images as im
from cvlab.params import (
    BoolParam,
    ChoiceParam,
    ColorParam,
    FloatParam,
    IntParam,
    ParamSpec,
    TextParam,
    bgr_to_hex,
)
from cvlab.registry import FilterDef

__all__ = ["widget_pour_parametre", "formulaire_parametres", "afficher_image",
           "tableau_informations"]


def widget_pour_parametre(
    spec: ParamSpec, valeur: Any, cle: str
) -> Any:
    """Affiche un widget pour ``spec`` et renvoie la valeur choisie.

    ``cle`` doit être unique dans la page : elle mêle l'identifiant de l'étape
    et le nom du paramètre, sinon deux étapes utilisant le même filtre
    partageraient leurs curseurs.
    """
    libelle = spec.label
    if getattr(spec, "unit", ""):
        libelle = f"{libelle} ({spec.unit})"
    aide = spec.help or None

    if isinstance(spec, BoolParam):
        return st.toggle(libelle, value=bool(valeur), key=cle, help=aide)

    if isinstance(spec, ChoiceParam):
        options = spec.keys
        try:
            index = options.index(spec.coerce(valeur))
        except ValueError:
            index = options.index(spec.default)
        return st.selectbox(libelle, options, index=index, key=cle, help=aide)

    if isinstance(spec, ColorParam):
        choisie = st.color_picker(
            libelle, value=bgr_to_hex(spec.coerce(valeur)), key=cle, help=aide
        )
        return spec.coerce(choisie)

    if isinstance(spec, TextParam):
        return st.text_input(
            libelle, value=str(valeur), max_chars=spec.max_length, key=cle,
            help=aide,
        )

    if isinstance(spec, IntParam):
        # Pour un paramètre contraint impair, un pas de 2 depuis une borne
        # impaire garantit que le curseur ne produit que des valeurs valides :
        # l'utilisateur ne peut pas sélectionner un noyau pair.
        if spec.odd_only:
            minimum = spec.min if spec.min % 2 == 1 else spec.min + 1
            maximum = spec.max if spec.max % 2 == 1 else spec.max - 1
            pas = 2
        else:
            minimum, maximum, pas = spec.min, spec.max, spec.step
        if minimum >= maximum:
            st.caption(f"{libelle} : {minimum} (valeur imposée)")
            return minimum
        return int(st.slider(
            libelle, min_value=int(minimum), max_value=int(maximum),
            value=int(spec.coerce(valeur)), step=int(pas), key=cle, help=aide,
        ))

    if isinstance(spec, FloatParam):
        return float(st.slider(
            libelle, min_value=float(spec.min), max_value=float(spec.max),
            value=float(spec.coerce(valeur)), step=float(spec.step), key=cle,
            help=aide,
        ))

    # Type inconnu : repli sur une saisie texte, sans jamais planter l'interface.
    st.warning(f"type de paramètre non géré par l'interface : {spec.kind}")
    return valeur


def formulaire_parametres(
    definition: FilterDef,
    valeurs: Mapping[str, Any],
    prefixe: str,
) -> dict[str, Any]:
    """Affiche tous les paramètres d'un filtre et renvoie les valeurs saisies.

    Les paramètres sont présentés en trois niveaux :

    * les paramètres principaux, directement visibles ;
    * les paramètres partageant un ``group``, rassemblés dans un dépliant ;
    * les paramètres ``advanced``, dans un dépliant « Réglages avancés ».
    """
    courantes = definition.coerce(valeurs)
    sortie: dict[str, Any] = {}

    principaux = [s for s in definition.params if not s.group and not s.advanced]
    avances = [s for s in definition.params if s.advanced]
    groupes: dict[str, list[ParamSpec]] = {}
    for spec in definition.params:
        if spec.group and not spec.advanced:
            groupes.setdefault(spec.group, []).append(spec)

    for spec in principaux:
        sortie[spec.name] = widget_pour_parametre(
            spec, courantes[spec.name], f"{prefixe}_{spec.name}"
        )

    for nom_groupe, specs in groupes.items():
        with st.expander(nom_groupe, expanded=len(groupes) == 1):
            # Deux colonnes : les groupes sont souvent des paires X/Y.
            colonnes = st.columns(2)
            for position, spec in enumerate(specs):
                with colonnes[position % 2]:
                    sortie[spec.name] = widget_pour_parametre(
                        spec, courantes[spec.name], f"{prefixe}_{spec.name}"
                    )

    if avances:
        with st.expander("Réglages avancés"):
            for spec in avances:
                sortie[spec.name] = widget_pour_parametre(
                    spec, courantes[spec.name], f"{prefixe}_{spec.name}"
                )

    return definition.coerce(sortie)


def afficher_image(
    image: np.ndarray,
    legende: str = "",
    largeur: int | None = None,
) -> None:
    """Affiche une image OpenCV (BGR) dans Streamlit, qui attend du RGB."""
    # Streamlit >= 1.3x : « width » accepte un entier de pixels ou "stretch".
    st.image(
        im.to_rgb(image),
        caption=legende or None,
        width=int(largeur) if largeur else "stretch",
    )


def tableau_informations(infos: Mapping[str, Any]) -> None:
    """Affiche les valeurs calculées renvoyées par un filtre."""
    if not infos:
        return
    for cle, valeur in infos.items():
        texte = str(valeur)
        if "\n" in texte:
            st.caption(f"{cle} :")
            st.code(texte, language=None)
        else:
            st.caption(f"**{cle}** : {texte}")
