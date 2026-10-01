"""Catalogue des filtres.

Un filtre est décrit par un :class:`FilterDef` : un identifiant stable, un nom
affichable, une famille, la liste de ses :class:`~cvlab.params.ParamSpec` et la
fonction qui fait le travail.

La fonction reçoit l'image en premier argument positionnel puis **tous** ses
paramètres en arguments nommés, déjà validés et déjà traduits en constantes
OpenCV. Elle renvoie soit une image, soit un couple ``(image, infos)`` où
``infos`` est un dictionnaire de valeurs calculées à afficher (par exemple le
seuil trouvé automatiquement par la méthode d'Otsu).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .params import ParamSpec

__all__ = [
    "FilterDef",
    "FilterError",
    "UnknownFilterError",
    "register",
    "get",
    "all_filters",
    "families",
    "filters_by_family",
    "FAMILY_ORDER",
]


class FilterError(RuntimeError):
    """Échec de l'application d'un filtre."""


class UnknownFilterError(KeyError):
    """Identifiant de filtre absent du catalogue."""


#: Ordre d'affichage des familles : il suit la progression des TP.
FAMILY_ORDER: tuple[str, ...] = (
    "Couleur & espaces colorimétriques",
    "Luminosité & contraste",
    "Géométrie",
    "Bruit",
    "Lissage & débruitage",
    "Contours & gradients",
    "Seuillage",
    "Morphologie",
    "Histogramme",
    "Fusion & opérations binaires",
    "Annotation",
)

#: Modes d'entrée/sortie acceptés par un filtre.
_INPUT_MODES = frozenset({"any", "gray", "color"})
_OUTPUT_MODES = frozenset({"same", "gray", "color"})


@dataclass(frozen=True, kw_only=True)
class FilterDef:
    """Description complète d'un filtre.

    Attributs
    ---------
    id
        Identifiant stable, utilisé dans les presets JSON et la CLI.
        **Ne jamais le renommer** sans migration : les presets enregistrés
        s'appuient dessus.
    name
        Nom affiché (français).
    family
        Famille d'appartenance, doit figurer dans :data:`FAMILY_ORDER`.
    summary
        Une phrase : ce que fait le filtre.
    theory
        Rappel du principe (formule, noyau, complexité), affiché dans l'aide
        et repris dans ``docs/``.
    params
        Paramètres exposés à l'utilisateur.
    tp
        Travaux pratiques d'origine, pour la traçabilité (``docs/07-...``).
    input_mode
        ``"any"`` (indifférent), ``"gray"`` (l'image est convertie en niveaux
        de gris avant l'appel) ou ``"color"`` (promue en BGR).
    output_mode
        ``"same"``, ``"gray"`` ou ``"color"`` : ce que le filtre produit.
    needs_reference
        Vrai si le filtre a besoin d'une deuxième image (fusion, ET binaire).
    realtime_safe
        Faux pour les filtres trop lents pour un flux webcam temps réel.
    """

    id: str
    name: str
    family: str
    summary: str
    fn: Callable[..., Any]
    theory: str = ""
    params: tuple[ParamSpec, ...] = ()
    tp: tuple[str, ...] = ()
    input_mode: str = "any"
    output_mode: str = "same"
    needs_reference: bool = False
    realtime_safe: bool = True
    notes: str = ""
    _index: dict[str, ParamSpec] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.id or " " in self.id:
            raise ValueError(f"identifiant de filtre invalide: {self.id!r}")
        if self.family not in FAMILY_ORDER:
            raise ValueError(
                f"{self.id}: famille inconnue {self.family!r} "
                f"(ajouter la famille à FAMILY_ORDER)"
            )
        if self.input_mode not in _INPUT_MODES:
            raise ValueError(f"{self.id}: input_mode invalide {self.input_mode!r}")
        if self.output_mode not in _OUTPUT_MODES:
            raise ValueError(f"{self.id}: output_mode invalide {self.output_mode!r}")
        names = [spec.name for spec in self.params]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"{self.id}: paramètres en double {sorted(duplicates)}")
        object.__setattr__(self, "_index", {spec.name: spec for spec in self.params})

    # ------------------------------------------------------------------
    # Paramètres
    # ------------------------------------------------------------------
    def spec(self, name: str) -> ParamSpec:
        """Spécification du paramètre ``name``."""
        try:
            return self._index[name]
        except KeyError as exc:
            raise KeyError(f"{self.id}: paramètre inconnu {name!r}") from exc

    def defaults(self) -> dict[str, Any]:
        """Dictionnaire ``nom -> valeur par défaut``."""
        return {spec.name: spec.default for spec in self.params}

    def coerce(self, values: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Valide un dictionnaire de valeurs et complète les manquantes.

        Les clés inconnues sont ignorées silencieusement : un preset
        enregistré avec une version antérieure de l'application reste
        utilisable même si un paramètre a disparu.
        """
        values = dict(values or {})
        out: dict[str, Any] = {}
        for spec in self.params:
            if spec.name in values and values[spec.name] is not None:
                out[spec.name] = spec.coerce(values[spec.name])
            else:
                out[spec.name] = spec.default
        return out

    def resolve(self, values: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Valide puis traduit les valeurs en arguments pour :attr:`fn`."""
        coerced = self.coerce(values)
        return {
            spec.name: spec.resolve(coerced[spec.name]) for spec in self.params
        }

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    def prepare(self, image: np.ndarray) -> np.ndarray:
        """Convertit l'image d'entrée selon :attr:`input_mode`."""
        from . import images as im  # import local : évite un cycle

        if self.input_mode == "gray":
            return im.to_gray(image)
        if self.input_mode == "color":
            return im.to_bgr(image)
        return im.normalise(image)

    def apply(
        self,
        image: np.ndarray,
        values: Mapping[str, Any] | None = None,
        reference: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Applique le filtre et renvoie ``(image, infos)``.

        Lève :class:`FilterError` en enveloppant toute erreur d'OpenCV, de
        façon à ce que l'interface affiche un message utile plutôt qu'une
        trace d'exception brute.
        """
        from . import images as im

        kwargs = self.resolve(values)
        source = self.prepare(image)

        if self.needs_reference:
            if reference is None:
                raise FilterError(
                    f"« {self.name} » a besoin d'une deuxième image "
                    "(panneau « Image de référence »)."
                )
            kwargs["reference"] = im.match_shape(source, reference)

        try:
            result = self.fn(source, **kwargs)
        except FilterError:
            raise
        except Exception as exc:  # pragma: no cover - dépend d'OpenCV
            raise FilterError(f"« {self.name} » a échoué : {exc}") from exc

        infos: dict[str, Any] = {}
        if isinstance(result, tuple):
            if len(result) != 2:
                raise FilterError(
                    f"{self.id}: la fonction doit renvoyer une image ou "
                    "un couple (image, infos)"
                )
            result, infos = result
            infos = dict(infos or {})

        output = im.normalise(result)
        if self.output_mode == "gray":
            output = im.to_gray(output)
        elif self.output_mode == "color":
            output = im.to_bgr(output)
        return output, infos

    def describe(self) -> str:
        """Fiche texte multi-lignes (utilisée par ``cvlab.cli --decrire``)."""
        lines = [
            f"{self.id}  —  {self.name}",
            f"  famille  : {self.family}",
            f"  résumé   : {self.summary}",
        ]
        if self.tp:
            lines.append(f"  origine  : {', '.join(self.tp)}")
        lines.append(
            f"  entrée   : {self.input_mode}   sortie : {self.output_mode}"
            + ("   (2 images)" if self.needs_reference else "")
        )
        if self.theory:
            lines.append(f"  principe : {self.theory}")
        if self.notes:
            lines.append(f"  notes    : {self.notes}")
        if self.params:
            lines.append("  paramètres :")
            lines.extend(f"    - {spec.describe()}" for spec in self.params)
        else:
            lines.append("  paramètres : aucun")
        return "\n".join(lines)


_REGISTRY: dict[str, FilterDef] = {}


def register(**kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Décorateur d'enregistrement d'un filtre.

    Exemple ::

        @register(
            id="filtre_median",
            name="Filtre médian",
            family="Lissage & débruitage",
            summary="Remplace chaque pixel par la médiane de son voisinage.",
            params=(IntParam(name="taille", label="Taille du noyau",
                             value=5, min=3, max=31, odd_only=True),),
            tp=("TP7 ex.3",),
        )
        def filtre_median(image, taille):
            return cv2.medianBlur(image, taille)
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        definition = FilterDef(fn=fn, **kwargs)
        if definition.id in _REGISTRY:
            raise ValueError(f"filtre déjà enregistré: {definition.id}")
        _REGISTRY[definition.id] = definition
        fn.filter_def = definition  # type: ignore[attr-defined]
        return fn

    return decorator


def get(filter_id: str) -> FilterDef:
    """Filtre d'identifiant ``filter_id``."""
    _ensure_loaded()
    try:
        return _REGISTRY[filter_id]
    except KeyError as exc:
        proches = ", ".join(sorted(_REGISTRY)[:8])
        raise UnknownFilterError(
            f"filtre inconnu: {filter_id!r} (ex. d'identifiants valides: {proches}…)"
        ) from exc


def all_filters() -> list[FilterDef]:
    """Tous les filtres, triés par famille puis par nom."""
    _ensure_loaded()
    return sorted(
        _REGISTRY.values(),
        key=lambda d: (FAMILY_ORDER.index(d.family), d.name),
    )


def families() -> list[str]:
    """Familles non vides, dans l'ordre de :data:`FAMILY_ORDER`."""
    present = {d.family for d in all_filters()}
    return [f for f in FAMILY_ORDER if f in present]


def filters_by_family() -> dict[str, list[FilterDef]]:
    """Dictionnaire ordonné ``famille -> filtres``."""
    out: dict[str, list[FilterDef]] = {f: [] for f in families()}
    for definition in all_filters():
        out[definition.family].append(definition)
    return out


_loaded = False


def _ensure_loaded() -> None:
    """Importe ``cvlab.filters`` au premier accès (remplit le catalogue)."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    from . import filters  # noqa: F401  (effet de bord : enregistrement)


def _reset_for_tests() -> None:  # pragma: no cover - utilitaire de test
    """Vide le catalogue. Réservé aux tests unitaires."""
    global _loaded
    _REGISTRY.clear()
    _loaded = False
