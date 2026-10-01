"""Spécifications de paramètres.

Chaque filtre déclare ses paramètres sous forme d'objets :class:`ParamSpec`.
Ces objets portent tout ce qu'une interface a besoin de savoir pour construire
un widget (type, borne minimale, borne maximale, pas, valeur par défaut, aide)
et tout ce que le noyau de traitement a besoin de savoir pour valider une
valeur venant de l'utilisateur (``coerce``).

C'est le point central de l'application : les interfaces (Streamlit, fenêtre
OpenCV avec trackbars, ligne de commande) ne connaissent *aucun* filtre en
particulier, elles savent seulement lire des ``ParamSpec``.

Toutes les bornes sont volontairement déclarées ici et nulle part ailleurs :
un paramètre invalide (noyau pair pour un flou médian, seuil de bloc trop
petit pour un seuillage adaptatif, ...) fait lever une exception à OpenCV,
donc la validation est faite en amont, une seule fois, pour toutes les
interfaces.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ParamSpec",
    "IntParam",
    "FloatParam",
    "BoolParam",
    "ChoiceParam",
    "ColorParam",
    "TextParam",
    "force_odd",
]


def force_odd(value: int, minimum: int = 1) -> int:
    """Renvoie l'entier impair le plus proche de ``value``, au moins ``minimum``.

    OpenCV exige un noyau de taille impaire pour ``GaussianBlur``,
    ``medianBlur``, ``Laplacian``, ``adaptiveThreshold``... Les trackbars et
    les curseurs produisent des entiers quelconques, on les corrige donc ici.
    """
    value = int(value)
    if value % 2 == 0:
        value += 1
    minimum = int(minimum)
    if minimum % 2 == 0:
        minimum += 1
    return max(minimum, value)


@dataclass(frozen=True, kw_only=True)
class ParamSpec:
    """Base commune à tous les paramètres.

    Attributs
    ---------
    name
        Nom de l'argument passé à la fonction du filtre (``snake_case``).
    label
        Libellé affiché à l'utilisateur (en français).
    help
        Explication courte du rôle du paramètre, affichée en infobulle.
    group
        Nom de groupe facultatif ; les interfaces regroupent les paramètres
        qui partagent un même groupe (ex. les 4 coins d'une homographie).
    advanced
        Si vrai, le paramètre est masqué derrière un dépliant « avancé ».
    """

    name: str
    label: str
    help: str = ""
    group: str = ""
    advanced: bool = False

    #: type logique, utilisé par les interfaces pour choisir un widget
    kind: str = field(default="", init=False)

    @property
    def default(self) -> Any:  # pragma: no cover - redéfini par les sous-classes
        raise NotImplementedError

    def coerce(self, value: Any) -> Any:  # pragma: no cover - redéfini
        raise NotImplementedError

    def resolve(self, value: Any) -> Any:
        """Convertit une valeur validée en valeur utilisable par OpenCV.

        Par défaut l'identité ; seul :class:`ChoiceParam` fait une vraie
        traduction (clé lisible -> constante ``cv2.*``).
        """
        return value

    def describe(self) -> str:
        """Résumé d'une ligne, utilisé par la CLI et la documentation."""
        return f"{self.name} ({self.kind}) = {self.default!r}"


@dataclass(frozen=True, kw_only=True)
class IntParam(ParamSpec):
    """Entier borné, éventuellement contraint à être impair."""

    value: int
    min: int
    max: int
    step: int = 1
    odd_only: bool = False
    unit: str = ""
    kind: str = field(default="int", init=False)

    def __post_init__(self) -> None:
        if self.min > self.max:
            raise ValueError(f"{self.name}: min ({self.min}) > max ({self.max})")
        if self.step <= 0:
            raise ValueError(f"{self.name}: le pas doit être strictement positif")
        # Vérifie que la valeur par défaut est elle-même valide.
        if self.coerce(self.value) != self.value:
            raise ValueError(
                f"{self.name}: valeur par défaut {self.value} invalide "
                f"(bornes [{self.min}, {self.max}], impair={self.odd_only})"
            )

    @property
    def default(self) -> int:
        return int(self.value)

    def coerce(self, value: Any) -> int:
        try:
            out = int(round(float(value)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{self.name}: entier attendu, reçu {value!r}") from exc
        if self.odd_only:
            out = force_odd(out, self.min)
        return max(self.min, min(self.max, out))

    def describe(self) -> str:
        bornes = f"[{self.min}..{self.max}]"
        extra = ", impair" if self.odd_only else ""
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.name} (entier{extra}) {bornes} défaut={self.value}{unit}"


@dataclass(frozen=True, kw_only=True)
class FloatParam(ParamSpec):
    """Réel borné."""

    value: float
    min: float
    max: float
    step: float = 0.01
    decimals: int = 2
    unit: str = ""
    kind: str = field(default="float", init=False)

    def __post_init__(self) -> None:
        if self.min > self.max:
            raise ValueError(f"{self.name}: min ({self.min}) > max ({self.max})")
        if self.step <= 0:
            raise ValueError(f"{self.name}: le pas doit être strictement positif")
        if not (self.min <= self.value <= self.max):
            raise ValueError(
                f"{self.name}: valeur par défaut {self.value} hors de "
                f"[{self.min}, {self.max}]"
            )

    @property
    def default(self) -> float:
        return float(self.value)

    def coerce(self, value: Any) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{self.name}: réel attendu, reçu {value!r}") from exc
        if out != out:  # NaN
            raise ValueError(f"{self.name}: valeur NaN refusée")
        return max(self.min, min(self.max, out))

    def describe(self) -> str:
        unit = f" {self.unit}" if self.unit else ""
        return (
            f"{self.name} (réel) [{self.min}..{self.max}] "
            f"défaut={self.value}{unit}"
        )


@dataclass(frozen=True, kw_only=True)
class BoolParam(ParamSpec):
    """Interrupteur."""

    value: bool
    kind: str = field(default="bool", init=False)

    @property
    def default(self) -> bool:
        return bool(self.value)

    def coerce(self, value: Any) -> bool:
        if isinstance(value, str):
            low = value.strip().lower()
            if low in {"true", "vrai", "1", "oui", "yes", "on"}:
                return True
            if low in {"false", "faux", "0", "non", "no", "off"}:
                return False
            raise ValueError(f"{self.name}: booléen attendu, reçu {value!r}")
        return bool(value)

    def describe(self) -> str:
        return f"{self.name} (booléen) défaut={self.value}"


@dataclass(frozen=True, kw_only=True)
class ChoiceParam(ParamSpec):
    """Choix dans une liste fermée.

    ``options`` associe un libellé lisible (ce qui est stocké dans les presets
    et affiché à l'utilisateur) à la valeur réellement passée à OpenCV
    (souvent une constante ``cv2.*``, parfois une simple chaîne interne).
    """

    value: str
    options: Mapping[str, Any]
    kind: str = field(default="choice", init=False)

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError(f"{self.name}: la liste d'options est vide")
        if self.value not in self.options:
            raise ValueError(
                f"{self.name}: valeur par défaut {self.value!r} absente des options"
            )

    @property
    def default(self) -> str:
        return self.value

    @property
    def keys(self) -> list[str]:
        return list(self.options.keys())

    def coerce(self, value: Any) -> str:
        key = str(value)
        if key not in self.options:
            raise ValueError(
                f"{self.name}: choix {key!r} inconnu "
                f"(attendu l'un de {', '.join(self.keys)})"
            )
        return key

    def resolve(self, value: Any) -> Any:
        return self.options[self.coerce(value)]

    def describe(self) -> str:
        return (
            f"{self.name} (choix: {', '.join(self.keys)}) défaut={self.value!r}"
        )


@dataclass(frozen=True, kw_only=True)
class ColorParam(ParamSpec):
    """Couleur BGR, l'ordre natif d'OpenCV (cf. TP2 §1.2)."""

    value: tuple[int, int, int]
    kind: str = field(default="color", init=False)

    def __post_init__(self) -> None:
        self.coerce(self.value)

    @property
    def default(self) -> tuple[int, int, int]:
        return tuple(int(c) for c in self.value)  # type: ignore[return-value]

    def coerce(self, value: Any) -> tuple[int, int, int]:
        if isinstance(value, str):
            value = _hex_to_bgr(value)
        if not isinstance(value, Sequence) or len(tuple(value)) != 3:
            raise ValueError(
                f"{self.name}: couleur BGR à 3 composantes attendue, reçu {value!r}"
            )
        out = []
        for component in value:
            try:
                comp = int(round(float(component)))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{self.name}: composante de couleur invalide {component!r}"
                ) from exc
            out.append(max(0, min(255, comp)))
        return (out[0], out[1], out[2])

    def describe(self) -> str:
        return f"{self.name} (couleur BGR) défaut={tuple(self.value)}"


@dataclass(frozen=True, kw_only=True)
class TextParam(ParamSpec):
    """Chaîne de caractères courte (texte d'annotation)."""

    value: str = ""
    max_length: int = 120
    kind: str = field(default="text", init=False)

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError(f"{self.name}: max_length doit être positif")

    @property
    def default(self) -> str:
        return self.value

    def coerce(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value)[: self.max_length]

    def describe(self) -> str:
        return f"{self.name} (texte, max {self.max_length}) défaut={self.value!r}"


def _hex_to_bgr(text: str) -> tuple[int, int, int]:
    """Convertit ``#rrggbb`` (format des sélecteurs web) en triplet BGR."""
    raw = text.strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6:
        raise ValueError(f"couleur hexadécimale invalide: {text!r}")
    try:
        red = int(raw[0:2], 16)
        green = int(raw[2:4], 16)
        blue = int(raw[4:6], 16)
    except ValueError as exc:
        raise ValueError(f"couleur hexadécimale invalide: {text!r}") from exc
    return (blue, green, red)


def bgr_to_hex(color: Sequence[int]) -> str:
    """Convertit un triplet BGR en ``#rrggbb`` pour les sélecteurs web."""
    blue, green, red = (int(c) for c in color)
    return f"#{red:02x}{green:02x}{blue:02x}"
