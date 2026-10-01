"""Chaîne de traitement.

Une :class:`Pipeline` est une liste ordonnée d':class:`Step`. Chaque étape
désigne un filtre du catalogue et les valeurs de ses paramètres. Appliquer la
chaîne consiste à passer l'image de l'étape *n* à l'étape *n+1*.

L'ordre compte, et c'est tout l'intérêt pédagogique de l'outil : flouter puis
détecter les contours ne donne pas du tout le même résultat que détecter les
contours puis flouter. De même, seuiller avant d'ouvrir (morphologie) est la
bonne séquence ; l'inverse n'a guère de sens.

Le format sérialisé (JSON) est volontairement simple et stable :

.. code-block:: json

    {
      "version": 1,
      "nom": "Contours de Canny",
      "etapes": [
        {"filtre": "filtre_gaussien", "actif": true,
         "parametres": {"taille_noyau": 5, "sigma_x": 0}},
        {"filtre": "canny", "actif": true,
         "parametres": {"seuil_bas": 50, "seuil_haut": 150}}
      ]
    }
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import images as im
from . import registry
from .registry import FilterDef, FilterError

__all__ = ["Step", "StepResult", "PipelineResult", "Pipeline", "PRESET_VERSION"]

#: Version du format de preset. À incrémenter en cas de changement cassant.
PRESET_VERSION = 1


@dataclass
class Step:
    """Une étape : un filtre et les valeurs de ses paramètres."""

    filter_id: str
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def definition(self) -> FilterDef:
        """Définition du filtre associé."""
        return registry.get(self.filter_id)

    @property
    def name(self) -> str:
        return self.definition().name

    def normalised_params(self) -> dict[str, Any]:
        """Valeurs validées et complétées par les valeurs par défaut."""
        return self.definition().coerce(self.params)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filtre": self.filter_id,
            "actif": bool(self.enabled),
            "parametres": self.normalised_params(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Step:
        """Relit une étape. Accepte les clés françaises et anglaises."""
        filter_id = data.get("filtre") or data.get("filter") or data.get("id")
        if not filter_id:
            raise ValueError("étape sans identifiant de filtre")
        params = data.get("parametres") or data.get("params") or {}
        if not isinstance(params, Mapping):
            raise ValueError(f"{filter_id}: « parametres » doit être un objet")
        enabled = data.get("actif", data.get("enabled", True))
        step = cls(filter_id=str(filter_id), params=dict(params),
                   enabled=bool(enabled))
        # Valide tout de suite : un preset corrompu doit échouer au chargement,
        # pas au milieu d'un calcul.
        step.definition()
        step.normalised_params()
        return step


@dataclass
class StepResult:
    """Trace de l'exécution d'une étape (pour l'affichage pas-à-pas)."""

    index: int
    filter_id: str
    name: str
    image: np.ndarray
    params: dict[str, Any]
    infos: dict[str, Any] = field(default_factory=dict)
    duree_ms: float = 0.0
    skipped: bool = False
    error: str | None = None


@dataclass
class PipelineResult:
    """Résultat complet : image finale et trace de chaque étape."""

    image: np.ndarray
    steps: list[StepResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def duree_totale_ms(self) -> float:
        return sum(s.duree_ms for s in self.steps)

    @property
    def applied(self) -> list[StepResult]:
        """Étapes réellement appliquées (ni désactivées ni en erreur)."""
        return [s for s in self.steps if not s.skipped and s.error is None]


@dataclass
class Pipeline:
    """Suite d'étapes appliquée à une image."""

    steps: list[Step] = field(default_factory=list)
    nom: str = "Chaîne sans titre"

    # ------------------------------------------------------------------
    # Édition
    # ------------------------------------------------------------------
    def add(
        self,
        filter_id: str,
        params: Mapping[str, Any] | None = None,
        index: int | None = None,
    ) -> Step:
        """Ajoute une étape (à la fin, ou à la position ``index``)."""
        definition = registry.get(filter_id)
        step = Step(
            filter_id=definition.id,
            params=definition.coerce(params),
            enabled=True,
        )
        if index is None:
            self.steps.append(step)
        else:
            self.steps.insert(max(0, min(len(self.steps), index)), step)
        return step

    def remove(self, index: int) -> None:
        """Supprime l'étape ``index``."""
        if not 0 <= index < len(self.steps):
            raise IndexError(f"étape inexistante: {index}")
        del self.steps[index]

    def move(self, index: int, delta: int) -> int:
        """Déplace une étape de ``delta`` positions. Renvoie la nouvelle position."""
        if not 0 <= index < len(self.steps):
            raise IndexError(f"étape inexistante: {index}")
        cible = max(0, min(len(self.steps) - 1, index + delta))
        if cible != index:
            self.steps.insert(cible, self.steps.pop(index))
        return cible

    def clear(self) -> None:
        self.steps.clear()

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    @property
    def needs_reference(self) -> bool:
        """Vrai si au moins une étape active réclame une deuxième image."""
        return any(
            s.enabled and s.definition().needs_reference for s in self.steps
        )

    # ------------------------------------------------------------------
    # Exécution
    # ------------------------------------------------------------------
    def apply(
        self,
        image: np.ndarray,
        reference: np.ndarray | None = None,
        collect: bool = True,
        stop_on_error: bool = False,
    ) -> PipelineResult:
        """Applique toutes les étapes actives.

        Paramètres
        ----------
        collect
            Conserver l'image intermédiaire de chaque étape. Mettre à ``False``
            sur un flux vidéo : on économise une copie par étape et par image.
        stop_on_error
            Par défaut une étape qui échoue est signalée puis **ignorée**, et la
            chaîne continue avec l'image précédente : l'interface reste utilisable
            pendant qu'on cherche le bon réglage. Avec ``True``, l'erreur est
            propagée (comportement attendu en ligne de commande).
        """
        courante = im.normalise(image)
        resultat = PipelineResult(image=courante)

        for index, step in enumerate(self.steps):
            definition = step.definition()

            if not step.enabled:
                if collect:
                    resultat.steps.append(
                        StepResult(
                            index=index, filter_id=step.filter_id,
                            name=definition.name, image=courante,
                            params=step.normalised_params(), skipped=True,
                        )
                    )
                continue

            debut = time.perf_counter()
            try:
                sortie, infos = definition.apply(courante, step.params, reference)
            except (FilterError, im.InvalidImageError, ValueError) as exc:
                message = f"Étape {index + 1} — {definition.name} : {exc}"
                if stop_on_error:
                    raise FilterError(message) from exc
                resultat.warnings.append(message)
                if collect:
                    resultat.steps.append(
                        StepResult(
                            index=index, filter_id=step.filter_id,
                            name=definition.name, image=courante,
                            params=step.normalised_params(),
                            duree_ms=(time.perf_counter() - debut) * 1000.0,
                            error=str(exc),
                        )
                    )
                continue

            duree = (time.perf_counter() - debut) * 1000.0
            courante = sortie
            if collect:
                resultat.steps.append(
                    StepResult(
                        index=index, filter_id=step.filter_id,
                        name=definition.name, image=sortie,
                        params=step.normalised_params(), infos=infos,
                        duree_ms=duree,
                    )
                )

        resultat.image = courante
        return resultat

    # ------------------------------------------------------------------
    # Sérialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": PRESET_VERSION,
            "nom": self.nom,
            "etapes": [step.to_dict() for step in self.steps],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Pipeline:
        """Relit une chaîne depuis un dictionnaire.

        Les étapes dont le filtre n'existe plus sont signalées par une
        ``ValueError`` : mieux vaut un échec explicite qu'une chaîne
        silencieusement incomplète.
        """
        if not isinstance(data, Mapping):
            raise ValueError("preset invalide : objet JSON attendu")
        version = data.get("version", PRESET_VERSION)
        if int(version) > PRESET_VERSION:
            raise ValueError(
                f"preset en version {version}, cette application lit au plus "
                f"la version {PRESET_VERSION}"
            )
        brutes = data.get("etapes", data.get("steps", []))
        if not isinstance(brutes, Iterable) or isinstance(brutes, (str, bytes)):
            raise ValueError("preset invalide : « etapes » doit être une liste")
        steps = [Step.from_dict(item) for item in brutes]
        return cls(steps=steps, nom=str(data.get("nom", "Chaîne importée")))

    @classmethod
    def from_json(cls, text: str | bytes) -> Pipeline:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"preset illisible (JSON invalide) : {exc}") from exc
        return cls.from_dict(data)

    def save(self, path: str | Path) -> Path:
        chemin = Path(path)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        chemin.write_text(self.to_json(), encoding="utf-8")
        return chemin

    @classmethod
    def load(cls, path: str | Path) -> Pipeline:
        chemin = Path(path)
        if not chemin.is_file():
            raise FileNotFoundError(f"preset introuvable : {chemin}")
        return cls.from_json(chemin.read_text(encoding="utf-8"))

    def summary(self) -> str:
        """Résumé d'une ligne par étape, pour la CLI et les journaux."""
        if not self.steps:
            return "(chaîne vide)"
        lignes = []
        for index, step in enumerate(self.steps, start=1):
            marque = " " if step.enabled else "✗"
            valeurs = ", ".join(
                f"{k}={v}" for k, v in step.normalised_params().items()
            )
            lignes.append(f"{marque} {index}. {step.name} ({valeurs})")
        return "\n".join(lignes)
