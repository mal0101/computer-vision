"""Entités de données partagées — le schéma public du pipeline.

Contrat entre les modules (le type de chaque étape est fixé ici une fois
pour toutes) :

* ``detection.detecter(image_analyse, config) -> ResultatDetection``
  (``ResultatDetection`` est défini dans ``detection.py`` : pièces + profil
  actif + masque binaire pour le HUD)
* ``classification.classifier_pieces(pieces, image_analyse, config, px_par_mm)``
  -> les pièces identifiées en place (le filtre ``centimes`` n'agit qu'au
  comptage, pas à l'identification)
* ``traitement.traiter_image(frame, etat) -> ResultatImage``
* ``suivi.Suiveur.maj(resultat, num_frame) -> EtatComptage`` (défini par suivi)

Toutes les coordonnées sont exprimées **à l'échelle d'analyse** (image
réduite à ``config.largeur_analyse``). ``ResultatImage.echelle`` convertit
vers l'échelle plein cadre du flux webcam (multiplier par ``echelle``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Classification",
    "PieceDetectee",
    "ResultatImage",
]

#: Certitudes attribuées à une pièce.
CERTITUDE_CONFIRME = "confirme"
CERTITUDE_TENTATIVE = "tentative"
CERTITUDE_INCONNU = "inconnu"

#: Familles de couleur (portées de teintes des dinars marocains).
FAMILLE_OR = "or"
FAMILLE_ARGENT = "argent"
FAMILLE_BIMETALLIQUE = "bimetallique"
#: couleur illisible (pièce sombre, territoire gravé, éclat) — pas de veto
FAMILLE_INCONNUE = "inconnue"

#: Polarités bimétalliques (inversées entre 5 DH et 10 DH).
POLARITE_CENTRE_OR = "centre_or"
POLARITE_CENTRE_ARGENT = "centre_argent"


@dataclass
class Classification:
    """Résultat de l'identification d'une pièce (M2).

    ``valeur_centimes`` vaut ``None`` pour un UNKNOWN : une pièce sans
    identification fiable ne sera **jamais** comptée dans le total.
    """

    valeur_centimes: int | None
    libelle: str
    #: meilleure_note − deuxième_meilleure (doit atteindre ``config.marge_classif``)
    marge: float
    #: diamètre mesuré − diamètre nominal de la variante retenue (mm)
    ecart_mm: float | None
    certitude: str


@dataclass
class PieceDetectee:
    """Un disque détecté, à l'échelle d'analyse."""

    centre: tuple[float, float]
    rayon: float
    aire: float
    #: « contour » ou « hough » (hough ne sert que de contre-vote/confirmation)
    source: str
    confiance: float = 1.0
    #: paramètres de l'ellipse de fit (axe_majeur ≥ axe_mineur, angle en degrés)
    axe_majeur: float | None = None
    axe_mineur: float | None = None
    angle: float = 0.0
    #: famille/polarité déduites de la couleur (remplies par classification)
    famille: str | None = None
    polarite: str | None = None
    classification: Classification | None = None
    #: motif d'incertitude (« inclinaison », « eclat », « occlusion »,
    #: « chevauchement », « trop_petit », « hors_portee »…) ou None
    motif_incertitude: str | None = None
    #: identifiant de piste attribué par le suivi (None tant que non suivi)
    piste_id: int | None = None

    @property
    def diametre(self) -> float:
        """Diamètre en pixels (2 × rayon) — la base du calcul en millimètres."""
        return 2.0 * self.rayon

    @property
    def est_comptable(self) -> bool:
        """Vrai si la pièce peut entrer dans un total (pas d'incertitude)."""
        return self.motif_incertitude is None


@dataclass
class ResultatImage:
    """Sortie de ``traitement.traiter_image`` pour une frame."""

    pieces: list[PieceDetectee]
    #: nom du profil de prétraitement actif (« defaut », « sel_poivre », …)
    profil: str
    #: pixels par millimètre à l'échelle d'analyse, ou None si non calibré
    echelle_px_mm: float | None
    #: temps de la frame en millisecondes (dernière mesure)
    analyse_ms: float
    #: largeur_pleine_cadre / largeur_analyse (≥ 1) pour replaquer le rendu
    echelle: float = 1.0
    #: messages destinés au HUD (diagnostics, avertissements)
    messages: list[str] = field(default_factory=list)
    #: masque binaire du profil actif (échelle d'analyse), pour la touche D
    masque: np.ndarray | None = None
    #: total confirmé de la frame (rempli par le suivi), en centimes
    total_centimes: int | None = None
