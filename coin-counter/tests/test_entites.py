"""Tests des entités partagées (schéma public du pipeline)."""

from __future__ import annotations

from pieces.entites import (
    CERTITUDE_CONFIRME,
    CERTITUDE_INCONNU,
    Classification,
    PieceDetectee,
    ResultatImage,
)


def _piece(r: float = 30.0) -> PieceDetectee:
    return PieceDetectee(centre=(100.0, 120.0), rayon=r, aire=3.14 * r * r,
                         source="contour")


def test_diametre() -> None:
    assert _piece(30.0).diametre == 60.0


def test_est_comptable_sans_incertitude() -> None:
    piece = _piece()
    assert piece.est_comptable
    piece.motif_incertitude = "inclinaison"
    assert not piece.est_comptable


def test_classification_unknown_non_comptable() -> None:
    """Une classification à valeur None ne peut pas entrer dans un total."""
    classement = Classification(
        valeur_centimes=None,
        libelle="inconnu",
        marge=0.05,
        ecart_mm=None,
        certitude=CERTITUDE_INCONNU,
    )
    piece = _piece()
    piece.classification = classement
    assert classement.valeur_centimes is None
    assert classement.certitude == CERTITUDE_INCONNU
    assert CERTITUDE_CONFIRME == "confirme"


def test_resultat_image_defauts() -> None:
    resultat = ResultatImage(
        pieces=[],
        profil="defaut",
        echelle_px_mm=None,
        analyse_ms=4.2,
    )
    assert resultat.echelle == 1.0
    assert resultat.messages == []
    assert resultat.masque is None
    assert resultat.total_centimes is None
