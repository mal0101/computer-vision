"""Tests des spécifications de paramètres."""

from __future__ import annotations

import pytest

from cvlab.params import (
    BoolParam,
    ChoiceParam,
    ColorParam,
    FloatParam,
    IntParam,
    TextParam,
    bgr_to_hex,
    force_odd,
)


@pytest.mark.parametrize(
    ("entree", "minimum", "attendu"),
    [(4, 1, 5), (5, 1, 5), (0, 1, 1), (0, 3, 3), (2, 3, 3), (31, 3, 31), (-4, 1, 1)],
)
def test_force_odd(entree: int, minimum: int, attendu: int) -> None:
    assert force_odd(entree, minimum) == attendu


def test_int_borne_et_impair() -> None:
    spec = IntParam(name="k", label="Noyau", value=5, min=3, max=31, odd_only=True)
    assert spec.coerce(8) == 9        # arrondi vers l'impair supérieur
    assert spec.coerce(-10) == 3      # borné en bas
    assert spec.coerce(1000) == 31    # borné en haut
    assert spec.coerce("7") == 7      # chaîne acceptée (venue de la CLI)
    assert spec.default == 5


def test_int_refuse_defaut_invalide() -> None:
    with pytest.raises(ValueError, match="valeur par défaut"):
        IntParam(name="k", label="k", value=4, min=3, max=9, odd_only=True)
    with pytest.raises(ValueError, match="min"):
        IntParam(name="k", label="k", value=5, min=9, max=3)


def test_int_rejette_texte() -> None:
    spec = IntParam(name="n", label="n", value=1, min=0, max=10)
    with pytest.raises(ValueError, match="entier attendu"):
        spec.coerce("beaucoup")


def test_float_borne_et_nan() -> None:
    spec = FloatParam(name="a", label="alpha", value=1.0, min=0.0, max=3.0)
    assert spec.coerce(-5) == 0.0
    assert spec.coerce(99) == 3.0
    assert spec.coerce("1.5") == 1.5
    with pytest.raises(ValueError, match="NaN"):
        spec.coerce(float("nan"))


def test_bool_depuis_chaine() -> None:
    spec = BoolParam(name="b", label="b", value=False)
    assert spec.coerce("vrai") is True
    assert spec.coerce("non") is False
    with pytest.raises(ValueError, match="booléen attendu"):
        spec.coerce("peut-être")


def test_choice_resolution() -> None:
    spec = ChoiceParam(name="m", label="Mode", value="A", options={"A": 10, "B": 20})
    assert spec.resolve("B") == 20
    assert spec.keys == ["A", "B"]
    with pytest.raises(ValueError, match="inconnu"):
        spec.coerce("C")
    with pytest.raises(ValueError, match="absente des options"):
        ChoiceParam(name="m", label="m", value="Z", options={"A": 1})


def test_color_hex_et_bornes() -> None:
    spec = ColorParam(name="c", label="Couleur", value=(0, 255, 0))
    assert spec.coerce("#ff0000") == (0, 0, 255)   # hex RVB -> tuple BGR
    assert spec.coerce("#f00") == (0, 0, 255)      # forme courte
    assert spec.coerce((-5, 300, 12)) == (0, 255, 12)
    assert bgr_to_hex((0, 0, 255)) == "#ff0000"
    with pytest.raises(ValueError):
        spec.coerce((1, 2))


def test_text_tronque() -> None:
    spec = TextParam(name="t", label="Texte", value="abc", max_length=5)
    assert spec.coerce("abcdefgh") == "abcde"
    assert spec.coerce(None) == ""
