"""Tests de la configuration : schéma, validation, formatage monétaire."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from pieces.config import (
    CONFIG_DEFAUT,
    DENOMINATIONS_DEFAUT,
    VERSION_CONFIG,
    charger_config,
    chemin_config_defaut,
    config_vers_dict,
    denominations_actives,
    dict_vers_config,
    formater_montant,
    sauver_config,
    valider_config,
)
from pieces.erreurs import ErreurConfiguration

# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------


def test_fichier_config_existe() -> None:
    chemin = chemin_config_defaut()
    assert chemin.is_file(), f"fichier absent : {chemin}"


def test_fichier_config_identique_aux_defauts() -> None:
    """Le JSON versionné doit être EXACTEMENT les défauts du code.

    Si ce test échoue après un changement de défaut : régénérer le fichier
    avec sauver_config(CONFIG_DEFAUT) (voir l'en-tête de config.py).
    """
    chargee = charger_config()
    assert config_vers_dict(chargee) == config_vers_dict(CONFIG_DEFAUT)


def test_fichier_config_est_du_json_valide() -> None:
    brut = json.loads(chemin_config_defaut().read_text(encoding="utf-8"))
    assert isinstance(brut, dict)
    assert brut["version"] == VERSION_CONFIG


def test_charger_config_relatif_ne_depend_pas_du_courant(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """charger_config() sans argument doit marcher depuis n'importe où."""
    monkeypatch.chdir(tmp_path)
    config = charger_config()
    assert config.largeur_analyse == 640


def test_charger_config_absent_leve(tmp_path) -> None:
    with pytest.raises(ErreurConfiguration, match="introuvable"):
        charger_config(tmp_path / "nimporte quoi.json")


def test_charger_config_json_casse_leve(tmp_path) -> None:
    cible = tmp_path / "casse.json"
    cible.write_text("{ pas du json", encoding="utf-8")
    with pytest.raises(ErreurConfiguration, match="JSON invalide"):
        charger_config(cible)


def test_sauver_charger_aller_retour(tmp_path) -> None:
    cible = sauver_config(CONFIG_DEFAUT, tmp_path / "c.json")
    rechargee = charger_config(cible)
    assert config_vers_dict(rechargee) == config_vers_dict(CONFIG_DEFAUT)


# ---------------------------------------------------------------------------
# Sérialisation
# ---------------------------------------------------------------------------


def test_dict_vers_config_typage_stricte() -> None:
    brut = config_vers_dict(CONFIG_DEFAUT)
    config = dict_vers_config(brut)
    # les compteurs du suivi doivent rester des entiers (pas 3.0)
    assert isinstance(config.suivi.hits_confirmer, int)
    assert isinstance(config.suivi.mediane_libelle, int)
    assert isinstance(config.version, int)
    assert isinstance(config.marge_classif, float)


@pytest.mark.parametrize(
    "cle,valeur,message",
    [
        ("version", 99, "non prise en charge"),
        ("largeur_analyse", 50, "hors bornes"),
        ("marge_classif", 0.0, "hors bornes"),
        ("centimes_par_defaut", "oui", "type bool"),
    ],
)
def test_dict_invalide_leve(cle: str, valeur: object, message: str) -> None:
    brut = config_vers_dict(CONFIG_DEFAUT)
    brut[cle] = valeur
    with pytest.raises(ErreurConfiguration, match=message):
        dict_vers_config(brut)


def test_champ_manquant_leve() -> None:
    brut = config_vers_dict(CONFIG_DEFAUT)
    del brut["portee_mm"]
    with pytest.raises(ErreurConfiguration, match="manquant"):
        dict_vers_config(brut)


def test_denominations_vides_levent() -> None:
    brut = config_vers_dict(CONFIG_DEFAUT)
    brut["denominations"] = []
    with pytest.raises(ErreurConfiguration, match="non vide"):
        dict_vers_config(brut)


def test_denomination_doublon_leve() -> None:
    brut = config_vers_dict(CONFIG_DEFAUT)
    brut["denominations"].append(dict(brut["denominations"][2]))  # type: ignore[index]
    with pytest.raises(ErreurConfiguration, match="deux fois"):
        dict_vers_config(brut)


def test_denomination_famille_inconnue_leve() -> None:
    brut = config_vers_dict(CONFIG_DEFAUT)
    brut["denominations"][0]["famille"] = "laiton"  # type: ignore[index]
    with pytest.raises(ErreurConfiguration, match="inconnue"):
        dict_vers_config(brut)


def test_bimetallique_sans_polarite_leve() -> None:
    brut = config_vers_dict(CONFIG_DEFAUT)
    # la dénomination 5 DH perd sa polarité (repérée par libellé : un index
    # fixe se décale silencieusement dès qu'on ajoute une dénomination)
    cinq_dh = next(
        d for d in brut["denominations"] if d["libelle"] == "5 DH"  # type: ignore[index,union-attr]
    )
    cinq_dh["polarite"] = None
    with pytest.raises(ErreurConfiguration, match="polarité obligatoire"):
        dict_vers_config(brut)


def test_portee_inversee_leve() -> None:
    brut = config_vers_dict(CONFIG_DEFAUT)
    brut["portee_mm"] = [30.0, 18.0]
    with pytest.raises(ErreurConfiguration, match="0 < bas < haut"):
        dict_vers_config(brut)


def test_seuil_hors_borne_leve() -> None:
    brut = config_vers_dict(CONFIG_DEFAUT)
    brut["seuils"]["circularite_min"] = 1.5  # type: ignore[index]
    with pytest.raises(ErreurConfiguration, match="circularite_min"):
        dict_vers_config(brut)


def test_suivi_hits_sup_a_fenetre_leve() -> None:
    brut = config_vers_dict(CONFIG_DEFAUT)
    brut["suivi"]["hits_confirmer"] = 9  # type: ignore[index]
    brut["suivi"]["fenetre_confirmer"] = 5  # type: ignore[index]
    with pytest.raises(ErreurConfiguration, match="hits_confirmer"):
        dict_vers_config(brut)


def test_variantes_hors_portee_levent() -> None:
    config = replace(
        CONFIG_DEFAUT,
        portee_mm=(18.5, 26.0),  # 29 mm (variante 5 DH) hors bornes
    )
    with pytest.raises(ErreurConfiguration, match="hors de portee_mm"):
        valider_config(config)


def test_valider_config_defaut_ne_leve_pas() -> None:
    valider_config(CONFIG_DEFAUT)  # ne doit rien lever


# ---------------------------------------------------------------------------
# Dénominations actives
# ---------------------------------------------------------------------------


def test_defaut_sans_centimes() -> None:
    actives = denominations_actives(CONFIG_DEFAUT)
    assert [d.valeur_centimes for d in actives] == [50, 100, 200, 500, 1000]


def test_avec_centimes_explicite() -> None:
    actives = denominations_actives(CONFIG_DEFAUT, centimes=True)
    assert [d.valeur_centimes for d in actives] == [5, 10, 20, 50, 100, 200, 500, 1000]


def test_option_defaut_sur_configuration_centimes() -> None:
    config = replace(CONFIG_DEFAUT, centimes_par_defaut=True)
    actives = denominations_actives(config)
    assert 10 in [d.valeur_centimes for d in actives]


def test_catalogue_separable_dans_chaque_famille() -> None:
    """Deux dénominations de même famille+polarité doivent être séparables.

    Remplace l'ancien test qui recopiait les diamètres littéraux (il changeait
    de valeur à chaque correction du catalogue sans jamais rien vérifier).
    Ici on teste la propriété qui compte réellement : à l'intérieur d'une
    famille, le veto de couleur n'aide plus, donc seul l'écart de diamètre
    sépare — il doit dépasser la marge de décision des deux côtés.
    """
    familles: dict[tuple[str, str | None], list[tuple[float, int]]] = {}
    for d in DENOMINATIONS_DEFAUT:
        for mm in d.variantes_mm:
            familles.setdefault((d.famille, d.polarite), []).append(
                (mm, d.valeur_centimes)
            )
    for cle, variantes in familles.items():
        variantes.sort()
        for (mm_a, val_a), (mm_b, val_b) in zip(variantes, variantes[1:], strict=False):
            if val_a == val_b:
                continue  # deux millésimes d'une même pièce : pas d'ambiguïté
            assert mm_b - mm_a > 2 * CONFIG_DEFAUT.marge_classif, (
                f"{cle} : {val_a} et {val_b} centimes ne sont séparés que de "
                f"{mm_b - mm_a:.2f} mm, en deçà de la marge de décision"
            )


# ---------------------------------------------------------------------------
# Formatage monétaire
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "centimes,attendu",
    [
        (1250, "12,50 DH"),
        (50, "0,50 DH"),
        (0, "0,00 DH"),
        (10000, "100,00 DH"),
        (5, "0,05 DH"),
        (15, "0,15 DH"),
        (1234567, "12345,67 DH"),
    ],
)
def test_formater_montant(centimes: int, attendu: str) -> None:
    assert formater_montant(centimes) == attendu


@pytest.mark.parametrize("mauvais", [12.5, "1250", None, True])
def test_formater_montant_type_invalide(mauvais: object) -> None:
    with pytest.raises(TypeError):
        formater_montant(mauvais)  # type: ignore[arg-type]
