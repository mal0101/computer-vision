"""Tests de la documentation.

La documentation fait partie du livrable : ces tests évitent qu'elle dérive du
code (référence des filtres périmée, lien mort, image manquante).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from cvlab import registry

RACINE = Path(__file__).resolve().parent.parent
DOCS = RACINE / "docs"
PAGES = sorted(DOCS.glob("*.md"))


def test_pages_presentes() -> None:
    attendues = {
        "README.md", "01-installation.md", "02-prise-en-main.md",
        "03-architecture.md", "04-interface-web.md", "05-interface-bureau.md",
        "06-ligne-de-commande.md", "07-theorie-filtrage.md",
        "08-reference-filtres.md", "09-parametres.md",
        "10-chaines-et-presets.md", "11-correspondance-tp.md",
        "12-etendre.md", "13-tests.md", "14-depannage.md", "15-api.md",
    }
    assert {p.name for p in PAGES} == attendues


def test_reference_des_filtres_a_jour() -> None:
    """La référence est générée : elle ne doit jamais diverger du catalogue."""
    import sys

    if str(RACINE) not in sys.path:
        sys.path.insert(0, str(RACINE))
    sys.path.insert(0, str(RACINE / "outils"))
    import generer_doc  # type: ignore[import-not-found]

    attendu = generer_doc.generer()
    actuel = (DOCS / "08-reference-filtres.md").read_text(encoding="utf-8")
    assert actuel == attendu, (
        "docs/08-reference-filtres.md est périmé — "
        "relancer : python outils/generer_doc.py"
    )


def test_tous_les_filtres_documentes() -> None:
    texte = (DOCS / "08-reference-filtres.md").read_text(encoding="utf-8")
    for definition in registry.all_filters():
        assert f"`{definition.id}`" in texte, definition.id
        for spec in definition.params:
            assert f"`{spec.name}`" in texte, f"{definition.id}.{spec.name}"


@pytest.mark.parametrize("page", PAGES, ids=[p.name for p in PAGES])
def test_liens_internes_valides(page: Path) -> None:
    """Aucun lien Markdown vers un fichier inexistant."""
    texte = page.read_text(encoding="utf-8")
    for cible in re.findall(r"\]\(([^)#]+?)(?:#[^)]*)?\)", texte):
        if cible.startswith(("http://", "https://", "mailto:")):
            continue
        chemin = (page.parent / cible).resolve()
        assert chemin.exists(), f"{page.name} : lien mort vers {cible}"


@pytest.mark.parametrize("page", PAGES, ids=[p.name for p in PAGES])
def test_images_presentes(page: Path) -> None:
    texte = page.read_text(encoding="utf-8")
    for cible in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", texte):
        if cible.startswith(("http://", "https://")):
            continue
        assert (page.parent / cible).is_file(), (
            f"{page.name} : image manquante {cible} — "
            "relancer : python outils/demo.py"
        )


def test_planches_illustrations_presentes() -> None:
    """Les planches citées dans la théorie doivent exister."""
    import sys

    sys.path.insert(0, str(RACINE / "outils"))
    import demo  # type: ignore[import-not-found]

    for nom in demo.PLANCHES:
        assert (DOCS / "images" / f"{nom}.png").is_file(), (
            f"planche manquante : {nom}.png — relancer python outils/demo.py"
        )


def test_readme_du_projet_mentionne_les_trois_interfaces() -> None:
    texte = (RACINE / "README.md").read_text(encoding="utf-8")
    for commande in ("streamlit run ui/streamlit_app.py",
                     "python ui/desktop.py",
                     "python -m cvlab.cli"):
        assert commande in texte, commande


def test_presets_tous_documentes() -> None:
    texte = (DOCS / "10-chaines-et-presets.md").read_text(encoding="utf-8")
    for fichier in sorted((RACINE / "presets").glob("*.json")):
        assert fichier.name in texte, f"preset non documenté : {fichier.name}"


def test_nombre_de_filtres_coherent() -> None:
    """Le chiffre annoncé dans la documentation doit être le bon."""
    total = len(registry.all_filters())
    for page in (DOCS / "README.md", DOCS / "08-reference-filtres.md"):
        texte = page.read_text(encoding="utf-8")
        assert f"{total} filtres" in texte, (
            f"{page.name} : le nombre de filtres annoncé n'est pas {total}"
        )
