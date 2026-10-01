"""Génère la référence des filtres à partir du catalogue.

    python outils/generer_doc.py              # écrit docs/08-reference-filtres.md
    python outils/generer_doc.py --verifier   # échoue si le fichier est périmé

La référence est produite depuis :mod:`cvlab.registry`, donc elle ne peut pas
diverger du code : un paramètre renommé, une borne modifiée ou un filtre ajouté
se retrouvent automatiquement dans la documentation. Le mode ``--verifier`` est
appelé par la suite de tests, ce qui interdit d'oublier la régénération.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from cvlab import registry  # noqa: E402
from cvlab.params import (  # noqa: E402
    BoolParam,
    ChoiceParam,
    ColorParam,
    FloatParam,
    IntParam,
    TextParam,
)

CIBLE = RACINE / "docs" / "08-reference-filtres.md"

ENTETE = """<!--
  FICHIER GÉNÉRÉ — NE PAS MODIFIER À LA MAIN.
  Source : le catalogue de cvlab.registry.
  Régénérer avec :  python outils/generer_doc.py
-->

# Référence des filtres

Fiche de chaque filtre : ce qu'il fait, le principe sous-jacent, et le rôle
exact de chacun de ses paramètres avec ses bornes.

Les bornes indiquées sont celles que l'application impose : un curseur ne peut
pas produire de valeur hors de ces limites, et les tailles de noyau marquées
« impair » n'avancent que de deux en deux. Toute valeur hors bornes fournie par
un preset ou par la ligne de commande est ramenée dans l'intervalle plutôt que
rejetée.

Conventions des colonnes :

- **entrée** — `any` : le filtre accepte couleur et niveaux de gris ;
  `gray` : l'image est convertie en niveaux de gris avant traitement ;
  `color` : l'image est promue en BGR.
- **sortie** — `same` : même nature que l'entrée ; `gray` / `color` : imposée.
- **origine** — le TP dont provient la notion ; `complément` signale un filtre
  ajouté pour compléter une famille, non traité tel quel en TP.

"""


def _type_lisible(spec: object) -> str:
    """Description courte du type et des bornes d'un paramètre."""
    if isinstance(spec, IntParam):
        borne = f"entier, {spec.min} à {spec.max}"
        if spec.odd_only:
            borne += ", impair"
        if spec.step != 1:
            borne += f", pas {spec.step}"
        return borne
    if isinstance(spec, FloatParam):
        return f"réel, {spec.min:g} à {spec.max:g}, pas {spec.step:g}"
    if isinstance(spec, BoolParam):
        return "booléen"
    if isinstance(spec, ChoiceParam):
        return "choix : " + ", ".join(f"`{k}`" for k in spec.keys)
    if isinstance(spec, ColorParam):
        return "couleur BGR"
    if isinstance(spec, TextParam):
        return f"texte, {spec.max_length} caractères au plus"
    return getattr(spec, "kind", "?")


def _echapper(texte: str) -> str:
    """Neutralise les caractères qui casseraient une cellule de tableau."""
    return texte.replace("|", "\\|").replace("\n", " ")


def generer() -> str:
    lignes = [ENTETE]

    groupes = registry.filters_by_family()
    total = sum(len(v) for v in groupes.values())
    lignes.append(f"**{total} filtres** répartis en {len(groupes)} familles.\n")

    # Sommaire.
    lignes.append("## Sommaire\n")
    for famille, definitions in groupes.items():
        ancre = _ancre(famille)
        lignes.append(f"- [{famille}](#{ancre}) — {len(definitions)} filtres")
        for definition in definitions:
            lignes.append(
                f"  - [{definition.name}](#{_ancre(definition.name)}) "
                f"`{definition.id}`"
            )
    lignes.append("")

    for famille, definitions in groupes.items():
        lignes.append(f"\n## {famille}\n")
        for definition in definitions:
            lignes.append(f"### {definition.name}\n")
            lignes.append(f"`{definition.id}`\n")
            lignes.append(f"{definition.summary}\n")

            meta = [
                f"**Origine** : {', '.join(definition.tp)}",
                f"**Entrée** : `{definition.input_mode}`",
                f"**Sortie** : `{definition.output_mode}`",
            ]
            if definition.needs_reference:
                meta.append("**Deuxième image requise**")
            if not definition.realtime_safe:
                meta.append("**Coûteux** (éviter en temps réel)")
            lignes.append(" · ".join(meta) + "\n")

            if definition.theory:
                lignes.append("**Principe**\n")
                lignes.append(f"{definition.theory}\n")
            if definition.notes:
                lignes.append(f"> **Note** — {definition.notes}\n")

            if definition.params:
                lignes.append("**Paramètres**\n")
                lignes.append("| Paramètre | Libellé | Type et bornes | Défaut | Rôle |")
                lignes.append("| --- | --- | --- | --- | --- |")
                for spec in definition.params:
                    defaut = spec.default
                    if isinstance(defaut, str):
                        defaut = f"`{defaut}`"
                    lignes.append(
                        f"| `{spec.name}` | {_echapper(spec.label)} "
                        f"| {_echapper(_type_lisible(spec))} | {defaut} "
                        f"| {_echapper(spec.help) or '—'} |"
                    )
                lignes.append("")
            else:
                lignes.append("*Aucun paramètre.*\n")

            lignes.append(
                f"```bash\npython -m cvlab.cli decrire {definition.id}\n```\n"
            )

    lignes.append(
        "\n---\n\n"
        "*Page générée par `outils/generer_doc.py` depuis le catalogue des "
        "filtres. Pour la modifier, changer les métadonnées du filtre dans "
        "`cvlab/filters/` puis relancer le générateur.*\n"
    )
    return "\n".join(lignes)


def _ancre(titre: str) -> str:
    """Ancre Markdown (GitHub) correspondant à un titre."""
    remplacements = {
        "é": "e", "è": "e", "ê": "e", "à": "a", "ç": "c", "ô": "o", "û": "u",
        "î": "i", "ù": "u", "â": "a", "ï": "i",
    }
    texte = titre.lower()
    for source, cible in remplacements.items():
        texte = texte.replace(source, cible)
    garde = [c if (c.isalnum() or c in " -") else "" for c in texte]
    return "".join(garde).strip().replace(" ", "-")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier", action="store_true",
                        help="ne rien écrire, échouer si le fichier est périmé")
    parser.add_argument("--sortie", default=str(CIBLE))
    args = parser.parse_args(argv)

    contenu = generer()
    chemin = Path(args.sortie)

    if args.verifier:
        if not chemin.is_file():
            print(f"{chemin} manquant : lancer python outils/generer_doc.py",
                  file=sys.stderr)
            return 1
        if chemin.read_text(encoding="utf-8") != contenu:
            print(
                f"{chemin} est périmé : relancer python outils/generer_doc.py",
                file=sys.stderr,
            )
            return 1
        print(f"{chemin.name} est à jour.")
        return 0

    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(contenu, encoding="utf-8")
    print(f"Écrit : {chemin} ({len(contenu.splitlines())} lignes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
