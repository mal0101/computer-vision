"""Interface en ligne de commande.

Utile pour appliquer une chaîne enregistrée à un lot d'images, pour consulter
le catalogue sans ouvrir l'interface graphique, et pour reproduire les
comparaisons chiffrées du TP7 (exercice 4).

Exemples ::

    # Catalogue
    python -m cvlab.cli lister
    python -m cvlab.cli lister --famille "Contours & gradients"
    python -m cvlab.cli decrire canny

    # Un filtre, des paramètres, une image
    python -m cvlab.cli appliquer photo.jpg -s canny:seuil_bas=60,seuil_haut=180 \
        -o sortie.png

    # Une chaîne complète enregistrée depuis l'interface web
    python -m cvlab.cli appliquer photo.jpg --preset presets/contours_canny.json \
        -o sortie.png --etapes

    # Comparaison chiffrée de plusieurs filtres (TP7 ex.4)
    python -m cvlab.cli comparer bruitee.jpg \
        -s filtre_moyenneur -s filtre_gaussien -s filtre_median -s filtre_bilateral
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import histogram as hist
from . import images as im
from . import io_utils, metrics, registry
from .pipeline import Pipeline, Step
from .registry import FilterError


# ---------------------------------------------------------------------------
# Analyse de la syntaxe « filtre:clé=valeur,clé=valeur »
# ---------------------------------------------------------------------------
def parse_step(expression: str) -> Step:
    """Transforme ``"canny:seuil_bas=60,seuil_haut=180"`` en :class:`Step`."""
    texte = expression.strip()
    if not texte:
        raise argparse.ArgumentTypeError("étape vide")

    filter_id, _, reste = texte.partition(":")
    definition = registry.get(filter_id.strip())

    valeurs: dict[str, Any] = {}
    if reste.strip():
        for morceau in reste.split(","):
            if "=" not in morceau:
                raise argparse.ArgumentTypeError(
                    f"« {morceau} » : syntaxe attendue clé=valeur"
                )
            cle, _, valeur = morceau.partition("=")
            cle = cle.strip()
            try:
                definition.spec(cle)
            except KeyError as exc:
                noms = ", ".join(s.name for s in definition.params) or "aucun"
                raise argparse.ArgumentTypeError(
                    f"{definition.id} : paramètre « {cle} » inconnu "
                    f"(disponibles : {noms})"
                ) from exc
            valeurs[cle] = valeur.strip()

    try:
        return Step(filter_id=definition.id, params=definition.coerce(valeurs))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _charger_source(
    chemin: str | None, webcam: int | None
) -> tuple[np.ndarray, str]:
    """Charge l'image d'entrée depuis un fichier ou la webcam."""
    if webcam is not None:
        return io_utils.capture_snapshot(index=webcam), f"webcam {webcam}"
    if not chemin:
        # ValueError et non SystemExit : main() la rattrape et renvoie un code
        # de retour, ce qui garde la commande testable.
        raise ValueError(
            "aucune source : donner un chemin d'image ou --webcam INDEX"
        )
    return io_utils.load_image(chemin), str(chemin)


# ---------------------------------------------------------------------------
# Sous-commandes
# ---------------------------------------------------------------------------
def cmd_lister(args: argparse.Namespace) -> int:
    """Affiche le catalogue des filtres."""
    groupes = registry.filters_by_family()
    total = 0
    for famille, definitions in groupes.items():
        if args.famille and args.famille.lower() not in famille.lower():
            continue
        print(f"\n{famille}")
        print("-" * len(famille))
        for definition in definitions:
            total += 1
            origine = f"  [{', '.join(definition.tp)}]" if definition.tp else ""
            print(f"  {definition.id:<32} {definition.name}{origine}")
            if args.verbeux:
                print(f"      {definition.summary}")
                for spec in definition.params:
                    print(f"        · {spec.describe()}")
    print(f"\n{total} filtre(s).")
    return 0


def cmd_decrire(args: argparse.Namespace) -> int:
    """Affiche la fiche détaillée d'un filtre."""
    try:
        definition = registry.get(args.filtre)
    except KeyError as exc:
        # KeyError s'affiche entre guillemets via print(exc) : on extrait le message.
        print(exc.args[0] if exc.args else exc, file=sys.stderr)
        return 2
    print(definition.describe())
    return 0


def cmd_appliquer(args: argparse.Namespace) -> int:
    """Applique une chaîne à une image et enregistre le résultat."""
    if args.preset:
        chaine = Pipeline.load(args.preset)
        if args.etape:
            chaine.steps.extend(args.etape)
    elif args.etape:
        chaine = Pipeline(steps=list(args.etape), nom="Chaîne (ligne de commande)")
    else:
        print(
            "aucune étape : utiliser -s/--etape ou --preset "
            "(voir « lister » pour les identifiants)",
            file=sys.stderr,
        )
        return 2

    image, origine = _charger_source(args.entree, args.webcam)
    reference = io_utils.load_image(args.reference) if args.reference else None
    if chaine.needs_reference and reference is None:
        print(
            "cette chaîne contient un filtre à deux images : "
            "fournir --reference CHEMIN",
            file=sys.stderr,
        )
        return 2

    try:
        resultat = chaine.apply(image, reference, collect=True, stop_on_error=True)
    except FilterError as exc:
        print(f"échec : {exc}", file=sys.stderr)
        return 1

    print(f"Source  : {origine}  {im.describe_image(image)}")
    print(f"Chaîne  : {chaine.nom}")
    print(chaine.summary())
    if args.etapes:
        for etape in resultat.steps:
            etat = "désactivée" if etape.skipped else f"{etape.duree_ms:.1f} ms"
            print(f"  {etape.index + 1}. {etape.name:<40} {etat}")
            for cle, valeur in etape.infos.items():
                texte = str(valeur).replace("\n", " | ")
                print(f"       {cle}: {texte}")
    for avertissement in resultat.warnings:
        print(f"  ⚠ {avertissement}", file=sys.stderr)

    print(f"Résultat: {im.describe_image(resultat.image)}")
    print(f"Durée totale : {resultat.duree_totale_ms:.1f} ms")

    mesures = metrics.compare(image, resultat.image)
    print("Écart à la source : " + "  ".join(
        f"{nom}={valeur:.2f}" for nom, valeur in mesures.items()
    ))

    if args.sortie:
        chemin = io_utils.save_image(resultat.image, args.sortie, args.qualite)
        print(f"Enregistré : {chemin}")
    if args.histogramme:
        trace = hist.draw_histogram(resultat.image, largeur=720, hauteur=420)
        chemin = io_utils.save_image(trace, args.histogramme)
        print(f"Histogramme enregistré : {chemin}")
    if args.sauver_preset:
        chemin = chaine.save(args.sauver_preset)
        print(f"Preset enregistré : {chemin}")
    return 0


def cmd_comparer(args: argparse.Namespace) -> int:
    """Compare plusieurs filtres sur la même image (TP7 ex.4)."""
    if not args.etape:
        print("donner au moins un filtre avec -s/--etape", file=sys.stderr)
        return 2

    image, origine = _charger_source(args.entree, args.webcam)
    reference = (
        io_utils.load_image(args.reference) if args.reference else image
    )
    print(f"Source    : {origine}")
    print(f"Référence : {args.reference or 'image source elle-même'}")
    print()
    entete = f"{'filtre':<34}{'MSE':>12}{'RMSE':>9}{'MAE':>9}{'PSNR dB':>10}{'ms':>9}"
    print(entete)
    print("-" * len(entete))

    lignes: list[tuple[str, float, float, float, float, float]] = []
    for etape in args.etape:
        chaine = Pipeline(steps=[etape])
        try:
            resultat = chaine.apply(image, reference, collect=True,
                                    stop_on_error=True)
        except FilterError as exc:
            print(f"{etape.filter_id:<34} échec : {exc}", file=sys.stderr)
            continue
        mesures = metrics.compare(reference, resultat.image)
        lignes.append((
            etape.name, mesures["MSE"], mesures["RMSE"], mesures["MAE"],
            mesures["PSNR (dB)"], resultat.duree_totale_ms,
        ))
        if args.dossier_sortie:
            cible = Path(args.dossier_sortie) / f"{etape.filter_id}.png"
            io_utils.save_image(resultat.image, cible)

    for nom, erreur, racine, absolue, rapport, duree in lignes:
        pic = "   inf" if rapport == float("inf") else f"{rapport:10.2f}"
        print(f"{nom:<34}{erreur:12.2f}{racine:9.2f}{absolue:9.2f}{pic}{duree:9.1f}")

    if lignes:
        meilleur = min(lignes, key=lambda ligne: ligne[1])
        print(
            f"\nMSE la plus faible : {meilleur[0]} ({meilleur[1]:.2f}). "
            "Attention : sur une image bruitée sans référence propre, une MSE "
            "faible signifie seulement « proche de l'entrée bruitée », pas "
            "« bien débruitée »."
        )
    if args.dossier_sortie:
        print(f"Images enregistrées dans {args.dossier_sortie}")
    return 0


def cmd_webcam(args: argparse.Namespace) -> int:
    """Vérifie l'accès à la webcam et enregistre éventuellement un instantané."""
    try:
        with io_utils.Camera(index=args.index, largeur=args.largeur,
                             hauteur=args.hauteur) as camera:
            image = camera.read()
            print(f"Webcam {args.index} ouverte, résolution {camera.resolution}")
            print(f"Image lue : {im.describe_image(image)}")
            if args.sortie:
                print(f"Enregistré : {io_utils.save_image(image, args.sortie)}")
    except io_utils.CameraError as exc:
        print(f"échec : {exc}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# Analyseur d'arguments
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cvlab.cli",
        description=(
            "Atelier de filtres OpenCV — version ligne de commande. "
            "Les interfaces graphiques sont lancées par « streamlit run "
            "ui/streamlit_app.py » et « python ui/desktop.py »."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sous = parser.add_subparsers(dest="commande", required=True)

    p_lister = sous.add_parser("lister", help="afficher le catalogue des filtres")
    p_lister.add_argument("--famille", help="ne garder que les familles correspondantes")
    p_lister.add_argument("-v", "--verbeux", action="store_true",
                          help="afficher résumé et paramètres")
    p_lister.set_defaults(func=cmd_lister)

    p_decrire = sous.add_parser("decrire", help="fiche détaillée d'un filtre")
    p_decrire.add_argument("filtre", help="identifiant du filtre")
    p_decrire.set_defaults(func=cmd_decrire)

    p_appliquer = sous.add_parser("appliquer", help="appliquer une chaîne à une image")
    p_appliquer.add_argument("entree", nargs="?", help="image d'entrée")
    p_appliquer.add_argument("--webcam", type=int, metavar="INDEX",
                             help="utiliser un instantané de webcam comme entrée")
    p_appliquer.add_argument("-s", "--etape", action="append", type=parse_step,
                             metavar="FILTRE[:clé=valeur,...]",
                             help="étape à appliquer (répétable, dans l'ordre)")
    p_appliquer.add_argument("--preset", help="chaîne enregistrée (JSON)")
    p_appliquer.add_argument("--reference", help="deuxième image (fusion, ET binaire)")
    p_appliquer.add_argument("-o", "--sortie", help="fichier image de sortie")
    p_appliquer.add_argument("--histogramme", metavar="FICHIER",
                             help="enregistrer aussi le tracé de l'histogramme")
    p_appliquer.add_argument("--sauver-preset", metavar="FICHIER",
                             help="enregistrer la chaîne utilisée au format JSON")
    p_appliquer.add_argument("--qualite", type=int, default=95,
                             help="qualité JPEG/WebP (défaut : 95)")
    p_appliquer.add_argument("--etapes", action="store_true",
                             help="détailler chaque étape")
    p_appliquer.set_defaults(func=cmd_appliquer)

    p_comparer = sous.add_parser(
        "comparer", help="comparer plusieurs filtres par MSE / PSNR (TP7 ex.4)"
    )
    p_comparer.add_argument("entree", nargs="?", help="image d'entrée")
    p_comparer.add_argument("--webcam", type=int, metavar="INDEX")
    p_comparer.add_argument("-s", "--etape", action="append", type=parse_step,
                            metavar="FILTRE[:clé=valeur,...]",
                            help="filtre à évaluer (répétable)")
    p_comparer.add_argument("--reference", metavar="IMAGE",
                            help="image propre de référence (sinon l'entrée elle-même)")
    p_comparer.add_argument("--dossier-sortie", metavar="DOSSIER",
                            help="enregistrer chaque résultat dans ce dossier")
    p_comparer.set_defaults(func=cmd_comparer)

    p_webcam = sous.add_parser("webcam", help="tester l'accès à la webcam")
    p_webcam.add_argument("--index", type=int, default=0)
    p_webcam.add_argument("--largeur", type=int)
    p_webcam.add_argument("--hauteur", type=int)
    p_webcam.add_argument("-o", "--sortie", help="enregistrer l'instantané")
    p_webcam.set_defaults(func=cmd_webcam)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (io_utils.ImageLoadError, io_utils.CameraError, FilterError,
            FileNotFoundError, ValueError) as exc:
        print(f"erreur : {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrompu", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
