"""Interface en ligne de commande de coin-counter (M3).

```bash
python -m pieces.cli --image photo.jpg             # image fixe → inventaire
python -m pieces.cli --camera 0 --centimes         # direct, avec 5c/10c/20c
python -m pieces.cli --sans-fenetre                # autotest, sans matériel
python -m pieces.cli --image photo.jpg --json rapport.json
```

Deux régimes, volontairement distincts :

* **image fixe** → un *inventaire* : « quelles pièces vois-tu ici, et combien
  font-elles ? ». Aucune confirmation temporelle n'a de sens sur une frame
  unique (c'est ce qui rendait l'ancien ``--image`` structurellement muet :
  le suivi exige trois présences, une image n'en fournit qu'une) ;
* **webcam** → un *total suivi*, où une pièce doit être revue plusieurs
  frames avant d'être comptée, et ne peut l'être qu'une fois.

L'échelle (px/mm) peut venir de trois sources, par ordre de préférence :
``--echelle-px-mm`` (mesurée, fournie), l'étalonnage interactif (touche
``c``), ou l'estimation au catalogue des diamètres marocains. La dernière
est toujours annoncée comme **estimée**, jamais comme mesurée.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import (
    Config,
    charger_config,
    formater_montant,
)
from .erreurs import ErreurApplication

__all__ = ["parcourir_arguments", "principal"]

VERSION_LIBELLE = "coin-counter 1.1"
#: bornes de bon sens pour --etalon-mm (une pièce, pas une assiette)
ETALON_MM_MIN = 10.0
ETALON_MM_MAX = 40.0


def _etalon_mm(valeur: str) -> float:
    """Valide ``--etalon-mm`` dès l'analyse des arguments.

    Sans ce contrôle, une valeur absurde produisait deux échecs distincts et
    tous deux mauvais : une valeur ≤ 0 faisait remonter une exception jusqu'à
    terminer la session (et perdre le total accumulé) à la première tentative
    d'étalonnage, tandis qu'une valeur énorme bouclait en silence, l'échec
    n'étant écrit que dans un journal que personne n'affichait.
    """
    try:
        nombre = float(valeur)
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"nombre attendu, reçu {valeur!r}") from err
    if not ETALON_MM_MIN <= nombre <= ETALON_MM_MAX:
        raise argparse.ArgumentTypeError(
            f"diamètre d'étalon {nombre} mm hors bornes plausibles "
            f"[{ETALON_MM_MIN:.0f}, {ETALON_MM_MAX:.0f}]"
        )
    return nombre


def _echelle_px_mm(valeur: str) -> float:
    try:
        nombre = float(valeur)
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"nombre attendu, reçu {valeur!r}") from err
    if nombre <= 0.0:
        raise argparse.ArgumentTypeError(f"échelle {nombre} : doit être > 0")
    return nombre


def parcourir_arguments(argument_brut: list[str] | None = None) -> argparse.Namespace:
    """Analyse les arguments CLI. Ne touche à aucune caméra."""
    analyseur = argparse.ArgumentParser(
        prog="coin-counter",
        description="Somme de dirhams marocains (traitement d'image classique).",
    )
    analyseur.add_argument("--config", metavar="JSON", default=None,
                           help="fichier de configuration (défaut : donnees/config.json)")
    analyseur.add_argument("--centimes", action="store_true",
                           help="compter aussi les pièces de 5c/10c/20c")
    analyseur.add_argument("--camera", type=int, default=None,
                           help="indice de la webcam (session en direct)")
    analyseur.add_argument("--image", metavar="JPG", default=None,
                           help="analyser une image fixe puis s'arrêter")
    analyseur.add_argument("--echelle-px-mm", type=_echelle_px_mm, default=None,
                           metavar="N",
                           help="échelle connue en pixels par millimètre à "
                                "l'échelle d'analyse (désactive l'estimation)")
    analyseur.add_argument("--sans-auto-echelle", action="store_true",
                           help="interdire l'estimation de l'échelle au catalogue")
    analyseur.add_argument("--etalon-mm", type=_etalon_mm, default=24.0,
                           help="diamètre réel (mm) de la pièce tenue pour "
                                "étalonner (défaut : 24,0 = 1 DH)")
    analyseur.add_argument("--limite", type=int, default=None,
                           help="nombre de frames traitées puis arrêt")
    analyseur.add_argument("--sortie", metavar="PNG", default=None,
                           help="écrire le HUD dans ce fichier")
    analyseur.add_argument("--json", metavar="JSON", default=None,
                           help="écrire un rapport lisible par une machine")
    analyseur.add_argument("--lister-pieces", action="store_true",
                           help="détailler chaque pièce sur la sortie standard")
    analyseur.add_argument("--sans-fenetre", action="store_true",
                           help="autotest du pipeline sur la scène de "
                                "référence : aucune caméra, aucune fenêtre")
    analyseur.add_argument("--version", action="version", version=VERSION_LIBELLE)
    return analyseur.parse_args(argument_brut)


def _charger_config(chemin: str | None) -> Config:
    """Charge la configuration — même chemin d'erreurs que ``charger_config``."""
    return charger_config(chemin) if chemin else charger_config()


# ---------------------------------------------------------------------------
# Rendu des résultats
# ---------------------------------------------------------------------------


def _ligne_echelle(px_par_mm: float | None, estimee: bool) -> str:
    if px_par_mm is None:
        return "échelle : indéterminée"
    origine = "estimée au catalogue" if estimee else "mesurée"
    return f"échelle : {px_par_mm:.2f} px/mm ({origine})"


def _detailler(pieces) -> None:
    for piece in pieces:
        classification = piece.classification
        libelle = classification.libelle if classification else "—"
        certitude = classification.certitude if classification else "—"
        motif = piece.motif_incertitude or ""
        print(
            f"  ({piece.centre[0]:7.1f},{piece.centre[1]:7.1f}) "
            f"Ø{piece.diametre:6.1f}px  {str(piece.famille or '—'):>13}  "
            f"{libelle:>10}  {certitude:>9}  {motif}"
        )


def _rapport(objet: dict, chemin: str) -> None:
    cible = Path(chemin)
    cible.parent.mkdir(parents=True, exist_ok=True)
    cible.write_text(
        json.dumps(objet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"rapport écrit : {cible}")


def _piece_json(piece) -> dict:
    classification = piece.classification
    return {
        "centre": [round(piece.centre[0], 2), round(piece.centre[1], 2)],
        "diametre_px": round(piece.diametre, 2),
        "famille": piece.famille,
        "polarite": piece.polarite,
        "libelle": classification.libelle if classification else None,
        "valeur_centimes": classification.valeur_centimes if classification else None,
        "certitude": classification.certitude if classification else None,
        "motif_incertitude": piece.motif_incertitude,
        "piste_id": piece.piste_id,
    }


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def _mode_image(args: argparse.Namespace, config: Config) -> int:
    """Image fixe → inventaire (aucun suivi, aucune caméra)."""
    import cv2

    from .traitement import inventorier_image

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"image illisible : {args.image}", file=sys.stderr)
        return 2

    inventaire = inventorier_image(
        frame,
        config,
        px_par_mm=args.echelle_px_mm,
        auto_echelle=not args.sans_auto_echelle,
        centimes=args.centimes,
    )
    for message in inventaire.messages:
        print(message)

    valides = [p for p in inventaire.pieces if p.motif_incertitude is None]
    print(f"pièces détectées : {len(valides)} (profil {inventaire.profil})")
    print(_ligne_echelle(inventaire.echelle_px_mm, inventaire.echelle_estimee))
    libelles = {d.valeur_centimes: d.libelle for d in config.denominations}
    detail = "  ".join(
        f"{libelles.get(v, '?')}×{n}" for v, n in sorted(inventaire.par_valeur.items())
    )
    print(f"TOTAL : {formater_montant(inventaire.total_centimes)}"
          + (f"   {detail}" if detail else ""))
    if args.lister_pieces:
        _detailler(inventaire.pieces)

    if args.sortie:
        from .rendu import dessiner_hud_inventaire

        destination = Path(args.sortie)
        destination.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destination), dessiner_hud_inventaire(frame, inventaire, config))
        print(f"HUD écrit : {destination}")

    if args.json:
        _rapport(
            {
                "mode": "image",
                "source": args.image,
                "profil": inventaire.profil,
                "echelle_px_mm": inventaire.echelle_px_mm,
                "echelle_estimee": inventaire.echelle_estimee,
                "analyse_ms": round(inventaire.analyse_ms, 3),
                "total_centimes": inventaire.total_centimes,
                "total": formater_montant(inventaire.total_centimes),
                "par_valeur": {str(k): v for k, v in inventaire.par_valeur.items()},
                "pieces": [_piece_json(p) for p in inventaire.pieces],
            },
            args.json,
        )
    return 0


def _mode_autotest(args: argparse.Namespace, config: Config) -> int:
    """Autotest : le pipeline complet sur une scène de vérité connue.

    L'ancien ``--sans-fenetre`` affichait une phrase et rendait 0 sans rien
    exécuter, tout en étant documenté comme « autotest du pipeline » : il ne
    pouvait détecter aucune régression. Ici la scène est synthétisée à une
    échelle **imposée**, donc son total est connu par construction, et le
    code de sortie reflète réellement l'état du pipeline.
    """
    from .scenes import scene_etalon
    from .traitement import inventorier_image

    scene = scene_etalon()
    inventaire = inventorier_image(
        scene.image, config, auto_echelle=True, centimes=args.centimes
    )
    attendu = (
        scene.total_centimes if args.centimes else scene.total_sans_centimes
    )
    valides = [p for p in inventaire.pieces if p.motif_incertitude is None]
    attendu_pieces = len(scene.inventaire)

    print(f"scène de référence : {attendu_pieces} pièces, "
          f"échelle imposée {scene.px_par_mm:.2f} px/mm")
    print(f"pièces détectées   : {len(valides)}/{attendu_pieces}")
    print(_ligne_echelle(inventaire.echelle_px_mm, inventaire.echelle_estimee))
    print(f"total obtenu       : {formater_montant(inventaire.total_centimes)}")
    print(f"total attendu      : {formater_montant(attendu)}")
    if args.lister_pieces:
        _detailler(inventaire.pieces)

    succes = (
        len(valides) == attendu_pieces
        and inventaire.total_centimes == attendu
    )
    print("autotest : " + ("RÉUSSI" if succes else "ÉCHOUÉ"))
    return 0 if succes else 1


def _mode_webcam(args: argparse.Namespace, config: Config) -> int:
    """Session en direct : total suivi, une pièce comptée une seule fois."""
    from .etalonnage import ObservateurEtalonnage
    from .etat import EtatApplication
    from .rendu import dessiner_hud
    from .suivi import Suiveur
    from .ui.bureau import Bureau, creer_capture

    etat = EtatApplication(
        config=config,
        suiveur=Suiveur(config, compter_centimes=args.centimes),
        observateur=ObservateurEtalonnage(),
        diametre_etalon_mm=args.etalon_mm,
        px_par_mm=args.echelle_px_mm,
        auto_echelle=not args.sans_auto_echelle,
    )

    capture = None
    try:
        # creer_capture est DANS le bloc gardé : il était auparavant appelé
        # avant le try, si bien que l'absence de caméra — le premier cas que
        # rencontre un utilisateur — remontait en trace d'appel brute.
        capture = creer_capture(args.camera)
        bureau = Bureau(config=config, etat=etat, capture=capture)
        dernier = bureau.reel(limite=args.limite)
        if args.sortie and dernier is not None and bureau.derniere_frame is not None:
            import cv2

            destination = Path(args.sortie)
            destination.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(destination), dessiner_hud(bureau.derniere_frame, dernier, etat)
            )
            print(f"HUD écrit : {destination}")
        print(_ligne_echelle(etat.px_par_mm, etat.echelle_estimee))
        print(
            f"TOTAL : {formater_montant(etat.suiveur.total_centimes)} "
            f"({etat.suiveur.par_valeur_total})"
        )
        if args.json:
            _rapport(
                {
                    "mode": "webcam",
                    "camera": args.camera,
                    "frames": etat.num_frame,
                    "echelle_px_mm": etat.px_par_mm,
                    "echelle_estimee": etat.echelle_estimee,
                    "total_centimes": etat.suiveur.total_centimes,
                    "total": formater_montant(etat.suiveur.total_centimes),
                    "par_valeur": {
                        str(k): v for k, v in etat.suiveur.par_valeur_total.items()
                    },
                    "pieces": [_piece_json(p) for p in (dernier.pieces if dernier else [])],
                },
                args.json,
            )
        return 0
    except (ErreurApplication, RuntimeError) as err:
        print(f"erreur : {err}", file=sys.stderr)
        return 1
    finally:
        if capture is not None:
            capture.release()


def principal(argument_brut: list[str] | None = None) -> int:
    """Point d'entrée ; retourne un code de sortie (0 = succès)."""
    args = parcourir_arguments(argument_brut)
    try:
        config = _charger_config(args.config)
    except (ErreurApplication, OSError, ValueError) as err:
        print(f"configuration invalide : {err}", file=sys.stderr)
        return 2

    if args.sans_fenetre:
        return _mode_autotest(args, config)
    if args.image is not None:
        return _mode_image(args, config)
    if args.camera is not None:
        return _mode_webcam(args, config)

    print(
        "rien à faire : précisez --image FICHIER, --camera N, ou "
        "--sans-fenetre (autotest). Voir --help.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(principal())
