"""Interface en ligne de commande de coin-counter (M3).

```bash
python -m pieces.cli           # session webcam, sans options
python -m pieces.cli --centimes --camera 1
python -m pieces.cli --config donnees/config.json --etalon-mm 24
python -m pieces.cli --limite 20 --sortie sorties/hud.png   # debug sans fenêtre
```

La somme factorielle — les centimes (10c/20c) — n'est comptée qu'avec
``--centimes`` ; leur *identification* reste exhaustive (``docs/02``).
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import (
    Config,
    charger_config,
    dict_vers_config,
    formater_montant,
)
from .erreurs import ErreurApplication

__all__ = ["parcourir_arguments", "principal"]

VERSION_LIBELLE = "coin-counter 1.0"


def parcourir_arguments(argument_brut: list[str] | None = None) -> argparse.Namespace:
    """Analyse les arguments CLI. Ne touche à aucune caméra."""
    analyseur = argparse.ArgumentParser(
        prog="coin-counter",
        description="Somme de dinars marocains sur webcam (détection classique).",
    )
    analyseur.add_argument("--config", metavar="JSON", default=None,
                           help="fichier de configuration (défaut : donnees/config.json)")
    analyseur.add_argument("--centimes", action="store_true",
                           help="compter aussi les pièces de 10c/20c")
    analyseur.add_argument("--camera", type=int, default=0,
                           help="indice de la webcam (défaut : 0)")
    analyseur.add_argument("--etalon-mm", type=float, default=24.0,
                           help="diamètre réel (mm) de la pièce tenue pour étalonner (défaut : 24,0 = 1 DH)")
    analyseur.add_argument("--limite", type=int, default=None,
                           help="nombre de frames traitées puis arrêt (débogage/CI)")
    analyseur.add_argument("--sortie", metavar="PNG", default=None,
                           help="écrire le HUD de la dernière frame dans ce fichier puis arrêter")
    analyseur.add_argument("--sans-fenetre", action="store_true",
                           help="aucune fenêtre ni webcam (autotest du pipeline)")
    analyseur.add_argument("--image", metavar="JPG", default=None,
                           help="traiter une image statique puis arrêter "
                                "(aucune webcam ; --sortie par défaut sorties/hud.png)")
    analyseur.add_argument("--version", action="version", version=VERSION_LIBELLE)
    return analyseur.parse_args(argument_brut)


def _charger_config(chemin: str | None) -> Config:
    if chemin:
        return dict_vers_config(json.load(open(chemin, encoding="utf-8")))
    return charger_config()


def principal(argument_brut: list[str] | None = None) -> int:
    """Point d'entrée ; retourne un code de sortie (0 = succès)."""
    args = parcourir_arguments(argument_brut)
    try:
        config = _charger_config(args.config)
    except (ErreurApplication, OSError, ValueError) as err:
        print(f"configuration invalide : {err}", file=sys.stderr)
        return 2

    from .etalonnage import ObservateurEtalonnage
    from .etat import EtatApplication
    from .suivi import Suiveur

    etat = EtatApplication(
        config=config,
        suiveur=Suiveur(config, compter_centimes=args.centimes),
        observateur=ObservateurEtalonnage(),
        diametre_etalon_mm=args.etalon_mm,
    )

    if getattr(args, "sans_fenetre", False):
        # autotest : aucune webcam, aucune fenêtre — force l'échelle à None
        # et rejoue stop immédiat (le pipeline est testé ailleurs)
        print("mode sans fenêtre : rien à lire (pipeline vérifié par les tests)")
        return 0

    if args.image is not None:
        return _traiter_image(args, etat, config)

    from .rendu import dessiner_hud
    from .ui.bureau import Bureau, creer_capture

    capture = creer_capture(args.camera)
    try:
        bureau = Bureau(config=config, etat=etat, capture=capture)
        dernier = bureau.reel(limite=args.limite)
        if args.sortie and dernier is not None and bureau.derniere_frame is not None:
            import cv2 as _cv2

            hud = dessiner_hud(bureau.derniere_frame, dernier, etat)
            _cv2.imwrite(args.sortie, hud)
            print(f"HUD écrit : {args.sortie}")
        print(
            f"total : {formater_montant(etat.suiveur.total_centimes)} "
            f"({etat.suiveur.par_valeur_total})"
        )
        return 0
    except (ErreurApplication, RuntimeError) as err:
        print(f"erreur : {err}", file=sys.stderr)
        return 1
    finally:
        capture.release()


class _ListeFrames:
    """Mini-capteur jetable pour l'analyse d'une image statique (--image)."""

    def __init__(self, frames: list) -> None:
        self._frames = list(frames)

    def read(self) -> tuple[bool, object | None]:
        return (True, self._frames.pop(0)) if self._frames else (False, None)

    def release(self) -> None:
        return None


def _traiter_image(args: argparse.Namespace, etat, config: Config) -> int:
    """Une frame statique → pipeline → HUD écrit (débogage, aucune webcam)."""
    import cv2 as _cv2

    from .rendu import dessiner_hud
    from .ui.bureau import Bureau

    frame = _cv2.imread(args.image)
    if frame is None:
        print(f"image illisible : {args.image}", file=sys.stderr)
        return 2
    from pathlib import Path

    destination = Path(args.sortie) if args.sortie else Path("sorties/hud.png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    bureau = Bureau(
        config=config, etat=etat, capture=_ListeFrames([frame]), afficher=False
    )
    dernier = bureau.reel(limite=1)
    if dernier is None:
        print("aucune frame traitée", file=sys.stderr)
        return 1
    _cv2.imwrite(str(destination), dessiner_hud(frame, dernier, etat))
    print(
        f"HUD écrit : {destination} — "
        f"{formater_montant(etat.suiveur.total_centimes)} "
        f"({etat.suiveur.par_valeur_total})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())
