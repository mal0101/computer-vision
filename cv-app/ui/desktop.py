"""Atelier de filtres — interface de bureau OpenCV (trackbars + webcam).

Cette interface est la continuation directe du TP3 : les paramètres sont réglés
par des **trackbars** (``cv2.createTrackbar``), la souris sert à désigner des
points sur l'image (``cv2.setMouseCallback``), et la source peut être un **flux
webcam continu** comme au TP1 §3.4 — ce qu'une page web ne peut pas faire sans
dépendance supplémentaire.

Lancement ::

    python ui/desktop.py                       # dernière image d'exemple des TP
    python ui/desktop.py image.jpg             # un fichier
    python ui/desktop.py --webcam              # flux webcam temps réel
    python ui/desktop.py --webcam 1 --largeur 1280 --hauteur 720
    python ui/desktop.py image.jpg --filtre canny
    python ui/desktop.py image.jpg --preset ../presets/contours_canny.json

Touches ::

    n / p      filtre suivant / précédent
    f          liste des filtres dans le terminal
    o          afficher / masquer l'original
    h          afficher / masquer l'histogramme
    i          afficher / masquer l'aide incrustée
    m          afficher / masquer les mesures d'écart (MSE, PSNR)
    r          réinitialiser les paramètres du filtre courant
    c          figer l'image courante de la webcam (gel / dégel)
    s          enregistrer l'image affichée
    ESC ou q   quitter

Souris ::

    déplacement        sonde de pixel (coordonnées et valeur BGR/HSV)
    glisser (gauche)   définit la région du filtre « Recadrer »
    clic gauche ×4     définit les 4 points du filtre « Déformation de perspective »

Les trackbars ne manipulent que des entiers positifs : les paramètres réels
sont donc exposés à une résolution fixe (voir :class:`EchelleTrackbar`), et les
paramètres impairs avancent de deux en deux. Les types que les trackbars ne
savent pas représenter (texte, couleur) gardent leur valeur par défaut, ce que
l'aide incrustée signale.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RACINE = Path(__file__).resolve().parent.parent
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cvlab import drawing as ds  # noqa: E402
from cvlab import histogram as hist  # noqa: E402
from cvlab import images as im  # noqa: E402
from cvlab import io_utils, metrics, registry  # noqa: E402
from cvlab.params import (  # noqa: E402
    BoolParam,
    ChoiceParam,
    ColorParam,
    FloatParam,
    IntParam,
    ParamSpec,
    TextParam,
)
from cvlab.pipeline import Pipeline  # noqa: E402
from cvlab.registry import FilterDef, FilterError  # noqa: E402

FENETRE = "Atelier de filtres (OpenCV)"
FENETRE_REGLAGES = "Reglages"  # ASCII : certains backends gèrent mal les accents
FENETRE_ORIGINAL = "Original"
FENETRE_HISTOGRAMME = "Histogramme"
TOUCHE_ECHAP = 27

#: Nombre de crans d'une trackbar représentant un réel.
CRANS_REEL = 200


@dataclass
class EchelleTrackbar:
    """Conversion entre la position entière d'une trackbar et la valeur réelle.

    ``cv2.createTrackbar`` n'accepte que des entiers de 0 à ``count``. Pour un
    paramètre réel sur [min, max], on découpe l'intervalle en ``CRANS_REEL``
    crans ; pour un entier, la position est la valeur décalée de ``min`` (les
    bornes négatives sont donc gérées) ; pour un paramètre impair, le pas vaut 2.
    """

    spec: ParamSpec
    maximum_trackbar: int

    @classmethod
    def pour(cls, spec: ParamSpec) -> EchelleTrackbar | None:
        """Échelle adaptée à ``spec``, ou ``None`` si le type est inreprésentable."""
        if isinstance(spec, IntParam):
            if spec.odd_only:
                bas = spec.min if spec.min % 2 == 1 else spec.min + 1
                haut = spec.max if spec.max % 2 == 1 else spec.max - 1
                return cls(spec=spec, maximum_trackbar=max(1, (haut - bas) // 2))
            return cls(spec=spec, maximum_trackbar=max(1, spec.max - spec.min))
        if isinstance(spec, FloatParam):
            return cls(spec=spec, maximum_trackbar=CRANS_REEL)
        if isinstance(spec, BoolParam):
            return cls(spec=spec, maximum_trackbar=1)
        if isinstance(spec, ChoiceParam):
            return cls(spec=spec, maximum_trackbar=max(1, len(spec.options) - 1))
        return None  # texte, couleur : pas de trackbar

    def vers_position(self, valeur: Any) -> int:
        """Valeur du paramètre -> position de la trackbar."""
        spec = self.spec
        if isinstance(spec, IntParam):
            entier = spec.coerce(valeur)
            if spec.odd_only:
                bas = spec.min if spec.min % 2 == 1 else spec.min + 1
                return max(0, min(self.maximum_trackbar, (entier - bas) // 2))
            return max(0, min(self.maximum_trackbar, entier - spec.min))
        if isinstance(spec, FloatParam):
            reel = spec.coerce(valeur)
            etendue = spec.max - spec.min
            if etendue <= 0:
                return 0
            return int(round((reel - spec.min) / etendue * self.maximum_trackbar))
        if isinstance(spec, BoolParam):
            return 1 if spec.coerce(valeur) else 0
        if isinstance(spec, ChoiceParam):
            return spec.keys.index(spec.coerce(valeur))
        return 0

    def vers_valeur(self, position: int) -> Any:
        """Position de la trackbar -> valeur du paramètre."""
        spec = self.spec
        position = max(0, min(self.maximum_trackbar, int(position)))
        if isinstance(spec, IntParam):
            if spec.odd_only:
                bas = spec.min if spec.min % 2 == 1 else spec.min + 1
                return spec.coerce(bas + position * 2)
            return spec.coerce(spec.min + position)
        if isinstance(spec, FloatParam):
            etendue = spec.max - spec.min
            return spec.coerce(spec.min + etendue * position / self.maximum_trackbar)
        if isinstance(spec, BoolParam):
            return bool(position)
        if isinstance(spec, ChoiceParam):
            return spec.keys[position]
        return spec.default


def _nom_trackbar(spec: ParamSpec, largeur: int = 22) -> str:
    """Nom court et sans accent : les trackbars d'OpenCV ont peu de place.

    Le nom sert aussi de clé pour ``cv2.getTrackbarPos`` : il doit être unique
    au sein de la fenêtre, d'où l'usage du nom technique du paramètre.
    """
    sans_accent = (
        spec.name.replace("é", "e").replace("è", "e").replace("ê", "e")
        .replace("à", "a").replace("ç", "c").replace("ô", "o").replace("û", "u")
    )
    return sans_accent[:largeur]


class AtelierBureau:
    """Boucle d'affichage, trackbars et interactions souris."""

    def __init__(
        self,
        source_image: np.ndarray | None,
        camera: io_utils.Camera | None,
        filtres: Sequence[FilterDef],
        index_initial: int = 0,
        chaine_prefixe: Pipeline | None = None,
        reference: np.ndarray | None = None,
        dossier_sortie: Path | None = None,
    ) -> None:
        self.image_fixe = source_image
        self.camera = camera
        self.filtres = list(filtres)
        self.index = max(0, min(len(self.filtres) - 1, index_initial))
        self.chaine_prefixe = chaine_prefixe or Pipeline()
        self.reference = reference
        self.dossier_sortie = dossier_sortie or RACINE / "sorties"

        # Valeurs courantes de chaque filtre, conservées quand on change de
        # filtre puis qu'on revient : les réglages ne sont pas perdus.
        self.valeurs: dict[str, dict[str, Any]] = {
            d.id: d.defaults() for d in self.filtres
        }
        self.echelles: dict[str, EchelleTrackbar] = {}

        self.afficher_original = False
        self.afficher_histogramme = False
        self.afficher_aide = True
        self.afficher_mesures = True
        self.gele = False
        self.image_gelee: np.ndarray | None = None

        self.position_souris = (0, 0)
        self.points_perspective: list[tuple[int, int]] = []
        self.glissement_debut: tuple[int, int] | None = None
        self.glissement_courant: tuple[int, int] | None = None
        self.message = ""
        self.message_expire = 0.0
        self.derniere_sortie: np.ndarray | None = None
        self.derniere_entree: np.ndarray | None = None
        self.duree_ms = 0.0
        self.infos: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Filtre courant
    # ------------------------------------------------------------------
    @property
    def filtre(self) -> FilterDef:
        return self.filtres[self.index]

    def notifier(self, texte: str, duree: float = 2.5) -> None:
        """Message incrusté temporaire, doublé dans le terminal."""
        self.message = texte
        self.message_expire = time.monotonic() + duree
        print(texte)

    # ------------------------------------------------------------------
    # Fenêtres et trackbars
    # ------------------------------------------------------------------
    def construire_fenetres(self) -> None:
        cv2.namedWindow(FENETRE, cv2.WINDOW_NORMAL)
        cv2.namedWindow(FENETRE_REGLAGES, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(FENETRE_REGLAGES, 520, 60)
        cv2.setMouseCallback(FENETRE, self._souris)
        self.construire_trackbars()

    def construire_trackbars(self) -> None:
        """Recrée la fenêtre de réglages pour le filtre courant.

        OpenCV ne sait pas supprimer une trackbar : changer de filtre impose
        donc de détruire la fenêtre et de la reconstruire.
        """
        cv2.destroyWindow(FENETRE_REGLAGES)
        cv2.namedWindow(FENETRE_REGLAGES, cv2.WINDOW_NORMAL)

        definition = self.filtre
        self.echelles = {}
        reglables = 0
        for spec in definition.params:
            echelle = EchelleTrackbar.pour(spec)
            if echelle is None:
                continue
            nom = _nom_trackbar(spec)
            self.echelles[nom] = echelle
            position = echelle.vers_position(self.valeurs[definition.id][spec.name])
            # Le callback est obligatoire sous certains backends ; la lecture
            # réelle se fait par getTrackbarPos dans la boucle d'affichage, ce
            # qui évite tout problème de concurrence.
            cv2.createTrackbar(nom, FENETRE_REGLAGES, position,
                               echelle.maximum_trackbar, lambda _valeur: None)
            reglables += 1

        hauteur = max(60, 26 * reglables + 40)
        cv2.resizeWindow(FENETRE_REGLAGES, 560, hauteur)

        non_reglables = [
            s.name for s in definition.params
            if isinstance(s, (TextParam, ColorParam))
        ]
        print(f"\n=== {definition.name} ({definition.id}) ===")
        print(definition.summary)
        if non_reglables:
            print(
                "  (paramètres non réglables par trackbar, valeur par défaut "
                f"utilisée : {', '.join(non_reglables)} — "
                "utiliser l'interface web pour les modifier)"
            )

    def lire_trackbars(self) -> dict[str, Any]:
        """Lit la position de chaque trackbar et met à jour les valeurs."""
        definition = self.filtre
        valeurs = dict(self.valeurs[definition.id])
        for spec in definition.params:
            nom = _nom_trackbar(spec)
            echelle = self.echelles.get(nom)
            if echelle is None:
                continue
            try:
                position = cv2.getTrackbarPos(nom, FENETRE_REGLAGES)
            except cv2.error:  # fenêtre fermée par l'utilisateur
                continue
            valeurs[spec.name] = echelle.vers_valeur(position)
        self.valeurs[definition.id] = definition.coerce(valeurs)
        return self.valeurs[definition.id]

    # ------------------------------------------------------------------
    # Souris (TP3 §1 à §4)
    # ------------------------------------------------------------------
    def _souris(self, evenement: int, x: int, y: int, _drapeaux: int,
                _param: object) -> None:
        self.position_souris = (x, y)

        if evenement == cv2.EVENT_LBUTTONDOWN:
            self.glissement_debut = (x, y)
            self.glissement_courant = (x, y)
            if self.filtre.id == "deformation_perspective":
                self.points_perspective.append((x, y))
                if len(self.points_perspective) >= 4:
                    self._appliquer_points_perspective()
                else:
                    self.notifier(
                        f"Point {len(self.points_perspective)}/4 placé "
                        "(ordre : haut-gauche, haut-droit, bas-droit, bas-gauche)"
                    )

        elif evenement == cv2.EVENT_MOUSEMOVE and self.glissement_debut is not None:
            self.glissement_courant = (x, y)

        elif evenement == cv2.EVENT_LBUTTONUP:
            if self.filtre.id == "recadrer" and self.glissement_debut is not None:
                self._appliquer_rectangle(self.glissement_debut, (x, y))
            self.glissement_debut = None
            self.glissement_courant = None

        elif evenement == cv2.EVENT_RBUTTONDOWN:
            self.points_perspective.clear()
            self.notifier("Points de perspective effacés")

    def _taille_entree(self) -> tuple[int, int]:
        """Dimensions de l'image sur laquelle les clics sont interprétés."""
        if self.derniere_entree is None:
            return (1, 1)
        hauteur, largeur = self.derniere_entree.shape[:2]
        return (largeur, hauteur)

    def _appliquer_rectangle(
        self, premier: tuple[int, int], second: tuple[int, int]
    ) -> None:
        """Traduit un glisser-déposer en paramètres du filtre « Recadrer »."""
        largeur, hauteur = self._taille_entree()
        x1, x2 = sorted((premier[0], second[0]))
        y1, y2 = sorted((premier[1], second[1]))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return
        valeurs = dict(self.valeurs["recadrer"])
        valeurs.update({
            "x": 100.0 * x1 / largeur,
            "y": 100.0 * y1 / hauteur,
            "largeur": 100.0 * (x2 - x1) / largeur,
            "hauteur": 100.0 * (y2 - y1) / hauteur,
        })
        self.valeurs["recadrer"] = registry.get("recadrer").coerce(valeurs)
        self.construire_trackbars()
        self.notifier(f"Recadrage ({x1}, {y1}) → ({x2}, {y2})")

    def _appliquer_points_perspective(self) -> None:
        """Traduit quatre clics en paramètres d'homographie (TP4 §3)."""
        largeur, hauteur = self._taille_entree()
        valeurs = dict(self.valeurs["deformation_perspective"])
        for indice, (x, y) in enumerate(self.points_perspective[:4], start=1):
            valeurs[f"p{indice}x"] = 100.0 * x / largeur
            valeurs[f"p{indice}y"] = 100.0 * y / hauteur
        self.valeurs["deformation_perspective"] = registry.get(
            "deformation_perspective"
        ).coerce(valeurs)
        self.points_perspective.clear()
        self.construire_trackbars()
        self.notifier("Homographie mise à jour depuis les 4 points")

    # ------------------------------------------------------------------
    # Acquisition et traitement
    # ------------------------------------------------------------------
    def image_source(self) -> np.ndarray | None:
        """Image courante : flux webcam, image gelée, ou fichier."""
        if self.gele and self.image_gelee is not None:
            return self.image_gelee
        if self.camera is not None:
            image = self.camera.read_optional()
            if image is None:
                return self.image_gelee
            self.image_gelee = image
            return image
        return self.image_fixe

    def traiter(self, source: np.ndarray) -> np.ndarray:
        """Applique le préfixe de chaîne puis le filtre courant."""
        entree = source
        if len(self.chaine_prefixe):
            entree = self.chaine_prefixe.apply(entree, self.reference,
                                               collect=False).image
        self.derniere_entree = entree

        definition = self.filtre
        valeurs = self.lire_trackbars()
        debut = time.perf_counter()
        try:
            sortie, self.infos = definition.apply(entree, valeurs, self.reference)
        except (FilterError, im.InvalidImageError, ValueError) as exc:
            self.infos = {}
            self.duree_ms = 0.0
            sortie = ds.dess_texte_encadre(
                im.to_bgr(entree), f"{definition.name}: {exc}"[:90], (12, 30),
                0.45, (120, 160, 255), (10, 10, 10),
            )
        else:
            self.duree_ms = (time.perf_counter() - debut) * 1000.0
        return sortie

    # ------------------------------------------------------------------
    # Incrustations
    # ------------------------------------------------------------------
    def incruster(self, affichee: np.ndarray, entree: np.ndarray) -> np.ndarray:
        """Ajoute l'aide, la sonde de pixel et les mesures sur l'image affichée."""
        toile = im.to_bgr(affichee)
        hauteur, largeur = toile.shape[:2]
        definition = self.filtre

        # Rectangle de recadrage en cours de tracé (TP3 §4).
        if self.glissement_debut and self.glissement_courant:
            toile = ds.dess_rectangle(toile, self.glissement_debut,
                                      self.glissement_courant, (80, 220, 120), 2,
                                      cv2.LINE_8)
        # Points de perspective déjà posés (TP4 §3).
        for indice, point in enumerate(self.points_perspective, start=1):
            toile = ds.dess_cercle(toile, point, 5, (60, 200, 255), 2)
            toile = ds.dess_text(toile, str(indice),
                                 (point[0] + 8, point[1] - 8), 0.5,
                                 (60, 200, 255), 1)

        lignes: list[str] = []
        if self.afficher_aide:
            lignes.append(
                f"[{self.index + 1}/{len(self.filtres)}] {definition.name}"
            )
            valeurs = self.valeurs[definition.id]
            if valeurs:
                resume = ", ".join(f"{k}={v}" for k, v in valeurs.items())
                lignes.extend(_couper(resume, 76))
            lignes.append(f"{self.duree_ms:.1f} ms   {largeur}x{hauteur}")
            for cle, valeur in list(self.infos.items())[:3]:
                texte = str(valeur).replace("\n", " ")
                lignes.append(f"{cle}: {texte}"[:76])

        if self.afficher_mesures and self.derniere_sortie is not None:
            mesures = metrics.compare(entree, self.derniere_sortie)
            pic = mesures["PSNR (dB)"]
            lignes.append(
                f"MSE {mesures['MSE']:.1f}  PSNR "
                + ("inf" if pic == float("inf") else f"{pic:.1f} dB")
            )

        # Sonde de pixel : coordonnées et valeur sous le curseur (TP3 §1).
        x, y = self.position_souris
        if 0 <= x < largeur and 0 <= y < hauteur:
            if im.is_gray(affichee):
                lignes.append(f"(x={x}, y={y})  intensite={int(affichee[y, x])}")
            else:
                bleu, vert, rouge = (int(c) for c in toile[y, x])
                lignes.append(f"(x={x}, y={y})  B={bleu} G={vert} R={rouge}")

        if self.afficher_aide:
            lignes.append("n/p filtre  o original  h histo  m mesures  "
                          "r reset  s sauver  i aide  ESC quitter")
            if self.camera is not None:
                lignes.append("c gel/degel du flux")

        if self.message and time.monotonic() < self.message_expire:
            lignes.append(f">> {self.message}")

        ordonnee = 22
        for ligne in lignes:
            toile = ds.dess_texte_encadre(toile, ligne, (12, ordonnee), 0.44,
                                          (235, 235, 235), (12, 12, 14), 1)
            ordonnee += 20
        return toile

    # ------------------------------------------------------------------
    # Actions clavier
    # ------------------------------------------------------------------
    def changer_filtre(self, delta: int) -> None:
        self.index = (self.index + delta) % len(self.filtres)
        self.points_perspective.clear()
        self.construire_trackbars()

    def reinitialiser(self) -> None:
        self.valeurs[self.filtre.id] = self.filtre.defaults()
        self.construire_trackbars()
        self.notifier(f"{self.filtre.name} : paramètres par défaut")

    def enregistrer(self) -> None:
        if self.derniere_sortie is None:
            return
        horodatage = time.strftime("%Y%m%d-%H%M%S")
        chemin = self.dossier_sortie / f"{self.filtre.id}-{horodatage}.png"
        io_utils.save_image(self.derniere_sortie, chemin)
        self.notifier(f"Enregistré : {chemin}")

    def lister_filtres(self) -> None:
        print("\nFiltres disponibles :")
        for position, definition in enumerate(self.filtres):
            marque = "→" if position == self.index else " "
            print(f" {marque} {position + 1:>2}. {definition.id:<32} {definition.name}")

    # ------------------------------------------------------------------
    # Boucle principale
    # ------------------------------------------------------------------
    def executer(self) -> int:
        self.construire_fenetres()
        self.lister_filtres()
        print(f"\n{FENETRE} — 'i' pour l'aide incrustée, ESC pour quitter.")

        while True:
            source = self.image_source()
            if source is None:
                print("aucune image à afficher", file=sys.stderr)
                return 1

            sortie = self.traiter(source)
            self.derniere_sortie = sortie
            assert self.derniere_entree is not None

            cv2.imshow(FENETRE, self.incruster(sortie, self.derniere_entree))

            if self.afficher_original:
                cv2.imshow(FENETRE_ORIGINAL, im.to_bgr(self.derniere_entree))
            if self.afficher_histogramme:
                cv2.imshow(FENETRE_HISTOGRAMME,
                           hist.draw_histogram(sortie, 640, 360))

            # 15 ms : compromis entre fluidité (≈ 60 im/s au plus) et
            # consommation processeur quand la source est une image fixe.
            touche = cv2.waitKey(15) & 0xFF
            if touche in (TOUCHE_ECHAP, ord("q")):
                break
            if touche == ord("n"):
                self.changer_filtre(1)
            elif touche == ord("p"):
                self.changer_filtre(-1)
            elif touche == ord("f"):
                self.lister_filtres()
            elif touche == ord("o"):
                self.afficher_original = not self.afficher_original
                if not self.afficher_original:
                    cv2.destroyWindow(FENETRE_ORIGINAL)
            elif touche == ord("h"):
                self.afficher_histogramme = not self.afficher_histogramme
                if not self.afficher_histogramme:
                    cv2.destroyWindow(FENETRE_HISTOGRAMME)
            elif touche == ord("i"):
                self.afficher_aide = not self.afficher_aide
            elif touche == ord("m"):
                self.afficher_mesures = not self.afficher_mesures
            elif touche == ord("r"):
                self.reinitialiser()
            elif touche == ord("s"):
                self.enregistrer()
            elif touche == ord("c") and self.camera is not None:
                self.gele = not self.gele
                self.notifier("Flux gelé" if self.gele else "Flux repris")

        cv2.destroyAllWindows()
        return 0


def _couper(texte: str, largeur: int) -> list[str]:
    """Découpe une chaîne en lignes d'au plus ``largeur`` caractères."""
    mots = texte.split(", ")
    lignes: list[str] = []
    courante = ""
    for mot in mots:
        candidat = f"{courante}, {mot}" if courante else mot
        if len(candidat) > largeur:
            if courante:
                lignes.append(courante)
            courante = mot
        else:
            courante = candidat
    if courante:
        lignes.append(courante)
    return lignes[:4]


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python ui/desktop.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", nargs="?", help="image à traiter")
    parser.add_argument("--webcam", nargs="?", type=int, const=0, default=None,
                        metavar="INDEX",
                        help="utiliser la webcam en flux continu (défaut : 0)")
    parser.add_argument("--largeur", type=int, help="largeur demandée à la webcam")
    parser.add_argument("--hauteur", type=int, help="hauteur demandée à la webcam")
    parser.add_argument("--filtre", help="identifiant du filtre affiché au démarrage")
    parser.add_argument("--famille", help="restreindre aux filtres d'une famille")
    parser.add_argument("--preset", help="chaîne JSON appliquée AVANT le filtre réglé")
    parser.add_argument("--reference", help="deuxième image (fusion, ET binaire)")
    parser.add_argument("--max-cote", type=int, default=1000, metavar="PX",
                        help="réduire l'image fixe si son plus grand côté dépasse")
    parser.add_argument("--sorties", help="dossier d'enregistrement (touche « s »)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    filtres = registry.all_filters()
    if args.famille:
        filtres = [
            d for d in filtres if args.famille.lower() in d.family.lower()
        ]
        if not filtres:
            print(f"aucune famille ne correspond à « {args.famille} »",
                  file=sys.stderr)
            return 2
    # Les filtres à deux images ne sont utilisables que si une référence est
    # fournie : sinon ils afficheraient une erreur à chaque image.
    reference = io_utils.load_image(args.reference) if args.reference else None
    if reference is None:
        filtres = [d for d in filtres if not d.needs_reference]

    index_initial = 0
    if args.filtre:
        try:
            definition = registry.get(args.filtre)
        except KeyError as exc:
            print(exc.args[0] if exc.args else exc, file=sys.stderr)
            return 2
        if definition not in filtres:
            filtres = [definition] + filtres
        index_initial = filtres.index(definition)

    camera: io_utils.Camera | None = None
    image: np.ndarray | None = None

    if args.webcam is not None:
        try:
            camera = io_utils.Camera(index=args.webcam, largeur=args.largeur,
                                    hauteur=args.hauteur).open()
        except io_utils.CameraError as exc:
            print(f"erreur : {exc}", file=sys.stderr)
            return 1
        print(f"Webcam {args.webcam} ouverte, résolution {camera.resolution}")
    else:
        chemin = args.image
        if not chemin:
            exemples = io_utils.sample_images()
            if not exemples:
                print(
                    "aucune image fournie et aucun exemple trouvé : "
                    "donner un chemin d'image ou utiliser --webcam",
                    file=sys.stderr,
                )
                return 2
            chemin = exemples[-1]
            print(f"Image d'exemple : {chemin}")
        try:
            image = io_utils.load_image(chemin)
        except io_utils.ImageLoadError as exc:
            print(f"erreur : {exc}", file=sys.stderr)
            return 1
        # Une photo de plusieurs mégapixels dépasse l'écran et ralentit la
        # boucle d'affichage sans rien apporter au réglage.
        cote = max(image.shape[:2])
        if args.max_cote > 0 and cote > args.max_cote:
            facteur = args.max_cote / cote
            image = cv2.resize(
                image,
                (int(image.shape[1] * facteur), int(image.shape[0] * facteur)),
                interpolation=cv2.INTER_AREA,
            )
            print(f"Image réduite à {image.shape[1]}×{image.shape[0]} px")

    prefixe = Pipeline.load(args.preset) if args.preset else None
    if prefixe is not None:
        print(f"Chaîne préalable « {prefixe.nom} » :\n{prefixe.summary()}")

    atelier = AtelierBureau(
        source_image=image,
        camera=camera,
        filtres=filtres,
        index_initial=index_initial,
        chaine_prefixe=prefixe,
        reference=reference,
        dossier_sortie=Path(args.sorties) if args.sorties else None,
    )
    try:
        return atelier.executer()
    except KeyboardInterrupt:
        return 130
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
