"""Bureau (M3) : boucle webcam de l'application.

Le module ne touche à la caméra que dans ``creer_capture`` / ``reel`` :
tout le reste (état, pipeline, rendu) est testable sans matériel. La boucle
accepte une capture **injectée** (fake en test) et un mode ``afficher=False``
sans fenêtre, pour la validation automatisée.

Touches pendant la session :

* ``Échap`` / ``q`` — quitter
* ``c`` — lancer/arrêter l'étalonnage « tenir une pièce » (échelle de session)
* ``d`` — afficher le masque de profilage
* ``r`` — remettre le comptage à zéro
"""

from __future__ import annotations

import cv2

from ..config import CONFIG_DEFAUT, Config
from ..entites import ResultatImage
from ..etalonnage import ObservateurEtalonnage
from ..etat import EtatApplication
from ..rendu import dessiner_hud
from ..suivi import Suiveur
from ..traitement import traiter_image

__all__ = ["Bureau", "creer_capture", "ETIQUETTE_FENETRE"]

ETIQUETTE_FENETRE = "coin-counter — somme des dirhams marocains"
#: nombre de frames consécutivement fautives au-delà duquel on abandonne
#: (une erreur isolée est un accident, cinquante sont une panne)
ERREURS_MAX = 50


def creer_capture(indice: int = 0) -> cv2.VideoCapture:
    """Ouvre la caméra (0 par défaut) et vérifie qu'elle répond."""
    capture = cv2.VideoCapture(indice)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"caméra d'indice {indice} introuvable ou occupée"
        )
    return capture


class Bureau:
    """Orchestrateur : expansion et mue de l'état à chaque frame."""

    def __init__(
        self,
        config: Config = CONFIG_DEFAUT,
        etat: EtatApplication | None = None,
        capture: cv2.VideoCapture | None = None,
        afficher: bool = True,
    ) -> None:
        self.config = config
        if etat is None:
            etat = EtatApplication(
                config=config,
                suiveur=Suiveur(config),
                observateur=ObservateurEtalonnage(),
            )
        self.etat = etat
        self.capture = capture
        self.afficher = bool(afficher)
        self._continuer = True
        self._erreurs = 0
        #: dernière frame brute lue (pour l'export HUD en CLI)
        self.derniere_frame: cv2.typing.MatLike | None = None

    # ------------------------------------------------------------------

    def reel(self, limite: int | None = None) -> ResultatImage | None:
        """Boucle principale ; ``limite`` borne le nombre de frames (debug,
        pas d'ouverture de fenêtre lorsque ``afficher=False``)."""
        traite = 0
        dernier: ResultatImage | None = None
        try:
            while self._continuer:
                if limite is not None and traite >= limite:
                    break
                if self.capture is None:
                    break
                ok, frame = self.capture.read()
                if not ok or frame is None:
                    break
                self.derniere_frame = frame
                try:
                    dernier = traiter_image(frame, self.etat)
                    if self.afficher:
                        hud = dessiner_hud(frame, dernier, self.etat)
                        cv2.imshow(ETIQUETTE_FENETRE, hud)
                except Exception as err:  # noqa: BLE001
                    # Une frame pathologique ne doit pas emporter la session
                    # entière : le total accumulé serait perdu. On la saute en
                    # le signalant, et la boucle continue.
                    self.etat.messages.append(f"frame ignorée : {err}")
                    self._erreurs += 1
                    if self._erreurs > ERREURS_MAX:
                        raise
                if self.afficher:
                    self._gerer_touche(cv2.waitKey(1) & 0xFF)
                traite += 1
            return dernier
        finally:
            # La capture est libérée ici ET par l'appelant : `release()` est
            # idempotent, mais un Bureau reste à usage unique — un second
            # `reel()` rendrait None en silence.
            if self.capture is not None and hasattr(self.capture, "release"):
                self.capture.release()
            if self.afficher:
                cv2.destroyAllWindows()

    def _gerer_touche(self, touche: int) -> None:
        if touche in (27, ord("q"), ord("Q")):  # Échap / q
            self._continuer = False
        elif touche in (ord("c"), ord("C")):
            self.etat.mode_etalonnage = not self.etat.mode_etalonnage
            if self.etat.observateur is not None:
                self.etat.observateur.reinitialiser()
            self.etat.messages.append(
                "étalonnage "
                + ("en cours" if self.etat.mode_etalonnage else "annulé")
            )
        elif touche in (ord("d"), ord("D")):
            self.etat.afficher_masque = not self.etat.afficher_masque
        elif touche in (ord("r"), ord("R")):
            self.etat.reinitialiser()
            self.etat.messages.append("comptage remis à zéro")
