# Raccourcis de développement de coin-counter.
#
#   make venv      vérifie (ou crée) l'environnement Python
#   make check     lint + suite de tests  ← porte de qualité principale
#   make test      exécute la suite de tests (hors marqueur perf)
#   make lint      vérifie le style avec ruff
#   make perf      exécute les tests de performance (-m perf)
#   make bench     produit le tableau des temps d'exécution
#   make autotest  autotest du pipeline (aucune caméra, vérité connue)
#   make webcam    lance le comptage en direct sur la webcam
#   make image     analyse une image statique (argument IMAGE=...)
#   make capture   capture une image d'étalonnage (sorties/capture.png)
#   make propre    nettoie les caches et les sorties

PYTHON ?= python3

# Environnement : un venv LOCAL au projet, créé depuis requirements.lock.
#
# Le Makefile visait auparavant ../cv-app/.venv en priorité — un dossier
# frère absent de tout extrait autonome de coin-counter, et absent du dépôt
# lui-même. Toutes les cibles échouaient alors sur un interpréteur
# introuvable. Le projet se suffit désormais à lui-même ; `make venv` crée
# .venv si besoin.
VENV := .venv
BIN := $(VENV)/bin
PY  := $(BIN)/python

.PHONY: venv check test lint perf bench autotest webcam image capture propre

venv:
	@if [ -x "$(PY)" ]; then \
		echo "Environnement : $(VENV)"; \
	else \
		echo "Environnement absent, création d'un venv local..."; \
		$(PYTHON) -m venv $(VENV); \
		$(BIN)/python -m pip install --upgrade pip; \
		$(BIN)/python -m pip install -r requirements.lock; \
	fi
	@$(PY) -c "import cv2, numpy, pytest" || \
		(echo "Dépendances manquantes : $(PY) -m pip install -r requirements.lock" && exit 1)

test: venv
	$(PY) -m pytest

lint: venv
	$(BIN)/ruff check .
	@echo "ruff : OK"

check: lint test

perf: venv
	$(PY) -m pytest -m perf

bench: venv
	$(PY) bench.py

webcam: venv
	$(PY) -m pieces.cli --camera 0

autotest: venv
	$(PY) -m pieces.cli --sans-fenetre

image: venv
	@test -n "$(IMAGE)" || (echo "usage : make image IMAGE=chemin/vers/image.jpg" && exit 1)
	$(PY) -m pieces.cli --image $(IMAGE)

capture: venv
	$(PY) -m pieces.cli --camera 0 --limite 1 --sortie sorties/capture.png

propre:
	rm -rf .pytest_cache .ruff_cache sorties
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
