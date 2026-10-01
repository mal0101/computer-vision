# Raccourcis de développement.
#
#   make venv      crée l'environnement et installe les dépendances
#   make web       lance l'interface web
#   make bureau    lance l'interface de bureau sur une image d'exemple
#   make webcam    lance l'interface de bureau sur la webcam
#   make test      exécute la suite de tests
#   make demo      produit des images de démonstration dans sorties/

PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin

.PHONY: venv web bureau webcam test lister demo propre

venv:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -r requirements.txt
	@echo "Environnement prêt : source $(BIN)/activate"

web:
	$(BIN)/streamlit run ui/streamlit_app.py

bureau:
	$(BIN)/python ui/desktop.py

webcam:
	$(BIN)/python ui/desktop.py --webcam

test:
	$(BIN)/python -m pytest

lister:
	$(BIN)/python -m cvlab.cli lister -v

demo:
	$(BIN)/python outils/demo.py

propre:
	rm -rf .pytest_cache sorties
	find . -path ./$(VENV) -prune -o -name __pycache__ -type d -exec rm -rf {} +
