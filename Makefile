PYTHON ?= python3.11
VENV   := .venv
BIN    := $(VENV)/bin

.PHONY: setup data inference strip-notebooks

## Create .venv, install common/ (editable) + dev tools, register the nbstripout git filter
setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"
	$(BIN)/nbstripout --install

## Download iWildCam v2.0 (~12 GB) and the WILDS ERM checkpoint
data:
	PYTHON=$(BIN)/python bash scripts/download_data.sh

## Write outputs/predictions.parquet
inference:
	$(BIN)/python scripts/run_inference.py

## Strip outputs from all notebooks in the working tree
strip-notebooks:
	find . -name "*.ipynb" -not -path "./$(VENV)/*" -not -path "*/.ipynb_checkpoints/*" -exec $(BIN)/nbstripout {} +
