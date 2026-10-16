PYTHON ?= python3.14
VENV   := .venv

# Use the activated conda env `ai-and-vis` (see environment.yml) if present, else .venv
ifeq ($(CONDA_DEFAULT_ENV),ai-and-vis)
BIN := $(CONDA_PREFIX)/bin
else
BIN := $(VENV)/bin
endif

.PHONY: setup data inference baselines strip-notebooks

## Create .venv, install common/ (editable) + dev tools, register the nbstripout git filter
setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[analysis,dev]" -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu132
	$(BIN)/nbstripout --install

## Download iWildCam v2.0 (~12 GB) and the WILDS ERM checkpoint
data:
	PYTHON=$(BIN)/python bash scripts/download_data.sh

## Write outputs/predictions.parquet
inference:
	$(BIN)/python scripts/run_inference.py

## Write outputs/baselines.csv: WILDS fine-tuned model vs. ImageNet linear probe (needs `make inference`)
baselines:
	$(BIN)/python scripts/evaluate_baselines.py

## Strip outputs from all notebooks in the working tree
strip-notebooks:
	find . -name "*.ipynb" -not -path "./$(VENV)/*" -not -path "*/.ipynb_checkpoints/*" -exec $(BIN)/nbstripout {} +
