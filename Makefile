PYTHON ?= python3.12
VENV   := .venv
BIN    := $(VENV)/bin
# Run from src/ directly: on some macOS setups Python skips the editable-install .pth file
# because the OS marks files inside .venv as hidden.
RUN    := PYTHONPATH=src $(BIN)/python -m causeway.cli

.PHONY: help setup lock data analysis env test lint format clean

help:
	@echo "setup   create .venv with Python 3.12 and install the pinned environment"
	@echo "data    download (checksum-verified), validate and process the dataset"
	@echo "analysis run the pre-specified analyses (Part I experiment; Part II benchmark)"
	@echo "test    run the test suite"
	@echo "lint    run ruff checks"
	@echo "format  apply ruff formatting fixes"
	@echo "env     print Python and dependency versions"
	@echo "lock    regenerate requirements.txt from the current environment"
	@echo "clean   remove derived data and reports (raw data is kept)"

$(BIN)/python:
	$(PYTHON) -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip

setup: $(BIN)/python
	$(BIN)/python -m pip install -r requirements.txt
	$(BIN)/python -m pip install --no-deps -e .

lock: $(BIN)/python
	@{ echo "# Fully pinned environment for Python 3.12 (generated with: make lock)."; \
	   echo "# Direct dependencies and their constraints are declared in pyproject.toml."; \
	   $(BIN)/python -m pip freeze --exclude-editable | grep -v '^causeway'; } > requirements.txt

data:
	$(RUN) data

analysis:
	$(RUN) experiment
	$(RUN) benchmark
	$(RUN) propensity
	$(RUN) estimation
	$(RUN) compare
	$(RUN) heterogeneity
	$(RUN) robustness
	$(RUN) findings

env:
	$(RUN) env

test:
	$(BIN)/python -m pytest

lint:
	$(BIN)/ruff check src tests scripts
	$(BIN)/ruff format --check src tests scripts

format:
	$(BIN)/ruff check --fix src tests scripts
	$(BIN)/ruff format src tests scripts

clean:
	find data/interim data/processed reports/validation reports/results reports/figures -type f ! -name .gitkeep -delete
