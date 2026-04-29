# =============================================================================
# RL IoT Defense System — Makefile
#
# Single entrypoint for routine developer and reproducibility tasks.
# Run `make help` to see the canonical list.
# =============================================================================

SHELL := /bin/bash

# Configurable variables (override on the CLI: `make train-rl ALGO=ppo SEED=1`)
PYTHON   ?= python
CONFIG   ?= config.yml
ALGO     ?= ppo
SEED     ?= 0
TIMESTEPS?= 500000
EPOCHS   ?= 100
DATA     ?= data/processed/ciciot2023
GEN_DIR  ?= artifacts/generator
RUNS_DIR ?= runs

.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
.PHONY: help
help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
	     /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } \
	     /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)

##@ Setup
.PHONY: install
install:  ## Install Python dependencies.
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

.PHONY: install-dev
install-dev: install  ## Install dev dependencies (pre-commit, lint).
	$(PYTHON) -m pip install pre-commit ruff black isort
	pre-commit install || true

##@ Quality
.PHONY: lint
lint:  ## Run ruff + black --check.
	ruff check src tests scripts main.py
	black --check src tests scripts main.py

.PHONY: format
format:  ## Auto-format with black + ruff --fix + isort.
	black src tests scripts main.py
	ruff check --fix src tests scripts main.py
	isort src tests scripts main.py

.PHONY: test
test:  ## Run pytest with quiet output.
	pytest -q

.PHONY: test-cov
test-cov:  ## Run pytest with coverage.
	pytest --cov=src --cov-report=term-missing --cov-report=html

##@ Pipeline (per-stage)
.PHONY: process-data
process-data:  ## Process raw CICIoT2023 -> data/processed/ciciot2023/.
	$(PYTHON) main.py --mode process-data --config $(CONFIG) --data-path $(DATA)

.PHONY: build-split-indices
build-split-indices:  ## Phase 1: build immutable train/val/test/OOD indices + hash manifest.
	$(PYTHON) -m scripts.data.build_split_indices \
	    --processed-dir $(DATA) --seed $(SEED)

.PHONY: plot-dataset
plot-dataset:  ## Phase 1: regenerate dataset overview figures (F0).
	$(PYTHON) -m scripts.data.plot_dataset_overview \
	    --processed-dir $(DATA) --out-dir docs/results/01_dataset

.PHONY: derive-stages
derive-stages:  ## Phase 4 prep: build stages.npy + manifest from state_indices.json.
	$(PYTHON) -m scripts.data.derive_stage_labels --data-path $(DATA)

.PHONY: phase-1
phase-1: build-split-indices plot-dataset  ## Run all Phase-1 deliverables.

.PHONY: phase-2
phase-2:  ## Phase 2: train LSTM Red Team and emit F1+F2 (~80 s on CPU).
	$(PYTHON) -m scripts.red_team.train_lstm \
	    --processed-dir $(DATA) --seed $(SEED)

.PHONY: phase-4
phase-4: derive-stages  ## Phase 4: train detector + RF + CNN1D, emit F11 (~3-5 min).
	$(PYTHON) -m scripts.detector.train_detector \
	    --processed-dir $(DATA) --seed $(SEED)

.PHONY: train-generator
train-generator:  ## Train the LSTM Red Team generator.
	$(PYTHON) main.py --mode train-generator --config $(CONFIG) \
	    --data-path $(DATA) --generator-path $(GEN_DIR) \
	    --generator-epochs $(EPOCHS)

.PHONY: train-rl
train-rl:  ## Train a single RL agent (override ALGO=ppo|dqn|a2c).
	$(PYTHON) main.py --mode train-rl --config $(CONFIG) \
	    --algorithm $(ALGO) --timesteps $(TIMESTEPS) \
	    --generator-path $(GEN_DIR) --data-path $(DATA)

.PHONY: train-all-rl
train-all-rl:  ## Train DQN, PPO, A2C sequentially (single seed).
	$(PYTHON) main.py --mode train-all-rl --config $(CONFIG) \
	    --timesteps $(TIMESTEPS) \
	    --generator-path $(GEN_DIR) --data-path $(DATA)

.PHONY: evaluate
evaluate:  ## Evaluate trained agent(s) -> results/benchmark/.
	$(PYTHON) main.py --mode evaluate --config $(CONFIG) \
	    --algorithms dqn ppo a2c --eval-episodes 100

##@ Reproducibility
.PHONY: reproduce-thesis
reproduce-thesis:  ## End-to-end thesis reproduction (data -> red -> blue -> bench).
	@echo ">>> 1/4 process-data";   $(MAKE) process-data
	@echo ">>> 2/4 train-generator"; $(MAKE) train-generator
	@echo ">>> 3/4 train-all-rl";    $(MAKE) train-all-rl
	@echo ">>> 4/4 evaluate";        $(MAKE) evaluate
	@echo "Done. Figures in docs/results/, raw data in $(RUNS_DIR)/."

##@ Maintenance
.PHONY: clean
clean:  ## Remove caches (safe).
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d -name .ruff_cache -prune -exec rm -rf {} +
	rm -rf .coverage htmlcov build dist *.egg-info

.PHONY: clean-runs
clean-runs:  ## Remove training artifacts and benchmarks (DESTRUCTIVE).
	@echo "This will delete artifacts/, runs/, results/, mlruns/."
	@read -p "Continue? [y/N] " ans; [ "$$ans" = "y" ] || exit 1
	rm -rf artifacts runs results mlruns

.PHONY: mlflow-ui
mlflow-ui:  ## Start MLflow UI on http://localhost:5000.
	mlflow ui --backend-store-uri mlruns
