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

# -----------------------------------------------------------------------------
# Phase 5: RL Blue Team — DQN/PPO/A2C × 5 seeds + F3/F4/T1
# -----------------------------------------------------------------------------
PHASE5_RUNS_ROOT ?= runs/phase5
PHASE5_TIMESTEPS ?= 250000
PHASE5_SEEDS ?= 0 1 2 3 4
PHASE5_ALGOS ?= dqn ppo a2c
PHASE5_PARALLEL ?= 1

.PHONY: phase-5-smoke
phase-5-smoke:  ## Phase 5 smoke: PPO seed 0 only, 5K timesteps (~20 s).
	$(PYTHON) -m scripts.blue_team.train_agent \
	    --algo ppo --seed 0 --smoke \
	    --out-dir runs/smoke/ppo_seed_0

.PHONY: phase-5-sweep
phase-5-sweep:  ## Phase 5: train DQN/PPO/A2C × 5 seeds (~3-7 h CPU).
	$(PYTHON) -m scripts.blue_team.run_phase5 \
	    --algos $(PHASE5_ALGOS) --seeds $(PHASE5_SEEDS) \
	    --total-timesteps $(PHASE5_TIMESTEPS) \
	    --out-root $(PHASE5_RUNS_ROOT) \
	    --parallel $(PHASE5_PARALLEL) \
	    --continue-on-failure

.PHONY: phase-5-figures
phase-5-figures:  ## Phase 5: render F3, F4, T1 from runs/phase5/.
	$(PYTHON) -m scripts.blue_team.plot_learning_curves \
	    --runs-root $(PHASE5_RUNS_ROOT) \
	    --out-dir docs/results/05_blue_team
	$(PYTHON) -m scripts.blue_team.plot_action_dist \
	    --runs-root $(PHASE5_RUNS_ROOT) \
	    --out-dir docs/results/05_blue_team
	$(PYTHON) -m scripts.blue_team.dump_hparams \
	    --runs-root $(PHASE5_RUNS_ROOT) \
	    --out-dir docs/results/05_blue_team

.PHONY: phase-5-gates
phase-5-gates:  ## Phase 5: evaluate G5.2-G5.7 against runs/phase5/.
	$(PYTHON) -m scripts.blue_team.evaluate_gates \
	    --runs-root $(PHASE5_RUNS_ROOT) \
	    --out-dir docs/results/05_blue_team

.PHONY: phase-5
phase-5: phase-5-sweep phase-5-figures phase-5-gates  ## Phase 5: full sweep + figures + gate scoreboard.

# -----------------------------------------------------------------------------
# Phase 6: RL Algorithm Benchmark — F5 + F6 + F7 + F8 from frozen Phase-5 ckpts
# -----------------------------------------------------------------------------
PHASE6_RUNS_ROOT     ?= runs/phase6
PHASE5_RUNS_ROOT     ?= runs/phase5
PHASE6_OUT_DIR       ?= docs/results/06_benchmark
PHASE6_N_EPISODES    ?= 30
PHASE6_N_DET_EPISODES ?= 150
PHASE6_RF_PATH       ?= artifacts/detector/random_forest.joblib

.PHONY: phase-6-smoke
phase-6-smoke:  ## Phase 6 smoke: 1 algo × 1 seed × 2 ep + 2 ep / baseline (~20 s CPU).
	$(PYTHON) -m scripts.benchmark.run_test_eval \
	    --smoke --out-root $(PHASE6_RUNS_ROOT)

.PHONY: phase-6-eval
phase-6-eval:  ## Phase 6: roll Phase-5 ckpts + 5 baselines on test_balanced (~10 min CPU).
	$(PYTHON) -m scripts.benchmark.run_test_eval \
	    --algos $(PHASE5_ALGOS) --seeds $(PHASE5_SEEDS) \
	    --n-episodes $(PHASE6_N_EPISODES) \
	    --n-deterministic-episodes $(PHASE6_N_DET_EPISODES) \
	    --phase5-runs-root $(PHASE5_RUNS_ROOT) \
	    --out-root $(PHASE6_RUNS_ROOT) \
	    --rf-path $(PHASE6_RF_PATH)

.PHONY: phase-6-figures
phase-6-figures:  ## Phase 6: render F5, F6, F7, F8 from runs/phase6/.
	$(PYTHON) -m scripts.benchmark.build_summary_table \
	    --runs-root $(PHASE6_RUNS_ROOT) \
	    --out-dir $(PHASE6_OUT_DIR)
	$(PYTHON) -m scripts.benchmark.plot_stage_action_cm \
	    --runs-root $(PHASE6_RUNS_ROOT) \
	    --out-dir $(PHASE6_OUT_DIR)
	$(PYTHON) -m scripts.benchmark.plot_overhead \
	    --runs-root $(PHASE6_RUNS_ROOT) \
	    --phase5-runs-root $(PHASE5_RUNS_ROOT) \
	    --out-dir $(PHASE6_OUT_DIR)
	$(PYTHON) -m scripts.benchmark.plot_baselines \
	    --runs-root $(PHASE6_RUNS_ROOT) \
	    --out-dir $(PHASE6_OUT_DIR)

.PHONY: phase-6
phase-6: phase-6-eval phase-6-figures  ## Phase 6: full eval sweep + F5/F6/F7/F8 figures.

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
