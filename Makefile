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
	     /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2 } \
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

##@ Pipeline (per-step)
.PHONY: process-data
process-data:  ## Process raw CICIoT2023 -> data/processed/ciciot2023/.
	$(PYTHON) main.py --mode process-data --config $(CONFIG) --data-path $(DATA)

.PHONY: build-split-indices
build-split-indices:  ## Dataset prep: build immutable train/val/test/OOD indices + hash manifest.
	$(PYTHON) -m scripts.data.build_split_indices \
	    --processed-dir $(DATA) --seed $(SEED)

.PHONY: plot-dataset
plot-dataset:  ## Dataset prep: regenerate dataset overview figures (F0).
	$(PYTHON) -m scripts.data.plot_dataset_overview \
	    --processed-dir $(DATA) --out-dir docs/results/01_dataset

.PHONY: derive-stages
derive-stages:  ## Detector prep: build stages.npy + manifest from state_indices.json.
	$(PYTHON) -m scripts.data.derive_stage_labels --data-path $(DATA)

.PHONY: dataset
dataset: build-split-indices plot-dataset  ## Run all dataset-preparation deliverables.

.PHONY: red-team
red-team:  ## Red-team: train LSTM episode generator and emit F1+F2 (~80 s on CPU).
	$(PYTHON) -m scripts.red_team.train_lstm \
	    --processed-dir $(DATA) --seed $(SEED)

.PHONY: detector
detector: derive-stages  ## Detector: train MLP + RF + CNN1D, emit F11 (~3-5 min).
	$(PYTHON) -m scripts.detector.train_detector \
	    --processed-dir $(DATA) --seed $(SEED)

# -----------------------------------------------------------------------------
# Blue Team — DQN/PPO/A2C × 5 seeds + F3/F4/T1
# -----------------------------------------------------------------------------
BLUE_TEAM_RUNS_ROOT ?= runs/blue_team
BLUE_TEAM_TIMESTEPS ?= 250000
BLUE_TEAM_SEEDS     ?= 0 1 2 3 4
BLUE_TEAM_ALGOS     ?= dqn ppo a2c
BLUE_TEAM_PARALLEL  ?= 1

.PHONY: blue-team-smoke
blue-team-smoke:  ## Blue-team smoke: PPO seed 0 only, 5K timesteps (~20 s).
	$(PYTHON) -m scripts.blue_team.train_agent \
	    --algo ppo --seed 0 --smoke \
	    --out-dir runs/smoke/ppo_seed_0

.PHONY: blue-team-sweep
blue-team-sweep:  ## Blue-team: train DQN/PPO/A2C × 5 seeds (~3-7 h CPU).
	$(PYTHON) -m scripts.blue_team.run_sweep \
	    --algos $(BLUE_TEAM_ALGOS) --seeds $(BLUE_TEAM_SEEDS) \
	    --total-timesteps $(BLUE_TEAM_TIMESTEPS) \
	    --out-root $(BLUE_TEAM_RUNS_ROOT) \
	    --parallel $(BLUE_TEAM_PARALLEL) \
	    --continue-on-failure

.PHONY: blue-team-figures
blue-team-figures:  ## Blue-team: render F3, F4, T1 from runs/blue_team/.
	$(PYTHON) -m scripts.blue_team.plot_learning_curves \
	    --runs-root $(BLUE_TEAM_RUNS_ROOT) \
	    --out-dir docs/results/05_blue_team
	$(PYTHON) -m scripts.blue_team.plot_action_dist \
	    --runs-root $(BLUE_TEAM_RUNS_ROOT) \
	    --out-dir docs/results/05_blue_team
	$(PYTHON) -m scripts.blue_team.dump_hparams \
	    --runs-root $(BLUE_TEAM_RUNS_ROOT) \
	    --out-dir docs/results/05_blue_team

.PHONY: blue-team-gates
blue-team-gates:  ## Blue-team: evaluate G5.2-G5.7 against runs/blue_team/.
	$(PYTHON) -m scripts.blue_team.evaluate_gates \
	    --runs-root $(BLUE_TEAM_RUNS_ROOT) \
	    --out-dir docs/results/05_blue_team

.PHONY: blue-team
blue-team: blue-team-sweep blue-team-figures blue-team-gates  ## Blue-team: full sweep + figures + gate scoreboard.

# -----------------------------------------------------------------------------
# Benchmark — F5 + F6 + F7 + F8 from frozen blue-team checkpoints
# -----------------------------------------------------------------------------
BENCHMARK_RUNS_ROOT      ?= runs/benchmark
BENCHMARK_OUT_DIR        ?= docs/results/06_benchmark
BENCHMARK_N_EPISODES     ?= 30
BENCHMARK_N_DET_EPISODES ?= 150
BENCHMARK_RF_PATH        ?= artifacts/detector/random_forest.joblib

.PHONY: benchmark-smoke
benchmark-smoke:  ## Benchmark smoke: 1 algo × 1 seed × 2 ep + 2 ep / baseline (~20 s CPU).
	$(PYTHON) -m scripts.benchmark.run_test_eval \
	    --smoke --out-root $(BENCHMARK_RUNS_ROOT)

.PHONY: benchmark-eval
benchmark-eval:  ## Benchmark: roll blue-team checkpoints + 5 baselines on test_balanced (~10 min CPU).
	$(PYTHON) -m scripts.benchmark.run_test_eval \
	    --algos $(BLUE_TEAM_ALGOS) --seeds $(BLUE_TEAM_SEEDS) \
	    --n-episodes $(BENCHMARK_N_EPISODES) \
	    --n-deterministic-episodes $(BENCHMARK_N_DET_EPISODES) \
	    --phase5-runs-root $(BLUE_TEAM_RUNS_ROOT) \
	    --out-root $(BENCHMARK_RUNS_ROOT) \
	    --rf-path $(BENCHMARK_RF_PATH)

.PHONY: benchmark-figures
benchmark-figures:  ## Benchmark: render F5, F6, F7, F8 from runs/benchmark/.
	$(PYTHON) -m scripts.benchmark.build_summary_table \
	    --runs-root $(BENCHMARK_RUNS_ROOT) \
	    --out-dir $(BENCHMARK_OUT_DIR)
	$(PYTHON) -m scripts.benchmark.plot_stage_action_cm \
	    --runs-root $(BENCHMARK_RUNS_ROOT) \
	    --out-dir $(BENCHMARK_OUT_DIR)
	$(PYTHON) -m scripts.benchmark.plot_overhead \
	    --runs-root $(BENCHMARK_RUNS_ROOT) \
	    --phase5-runs-root $(BLUE_TEAM_RUNS_ROOT) \
	    --out-dir $(BENCHMARK_OUT_DIR)
	$(PYTHON) -m scripts.benchmark.plot_baselines \
	    --runs-root $(BENCHMARK_RUNS_ROOT) \
	    --out-dir $(BENCHMARK_OUT_DIR)

.PHONY: benchmark
benchmark: benchmark-eval benchmark-figures  ## Benchmark: full eval sweep + F5/F6/F7/F8 figures.

##@ Ablation — reward-component sweep + OOD-class robustness (PLAN: docs/results/07_ablation/PLAN.md)
ABLATION_OUT_DIR         ?= docs/results/07_ablation
ABLATION_OOD_RUNS_ROOT   ?= runs/ablation/ood
ABLATION_OOD_CLASSES     ?= DDoS-HTTP_Flood Mirai-udpplain VulnerabilityScan XSS
ABLATION_OOD_POLICIES    ?= recommended_action rf_acting dqn ppo a2c random always_observe always_block
ABLATION_OOD_N_EPISODES  ?= 30
ABLATION_OOD_N_DET_EPISODES ?= 150

.PHONY: ablation-ood-smoke
ablation-ood-smoke:  ## Ablation F15 smoke: 1 OOD class × 2 policies × 1 seed × 2 ep (~10 s).
	$(PYTHON) -m scripts.ablation.run_ood_eval \
	    --smoke --out-root $(ABLATION_OOD_RUNS_ROOT)

.PHONY: ablation-ood-eval
ablation-ood-eval:  ## Ablation F15: 4 OOD classes × 8 policies eval (~1 h CPU).
	$(PYTHON) -m scripts.ablation.run_ood_eval \
	    --ood-classes $(ABLATION_OOD_CLASSES) \
	    --policies $(ABLATION_OOD_POLICIES) \
	    --seeds $(BLUE_TEAM_SEEDS) \
	    --n-episodes $(ABLATION_OOD_N_EPISODES) \
	    --n-deterministic-episodes $(ABLATION_OOD_N_DET_EPISODES) \
	    --phase5-runs $(BLUE_TEAM_RUNS_ROOT) \
	    --out-root $(ABLATION_OOD_RUNS_ROOT) \
	    --rf-path $(BENCHMARK_RF_PATH)

.PHONY: ablation-ood-figure
ablation-ood-figure:  ## Ablation: render F15 from runs/ablation/ood/.
	$(PYTHON) -m scripts.ablation.plot_ood_robustness \
	    --runs-root $(ABLATION_OOD_RUNS_ROOT) \
	    --out-dir $(ABLATION_OUT_DIR) \
	    --ood-classes $(ABLATION_OOD_CLASSES) \
	    --policies $(ABLATION_OOD_POLICIES)

.PHONY: ablation-ood
ablation-ood: ablation-ood-eval ablation-ood-figure  ## Ablation F15: full OOD eval + figure.

# F9 — Reward-component ablation sweep
ABLATION_REWARD_RUNS_ROOT ?= runs/ablation/reward_sweep
ABLATION_REWARD_TIMESTEPS ?= 250000
ABLATION_REWARD_ALGO      ?= ppo

.PHONY: ablation-reward-smoke
ablation-reward-smoke:  ## Ablation F9 smoke: 1 cell × 1 seed × 5K (~30 s).
	$(PYTHON) -m scripts.ablation.run_reward_sweep \
	    --smoke --algo $(ABLATION_REWARD_ALGO) \
	    --out-root $(ABLATION_REWARD_RUNS_ROOT)

.PHONY: ablation-reward-sweep
ablation-reward-sweep:  ## Ablation F9: PPO × 5 seeds × 12 cells (~6 h CPU).
	$(PYTHON) -m scripts.ablation.run_reward_sweep \
	    --algo $(ABLATION_REWARD_ALGO) \
	    --seeds $(BLUE_TEAM_SEEDS) \
	    --total-timesteps $(ABLATION_REWARD_TIMESTEPS) \
	    --out-root $(ABLATION_REWARD_RUNS_ROOT) \
	    --continue-on-failure

.PHONY: ablation-reward-figure
ablation-reward-figure:  ## Ablation: render F9 from runs/ablation/reward_sweep/.
	$(PYTHON) -m scripts.ablation.plot_reward_ablation \
	    --runs-root $(ABLATION_REWARD_RUNS_ROOT) \
	    --out-dir $(ABLATION_OUT_DIR)

.PHONY: ablation-reward
ablation-reward: ablation-reward-sweep ablation-reward-figure  ## Ablation F9: full sweep + figure.

# F10 — Attack-aggressiveness sweep (PPO + oracle rule × 6 p values × 5 seeds)
ABLATION_AGGR_RUNS_ROOT ?= runs/ablation/aggressiveness

.PHONY: ablation-aggressiveness-smoke
ablation-aggressiveness-smoke:  ## Ablation F10 smoke: 2 p values × 1 seed × 5K (~30 s).
	$(PYTHON) -m scripts.ablation.run_aggressiveness_sweep \
	    --smoke --out-root $(ABLATION_AGGR_RUNS_ROOT)

.PHONY: ablation-aggressiveness-sweep
ablation-aggressiveness-sweep:  ## Ablation F10: PPO × 6 p values × 5 seeds + oracle rule (~1.5 h CPU).
	$(PYTHON) -m scripts.ablation.run_aggressiveness_sweep \
	    --seeds $(BLUE_TEAM_SEEDS) \
	    --total-timesteps $(ABLATION_REWARD_TIMESTEPS) \
	    --out-root $(ABLATION_AGGR_RUNS_ROOT) \
	    --continue-on-failure

.PHONY: ablation-aggressiveness-figure
ablation-aggressiveness-figure:  ## Ablation: render F10 from runs/ablation/aggressiveness/.
	$(PYTHON) -m scripts.ablation.plot_aggressiveness \
	    --runs-root $(ABLATION_AGGR_RUNS_ROOT) \
	    --out-dir $(ABLATION_OUT_DIR)

.PHONY: ablation-aggressiveness
ablation-aggressiveness: ablation-aggressiveness-sweep ablation-aggressiveness-figure  ## Ablation F10: full sweep + figure.

# F12 — Security-vs-availability Pareto (plotter-only; reads F9 + F10 + benchmark)
.PHONY: ablation-pareto
ablation-pareto:  ## Ablation F12: render Pareto plot from F9 + F10 + benchmark outputs.
	$(PYTHON) -m scripts.ablation.plot_pareto \
	    --phase6-runs $(BENCHMARK_RUNS_ROOT) \
	    --phase7-f9-runs $(ABLATION_REWARD_RUNS_ROOT) \
	    --phase7-f10-runs $(ABLATION_AGGR_RUNS_ROOT) \
	    --out-dir $(ABLATION_OUT_DIR)

# Close (G7 scoreboard + RESULTS.md)
.PHONY: ablation-close
ablation-close:  ## Ablation: assemble G7 scoreboard + RESULTS.md skeleton.
	$(PYTHON) -m scripts.ablation.close_ablation \
	    --out-dir $(ABLATION_OUT_DIR)

# Top-level ablation chains
.PHONY: ablation-figures
ablation-figures: ablation-ood-figure ablation-reward-figure ablation-aggressiveness-figure ablation-pareto  ## Ablation: render F9/F10/F12/F15 from existing runs/ablation/.

.PHONY: ablation
ablation: ablation-ood ablation-reward ablation-aggressiveness ablation-pareto  ## Ablation: full F9 + F10 + F12 + F15 (~7.5 h CPU walk-away).

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

##@ Thesis (Docker)
THESIS_IMAGE ?= rl-iot-thesis
THESIS_TEX   ?= $(PWD)/tex

.PHONY: thesis-image
thesis-image:  ## Build the minimal Docker image for thesis compilation (one-off, ~3-4 min first run).
	docker build --tag $(THESIS_IMAGE) --file tex/Dockerfile tex/

.PHONY: thesis
thesis:  ## Compile tex/thesis.pdf via Docker (builds image if not present).
	@if ! docker image inspect $(THESIS_IMAGE) >/dev/null 2>&1; then \
	  echo "==> Image '$(THESIS_IMAGE)' not found — building first..."; \
	  $(MAKE) thesis-image; \
	fi
	docker run --rm --volume "$(THESIS_TEX)":/work $(THESIS_IMAGE)

.PHONY: thesis-rebuild
thesis-rebuild: thesis-image thesis  ## Force-rebuild Docker image, then compile thesis.

##@ Reproducibility
.PHONY: reproduce-thesis
reproduce-thesis:  ## End-to-end thesis reproduction (data -> red -> blue -> bench).
	@echo ">>> 1/4 process-data";   $(MAKE) process-data
	@echo ">>> 2/4 train-generator"; $(MAKE) train-generator
	@echo ">>> 3/4 train-all-rl";    $(MAKE) train-all-rl
	@echo ">>> 4/4 benchmark";       $(MAKE) benchmark
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
