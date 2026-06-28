# =============================================================================
# RL IoT Defense System — Makefile
#
# Single entrypoint for routine developer and reproducibility tasks.
# Run `make help` to see the canonical list.
# =============================================================================

SHELL := /bin/bash

# Configurable variables (override on the CLI: `make train-rl ALGO=ppo SEED=1`)
PYTHON   ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
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
	$(PYTHON) -m pytest -q

.PHONY: test-cov
test-cov:  ## Run pytest with coverage.
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing --cov-report=html

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
	    --processed-dir $(DATA) --out-dir docs/results/dataset

.PHONY: derive-stages
derive-stages:  ## Detector prep: build stages.npy + manifest from state_indices.json.
	$(PYTHON) -m scripts.data.derive_stage_labels --data-path $(DATA)

.PHONY: dataset
dataset: build-split-indices plot-dataset  ## Run all dataset-preparation deliverables.

.PHONY: detector
detector: derive-stages  ## Detector: train MLP + RF, emit F11 (~3-5 min).
	$(PYTHON) -m scripts.detector.train_detector \
	    --processed-dir $(DATA) --seed $(SEED)

.PHONY: detector-figure
detector-figure:  ## Detector: re-plot per_stage_recall (Fig 4.3) from F11_summary.json (no retrain).
	$(PYTHON) -m scripts.detector.plot_per_stage_recall \
	    --summary docs/results/stage-detector/F11_summary.json \
	    --out-dir docs/results/stage-detector

# -----------------------------------------------------------------------------
# Blue Team — DQN/PPO/A2C × 10 seeds + F3/F4/T1
# -----------------------------------------------------------------------------
BLUE_TEAM_RUNS_ROOT     ?= runs/blue_team
BLUE_TEAM_TIMESTEPS     ?= 1500000
BLUE_TEAM_SEEDS         ?= 0 1 2 3 4 5 6 7 8 9
BLUE_TEAM_ALGOS         ?= dqn ppo a2c
BLUE_TEAM_PARALLEL      ?= 1
BLUE_TEAM_IMPACT_TERM   ?= false  # Blue-Team Training primary contract: false (locked)
# JSON forwarded to train_agent --reward-overrides, e.g. {"aliasing_rate":0.2}
BLUE_TEAM_REWARD_OVERRIDES ?=

.PHONY: blue-team-smoke
blue-team-smoke:  ## Blue-team smoke: PPO seed 0 only, 5K timesteps (~20 s).
	$(PYTHON) -m scripts.blue_team.train_agent \
	    --algo ppo --seed 0 --smoke \
	    --impact-is-terminal $(BLUE_TEAM_IMPACT_TERM) \
	    $(if $(BLUE_TEAM_REWARD_OVERRIDES),--reward-overrides '$(BLUE_TEAM_REWARD_OVERRIDES)',) \
	    --out-dir runs/smoke/ppo_seed_0

.PHONY: blue-team-sweep
blue-team-sweep:  ## Blue-team: train DQN/PPO/A2C × 10 seeds (~3-7 h CPU).
	$(PYTHON) -m scripts.blue_team.run_sweep \
	    --algos $(BLUE_TEAM_ALGOS) --seeds $(BLUE_TEAM_SEEDS) \
	    --total-timesteps $(BLUE_TEAM_TIMESTEPS) \
	    --out-root $(BLUE_TEAM_RUNS_ROOT) \
	    --parallel $(BLUE_TEAM_PARALLEL) \
	    --impact-is-terminal $(BLUE_TEAM_IMPACT_TERM) \
	    $(if $(BLUE_TEAM_REWARD_OVERRIDES),--reward-overrides '$(BLUE_TEAM_REWARD_OVERRIDES)',) \
	    --continue-on-failure

.PHONY: blue-team-figures
blue-team-figures:  ## Blue-team: render F3, F4, T1 from runs/blue_team/.
	$(PYTHON) -m scripts.blue_team.plot_learning_curves \
	    --runs-root $(BLUE_TEAM_RUNS_ROOT) \
	    --out-dir docs/results/blue-team-training
	$(PYTHON) -m scripts.blue_team.plot_action_dist \
	    --runs-root $(BLUE_TEAM_RUNS_ROOT) \
	    --force-algo ppo \
	    --out-dir docs/results/blue-team-training
	$(PYTHON) -m scripts.blue_team.dump_hparams \
	    --runs-root $(BLUE_TEAM_RUNS_ROOT) \
	    --out-dir docs/results/blue-team-training

.PHONY: blue-team-gates
blue-team-gates:  ## Blue-team: evaluate G5.2-G5.7 against runs/blue_team/.
	$(PYTHON) -m scripts.blue_team.evaluate_gates \
	    --runs-root $(BLUE_TEAM_RUNS_ROOT) \
	    --out-dir docs/results/blue-team-training

.PHONY: blue-team
blue-team: blue-team-sweep blue-team-figures blue-team-gates  ## Blue-team: full sweep + figures + gate scoreboard.

# -----------------------------------------------------------------------------
# Benchmark — evaluation from frozen blue-team checkpoints
# -----------------------------------------------------------------------------
BENCHMARK_RUNS_ROOT      ?= runs/benchmark
BENCHMARK_OUT_DIR        ?= docs/results/benchmark
BENCHMARK_N_EPISODES     ?= 30
BENCHMARK_N_DET_EPISODES ?= 300
BENCHMARK_RF_PATH ?= artifacts/detector/random_forest.joblib
# Proximity-coupled escalation regime: no fixed attacker budget. Escalation
# pressure scales with proximity to IMPACT (sigma_min=0.4, lambda=stage/4).

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
	    --blue-team-runs-root $(BLUE_TEAM_RUNS_ROOT) \
	    --out-root $(BENCHMARK_RUNS_ROOT) \
	    --rf-path $(BENCHMARK_RF_PATH)

.PHONY: benchmark
benchmark: benchmark-eval  ## Benchmark: full eval sweep from frozen blue-team checkpoints.

##@ Ablation — reward-component sweep + OOD-class robustness (PLAN: docs/results/ablation/PLAN.md)
ABLATION_OUT_DIR         ?= docs/results/ablation
ABLATION_PARALLEL        ?= 1
ABLATION_OOD_RUNS_ROOT   ?= runs/ablation/ood
ABLATION_OOD_CLASSES     ?= VulnerabilityScan Recon-OSScan XSS SqlInjection Mirai-udpplain DNS_Spoofing DDoS-HTTP_Flood DoS-SYN_Flood DDoS-SlowLoris DDoS-ACK_Fragmentation
ABLATION_OOD_POLICIES    ?= recommended_action rf_acting dqn ppo a2c random always_observe always_block
ABLATION_OOD_N_EPISODES  ?= 30
ABLATION_OOD_N_DET_EPISODES ?= 300
# Commensurability: the OOD eval MUST run at the same operating point as the
# held-out benchmark, else OOD reward is on an incomparable (unbounded) axis.
ABLATION_OOD_REWARD_MODE ?= outcome
# Canonical current-contract blue-team checkpoints live under
# runs/redesign/alpha_<NN>/<algo>/seed_<n>/best_model.zip (NOT runs/blue_team,
# which is the legacy/pre-redesign path and is absent on fresh checkouts). The
# headline operating point is alpha=0.4, so OOD eval loads from alpha_04.
ABLATION_OOD_BLUE_TEAM_RUNS ?= runs/redesign/alpha_04
# Partial-observability operating point: these flags MUST be passed or the
# runner falls back to a fully-observable MDP (aliasing_rate=0, no session
# coherence) that mismatches the trained checkpoints' contract and trips the
# train/eval parity assertion. Mirrors the locked HeadlineAlpha=0.4 regime.
ABLATION_OOD_ALIASING_RATE ?= 0.4
ABLATION_OOD_PROXIMITY_MIN ?= 0.4
ABLATION_OOD_POMDP_FLAGS ?= --aliasing-rate $(ABLATION_OOD_ALIASING_RATE) \
	    --session-coherent --no-post-transition-leak \
	    --proximity-coupled --proximity-min-escalation $(ABLATION_OOD_PROXIMITY_MIN)

.PHONY: ablation-ood-smoke
ablation-ood-smoke:  ## Ablation F15 smoke: 1 OOD class × 2 policies × 1 seed × 2 ep (~10 s). Writes to throwaway dir, never canonical.
	$(PYTHON) -m scripts.ablation.run_ood_eval \
	    --smoke --out-root runs/ablation/_smoke/ood

.PHONY: ablation-ood-eval
ablation-ood-eval:  ## Ablation: zero-day OOD eval, 10 held-out classes × 8 policies × 10 seeds (~2 h CPU).
	$(PYTHON) -m scripts.ablation.run_ood_eval \
	    --ood-classes $(ABLATION_OOD_CLASSES) \
	    --policies $(ABLATION_OOD_POLICIES) \
	    --seeds $(BLUE_TEAM_SEEDS) \
	    --n-episodes $(ABLATION_OOD_N_EPISODES) \
	    --n-deterministic-episodes $(ABLATION_OOD_N_DET_EPISODES) \
	    --blue-team-runs $(ABLATION_OOD_BLUE_TEAM_RUNS) \
	    --out-root $(ABLATION_OOD_RUNS_ROOT) \
	    --reward-mode $(ABLATION_OOD_REWARD_MODE) \
	    $(ABLATION_OOD_POMDP_FLAGS) \
	    --rf-path $(BENCHMARK_RF_PATH)

.PHONY: ablation-ood-figure
ablation-ood-figure:  ## Ablation: render F15 from runs/ablation/ood/.
	$(PYTHON) -m scripts.ablation.plot_ood_robustness \
	    --runs-root $(ABLATION_OOD_RUNS_ROOT) \
	    --out-dir $(ABLATION_OUT_DIR) \
	    --ood-classes $(ABLATION_OOD_CLASSES) \
	    --policies $(ABLATION_OOD_POLICIES) \
	    --blue-team-sweep-manifest $(ABLATION_OOD_BLUE_TEAM_RUNS)/sweep_manifest.json

.PHONY: ablation-ood
ablation-ood: ablation-ood-eval ablation-ood-figure  ## Ablation F15: full OOD eval + figure.

# ------------------------------------------------------------------------------
# Coupled-vs-decoupled reward ablation (the reward-design control)
ABLATION_COUPLING_RUNS_ROOT ?= runs/ablation/reward_coupling
ABLATION_COUPLING_TIMESTEPS ?= 1000000

.PHONY: ablation-reward-coupling-smoke
ablation-reward-coupling-smoke:  ## Coupling smoke: coupled+outcome × 1 algo × 1 seed × 5K (~1 min).
	$(PYTHON) -m scripts.ablation.run_reward_coupling \
	    --smoke \
	    --out-root $(ABLATION_COUPLING_RUNS_ROOT)

.PHONY: ablation-reward-coupling-sweep
ablation-reward-coupling-sweep:  ## Coupling: {coupled,outcome} × 3 algos × 10 seeds + RF-Acting (~6 h CPU).
	$(PYTHON) -m scripts.ablation.run_reward_coupling \
	    --seeds $(BLUE_TEAM_SEEDS) \
	    --total-timesteps $(ABLATION_COUPLING_TIMESTEPS) \
	    --out-root $(ABLATION_COUPLING_RUNS_ROOT) \
	    --parallel $(ABLATION_PARALLEL) \
	    --continue-on-failure

.PHONY: ablation-reward-coupling-figure
ablation-reward-coupling-figure:  ## Coupling: render coupled-vs-outcome gap figure.
	$(PYTHON) -m scripts.ablation.plot_reward_coupling \
	    --out-root $(ABLATION_COUPLING_RUNS_ROOT) \
	    --seeds $(BLUE_TEAM_SEEDS) \
	    --out-dir $(ABLATION_OUT_DIR)

.PHONY: ablation-reward-coupling
ablation-reward-coupling: ablation-reward-coupling-sweep ablation-reward-coupling-figure  ## Coupling: full sweep + figure.

# F10 — Attack-aggressiveness sweep (PPO + oracle rule × 6 p values × 10 seeds)
ABLATION_AGGR_RUNS_ROOT ?= runs/ablation/aggressiveness
ABLATION_AGGR_TIMESTEPS ?= 250000

.PHONY: ablation-aggressiveness-smoke
ablation-aggressiveness-smoke:  ## Ablation F10 smoke: 2 p values × 1 seed × 5K (~30 s). Writes to throwaway dir, never canonical.
	$(PYTHON) -m scripts.ablation.run_aggressiveness_sweep \
	    --smoke --out-root runs/ablation/_smoke/aggressiveness

.PHONY: ablation-aggressiveness-sweep
ablation-aggressiveness-sweep:  ## Ablation F10: PPO × 6 p values × 10 seeds + oracle rule (~1.5 h CPU).
	$(PYTHON) -m scripts.ablation.run_aggressiveness_sweep \
	    --seeds $(BLUE_TEAM_SEEDS) \
	    --total-timesteps $(ABLATION_AGGR_TIMESTEPS) \
	    --out-root $(ABLATION_AGGR_RUNS_ROOT) \
	    --parallel $(ABLATION_PARALLEL) \
	    --continue-on-failure

.PHONY: ablation-aggressiveness-figure
ablation-aggressiveness-figure:  ## Ablation: render F10 from runs/ablation/aggressiveness/.
	$(PYTHON) -m scripts.ablation.plot_aggressiveness \
	    --runs-root $(ABLATION_AGGR_RUNS_ROOT) \
	    --out-dir $(ABLATION_OUT_DIR)

.PHONY: ablation-aggressiveness
ablation-aggressiveness: ablation-aggressiveness-sweep ablation-aggressiveness-figure  ## Ablation F10: full sweep + figure.

# F17 — Evasive-attacker sweep (on-contract outcome reward; locked overrides in runner)
ABLATION_EVASION_RUNS_ROOT ?= runs/ablation/evasion
ABLATION_EVASION_TIMESTEPS ?= 1500000

.PHONY: ablation-evasion-smoke
ablation-evasion-smoke:  ## Ablation F17 smoke: 2 evasion values × 1 seed × 5K (~30 s). Writes to throwaway dir, never canonical.
	$(PYTHON) -m scripts.ablation.run_evasion_sweep \
	    --smoke --out-root runs/ablation/_smoke/evasion

.PHONY: ablation-evasion-sweep
ablation-evasion-sweep:  ## Ablation F17: PPO × 4 evasion values × 10 seeds (~1 h CPU).
	$(PYTHON) -m scripts.ablation.run_evasion_sweep \
	    --seeds $(BLUE_TEAM_SEEDS) \
	    --total-timesteps $(ABLATION_EVASION_TIMESTEPS) \
	    --out-root $(ABLATION_EVASION_RUNS_ROOT) \
	    --parallel $(ABLATION_PARALLEL) \
	    --continue-on-failure

.PHONY: ablation-evasion-figure
ablation-evasion-figure:  ## Ablation: render F17 from runs/ablation/evasion/.
	$(PYTHON) -m scripts.ablation.plot_evasion_sweep \
	    --runs-root $(ABLATION_EVASION_RUNS_ROOT) \
	    --out-dir $(ABLATION_OUT_DIR)

.PHONY: ablation-evasion
ablation-evasion: ablation-evasion-sweep ablation-evasion-figure  ## Ablation F17: full sweep + figure.

# F12 — Security-vs-availability Pareto (plotter-only; reads F10 + benchmark)
.PHONY: ablation-pareto
ablation-pareto:  ## Ablation F12: render Pareto plot from F10 + benchmark outputs.
	$(PYTHON) -m scripts.ablation.plot_pareto \
	    --benchmark-runs $(BENCHMARK_RUNS_ROOT) \
	    --ablation-aggressiveness-runs $(ABLATION_AGGR_RUNS_ROOT) \
	    --out-dir $(ABLATION_OUT_DIR)

# Close (G7 scoreboard + RESULTS.md)
.PHONY: ablation-close
ablation-close:  ## Ablation: assemble G7 scoreboard + RESULTS.md skeleton.
	$(PYTHON) -m scripts.ablation.close_ablation \
	    --out-dir $(ABLATION_OUT_DIR)

# Top-level ablation chains
.PHONY: ablation-figures
ablation-figures: ablation-ood-figure ablation-aggressiveness-figure ablation-evasion-figure ablation-pareto  ## Ablation: render F10/F12/F15/F17 from existing runs/ablation/.

.PHONY: ablation
ablation: ablation-ood ablation-aggressiveness ablation-evasion ablation-pareto  ## Ablation: full F10 + F12 + F15 + F17 (~8.5 h CPU walk-away).

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

##@ Thesis (Podman, Docker fallback — FEEC CCPG 001-2015 / abnTeX2 template)
THESIS_IMAGE ?= rl-iot-thesis

.PHONY: thesis-image
thesis-image:  ## Build the minimal container image for thesis compilation (one-off, ~3-6 min first run).
	bash tex/build.sh --rebuild --draft

.PHONY: thesis
thesis:  ## Compile tex/main.pdf (full: pdflatex × 3 + bibtex) via container engine.
	$(MAKE) render-tables
	bash tex/build.sh

.PHONY: thesis-draft
thesis-draft:  ## Single fast pdflatex pass (no bibtex, no ToC fix-up).
	$(MAKE) render-tables
	bash tex/build.sh --draft

.PHONY: thesis-rebuild
thesis-rebuild:  ## Force-rebuild container image, then compile thesis.
	$(MAKE) render-tables
	bash tex/build.sh --rebuild

##@ Figure / Table Synchronisation (anti-drift)
.PHONY: export-figure-pdfs
export-figure-pdfs:  ## Wrap docs/results/**/F*.png into same-named F*.pdf (raster-in-PDF).
	$(PYTHON) scripts/thesis/export_pdfs.py

.PHONY: sync-figures
sync-figures: export-figure-pdfs  ## Export PDFs then copy them from docs/results/ → tex/figs/
	@echo "Syncing figures ..."
	@cp docs/results/stage-detector/per_stage_recall.pdf tex/figs/ 2>/dev/null || true
	@cp docs/results/blue-team-training/F3_*.pdf tex/figs/ 2>/dev/null || true
	@cp docs/results/blue-team-training/F4_*.pdf tex/figs/ 2>/dev/null || true
	@cp docs/results/ablation/F10_*.pdf tex/figs/ 2>/dev/null || true
	@cp docs/results/ablation/F12_*.pdf tex/figs/ 2>/dev/null || true
	@cp docs/results/ablation/Falpha_curve.pdf tex/figs/ 2>/dev/null || true
	@cp docs/results/ablation/F15_*.pdf tex/figs/ 2>/dev/null || true
	@cp docs/results/ablation/F15b_recall_vs_advantage.pdf tex/figs/ 2>/dev/null || true
	@cp docs/results/ablation/F17_*.pdf tex/figs/ 2>/dev/null || true
	@echo "Done."

.PHONY: stale-check
stale-check:  ## List tex/figs PDFs older than their docs/results source.
	@echo "Checking for stale figures ..."
	@for src in docs/results/*/*.pdf; do \
		 tgt="tex/figs/$$(basename $$src)"; \
		 if [ ! -f "$$tgt" ] || [ "$$src" -nt "$$tgt" ]; then \
		   echo "STALE: $$src → $$tgt"; \
		 fi; \
	done
	@echo "Stale check complete."

.PHONY: render-tables
render-tables:  ## Regenerate tex/generated/*.tex from canonical JSONs.
	$(PYTHON) scripts/thesis/render_tables.py

.PHONY: gen-results-index
gen-results-index:  ## Auto-generate docs/RESULTS_INDEX.md from canonical JSONs.
	$(PYTHON) scripts/thesis/gen_results_index.py

.PHONY: verify-fresh
verify-fresh:  ## Fail if any derived artifact is older than its canonical JSON source.
	$(PYTHON) scripts/thesis/verify_fresh.py

.PHONY: verify-fresh-fix
verify-fresh-fix:  ## Re-run render-tables + gen-results-index if any artifact is stale.
	$(PYTHON) scripts/thesis/verify_fresh.py --fix

##@ Reproducibility
.PHONY: reproduce-thesis
reproduce-thesis:  ## End-to-end thesis reproduction (full chain).
	@echo ">>> 1/6 process-data";   $(MAKE) process-data
	@echo ">>> 2/6 detector";        $(MAKE) detector
	@echo ">>> 3/6 blue-team";       $(MAKE) blue-team BLUE_TEAM_IMPACT_TERM=false
	@echo ">>> 4/6 benchmark";       $(MAKE) benchmark
	@echo ">>> 5/6 ablation";        $(MAKE) ablation
	@echo ">>> 6/6 smoke";           $(MAKE) smoke
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
