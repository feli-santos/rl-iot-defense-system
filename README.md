# RL IoT Defense System — Adversarial RL for Kill-Chain-Aware IoT Defense

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 428 passed](https://img.shields.io/badge/tests-428%20passed-brightgreen.svg)](#)
[![Phases: 0–7 closed](https://img.shields.io/badge/phases-0--7%20closed-brightgreen.svg)](#phases-as-chapters)
[![Release: v0.2.0](https://img.shields.io/badge/release-v0.2.0-blue.svg)](#)

> **TL;DR.** A reproducible MSc-thesis codebase: an adversarial
> reinforcement-learning framework for kill-chain-aware defense on
> real IoT traffic (CICIoT2023). We train DQN / PPO / A2C defenders
> against a fixed 5×5 first-order Markov attacker that walks the
> kill chain under a finite intrusion budget, and ship the empirical
> machinery — manifest hash-chains, audit-first PLANs, exit-gate
> scoreboards — that lets every figure in the thesis be regenerated
> from raw data with the named Makefile targets (`make dataset`, …, `make ablation`).

---

## Headline thesis claims

The thesis chapter rests on **three primary claims** plus one
pre-registered finding, all backed by gate-passing artefacts under
[`docs/results/`](docs/results/):

1. **(G6.2 — primary contract `impact_is_terminal=False`, finite `attacker_budget=40`, 10 seeds × 300 episodes)**
   On `test_balanced` (CICIoT2023, 5-stage kill chain), the three deployable RL defenders are
   **statistically tied** on reward: **A2C +278.5 (CI [+251.1, +308.8]) / PPO +274.5 / DQN +267.8**
   vs. an oracle recommended-action ceiling of **+543.1** (CI [+536.6, +549.4]) that has free access
   to the hidden `attack_stage` — i.e., the best deployable RL agent (A2C) captures **51.3 %** of
   the oracle ceiling (PPO 50.5 %, DQN 49.3 %, RF-Acting 82.5 %). The trade-off against the
   detector-coupled RF-Acting baseline is honest, not a domination claim: RF-Acting earns more reward
   (**+448.2**) but depends on an upstream stage classifier and runs **~176× slower** (16.505 ms p50,
   failing the 3 ms budget; trained RL ~0.094 ms p50, detector-free). Benign FPR is now **below 1 %**
   for all RL agents (DQN 0.46 %, PPO 0.89 %, A2C 0.66 %). always-BLOCK is the worst-scoring policy
   of all (−2005.06, indiscriminate over-force). *See
   [`docs/results/benchmark/RESULTS.md`](docs/results/benchmark/RESULTS.md) and
   `docs/results/benchmark/F5_summary.json`.*

2. **(Phase 7, G7.3)** Under the tug-of-war dynamics, PPO mean reward grows
   **monotonically** with the environment-difficulty parameter `p_down` (the
   tug-of-war de-escalation success probability), rising from **−111.1** at
   `p_down = 0.0` (harshest environment) to **+145.4** at `p_down = 1.0`; the
   headline operating point is `p_down = 0.90`. The trend is monotone
   non-decreasing across the full sweep. *See
   [`docs/results/ablation/F10_aggressiveness.png`](docs/results/ablation/F10_aggressiveness.png).*

3. **(G7.2 / D7.1.1)** Within the tug-of-war reward formulation, no single-axis
   0.5×/2× perturbation of any reward coefficient closes the deployable gap to
   the oracle ceiling. A *structural* env-semantics change
   (`impact_is_terminal=False`) is the highest-impact lever: it raises the
   structural mitigated-impact rate to **0.850** at reward **+278.5**, versus
   **0.0** for the mis-specified baseline contract — a decisive improvement that
   localises reward-mis-specification as the principal historical limitation. The
   **primary** security KPI is now `prevention_rate` (oracle 1.00 / a2c 0.60 /
   ppo 0.33 / dqn 0.54); `mitigated_impact_rate` is retired as a headline metric.
   *See [`docs/results/ablation/F9_reward_ablation.png`](docs/results/ablation/F9_reward_ablation.png).*

**Pre-registered finding (G7.9, now PASS — detector-independence dividend).** On the
held-out OOD class `VulnerabilityScan` (where the supervised detector is structurally
blind, recall ≈ 0.000), the detector-free RL policy **wins outright**: PPO's mean OOD
reward (**+298.3**) is within seed-noise of its in-distribution reward, while the
detector-coupled RF-Acting baseline **collapses to −4430.6** (Δ = **+4728.9** in RL's
favour). A blind detector mis-predicts the stage, under-forces under the tug-of-war
rule, and lets the attacker advance unchecked — exactly the failure mode the
detector-free agent avoids. This is the sharpest evidence for the POMDP /
detector-independence thesis. *See
[`docs/results/ablation/F15_ood_robustness.png`](docs/results/ablation/F15_ood_robustness.png)
and §6.2 of `docs/results/ablation/RESULTS.md`.*

---

## What's in this repo

```
rl-iot-defense-system/
├── src/                      # Library code (importable)
│   ├── algorithms/           # Adversarial-RL algorithm wrappers (SB3 backed)
│   ├── benchmark/            # Phase-6 baselines + eval runner + latency bench
│   ├── blue_team/            # Phase-5 env factory, callbacks, run config
│   ├── detector/             # Phase-4 supervised stage detector (RF + MLP)
│   ├── environment/          # Phase-3 AdversarialIoTEnv (Gymnasium-compatible)
│   ├── generator/            # Phase-2 Markov attacker (5×5 kill-chain process)
│   ├── training/             # Generic training-manager
│   └── utils/                # Dataset processor + realisation engine + I/O
├── scripts/                  # Phase-pinned runners, plotters, gate evaluators
│   ├── data/  red_team/  detector/  blue_team/  benchmark/  ablation/
│   └── (each subdir is owned by exactly one phase; see Makefile)
├── tests/                    # 428 unit + integration tests (pytest)
├── docs/results/             # Canonical thesis figures + RESULTS chapters
│   ├── dataset/   environment/   stage-detector/
│   ├── blue-team-training/ benchmark/  ablation/
│   └── (each chapter has PLAN.md + RESULTS.md + manifests + figures)
├── docs/                     # Consolidated knowledge base
│   ├── ARCHITECTURE.md       # Module map + adversarial loop + config flow
│   ├── ENVIRONMENT.md        # Obs/actions/reward/budget mechanics
│   ├── RESULTS.md            # budget=40 headline + gate scoreboard
│   ├── STATUS.md             # Live status, locked decisions, journal
│   ├── dataset_card.md  kill-chain-mapping.md
│   └── RESULTS_INDEX.md      # Auto-generated figure index
├── data/img/                 # Static dataset diagrams (CICIoT2023 topology)
├── notebooks/                # Exploratory only; not on the thesis path
├── config.yml                # Single source of hyperparameters
├── Makefile                  # `make phase-N` reproduction recipes
├── CITATION.cff              # How to cite this work
└── LICENSE                   # MIT
```

`runs/` (per-phase outputs) and `data/processed/` (CICIoT2023 features)
are **gitignored** and live only on the user's machine. The recipe to
re-create them from raw CSV is in [§ Reproducibility](#reproducibility).

---

## Phases as chapters

The thesis chapter is organised as **eight closed phases** (0–7), each
with a locked `PLAN.md`, an exit-gate scoreboard `G<N>_scoreboard.json`,
a hand-written `RESULTS.md`, and at least one canonical figure under
`docs/results/<area>/`. Phase 10 (this README, code-cleanup, release
tag) is documentation-only.

| # | Phase | What it produces | Headline gate |
|---|---|---|---|
| **1** | Dataset & splits | F0 dataset overview · `data/processed/ciciot2023/` · immutable train/val/test/OOD index manifests | Hashes pin every downstream split |
| **2** | Tug-of-war attacker | Reactive `MarkovAttacker` on the kill chain: a proportionate defender action de-escalates one stage (p_down=0.90; ISOLATE 0.98), under-force advances (p_up=0.90), over-force holds; strictly sequential, absorbing IMPACT | Stage dynamics drive the adversarial environment |
| **3** | Environment v2 | `AdversarialIoTEnv` (Gymnasium); 29-feature obs; 5 actions; kill-chain reward | G3.1–G3.6 PASS (env contracts + reward shape) |
| **4** | Stage detector | F11 per-stage recall (Random Forest + MLP); RF-acting baseline export | G4 PASS — but `VulnerabilityScan` recall = 0.000 (audit-AF1 surface) |
| **5** | RL Blue Team | F3 (reward curves DQN/PPO/A2C × 10 seeds) · F4 (action distribution evolution) · T1 (hyperparams) | G5 PASS — all three algorithms converge above random |
| **6** | RL benchmark | F5 (security metrics) · F6 (stage × action confusion) · F7 (latency CDF + train time) · F8 (RL vs non-RL baselines) | G6 PASS — A2C best deployable RL (+278.5 @ budget=40); oracle ceiling +543.1 (reframed D6.2.1, audit-AF2) |
| **7** | Ablations + OOD | F9 (reward sweep) · F10 (environment-difficulty) · F12 (Pareto) · F15 (held-out OOD class) · F16 (budget sweep) · F17 (evasion sweep) | **8 PASS / 2 FAIL-WITH-FINDING** across G7.1–G7.10 — both FAIL gates pre-registered (D7.1.1, R7.3) |

### Phase reproduction recipes

Every phase is reproducible end-to-end with the corresponding
named Makefile target. CPU-only times below assume an Apple-silicon
laptop or comparable.

```bash
make dataset                 # Phase 1: Dataset splits + F0 (~1 min)
make detector                # Phase 4: Stage detector (RF + MLP) + F11 (~3-5 min)
make blue-team-smoke         # Phase 5 smoke: PPO seed 0, 5K steps (~20 s)
make blue-team               # Phase 5: Full sweep DQN/PPO/A2C × 10 seeds + F3/F4/T1 (~3-7 h CPU)
make benchmark-smoke         # Phase 6 smoke: 1 algo × 1 seed × 2 episodes (~20 s)
make benchmark               # Phase 6: Eval + F5/F6/F7/F8 (~10 min CPU after blue-team)
make ablation-ood-smoke      # Phase 7 smoke: 1 OOD class × 2 policies × 1 seed × 2 ep (~10 s)
make ablation                # Phase 7: Full F9/F10/F12/F15 + closeout (~7.5 h CPU walk-away)
```

`make help` prints every target with a one-line description.

### Phase 7 final gate scoreboard

| Gate | Threshold | Status | Headline value |
|---|---|:---:|---|
| **G7.1** | `pytest -q` ≥ 428 passed; zero new skips | **PASS** | 428 passed |
| **G7.2** | F9 best reward-comparable mean test reward > Phase-6 deployable best by ≥ 1 σ | **FAIL-WITH-FINDING (D7.1.1)** | best reward-comparable cell = `impact_is_terminal_false` +278.5 (mit-rate 0.850 vs 0.0 mis-specified baseline) |
| **G7.3** | PPO p_down = 0.0 < p_down = 0.6 by ≥ 1 σ AND rule monotone | **PASS** | environment-difficulty (p_down) sweep: p_down=0.0 mean −111.1; p_down=1.0 mean +145.4 |
| **G7.4** | Pareto frontier ≥ 3 distinct dominant points | **FAIL-WITH-FINDING (R7.3)** | n_distinct = 1 / 32 — under perfect perception the oracle dominates at (security_gain=1.0, availability_cost=0.0); interior RL placement quantifies the cost of partial observability |
| **G7.5** | Phase-3 frozen tests pass with `impact_is_terminal=True` | **PASS** | full pytest green |
| **G7.6** | No regression on Phase-3/4/5/6 frozen tests | **PASS** | — |
| **G7.7** | F9 / F10 / F12 / F15 manifests SHA-pinned | **PASS** | all 4 present |
| **G7.8** | F15 4 × 8 OOD matrix complete, no NaN | **PASS** | 32 / 32 cells |
| **G7.9** | On VulnerabilityScan, trained RL > RF-acting by ≥ 1 σ | **PASS** | detector-free PPO +298.3 vs detector-coupled RF-acting −4430.6 (Δ = +4728.9) |

Both FAIL-WITH-FINDING gates (G7.2 → D7.1.1, G7.4 → R7.3) were **pre-registered** in `docs/results/ablation/PLAN.md` §6/§8 — neither is a goalpost move. G7.9 (held-out OOD class) now **passes**: the detector-free RL policies decisively beat detector-coupled RF-acting on `VulnerabilityScan`, where the supervised stage detector is structurally blind (recall ≈ 0.000).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                  ADVERSARIAL TRAINING LOOP (Phase 3)                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────────┐         ┌────────────────────────────────┐ │
│   │   RED TEAM          │         │   BLUE TEAM (Phase 5/6/7)      │ │
│   │   (Phase 2)         │         ├────────────────────────────────┤ │
│   ├─────────────────────┤  next   │                                │ │
│   │  5x5 Markov attacker│  stage  │   DQN / PPO / A2C  (SB3)       │ │
│   │  (first-order kill- │ ──────► │                                │ │
│   │  chain process;     │         │   29-feature observation       │ │
│   │  finite budget)     │         │   (window=5, deltas on)        │ │
│   └─────────────────────┘         │                                │ │
│            │                      │   5 actions (force continuum): │ │
│            │ stage label          │     OBSERVE / LOG /            │ │
│            ▼                      │     RESTRICT / BLOCK /         │ │
│   ┌─────────────────────┐         │     ISOLATE                    │ │
│   │  RealisationEngine  │         └────────────────────────────────┘ │
│   │  (Phase 1)          │                       │                    │
│   ├─────────────────────┤                       │ action             │
│   │  Samples real       │                       │                    │
│   │  CICIoT2023 row from│                       ▼                    │
│   │  the stage's pool   │         ┌────────────────────────────────┐ │
│   │  (allowed_indices)  │ feature │   Kill-Chain Reward            │ │
│   └─────────────────────┘ vector  │   (Phase 3, calibrated)        │ │
│            │                      │                                │ │
│            └─────────────────────►│   defense_success_bonus,       │ │
│                                   │   impact_penalty,              │ │
│                                   │   penalty_overreact_benign…    │ │
│                                   └────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

Detailed designs:
- **Environment contract (obs/actions/reward/budget):** [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)
- **Architecture (module map + adversarial loop + config flow):** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- **Kill-chain mapping (CICIoT2023 → 5 stages):** [`docs/kill-chain-mapping.md`](docs/kill-chain-mapping.md)
- **Results (budget=40 headline + gate scoreboard):** [`docs/RESULTS.md`](docs/RESULTS.md)
- **Status, locked decisions & journal:** [`docs/STATUS.md`](docs/STATUS.md)

---

## Quick start

### Prerequisites

- Python **3.9+** (tested on 3.9.6 macOS Tahoe; CI tests on Linux + macOS)
- `make` (GNU Make 3.81+ or BSD Make)
- ~30 GB free disk for the processed CICIoT2023 dataset + model
  checkpoints (raw CICIoT2023 CSVs not included; see below)

### Install

```bash
git clone https://github.com/feli-santos/rl-iot-defense-system.git
cd rl-iot-defense-system
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make test                    # 428 passed in ~60-90 s on CPU
```

### Dataset

This project consumes the
[**CICIoT2023**](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
dataset. The raw CSVs are **not** redistributed in this repo (CIC
licensing). Download them from the official source, place them under
`data/raw/ciciot2023/`, then run:

```bash
make process-data            # raw CSVs → data/processed/ciciot2023/
make build-split-indices     # immutable train/val/test/OOD splits + hash manifest
```

Phase-1 manifests pin every downstream split with a SHA-256 — see
[`docs/dataset_card.md`](docs/dataset_card.md).

### Run a smoke check

```bash
source .venv/bin/activate
make phase-5-smoke           # PPO seed 0, 5K timesteps, ~20 s
make phase-6-smoke           # 1 algo × 1 seed × 2 episodes, ~20 s
make phase-7-ood-smoke       # 1 OOD class × 2 policies × 1 seed × 2 ep, ~10 s
```

If any smoke fails on a fresh checkout, surface it as a bug — the
project's protocol (see [§ Operating principles](#operating-principles))
treats smoke-test breakage as the canary for design / env-config drift.

---

## Reproducibility

Every thesis figure ships with a sibling `manifest.json` containing:

```json
{
  "figure_id":   "F9",
  "title":       "Reward-component ablation",
  "produced_by": "scripts/ablation/plot_reward_ablation.py",
  "git_sha":     "<commit at generation time>",
  "inputs":  [{"path": "...eval_test.jsonl", "sha256": "<hash>"}],
  "outputs": [{"path": "F9_reward_ablation.png", "sha256": "<hash>"}]
}
```

The manifests form a **hash chain** anchored at `data/processed/ciciot2023/`
(Phase 1) and reaching every figure in `docs/results/<area>/`. Verify
the chain end-to-end with:

```bash
python -m scripts.reproducibility_smoke           # verify all manifests
python -m scripts.reproducibility_smoke --strict  # exit 1 on any hash miss
```

A figure that doesn't have a manifest (or whose hashes don't reconcile)
is **not** considered defense-ready. See
[`docs/STATUS.md`](docs/STATUS.md) for the reproducibility protocol and
the per-area `docs/results/<area>/RESULTS.md` for authoring conventions.

### What's deterministic vs. seeded

- **Deterministic**: dataset splits (Phase 1), kill-chain stage labels
  (Phase 4 prep), all `manifest.json` hashes (modulo Python and library
  patch versions; see `requirements.txt` for pinned versions).
- **Seeded**: RL training (10 seeds per algo, exposed via
  `--seed N`), evaluation episode rollouts (per-seed RNG; bootstrap CIs
  reported throughout RESULTS).

---

## Inspiring work

This project takes its conceptual cue from the IoTWarden paper, which
established the trigger-action-attack / RL-defense paradigm we build
on. IoTWarden is **inspiration**, not a baseline; the dataset, MDP,
action space, and red team are all different and no head-to-head
numerical comparison is made or claimed.

> Alam, Md M., Jahan, I., & Wang, W. (2024).
> **IoTWarden: A Deep Reinforcement Learning Based Real-time Defense
> System to Mitigate Trigger-action IoT Attacks.**
> *arXiv preprint arXiv:2401.08141.*

Notable differences from IoTWarden's setup:

- **Environment.** IoTWarden uses a hand-crafted IFTTT trigger graph
  with a small synthetic state space. We use a 29-feature
  CICIoT2023-derived observation vector with realistic per-stage
  feature distributions (`RealizationEngine`, Phase 3).
- **Red team.** IoTWarden samples attack triggers from a fixed
  schedule. We use a fixed 5x5 first-order Markov attacker
  (`MarkovAttacker`, Phase 2) — an upper-triangular kill-chain
  transition process with an absorbing IMPACT state, operating under a
  finite intrusion budget — whose sampled stage trajectory drives the
  realisation engine.
- **Action space.** IoTWarden uses a binary block-or-not action. We
  use a 5-level graduated **force continuum** (OBSERVE → LOG →
  RESTRICT → BLOCK → ISOLATE), which lets the policy under- and
  over-react in measurable ways.
- **Reproducibility.** We ship a hash-chain-pinned set of
  `make phase-N` recipes that regenerate every figure end-to-end.

One design choice we deliberately kept from IoTWarden is the
stage-action *recommended-action* mapping (BENIGN→OBSERVE,
RECON→LOG, ACCESS→RESTRICT, MANEUVER→BLOCK, IMPACT→ISOLATE), which
seeds our oracle baseline policy.

---

## Operating principles

The eight closed phases share a common protocol that the codebase enforces.
The revision history and all locked empirical decisions are tracked in
[`docs/results/`](docs/results/) (per-phase `PLAN.md` + `RESULTS.md`); see also
[`docs/archive/HANDOFF.md`](docs/archive/HANDOFF.md) for the historical Phase-7→10
handoff record.

1. **Audit-first.** Every new phase opens with a `PLAN.md` that contains
   audit findings, deliverables, exit gates, sequencing, and what we are
   **not** doing. Plan goes through a "lock decisions" commit *before*
   any implementation lands.
2. **Empirical gates.** Every phase has named exit gates `G<N>.<i>` with
   explicit numerical thresholds. When a gate fails, we treat the
   failure as **diagnostic** — historical record shows phases 3–7 each
   turned at least one FAIL into a thesis-credible finding via dated
   `D<N>.X.1` decisions in PLAN §8.
3. **Hash-chain everything.** See [§ Reproducibility](#reproducibility).
4. **Honest commit history.** Bugs found mid-phase are fixed as
   `fix(phase-<N>):` commits attributed to the discovering phase, with
   the issue logged in that phase's `RESULTS.md` §5. Earlier-phase
   numbers are *never* retroactively touched without a dated decision.
5. **Mentor-mode communication.** Brief, direct, lead with the result.
   Cite numbers, paper figures, gate IDs, commit SHAs by name.

---

## Tests

```bash
make test                    # 428 passed in ~60-90 s on CPU
make test-cov                # with coverage
```

Test layout (one-to-one with `src/` modules and the cross-phase
parsers):

| Test file | Coverage area |
|---|---|
| `test_dataset_processor.py` · `test_realization_engine*.py` · `test_label_mapper.py` · `test_build_split_indices.py` · `test_derive_stage_labels.py` | Phase 1 — data pipeline + realisation engine |
| `test_attack_sequence_generator.py` · `test_episode_generator.py` · `test_generator_trainer.py` · `test_red_team_helpers.py` · `test_transition_mask.py` | Phase 2 — Red Team |
| `test_adversarial_env.py` · `test_phase3_env_gates.py` · `test_phase31_impact_terminal.py` · `test_adversarial_algorithm.py` | Phase 3 — Environment v2 |
| `test_detector.py` | Phase 4 — supervised stage detector |
| `test_blue_team_*.py` · `test_train_agent_reward_overrides.py` | Phase 5/7 — Blue Team training |
| `test_baseline_policies.py` · `test_benchmark_eval_runner.py` · `test_benchmark_latency.py` | Phase 6 — RL benchmark |
| `test_close_ablation_parsers.py` | Phase 7 — gate-evaluator parsers (audit-fix `7537493`) |

Real-data smoke tests are guarded with
`pytest.skipif(not Path('data/processed/...').exists(), ...)`; the
428 reported above is the synthetic-only count.

---

## How to cite this work

If you use this codebase or its figures in a publication, please cite:

```bibtex
@misc{santos2026rliotdefense,
  author       = {Santos, Felipe},
  title        = {{RL IoT Defense System: Adversarial Reinforcement Learning
                   for Kill-Chain-Aware IoT Defense}},
  year         = {2026},
  version      = {v0.1.0},
  howpublished = {\url{https://github.com/feli-santos/rl-iot-defense-system}},
  note         = {MSc thesis software release}
}
```

A machine-readable citation is in [`CITATION.cff`](CITATION.cff).

---

## License

[MIT](LICENSE) © 2025–2026 Felipe Santos.

The CICIoT2023 dataset is governed by its own
[license terms](https://www.unb.ca/cic/datasets/iotdataset-2023.html); we
neither redistribute the raw CSVs nor any derivatives that would violate
those terms. The hash-chain manifests cite each input split by SHA-256
without exposing its content.

---

## Acknowledgements

This thesis builds on the IoTWarden line of work
(Alam et al., 2024) and the CICIoT2023 dataset published by the
Canadian Institute for Cybersecurity. We thank the Stable-Baselines3,
Gymnasium, and PyTorch communities for the libraries this project
stands on.
