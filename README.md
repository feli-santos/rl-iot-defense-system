# RL IoT Defense System — Adversarial RL for Kill-Chain-Aware IoT Defense

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 432 passed](https://img.shields.io/badge/tests-432%20passed-brightgreen.svg)](#)
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
   On `test_balanced` (CICIoT2023, 5-stage kill chain), trained RL defenders earn
   **PPO +1034.7 (CI [+998.1, +1069.8]) / DQN +1028.9 / A2C +973.1 mean reward** vs. an
   oracle recommended-action ceiling of **+1393.8** (CI [+1366.9, +1420.6]) that has free access
   to the hidden `attack_stage` — i.e., the best deployable RL agent (PPO) captures **74.2 %** of
   the oracle ceiling. Among deployable policies, any trained RL algo achieves **~142.8× faster
   inference** than RF-Acting (0.096 ms vs 13.692 ms p50). Benign FPR: DQN 7.5 %, PPO 8.7 %,
   A2C 7.7 %. Non-RL trivial baselines (random, always-OBSERVE, always-BLOCK) never come within
   1 σ. *See [`docs/results/benchmark/RESULTS.md`](docs/results/benchmark/RESULTS.md) and
   `docs/results/benchmark/F5_summary.json`.*

2. **(Phase 7, G7.3)** With the Phase-3 reward function held fixed, PPO
   mean reward grows **monotonically** with the
   `p_defender_de-escalation` parameter, increasing roughly tenfold from
   p = 0.0 (CI 134, 141) to p = 0.6 (CI 1280, 1359). The trend is
   monotone non-decreasing across the full sweep. *See
   [`docs/results/ablation/aggressiveness.png`](docs/results/ablation/aggressiveness.png).*

3. **(G7.2 / D7.1.1 partial)** Within the Phase-3 reward formulation,
   no single-axis 0.5×/2× perturbation of any reward coefficient closes
   the deployable gap to the oracle ceiling. A *structural* env-semantics
   change (`impact_is_terminal=False`) is the highest-impact lever: in an
   ablation probe (PPO-only, n=30 episodes), it raises the mitigated-impact
   rate from 0.153 to **0.840** (5.5×) and mean reward to **+1544.4**. At
   full benchmark scale (n=300, 10 seeds, all three algorithms), the trained
   policies under this primary contract achieve mitigated-impact rates of
   **0.26–0.32** — a genuine improvement over the mis-specified baseline,
   while demonstrating that reward-mis-specification is the principal
   limitation. *See [`docs/results/ablation/reward_ablation.png`](docs/results/ablation/reward_ablation.png).*

**Pre-registered finding (G7.9, D7.9.1).** On the held-out OOD class
`VulnerabilityScan`, RL is **robust to** but not **better at** the
distribution shift: PPO's mean OOD reward (+1355.2) is within seed-noise
of its in-distribution mean (+1320.2). RF-acting's higher OOD reward
(+1680.0, Δ = −324.8 vs PPO) is *not* evidence of RF working
(detector recall = 0.001) — it is evidence that "do nothing" is locally
rewarded when the reward is dominated by avoiding disproportionate-penalty
costs. *See [`docs/results/ablation/ood_robustness.png`](docs/results/ablation/ood_robustness.png) and §6.2 of `docs/results/ablation/RESULTS.md`.*

---

## What's in this repo

```
rl-iot-defense-system/
├── src/                      # Library code (importable)
│   ├── algorithms/           # Adversarial-RL algorithm wrappers (SB3 backed)
│   ├── benchmark/            # Phase-6 baselines + eval runner + latency bench
│   ├── blue_team/            # Phase-5 env factory, callbacks, run config
│   ├── detector/             # Phase-4 supervised stage detector (RF + 1D-CNN)
│   ├── environment/          # Phase-3 AdversarialIoTEnv (Gymnasium-compatible)
│   ├── generator/            # Phase-2 Markov attacker (5×5 kill-chain process)
│   ├── training/             # Generic training-manager
│   └── utils/                # Dataset processor + realisation engine + I/O
├── scripts/                  # Phase-pinned runners, plotters, gate evaluators
│   ├── data/  red_team/  detector/  blue_team/  benchmark/  ablation/
│   └── (each subdir is owned by exactly one phase; see Makefile)
├── tests/                    # 432 unit + integration tests (pytest)
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
| **2** | Markov attacker | Fixed 5×5 first-order kill-chain transition matrix (`MarkovAttacker`); upper-triangular, absorbing IMPACT | Stage dynamics drive the adversarial environment |
| **3** | Environment v2 | `AdversarialIoTEnv` (Gymnasium); 29-feature obs; 5 actions; kill-chain reward | G3.1–G3.6 PASS (env contracts + reward shape) |
| **4** | Stage detector | F11 per-stage recall (Random Forest + 1D-CNN); RF-acting baseline export | G4 PASS — but `VulnerabilityScan` recall = 0.001 (audit-AF1 surface) |
| **5** | RL Blue Team | F3 (reward curves DQN/PPO/A2C × 10 seeds) · F4 (action distribution evolution) · T1 (hyperparams) | G5 PASS — all three algorithms converge above random |
| **6** | RL benchmark | F5 (security metrics) · F6 (stage × action confusion) · F7 (latency CDF + train time) · F8 (RL vs non-RL baselines) | G6 PASS — PPO best deployable RL (+1034.7 @ budget=40); oracle ceiling +1393.8 (reframed D6.2.1, audit-AF2) |
| **7** | Ablations + OOD | F9 (reward sweep) · F10 (aggressiveness) · F12 (Pareto) · F15 (held-out OOD class) · F16 (budget sweep) · F17 (evasion sweep) | **8 PASS / 2 FAIL-WITH-FINDING** across G7.1–G7.10 — both FAIL gates pre-registered (D7.1.1, D7.9.1) |

### Phase reproduction recipes

Every phase is reproducible end-to-end with the corresponding
named Makefile target. CPU-only times below assume an Apple-silicon
laptop or comparable.

```bash
make dataset                 # Phase 1: Dataset splits + F0 (~1 min)
make detector                # Phase 4: Stage detector (RF + 1D-CNN) + F11 (~3-5 min)
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
| **G7.1** | `pytest -q` ≥ 430 passed; zero new skips | **PASS** | 432 passed |
| **G7.2** | F9 best reward-comparable mean test reward > Phase-6 deployable best by ≥ 1 σ | **PASS-WITHOUT-STRETCH** | ablation probe best = `impact_is_terminal_false` PPO +1544.4; benchmark-scale mit-rate = 0.26–0.32 |
| **G7.3** | PPO p = 0.0 < p = 0.6 by ≥ 1 σ AND rule monotone | **PASS** | p = 0.0 CI (134, 141); p = 0.6 CI (1280, 1359) |
| **G7.4** | Pareto frontier ≥ 3 distinct dominant points | **FAIL-WITH-FINDING (R7.3)** | n_distinct = 1 / 32 — trade-off surface is ~linear |
| **G7.5** | Phase-3 frozen tests pass with `impact_is_terminal=True` | **PASS** | full pytest green |
| **G7.6** | No regression on Phase-3/4/5/6 frozen tests | **PASS** | — |
| **G7.7** | F9 / F10 / F12 / F15 manifests SHA-pinned | **PASS** | all 4 present |
| **G7.8** | F15 4 × 8 OOD matrix complete, no NaN | **PASS** | 32 / 32 cells |
| **G7.9** | On VulnerabilityScan, trained RL > RF-acting by ≥ 1 σ | **FAIL-WITH-FINDING (D7.9.1)** | PPO +1355.2 vs RF +1680.0 (Δ = −324.8) |

Both FAIL gates were **pre-registered** in `docs/results/ablation/PLAN.md` §6/§8 — neither is a goalpost move.

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
│            ▼                      │     THROTTLE / BLOCK /         │ │
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
make test                    # 432 passed in ~60-90 s on CPU
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
  THROTTLE → BLOCK → ISOLATE), which lets the policy under- and
  over-react in measurable ways.
- **Reproducibility.** We ship a hash-chain-pinned set of
  `make phase-N` recipes that regenerate every figure end-to-end.

One design choice we deliberately kept from IoTWarden is the
stage-action *recommended-action* mapping (BENIGN→OBSERVE,
RECON→LOG, ACCESS→THROTTLE, MANEUVER→BLOCK, IMPACT→ISOLATE), which
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
make test                    # 432 passed in ~60-90 s on CPU
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
432 reported above is the synthetic-only count.

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
