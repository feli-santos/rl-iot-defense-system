# RL IoT Defense System — Adversarial RL for Kill-Chain-Aware IoT Defense

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 473 passed](https://img.shields.io/badge/tests-473%20passed-brightgreen.svg)](#)
[![Thesis: 84 pages](https://img.shields.io/badge/thesis-84%20pages-blue.svg)](#thesis)

> A reproducible MSc-thesis codebase: an adversarial reinforcement-learning
> framework for kill-chain-aware defense on real IoT traffic (CICIoT2023).
> A windowed PPO defender is trained against a reactive tug-of-war attacker
> under genuine partial observability, and shown to beat a tuned supervised
> classifier exactly where per-flow classification becomes ambiguous.

---

## Headline results

The thesis rests on **three empirical threads**, all backed by manifest-pinned
artefacts under [`docs/results/`](docs/results/):

1. **Partial-observability crossover.** Sweeping the observation-aliasing rate
   α traces a clean crossover: at α=0 a windowed PPO defender and a tuned
   RandomForest classifier **tie** (PPO +131.3 vs RF +134.8, overlapping CIs),
   proving the environment does not favour RL by construction. As α rises, PPO
   holds flat (→+117.4) while RF degrades monotonically (→+67.7). From α=0.4
   onward the CIs are disjoint. The α=0 tie is the anchor; the crossover shape
   is the contribution.

2. **Reward-coupling ablation.** A reward shaped by a stage-action
   proportionality term hands the agent a privileged classification target,
   making a supervised classifier sufficient. We test this by training under
   both a **coupled** (shaped) and a **decoupled** (sparse outcome-only) reward.
   The best RL agent beats the tuned RF under **both** contracts (coupled gap
   −63.1; outcome gap −43.1), so the RL advantage is not an artefact of reward
   shaping.

3. **Out-of-distribution generalisation.** On 10 held-out zero-day attack
   classes, the windowed PPO defender prevents a substantially larger fraction
   of intrusions than RF-Acting on **every** class (PPO 0.37–0.65 vs RF
   0.00–0.18), and the advantage is **independent of the upstream detector's
   per-class recall** — a temporal-control property no memoryless per-flow
   classifier can reproduce.

**Honest finding.** Only the long-rollout on-policy PPO agent trains stably
under the sparse outcome reward; DQN and A2C fail to converge in this regime
and are reported as such rather than omitted.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                  ADVERSARIAL TRAINING LOOP                           │
├──────────────────────────────────────────────────────────────────────┤
│   ┌─────────────────────┐         ┌────────────────────────────────┐ │
│   │  Reactive Tug-of-War│         │   Blue Team                    │ │
│   │  Attacker           │  stage  │   DQN / PPO / A2C (SB3)       │ │
│   │  (proximity-coupled │ ──────► │   290-dim windowed observation │ │
│   │  escalation; no     │         │   5 actions: OBSERVE/LOG/      │ │
│   │  fixed budget)      │         │   RESTRICT/BLOCK/ISOLATE       │ │
│   └─────────────────────┘         └────────────────────────────────┘ │
│            │                      │ action                           │
│            ▼                      ▼                                  │
│   ┌─────────────────────┐  ┌──────────────────────────────────────┐ │
│   │  Realization Engine │  │  Kill-Chain Reward                    │ │
│   │  (session-coherent, │  │  (outcome: prevention bonus +        │ │
│   │  aliasing-aware     │  │  action cost; or coupled:            │ │
│   │  feature sampling)  │  │  proportionality shaping)            │ │
│   └─────────────────────┘  └──────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

The defender **never observes the true kill-chain stage** (POMDP). Partial
observability is controlled by the observation-aliasing rate α: with
probability α a step emits a feature row from an adjacent stage, so no single
row identifies the stage and the policy must integrate evidence over time.

Detailed designs:
- **[Architecture](docs/ARCHITECTURE.md)** — module map, adversarial loop, config flow
- **[Environment](docs/ENVIRONMENT.md)** — observation/actions/reward mechanics
- **[Kill-chain mapping](docs/kill-chain-mapping.md)** — CICIoT2023 → 5 stages
- **[Dataset card](docs/dataset_card.md)** — provenance, splits, feature provenance

---

## Quick start

### Prerequisites

- Python **3.9+** (developed and tested on 3.9)
- `make` (GNU Make or BSD Make)
- ~30 GB free disk for processed CICIoT2023 features + model checkpoints
- **Podman** (for thesis PDF builds; Docker is a fallback)

### Install

```bash
git clone https://github.com/feli-santos/rl-iot-defense-system.git
cd rl-iot-defense-system
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make test                    # 473 passed in ~60-90 s on CPU
```

### Dataset

This project consumes the
[**CICIoT2023**](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
dataset. Raw CSVs are **not** redistributed (CIC licensing). Download them
from the official source, place under `data/raw/ciciot2023/`, then:

```bash
make process-data            # raw CSVs → data/processed/ciciot2023/
make build-split-indices     # immutable train/val/test/OOD splits + hash manifest
```

### Smoke checks

```bash
make blue-team-smoke         # PPO seed 0, 5K steps (~20 s)
make benchmark-smoke         # 1 algo × 1 seed × 2 episodes (~20 s)
make ablation-ood-smoke      # 1 OOD class × 2 policies × 1 seed × 2 ep (~10 s)
```

If any smoke fails on a fresh checkout, treat it as a bug.

---

## Reproducibility

Every thesis figure ships with a sibling `manifest.json` (git SHA + input/output
SHA-256). The manifests form a hash chain anchored at the processed dataset and
reaching every figure in `docs/results/<area>/`.

```bash
python -m scripts.reproducibility_smoke           # verify all manifests
python -m scripts.reproducibility_smoke --strict  # exit 1 on any hash miss
```

### Full pipeline

```bash
make dataset                 # splits + dataset overview (~1 min)
make detector                # stage detector: RF + MLP (~3-5 min)
make blue-team               # full sweep DQN/PPO/A2C × 10 seeds (~3-7 h CPU)
make benchmark               # held-out benchmark eval (~10 min)
make ablation                # alpha-curve + coupling + OOD + sweeps (~2-4 h CPU)
```

`make help` prints every target with a one-line description.

### What's deterministic vs. seeded

- **Deterministic**: dataset splits, kill-chain stage labels, all manifest hashes
  (modulo Python/library patch versions; see `requirements.txt`).
- **Seeded**: RL training (10 seeds per algo via `--seed N`), evaluation
  rollouts (per-seed RNG; bootstrap CIs reported throughout).

---

## Thesis

The thesis PDF is the comprehensive documentation for this project. It builds
with Podman/TeXLive via `make thesis` (wraps `bash tex/build.sh`). The root
file is `tex/main.tex` (abnTeX2/FEEC template).

```bash
make thesis                  # build tex/main.pdf (~84 pages)
```

Numbers in the thesis are **macro-driven, never hand-typed**. Canonical
experiment JSONs live under `docs/results/<area>/`; `make render-tables`
regenerates `tex/generated/{numbers,tables}.tex` from them.

---

## Repository structure

```
rl-iot-defense-system/
├── src/                      # Library code (importable)
│   ├── algorithms/           # SB3-backed DQN/PPO/A2C wrappers
│   ├── benchmark/            # Baselines + eval runner
│   ├── blue_team/            # Env factory, callbacks, run config
│   ├── detector/             # Supervised stage detector (RF + MLP)
│   ├── environment/          # AdversarialIoTEnv (Gymnasium)
│   ├── generator/            # Markov attacker (kill-chain process)
│   └── utils/                # Dataset processor + realization engine
├── scripts/                  # Runners, plotters, thesis tooling
│   ├── data/  detector/  blue_team/  benchmark/  ablation/  thesis/
├── tests/                    # 473 unit + integration tests (pytest)
├── docs/                     # Architecture, environment, dataset docs
│   ├── results/              # Canonical figures + manifest hash-chains
│   ├── ARCHITECTURE.md       # Module map + adversarial loop
│   ├── ENVIRONMENT.md        # Obs/actions/reward mechanics
│   ├── dataset_card.md       # Dataset provenance
│   └── kill-chain-mapping.md # CICIoT2023 → 5-stage projection
├── tex/                      # Thesis LaTeX (abnTeX2/FEEC)
│   ├── main.tex              # Root document
│   ├── figs/                 # Figures (vector PDF)
│   └── build.sh              # Podman/TeXLive build script
├── config.yml                # Single source of hyperparameters
├── Makefile                  # Reproduction recipes (source of truth)
└── LICENSE                   # MIT
```

`runs/`, `data/processed/`, `artifacts/`, `mlruns/` are gitignored and live
only on the user's machine.

---

## Inspiring work

This project takes its conceptual cue from IoTWarden, which established the
trigger-action-attack / RL-defense paradigm. IoTWarden is **inspiration**, not
a baseline; the dataset, MDP, action space, and red team are all different and
no head-to-head numerical comparison is made or claimed.

> Alam, Md M., Jahan, I., & Wang, W. (2024). **IoTWarden: A Deep Reinforcement
> Learning Based Real-time Defense System to Mitigate Trigger-action IoT
> Attacks.** *arXiv preprint arXiv:2401.08141.*

---

## How to cite

```bibtex
@misc{santos2026rliotdefense,
  author       = {Santos, Felipe},
  title        = {{RL IoT Defense System: Adversarial Reinforcement Learning
                   for Kill-Chain-Aware IoT Defense}},
  year         = {2026},
  howpublished = {\url{https://github.com/feli-santos/rl-iot-defense-system}},
  note         = {MSc thesis software release}
}
```

---

## License

[MIT](LICENSE) © 2025–2026 Felipe Santos.

The CICIoT2023 dataset is governed by its own
[license terms](https://www.unb.ca/cic/datasets/iotdataset-2023.html); we
neither redistribute the raw CSVs nor any derivatives that would violate those
terms.

---

## Acknowledgements

This thesis builds on the IoTWarden line of work (Alam et al., 2024) and the
CICIoT2023 dataset published by the Canadian Institute for Cybersecurity. We
thank the Stable-Baselines3, Gymnasium, and PyTorch communities for the
libraries this project stands on.
