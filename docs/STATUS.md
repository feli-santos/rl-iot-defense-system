# Project Status — Adversarial-RL IoT Defense (prevention pivot)

> Single load-bearing state/decisions/progress document. Supersedes the old
> `docs/review/` ledger set. Edit the live sections in place; append to the
> Journal (never rewrite history). The thesis under `tex/` plus this file are
> the canonical knowledge source for future work.

Branch: `feat/prevention-pivot` · Venue lens: IEEE Internet of Things Journal
(IoT-J) · Author: Felipe Augusto Oliveira dos Santos, MSc UNICAMP/FEEC (advisor
Prof. Dr. Denis Fantinato).

---

## 1. The pivot in one paragraph

The LSTM red-team — a frozen 97.7%-fidelity imitator of a hand-built 5×5 Markov
matrix, with no expressive power and no adaptivity — was **cut** and that Markov
transition matrix was **promoted to be the attacker**. A **finite attacker
budget** was added so correct, timely kill-chain defense can **exhaust the
attacker before IMPACT**: compromise becomes *preventable* and `compromise_rate`
drops below 1.0 as a function of policy quality. An **evasion-before-commit
reactive attacker** (probabilistic stall when recently blocked at RECON/ACCESS)
adds a defender-coupled adaptive axis. The Cyber Kill Chain stays and is now
*more* central (semantic spine + anti-toy-MDP shield). Every observation is a
real CICIoT2023 flow row sampled per stage by the `RealizationEngine`.

## 2. Locked decisions (fixed contracts — do not silently change)

- **Primary reward contract: `impact_is_terminal=False`** for training and
  benchmark. `impact_is_terminal=True` is retained only as a
  reward-misspecification case study. Code default is `True`
  (`run_config.py`/`adversarial_env.py`); scripts pass `False` explicitly.
- **`attacker_budget=40`** (LOCKED) — calibrated in the budget micro-sweep
  (commit `727e21f`). It is an experiment contract injected at runtime via
  Make/CLI, **not** a code default (code default is `None` = unbounded).
- **10 seeds `{0..9}`** for DRL; baselines/oracle run 1 seed. **n=300 episodes**
  for every policy in benchmark and OOD ablation. `p_defender_de-escalation=0.6`
  default.
- Legacy artifacts are removed by **pure git-rm** (history is the only record).
- Kill-chain stages: `0 BENIGN, 1 RECON, 2 ACCESS, 3 MANEUVER, 4 IMPACT`.
  Actions: `0 OBSERVE, 1 LOG, 2 RESTRICT, 3 BLOCK, 4 ISOLATE`.

## 3. Canonical headline (budget=40, from `docs/results/benchmark/F5_summary.json`)

| Policy | Mean reward (95% CI) | p50 latency | benign FPR |
|---|---|---|---|
| **A2C** (best deployable RL by reward) | **+278.5** [+251.1, +308.8] | 0.094 ms | 0.7% |
| PPO | +274.5 [+252.2, +294.6] | 0.094 ms | 0.9% |
| DQN | +267.8 [+215.3, +321.7] | 0.063 ms | 0.5% |
| `recommended_action` oracle (ceiling) | **+543.1** [+536.6, +549.4] | — | — |
| RF-Acting (hero comparator) | +448.2 | 16.505 ms | 100% |
| always_block / random / always_observe | −2005.1 / −573.9 / −393.2 | — | — |

Best deployable RL (A2C) captures **51.3%** of the oracle ceiling at
**~176× lower latency** than RF-Acting (whose 16.5 ms p50 fails the 3 ms budget).
The three DRL agents are statistically indistinguishable on reward (overlapping
CIs). All RL benign FPR is **below 1%** (DQN 0.5% / PPO 0.9% / A2C 0.7%); only
random (41%) and always_block (100%) breach the 1% threshold. RF-Acting attains
higher reward but is detector-coupled and ~176× slower — an honest deployment
trade-off, not a reward-domination claim. F9 reward-ablation structural strand:
reward +278.5, mit-rate 0.850 (vs 0.0 for the mis-specified baseline contract).
Under the tug-of-war + finite budget, `prevention_rate` (primary KPI) varies by
policy (oracle 1.00, A2C 0.60, PPO 0.33); always_block prevents 100% but only by
indiscriminate force (100% benign FPR), making it the worst-scoring policy.

## 4. Gate scoreboard (Ablation & Robustness, `docs/results/ablation/G7_scoreboard.json`)

**8 PASS / 2 FAIL-WITH-FINDING / 0 FAIL across G7.1–G7.10.** Both FAILs are
pre-registered findings, not regressions:

- **G7.2 → D7.1.1**: reward-shaping raw-reward ceiling, but the security-KPI
  strand passes (structural mit-rate 0.850 vs 0.0 for the mis-specified baseline).
- **G7.4 → F12 Pareto**: FAIL-WITH-FINDING — under perfect stage perception the
  security/availability frontier collapses to a single dominant point (the
  oracle, security_gain=1.0 at near-zero availability cost), strictly dominating
  always_block and every interior learned policy; the interior placement of the
  RL agents *visualises the cost of partial observability* (POMDP).

Note: **G7.9 now PASSES** (was D7.9.1 FAIL-WITH-FINDING pre-redesign). On the
VulnerabilityScan OOD class — where the supervised detector is blind (recall
0.001) — detector-free RL (PPO +298.3) **beats** detector-coupled RF-Acting
(−4430.6) by +4728.9: the blind detector mis-predicts → recommends under-forcing →
the tug-of-war attacker advances unchecked to IMPACT (catastrophic reward), while
the detector-free policy stays robust. This is the sharpest demonstration of the
detector-independence dividend.

Other gates: G7.3 environment-difficulty p_down sweep (IoTWarden-shaped),
G7.4 budget-hump peak @40, G7.5–G7.7 manifests, G7.8 OOD coverage 32/32, G7.10
evasion robustness (F17: PPO reward degrades gracefully across evasion 0→0.75).

## 5. Caveat dispositions (implementation hazards)

| ID | Caveat | Status |
|---|---|---|
| C1 | Budget step-cost charged inside the progression branch only; de-escalation charges `reset_cost` (=2) once (no double-charge). | Implemented. |
| C2 | Grace clamp downgrades pre-`min_episode_length`(20) IMPACT→MANEUVER; report prevention conditioned on `step≥20`. | Implemented; headline metric `prevent_pg`. |
| C3 | De-escalation has coupled drain-vs-prolong effects — empirical, not paper-resolvable. | Resolved by budget calibration (F16 pivot @40). |
| C6 | Which 29 features & why. | RESOLVED `202859c` — `docs/results/dataset/feature_provenance.json`. |
| C7 | ~~THROTTLE never triggers de-escalation → dominated action.~~ | OBSOLETE after tug-of-war redesign: action renamed RESTRICT and is now the *proportional recommended action* at ACCESS, de-escalating the attacker w.p. `p_down=0.90` and draining budget; ISOLATE (`p_down_isolate=0.98`) is strictly stronger than BLOCK. The graded RESTRICT<BLOCK<ISOLATE ladder is fully exercised — no dominated action remains. |
| C9 | Eval-env `impact_is_terminal` defaulted True while training used False. | RESOLVED `c0b39d0`. |
| C10 | Benchmark `eval_manifest.json` recorded `attacker_budget=None` despite budget=40 applying. | RESOLVED `a809bc1` (manifest now self-contained; regression guard `tests/test_run_test_eval_manifest.py`). |

## 6. Known stale-code notes (not on the thesis reproduction path)

- `main.py` `train_rl` is **stale/broken** against the current env (passes
  removed reward kwargs; requires the deleted `attack_sequence_generator.pth`).
  The canonical training entry is `scripts/blue_team/train_agent.py`.
- Stale "LSTM"/"Red-Team" docstrings remain in `config.yml`,
  `src/generator/episode_generator.py`, `run_config.py`, and
  `config_loader._validate_config` (still requires a legacy `attack_generator`
  section). Harmless; off the live loop.

## 7. Journal (newest first — append, never rewrite)

- `7bbe607` — drop CHANGELOG; make-thesis toolchain Podman-first; AGENTS thesis
  block fixed.
- `69cefba` — rename thesis sources to English (principal→main, preambulo→
  preamble, introducao→introduction, conclusao→conclusion, apendice→appendix,
  tese.bib→thesis.bib); drop Overleaf integration (`.olcli.json`/`.olignore`).
- `c07b305` — refresh AGENTS/README canonical headline to budget=40.
- `d7b6d75` — cut Red-Team LSTM-imitation gate artifacts (G1–G5) + delink
  dangling tooling.
- `202859c` — C6 resolved: `feature_provenance.json` (29-col basis).
- `6ae1441` — centralized PNG→PDF figure export + F-named PDFs.
- `a8b2dca` — repoint thesis toolchain to F-named summaries; budget=40 reconcile.
- `ed7549c` — fold G7.10 (evasion) into the unified G7 scoreboard.
- `a808514` — F17 evasion-reactive sweep results + plotter (G7.10).
- `7e70d7b` — Phase-D ablation results (F9/F10/F12/F15/F16) + closer fixes.
- `390c9e0` — Phase-D budget=40 re-train + benchmark.
- `727e21f` — budget calibration HARD GATE: `attacker_budget=40` LOCKED.
- `cda1ccc` / `8221c22` / `a3eba2a` — reward_mode ablation / evasion-before-commit
  / finite attacker budget mechanics.
- `321eeb8` / `6b3c251` — delete LSTM modules / rewire env to `MarkovAttacker`.
