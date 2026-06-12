# Results

Canonical experiment results at the locked `attacker_budget=40`. Numbers are
sourced from the gate-passing JSONs under `docs/results/<area>/` and rendered
into the thesis via `make render-tables` (never hand-typed in `tex/`). See
`docs/STATUS.md` for decisions/caveats and `docs/RESULTS_INDEX.md` (generated)
for the per-figure manifest index.

## Held-out benchmark (F5, `test_balanced`, 10 seeds × 300 episodes)

Source: `docs/results/benchmark/F5_summary.json`.

| Policy | Mean reward (95% CI) | p50 latency | benign FPR |
|---|---|---|---|
| **A2C** — best deployable RL | **+278.5** [+251.1, +308.8] | 0.094 ms | 0.66% |
| PPO | +274.5 | 0.094 ms | 0.89% |
| DQN | +267.8 | 0.063 ms | 0.46% |
| `recommended_action` oracle (ceiling) | **+543.1** [+536.6, +549.4] | — | — |
| RF-Acting (hero comparator) | +448.2 | 16.505 ms | — |
| always_observe | −393.2 | — | — |
| random | −573.9 | — | — |
| always_block | **−2005.1** (worst) | — | — |

- The three deployable DRL agents are **statistically tied** on reward (overlapping
  95% CIs); best-by-reward A2C captures **51.3%** of the oracle ceiling (PPO 50.5%,
  DQN 49.3%, RF-Acting 82.5%).
- The RL agents run at **~176× lower latency** than RF-Acting (~0.094 ms vs 16.505 ms
  p50; RF-Acting fails the 3 ms budget). This is an honest deployment trade-off:
  RF-Acting earns higher reward but is detector-coupled and slow; the RL agents are
  detector-free and sub-0.1 ms.
- Benign FPR is now **below 1% for all three RL agents** (DQN 0.46%, A2C 0.66%,
  PPO 0.89%); only `random` (41.3%) and `always_block` (100%) breach the 1% threshold.
- **`prevention_rate` is the primary security KPI** (oracle 1.00, A2C 0.60, DQN 0.54,
  PPO 0.33); `mitigated_impact_rate` is retired. `compromise_rate` varies by policy
  (A2C 0.403, DQN 0.463, PPO 0.67, oracle 0.00, always_block 0.00). `always_block`
  prevents every compromise but by indiscriminate force, so it is the worst-scoring
  policy on reward.

## Ablation & Robustness gate scoreboard (G7)

Source: `docs/results/ablation/G7_scoreboard.json` — **8 PASS / 2
FAIL-WITH-FINDING / 0 FAIL across G7.1–G7.10**. Both FAIL-WITH-FINDINGs are
pre-registered (G7.2 → D7.1.1, G7.4 → R7.3):

| Gate | Result | Note |
|---|---|---|
| G7.2 | FAIL-WITH-FINDING (D7.1.1) | reward-shaping raw ceiling; security-KPI strand passes (F9 structural mit-rate 0.850 @ +278.5 vs 0.0 for the mis-specified baseline) |
| G7.3 | PASS | environment-difficulty (`p_down`) sweep: PPO reward monotone, −111.1 @ p_down=0.0 → +145.4 @ p_down=1.0 (headline p_down=0.90) |
| G7.4 | FAIL-WITH-FINDING (R7.3) | Pareto frontier collapses to a single dominant point under perfect perception (F12): the oracle dominates at (security_gain=1.0, availability_cost=0.0); interior RL placement quantifies the cost of partial observability. Budget pivot peaks @ budget=40 (F16) PASS |
| G7.5–G7.7 | PASS | manifest hash-chains present + SHA-pinned |
| G7.8 | PASS | OOD coverage 32/32 |
| G7.9 | **PASS** | VulnerabilityScan OOD detector-independence dividend: detector-free RL wins outright. Detector-free PPO **+298.3** vs detector-coupled RF-Acting **−4430.6** (Δ **+4728.9**); RF's blind detector (recall ~0.000) systematically under-forces and the attacker advances unchecked |
| G7.10 | PASS | evasion-before-commit robustness (F17): reward degrades gracefully (PPO +271.6 @ evasion 0.0 → +270.7 @ evasion 0.75) |

## Figure map (thesis F-IDs → source)

| F-ID | Figure | Area |
|---|---|---|
| F3 | Blue-team learning curves (DQN/PPO/A2C × 10 seeds) | blue-team-training |
| F4 | Per-stage action-distribution evolution (PPO) | blue-team-training |
| F5 | Security-metrics table (`test_balanced`) | benchmark |
| F6 | Per-policy stage×action confusion matrices | benchmark |
| F7 | Inference-latency CDFs + train time | benchmark |
| F8 | RL vs non-RL baselines (reward + CIs) | benchmark |
| F9 | Reward-component ablation (12-cell sweep) | ablation |
| F10 | Environment-difficulty sweep (p_down) | ablation |
| F12 | Pareto frontier | ablation |
| F15 | Single-stage OOD-feature injection | ablation |
| F16 | Prevention-vs-budget sweep (prevention spine) | ablation |
| F17 | Evasion-before-commit sweep | ablation |

F-named PDFs are exported from the committed PNGs via `make export-figure-pdfs`
and staged into `tex/figs/` via `make sync-figures`. Provenance for every figure
is recorded in a sibling `F*_manifest.json` (git SHA + input/output SHA-256),
verified end-to-end by `python -m scripts.reproducibility_smoke`.
