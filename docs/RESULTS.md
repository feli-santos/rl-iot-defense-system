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
| **PPO** — best deployable RL | **+1034.7** [+998.1, +1069.8] | 0.096 ms | 8.7% |
| DQN | +1028.9 | 0.061 ms | 7.5% |
| A2C | +973.1 | 0.094 ms | 7.7% |
| `recommended_action` oracle (ceiling) | **+1393.8** [+1366.9, +1420.6] | — | — |
| RF-Acting (hero comparator) | +1323.0 | 13.692 ms | — |
| always_block | −219.4 | — | — |
| random | +211.1 | — | — |
| always_observe | −416.1 | — | — |

- Best deployable RL (PPO) captures **74.2%** of the oracle ceiling.
- PPO runs at **~142.8× lower latency** than RF-Acting (0.096 ms vs 13.692 ms p50).
- Under the finite budget, `compromise_rate` varies by policy (PPO ~0.68,
  always_block ~0.36, oracle ~0.47) — no longer 1.0-for-everything.

## Ablation & Robustness gate scoreboard (G7)

Source: `docs/results/ablation/G7_scoreboard.json` — **8 PASS / 2
FAIL-WITH-FINDING / 0 FAIL across G7.1–G7.10**. Both FAILs are pre-registered:

| Gate | Result | Note |
|---|---|---|
| G7.2 | FAIL-WITH-FINDING (D7.1.1) | reward-shaping raw ceiling; security-KPI strand passes (F9 structural mit-rate 0.867 vs DQN 0.153) |
| G7.3 | PASS | aggressiveness sweep (IoTWarden-shaped, monotone in p_de_esc) |
| G7.4 | PASS | Pareto frontier + budget-hump peak @ budget=40 (F12/F16) |
| G7.5–G7.7 | PASS | manifest hash-chains present + SHA-pinned |
| G7.8 | PASS | OOD coverage 32/32 |
| G7.9 | FAIL-WITH-FINDING (D7.9.1) | VulnerabilityScan OOD: RL +1109.5 < RF-Acting +1443.4 (RF recall 0.001, cheap-default mapping wins). Claim narrows to "RL robust to, not better at" |
| G7.10 | PASS | evasion-before-commit robustness (F17): reward degrades gracefully 1018.4→982.0 across evasion 0→0.75 |

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
| F10 | Aggressiveness sweep (p_de_esc) | ablation |
| F12 | Pareto frontier | ablation |
| F15 | Single-stage OOD-feature injection | ablation |
| F16 | Prevention-vs-budget sweep (prevention spine) | ablation |
| F17 | Evasion-before-commit sweep | ablation |

F-named PDFs are exported from the committed PNGs via `make export-figure-pdfs`
and staged into `tex/figs/` via `make sync-figures`. Provenance for every figure
is recorded in a sibling `F*_manifest.json` (git SHA + input/output SHA-256),
verified end-to-end by `python -m scripts.reproducibility_smoke`.
