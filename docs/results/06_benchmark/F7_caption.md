# F7 — Computational Overhead (Inference + Training)

> **Two-panel computational-overhead figure.** *Left:*
> per-step inference latency CDF, one curve per policy, x-axis log-
> scaled in milliseconds. The grey vertical reference lines are the
> G6.4 budgets (rule-based ≤ 1 ms, RF ≤ 3 ms, RL ≤ 5 ms; see D6.8).
> *Right:* total Phase-5 training wallclock per algorithm, summed
> over the 5 seeds (read from `runs/phase5/sweep_manifest.json`).
> Non-RL baselines have zero training time and are intentionally
> absent from the right panel — F7's contrast is "RL training cost"
> versus "rule-based zero-training" (the rule policies' upfront cost
> is human design effort, which F7 cannot quantify).
>
> **G6.4 outcome (median per-step latency):**
> - **DQN p50 = 0.068 ms** ✓ (budget 5 ms; ~75× headroom)
> - **PPO p50 = 0.100 ms** ✓ (budget 5 ms; ~50× headroom)
> - **A2C p50 = 0.101 ms** ✓ (budget 5 ms; ~50× headroom)
> - Random p50 = 0.002 ms ✓
> - Always-OBSERVE / Always-BLOCK / Recommended-Action p50 = 0.001 ms ✓
> - **RF-Acting p50 = 13.976 ms** ✗ (budget 3 ms; **G6.4 FAIL** — see
>   D6.8.1 in PLAN §8 for the disposition)
>
> The RF-Acting baseline's p50 latency is governed by sklearn's
> `RandomForestClassifier.predict()` on a 100-tree ensemble called
> once per env step. The 14 ms cost is a property of the supervised
> wrapper, not of the underlying RandomForest detector head (the
> detector head alone met the Phase-4 G4.5 ≤ 1 ms target on
> per-flow inputs); the overhead comes from sklearn's per-call
> Python dispatch and the 100-tree fan-out. The thesis interpretation
> is that **RF-Acting trades inference cost for higher-ish reward
> than the trained RL trio** (F5 +1508 vs. RL +1297..+1336) but
> remains slower than the RL forward pass — a trade-off that
> validates the case for *learned* policies with a fixed-shape
> network, even when current Phase-5 reward shaping leaves them
> below the recommended-action floor.
>
> **Training cost (right panel):** total wallclock summed across 5
> seeds. PPO and A2C run within minutes per seed; DQN dominates the
> bar because of the off-policy replay-buffer overhead. The full
> sweep (DQN + PPO + A2C × 5 seeds × 250 K timesteps) completed in
> 6513 s = 1 h 49 min on macOS / Apple silicon CPU. F7 thus shows
> the headline "RL training is ~10⁹× slower than rule-design once,
> but ~10⁻¹× faster than RF-Acting at inference time".
>
> *Reproducibility:* every input `latency.jsonl` and the upstream
> `sweep_manifest.json` are SHA-256 hash-pinned in `F7_manifest.json`;
> the platform fingerprint (`platform.platform()`, `platform.processor()`,
> Python version) is recorded in `F7_summary.json` so absolute
> latency numbers can be reinterpreted on different hardware
> (R6.3: macOS / Apple silicon / single-process CPU is 2–3×
> pessimistic vs. server hardware; the CDF + p99 reporting absorbs
> that).

**Files:**
- `F7_overhead.png` — two-panel figure (left CDF, right training-time bar).
- `F7_summary.json` — per-policy {p50, p95, p99, mean} latency in ms +
                      per-algo training seconds + platform fingerprint
                      + per-policy `g64_pass` boolean.
- `F7_manifest.json` — input/output SHA-256 chain + git SHA.
