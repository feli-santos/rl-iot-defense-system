# F6 — Stage × Action Decision Distribution per Policy

> **Per-policy stage × action heatmaps on `test_balanced`.** Rows
> are kill-chain stages (BENIGN → IMPACT); columns are defensive
> actions (OBSERVE → ISOLATE). Cell colour = empirical fraction of
> decisions at that stage that chose that action (rows sum to 1.0).
> The red box overlays the proportionality band
> `|action − recommended(stage)| ≤ 1`. The G6.3 score printed below
> each panel is the mean fraction of decisions inside the band,
> averaged over **non-IMPACT stages only** (per D6.7 — the IMPACT-
> stage de-escalation finding from G5.4 is documented as the
> Phase-7 hand-off, not silently relaxed here).
>
> **Result (G6.3 PASS for all trained-RL algos):**
> DQN 0.785, PPO 0.712, A2C 0.746 — all clear the 0.70 threshold.
> The trained agents *do* learn proportional behaviour on the four
> non-IMPACT stages; the per-stage rows on the top of each RL panel
> place the bulk of decision mass on or adjacent to the diagonal,
> with a recognisable "force-continuum" structure (BENIGN → mostly
> OBSERVE/LOG; ACCESS → THROTTLE; MANEUVER → BLOCK). The IMPACT row
> is where every RL algo deviates from the rule policy and chooses
> sub-ISOLATE actions, exposing the de-escalation-farming behaviour
> introduced in Phase-5 G5.4 (a non-trivial fraction of total reward
> in training came from the +250 de-escalation bonus rather than
> from successful IMPACT mitigation; see also D6.2.1 on why this
> doesn't translate to a test-split reward win).
>
> **Reference panels (non-RL):** Random, Always-OBSERVE, Always-BLOCK
> are included to anchor the heatmap visually — their G6.3 scores
> 0.50–0.55 are the floor for "any policy with information" against
> the band metric. The Recommended-Action and RF-Acting panels are
> deliberately omitted: by construction the recommended-action
> matrix is a perfect identity (G6.3 = 1.00) and would carry no
> information, while RF-Acting's matrix mirrors the RandomForest's
> per-stage recall (Phase-4 macro-F1 ≈ 0.79) on a different decision
> axis already covered by Phase-4 F11.
>
> *Reproducibility:* per-policy 5×5 matrices in `F6_summary.json`;
> input JSONLs SHA-256 hash-pinned in `F6_manifest.json`. Same
> input artefacts as F5 (`runs/phase6/eval_manifest.json`).

**Files:**
- `F6_stage_action_cm.png` — multi-panel figure (2×3, trained-RL on top).
- `F6_summary.json`        — per-policy 5×5 matrices + G6.3 scores +
                              boolean per-policy pass flags.
- `F6_manifest.json`       — input/output SHA-256 chain + git SHA.
