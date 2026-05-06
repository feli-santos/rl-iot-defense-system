# F8 — RL vs Non-RL Baselines (Mean Reward, 95 % Bootstrap CI)

> **Horizontal bar chart of per-policy mean episodic reward on
> `test_balanced` with 95 % bootstrap CIs**, sorted descending.
> Trained-RL bars in blue; non-RL baselines in grey. The red dashed
> line marks the **oracle Recommended-Action ceiling (+1624.4)** —
> the rule-based policy that has free access to the true attack
> stage (`info["attack_stage"]`) and applies the stage-action
> recommended mapping. It is **not a deployable defender**, but a
> measurement instrument quantifying the value of perfect stage
> detection (audit AF2; see `RESULTS.md` §6.1). Every CI annotation
> is rendered next to its bar.
>
> **G6.5 PASS for all three trained-RL algos:** DQN / PPO / A2C bars
> have CIs that do **not overlap** any non-RL baseline's CI; the gate
> is satisfied. Per D6.2.1, the *direction* of the separation is no
> longer constrained — F5 (and now F8) make the test-split rank order
> visible at a glance:
>
> 1. **Recommended-Action (rule)**     +1624 (1572, 1672) — best
> 2. **RF-Acting (supervised + rules)**+1508 (1455, 1565)
> 3. **DQN**                           +1336 (1265, 1407)
> 4. **PPO**                           +1313 (1253, 1372)
> 5. **A2C**                           +1297 (1267, 1337)
> 6. **Always-BLOCK**                  +520  (483,  554)
> 7. **Random**                        +390  (384,  398)
> 8. **Always-OBSERVE**                −418  (−421, −415) — worst
>
> Two takeaway clusters separate cleanly:
>
> - **The "supervised + rules" cluster (top two rows)** dominates by
>   ~+170 reward. Recommended-Action and RF-Acting both translate the
>   per-stage proportional mapping into action; they differ only on
>   whether the stage is read from `info["attack_stage"]` (oracle) or
>   from `RandomForest.predict(...)` (Phase-4 macro-F1 ≈ 0.79). The
>   ~+116 reward gap between the two quantifies the cost of trading
>   oracle stage knowledge for a learned classifier.
>
> - **The trained-RL cluster** sits ~+290 below the oracle ceiling
>   but ~+780 above the random-policy bar — the agents *did* learn
>   something useful (G6.3 PASS confirms proportional behaviour on
>   non-IMPACT stages), but Phase-3 reward shaping rewards de-
>   escalation farming (G5.4) more than it rewards true mitigation,
>   so the val-split optimum does not generalise. This is the
>   D6.2.1 finding rendered in one image.
>
> **Reproducibility:** F8 is generated *from* `F5_summary.json`
> (single source of truth for per-policy means + bootstrap CIs).
> `F8_manifest.json` hash-pins F5_summary.json and the upstream
> `runs/phase6/eval_manifest.json` plus the producing git SHA. No
> retraining occurred; checkpoints are the Phase-5 `runs/phase5/`
> artefacts evaluated deterministically on test_balanced.

**Files:**
- `F8_baselines.png` — horizontal bar chart with bootstrap-CI whiskers
                       and oracle recommended-action ceiling annotated.
- `F8_summary.json`  — sorted-desc per-policy {mean, ci_low, ci_high} +
                       per-RL-policy G6.5 pass/fail and overlap list.
- `F8_manifest.json` — input/output SHA-256 chain + git SHA.
