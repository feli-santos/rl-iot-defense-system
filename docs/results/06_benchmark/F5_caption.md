# F5 — Final Security Metrics on `test_balanced`

> **Final security metrics on the held-out `test_balanced` split.**
> Trained RL (DQN / PPO / A2C, 5 seeds × 30 deterministic episodes /
> seed, n=150 each) vs. four non-RL baselines: random policy seeded
> 5 ways (n=150); deterministic always-OBSERVE / always-BLOCK /
> recommended-action / RF-Acting at single seed × 150 episodes (D6.3).
> The **recommended-action** rule baseline ⓞ achieves the best mean
> reward (+1624) — but it has free oracle access to
> `info["attack_stage"]`, so it is the *upper bound on the value of
> perfect stage detection*, not a deployable competing baseline
> (audit AF2 / Step-6 F2 reframe; Step-8 doc-fix). The best
> deployable agent (DQN, +1336) captures **82 % of the oracle
> ceiling** without ever seeing a stage; the remaining +288-reward
> gap is the **Phase-7 target** (D6.2.1) — Phase 7 closed 71 % of
> it via `impact_is_terminal=False` (PPO +1542 / mit-rate 0.900).
> See RESULTS.md §6.1 for the full reframe.
> All policies fully reach IMPACT under the upper-triangular Phase-2
> LSTM (compromise rate = 100 %); they differ on the **proportion of
> episodes that ended in `impact_mitigated`** (i.e., the agent
> isolated the host *during* the compromise) — `always-BLOCK` is the
> only policy that mitigates IMPACT in 100 % of episodes (because
> ISOLATE is one step beyond BLOCK on the force continuum, and
> always-BLOCK gets there reliably; the cost penalty is
> reflected in its lower mean reward). MTTC is statistically
> indistinguishable across policies (~19 steps), confirming that all
> Phase-2 episodes follow the same upper-triangular generator and
> that the differentiation lives in the per-step reward shaping, not
> in the time-to-compromise. Mean episode length ~20.7 steps matches
> the Phase-3 `min_episode_length=20` floor. Inference latency is
> well within the 5 ms / step RL budget for every trained algo
> (p50 ≤ 0.11 ms); rule-based policies clock under 0.003 ms; the RF-
> Acting baseline at p50 ≈ 14 ms exceeds the 3 ms budget set in G6.4
> (see F7 for the full distribution and RESULTS §5 for the budget
> revision).
>
> *Reproducibility:* every input JSONL and the upstream
> `runs/phase6/eval_manifest.json` are SHA-256 hash-pinned in
> `F5_manifest.json`; the producing git SHA is recorded there. No
> retraining was performed — checkpoints are the Phase-5
> `runs/phase5/<algo>/seed_<k>/model.zip` artefacts evaluated
> deterministically on `test_balanced`.

**Files:**
- `F5_summary.json` — machine-readable per-policy metrics + bootstrap CIs.
- `F5_summary.md`   — Markdown table for the thesis chapter.
- `F5_summary.csv`  — flat per-row metrics.
- `F5_table.png`    — rendered table (best row highlighted, divider
                      between trained-RL and non-RL baselines).
- `F5_manifest.json` — input/output SHA-256 chain + git SHA.
