# Figure F2 — Empirical 5×5 transition matrix vs ground truth

**LaTeX caption (proposed).**
*Empirical Kill-Chain stage-transition probabilities recovered from
10 000 LSTM-generated episodes (centre) compared with the ground-truth
synthetic transition matrix (left) and their element-wise difference
(right, signed colour scale). Cell (i, j) is P(stage_{t+1}=j | stage_t=i).
Stages: 0 = BENIGN, 1 = RECON, 2 = ACCESS, 3 = MANEUVER, 4 = IMPACT.*

**What to look for.**
1. The two heatmaps are visually indistinguishable. Maximum absolute
   deviation across all 25 cells is **0.012** (1.2 percentage points),
   easily within sampling noise for 10 000 rollouts.
2. The IMPACT row stays at `T[4,4] ≈ 1.000` — the LSTM correctly learned
   the absorbing-state property of the IMPACT stage and never generates
   illegal back-transitions out of it.
3. Mean per-row KL divergence is **0.021** — about 2× tighter than the
   0.05 threshold from PLAN §3.2. The Red Team faithfully reproduces the
   intended attack-progression grammar.

**Why this matters.**
Phase 4 (the supervised stage detector) and Phase 5 (the RL Blue Team)
both consume episodes drawn from this LSTM. F2 is the formal evidence
that the Red Team is *grammatically faithful* — i.e. when downstream
phases evaluate the agent's response to attack progressions, those
progressions are statistically indistinguishable from the canonical
Kill-Chain progression.

**How it was generated.**
`PYTHONPATH=. python -m scripts.red_team.train_lstm --no-mlflow` produces
this figure together with F1, the JSON summary, and the manifest.
