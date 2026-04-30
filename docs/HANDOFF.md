# Handoff prompt — RL IoT Defense thesis (CICIoT2023)

> **Hand this whole document to the next coding agent (or future-you).** Below
> the metadata block there is a single prompt that tells the agent how to
> orient itself, how to verify the prior phases are real, and how to pick up
> at Phase 7. The prompt is self-contained: an agent following it should
> need nothing other than this file, the repo, and CPU.

---

## Quick metadata (for the human reading this)

| | |
|---|---|
| **Repo** | `git@github.com-personal:feli-santos/rl-iot-defense-system.git` |
| **Branch** | `feature/reward-shaping` (40+ commits ahead of `origin/...` at handoff) |
| **Last commit at handoff** | `d3e8ae1` (`docs(phase-6): close — RESULTS + CHANGELOG + G6 scoreboard`) |
| **Total tests** | 420/420 passing |
| **Phases complete** | 0, 1, 2, 3, 4, 5, **6** |
| **Next phase** | **Phase 7 — Ablations: reward components + attack aggressiveness + Pareto** (F9 + F10 + F12). The optional `impact_is_terminal` env-config flag deferred from Phase 6 (D6.6) is bundled into Phase 7's reward-shaping ablation. |
| **Inspiring papers** | `docs/papers/IoTWarden- A Deep Reinforcement Learning Based Real-time Defense System...pdf` (especially Fig. 6 for attack-aggressiveness sensitivity) |
| **Thesis-figure map** | `docs/thesis_results_map.md` (F0..F14, T1) — Phase 7 owns F9, F10, F12 |
| **Reproducibility** | every figure has a sibling `manifest.json` with SHA-256 hash chain pinned to the producing git SHA |

### Phase-6 headline numbers (just shipped)

Final ranking by mean episodic reward on the **held-out `test_balanced`**
split (D6.2 — first use of this split for blue-team metrics; n = 150
deterministic episodes per policy, 5 seeds × 30 ep for RL + random,
1 seed × 150 ep for the deterministic baselines):

| # | Policy                          | Mean reward |    95 % CI    | Cluster        |
|---|---------------------------------|------------:|---------------|----------------|
| 1 | **Recommended-Action (rule)** ★ |    **+1624** | (1572, 1672)  | supervised+rules |
| 2 | RF-Acting (supervised + rules)  |       +1508 | (1455, 1565)  | supervised+rules |
| 3 | DQN                             |       +1336 | (1265, 1407)  | trained-RL     |
| 4 | PPO                             |       +1313 | (1253, 1372)  | trained-RL     |
| 5 | A2C                             |       +1297 | (1267, 1337)  | trained-RL     |
| 6 | Always-BLOCK                    |        +520 | (483, 554)    | non-RL floor   |
| 7 | Random                          |        +390 | (384, 398)    | non-RL floor   |
| 8 | Always-OBSERVE                  |        −418 | (−421, −415)  | non-RL floor   |

★ = best by mean reward (D6.10 tie-break: lower p95 latency).

### Why Phase 6 ended `FAIL-WITH-FINDING` on G6.2 — *the most important context for Phase 7*

The **single most important Phase-6 finding (D6.2.1)**: on `test_balanced`,
the IoTWarden recommended-action rule baseline **+1624** strictly dominates
every trained RL algorithm (DQN +1336, PPO +1313, A2C +1297). Bootstrap CIs
do not overlap (DQN max 1407 < rec-action min 1572) — the gap is
statistically real.

Phase-5's "trained RL beats baseline by ~25×" headline was a **val-split
selection-bias artefact**: the trained agents converged on a de-escalation-
farming strategy (~6.30 de-escalations/episode × +250 = +1575/episode in
the reward signal) that scored well in-distribution but does not generalise.

Critically, **every other Phase-6 gate passed**:

- **G6.3 PASS:** trained agents *do* learn proportional behaviour on
  non-IMPACT stages (DQN 0.785, PPO 0.712, A2C 0.746; threshold 0.70).
  They learned the right local policy; they just optimised the wrong
  global objective on the IMPACT row.
- **G6.4 PASS-WITH-FINDING:** RL inference latency 0.07–0.10 ms (≥30×
  headroom on the 5 ms budget); RF-Acting 14 ms ✗ (D6.8.1; sklearn per-
  call dispatch property, not a real budget violation — production
  would batch / compile).
- **G6.5 PASS:** every RL CI is statistically separated from every
  non-RL CI (in *both* directions; the rule floor sits clearly above
  the RL cluster).
- **G6.6 PASS:** zero regression on Phase-3 / 4 / 5 frozen tests.
- **G6.7 PASS:** SHA-256 hash chain on every figure manifest.

The thesis chapter therefore reframes from "RL beats baselines by 25×" to:

> *"DQN/PPO/A2C all dominate the random-policy and always-OBSERVE
> baselines by ≥3.3× on the held-out test split, but are dominated in
> turn by the IoTWarden hand-crafted recommended-action rule baseline.
> We identify the gap as a Phase-3 reward-shaping artefact (the
> de-escalation bonus rewards a strategy that scores well
> in-distribution but does not generalise) and motivate the Phase-7
> reward-component ablation as the remediation."*

— more defensible because (a) the gap is precisely characterised, (b) the
remediation is already scoped (Phase 7), and (c) the result is consistent
with everything Phase-5 G5.4 already said.

### Cross-quadrant trade-off (Phase 7 must close it)

| Policy class       | Reward (test) | Latency (p50) |
|--------------------|--------------:|--------------:|
| Recommended-Action |    +1624 ★    | 0.001 ms      |
| RF-Acting          |    +1508      | 14.0 ms ✗     |
| Trained RL trio    |  +1297..+1336 | 0.10 ms       |
| Random             |    +390       | 0.002 ms      |
| Always-OBSERVE     |    −418       | 0.001 ms      |

— RF-Acting wins reward but loses inference cost; RL wins inference cost
but loses reward; **Phase 7's job is to *get both*** (RL-grade latency +
supervised-grade test-split reward).

### Phase-7 inputs already on disk

- 15 trained Phase-5 model checkpoints at `runs/phase5/<algo>/seed_<k>/model.zip`
  (gitignored; re-run `make phase-5-sweep` if absent).
- Phase-6 baseline rollouts at `runs/phase6/<policy>/seed_<k>/{eval_test,latency}.jsonl`
  (also gitignored; re-run `make phase-6` if absent).
- Phase-6 hash-pinned manifests in `docs/results/06_benchmark/F{5,6,7,8}_manifest.json`.
- The `recommended-action` rule baseline mean reward on `test_balanced`
  (+1624 ± 50) is the **target** Phase 7 must close the gap to.

---

## The prompt — paste this verbatim into the next agent

```
You are taking over as mentor/engineer on a Master's thesis project at
/Users/felipe.santos/Projects/rl-iot-defense-system on the
`feature/reward-shaping` branch. The thesis is an extension of IoTWarden
(Bhattacharjee et al. 2023, see docs/papers/) using the CICIoT2023
dataset. Seven phases (0-6) are complete; you are picking up at Phase 7
(Ablations — F9 reward-component sweep + F10 attack-aggressiveness
sweep + F12 security-vs-availability Pareto).

Your operating principles, learned from this project's history, are:

  1. Audit-first: read the relevant code and the IoTWarden paper Fig. 6
     before writing any new code, and write a PLAN.md for each phase
     BEFORE touching code. PLAN.md must contain (a) the audit findings,
     (b) the deliverables, (c) the exit gates, (d) a sequencing table,
     (e) what we are NOT doing, (f) the risks tracked. The PLAN goes
     through a "lock decisions" commit before any implementation.

  2. Empirical gates: every phase has named exit gates G<phase>.<i>
     with explicit numerical thresholds. Run them on real data before
     calling the phase done. When a gate fails, treat the failure as
     diagnostic — it usually means the gate or the design has a hole,
     not that the phase is doomed. Phases 3, 4, 5, AND 6 all turned
     "FAIL" into thesis-credible findings via dated D-decisions in
     PLAN §8 (B1-B6, D2.1, D5.4.1, D6.2.1, D6.8.1).

  3. Hash-chain everything: each thesis figure ships with a
     manifest.json that lists SHA-256 of the inputs and outputs and
     the producing git SHA. This is what lets the defense narrative
     stay reproducible.

  4. Honest commit history: when a prior phase's bug or
     selection-bias artefact is discovered mid-phase, fix it as a
     `fix(phase-<N>):` commit attributed to the discovering phase,
     document it in the discovering phase's RESULTS.md §5, and decide
     *consciously* whether to rebuild downstream artefacts. (Phase 4
     did this for the Phase-1 OOD leakage; Phase 6 surfaced the
     Phase-5 val-split bias as D6.2.1 without modifying Phase 5's
     numbers — it just reframed them.)

  5. Mentor-mode communication: brief, direct, lead with the result.
     Cite numbers, paper figures, and gate IDs by name. Don't bury
     the lede.

=== STEP 0: Acclimate ===

Read these in order, in full:
  - docs/HANDOFF.md (this file)
  - CHANGELOG.md (top to bottom — seven [Unreleased] sections, one
    per completed phase, with gate scoreboards and findings)
  - docs/thesis_results_map.md (figure -> phase mapping; Phase 7 is
    F9 / F10 / F12, Phase 8 is F13 / F14)
  - docs/results/00_phase0_diagnosis.md (the original failure mode
    that motivated the entire restart)
  - docs/results/01_dataset/ (Phase 1 — dataset card + F0 figures)
  - docs/results/02_red_team/PLAN.md and RESULTS.md (Phase 2 LSTM)
  - docs/results/03_env/PLAN.md and RESULTS.md (Phase 3 env v2 —
    PARTICULARLY §3 + §8 because Phase 7's reward-component sweep
    edits exactly the constants Phase 3 calibrated)
  - docs/results/04_detector/PLAN.md and RESULTS.md (Phase 4 F11)
  - docs/results/05_blue_team/PLAN.md and RESULTS.md (Phase 5 F3/F4/T1)
    — and `G5_scoreboard.json` for the per-gate result.
  - docs/results/06_benchmark/PLAN.md and RESULTS.md (Phase 6 RL
    benchmark; READ §6 in full — that is the thesis's headline
    finding) — and `G6_scoreboard.json`.

You DO need to (re-)read the IoTWarden paper for Phase 7 — specifically
their **Fig. 6 (sensitivity to attack aggressiveness)**, which is what
F10 reproduces / extends on the CICIoT2023 env. You probably do NOT
need to read it for Phase 8 (robustness is novel to this thesis).

Then verify the project is in the state CHANGELOG claims:

  cd /Users/felipe.santos/Projects/rl-iot-defense-system
  git log --oneline -25
  source .venv/bin/activate
  pytest -q                                        # expect 420 passed
  ls docs/results/06_benchmark/F5_table.png        # exists
  ls docs/results/06_benchmark/F6_stage_action_cm.png   # exists
  ls docs/results/06_benchmark/F7_overhead.png     # exists
  ls docs/results/06_benchmark/F8_baselines.png    # exists
  cat docs/results/06_benchmark/G6_scoreboard.json | jq '.gates'
  ls runs/phase6/   # if missing, runs/ is gitignored; re-run `make phase-6`
  ls runs/phase5/   # if missing, re-run `make phase-5-sweep`

Read the G6 scoreboard JSON in full. Confirm gate statuses match the
RESULTS.md §2 scoreboard. If anything disagrees, STOP and surface the
discrepancy before proceeding.

If `runs/phase5/` is missing on this machine (gitignored), the Phase-5
sweep needs to be re-run for Phase 7 to retrain agents under different
reward shapings:

  make phase-5-sweep PHASE5_TIMESTEPS=250000   # ~108 minutes CPU

If `runs/phase6/` is missing, regenerate it (Phase 7 may not need it
directly — depends on whether you fold Phase-6's `recommended-action`
floor into F9 as a reference line):

  make phase-6                                 # ~10 min CPU + figures

=== STEP 1: Phase 7 audit (no code yet) ===

Before any code, read:

  - src/environment/adversarial_env.py  (Phase-3 frozen contract;
    pay attention to AdversarialEnvConfig fields: every reward
    coefficient + p_defender_deescalation + max_steps + min_episode_length)
  - src/environment/__init__.py
  - src/blue_team/{callbacks.py, env_factory.py, run_config.py,
    aggregation.py}      (Phase-5 substrate; eval_runner from Phase 6
    can be reused too)
  - src/benchmark/{baseline_policies.py, eval_runner.py, latency.py}
    (Phase-6 substrate; the rollout harness already accepts any
    Policy callable, so Phase-7-trained agents drop in unchanged)
  - scripts/blue_team/{train_agent.py, run_phase5.py}  (the training
    entrypoint and the algo-x-seed sweep driver — Phase 7 may need
    a parameterised variant that takes reward overrides)
  - scripts/benchmark/{run_test_eval.py, build_summary_table.py}
    (Phase 6's evaluation pipeline; F9 will reuse the F5-builder
    aggregation almost verbatim, just with extra rows per
    reward-config)
  - tests/test_phase3_env_gates.py      (Phase-3 contract under test)
  - docs/results/03_env/PLAN.md §8.B6   (the calibration argument for
    `defense_success_bonus = 250` — this is what Phase 7 sweeps)
  - docs/results/05_blue_team/PLAN.md §8.D5.4.1
    (the de-escalation-farming finding — Phase 7's diagnostic target)
  - docs/results/06_benchmark/PLAN.md §8.D6.2.1 + D6.8.1
    (the test-split selection-bias finding + RF-Acting latency
    finding — Phase 7 must explain why its reward sweep closes the
    rule-baseline gap on test_balanced)
  - The IoTWarden paper, Fig. 6 caption — that is what F10
    reproduces.

The Phase-3 env's `info` dict already carries everything Phase-7
ablations need (`defender_deescalations`, `compromised`, `mttc_steps`,
etc.). The Phase-3 env's reward function `_calculate_reward` and the
terminal IMPACT path `_step_at_impact` are the **only places** Phase 7
should touch in `src/environment/`. The frozen-contract tests in
`tests/test_phase3_env_gates.py` and `test_adversarial_env.py` will
catch any change to the obs / action / step contract — if they fail,
fix the change, not the tests (G7.6 will mirror G6.6 exactly).

The thesis-results map (`docs/thesis_results_map.md`) lists for Phase 7:

  - F9 — Reward-component ablation. Sweep at least
         {defense_success_bonus, penalty_missed_impact,
         reward_proportional, penalty_disproportionate,
         reward_benign_passive} on a small grid (3-5 levels each
         won't fit; pick a sparse design — e.g., one-at-a-time at
         {0.5×, 1×, 2×} of the Phase-3 default). Plot mean test
         reward (and mitigated_impact_rate) vs. each component;
         the headline is "which component drives the
         test-split gap to the rule baseline closed?". This figure
         must include the **+1624 recommended-action floor** as a
         horizontal reference line.

  - F10 — Sensitivity to attack aggressiveness. Sweep
         `p_defender_deescalation` ∈ {0.0, 0.2, 0.4, 0.6 (default),
         0.8, 1.0} (or similar) and re-train one algorithm (PPO is
         the obvious pick — best Phase-5 reward) at each level on
         5 seeds. Plot mean test reward vs. aggressiveness for the
         trained policy AND for the recommended-action rule.
         Aligned with IoTWarden Fig. 6.

  - F12 — Security-vs-availability Pareto. For each
         (reward_config, p_defender_deescalation) point produced
         by F9 and F10, plot {security gain (mitigated_impact_rate
         or 1 − compromise_rate), availability cost (action_cost
         or BLOCK+ISOLATE share)} as a 2-D scatter. The Pareto
         frontier is the thesis's "operating-point chooser": every
         point on it represents a viable defender preference. This
         is the thesis's policy-design contribution.

  - **Optional (D6.6 deferred):** the `impact_is_terminal: bool`
     env-config flag. When False, the env transitions to IMPACT
     but does NOT terminate the same step; `_step_at_impact` runs
     on the next step instead, giving the agent an explicit IMPACT
     decision. Default True to preserve Phase-5/6 frozen contract.
     Phase 7 sweeps this as one axis of the F9 grid.

Cross-reference: Phase-6's `runs/phase6/eval_manifest.json` already
hash-pins the Phase-5 model.zips + RF model + scaler + splits manifest;
Phase 7's manifests must do the same for any new sweep results.

=== STEP 2: Write docs/results/07_ablation/PLAN.md ===

Cover, in this order:

  §1 Why Phase 7 exists (one paragraph; lead with D6.2.1 — the
     recommended-action floor of +1624 is what Phase 7 must close
     the gap to. Cite the thesis claim it supports — "we identify
     the precise reward-shaping component that, when retuned,
     closes the generalisation gap between Phase-3 in-distribution
     training reward and held-out test reward, and we plot the
     security-vs-availability Pareto frontier produced by the
     resulting policy family").

  §2 Audit findings:
       - what code already exists from Phases 3-6 (a lot — every
         training piece, eval piece, aggregation piece exists; only
         the *grid driver* is new)
       - what gaps remain (no `scripts/ablation/` directory; no
         parameterised env-config override for the training entry-
         point — you'll need to add a `--reward-overrides` JSON arg
         to `scripts/blue_team/train_agent.py`)
       - whether to fold the optional `impact_is_terminal` flag
         into Phase 7 (recommended: yes, as one axis of F9 — see
         D6.6 hand-off in 06_benchmark PLAN §8)

  §3 Concrete deliverables:
       3.1 code:
            - Add `impact_is_terminal: bool = True` to
              `AdversarialEnvConfig`. Default preserves the Phase-3
              frozen contract; new `impact_is_terminal=False` codepath
              re-uses `_step_at_impact` on the next step, NOT inline
              with `_calculate_reward`.
            - Parameterise `scripts/blue_team/train_agent.py` with a
              `--reward-overrides` JSON argument that overrides
              specific `AdversarialEnvConfig` fields without forking
              the script.
            - `scripts/ablation/__init__.py`
            - `scripts/ablation/run_reward_sweep.py` — driver for
              F9. Specifies a small grid (one-at-a-time around the
              Phase-3 defaults at {0.5×, 1×, 2×}) and fans out
              `train_agent.py` with the right `--reward-overrides`
              JSON per cell. PPO only (best from Phase 5 / Phase 6),
              3-5 seeds per cell. Total compute budget ≈ 3-6 h CPU.
            - `scripts/ablation/run_aggressiveness_sweep.py` — driver
              for F10. Sweeps `p_defender_deescalation` at
              {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} × 5 seeds × PPO. Total
              compute ≈ 1-2 h CPU.
            - `scripts/ablation/plot_reward_ablation.py` (F9) —
              consumes runs/phase7/reward_sweep/<cell>/seed_*/eval_test.jsonl
              and renders the per-component effect plot with the
              +1624 rule floor as a horizontal reference.
            - `scripts/ablation/plot_aggressiveness.py` (F10) —
              renders sensitivity curve aligned with IoTWarden Fig. 6.
            - `scripts/ablation/plot_pareto.py` (F12) — Pareto-
              frontier scatter on (security, availability) using
              every (reward_config, aggressiveness) point.

       3.2 OPTIONAL: re-run Phase-6 evaluations on the best
           Phase-7 reward-config so the closing chapter shows
           "with the Phase-7 reward retuning, RL meets / exceeds
           the recommended-action rule baseline at +X". Lock as
           optional — only ship if F9 actually finds a winning
           config; otherwise the thesis's honest finding stays
           "we identified the gap, characterised it, and the
           sweep failed to close it" (which is also defensible).

       3.3 tests (synthetic-only):
            - `tests/test_phase31_impact_terminal.py` — pin the
              `impact_is_terminal=False` codepath: env terminates
              one step LATER than with `impact_is_terminal=True`,
              the agent gets an explicit IMPACT-row decision, and
              the action histogram for that decision is captured
              in `action_counts_by_stage["4"]`.
            - `tests/test_train_agent_reward_overrides.py` — the
              `--reward-overrides` JSON arg is parsed correctly,
              hash-pinned in the run_manifest, and reflected in
              the env config used by SB3.
            - Run-driver tests are out of scope (real-data
              dependent).

       3.4 exit gates G7.1..G7.k (each with a numerical threshold).
            See "Recommended gates" below.

       3.5 figures produced (F9, F10, F12) with caption sketches.

  §4 Sequencing table (commits + estimated cost). Phase 7 is the
     single most expensive phase in compute terms because it
     re-trains. Estimated 6-8 commits, 1-2 days of work, 4-8 h CPU.

  §5 What we are NOT doing (defer to Phase 8):
       - OOD-class evaluation (Phase 8, F14).
       - Robustness to observation noise / drift (Phase 8, F13).
       - Re-training at full 500 K timesteps (Phase 5's D5.3.1
         locked us at 250 K; Phase 7 inherits that).
       - Re-implementing IoTWarden's DQN — already retired
         (`ecfb584`).
       - Hyperparameter sweeps within an algorithm — T1 in Phase 5
         already locked one config per algo.

  §6 Risks tracked (R1..Rk with mitigations). The big one is R7.1:
     "the reward sweep does not close the rule-baseline gap" — the
     thesis must be written so this outcome is *also* defensible
     (it characterises the limit of Phase-3-style reward shaping
     and motivates Phase-9-onwards work).

  §7 Cross-references to the thesis chapter outline. F9/F10/F12 all
     feed Chapter "Empirical Results" §6.5 (Ablations) + §6.6
     (Operating-Point Pareto).

  §8 Locked design decisions (after mentor sign-off).

Recommended Phase-7 exit gates to start the discussion (the user will
edit them):

  G7.1  Full pytest suite green (target ≥ 430+ tests). Phase-3 env
        contract tests still pass with `impact_is_terminal=True`
        (default).

  G7.2  F9 reward-component sweep produces at least one
        (reward_config) cell whose mean test-split reward, averaged
        over 5 seeds, exceeds the **Phase-6 trained-RL ceiling
        (+1336, DQN seed 0)** by at least 1 sigma of the per-cell
        bootstrap CI. Stretch goal: meet the +1624 recommended-action
        floor (D6.2.1). Acceptable failure mode (turns into a
        finding): the sweep does not close the gap; document why.

  G7.3  F10 attack-aggressiveness sweep produces a monotone-ish curve:
        as `p_defender_deescalation` decreases, mean test reward
        decreases for both PPO and the recommended-action rule.
        Threshold: PPO mean test reward at p=0.0 < PPO mean test
        reward at p=0.6, with at least 1 sigma separation.

  G7.4  F12 Pareto frontier has ≥ 3 distinct dominant points (no
        single config dominates all of {security, availability}).
        This validates the operating-point-choice contribution.

  G7.5  No regression on Phase-3 frozen tests when
        `impact_is_terminal=True` (the default). When False, a new
        test file (test_phase31_impact_terminal.py) covers the
        deferred codepath.

  G7.6  No regression on Phase-3/4/5/6 frozen tests overall —
        same hard-stop as G6.6.

  G7.7  Reproducibility — F9/F10/F12 each carry a manifest.json
        hash-pinning eval JSONLs + reward_config JSON per cell +
        git SHA.

The user's standing instructions for D-decisions:
  - The user has consistently said "I leave the decisions to you.
    What do you think is the best for my thesis defense/results?"
    Take that as the default for Phase 7 too. Make calls
    confidently, document them in PLAN §8 with rationale, and only
    ask the user when you genuinely need a value judgement, not
    when you can defend the call yourself.
  - Two judgement calls deserve explicit user sign-off before
    locking the PLAN: (a) whether to fold the
    `impact_is_terminal` flag into F9 or ship it as a standalone
    "Phase 7.1" with separate tests; (b) the size of the F9 grid
    (a 3-component × 3-level one-at-a-time sweep is 9 cells × 5
    seeds = 45 training runs ≈ 4 h CPU; a full 5-component × 3-
    level grid would be 81 × 5 = 405 ≈ 36 h CPU — the latter is
    out of scope unless the user explicitly approves the budget).

=== STEP 3: Lock the PLAN, then implement step-by-step ===

Same protocol as Phases 2-6: commit the PLAN, then implement one
substep per commit. After each substep, run pytest -q and verify
zero regressions on Phase-3/4/5/6 frozen tests.

Phase 7 DOES include re-training (the headline cost). Total wallclock
estimate (sparse F9 grid + F10 + figures):
  - F9 sweep: ~4 h CPU (9 cells × 5 seeds × ~6 min/seed at PPO 250 K)
  - F10 sweep: ~1.5 h CPU (6 levels × 5 seeds × ~3 min/seed at PPO 100 K
    if you use a shorter horizon for sensitivity; otherwise ~3 h)
  - Eval rollouts on test_balanced for every cell: ~30 min
  - Figure rendering: < 5 min
  - Total: 6-8 h CPU. Plan to start the sweep, walk away, come back.

=== STEP 4: Close Phase 7 ===

  - Run all gates G7.1..G7.k on real data.
  - Render F9 + F10 + F12 with manifests.
  - Write docs/results/07_ablation/RESULTS.md sister to PLAN.md.
  - Prepend a Phase-7 section to CHANGELOG.md.
  - Ensure pytest -q is green.
  - Tell the user the gate scoreboard, the headline numbers, and
    the findings worth defending. SPECIFICALLY state whether the
    reward sweep closed the +290-reward gap to the rule baseline,
    and if not, what the limit appears to be.

=== House rules ===

  - Always cd to the repo root before running commands.
  - Always use `source .venv/bin/activate` for python; bare
    `python` is not on PATH.
  - The processed dataset (`data/processed/ciciot2023/`),
    `runs/phase5/`, and `runs/phase6/` are gitignored and live only
    on the user's machine. Synthetic-data tests must NEVER depend on
    them. Real-data smoke tests should mark themselves
    `pytest.skipif(not Path('data/processed/...').exists(), ...)`.
  - When a gate "fails", first ask "is the gate or the
    implementation wrong?" Phases 3, 4, 5, AND 6 all had gates that
    turned out to be wrong on the first contact with reality, and
    updating the gate (with rationale captured in PLAN §8 D-decision)
    was the right move every time.
  - Commit messages follow conventional commits and cite the PLAN
    section being implemented (`feat(phase-7,§3.1.2): ...`).

=== If the user is asking you to *review*, not implement ===

If the user's first instruction after handoff is "review what was done
so far", do this and only this until they say "go":

  1. Read everything in STEP 0.
  2. Run `pytest -q` and confirm 420/420.
  3. Inspect F5 + F6 + F7 + F8 PNGs (open them) and the F5/F6/F7/F8
     summary JSONs.
  4. Walk back through commits 6eaafdc..d3e8ae1 with `git show
     --stat` to see what changed when. Confirm the *narrative* in
     06_benchmark/RESULTS.md matches the actual commits.
  5. Look for one of:
       (a) numerical mismatches between RESULTS docs and JSON files,
       (b) tests that exist but are skipped without good reason,
       (c) artefacts referenced in CHANGELOG but absent on disk,
       (d) commits that touched unrelated files,
       (e) regressions in older tests masked by newer ones,
       (f) Phase-6 D6.2.1's "FAIL-WITH-FINDING" interpretation —
           does the user agree that the rule-baseline-dominance
           result is thesis-defensible (reframed contribution),
           or does the gate failure genuinely block downstream
           work? My read: it's defensible AND it sharpens the
           Phase-7 target — but the user's defense committee
           may push back, so be ready with the argument.
       (g) Phase-6 D6.8.1's "PASS-WITH-FINDING" interpretation —
           is the 14 ms RF-Acting latency a fair comparison or
           an unfair handicap? Document either reading.
  6. Write a one-page audit to /tmp/handoff_audit.md and present it.
     Either green-light Phase 7 or list the blockers.

The user is preparing to defend a Master's thesis. Be honest. If
something is shaky, say so plainly with the evidence. The audit-first
protocol that has paid off four times (Phase 3 env bugs B1-B6,
Phase 4 discovering the Phase-1 OOD leakage, Phase 5 reframing
G5.3/G5.4 from the probe, Phase 6 reframing G6.2 from the held-out
split) only works if the *next* agent also follows it.
```

---

## Quick links for the human

- Final Phase-6 figures: `docs/results/06_benchmark/F5_table.png`,
  `F6_stage_action_cm.png`, `F7_overhead.png`, `F8_baselines.png`
- Phase-6 numbers: `docs/results/06_benchmark/F5_summary.json`,
  `F6_summary.json`, `F7_summary.json`, `F8_summary.json`,
  `G6_scoreboard.json`
- Phase-6 captions: `docs/results/06_benchmark/F{5,6,7,8}_caption.md`
- Phase-5 model checkpoints (Phase-7 will load these and retrain):
  `runs/phase5/<algo>/seed_<k>/model.zip` (gitignored — re-run
  `make phase-5-sweep` if missing)
- Phase-6 baseline rollouts:
  `runs/phase6/<policy>/seed_<k>/{eval_test,latency}.jsonl` (gitignored —
  re-run `make phase-6-eval` if missing)
- Most recent gate scoreboard: `CHANGELOG.md` top section (Phase 6)
- All phase RESULTS docs:
  `docs/results/{02_red_team,03_env,04_detector,05_blue_team,06_benchmark}/RESULTS.md`
- Test count history: 254 (Phase 0) → 266 (Phase 1) → 283 (Phase 2)
  → 296 (Phase 3) → 329 (Phase 4) → 376 (Phase 5) → **420 (Phase 6)**
- Figure-to-phase map: `docs/thesis_results_map.md`
  (Phase 7 → F9 / F10 / F12 ablations + Pareto;
   Phase 8 → F13 / F14 robustness)
