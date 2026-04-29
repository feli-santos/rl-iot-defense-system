# Handoff prompt — RL IoT Defense thesis (CICIoT2023)

> **Hand this whole document to the next coding agent (or future-you).** Below
> the metadata block there is a single prompt that tells the agent how to
> orient itself, how to verify the prior phases are real, and how to pick up
> at Phase 6. The prompt is self-contained: an agent following it should
> need nothing other than this file, the repo, and CPU.

---

## Quick metadata (for the human reading this)

| | |
|---|---|
| **Repo** | `git@github.com-personal:feli-santos/rl-iot-defense-system.git` |
| **Branch** | `feature/reward-shaping` (30+ commits ahead of `origin/...` at handoff) |
| **Last commit at handoff** | `<this commit>` (docs: retire IoTWarden head-to-head, scope Phase 6 as RL algorithm benchmark) |
| **Total tests** | 376/376 passing |
| **Phases complete** | 0, 1, 2, 3, 4, 5 |
| **Next phase** | **Phase 6 — RL Algorithm Benchmark** (F5 + F6 + F7 + F8). Optional Phase-3.1 patch (`impact_is_terminal` config flag) is bundled with the Phase-6 PLAN. |
| **Inspiring paper** | `docs/papers/IoTWarden- A Deep Reinforcement Learning Based Real-time Defense System...pdf` |
| **Thesis-figure map** | `docs/thesis_results_map.md` (F0..F14, T1) — UPDATED in this commit |
| **Reproducibility** | every figure has a sibling `manifest.json` with SHA-256 hash chain pinned to the producing git SHA |

### Phase-5 headline numbers (just shipped)

| Algo | Mean eval reward (last 10 % × 5 seeds) | Mean MTTC | Mitigated-impact rate |
|---|---:|---:|---:|
| **PPO** | **+1350.7** | 19.24 | 0.263 |
| A2C     | +1325.6     | 19.26 | 0.242 |
| DQN     | +1300.1     | 19.25 | 0.236 |

- **Recommended-policy floor** ~+50, so trained agents beat the IoTWarden hand-crafted baseline by **~25×**.
- 15 trained model checkpoints at `runs/phase5/<algo>/seed_<k>/model.zip`, ready for Phase 6 to consume.
- G5.4 PASS-WITH-FINDING: **the agent farms de-escalation bonuses (~6.30/episode × +250 = +1575) and accepts the IMPACT loss**. Phase 7's reward-component ablation will sweep this; Phase 6 is *not* expected to fix it.

### Why we **retired** the IoTWarden head-to-head as a phase

The earlier draft of Phase 6 was a faithful re-implementation of IoTWarden's
DQN configuration. The user (correctly) pointed out that:

1. IoTWarden's hparams are tuned to *their* env (trigger-action IoT, no LSTM
   kill-chain prior, different reward shaping). Re-running them on our
   Phase-3 env compares "their hparams + our env" vs "our hparams + our env",
   which is **not a like-for-like head-to-head**.
2. Phase 5 already cites the IoTWarden recommended-action policy as the
   baseline (the `info["recommended_action"]` field). The thesis can claim
   "we beat the IoTWarden recommended-action policy by 25× on the same env"
   without re-implementing their DQN.
3. The user's actual contribution is the **cross-algorithm benchmark**
   (DQN vs PPO vs A2C, plus rule-based and supervised baselines), which is
   what F5/F6/F7/F8 in `docs/thesis_results_map.md` already describe.

So Phase 6 is now scoped as the **RL Algorithm Benchmark**, producing
F5 + F6 + F7 + F8 from the existing Phase-5 checkpoints. No re-training of
the trio. (The phase-numbering downstream slides: ablation 8→7,
robustness 9→8.)

---

## The prompt — paste this verbatim into the next agent

```
You are taking over as mentor/engineer on a Master's thesis project at
/Users/felipe.santos/Projects/rl-iot-defense-system on the
`feature/reward-shaping` branch. The thesis is an extension of IoTWarden
(Bhattacharjee et al. 2023, see docs/papers/) using the CICIoT2023
dataset. Six phases (0-5) are complete; you are picking up at Phase 6
(RL Algorithm Benchmark — F5 + F6 + F7 + F8).

Your operating principles, learned from this project's history, are:

  1. Audit-first: read the relevant code and a paper section before
     writing any new code, and write a PLAN.md for each phase BEFORE
     touching code. PLAN.md must contain (a) the audit findings, (b)
     the deliverables, (c) the exit gates, (d) a sequencing table, (e)
     what we are NOT doing, (f) the risks tracked. The PLAN goes
     through a "lock decisions" commit before any implementation.

  2. Empirical gates: every phase has named exit gates G<phase>.<i>
     with explicit numerical thresholds. Run them on real data before
     calling the phase done. When a gate fails, treat the failure as
     diagnostic — it usually means the gate or the design has a hole,
     not that the phase is doomed. Phases 3, 4, AND 5 all turned
     "FAIL" into thesis-credible findings via dated D-decisions in
     PLAN §8 (B1-B6, D2.1, D5.4.1).

  3. Hash-chain everything: each thesis figure ships with a
     manifest.json that lists SHA-256 of the inputs and outputs and
     the producing git SHA. This is what lets the defense narrative
     stay reproducible.

  4. Honest commit history: when a prior phase's bug is discovered
     mid-phase, fix it as a `fix(phase-<N>):` commit attributed to
     the discovering phase, document it in the discovering phase's
     RESULTS.md §5, and decide *consciously* whether to rebuild
     downstream artefacts. (Phase 4 did this for the Phase-1 OOD
     leakage; Phase 5 surfaced the Phase-3 R2 reward-shaping
     interaction without modifying Phase 3.)

  5. Mentor-mode communication: brief, direct, lead with the result.
     Cite numbers, paper figures, and gate IDs by name. Don't bury
     the lede.

=== STEP 0: Acclimate ===

Read these in order, in full:
  - docs/HANDOFF.md (this file)
  - CHANGELOG.md (top to bottom — six [Unreleased] sections, one per
    completed phase, with gate scoreboards and findings)
  - docs/thesis_results_map.md (figure -> phase mapping, recently
    updated to retire IoTWarden head-to-head)
  - docs/results/00_phase0_diagnosis.md (the original failure mode
    that motivated the entire restart)
  - docs/results/01_dataset/ (Phase 1 — dataset card + F0 figures)
  - docs/results/02_red_team/PLAN.md and RESULTS.md (Phase 2 LSTM)
  - docs/results/03_env/PLAN.md and RESULTS.md (Phase 3 env v2)
  - docs/results/04_detector/PLAN.md and RESULTS.md (Phase 4 F11)
  - docs/results/05_blue_team/PLAN.md and RESULTS.md (Phase 5 F3/F4/T1)
    — and `G5_scoreboard.json` for the per-gate result.

You do NOT need to read the IoTWarden paper for Phase 6 (the thesis
already cites it as the inspiring paper; Phase 5 cites the
recommended-action policy as the rule-based baseline). You DO need to
re-read it for Phase 7 (sensitivity ablation aligned with their
Fig. 6) and possibly Phase 8 (robustness).

Then verify the project is in the state CHANGELOG claims:

  cd /Users/felipe.santos/Projects/rl-iot-defense-system
  git log --oneline -25
  source .venv/bin/activate
  pytest -q                                        # expect 376 passed
  ls docs/results/05_blue_team/F3_learning_curves.png   # exists
  cat docs/results/05_blue_team/G5_scoreboard.json | jq '.gates'
  ls runs/phase5/  # if missing, runs/ is gitignored — see Phase-5 RESULTS.md

Read the G5 scoreboard JSON in full. Confirm gate statuses match the
RESULTS.md §2 scoreboard. If anything disagrees, STOP and surface the
discrepancy before proceeding.

If `runs/phase5/` is missing on this machine (gitignored), the Phase-5
sweep needs to be re-run for Phase 6 to consume the model checkpoints:

  make phase-5-sweep PHASE5_TIMESTEPS=250000   # ~108 minutes CPU

=== STEP 1: Phase 6 audit (no code yet) ===

Before any code, read:

  - src/blue_team/{callbacks.py, env_factory.py, run_config.py,
    aggregation.py}      (Phase-5 substrate, ready to reuse)
  - scripts/blue_team/   (train_agent, run_phase5, evaluate_gates,
                          plot_learning_curves, plot_action_dist,
                          dump_hparams)
  - src/environment/adversarial_env.py  (Phase-3 frozen contract)
  - tests/test_phase3_env_gates.py      (Phase-3 contract under test)
  - docs/results/05_blue_team/PLAN.md §8  (D5.1..D5.11 + D5.3.1..D5.10.1)

The Phase-3 env's `info` dict carries: `compromised`, `mttc_steps`,
`first_attack_step`, `compromise_step`, `defender_deescalations`,
`recommended_action`, `attack_stage`, `last_action`, `outcome`.
The Phase-5 sweep produced 15 trained checkpoints at
`runs/phase5/<algo>/seed_<k>/model.zip` plus per-run JSONL logs.
Phase 6 should treat those as **frozen reference numbers** (do NOT
retrain).

The thesis-results map (`docs/thesis_results_map.md`) lists for Phase 6:

  - F5 — Final security metrics table. Per-(algo, baseline) summary of
         {mean reward, MTTC, compromise rate, mitigated-impact rate,
         per-action share, latency}. Baselines = random, always-OBSERVE,
         always-BLOCK, recommended-action, RandomForest detector
         (Phase-4 RF as a reference point).
  - F6 — Stage × Action confusion matrices, one 5×5 heatmap per algo.
         Tells the per-stage proportionality story visually.
  - F7 — Computation overhead. Latency CDF (per-step inference time)
         + training time per algo, aligned with IoTWarden Fig. 4(b).
  - F8 — Bar chart of mean reward (or mitigated-impact rate, your
         call): RL vs random vs always-OBSERVE vs always-BLOCK vs
         RandomForest-acting-policy vs recommended-action.

Cross-reference: Phase-5's `runs/phase5/sweep_manifest.json` already
hash-pins the input JSONLs; Phase 6's manifests must do the same.

=== STEP 2: Write docs/results/06_benchmark/PLAN.md ===

Cover, in this order:

  §1 Why Phase 6 exists (one paragraph; the user explicitly said the
     cross-algorithm benchmark is part of their contribution. Cite
     the thesis claim it supports — "we present the first head-to-head
     benchmark of model-free RL algorithms on the CICIoT2023 kill-chain
     defense problem").

  §2 Audit findings:
       - what code already exists from Phases 3-5 (a lot)
       - what gaps remain (the `scripts/benchmark/` directory does NOT
         exist yet; you must create it)
       - whether to extend Phase-5 model checkpoints with eval rollouts
         on `test_balanced` (not just `val_balanced`) — recommended.

  §3 Concrete deliverables:
       3.1 code:
            - `src/benchmark/baseline_policies.py` — wrap the rule-based
              policies (random, always-X, recommended-action) as SB3-
              compatible callables that emit the same JSONL schema as
              EpisodeJSONLCallback. The RandomForest detector wrapper
              is the only non-trivial one; it acts on each step's
              detector probability and applies a deterministic mapping
              from predicted stage to recommended action.
            - `scripts/benchmark/eval_runner.py` — given a list of
              (policy_name, callable) tuples, run N episodes per
              policy on the eval-split env, write the JSONL.
            - `scripts/benchmark/build_summary_table.py` (F5) —
              consume runs/phase5/<algo>/seed_*/eval.jsonl plus the
              new baseline JSONLs, emit T-shaped summary CSV/MD/PNG.
            - `scripts/benchmark/plot_stage_action_cm.py` (F6) — per
              algo, build 5×5 (decision_stage × action) confusion
              matrix from `action_counts_by_stage` and render.
            - `scripts/benchmark/plot_overhead.py` (F7) — measure
              per-step inference time per algo with timeit; plot
              latency CDF + bar of training-time-from-run_manifest.
            - `scripts/benchmark/plot_baselines.py` (F8) — bar chart
              comparing all policies on a single metric (default:
              mean eval reward; configurable).

       3.2 OPTIONAL — Phase-3.1 patch:
           Add `impact_is_terminal: bool = True` to
           `AdversarialEnvConfig`. When False, the env transitions
           to IMPACT but does NOT terminate the same step;
           `_step_at_impact` runs on the next step instead, giving
           the agent an explicit IMPACT decision. Default True to
           preserve Phase-5 frozen contract. New gate G6.X may
           sweep this for one of the four algos to study the
           credit-assignment effect (D5.4.1 + Finding-2 follow-up).

           This is locked as **optional** in Phase 6 — only ship if
           the rest of Phase 6 has time-budget left. Otherwise it
           moves to Phase 7 alongside the reward-component ablation.

       3.3 tests (synthetic-only):
            - `tests/test_baseline_policies.py` — random emits valid
              actions; always-X emits the right constant; recommended-
              action consults info["recommended_action"]; RF detector
              wrapper emits the right action given a known stage.
            - `tests/test_benchmark_eval_runner.py` — round-trip
              JSONL of a baseline policy.
            - `tests/test_phase31_impact_terminal.py` (only if §3.2
              patch is implemented).

       3.4 exit gates G6.1..G6.k (each with a numerical threshold).
            See "Recommended gates" below.

       3.5 figures produced (F5, F6, F7, F8) with caption sketches.

  §4 Sequencing table (commits + estimated cost). Phase 6 is much
     smaller than Phase 5 — no new sweep. Estimated 4-6 commits,
     1 day of work.

  §5 What we are NOT doing (defer to Phase 7 / Phase 8):
       - Reward-component sweep (Phase 7).
       - Attack-aggressiveness sweep (Phase 7).
       - OOD-class evaluation (Phase 8).
       - The Phase-3.1 `impact_is_terminal` flag becomes a reward-
         shaping ablation in Phase 7 if Phase 6 doesn't ship it.

  §6 Risks tracked (R1..Rk with mitigations).

  §7 Cross-references to the thesis chapter outline.

  §8 Locked design decisions (after mentor sign-off).

Recommended Phase-6 exit gates to start the discussion (the user will
edit them):

  G6.1  Full pytest suite green (target ~390+ tests).

  G6.2  F5 (security metrics table) reports each of {DQN, PPO, A2C,
        random, always-OBSERVE, always-BLOCK, recommended-action,
        RF-acting-policy} on (mean_reward, mean_mttc,
        compromise_rate, mitigated_impact_rate, mean_episode_length,
        mean_inference_latency_ms). Threshold: every cell is
        non-NaN; trained-RL row dominates rule-based row in
        mean_reward by >= 5x.

  G6.3  F6 (stage×action CM) shows per-algo proportionality on the
        diagonal ±1 band. Quantitative gate: for each algo, the
        sum of probability mass within the proportionality band
        (|action - recommended| <= 1) is >= 0.7 averaged over
        non-IMPACT stages.

  G6.4  F7 (overhead) reports inference latency <= 5 ms / step for
        all three RL algos and <= 1 ms / step for the rule-based
        baselines. (Phase-4 G4.5 was 1ms for the detector head
        alone; the SB3 model on top adds policy-network forward
        pass.)

  G6.5  F8 (bar chart): the trained-RL bars (DQN/PPO/A2C) clearly
        separate from the rule-based bars at 95% bootstrap CI.

  G6.6  No regression on Phase-3 / Phase-4 / Phase-5 frozen tests
        (test_phase3_env_gates.py, test_adversarial_env.py,
         test_blue_team_*.py, test_detector.py).

  G6.7  Reproducibility — F5/F6/F7/F8 each carry a manifest.json
        hash-pinning eval JSONLs + baseline JSONLs + git SHA.

Do NOT propose to retrain the Phase-5 trio, change the env, or
modify the Phase-1 splits. Phases 1/2/3/4/5 are frozen by contract.

The user's standing instructions for D-decisions:
  - The user has consistently said "I leave the decisions to you.
    What do you think is the best for my thesis defense/results?"
    Take that as the default for Phase 6 too. Make calls
    confidently, document them in PLAN §8 with rationale, and only
    ask the user when you genuinely need a value judgement, not
    when you can defend the call yourself.

=== STEP 3: Lock the PLAN, then implement step-by-step ===

Same protocol as Phases 2-5: commit the PLAN, then implement one
substep per commit. After each substep, run pytest -q and verify
zero regressions.

Phase 6 does NOT include a re-train. Total wallclock should be
< 30 min for evaluation + figure rendering, dominated by:
  - inference-latency benchmarking (the trickiest measurement)
  - rolling out the rule-based baselines for ~500 episodes each on
    the eval split.

=== STEP 4: Close Phase 6 ===

  - Run all gates G6.1..G6.k on real data.
  - Render F5 + F6 + F7 + F8 with manifests.
  - Write docs/results/06_benchmark/RESULTS.md sister to PLAN.md.
  - Prepend a Phase-6 section to CHANGELOG.md.
  - Ensure pytest -q is green.
  - Tell the user the gate scoreboard, the headline numbers, and the
    findings worth defending.

=== House rules ===

  - Always cd to the repo root before running commands.
  - Always use `source .venv/bin/activate` for python; bare
    `python` is not on PATH.
  - The processed dataset (`data/processed/ciciot2023/`) and
    `runs/phase5/` are gitignored and live only on the user's
    machine. Synthetic-data tests must NEVER depend on them. Real-data
    smoke tests should mark themselves
    `pytest.skipif(not Path('data/processed/...').exists(), ...)`.
  - When a gate "fails", first ask "is the gate or the
    implementation wrong?" Phases 3, 4, AND 5 all had gates that
    turned out to be wrong on the first contact with reality, and
    updating the gate (with rationale captured in PLAN §8) was the
    right move every time.
  - Commit messages follow conventional commits and cite the PLAN
    section being implemented (`feat(phase-6,§3.1.2): ...`).

=== If the user is asking you to *review*, not implement ===

If the user's first instruction after handoff is "review what was done
so far", do this and only this until they say "go":

  1. Read everything in STEP 0.
  2. Run `pytest -q` and confirm 376/376.
  3. Inspect F3 + F4 PNGs (open them) and the F3/F4 summary JSONs.
  4. Walk back through commits 9b70d7d..<latest> with `git show
     --stat` to see what changed when. Confirm the *narrative* in
     RESULTS.md matches the actual commits.
  5. Look for one of:
       (a) numerical mismatches between RESULTS docs and JSON files,
       (b) tests that exist but are skipped without good reason,
       (c) artefacts referenced in CHANGELOG but absent on disk,
       (d) commits that touched unrelated files,
       (e) regressions in older tests masked by newer ones,
       (f) Phase-5 D5.4.1's "PASS-with-finding" interpretation —
           does the user agree that the de-escalation-farming
           result is thesis-defensible, or does the gate failure
           genuinely block downstream work?
  6. Write a one-page audit to /tmp/handoff_audit.md and present it.
     Either green-light Phase 6 or list the blockers.

The user is preparing to defend a Master's thesis. Be honest. If
something is shaky, say so plainly with the evidence. The audit-first
protocol that has paid off three times (Phase 3 env bugs B1-B6,
Phase 4 discovering the Phase-1 OOD leakage, Phase 5 reframing
G5.3/G5.4 from the probe) only works if the *next* agent also
follows it.
```

---

## Quick links for the human

- Final Phase-5 figures: `docs/results/05_blue_team/F3_learning_curves.png`, `F4_action_distribution.png`
- Phase-5 numbers: `docs/results/05_blue_team/F3_summary.json`, `F4_summary.json`, `G5_scoreboard.json`
- Phase-5 hparams: `docs/results/05_blue_team/T1_hparams.md` (markdown) and `T1_hparams.json`
- Phase-5 model checkpoints: `runs/phase5/<algo>/seed_<k>/model.zip` (gitignored — re-run `make phase-5-sweep` if missing)
- Most recent gate scoreboard: `CHANGELOG.md` top section
- All phase RESULTS docs: `docs/results/{02_red_team,03_env,04_detector,05_blue_team}/RESULTS.md`
- Test count history: 254 (Phase 0) → 266 (Phase 1) → 283 (Phase 2) → 296 (Phase 3) → 329 (Phase 4) → **376 (Phase 5)**
- Figure-to-phase map: `docs/thesis_results_map.md` (Phase 6 → F5/F6/F7/F8 RL benchmark; Phase 7 → F9/F10/F12 ablations; Phase 8 → F13/F14 robustness)
