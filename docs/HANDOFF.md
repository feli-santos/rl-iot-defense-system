# Handoff prompt — RL IoT Defense thesis (CICIoT2023)

> **Hand this whole document to the next coding agent (or future-you).** Below
> the metadata block there is a single prompt that tells the agent how to
> orient itself, how to verify the prior phases are real, and how to pick up
> at Phase 5. The prompt is self-contained: an agent following it should
> need nothing other than this file, the repo, and CPU.

---

## Quick metadata (for the human reading this)

| | |
|---|---|
| **Repo** | `git@github.com-personal:feli-santos/rl-iot-defense-system.git` |
| **Branch** | `feature/reward-shaping` (21 commits ahead of `origin/...` at handoff) |
| **Last commit at handoff** | `99c6a54` (docs(phase-4): close Phase 4) |
| **Total tests** | 329/329 passing |
| **Phases complete** | 0, 1, 2, 3, 4 |
| **Next phase** | Phase 5 — RL Blue Team v2 (DQN/PPO/A2C × 5 seeds → F3, F4) |
| **Inspiring paper** | `docs/papers/IoTWarden- A Deep Reinforcement Learning Based Real-time Defense System...pdf` |
| **Thesis-figure map** | `docs/thesis_results_map.md` (F0..F14, T1) |
| **Reproducibility** | every figure has a sibling `manifest.json` with SHA-256 hash chain pinned to the producing git SHA |

---

## The prompt — paste this verbatim into the next agent

```
You are taking over as mentor/engineer on a Master's thesis project at
/Users/felipe.santos/Projects/rl-iot-defense-system on the
`feature/reward-shaping` branch. The thesis is an extension of IoTWarden
(Bhattacharjee et al. 2023, see docs/papers/) using the CICIoT2023
dataset. Five phases (0-4) are complete; you are picking up at Phase 5
(RL Blue Team training).

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
     not that the phase is doomed.

  3. Hash-chain everything: each thesis figure ships with a
     manifest.json that lists SHA-256 of the inputs and outputs and
     the producing git SHA. This is what lets the defense narrative
     stay reproducible.

  4. Honest commit history: when a prior phase's bug is discovered
     mid-phase, fix it as a `fix(phase-<N>):` commit attributed to
     the discovering phase, document it in the discovering phase's
     RESULTS.md §5, and decide *consciously* whether to rebuild
     downstream artefacts.

  5. Mentor-mode communication: brief, direct, lead with the result.
     Cite numbers, paper figures, and gate IDs by name. Don't bury
     the lede.

=== STEP 0: Acclimate ===

Read these in order, in full:
  - docs/HANDOFF.md (this file)
  - CHANGELOG.md (top to bottom — five [Unreleased] sections, one per
    completed phase, with gate scoreboards and findings)
  - docs/thesis_results_map.md (figure -> phase mapping)
  - docs/results/00_phase0_diagnosis.md (the original failure mode
    that motivated the entire restart)
  - docs/results/01_dataset/ (Phase 1 — dataset card + F0 figures)
  - docs/results/02_red_team/PLAN.md and RESULTS.md (Phase 2 LSTM)
  - docs/results/03_env/PLAN.md and RESULTS.md (Phase 3 env v2)
  - docs/results/04_detector/PLAN.md and RESULTS.md (Phase 4 F11)
  - docs/papers/IoTWarden- A Deep Reinforcement Learning Based Real-time
    Defense System...pdf (skim sections 3-5; figs 4-5 are the templates
    for our F3/F4)

Then verify the project is in the state CHANGELOG claims:

  cd /Users/felipe.santos/Projects/rl-iot-defense-system
  git log --oneline -25
  source .venv/bin/activate
  pytest -q                                        # expect 329 passed
  ls docs/results/04_detector/F11_per_stage_recall.png   # exists
  cat docs/results/04_detector/F11_summary.json | head -40
  ls artifacts/detector/                                  # 3 checkpoints

Read the F11 summary JSON in full. Confirm gate statuses match the
RESULTS.md §2 scoreboard. If anything disagrees, STOP and surface the
discrepancy before proceeding.

=== STEP 1: Phase 5 audit (no code yet) ===

Before any code, read:
  - src/algorithms/adversarial_algorithm.py        (existing SB3 wrapper)
  - src/environment/adversarial_env.py             (Phase-3 env, frozen)
  - src/utils/realization_engine.py                (split-aware factory)
  - tests/test_adversarial_algorithm.py            (what is tested today)
  - tests/test_phase3_env_gates.py                 (Phase-3 contract)
  - main.py                                        (--mode train-rl entrypoint)

The Phase-3 env's `info` dict exposes (Phase-3 RESULTS.md §3):
    compromised, mttc_steps, first_attack_step, compromise_step,
    defender_deescalations, recommended_action.
Phase 5 will need to log these (in addition to episode reward) for F3/F4.

The Phase-4 detector at artifacts/detector/stage_detector.pt:
    StageDetector, 4357 params, 0.039 ms / sample CPU.
    Public API: predict(X), predict_proba(X), from_checkpoint(path).
The agent can OPTIONALLY consume detector outputs as part of its
observation. Phase 5 should treat that as an *ablation*, not a default.

The thesis-results map calls for F3 ("RL episodic reward curves
DQN/PPO/A2C × 5 seeds") and F4 ("Action-distribution evolution over
training"). They are templated on IoTWarden Fig. 4(a) and Fig. 5
respectively. Reading those figures *first* will save you design
mistakes.

=== STEP 2: Write docs/results/05_blue_team/PLAN.md ===

Cover, in this order:

  §1 Why Phase 5 exists (one paragraph; cite the thesis claim it
     supports)
  §2 Audit findings (what code already exists and what gaps it has)
  §3 Concrete deliverables:
       3.1 code (src/blue_team/, scripts/blue_team/, tests/)
       3.2 tests
       3.3 exit gates G5.1..G5.k (each with a numerical threshold)
       3.4 figures produced (F3, F4) with caption sketches
  §4 Sequencing table (commits + estimated cost)
  §5 What we are NOT doing (defer to Phase 8 / Phase 9 etc.)
  §6 Risks tracked (R1..Rk with mitigations)
  §7 Cross-references to the thesis chapter outline
  §8 Locked design decisions (after mentor sign-off)

Recommended Phase-5 exit gates to start the discussion (the user will
edit them):

  G5.1  All new tests green; full suite green.
  G5.2  At least one of {DQN, PPO, A2C} achieves mean episodic reward
        > 0 on the eval split, averaged over the last 10% of training
        and over 5 seeds. (Sanity: the recommended-action policy
        already nets > 0; an RL agent should at least match it.)
  G5.3  Mean MTTC at convergence (last 10% of training) >
        max(min_episode_length, 1.5 * MTTC at start).
        (The agent must demonstrably *delay* compromise, not just
         act randomly.)
  G5.4  Compromise rate at convergence < 0.5 averaged over 5 seeds.
        (Not just MTTC, also fewer compromises overall.)
  G5.5  Action-distribution at convergence is non-degenerate:
        no single action accounts for > 70% of all decisions
        averaged over the last 10% of training.
  G5.6  No regression on Phase-3 env tests (test_phase3_env_gates.py
        and test_adversarial_env.py still green after any env-side
        changes).

Do NOT propose to retrain the LSTM or change the env in Phase 5.
Phase 3 is frozen by contract (PLAN §3.4 in Phase 4 follows the same
pattern). If something seems wrong, stop and surface it.

The user's standing instructions for D-decisions:
  - The user said in Phase 4: "I leave the decisions to you. What do
    you think is the best for my thesis defense/results?" Take that
    instruction as the default for Phase 5 too. Make calls
    confidently, document them in PLAN §8 with rationale, and only
    ask the user when you genuinely need a value judgement (e.g.,
    "should we use MlpPolicy or roll a custom Transformer encoder
    here?"), not when you can defend the call yourself.

=== STEP 3: Lock the PLAN, then implement step-by-step ===

Same protocol as Phases 2-4: commit the PLAN, commit the D-lock, then
implement one substep per commit. After each substep, run pytest -q
and verify zero regressions.

Phase 5 is much larger than Phases 2-4: training 3 algorithms × 5
seeds × ~500K timesteps will take several CPU-hours. Plan for:
  - using stable_baselines3 (already in requirements.txt)
  - logging to tensorboard AND a JSON-per-run file under
    runs/<algo>/<seed>/episodes.jsonl
  - aggregating across seeds with bootstrap CIs at plot time
  - parallelising via subprocess (one Python process per algo×seed)
    or VecEnv (one process, multiple env copies); both are valid,
    pick what gives you the cleanest log story.

The figures are the deliverables. Don't waste a commit until you have
F3 and F4 rendering at thesis quality with shaded confidence bands.

=== STEP 4: Close Phase 5 ===

  - Run all gates G5.1..G5.k on real data.
  - Render F3 and F4 with manifests.
  - Write docs/results/05_blue_team/RESULTS.md sister to PLAN.md.
  - Prepend a Phase-5 section to CHANGELOG.md.
  - Ensure pytest -q is green.
  - Tell the user the gate scoreboard, the headline numbers, and the
    findings worth defending.

=== House rules ===

  - Always cd to the repo root before running commands.
  - Always use `source .venv/bin/activate` for python; bare
    `python` is not on PATH.
  - The processed dataset (`data/processed/ciciot2023/`) is gitignored
    and lives only on the user's machine. Synthetic-data tests must
    NEVER depend on it. Real-data smoke tests should mark themselves
    `pytest.skipif(not Path('data/processed/...').exists(), ...)`.
  - When a gate "fails", first ask "is the gate or the
    implementation wrong?" Phases 3 and 4 both had gates that turned
    out to be wrong on the first contact with reality, and updating
    the gate (with rationale captured in PLAN §8) was the right
    move both times.
  - Commit messages follow conventional commits and cite the PLAN
    section being implemented (`feat(phase-5,§3.1.2): ...`).

=== If the user is asking you to *review*, not implement ===

If the user's first instruction after handoff is "review what was done
so far", do this and only this until they say "go":

  1. Read everything in STEP 0.
  2. Run `pytest -q` and confirm 329/329.
  3. Inspect the F11 PNG (open it) and the F11_summary.json numbers.
  4. Walk back through commits 4fd3460..99c6a54 with `git show
     --stat` to see what changed when. Confirm the *narrative* in
     RESULTS.md matches the actual commits.
  5. Look for one of:
       (a) numerical mismatches between RESULTS docs and JSON files,
       (b) tests that exist but are skipped without good reason,
       (c) artefacts referenced in CHANGELOG but absent on disk,
       (d) commits that touched unrelated files,
       (e) regressions in older tests masked by newer ones.
  6. Write a one-page audit to /tmp/handoff_audit.md and present it.
     Either green-light Phase 5 or list the blockers.

The user is preparing to defend a Master's thesis. Be honest. If
something is shaky, say so plainly with the evidence. The audit-first
protocol that has paid off twice (Phase 3 env bugs B1-B6, Phase 4
discovering the Phase-1 OOD leakage) only works if the *next* agent
also follows it.
```

---

## Quick links for the human

- Final Phase-4 figure: `docs/results/04_detector/F11_per_stage_recall.png`
- Phase-4 numbers: `docs/results/04_detector/F11_summary.json`
- Most recent gate scoreboard: `CHANGELOG.md` top section
- All phase RESULTS docs: `docs/results/{02_red_team,03_env,04_detector}/RESULTS.md`
- Test count history: 254 (Phase 0) → 266 (Phase 1) → 283 (Phase 2) → 296 (Phase 3) → 329 (Phase 4)
