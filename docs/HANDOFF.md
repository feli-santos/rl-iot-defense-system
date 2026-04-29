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
| **Branch** | `feature/reward-shaping` (29 commits ahead of `origin/...` at handoff) |
| **Last commit at handoff** | `e02b651` (docs(phase-5): close Phase 5) |
| **Total tests** | 376/376 passing |
| **Phases complete** | 0, 1, 2, 3, 4, 5 |
| **Next phase** | Phase 6 — IoTWarden head-to-head baseline (direct comparison against IoTWarden's own DQN setup) |
| **Inspiring paper** | `docs/papers/IoTWarden- A Deep Reinforcement Learning Based Real-time Defense System...pdf` |
| **Thesis-figure map** | `docs/thesis_results_map.md` (F0..F14, T1) |
| **Reproducibility** | every figure has a sibling `manifest.json` with SHA-256 hash chain pinned to the producing git SHA |

### Phase-5 headline numbers (just shipped)

| Algo | Mean eval reward (last 10 % × 5 seeds) | Mean MTTC | Mitigated-impact rate |
|---|---:|---:|---:|
| **PPO** | **+1350.7** | 19.24 | 0.263 |
| A2C     | +1325.6     | 19.26 | 0.242 |
| DQN     | +1300.1     | 19.25 | 0.236 |

- **Recommended-policy floor** ~+50, so trained agents beat the IoTWarden hand-crafted baseline by **~25×**.
- 15 trained model checkpoints at `runs/phase5/<algo>/seed_<k>/model.zip`, ready for Phase 6 / 7 to consume.
- G5.4 PASS-WITH-FINDING: **the agent farms de-escalation bonuses (~6.30/episode × +250 = +1575) and accepts the IMPACT loss**. Phase 8's reward-component ablation will sweep this; Phase 6 is *not* expected to fix it.

---

## The prompt — paste this verbatim into the next agent

```
You are taking over as mentor/engineer on a Master's thesis project at
/Users/felipe.santos/Projects/rl-iot-defense-system on the
`feature/reward-shaping` branch. The thesis is an extension of IoTWarden
(Bhattacharjee et al. 2023, see docs/papers/) using the CICIoT2023
dataset. Six phases (0-5) are complete; you are picking up at Phase 6
(IoTWarden head-to-head baseline).

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
     not that the phase is doomed. The Phase-3, Phase-4, AND Phase-5
     gate revisions all turned "FAIL" into thesis-credible findings
     via dated D-decisions in PLAN §8 (e.g., D5.4.1, D2.1).

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
  - docs/thesis_results_map.md (figure -> phase mapping)
  - docs/results/00_phase0_diagnosis.md (the original failure mode
    that motivated the entire restart)
  - docs/results/01_dataset/ (Phase 1 — dataset card + F0 figures)
  - docs/results/02_red_team/PLAN.md and RESULTS.md (Phase 2 LSTM)
  - docs/results/03_env/PLAN.md and RESULTS.md (Phase 3 env v2)
  - docs/results/04_detector/PLAN.md and RESULTS.md (Phase 4 F11)
  - docs/results/05_blue_team/PLAN.md and RESULTS.md (Phase 5 F3/F4/T1)
    — and `G5_scoreboard.json` for the per-gate result.
  - docs/papers/IoTWarden- A Deep Reinforcement Learning Based Real-time
    Defense System...pdf — for Phase 6 you should read sections 3.1
    (system model), 3.2 (problem formulation), 4 (DQN-based defender),
    Tab. I (hyperparameters), and Fig. 4 (results) carefully. Phase 6
    *is* the head-to-head with that paper's reported numbers.

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
`recommended_action`, `attack_stage`, `last_action`. Phase 5's
`EpisodeJSONLCallback` already serialises the full schema. Phase 6
should reuse it verbatim.

The Phase-5 sweep produced 15 trained checkpoints at
`runs/phase5/<algo>/seed_<k>/model.zip` plus per-run JSONL logs and
manifests. Phase 6 should treat those as **frozen reference numbers**
to be benchmarked against the IoTWarden DQN setup; do NOT retrain
them.

The IoTWarden paper reports (Tab. I in their paper):

  - DQN with a small MLP (specific hidden sizes — read the paper)
  - episode length, reward shaping, action space (5 actions; matches
    ours)
  - reported convergence reward and MTTC (read Fig. 4 in their paper)

Phase 6's job is the *direct head-to-head*: re-implement IoTWarden's
DQN configuration as faithfully as their paper specifies, train it on
**our Phase-3 env** (so the comparison is on identical attacker/env
mechanics, not just the same dataset), and report the head-to-head
delta against our Phase-5 DQN/PPO/A2C numbers.

=== STEP 2: Write docs/results/06_iotwarden_baseline/PLAN.md ===

Cover, in this order:

  §1 Why Phase 6 exists (one paragraph; cite the thesis claim it
     supports — "we beat IoTWarden's reported numbers because..."
     or "we *match* them on env-controlled mechanics, demonstrating
     the contribution is the env + reward design, not the algorithm")
  §2 Audit findings (what code already exists from Phases 3-5 and
     what gaps remain). Include a faithful read of IoTWarden Tab. I
     and Fig. 4 — quote exact numbers.
  §3 Concrete deliverables:
       3.1 code (src/blue_team/iotwarden_config.py with a frozen
                 IoTWarden-paper hparams dict)
       3.2 tests (every reported hparam has a test that asserts
                  the dict matches the paper's exact value)
       3.3 exit gates G6.1..G6.k (each with a numerical threshold)
       3.4 figures produced (F8 — RL vs random / always-OBSERVE /
                              always-BLOCK / RF / IoTWarden-DQN —
                              listed in thesis_results_map.md as
                              Phase 4+7, but the IoTWarden bar of F8
                              is what Phase 6 produces)
  §4 Sequencing table (commits + estimated cost)
  §5 What we are NOT doing (defer to Phase 7 / Phase 8 etc.)
  §6 Risks tracked (R1..Rk with mitigations)
  §7 Cross-references to the thesis chapter outline
  §8 Locked design decisions (after mentor sign-off)

Recommended Phase-6 exit gates to start the discussion (the user will
edit them):

  G6.1  All new tests green; full suite green (target ~380+ tests).
  G6.2  IoTWarden DQN reproduction across 5 seeds:
          mean eval reward >= -100  (i.e., the IoTWarden-as-reported
          configuration *can* learn on our env). If this fails on
          the first run, ask "is the gate or the implementation
          wrong?" — IoTWarden's hparams may not match our Phase-3
          env scale exactly, which is a thesis finding by itself
          (env-as-controlled-variable shows the *delta* is in the
          algorithm-tuning, not the architecture).
  G6.3  Head-to-head delta: our Phase-5 best (PPO +1350.7) vs
          IoTWarden DQN reproduction. The expected gap is large
          (Phase-5 PPO uses a careful Phase-3 env + tuned hparams;
          IoTWarden's hparams are paper-default for a different env).
          The *delta* is what F8 reports, not the absolute number.
  G6.4  No regression on Phase-3 / Phase-4 / Phase-5 frozen tests
          (test_phase3_env_gates.py, test_adversarial_env.py,
           test_blue_team_*.py).
  G6.5  Reproducibility — Phase 6 produces an
          `iotwarden_baseline_summary.json` and a hash-pinned
          `manifest.json` for F8's IoTWarden bar.

Do NOT propose to retrain the LSTM, change the env, or modify the
Phase-5 model checkpoints. Phases 2/3/4/5 are frozen by contract.

The user's standing instructions for D-decisions:
  - The user has consistently said "I leave the decisions to you.
    What do you think is the best for my thesis defense/results?"
    Take that as the default for Phase 6 too. Make calls
    confidently, document them in PLAN §8 with rationale, and only
    ask the user when you genuinely need a value judgement (e.g.,
    "should we replicate IoTWarden's exact MLP topology even though
    it diverges from the SB3 default?"), not when you can defend the
    call yourself.

=== STEP 3: Lock the PLAN, then implement step-by-step ===

Same protocol as Phases 2-5: commit the PLAN, then implement one
substep per commit. After each substep, run pytest -q and verify
zero regressions.

Phase 6 is much smaller than Phase 5 — there is no new env, no new
sweep necessarily, and most of the heavy lifting is in faithful
hparam re-implementation + a single 5-seed IoTWarden-DQN training run
(if it converges, it converges in <1 h; if not, that's a *finding*).

The figure F8 is the deliverable. Don't burn a commit until F8 is
rendering at thesis quality with the IoTWarden bar correctly labelled.

=== STEP 4: Close Phase 6 ===

  - Run all gates G6.1..G6.k on real data.
  - Render F8 with manifest.
  - Write docs/results/06_iotwarden_baseline/RESULTS.md sister to PLAN.md.
  - Prepend a Phase-6 section to CHANGELOG.md.
  - Ensure pytest -q is green.
  - Tell the user the gate scoreboard, the headline numbers, and the
    findings worth defending. Specifically, the "is the contribution
    the algorithm or the env?" question Phase 6 was designed to
    answer.

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
  4. Walk back through commits 9b70d7d..e02b651 with `git show
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
- Figure-to-phase map for Phase 6+: `docs/thesis_results_map.md` (Phase 6 → F8 IoTWarden bar; Phase 7 → F5/F6/F7; Phase 8 → F9/F10/F12; Phase 9 → F13/F14)
