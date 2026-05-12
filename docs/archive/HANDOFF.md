# Handoff prompt — RL IoT Defense thesis (CICIoT2023)

> **STATUS (2026-05-05): SUPERSEDED — historical record.**
>
> This file documented the Phase-7 → Phase-10 closeout decision (the
> "next agent picks Phase 8 vs Phase 10" branch point). It is preserved
> intact because the audit narrative in §"Audit-fix cycle on
> 2026-05-01" is referenced from `docs/results/07_ablation/RESULTS.md`
> §5 and from the `7537493` / `396f827` commit messages.
>
> **For the current state of the project**, read
> [`docs/mentor_review/`](mentor_review/) instead — specifically the
> highest-numbered `<NN>_HANDOFF.md` in that directory. Phase 7 and
> Phase 10 are both closed; the project is now in mentor-review mode
> and finalising the dissertation.
>
> The Phase 8 (F13 noise/drift robustness) section below is **not the
> current next step**; per `docs/mentor_review/00_framing.md` it has
> been reframed as future-work in Chapter 5 of the thesis rather than
> a phase of work to execute.

---

## (historical) Hand this whole document to the next coding agent (or future-you).

The prompt below is self-contained: an agent following it should need
nothing other than this file, the repo, and CPU. **But:** the
mentor-review walkthrough has now superseded the
"Phase 8 vs Phase 10" decision branch this prompt documents. New
agents should defer to `docs/mentor_review/<latest>_HANDOFF.md`.

---

## Quick metadata (for the human reading this)

| | |
|---|---|
| **Repo** | `git@github.com-personal:feli-santos/rl-iot-defense-system.git` |
| **Branch** | `feature/reward-shaping` (~52 commits ahead of `origin/...` at handoff) |
| **Last commit at handoff** | `396f827` (`docs(phase-7): close — RESULTS.md hand-fill + D7.1.1/D7.9.1 activation + G7 scoreboard + figures`) |
| **Total tests** | **454/454** passing, 0 skipped |
| **Phases complete** | 0, 1, 2, 3, 4, 5, 6, **7 (CLOSED 2026-05-01)** |
| **Phase-7 verdict** | **7 PASS / 2 FAIL-WITH-FINDING** across G7.1–G7.9. Both FAIL gates were pre-registered in PLAN §6 (R7.3 → G7.4) and §8 (D7.9.1 → G7.9). |
| **Next phase** | **D2 decision required** — Phase 8 (F13 noise/drift robustness) **OR** Phase 10 (open-source hygiene + README rewrite). See "Decision points for the next agent" below. |
| **Inspiring papers** | `docs/papers/IoTWarden- A Deep Reinforcement Learning Based Real-time Defense System...pdf` (Fig. 6 was reproduced as F10, G7.3 PASS) |
| **Thesis-figure map** | `docs/thesis_results_map.md` (F0..F15, T1) — Phase 7 closed F9/F10/F12/**F15**; Phase 8 owns F13/F14 |
| **Reproducibility** | every figure has a sibling `manifest.json` with SHA-256 hash chain pinned to the producing git SHA |

---

## Phase-7 closeout summary (the most important context)

### Final gate scoreboard (`docs/results/07_ablation/G7_scoreboard.json`)

| Gate | Threshold | Status | Headline value |
|---|---|:---:|---|
| **G7.1** | pytest -q ≥ 430 passed; zero new skips | **PASS** | 454 passed, 2 warnings |
| **G7.2** | F9 best reward-comparable cell mean test reward > Phase-6 DQN +1336 by ≥1σ | **PASS-WITHOUT-STRETCH** (D7.1.1 partially activated) | reward-comparable best=`impact_is_terminal_false` (+1542); security-KPI best=`impact_is_terminal_false` (mit=0.900); meets_oracle_stretch=False |
| **G7.3** | PPO p=0.0 < p=0.6 by ≥1σ AND rule monotone | **PASS** | p=0.0 CI=(134, 141); p=0.6 CI=(1280, 1359) — IoTWarden Fig. 6 replicated |
| **G7.4** | Pareto frontier ≥ 3 distinct dominant points | **FAIL-WITH-FINDING (R7.3)** | n_distinct=1/32 — trade-off surface is approximately linear |
| **G7.5** | Phase-3 frozen tests pass with `impact_is_terminal=True` | **PASS** | full pytest green ⇒ Phase-3 contract preserved |
| **G7.6** | No regression on Phase-3/4/5/6 frozen tests | **PASS** | 454/454 |
| **G7.7** | F9/F10/F12/F15 manifests SHA-pinned | **PASS** | all 4 present |
| **G7.8** | F15 4×8 OOD matrix complete, no NaN | **PASS** | 32/32 cells |
| **G7.9** | On VulnerabilityScan, trained RL > RF-Acting by ≥1σ | **FAIL-WITH-FINDING (D7.9.1 fully activated)** | DQN +1313 (CI 1228–1387) vs RF +1611 (CI 1556–1666); Δ=−298 |

### What's defensible after Phase 7 (the thesis claims)

1. **G7.2 / D7.1.1 partial:** "Within the Phase-3 reward formulation, no
   single-axis 0.5×/2× reward-coefficient perturbation moves PPO mean
   reward by ≥ 1σ. Closing the +288 deployable gap to the oracle ceiling
   required a structural env-semantics change (`impact_is_terminal=False`),
   which recovers **71 % of the gap** and improves mitigated-impact rate
   from DQN's 0.153 to **0.900** (5.9×). The remaining −82.5 to the oracle
   is the cost of operating without oracle stage knowledge." (RESULTS §6.1.)
2. **G7.3 PASS:** "On CICIoT2023, with a 29-feature state and Kill-Chain
   reward, we replicate the IoTWarden Fig. 6 qualitative shape: PPO mean
   reward grows monotonically with `p_defender_deescalation`. Validates
   the Phase-3 reward function as having the source paper's qualitative
   behaviour despite the realer environment." (RESULTS §6.3.)
3. **G7.9 / D7.9.1 narrowed:** "RL is **robust to** (not **better at**)
   the OOD class — DQN's mean OOD reward (+1313) is within seed-noise of
   its in-distribution mean (+1336); generalisation does not collapse
   the policy. RF-Acting's stronger OOD reward (+1611) is *not* evidence
   of RF working (recall = 0.001) — it is evidence that 'do nothing' is
   locally-good when the Phase-3 reward function is dominated by avoiding
   disproportionate-penalty costs." (RESULTS §6.2 — includes 3 pre-rebuttals
   for the defense committee.)
4. **G7.4 / R7.3 finding:** "Under the Phase-3 reward formulation the
   security-vs-availability trade-off is approximately linear; operating-
   point selection reduces to a single scalar weighting. Future work that
   wants a non-trivial Pareto front needs *non-linear* reward composition
   (e.g. a hard mit-rate constraint)." (RESULTS §6.4.)

### Audit-fix cycle on 2026-05-01 (read this carefully if you're auditing)

The auto-finalizer landed Phase-7 artefacts at 23:01 on 2026-04-30 with the
verdict **4 PASS / 5 FAIL-WITH-FINDING**. A same-day audit caught and
corrected 3 issues *before* the chapter locked, in commits `7537493` (fix)
and `396f827` (docs):

1. **G7.2 evaluator was reward-scaling-blind.** It picked
   `defense_success_bonus_x2p0` (+2926) as the winner — but that cell's
   reward function is constructed to earn ~2× the bonus per defense
   success, so +2926 is not commensurable with Phase-6's DQN +1336.
   Corrected: `_evaluate_g72` now splits into two strands —
   apples-to-apples raw reward (only Phase-3-reward-fn-preserving cells
   qualify) and security-KPI fallback (mitigated-impact rate is
   commensurable across cells). Activates D7.1.1 partially: both strands
   now agree honestly, with `impact_is_terminal_false` winning. See
   `tests/test_close_phase7_parsers.py` for the pinned logic (12 tests).
2. **`close_phase7._run_pytest_count` exit-code bug.** The parser gated
   on `proc.returncode == 0`, which spuriously reported false on
   warning-only runs. False-fail then cascaded to G7.5/G7.6 (both
   piggyback on G7.1). Fixed: gate on `passed > 0 and failed == 0 and
   errors == 0` from the trailing summary line. Pytest exit codes are
   *not* the source of truth.
3. **Test count went 442 → 454** (+12 from `test_close_phase7_parsers.py`).

The corrected scoreboard is **7 PASS / 2 FAIL-WITH-FINDING**, with both
FAIL-WITH-FINDING gates pre-registered (R7.3 and D7.9.1) — neither is a
goalpost move. AF3 protocol-continuity preserved (D5.4.1, D6.2.1, D7.X.1
precedents).

---

## Decision points for the next agent

The user's standing instruction is "make the call, document it, only ask if
it's a value judgement."

### D2 — Phase 8 (F13 noise/drift robustness) vs Phase 10 (open-source hygiene)

Phase 7 closed cleanly. The next-step choice depends on the user's time-
to-defense:

- **Phase 8 — F13 (Tier 3, novel work)**: inject Gaussian noise into
  observations / drift the realiser's per-stage means and re-run F5 / F8
  to see how mean reward degrades. Adds a **fourth thesis claim** (a
  robustness chapter). ~3 h CPU + ~1 day human. Needs a fresh PLAN.md
  (audit-first protocol) before code.
- **Phase 10 — open-source hygiene** (audit AF4):
  - Delete `src/benchmarking/` (singular; three files, dead but consumed
    by 43 tests in `test_benchmark_runner.py` + `test_metrics_collector.py`).
    Phase-5/6/7 pipelines use `src/benchmark/` (no g) instead.
  - Delete three pre-restart orphans: `scripts/evaluate_generator.py`,
    `scripts/measure_improved_targets.py`, `scripts/separability_analysis.py`.
  - Rewrite root `README.md` (still pre-restart; doesn't mention any of
    the eight completed phases or `make phase-N` targets).
  - Tag the repo (`v0.1.0` or `phase-7-closeout`).
  - ~1 day human, ~0 h CPU.

**Default if user has < 1 week to defense**: Phase 10 (cleanup + README +
tag) — it's ship-blocking for the open-source release the thesis cites.
Phase 8 if there's budget for a fourth thesis figure thread.

### D3 — Audit Phase-7 numbers if the next agent is uneasy

If the next agent re-reads the RESULTS narrative and disagrees with any
specific claim:

- All raw numbers live in `docs/results/07_ablation/F{9,10,12,15}_summary.json`
  and the gate verdicts in `docs/results/07_ablation/G7_scoreboard.json`.
- The two-strand G7.2 evaluator is in `scripts/ablation/plot_reward_ablation.py::_evaluate_g72`
  (committed in `7537493`); 6 unit tests in `tests/test_close_phase7_parsers.py`
  pin its behaviour.
- Re-running on the local machine: `python -m scripts.ablation.plot_reward_ablation`
  + `python -m scripts.ablation.close_phase7` will reproduce the figures
  from the cached `runs/phase7/.../eval_test.jsonl` (gitignored; ~7.5 h
  CPU only if missing).
- If the agent thinks the +205.6 win in G7.2 is suspicious, the protocol
  is: re-run `impact_is_terminal_false` × PPO with **fresh seed pool**
  (e.g. seeds 5–9 instead of 0–4) and confirm the mean is stable. Same
  protocol Phase 6 used for D6.2.1.

### D4 — Push origin

`git push origin feature/reward-shaping` whenever the user is ready. The
branch is +13 ahead at handoff time but +14 with this HANDOFF rewrite
commit; check `git log --oneline origin/feature/reward-shaping..HEAD`.

---

## The prompt — paste this verbatim into the next agent

```
You are taking over as mentor/engineer on a Master's thesis project at
/Users/felipe.santos/Projects/rl-iot-defense-system on the
`feature/reward-shaping` branch. The thesis is an extension of IoTWarden
(Bhattacharjee et al. 2023, see docs/papers/) using the CICIoT2023
dataset.

**EIGHT PHASES (0-7) ARE NOW CLOSED.** Phase 7's gate scoreboard is
**7 PASS / 2 FAIL-WITH-FINDING** with both FAIL gates pre-registered
in PLAN §6 (R7.3 → G7.4) and §8 (D7.9.1 → G7.9). The chapter locks at
commit `396f827`. Tests: 454/454.

The most likely next move is one of:

  - "Start Phase 10 (open-source hygiene)" → §3.1 below
  - "Start Phase 8 (F13 robustness)" → §3.2 below
  - "Audit/review what was done" → §3.3 below
  - "Push origin" → just run `git push origin feature/reward-shaping`

Your job depends on what the user gives you as the first instruction.

=== STEP 0: Acclimate ===

Read these in order, in full:
  - docs/HANDOFF.md (this file)
  - CHANGELOG.md (top to bottom — eight [Unreleased] sections, one per
    closed phase; the top one is Phase 7's full closeout block with
    the corrected 7/2 scoreboard and the same-day audit-fix narrative)
  - docs/thesis_results_map.md (figure → phase mapping; F0..F15 + T1
    are all Tier 1 / Tier 2 / Tier 3 placed; F13 + F14 are Phase 8
    territory)
  - docs/results/07_ablation/PLAN.md (Phase-7 plan; §8 D7.1.1 + D7.9.1
    are now activated entries with dated rationale, not placeholders)
  - docs/results/07_ablation/RESULTS.md (the closeout doc with full
    §6 narrative — read §1 headlines + §6 thesis claims especially
    carefully; §5 documents the 3 cross-phase findings discovered
    during Phase 7 (smoke fixes, G7.2 evaluator audit, pytest parser))
  - docs/results/07_ablation/G7_scoreboard.json (canonical per-gate
    record — 7 PASS / 2 FAIL-WITH-FINDING)
  - docs/results/07_ablation/F{9,10,12,15}_summary.json (live numbers
    + per-gate evaluator output)
  - docs/results/06_benchmark/RESULTS.md §6 (the AF2-reframed
    oracle-ceiling framing — Phase-7 §1 cites this)

Then verify the project is in the state CHANGELOG claims:

  cd /Users/felipe.santos/Projects/rl-iot-defense-system
  git log --oneline -25
  source .venv/bin/activate
  pytest -q                                    # expect 454 passed
  ls docs/results/07_ablation/                 # expect F9/F10/F12/F15 figures + JSONs + manifests + RESULTS.md + G7_scoreboard.json + PLAN.md
  cat docs/results/07_ablation/G7_scoreboard.json | jq '.n_pass, .n_fail'  # expect 7, 2
  cat docs/results/07_ablation/F9_summary.json | jq '.gates."G7.2".best_cell, .gates."G7.2".best_mean_reward'  # expect "impact_is_terminal_false", 1541.9...

If pytest != 454/454, STOP and surface the discrepancy. The Phase-7
sweep cell directories under runs/phase7/{ood,reward_sweep,aggressiveness}/
are gitignored and may not exist on a fresh checkout — that is fine
unless the user wants to re-run the plotters; for narrative work, the
docs/results/07_ablation/F*_summary.json files have all numbers cached.

If runs/phase5/ and runs/phase6/ are missing AND the user wants to
regenerate Phase-7 plots, regenerate them first:

  make phase-5-sweep PHASE5_TIMESTEPS=250000   # ~108 minutes CPU (one-off)
  make phase-6                                 # phase-6-eval + phase-6-figures (~10 min)
  make phase-7                                 # ~7.5 h CPU walk-away

=== STEP 1: Triage and act on the user's first instruction ===

The user is most likely to say one of:

  "go" / "next" / "Phase 10" → §3.1 (Phase 10 hygiene)
  "Phase 8" / "F13" → §3.2 (Phase 8 robustness)
  "review" / "audit Phase 7" → §3.3 (read-only audit pass)
  "push" → `git push origin feature/reward-shaping`

If the user gives a different instruction, default to §3.3 (audit Phase
7 first); the right next move depends on whether the gate verdicts are
honest under a fresh pair of eyes.

=== §3.1 — Phase 10: open-source hygiene + README + release tag ===

Recommended default if the user has < 1 week to defense — this is
ship-blocking for the open-source release that the thesis cites.

Sequence (~1 day human, ~0 h CPU):

  C1. docs/results/10_release/PLAN.md (audit-first; commit BEFORE code)
      - Audit: src/benchmarking/ (singular) is dead — Phase-5/6/7 use
        src/benchmark/ (no g). 43 tests in test_benchmark_runner.py +
        test_metrics_collector.py exclusively consume the dead package.
      - Audit: scripts/evaluate_generator.py,
        scripts/measure_improved_targets.py,
        scripts/separability_analysis.py are pre-restart orphans (search
        with `grep -r 'evaluate_generator\|measure_improved_targets\|separability_analysis' --include='*.py'` confirms no consumer).
      - Audit: README.md (root) is pre-restart — does not mention Phases
        0–7, F-figures, or `make phase-N` targets.
      - Audit: CITATION.cff exists but has no thesis-specific Person.
      - Deliverables list: code deletions + README rewrite + tag.
      - Exit gates: G10.1 pytest -q == 411 passed (454 - 43 dead),
        G10.2 no `from src.benchmarking` consumer remaining,
        G10.3 README.md mentions all eight phases + reproducibility,
        G10.4 git tag `v0.1.0` exists.
  C2. fix(phase-10): delete dead src/benchmarking/ package + tests
      - rm src/benchmarking/{benchmark_analyzer,benchmark_runner,metrics_collector}.py
      - rm src/benchmarking/__init__.py if any
      - rm tests/test_benchmark_runner.py tests/test_metrics_collector.py
      - pytest -q expected to drop 454 → 411
  C3. fix(phase-10): delete three pre-restart orphan scripts
  C4. docs(phase-10): rewrite root README.md (project overview, eight
      phases as numbered chapters with their headline F-figures inline,
      `make phase-N` reproduction recipes, dataset citation, IoTWarden
      paper citation)
  C5. docs(phase-10): close — RESULTS.md + CHANGELOG block + tag v0.1.0
  C6. git push origin feature/reward-shaping --tags

=== §3.2 — Phase 8: F13 noise/drift robustness ===

Recommended if the user has ≥ 1 week to defense and the Phase-7 results
are strong (they are). Adds a fourth thesis claim (robustness chapter).

Sequence (~3 h CPU + ~1 day human, audit-first):

  C1. docs/results/08_robustness/PLAN.md (committed FIRST)
      - Audit: existing realiser RealizationEngine.from_split_manifest
        in src/utils/realization_engine.py — what's the obs vector
        shape, what's the per-stage feature distribution, where can
        we inject perturbations?
      - F13 = inject Gaussian noise σ ∈ {0.1, 0.5, 1.0} on the
        normalised obs vector, re-eval each Phase-5 trained checkpoint
        (no retraining, like F15), plot mean reward as a function of σ.
      - F14 (optional / Tier 3 stretch) = train-time OOD-class
        augmentation: synthetic feature blending of one OOD class into
        the train pool, then re-evaluate F15 and check whether the
        retrained DQN/PPO/A2C now beat RF-Acting on VulnerabilityScan
        (i.e. inverse of D7.9.1).
      - PLAN §8 must cite D7.1.1 + D7.9.1 + D6.2.1 + D5.4.1 as
        precedents (AF3 protocol-continuity).
      - G8 exit gates with thresholds.
  C2. feat(phase-8,§3.1): obs-noise injection in env_factory.py +
      synthetic tests pinning the codepath
  C3. feat(phase-8,§3.2): F13 driver + plotter + manifest
  C4. (optional C5 if F14 ships): feat(phase-8,§3.3): F14 retrain +
      re-eval
  C5. docs(phase-8): close — RESULTS.md + G8 scoreboard + CHANGELOG

=== §3.3 — Audit Phase 7 (read-only) ===

If the user's first instruction is any phrasing of audit / review /
look at the results, do this and only this until they say "go":

  1. Read everything in STEP 0 above.
  2. Run the gate-by-gate scoreboard verification:
       pytest -q                                # expect 454 passed
       cat docs/results/07_ablation/G7_scoreboard.json | jq '{n_pass, n_fail, gates: [.gates[] | {id, passes, value}]}'
  3. Inspect the four PNGs:
       open docs/results/07_ablation/F9_reward_ablation.png
       open docs/results/07_ablation/F10_aggressiveness.png
       open docs/results/07_ablation/F12_pareto.png
       open docs/results/07_ablation/F15_ood_robustness.png
  4. Walk back through commits cdb609a..HEAD with `git show --stat`
     and confirm the *narrative* in 07_ablation/RESULTS.md matches
     the actual commits + JSONs.
  5. Look for one of:
       (a) numerical mismatches between RESULTS docs and JSON files
           (the close_phase7 script regenerates RESULTS skeleton from
           the JSONs; the §6 narrative was hand-filled in 396f827);
       (b) tests that exist but are skipped without good reason;
       (c) artefacts referenced in CHANGELOG but absent on disk;
       (d) commits that touched unrelated files;
       (e) regressions in older tests masked by newer ones;
       (f) the two-strand G7.2 logic — does the verdict match the
           direction of the bootstrap CIs in F9_summary.json#rows?
           The reward-comparable strand only counts axis ∈
           {baseline, impact_terminal} cells; the security-KPI strand
           counts all cells (axis="reward" included). Both must agree
           on impact_is_terminal_false for the current verdict to
           hold;
       (g) the hybrid OOD realiser correctness — F15's central
           design choice is that the agent sees in-distribution
           features at every non-OOD-stage step and OOD features
           only at the OOD class's stage. If the F15 mean rewards
           on the three classes whose OOD stage isn't IMPACT look
           much worse than Phase-6 baselines, that's the hybrid
           realiser working as designed; on the IMPACT-only class
           (DDoS-HTTP_Flood) the means should be closer to Phase-6
           because IMPACT is where the test fires.
  6. Write a one-page audit to /tmp/handoff_audit.md and present it.
     Either green-light Phase 8 / Phase 10 or list the blockers.

=== Operating principles (NOT NEGOTIABLE — eight closed phases earned them) ===

  1. Audit-first: read the relevant code and PLAN.md before writing
     any new code, and write a PLAN.md for each NEW phase BEFORE
     touching code. PLAN.md must contain (a) the audit findings,
     (b) the deliverables, (c) the exit gates, (d) a sequencing
     table, (e) what we are NOT doing, (f) the risks tracked. The
     PLAN goes through a "lock decisions" commit before any
     implementation.

  2. Empirical gates: every phase has named exit gates G<phase>.<i>
     with explicit numerical thresholds. Run them on real data
     before calling the phase done. When a gate fails, treat the
     failure as diagnostic — it usually means the gate or the
     design has a hole, not that the phase is doomed. Phases 3, 4,
     5, 6, AND 7 all turned "FAIL" into thesis-credible findings
     via dated D-decisions in PLAN §8 (B1-B6, D2.1, D5.4.1,
     D6.2.1, D6.8.1, D7.1.1 partial, D7.9.1 full).

  3. Hash-chain everything: each thesis figure ships with a
     manifest.json listing SHA-256 of inputs and outputs and the
     producing git SHA. This is what lets the defense narrative
     stay reproducible.

  4. Honest commit history: when a prior phase's bug or
     selection-bias artefact is discovered mid-phase, fix it as a
     `fix(phase-<N>):` commit attributed to the discovering phase,
     document it in the discovering phase's RESULTS.md §5, and
     decide *consciously* whether to rebuild downstream artefacts.
     Phase 7's `87b80dc` (smoke surfaced 3 latent bugs) and
     `7537493` (audit-fix G7.2 evaluator + pytest parser) are the
     model: each named a specific issue, fixed it with tests, and
     no Phase-3/4/5/6 numbers were retroactively touched.

  5. Mentor-mode communication: brief, direct, lead with the result.
     Cite numbers, paper figures, gate IDs, commit SHAs by name.
     Don't bury the lede. The user is preparing a thesis defense; be
     honest. If something is shaky, say so plainly with the evidence.

=== House rules (NOT NEGOTIABLE) ===

  - Always cd to the repo root before running commands.
  - Always use `source .venv/bin/activate` for python; bare
    `python` is not on PATH.
  - The processed dataset (`data/processed/ciciot2023/`),
    `runs/phase5/`, `runs/phase6/`, `runs/phase7/` are gitignored
    and live only on the user's machine. Synthetic-data tests must
    NEVER depend on them. Real-data smoke tests should mark
    themselves `pytest.skipif(not Path('data/processed/...').exists(), ...)`.
  - When a gate "fails", first ask "is the gate or the
    implementation wrong?" Phases 3, 4, 5, 6, AND 7's smoke run +
    audit cycle all had gates / designs / evaluators that turned
    out to be wrong on the first contact with reality, and updating
    the gate / design / evaluator (with rationale captured in PLAN
    §8 D-decision or a `fix(phase-N):` commit) was the right move
    every time.
  - Commit messages follow conventional commits and cite the PLAN
    section being implemented (`feat(phase-8,§3.1.2): ...`).
```

---

## Quick links for the human

- Phase-7 PLAN (with D7.1.1 + D7.9.1 activated): `docs/results/07_ablation/PLAN.md`
- Phase-7 RESULTS (full §6 narrative): `docs/results/07_ablation/RESULTS.md`
- Phase-7 figures: `docs/results/07_ablation/F{9,10,12,15}_*.png`
- Phase-7 numbers: `docs/results/07_ablation/F{9,10,12,15}_summary.json`
- Phase-7 manifests: `docs/results/07_ablation/F{9,10,12,15}_manifest.json`
- Phase-7 captions: `docs/results/07_ablation/F{9,10,12,15}_caption.md`
- Phase-7 scoreboard: `docs/results/07_ablation/G7_scoreboard.json` (7 PASS / 2 FAIL-WITH-FINDING)
- Background runner logs (kept for forensics): `logs/phase7/{phase7,ood,reward,aggressiveness,pareto,finalize}.log`
- All phase RESULTS: `docs/results/{02_red_team,03_env,04_detector,05_blue_team,06_benchmark,07_ablation}/RESULTS.md`
- CHANGELOG: top section is Phase 7 closeout (`396f827`); previous Phase-6 closeout is `d3e8ae1`
- Test count history: 254 (Phase 0) → 266 (Phase 1) → 283 (Phase 2) → 296 (Phase 3) → 329 (Phase 4) → 376 (Phase 5) → 420 (Phase 6) → 442 (Phase 7 §3.3) → **454 (Phase 7 audit-fix)**
- Figure-to-phase map: `docs/thesis_results_map.md` (Phase 7 closed F9 / F10 / F12 / **F15**; Phase 8 → F13 / F14)
- Phase-7 implementation commits: `cdb609a` C1 → `df14cc9` C2 → `4d19e81` C3 → `7268428` C4 → `e2dd145` C5 → `8c2636a` C6 → `e11485f` C7+C8 → `87b80dc` smoke fixes → `e2d5f9d` C9 scaffold → `7537493` **audit-fix (G7.2 + pytest parser)** → `396f827` **closeout (RESULTS hand-fill + D7.1.1/D7.9.1 activation + figures)**
