# Handoff prompt — RL IoT Defense thesis (CICIoT2023)

> **Hand this whole document to the next coding agent (or future-you).** Below
> the metadata block there is a single prompt that tells the agent how to
> orient itself, how to verify the prior phases are real, and how to decide
> what's next. The prompt is self-contained: an agent following it should
> need nothing other than this file, the repo, and CPU.

---

## Quick metadata (for the human reading this)

| | |
|---|---|
| **Repo** | `git@github.com-personal:feli-santos/rl-iot-defense-system.git` |
| **Branch** | `feature/reward-shaping` (~50 commits ahead of `origin/...` at handoff) |
| **Last commit at handoff** | `e2d5f9d` (`feat(phase-7,§4): close_phase7.py + background runner scripts (C9 scaffolding)`) |
| **Total tests** | **442/442** passing, 0 skipped (was 420 at last handoff) |
| **Phases complete** | 0, 1, 2, 3, 4, 5, 6 (closed); **Phase 7 implementation = 100 % committed but real-data CPU runs are mid-flight at the moment of this handoff** |
| **Phase-7 status** | Code + tests + smoke-validated end-to-end. **Background runner kicked off ~3:41 PM 2026-04-30 → ETA ~10:41 PM** (~7 h walk-away). Auto-finalizer (`scripts/ablation/_finalize_phase7_background.sh`) is polling for completion and will run `close_phase7.py` to materialise C9 deliverables (`G7_scoreboard.json` + `RESULTS.md` + `CHANGELOG.md` prepend) without human intervention. |
| **Next phase after Phase-7 closure** | **Review Phase-7 results, then either Phase 8 (F13 noise/drift robustness) or Phase 10 (open-source hygiene)** depending on time-to-defense priorities. See "Decision points for the next agent" below. |
| **Inspiring papers** | `docs/papers/IoTWarden- A Deep Reinforcement Learning Based Real-time Defense System...pdf` (Fig. 6 was reproduced in F10) |
| **Thesis-figure map** | `docs/thesis_results_map.md` (F0..F15, T1) — Phase 7 owns F9/F10/F12/**F15** (the last is audit-AF1 Tier-1) |
| **Reproducibility** | every figure has a sibling `manifest.json` with SHA-256 hash chain pinned to the producing git SHA; all Phase-7 figure scripts already wire G7.2/G7.3/G7.4/G7.8/G7.9 evaluators inline |

---

## Phase-7 status at handoff (the most important context)

### What's committed (12 commits, all PHASE-7-related)

| Commit | Title | Effect |
|---|---|---|
| `cdb609a` | C1 — `docs(phase-6,§6.4): reframe rec-action floor as oracle upper bound (audit AF2)` | Phase-6 RESULTS reads "DQN +1336 = 82 % of oracle ceiling +1624" — not "RL lost by 290". |
| `df14cc9` | C2 — `docs(phase-7): audit & PLAN — F9 + F10 + F12 + F15 ablations` | Locked PLAN.md (566 lines) + thesis_results_map.md F15 promotion. |
| `4d19e81` | C3 — `feat(phase-7,§3.1.1): impact_is_terminal flag in AdversarialEnvConfig` | Default `True` preserves Phase-3 contract; `False` = explicit IMPACT-row decision step. **+8 tests (420 → 428).** |
| `7268428` | C4 — `feat(phase-7,§3.1.2): --reward-overrides + --p-defender-deescalation + --impact-is-terminal in train_agent.py` | `EnvConfigSerializable` extended 7 → 18 fields. Pre-Phase-7 manifests still deserialise. **+14 tests (428 → 442).** |
| `e2dd145` | C5 — `feat(phase-7,§3.1.3): F15 OOD-class robustness eval runner + plotter (audit-AF1)` | New `scripts/ablation/{run_ood_eval, plot_ood_robustness}.py`. |
| `8c2636a` | C6 — `feat(phase-7,§3.1.4): F9 reward-component sweep driver + plotter (D7.1)` | 12-cell sparse one-at-a-time grid (5 components × {0.5×, 1×, 2×} + impact_is_terminal binary) × PPO × 5 seeds = 60 runs. |
| `e11485f` | C7+C8 — `feat(phase-7,§3.1.5,§3.1.6): F10 aggressiveness sweep + F12 Pareto plot` | F10: 6 p-values × PPO × 5 seeds + oracle rule reference. F12: plotter-only, derived from F9 + F10 + Phase-6. |
| `87b80dc` | **`fix(phase-7): three smoke-surfaced bugs (hybrid OOD realiser + train/eval window match + relative_to)`** | Critical: each OOD class is single-stage, so the original "constrain to OOD indices" approach crashed on `env.reset()`. Replaced with hybrid realiser (in-distribution train pool + OOD overlay at the OOD class's stage). Also fixed `--smoke` train/eval obs-shape mismatch and `Path.relative_to` bug. **Without this commit, 7.5 h of CPU would have crashed instantly.** |
| `e2d5f9d` | C9 scaffold — `feat(phase-7,§4): close_phase7.py + background runner scripts (C9 scaffolding)` | `scripts/ablation/close_phase7.py` reads all four `F*_summary.json#gates` blocks and emits canonical `G7_scoreboard.json` + populated-with-live-numbers `RESULTS.md` + `CHANGELOG.md` prepend. Plus the two ops-only background helper scripts. |

### What's running right now (background processes alive on the user's machine at handoff)

| PID | Role | Started |
|---:|---|---|
| 15496 | top-level `_run_phase7_background.sh` parent | 15:41 |
| 15517 | `python -m scripts.ablation.run_aggressiveness_sweep` (F10) | 15:41 |
| 15518 | `python -m scripts.ablation.run_ood_eval` (F15, audit-AF1) | 15:41 |
| 15519 | `python -m scripts.ablation.run_reward_sweep` (F9) | 15:41 |
| 15541, 15542, … | `train_agent.py` PPO subprocesses (one per cell × seed) | rolling |
| 16226 | `_finalize_phase7_background.sh` watcher | 15:45 |

ETA: ~10:41 PM 2026-04-30. F9 is the rate-limiter (60 PPO × 250K runs serial within its driver). F10 (~1.5 h) and F15 (~1 h) finish much earlier and idle while F9 grinds.

### What lands automatically when phase7.end appears

The watcher (`_finalize_phase7_background.sh`, PID 16226) polls
`logs/phase7/phase7.end` every 60 s. Once the parent runner writes
that file (after `make phase-7-pareto` returns), the watcher runs
`python -m scripts.ablation.close_phase7` which produces:

  1. `docs/results/07_ablation/G7_scoreboard.json` — canonical
     per-gate threshold + value + status (mirrors Phase-6 G6
     scoreboard shape).
  2. `docs/results/07_ablation/RESULTS.md` — §1–§9 doc with **live
     numbers** from the `F<N>_summary.json` files (best-cell reward
     in F9, RL-vs-RF Δ on `VulnerabilityScan` in F15, Pareto
     frontier point count in F12, etc.). Hand-fillable narrative
     sections marked `(Hand-fill ...)` for the next agent / the user
     to flesh out before locking the chapter.
  3. **CHANGELOG.md** — Phase-7 `[Unreleased]` block prepended with
     gate scoreboard + headline pointers.

**The next agent should NOT re-run any of this** — read the artefacts
that landed, then either (a) finish the hand-fillable narrative
sections in RESULTS.md, (b) decide on Phase 8 vs Phase 10, or
(c) audit the Phase-7 numbers if the user requests review-not-act.

### How to check progress before phase7.end exists

```bash
# parent runner status
ps -p $(cat logs/phase7/runner.pid)

# per-sweep tails
tail -f logs/phase7/phase7.log
tail -f logs/phase7/{ood,reward,aggressiveness}.log

# cells produced so far
find runs/phase7/ -name eval_test.jsonl | wc -l   # target: 60 (F9) + 30 (F10) + 32 (F15) + 6 (rule reference) = 128
```

If `phase7.end` already exists when the next agent reads this:

```bash
# Confirm finalization worked.
cat logs/phase7/finalize.log
ls docs/results/07_ablation/  # should have F9/F10/F12/F15_*.{png,json,md} + G7_scoreboard.json + RESULTS.md
head -60 CHANGELOG.md         # should start with the Phase-7 [Unreleased] block
```

---

## Decision points for the next agent (hand from previous mentor-agent)

After the auto-finalization lands, you face four orthogonal next-step choices. The user's standing instruction is "make the call, document it, only ask if it's a value judgement." Suggested defaults:

### D1 — Hand-fill the RESULTS.md narrative sections

**Default: do this immediately after phase7.end.** The auto-generated `RESULTS.md` has placeholder `(Hand-fill ...)` blocks in §5, §6.1–§6.4. Each maps to a gate verdict (G7.2, G7.9, G7.3, G7.4) whose interpretation is already pre-written in `F<N>_summary.json#gates.G*.interpretation`. The hand-fill is just narrative phrasing for the thesis chapter (1 paragraph each, ~1 h work).

The two thesis-headline sections to write last and most carefully are:

- **§6.1 (G7.2 / D7.1.1 if needed)**: did the F9 reward-component sweep close the +288 deployable gap to the oracle ceiling? If yes, name the winning component and cite the +Δ. If no, frame as D7.1.1 — the linear sweep characterised the limit of one-at-a-time Phase-3-style reward shaping.
- **§6.2 (G7.9 / D7.9.1 if needed; audit-AF1 HEADLINE)**: did trained RL beat RF-Acting on `VulnerabilityScan` by ≥1σ? If yes, the thesis claim "RL closes the OOD gap by acting on raw features" lands. If no, narrow the claim to "RL is *robust to* (not *better at*) the OOD class" per D7.9.1 — still defensible, just narrower.

### D2 — Phase 8 (F13 noise/drift robustness) vs Phase 10 (open-source hygiene)

Both are scoped in `docs/thesis_results_map.md` Tier 3. The user's time-to-defense decides:

- **Phase 8 — F13 (Tier 3)**: inject Gaussian noise into observations / drift the realiser's per-stage means and re-run F5 / F8 to see how mean reward degrades. ~3 h CPU + ~1 day human. Adds a **fourth thesis claim** (robustness chapter) but is *novel work* — needs a fresh PLAN.md + audit cycle.
- **Phase 10 — open-source hygiene** (audit AF4 from 2026-04-30, still in this file's audit ledger):
  - Delete `src/benchmarking/` (singular, three files, dead but consumed by 43 tests in `test_benchmark_runner.py` + `test_metrics_collector.py`). Phase-5/6 pipelines use `src/benchmark/` (no g) instead.
  - Delete three pre-restart orphans: `scripts/evaluate_generator.py`, `scripts/measure_improved_targets.py`, `scripts/separability_analysis.py`.
  - Rewrite root `README.md` (still pre-restart; doesn't mention any of the seven completed phases, F-figures, or `make phase-N` targets).
  - Plus: tag the repo, write a `CITATION.cff` for the thesis, decide on PyPI / Docker.
  - ~1 day human, ~0 h CPU.

**Default if user has < 1 week to defense**: Phase 10 (cleanup + README + tag) — it's ship-blocking for the open-source release the thesis cites. Phase 8 if there's budget for a fourth thesis figure thread. Phase 11 (CI / pre-commit / lint sweep) is a Phase-10 sub-deliverable; not separate.

### D3 — Audit Phase-7 numbers if G7.2 / G7.9 looks weird

If `close_phase7` reports something that contradicts Phase-6's framing — e.g., **G7.2 PASS but the winning cell beats DQN +1336 by only 2σ** (close to noise) — re-run the relevant cell with a fresh seed pool to confirm the result isn't a single-seed artefact. Same protocol Phase 6 used for D6.2.1 (held-out split as the validation gate).

### D4 — Push the branch (the project has been local-only for ~50 commits)

`git push origin feature/reward-shaping` whenever the user is ready. The branch is currently +12 ahead at the time of this handoff but will be +12+more by the time the next agent reads this; check `git log --oneline origin/feature/reward-shaping..HEAD` to see the gap.

---

## Phase-7 PLAN snapshot (for context)

The locked PLAN at `docs/results/07_ablation/PLAN.md` has 9 sections + 10 D-decisions + 7 risks + 9 exit gates. Key D-decisions to know about:

| ID | Decision | Why it matters for review |
|---|---|---|
| **D7.1** | F9 grid is sparse one-at-a-time (12 cells, not 405) | If G7.2 fails and the user wants more coverage, this is the lever to revisit |
| **D7.2** | F9 + F10 train PPO only (D7.2 — best Phase-5 / Phase-6 trio) | If results are surprising, re-run with DQN/A2C for triangulation |
| **D7.3** | `impact_is_terminal` folded into F9 as one binary axis | Default `True` preserves Phase-3 contract; `False` is the F9 ablation |
| **D7.4** | F15 uses Phase-3 frozen reward config (isolates generalisation from reward-shaping) | If you want to test "best-F9-cell on OOD", that's a §3.2 OPTIONAL bonus, not the headline F15 |
| **D7.6** | F15 reuses frozen Phase-5 trained checkpoints (no retraining) | This is what makes F15 ~1 h CPU instead of ~7.5 h |
| **D7.9** | G7.2 success bar = DQN +1336 (deployable best), not PPO +1313 | Document |

### Pre-emptive D-decisions (already logged in PLAN §8)

- **D7.1.1** (placeholder) — activates if G7.2 fails. PASS-WITH-FINDING reformulation: "linear sweep characterised the limit of one-at-a-time Phase-3-style reward shaping."
- **D7.9.1** (placeholder) — activates if G7.9 fails. Narrow the thesis claim from "RL closes the OOD gap" to "RL is *robust to* the OOD class."

These both follow the AF3 protocol-continuity pattern (PASS-WITH-FINDING with original threshold preserved verbatim). The audit-first protocol's twin precedents are **D5.4.1** (Phase 5 de-escalation farming) and **D6.2.1** (Phase 6 rule-baseline dominance) — both reformulated under empirical pressure with the original threshold preserved.

---

## Phase-6 headline numbers (carried forward; same numbers as previous HANDOFF)

| # | Policy                          | Mean reward | 95 % CI | Cluster | Stage knowledge |
|---|---------------------------------|------------:|---|---|---|
| ★ | **Recommended-Action (rule)** ⓞ |    **+1624** | (1572, 1672) | oracle ceiling | true stage (oracle) |
| 1 | **DQN** (best deployable)        |    **+1336** | (1265, 1407) | trained-RL | none |
| 2 | PPO                              |       +1313 | (1253, 1372) | trained-RL | none |
| 3 | A2C                              |       +1297 | (1267, 1337) | trained-RL | none |
| 4 | RF-Acting (supervised + rules)   |       +1508 | (1455, 1565) | supervised+rules | RF-predicted stage |
| 5 | Always-BLOCK                     |        +520 | (483, 554)   | non-RL floor | none |
| 6 | Random                           |        +390 | (384, 398)   | non-RL floor | none |
| 7 | Always-OBSERVE                   |        −418 | (−421, −415) | non-RL floor | none |

ⓞ = oracle baseline (free `info["attack_stage"]` access; not deployable).
**Best deployable = DQN +1336 = 82 % of oracle ceiling +1624.** The +288 reward gap is the Phase-7 target.

---

## Audit findings status (carried forward, updated)

### AF1 — Promote OOD-conditioned eval to Tier 1 — **DONE in Phase 7**

F15 was scoped, implemented, and (as of this handoff) is being evaluated by the background runner. Plotter writes G7.8 + G7.9 verdicts to `F15_summary.json#gates` and `close_phase7.py` aggregates them.

### AF2 — Reframe Phase-6 §6.4 as oracle upper bound — **DONE in C1 (cdb609a)**

Phase-6 RESULTS.md now leads with "DQN +1336 = 82 % of oracle ceiling +1624" and frames the +288 absolute as the Phase-7 target.

### AF3 — Pre-empt the "moving-the-goalpost" reading — **STATIC defense-deck action; D7.1.1 + D7.9.1 placeholders preserve continuity**

Phase-7 PLAN §8 explicitly cites D5.4.1 + D6.2.1 as precedents and pre-registers D7.1.1 + D7.9.1 as the same-protocol fallbacks for G7.2 + G7.9 if they fail. The user still owes a defense-deck slide titled something like "When gates change: D5.4.1 / D6.2.1 / D7.X.1" listing the four reformulations and the original-threshold-preserved invariant.

### AF4 — Code hygiene before Phase 10 — **DEFERRED (still relevant)**

Unchanged from previous HANDOFF; same three items:

  1. `src/benchmarking/` (singular, dead) consumed by 43 tests
     (`test_benchmark_runner.py` + `test_metrics_collector.py`).
     Phase-5/6/7 pipelines use `src/benchmark/` (no g). Recommended:
     delete the package and its tests as one Phase-10 hygiene commit.
  2. Three pre-restart orphan scripts:
     `scripts/evaluate_generator.py`,
     `scripts/measure_improved_targets.py`,
     `scripts/separability_analysis.py`. Safe to delete.
  3. Repo-root `README.md` is pre-restart — does not mention Phases
     0-7, F-figures, or `make phase-N` targets. Phase 10 rewrites.

These are **not** Phase-7 work and the Phase-7 PLAN does not bundle them.

---

## The prompt — paste this verbatim into the next agent

```
You are taking over as mentor/engineer on a Master's thesis project at
/Users/felipe.santos/Projects/rl-iot-defense-system on the
`feature/reward-shaping` branch. The thesis is an extension of IoTWarden
(Bhattacharjee et al. 2023, see docs/papers/) using the CICIoT2023
dataset. Seven phases (0-6) are CLOSED; **Phase 7 implementation is
100 % committed** (12 commits, c1=cdb609a … e2d5f9d) but **the
real-data CPU runs are still in flight** at the moment you start. A
background runner + auto-finalizer are alive on the user's machine and
will land the C9 deliverables (G7_scoreboard.json + RESULTS.md +
CHANGELOG entry) without further intervention from you when phase7.end
appears (~10:41 PM 2026-04-30 if the runs started ~3:41 PM that same
day; check `cat logs/phase7/phase7.start` to confirm).

Your job depends on what state Phase-7 is in when the user gives you
the first instruction. Triage decision tree:

  (a) If `logs/phase7/phase7.end` does NOT exist when the user gives
      you the first instruction:
      - Phase-7 is still running. Tail logs/phase7/phase7.log to see
        progress. Estimate remaining wall-clock from the start
        timestamp (logs/phase7/phase7.start) + 7 hours.
      - Do NOT touch anything in runs/phase7/ — the running drivers
        are writing there.
      - Use the wait time to do read-only work: review the Phase-7
        commits, draft the §5/§6 hand-fill skeleton for RESULTS.md,
        or run `make phase-7-figures` ONLY if you want to confirm
        the partial-output figures still render (they will be
        incomplete until phase7.end).

  (b) If `logs/phase7/phase7.end` exists but `docs/results/07_ablation/G7_scoreboard.json`
      does NOT:
      - The auto-finalizer (`scripts/ablation/_finalize_phase7_background.sh`,
        PID logged in logs/phase7/finalize.pid) may have crashed
        before close_phase7 ran. Check logs/phase7/finalize.log for
        the failure. Re-run manually:
        `python -m scripts.ablation.close_phase7`

  (c) If `docs/results/07_ablation/G7_scoreboard.json` exists:
      - Auto-finalization succeeded. Read the scoreboard +
        F9/F10/F12/F15 captions + the auto-generated RESULTS.md.
        Your job is to FILL IN the (Hand-fill ...) sections (§5,
        §6.1-§6.4 — the thesis narrative). Each section maps to a
        gate verdict already pre-written in
        F<N>_summary.json#gates.G*.interpretation; the hand-fill is
        the chapter-prose phrasing of the same content.

**BEFORE YOU START STEP 0**: read this entire HANDOFF document. The
"Decision points for the next agent" section above lists four
orthogonal choices the user may want you to make (D1: hand-fill
RESULTS, D2: Phase 8 vs Phase 10 next, D3: re-audit if G7.2/G7.9 looks
weird, D4: push origin). The user's standing instruction is "make the
call, document it, only ask if it's a value judgement."

Your operating principles, learned from this project's history (NOT
NEGOTIABLE — six closed phases earned them):

  1. Audit-first: read the relevant code and PLAN.md before writing
     any new code, and write a PLAN.md for each NEW phase BEFORE
     touching code. PLAN.md must contain (a) the audit findings,
     (b) the deliverables, (c) the exit gates, (d) a sequencing table,
     (e) what we are NOT doing, (f) the risks tracked. The PLAN goes
     through a "lock decisions" commit before any implementation.

  2. Empirical gates: every phase has named exit gates G<phase>.<i>
     with explicit numerical thresholds. Run them on real data before
     calling the phase done. When a gate fails, treat the failure as
     diagnostic — it usually means the gate or the design has a hole,
     not that the phase is doomed. Phases 3, 4, 5, 6, AND 7 all
     turned "FAIL" into thesis-credible findings via dated D-decisions
     in PLAN §8 (B1-B6, D2.1, D5.4.1, D6.2.1, D6.8.1, and the
     pre-registered D7.1.1 / D7.9.1 placeholders).

  3. Hash-chain everything: each thesis figure ships with a
     manifest.json listing SHA-256 of inputs and outputs and the
     producing git SHA. This is what lets the defense narrative
     stay reproducible.

  4. Honest commit history: when a prior phase's bug or
     selection-bias artefact is discovered mid-phase, fix it as a
     `fix(phase-<N>):` commit attributed to the discovering phase,
     document it in the discovering phase's RESULTS.md §5, and decide
     *consciously* whether to rebuild downstream artefacts.
     Phase 7's `87b80dc` is the model: smoke surfaced 3 bugs that
     would have crashed 7.5 h of CPU; one fix commit named all three;
     no Phase-3/4/5/6 numbers were retroactively touched.

  5. Mentor-mode communication: brief, direct, lead with the result.
     Cite numbers, paper figures, gate IDs, commit SHAs by name.
     Don't bury the lede. The user is preparing a thesis defense; be
     honest. If something is shaky, say so plainly with the evidence.

=== STEP 0: Acclimate ===

Read these in order, in full:
  - docs/HANDOFF.md (this file)
  - CHANGELOG.md (top to bottom — eight [Unreleased] sections after
    auto-finalization, one per closed phase, with gate scoreboards
    and findings; the top one is Phase 7's, prepended by
    close_phase7.py)
  - docs/thesis_results_map.md (figure → phase mapping; all of
    F0..F15 + T1 are now Tier 1 / Tier 2 / Tier 3 placed; F13 + F14
    are the Phase-8 territory, T1 is Phase 5)
  - docs/results/07_ablation/PLAN.md (the locked Phase-7 plan — read
    §1 motivation, §3 deliverables, §3.4 gates, §6 risks, §8
    D-decisions in full)
  - docs/results/07_ablation/RESULTS.md (auto-generated RESULTS doc;
    fill the (Hand-fill ...) sections per D1)
  - docs/results/07_ablation/G7_scoreboard.json (canonical per-gate
    record)
  - docs/results/07_ablation/F{9,10,12,15}_{summary.json,caption.md,
    manifest.json} — every figure's live numbers + caption +
    SHA-256 chain
  - docs/results/06_benchmark/RESULTS.md §6.1 + §6.4 (the AF2-
    reframed oracle-ceiling framing — Phase-7 §1 cites this)
  - docs/results/{02_red_team,03_env,04_detector,05_blue_team}/RESULTS.md
    — Phases 2-5 closures; same shape as Phase 6 / Phase 7 RESULTS

You probably do NOT need to re-read the IoTWarden paper unless you're
starting Phase 8. F10 (the IoTWarden Fig. 6 re-implementation) was the
last reason to read it.

Then verify the project is in the state CHANGELOG claims:

  cd /Users/felipe.santos/Projects/rl-iot-defense-system
  git log --oneline -25
  source .venv/bin/activate
  pytest -q                                    # expect 442 passed
  ls docs/results/07_ablation/                 # F9/F10/F12/F15_*.{png,json,md} + G7_scoreboard.json + RESULTS.md
  cat docs/results/07_ablation/G7_scoreboard.json | jq '.gates[]|{id,passes,value}'
  ls runs/phase7/{ood,reward_sweep,aggressiveness}  # cell directories (gitignored)
  cat logs/phase7/phase7.log
  cat logs/phase7/finalize.log

If pytest != 442/442, STOP and surface the discrepancy. If the
runs/phase7/ directories are missing on this machine, the user wiped
them — re-run the sweeps:

  bash scripts/ablation/_run_phase7_background.sh &
  bash scripts/ablation/_finalize_phase7_background.sh &

(They take ~7.5 h total. Do not wait synchronously; tell the user
they can come back later.)

If runs/phase5/ and runs/phase6/ are missing, regenerate them first:

  make phase-5-sweep PHASE5_TIMESTEPS=250000   # ~108 minutes CPU (one-off)
  make phase-6                                 # phase-6-eval + phase-6-figures (~10 min)

=== STEP 1: Triage and act on the user's first instruction ===

The user is most likely to say one of:

  "review the Phase-7 results"  →  go to STEP 2
  "go" / "proceed" / "next phase"  →  go to STEP 3 (D2 decision)
  "the runs are still going; what's the status?"  →  show
     `tail logs/phase7/phase7.log` + `find runs/phase7/ -name 'eval_test.jsonl' | wc -l`
     and the active python processes (`ps -ef | grep -E 'scripts.(blue_team|ablation)' | grep -v grep`)
  "fill in RESULTS.md"  →  go to STEP 2

If the user gives a different instruction, default to STEP 2 (review
Phase-7 first); the right next move depends on whether the gate
verdicts are honest.

=== STEP 2: Review Phase-7 results (mandatory before any new work) ===

Run-only:

  pytest -q                                # confirm 442/442
  cat docs/results/07_ablation/G7_scoreboard.json | jq '.n_pass, .n_fail'
  cat docs/results/07_ablation/F9_summary.json | jq '.gates."G7.2"'
  cat docs/results/07_ablation/F15_summary.json | jq '.gates."G7.9"'  # the audit-AF1 HEADLINE
  cat docs/results/07_ablation/F10_summary.json | jq '.gates."G7.3"'
  cat docs/results/07_ablation/F12_summary.json | jq '.gates."G7.4"'

For each gate verdict (pass/fail), check that:

  (a) the F<N>_summary.json values are sensible (no NaNs, CI widths
      not absurdly wide, mean rewards in the expected ±2000 range)
  (b) the RESULTS.md auto-generated section reflects the same numbers
  (c) any FAIL-WITH-FINDING gates have their D7.X.1 placeholder
      activated in PLAN §8 (the placeholders are pre-registered;
      activation = filling in the date + observed numbers + new
      thesis-claim wording)

Write a 1-page review note to /tmp/phase7_review.md listing:

  - The ALL-GATES PASS / mixed verdict tally
  - Whether G7.2 closed the +288 deployable gap (and if so, by which
    cell + how much)
  - Whether G7.9 delivered the "RL closes the OOD gap" claim on
    VulnerabilityScan
  - Any anomaly worth a deeper look (CI width > 200, multi-σ
    non-overlap, rule baseline behaving strangely, etc.)
  - A go / no-go recommendation for Phase 8 (vs Phase 10 cleanup)

Present the review note to the user. Do NOT proceed to STEP 3
without their go-ahead.

=== STEP 3: Decide between Phase 8, Phase 10, or D1-D4 mop-up ===

Per "Decision points for the next agent" above, the choices are:

  D1 — Hand-fill the RESULTS.md (Hand-fill ...) sections
       (~1 h work; finish the thesis chapter)
  D2 — Phase 8 (F13 + maybe F14)  vs  Phase 10 (open-source hygiene)
  D3 — Re-audit Phase-7 numbers if anything is shaky
  D4 — Push origin

The user's standing instruction:
  - "I leave the decisions to you. What do you think is the best
    for my thesis defense/results?"
  - The default for D2 if user has < 1 week to defense is Phase 10
    (the README is a deal-breaker for the open-source release; Phase 8
    adds a fourth thesis claim but is novel work that needs its own
    PLAN cycle). If user has ≥ 1 week and the Phase-7 results are
    strong, Phase 8 is the better defense narrative.

Only ask the user when you genuinely need a value judgement. Make
calls confidently otherwise; document them in the next phase's
PLAN §8 with rationale.

=== STEP 4: If starting Phase 8 — same protocol as Phases 2-7 ===

  - docs/results/08_robustness/PLAN.md committed FIRST (audit-first).
  - F13 = noise / drift on the realiser; F14 = train-time OOD-class
    augmentation (complement of F15's eval-time OOD).
  - PLAN §8 must cite D7.1.1 + D7.9.1 + D6.2.1 + D5.4.1 as
    precedents (AF3 protocol-continuity).
  - Synthetic-only tests (mirror C3 + C4 pattern from Phase 7).
  - G8 exit gates with thresholds.
  - Real-data sweep + figure rendering.
  - Closeout: G8_scoreboard.json + RESULTS.md + CHANGELOG block.

=== STEP 4-alt: If starting Phase 10 — hygiene + open-source ===

  - docs/results/10_release/PLAN.md committed FIRST.
  - Delete `src/benchmarking/` + `test_benchmark_runner.py` +
    `test_metrics_collector.py` after confirming no Phase-5/6/7
    consumer (search `from src.benchmarking` across all sources).
  - Delete `scripts/evaluate_generator.py`,
    `scripts/measure_improved_targets.py`,
    `scripts/separability_analysis.py`.
  - Rewrite root `README.md`: project overview, the seven phases as
    numbered chapters with their headline figures inline, the
    `make phase-N` reproduction recipes, the dataset citation,
    the IoTWarden paper citation.
  - Update `CITATION.cff` for the thesis.
  - Tag the repo (`v0.1.0` or `phase-7-closeout`).
  - One CHANGELOG entry.

=== House rules (NOT NEGOTIABLE) ===

  - Always cd to the repo root before running commands.
  - Always use `source .venv/bin/activate` for python; bare
    `python` is not on PATH (Python 3.9.6 from CommandLineTools).
  - The processed dataset (`data/processed/ciciot2023/`),
    `runs/phase5/`, `runs/phase6/`, `runs/phase7/` are gitignored
    and live only on the user's machine. Synthetic-data tests must
    NEVER depend on them. Real-data smoke tests should mark
    themselves `pytest.skipif(not Path('data/processed/...').exists(), ...)`.
  - When a gate "fails", first ask "is the gate or the
    implementation wrong?" Phases 3, 4, 5, 6, AND 7's smoke run all
    had gates / designs that turned out to be wrong on the first
    contact with reality, and updating the gate or the design (with
    rationale captured in PLAN §8 D-decision or a `fix(phase-N):`
    commit) was the right move every time.
  - Commit messages follow conventional commits and cite the PLAN
    section being implemented (`feat(phase-8,§3.1.2): ...`).
  - Test the F15 `--smoke` if you ever change the OOD codepath. The
    smoke is the only thing that catches the "single-stage OOD class"
    design issue (87b80dc fix).

=== If the user is asking you to *review*, not implement ===

If the user's first instruction after handoff is "review what was
done so far" (any phrasing of audit / review / look at the results),
do this and only this until they say "go":

  1. Read everything in STEP 0.
  2. Run STEP 2 (the gate-by-gate scoreboard review).
  3. Inspect the four PNGs:
       open docs/results/07_ablation/F9_reward_ablation.png
       open docs/results/07_ablation/F10_aggressiveness.png
       open docs/results/07_ablation/F12_pareto.png
       open docs/results/07_ablation/F15_ood_robustness.png
  4. Walk back through commits cdb609a..HEAD with `git show --stat`
     to see what changed when. Confirm the *narrative* in
     07_ablation/RESULTS.md matches the actual commits.
  5. Look for one of:
       (a) numerical mismatches between RESULTS docs and JSON files,
       (b) tests that exist but are skipped without good reason,
       (c) artefacts referenced in CHANGELOG but absent on disk,
       (d) commits that touched unrelated files,
       (e) regressions in older tests masked by newer ones,
       (f) Phase-7 G7.2 / G7.9 verdicts — does the verdict match the
           direction of the bootstrap CIs in the F<N>_summary.json
           rows? (Plotter logic is in close_phase7.py + the four
           plot_*.py files; if you suspect a logic bug, re-derive
           the verdict from the raw rows.)
       (g) The hybrid OOD realiser correctness — F15's central
           design choice is that the agent sees in-distribution
           features at every non-OOD-stage step and OOD features
           only at the OOD class's stage. If the F15 mean rewards
           look wildly different from Phase-6 baselines on the
           three classes whose OOD stage isn't IMPACT, that's the
           hybrid realiser working as designed; on the IMPACT-only
           class (DDoS-HTTP_Flood) the means should be closer to
           Phase-6 because the IMPACT row is where the test fires.
  6. Write a one-page audit to /tmp/handoff_audit.md and present it.
     Either green-light Phase 8 / Phase 10 or list the blockers.

The user is preparing to defend a Master's thesis. Be honest. If
something is shaky, say so plainly with the evidence. The audit-first
protocol that has paid off five times (Phase 3 env bugs B1-B6,
Phase 4 discovering the Phase-1 OOD leakage, Phase 5 reframing G5.4
from the probe, Phase 6 reframing G6.2 from the held-out split,
Phase 7's smoke run catching the single-stage OOD class design issue
in 87b80dc) only works if the *next* agent also follows it.
```

---

## Quick links for the human

- Phase-7 PLAN: `docs/results/07_ablation/PLAN.md`
- Phase-7 RESULTS (auto-generated by close_phase7.py): `docs/results/07_ablation/RESULTS.md`
- Phase-7 figures: `docs/results/07_ablation/F{9,10,12,15}_*.png`
- Phase-7 numbers: `docs/results/07_ablation/F{9,10,12,15}_summary.json`
- Phase-7 manifests: `docs/results/07_ablation/F{9,10,12,15}_manifest.json`
- Phase-7 captions: `docs/results/07_ablation/F{9,10,12,15}_caption.md`
- Phase-7 scoreboard: `docs/results/07_ablation/G7_scoreboard.json`
- Background runner logs: `logs/phase7/{phase7,ood,reward,aggressiveness,pareto,finalize}.log`
- Background runner PIDs: `logs/phase7/{runner,finalize,phase7}.pid` / `phase7.pids`
- All phase RESULTS: `docs/results/{02_red_team,03_env,04_detector,05_blue_team,06_benchmark,07_ablation}/RESULTS.md`
- CHANGELOG: top section will be Phase 7 after auto-finalization (otherwise Phase 6's `d3e8ae1` block)
- Test count history: 254 (Phase 0) → 266 (Phase 1) → 283 (Phase 2) → 296 (Phase 3) → 329 (Phase 4) → 376 (Phase 5) → 420 (Phase 6) → **442 (Phase 7)**
- Figure-to-phase map: `docs/thesis_results_map.md` (Phase 7 owns F9 / F10 / F12 / **F15**; Phase 8 → F13 / F14)
- Phase-7 implementation commits at handoff: `cdb609a` C1 → `df14cc9` C2 → `4d19e81` C3 → `7268428` C4 → `e2dd145` C5 → `8c2636a` C6 → `e11485f` C7+C8 → `87b80dc` smoke fixes → `e2d5f9d` C9 scaffold → (auto) Phase-7 closeout via finalizer
