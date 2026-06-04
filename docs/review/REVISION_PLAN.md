# Thesis Revision Execution Plan

**Status:** NOT STARTED
**Owner:** revision agent (multi-session)
**Source reviews:** docs/review/THESIS_REVIEW.md + 3 external IEEE/journal reviews
**Last updated:** 2026-06-02

> READ THIS FIRST. This document is the single source of truth for the thesis
> revision. It is designed to be resumed from a cold start in any session.
> Before doing anything, read the "Resume Protocol" section, then the
> "Progress Ledger" at the bottom to see what is already done.

---

## 0. Hard Rules (never violate)

1. **pytest stays green at every commit.** Run `make format && make lint && make test`
   (in that order, per AGENTS.md) before every commit. CI runs `pytest -q`.
2. **Numbers in the thesis come from JSON, never hand-typed.** After the
   JSON→tex generator (task 2.5) exists, all headline numbers are `\input{}`
   macros. Hand-copying numbers is what caused every BLOCKER in THESIS_REVIEW.md.
3. **NEVER write "Phase N" / "PHASE N" into the thesis prose (tex/*.tex).**
   Use the semantic stage names (table below). "Phase 0–6" labels in THIS
   document are agent work-ordering only.
4. **Commit in baby steps.** One logical change per commit. Update the Progress
   Ledger after each commit.
5. **Update AGENTS.md ledger + this file's Progress Ledger** whenever a task
   completes, so the next session can resume.
6. **Back up `docs/results/` before any destructive re-run** (`cp -r docs/results docs/results.bak.$(date +%Y%m%d)`).
7. **Canonical data = `docs/results/**/*.json`** pinned by SHA-256 manifest chains.
   If prose and JSON disagree, JSON wins; regenerate prose from JSON.

### Semantic stage-name map (use these in thesis prose, NOT "Phase N")

| Code dir | Thesis stage name | (internal phase #) |
|---|---|---|
| `src/generator/` | Red-Team LSTM stage | 2 |
| `src/environment/` | Adversarial Environment stage | 3 |
| `src/detector/` | Stage Detector stage | 4 |
| `src/blue_team/` | Blue-Team Training stage | 5 |
| `src/benchmark/` | Held-Out Benchmark stage | 6 |
| `src/ablation/` (scripts/ablation) | Ablation & Robustness stage | 7 |

---

## 1. Author Decisions (locked)

These were chosen by the author and drive the whole plan. Do not relitigate.

1. **Seed count:** Re-run with **10 seeds** (`{0..9}`), 30 runs. (Resolves B1/B3/M5.)
2. **Primary contract:** **Full pivot** — re-train AND benchmark under
   `impact_is_terminal=False` as the PRIMARY result; retain `impact_is_terminal=True`
   as a smaller "reward-mis-specification case study". (Resolves the cross-review
   "primary benchmark under broken contract" BLOCKER.)
3. **New ablations:** Run **all three** — (1) RL trained WITH stage_pred input,
   (2) RF-Acting latency-reward sweep over {10,20,50,100} trees, (3) non-monotonic
   attacker stress test (small retreat probability).
4. **FPR remediation:** **Full constrained-MDP / Lagrangian** — add a `-beta*FPR`
   (or block-on-benign rate) penalty, retrain, report reward-vs-FPR tradeoff.
5. **Title:** **Revise** to remove "Proactive Attack Prediction". Pick from the
   candidates in §7.

---

## 2. Resume Protocol (cold-start checklist)

1. `git log --oneline -15` and read the Progress Ledger (§8) to see last completed task.
2. `make test` to confirm green baseline. Record count.
3. `git rev-parse HEAD` — record current SHA.
4. Check `docs/results/**/*.json` for which re-runs have landed
   (look at `n_seeds` fields and `impact_is_terminal` in summaries).
5. Continue from the first unchecked task in §8, respecting dependencies in §3.

---

## 3. Dependency Order (critical)

```
Phase 0 (memory) ─┐
Phase 1 (cheap text/format) ─┤  (independent, can run anytime, no data dep)
Phase 2 (code changes) ──────┤
                             ▼
Phase 3 (contract design) ── must precede ──► Phase 4 (big re-runs)
                                                   │
                                                   ▼
                                            Phase 5 (prose rewrite, needs new JSON)
                                                   │
                                                   ▼
                                            Phase 6 (final verify)
```
Prose numbers are UNKNOWN until Phase 4 finishes. Do NOT write headline numbers
before then. Abstract is written LAST.

---

## 4. Phase-by-Phase Tasks

### Phase 0 — Setup & Memory (no re-run)
- [ ] 0.1 Write this file (`docs/review/REVISION_PLAN.md`) and the 3 external
      reviews to `docs/review/` so they are git-tracked.
- [ ] 0.2 Add `## Thesis Revision Progress` ledger to AGENTS.md (decisions +
      phase checklist + "current canonical numbers" placeholders).
- [ ] 0.3 Record baseline: `make test` count, `git rev-parse HEAD`.
- [ ] 0.4 Commit: `docs: add thesis revision plan + AGENTS.md progress ledger`

### Phase 1 — Cheap Text / Format Fixes (no re-run; parallelizable)
- [ ] 1.1 `SUB\section` → `\subsection` (25 occ): methodology.tex (11,14,20,47,55,
      58,96,106,116,165,168,171,198,203,206), background.tex (28,39,54,95,98,107,132),
      conclusao.tex (58,66,74). [M1]
- [ ] 1.2 Remove duplicate `\usepackage[utf8]{inputenc}` at principal.tex:15. [n1]
- [ ] 1.3 `tex/thesis.pdf` → `tex/principal.pdf` in apendice.tex (4,80,86). [M4]
- [ ] 1.4 Standardize "thesis"→"dissertation" in English body (background.tex heavy). [S2.4]
- [ ] 1.5 Verify ToC depth ≤3 after 1.1 (the SUB fix restores hierarchy). [S2.5/2.6]
- [ ] 1.6 tese.bib: wrap all titles in `{{...}}` (n2); add reward-hacking cites
      (Amodei 2016, Krakovna 2020, Leike 2018); add 2–3 recent (2024–25) RL-IDS
      refs; footnote IoTWarden/Alam2024 as preprint (m5). [P1.5, P1.6, S8]
- [ ] 1.7 `make thesis` compiles after SUB fix; commit per logical group (3–4 commits).

### Phase 2 — Code Changes Before Re-runs (each with tests, pytest green)
- [ ] 2.1 **stage_pred-in-training** plumbing. Inject frozen classifier handle in
      `AdversarialEnv.__init__` (adversarial_env.py:329-340); add predicted-stage
      to obs in `_build_observation()` (:637-646); widen obs space (:342-350);
      gate behind new flag `include_stage_pred` (default False). Unit tests. [rev 2.4(1), C1]
- [ ] 2.2 **RF tree-count sweep** plumbing. Add `--n-estimators` to
      train_detector.py:375; thread `RandomForestConfig(n_estimators=...)`
      (random_forest.py:36,63). Test. [rev 2.4(2), S4, C2]
- [ ] 2.3 **Non-monotonic attacker** plumbing. Add `retreat_prob` to red-team
      sampling (transition_mask.py:73-80 / attack_sequence_generator.py sample_next),
      default 0 (monotonic preserved). Test. [rev 2.4(3), 3.5]
- [ ] 2.4 **Lagrangian FPR penalty**. Add episode-level benign-block accumulator
      in env + terminal `-beta*FPR` correction in `_calculate_reward`/`step`
      (:685-742 / :525-558). New config `fpr_penalty_beta` (default 0). Calibration
      tests. (Largest new-code item; only per-step block-on-benign exists today.) [rev 2.2, C4, Dir6]
- [ ] 2.5 **JSON→tex generator** (anti-drift). New `scripts/thesis/render_tables.py`:
      reads F5/F8/F9/F10/F15 summaries + G5/G6/G7 scoreboards + benign_fpr.json;
      emits `tex/generated/*.tex` table fragments + `tex/generated/numbers.tex`
      `\newcommand` macros (e.g. `\BestAgentReward`, `\OracleCeiling`, `\OracleCapturePct`,
      `\LatencyRatio`, `\BenignFPR`). results.tex `\input`s these. Test that it runs
      and macros resolve. [THESIS_REVIEW P4.7, B3]
- [ ] 2.6 Makefile: `BLUE_TEAM_SEEDS ?= 0 1 2 3 4 5 6 7 8 9`; add targets for the
      3 new ablations + RF sweep + `render-tables`; fix `reproduce-thesis` to full
      chain (dataset→red-team→detector→blue-team@250k→benchmark→ablation→smoke). [B1, m1]
- [ ] Commit per item. Run smoke targets after 2.1–2.4 to catch env/config drift.

### Phase 3 — Primary-Contract Design (short, no big run)
- [ ] 3.1 Confirm `impact_is_terminal=False` is trained+benchmarked primary;
      `=True` is the smaller case-study run.
- [ ] 3.2 Equalize benchmark to **n=300 episodes for ALL policies** (fix n=150/300
      asymmetry). [rev S/M7]
- [ ] 3.3 Confirm `p_de_esc=0.6` carried through both contracts.
- [ ] 3.4 Record decisions in AGENTS.md ledger. Commit.

### Phase 4 — Big Re-runs (walk-away; back up docs/results first)
**Staleness-Elimination Protocol (new, critical):**
- **4.0 Pre-run clean slate:**
  - Back up: `cp -r docs/results docs/results.bak.$(date +%Y%m%d)` and `cp -r tex/figs tex/figs.bak.$(date +%Y%m%d)`.
  - `make clean-runs` (wipes `artifacts/ runs/ results/ mlruns/`).
  - Verify `runs/` is empty; remove any leftover `.log` files or stray dirs under `runs/`.
  - **Delete all figure PDFs and JSONs** from both `docs/results/**` and `tex/figs/`
    for everything Phase 4 regenerates (F3,F4,F5,F6,F7,F8,F9,F10,F12,F15 + summaries
    + scoreboards + benign_fpr.json), so a missing-output bug fails loudly instead
    of leaving a stale file. Keep figures Phase 4 does NOT touch (F0, F1, F2, F11,
    FA_*, architecture).
- **4.x Per-stage, regenerate + sync:**
  - Each plot step writes to `docs/results/<NN>_*/`.
  - After every plot step, run `make sync-figures` (new target, writes into Makefile
    in task 2.6) which copies every thesis-consumed PDF from `docs/results/**`
    → `tex/figs/` deterministically. Fails if source is missing.
  - For figures whose scripts already write directly to `tex/figs/` (FA_action_cost,
    FA_window — Appendix D), leave as-is.
- **4.7 Post-run stale check:**
  - Assert every `\includegraphics` path in `tex/*.tex` resolves to a file
    regenerated in this run (`make stale-check`, new target from task 2.6).
  - `reproducibility_smoke` PASS.
  - Diff `docs/results.bak.*` vs new `docs/results` to produce a "what changed" report.

**Run stages (by dependency):**
- [ ] 4.1 `make dataset`, `make red-team`, `make detector` (+ RF {10,20,50,100} variants).
- [ ] 4.2 Blue-team sweep: 10 seeds × 3 algos under `impact_is_terminal=False`
      (primary); then smaller `=True` case-study run.
- [ ] 4.3 Benchmark eval (n=300 all policies) → F5/F6/F7/F8 + benign_fpr.json + G6.
- [ ] 4.4 Ablations: F9 (10 seeds), F10 (10 seeds), F15 (10 seeds), + stage_pred-in-training,
      + RF tree-count latency-reward sweep, + non-monotonic stress test,
      + FPR-penalty beta-sweep (reward-vs-FPR curve).
- [ ] 4.5 F12 Pareto: with new RF tree-count + FPR points, build a real ≥3-point
      frontier; else remove. [M2]
- [ ] 4.6 Statistical tests n=10: ALL pairwise (incl. PPO-vs-RF, A2C-vs-RF) +
      Bonferroni. [S3, M5]
- [ ] 4.7 `make reproduce-thesis` + reproducibility_smoke PASS; record wallclock;
      commit regenerated `docs/results/**` + figures.
      Commit: `data: 10-seed re-run, impact_is_terminal=False primary, new ablations`
- [ ] 4.8 Run `scripts/thesis/render_tables.py`; commit `tex/generated/*`.

### Phase 5 — Prose Rewrite From New Data (uses macros; nothing hand-typed)
Use semantic stage names, NOT "Phase N".
- [ ] 5.1 **Title** (preambulo.tex:169): pick from §7; reframe all "proactive"
      usages → stage-anticipation tooling, agent is reactive-mitigation. [rev 1.1, 3.2]
- [ ] 5.2 **Abstract/Resumo** (principal.tex; write LAST): new best agent, oracle
      capture %, explicit ~FPR caveat + FPR-fix result, latency ratio for actual
      best agent, fix garbled "chain of hashes" sentence, mention reward-mis-spec. [B6]
- [ ] 5.3 **Introduction contributions** (introducao.tex): replace leakage-bug
      contribution with reward-mis-specification structural analysis; caveat latency
      with FPR; numbers via macros. [rev 3.2, 3.3]
- [ ] 5.4 **Results** (results.tex): regenerate tables from JSON; pivot primary to
      `=False`; demote `=True` to case study; replace F5_table.pdf image with native
      LaTeX table; "floor"→"oracle ceiling"; add benign-FPR column; reframe OOD as
      limitation; remove gate-code jargon→appendix; fix Fig x-axis 250k vs 500k. [B2-B7,M3,M7]
- [ ] 5.5 **Methodology** (methodology.tex): 29-feature appendix table; disclose
      i.i.d. feature-sampling assumption; reward pseudocode (Algorithm 1); justify
      MLP-vs-RF gap; OOD-class selection criteria; Mirai-greeth/greip MANEUVER
      justification; clarify oracle is current-stage. [rev 2.3,5.3,5.7,S9,S10]
- [ ] 5.6 **Conclusions** (conclusao.tex): rename "Findings Worth Defending"→
      "Principal Empirical Findings"; remove JSON-scoreboard sentence; add
      compromise_rate≡1.0 = reactive-mitigation limitation; elevate FPR +
      reward-mis-spec; reprioritize future work (FPR/OOD-aug first). [rev 7.x, 8.3]
- [ ] 5.7 **Background/Related work** (background.tex): +15–25 citations; algorithmic-
      choice discussion (why PPO/A2C/DQN); consolidate oracle "measurement instrument"
      repetition to one statement. [rev 3.5, 4.2, S8]
- [ ] 5.8 **Threats to validity**: simulator fidelity, reward-coefficient sensitivity,
      FPR operational-cost back-of-envelope, n=10 power-analysis footnote. [rev 4.6,8.4,8.7]
- [ ] 5.9 **Appendices** (apendice.tex): replace placeholder pages
      (ficha catalográfica + signatures — AUTHOR-SUPPLIED docs, leave clear stubs);
      fix 82%/81%, test counts, seed count→10, hash-verification caveat. [M6, rev 2.1, 8.1]
- [ ] Commit per chapter (~6–8 commits).

### Phase 6 — Final Verification
- [ ] 6.1 `make thesis` compiles clean; no "SUB" in PDF; numbers match JSON.
- [ ] 6.2 `make format && make lint && make test` green.
- [ ] 6.3 `python -m scripts.reproducibility_smoke` PASS.
- [ ] 6.4 Subagent verification: every headline number in PDF == canonical JSON.
- [ ] 6.5 Final AGENTS.md + ledger update.
      Commit: `fix(thesis): reconcile all numbers with 10-seed impact_is_terminal=False re-run`

---

## 5. Ground-Truth Numbers At Time Of Planning (5-seed, =True data)

These are the OLD canonical values (pre-re-run), kept for reference so the agent
can detect when the re-run has landed (numbers + n_seeds will change).

- Best deployable RL on test (5-seed, =True): **DQN +1336.3** (NOT PPO +1312.6).
- Oracle ceiling (test): **+1624.4** (thesis prose wrongly says +1647.6).
- RF-Acting: +1507.9 | A2C +1296.7 | always_block +519.7 | random +390.3 | always_observe −418.2.
- Oracle capture (DQN): 1336.3/1624.4 = **82%** (prose says 81% on PPO).
- Latency p50: DQN 0.068ms, PPO 0.100ms, A2C 0.101ms, RF 13.976ms.
- mitigated_impact_rate (=True): DQN 0.153, PPO 0.28, A2C 0.253; always_block 1.0.
- F9 `impact_is_terminal=False`: +1541.9, mit-rate 0.900.
- benign_fpr.json: PPO 9.6%, A2C 9.4%, DQN 12.7%.
- compromise_rate ≡ 1.0 across all policies (structural).
- Source files: docs/results/benchmark/main_results.json, G6_scoreboard.json;
  blue-team-training/blue_team_acceptance.json; ablation/reward_ablation.json, F15_summary.json.

---

## 6. Known Code Facts (from exploration, save re-discovery)

- `impact_is_terminal` is NOT in config.yml; default `True` at
  src/environment/adversarial_env.py:201; consumed at :525; threaded via
  blue_team/run_config.py:76, env_factory.py:65, scripts CLI.
- Obs built in `_build_observation()` adversarial_env.py:637-646; space at :342-350.
  **No `stage_pred` field exists in obs/info today** — 2.1 must add it.
- Reward in `_calculate_reward()` :685-742; per-step block-on-benign penalties
  exist (penalty_block_benign :223, applied :721-722) but NO episode-level FPR
  accumulator — 2.4 must add it.
- RF `n_estimators=100` at random_forest.py:36; train_detector.py:429 has no
  `--n-estimators` flag — 2.2 must add it.
- Monotonic mask in transition_mask.py:60-91 (`allow_regression` flag :39).
- benign FPR computed only in plot_stage_action_cm.py:265-324 → benign_fpr.json.
- **No JSON→tex tooling exists; all tables hand-written** — 2.5 builds it.
- `main.py` + config.yml reward section are STALE/off-repro-path (main.py would
  TypeError on nonexistent kwargs). Live coefficients = EnvConfigSerializable /
  _PHASE3_DEFAULTS. Do NOT trust config.yml reward values. (Optional fix only.)
- Raw + processed CICIoT2023 present locally; runs/phase5 present. Re-run feasible.

---

## 7. Title Candidates (pick one in task 5.1)

Current: *"An Adaptive Defense System for IoT Networks Using Proactive Attack Prediction and Deep Reinforcement Learning"*

1. **"A Kill-Chain-Aware Deep Reinforcement Learning Framework for IoT Intrusion
   Response: A Reproducible Benchmark on CICIoT2023"**
   — emphasizes the reproducibility/benchmark contribution; drops "proactive".
2. **"Kill-Chain-Aware Deep Reinforcement Learning for Autonomous IoT Network
   Defense on CICIoT2023"**
   — concise; "autonomous" instead of "proactive"; names dataset.
3. **"Adaptive IoT Network Defense via Kill-Chain-Aware Deep Reinforcement Learning
   and LSTM Threat Modeling"**
   — keeps "adaptive", frames LSTM honestly as threat modeling (not prediction by agent).
4. **"An Adaptive Kill-Chain-Aware Defense Framework for IoT Networks Using Deep
   Reinforcement Learning and LSTM-Based Attack-Stage Modeling"**
   — closest to original wording; swaps "Proactive Attack Prediction" →
   "LSTM-Based Attack-Stage Modeling" (most honest minimal change).

Recommendation: **#1** if leaning into the reproducibility story (reviews praise
it most); **#4** for minimal deviation from the registered title.

---

## 8. Progress Ledger (update after every commit)

| Task | Status | Commit SHA | Notes |
|---|---|---|---|
| 0.1 plan file | ☐ | | |
| 0.2 AGENTS.md ledger | ☐ | | |
| ... | | | |

**Baseline (pre-revision):** HEAD=<sha>, `make test`=<N> passed.
**Current canonical numbers:** see §5 until Phase 4 re-run lands, then update here.
