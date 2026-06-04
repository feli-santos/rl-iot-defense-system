# Context Ledger — Prevention Pivot

> **Status:** ACTIVE. This is the single load-bearing state-tracking document for the
> thesis re-centering ("prevention pivot"). It is **append-mostly**: edit the live-state
> sections (1, 2, 8) in place; never delete an entry from the Drift Register (§5),
> Hypotheses Register (§3), or Journal (§10) — supersede it instead.
>
> **Why this file is load-bearing:** the rename from codename artifacts (`F5`, `G7`,
> `06_benchmark`, …) to semantic names is a **pure delete** (git-rm; we rely on git
> history only — no `PROVENANCE.md`, no safety-tag). Therefore **this ledger is the SOLE
> human-readable record of the codename→semantic mapping** (§6) and of every locked
> decision behind the pivot. If this file is wrong, provenance is lost.

---

## 0. Pin

| Field | Value |
|---|---|
| **Pre-pivot baseline tag** | `v0.3.0` → commit `1f57936` (annotated) |
| **Pivot branch** | `feat/prevention-pivot` (off `1f57936`) |
| **Prior thesis HEAD** | `cd4841b` (thesis revision complete, 86pp) |
| **Venue lens** | IEEE Internet of Things Journal (IoT-J). "Any journal" is the MSc floor; IoT-J is the aim. |
| **Author** | Felipe Augusto Oliveira dos Santos — MSc, UNICAMP/FEEC. Advisor: Prof. Dr. Denis Fantinato. |
| **Headline claim (target)** | Environment + reward modeling determine whether a deep-RL defender can **PREVENT** compromise (not merely mitigate). Finite attacker budget makes `compromise_rate` a function of policy quality. |
| **Current title** | TBD at Phase F. Working title MUST drop "LSTM" and likely "Adaptive". Old title (retired): *"An Adaptive Kill-Chain-Aware Defense Framework for IoT Networks Using Deep Reinforcement Learning and LSTM-Based Attack-Stage Modeling."* |
| **Python** | `.venv` = 3.9.6 (results platform). CI = 3.10 + 3.11. (system python is 3.14 — do NOT use it; always `.venv/bin/python`.) |
| **Test baseline** | 459 passed, 2 warnings (canonical at `1f57936`). Target after Phase C: ~471–474 (new budget/Markov/outcome tests). |

---

## 1. Current State

- **Phase:** A (writing this ledger). Mode: BUILD (no longer read-only).
- **Repo:** clean working tree on `feat/prevention-pivot` at `1f57936`. `tex/tese.pdf`
  remains untracked by design (stray duplicate; AGENTS.md rule).
- **Nothing re-run yet.** All headline numbers below are **pre-pivot** and will be
  REPLACED by the Phase-D re-run on the Markov+budget MDP. No pre-pivot number is
  contractually retained.
- **Next action after Phase A:** Phase B (rename emitters, prove suite green on OLD data).

### The pivot in one paragraph
We drop the LSTM red-team (a frozen 97.7%-fidelity imitator of a hand-built 5×5 Markov
matrix — no expressive power, no adaptivity, falsely in the title) and promote that Markov
transition matrix to *be* the attacker. We add a **finite attacker budget** so that correct,
timely kill-chain defense can **exhaust the attacker before IMPACT** → compromise becomes
*preventable* and `compromise_rate` drops below 1.0 as a function of policy quality. We add an
**evasion-before-commit reactive attacker** (probabilistic stall/retreat in anticipation when
recently blocked). The Cyber Kill Chain stays — it is now *more* central (semantic spine +
anti-toy-MDP shield). Every observation remains a **real CICIoT2023 flow row** sampled per
stage by the RealizationEngine (this is the dataset's load-bearing role).

---

## 2. Long-Running Experiment State

> Update this table as runs launch/finish. Times are walk-away CPU estimates.
> "Substrate" = MDP the run assumes. After the pivot, the live substrate is **Markov + budget**.

| Experiment | Substrate | Status | Owner phase | Est. time | Output (semantic name) |
|---|---|---|---|---|---|
| Budget micro-sweep (calibration) | Markov+budget | NOT STARTED | C-cal (HARD GATE) | ~1–2 h | `ablation/budget_calibration` |
| Blue-team smoke | Markov+budget | NOT STARTED | D-smoke | ~20 s | (transient) |
| 1-seed budget probe | Markov+budget | NOT STARTED | D-smoke | ~min | (transient) |
| Blue-team sweep (10 seed × 3 algo) | Markov+budget | NOT STARTED | D | ~3–7 h | `blue-team-training/*` |
| Held-out benchmark | Markov+budget | NOT STARTED | D | ~10 min | `benchmark/benchmark_main_results` |
| Cocked-trigger spine (True vs False) | Markov, **budget=None** | NOT STARTED | D (disjoint cells) | (part of sweep) | `benchmark/reward_contract_*` |
| Outcome-only reward cell | Markov+budget | NOT STARTED | D | (rides F9 harness) | `ablation/reward_ablation` |
| Evasion-reactive attacker condition | Markov+budget+evasion | NOT STARTED | D | (part of sweep) | `ablation/reactive_attacker` |
| OOD robustness | Markov+budget | NOT STARTED | D | ~part of 7.5 h | `ablation/ood_robustness` |
| Budget/aggressiveness sensitivity | Markov+budget | NOT STARTED | D | ~6 h | `ablation/*_sensitivity` |

---

## 3. Hypotheses Register

> H = hypothesis under test by the re-run. Each has a **clean** and a **fallback** reading
> so we are never tempted to pretend a messy curve is clean (locked user posture, m0065).

| ID | Hypothesis | Clean outcome | Fallback reading (still publishable) | Decided by |
|---|---|---|---|---|
| **H1 (SPINE)** | Finite budget makes prevention a function of policy quality. | `compromise_rate` is a clean monotone/sigmoid fn of budget; better policies shift the curve. → "prevention" headline. | Non-monotone/degenerate curve → pivot spine to *"we characterize WHEN prevention is/isn't achievable"*. | Phase C-cal |
| **H2** | Naive `impact_is_terminal=True` induces a cocked-trigger reward-misspecification pathology. | True-contract benchmark shows earlier/over-aggressive blocking + higher benign FPR than False, within-algorithm. | If it doesn't replicate under budget, retain as `budget=None` case study OR demote. | Phase D |
| **H3** | Proportionality-shaped RL matches the RF-Acting+rules pipeline at ~146× lower latency but cannot exceed it. | Latency win re-derives automatically (forward-pass property, MDP-independent). | (Lowest risk — essentially guaranteed.) | Phase D |
| **H4** | Outcome-only reward degenerates to blunt late-blocking (proportionality shaping is load-bearing). | Outcome-only agent learns delay/mitigate, not prevent; worse proportionality. | Either direction is a finding about shaping. | Phase D |
| **H5** | RL inherits but does not cure the detector's VulnerabilityScan blind spot (recall ≈0.001). | RL robust-to but not better-than baselines on OOD class. | (Already observed pre-pivot; expected to hold.) | Phase D |

---

## 4. Curation Ledger (KEEP / CUT / DEFER)

| Item | Decision | Rationale |
|---|---|---|
| Cyber Kill Chain (5 stages) | **KEEP (promote)** | Semantic spine + anti-toy-MDP shield (Lockheed-Martin citable). More central post-pivot. |
| LSTM red-team (`src/generator/attack_sequence_generator.py`, `src/training/generator_trainer.py`) | **CUT (delete)** | Frozen imitator of the Markov matrix; no expressive power/adaptivity; falsely in title. |
| 5×5 Markov transition matrix (`episode_generator.py:220-270`) | **KEEP (promote to attacker)** | Becomes the actual attacker dynamics. |
| Real CICIoT2023 per-stage observation (RealizationEngine) | **KEEP (foreground)** | The dataset's load-bearing role; the real-observation anchor for face validity. |
| Stage detector (MLP/RF/CNN1D) | **KEEP (reframe)** | Baseline apparatus (RF-Acting hero comparator + optional `include_stage_pred` obs-aug), NOT a contribution (C8). |
| Multi-baseline suite (random / always-observe / always-block / recommended_action oracle / RF-Acting) | **KEEP** | Genuine strength; RF-Acting is the hero comparator, oracle is the ceiling. |
| Reproducibility (SHA-256 manifest hash-chain) | **DEMOTE** | 2026 table-stakes/hygiene, not a primary contribution. Move from contribution #5 → appendix/subsection. |
| Red-Team LSTM-imitation gates (G1–G5) | **CUT** | Measure imitation fidelity of a module we are deleting. |
| Unbounded-MDP `compromise_rate=1.0` result | **KEEP as single contrast** | One figure/subsection: "why naive absorbing-IMPACT modeling makes prevention impossible" = control proving prevention comes from the agent, not env rigging. User may veto retention at Phase F. |
| Falsified OOD mechanism prose (`results.tex:287,290`) | **CUT (delete sentences)** | Factually false (see §5/D-OOD). Replace with honest bounded-generalization finding. |
| Finite attacker budget | **ADD** | New headline mechanic (see §7). |
| `reward_mode="outcome_only"` | **ADD** | New ablation code path (~25–45 LOC). |
| Evasion-before-commit reactive attacker | **ADD** | The "adaptive" substance (defender-coupled), modest code. |

---

## 5. Drift Register

> Pre-existing inconsistencies (D1–D10) discovered in the read-only audit + the confirmed
> code bug (C9). **Never delete a row** — mark RESOLVED with the fixing commit when handled.
> "Phase" = where it gets fixed.

| ID | Drift / defect | Status | Fix phase |
|---|---|---|---|
| **D1** | Test-count 459-vs-411 conflict. | RESOLVED (audit): **459 canonical** (`test_count.json` + RESULTS + AGENTS); the 411 in G4/G5/G7 scoreboards + `02_red_team/RESULTS` ("411 at HEAD") are STALE. | B/E (scoreboards retired in rename) |
| **D2** | `F9_summary.json` nested `gates.G7.2` block + `headline` still carry pre-rerun +1624/+1336.3/0.153 although top-level updated. | OPEN | D/E (regenerated) |
| **D3** | `architecture.md`/`decisions.md` say `MlpPolicy`; code uses `MultiInputPolicy` (Dict obs). | OPEN | F |
| **D4** | `reproducibility.md` says Python 3.12 (CI is 3.10–3.11) and omits the hash-chain. | OPEN | F |
| **D5** | `environment.md` says "reward vs previous stage", contradicting the decision-time convention. | OPEN | F |
| **D6** | `memory-bank/` referenced in AGENTS.md but MISSING. | OPEN | B/F (create or de-reference) |
| **D7** | `REVISION_PLAN.md` header says "NOT STARTED" but the revision is done. | OPEN | F (or supersede) |
| **D8** | `runs/phase5` vs `runs/blue_team` path divergence in scoreboard strings. | OPEN | C (provenance fix) |
| **D9** | AGENTS.md references `--verify-manifests` flag + `scripts.ablation.close_phase7` — NEITHER EXISTS (real script: `close_ablation.py`). Reproducibility-verify claim currently unbacked by code. | OPEN | C (provenance fix) |
| **D10** | Dev tools (ruff/black/pre-commit) not installed in `.venv` (system-level only). | OPEN | C (provenance fix) |
| **D-OOD** | `results.tex:287,290` explains VulnerabilityScan with a FALSEHOOD ("RF mostly predicts BENIGN→OBSERVE which is cheap"). Falsified 3 ways: detector predicts IMPACT (92%), always_observe is the WORST policy there (-420.9), RF OOD distribution never saved. Finding itself is GENUINE. | OPEN | F (delete mechanism, keep finding) |
| **D-MIRAI** | `methodology.tex:59` cosmetically mislabels Mirai-udpplain as IMPACT (mapper says MANEUVER). | OPEN | F |

### Caveat Ledger (C1–C9) — implementation hazards for the new mechanics

| ID | Caveat | Disposition |
|---|---|---|
| **C1** | `step()` :544–552 is IF/ELSE: de-escalation XOR `_advance_attack`. Budget step-cost decrement belongs INSIDE the else-branch at :551; de-escalation charges `reset_cost` only (avoid double-charge). Post-reset BENIGN→RECON re-climb steps are partially free under the "drain only when stage≥RECON" gate → calibration MORE sensitive. | Implement per this in Phase C. |
| **C2** | Grace clamp (:562–570) downgrades pre-`min_episode_length`(20) IMPACT to MANEUVER. Early-exhaustion "preventions" in the grace window are NOT defender-attributable → inflate prevention-rate. | Report prevention-rate CONDITIONED on `step≥min_episode_length`; exhaustion check sits AFTER the grace clamp. |
| **C3** | **(biggest scientific risk, unresolvable on paper)** De-escalation has COUPLED opposing effects: resets cost budget (drain) BUT restart the episode clock → longer episodes → more step-cost drain AND more opportunity. "Good defender prevents" vs "good defender just prolongs" is EMPIRICAL. | Resolved only by Phase C-cal. |
| **C4** | Naive reactive-retreat would double-count the existing de-escalation reset. | Resolved → evasion-before-commit (anticipatory, orthogonal). |
| **C5** | Dropping LSTM = MODULE-DELETION (111 grep matches in `src/`), not a comment sweep. Dead: `attack_sequence_generator.py`, `generator_trainer.py`, `dataset_loader.py:load_lstm_data`, `config_loader.py:get_lstm_config`, parts of `dataset_processor.py`, `episode_generator.py:episodes_to_training_sequences/episodes_to_numpy`. Env: drop import `adversarial_env.py:49` + `.pth` load :386–388 + rewire `_advance_attack:719-739`. | Phase C. |
| **C6** | "Which 29 features & why" — selection via `variance_threshold=0.01`, `correlation_threshold=0.95`; NOT confirmed a committed artifact LISTS the 29 surviving column names. Reviewer will ask. | DEFERRED → verify Phase E, prose Phase F. |
| **C7** | THROTTLE (action=2) can never trigger de-escalation (`_maybe_defender_deescalation` requires action≥3, :689) → strictly DOMINATED action under prevention spine. | Note/justify in Phase F prose. |
| **C8** | Detector role shrinks (attacker no longer neural). Stays as baseline apparatus. | Framing, Phase F. |
| **C9** | **(CONFIRMED REAL BUG)** `run_test_eval.py:_eval_env_spec()` (:147) builds `EnvConfigSerializable(...)` with defaults → `impact_is_terminal=True` (run_config.py:76), but training passes `--impact-is-terminal false`. Every RL agent in `F5_summary.json` was TRAINED under False but EVALUATED under True = train/eval terminal-contract MISMATCH. Affects all F5 RL mitigation/reward numbers; does NOT change structural `compromise_rate=1.0`. | FIX for free in Phase D re-run. |

---

## 6. Rename Taxonomy (codename → semantic)

> **Pure delete** (git-rm). This table is the sole record. `render_tables.py` macro NAMES are
> already semantic — KEEP them. JSON internal keys (`rows`, `mean_reward`, `policy`) are schema
> — KEEP them. Only rename **dirs / files / hardcoded paths**. Retire the `scoreboard`/`gate`/`G#.#`
> vocabulary and eliminate "Phase N" from scientific output (it remains fine in dev docs).

### Directories
| Codename | Semantic |
|---|---|
| `01_dataset/` | `dataset/` |
| `02_red_team/` | `red-team-model/` *(largely CUT — LSTM gates removed; keep only what survives)* |
| `03_env/` | `environment/` |
| `04_detector/` | `stage-detector/` |
| `05_blue_team/` | `blue-team-training/` |
| `06_benchmark/` | `benchmark/` |
| `07_ablation/` | `ablation/` |

### Files (figure/summary JSONs)
| Codename | Semantic |
|---|---|
| `F0_summary` | `dataset_summary` |
| `F1_summary` | `red_team_gates` *(likely CUT with LSTM)* |
| `F5_summary` | `benchmark_main_results` |
| `F6_summary` | `stage_action_proportionality` |
| `F7_summary` | `latency_profile` |
| `F8_summary` | `reward_ranking` |
| `F9_summary` | `reward_ablation` |
| `F10_summary` | `aggressiveness_sweep` |
| `F11_summary` | `detector_summary` |
| `F12_summary` | `pareto_frontier` |
| `F15_summary` | `ood_robustness` |
| `benign_fpr` | `benign_fpr` (keep) |
| `G4/G5/G6/G7_scoreboard` | `*_acceptance` (vocabulary retired) |

### New (post-pivot) outputs
`benchmark/reward_contract_{true,false}`, `ablation/budget_calibration`,
`ablation/reactive_attacker`, prevention-curve figure under `benchmark/` or `ablation/`.

---

## 7. Figure Staging

> Figures the thesis will need from the re-run. Status updated in Phase E.

| Figure | Source (semantic) | Status |
|---|---|---|
| Prevention curve: `compromise_rate` vs budget, per policy (conditioned on step≥min_len) | `ablation/budget_calibration` | NOT STARTED |
| Benchmark ranking table (reward, mit-rate, FPR, latency, compromise_rate) | `benchmark/benchmark_main_results` | NOT STARTED |
| Latency–reward trade-off (the ~146× pillar) | `benchmark/latency_profile` | NOT STARTED |
| Reward-contract True-vs-False (within-algorithm) + per-stage action dist + FPR | `benchmark/reward_contract_*` | NOT STARTED |
| Outcome-only vs proportional behavior | `ablation/reward_ablation` | NOT STARTED |
| OOD robustness (honest finding; falsified mechanism deleted) | `ablation/ood_robustness` | NOT STARTED |
| Budget/aggressiveness sensitivity robustness | `ablation/*_sensitivity` | NOT STARTED |
| Unbounded-MDP `compromise_rate=1.0` contrast (single panel) | (control) | NOT STARTED |

---

## 8. Prose Staging (Phase F)

| Section | Action |
|---|---|
| Title | Drop "LSTM"; likely drop "Adaptive" (or redefine as defender-side). New title TBD after curve. |
| Abstract/Resumo | Rewrite around PREVENTION spine; remove LSTM; macro-driven numbers only. |
| Introduction | Re-center 5 pillars under prevention spine; demote reproducibility. |
| Background | Add "why not just classify?" section (RF macro-F1 0.90 + latency makes it airtight). |
| Methodology | Rewrite attacker as kill-chain stochastic process (Markov + budget + evasion); foreground real-observation anchor; add feature-provenance (C6); threats-to-validity designed-dynamics concession + sensitivity sweeps; note C7 THROTTLE dominance. |
| Results | DELETE falsified OOD mechanism (D-OOD); fix algorithm-confound wording (PPO-vs-A2C, not "sample size"); new prevention curve. |
| Conclusion | Re-state findings under prevention spine; honest fallback if H1 messy. |
| Appendix | Reproducibility hash-chain demoted here; feature list (C6); D-MIRAI label fix. |

---

## 9. Verification Gate Status

| Gate | Command | Last result (at `1f57936`) |
|---|---|---|
| Tests | `.venv/bin/python -m pytest -q` | 459 passed, 2 warnings (~90s) |
| Freshness | `make verify-fresh` | GREEN (derived artifacts reconcile with JSON) |
| Lint (ruff) | `ruff check` | 21 errors (UP031/F401 known baseline) — tool only at system level (D10) |
| Format (black) | `black --check` | would reformat 4 files (known baseline) — tool only at system level (D10) |
| Manifest verify | `--verify-manifests` | **UNBACKED** — flag/script don't exist (D9) |

---

## 10. Open Questions

1. **H1 curve shape** — clean sigmoid vs degenerate? (Phase C-cal, hard gate.)
2. **C3** — does a good defender *prevent* or merely *prolong*? (empirical, C-cal.)
3. **Default budget** — `40` is a hypothesis; micro-sweep `{20,30,40,50,60,80,None}` locks it.
4. **C6** — is there a committed artifact listing the 29 selected feature columns?
5. **Unbounded contrast retention** — keep the `compromise_rate=1.0` panel? (user veto at Phase F.)
6. **Title** — finalize after the prevention curve is known.

---

## 11. Journal (append-only)

> Newest at bottom. One line per material event: decision, run launched/finished, commit, gate result.

- `1f57936` — committed AGENTS.md cleanup; tagged `v0.3.0` (pre-pivot baseline); branched `feat/prevention-pivot`.
- Phase A — wrote this Context Ledger (first write). Encodes: prevention spine + C-cal fallback (H1),
  5-pillar shape, LSTM→Markov drop, finite-budget spec, evasion-before-commit reactive attacker,
  kill-chain-stays + face-validity strategy, rename taxonomy (pure-delete; this file = sole record),
  drift register D1–D10 + D-OOD/D-MIRAI, caveat ledger C1–C9 (incl. confirmed C9 eval-env bug).
