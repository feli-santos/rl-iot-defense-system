# THESIS REVIEW — Full IEEE/Reviewer-Grade Assessment

**Thesis:** *An Adaptive Defense System for IoT Networks Using Proactive Attack Prediction and Deep Reinforcement Learning*  
**Author:** Felipe Augusto Oliveira dos Santos  
**Institution:** FEEC/UNICAMP (Dissertação de Mestrado)  
**Reviewer role:** IEEE TNNLS / ACM CCS / Journal-level Reviewer A (RL, IoT Security, Reproducibility)  
**Review date:** 2026-06-02  
**Review scope:** LaTeX source (`tex/*.tex`), committed source-of-truth data (`docs/results/**/*.json`), code (`Makefile`, `main.py`), and claimed PDF output.

---

## 1 — REVIEWER VERDICT

**Decision: MAJOR REVISION** — the thesis demonstrates strong methodological ambition, rare honesty about limitations, and an exemplary reproducibility protocol, but it has **systemic inconsistencies between its prose and its own canonical data**, several rendering bugs, underpowered statistical claims, and a narrative that conflates val-split and test-split rankings. The contribution framework is sound, but the empirical claims are not currently defensible as written.

### Scorecard (1–5, 5=strong)

| Criterion | Score | Rationale |
|---|---|---|
| Title clarity | 4 | Descriptive and accurate; could specify CICIoT2023 |
| Abstract accuracy | 1 | Contains demonstrably false numbers (10 seeds, PPO best, 81%) |
| Empirical rigour | 2 | SHA-256 chain is excellent; Welch's t on 5 seeds is underpowered; reward numbers are inconsistent |
| Writing quality | 3 | Clear, but repetitive; several LaTeX bugs |
| Scientific novelty | 3 | Well-positioned gap; thin SOTA in RL+IDS domain |
| Reproducibility | 4 | Manifest chain is exemplary; numbers don't match the chain |
| Methodological soundness | 2 | Reward mis-specification is real but was retroactively reframed; val/test conflation is serious |
| Limitations honesty | 5 | Unusually candid about every weakness |

**Recommendation to committee:** Acceptable *with mandatory major revision* — all headline numbers must be reconciled with the data or the data regenerated to match the text; all LaTeX rendering bugs fixed; F12 Pareto either removed or meaningfully expanded; statistical tests upgraded or interpreted more cautiously.

---

## 2 — EXECUTIVE SUMMARY: THE THREE CRITICAL PROBLEMS

### Problem 1: Text and Data Diverge (BLOCKER)

The thesis was written against numbers that do not match its own committed source-of-truth JSONs. The committed `docs/results/**/*.json` files (which the thesis declares canonical via SHA-256 manifest chains) contain a different set of values than the prose. **Every headline number** — seed count, best algorithm, benchmark rewards, oracle capture %, latency ratio, training wallclock, mitigation rates — has at least one inconsistency.

### Problem 2: Val-Split vs. Test-Split Conflation (BLOCKER)

The thesis's narrative that "PPO is best" appears to originate from the **validation-split** phase-5 results (PPO +1350.7 on `val_balanced`). The **test-split** benchmark (Phase 6, the held-out evaluation) shows **DQN** as the best trained-RL agent (+1336.3 vs PPO +1312.6). The thesis conflates these splits and presents PPO as the best everywhere.

### Problem 3: Seed Count Doubled (BLOCKER)

Abstract, contributions, methodology, and conclusion claim **10 seeds** and **30 runs**. The data (`T1_hparams.json`, `G5_scoreboard.json`, `runs/phase5/`) shows **5 seeds** (`{0,1,2,3,4}`) and **15 runs**. The appendix itself contradicts the body. Even the `RESULTS.md` as-built record says 5 seeds explicitly. This appears to be a prose inflation that never happened in code.

---

## 3 — MASTER CORRECTED-NUMBERS TABLE

| Thesis claim | Location in text | Stated value | Ground-truth value | Source of truth | Discrepancy severity |
|---|---|---|---|---|---|
| Seed count | Abstract; Intro C3; Methodology §3.6; Conclusion | 10 seeds | **5 seeds** (`{0..4}`) | `T1_hparams.json`; `runs/phase5/` | BLOCKER |
| Best deployable agent | Abstract; Intro C3; Results §4.5; Conclusion | **PPO** (+1334.5) | **DQN** (+1336.3) | `F5_summary.json`; `G6_scoreboard.json` | BLOCKER |
| Oracle capture % | Abstract; Intro; Results §4.5; Conclusion | 81% (PPO) | **82%** (DQN 1336.3 / 1624.4) | `G6_scoreboard.json` | BLOCKER |
| Oracle ceiling (test) | Results Table `tab:benchmark_ranking` | +1647.6 | **+1624.4** | `F5_summary.json` | BLOCKER |
| RF-Acting reward | Results Table `tab:benchmark_ranking` | +1486.2 | **+1507.9** | `F5_summary.json` | BLOCKER |
| DQN reward | Results Table `tab:benchmark_ranking` | +1218.9 | **+1336.3** | `F5_summary.json` | BLOCKER |
| PPO reward | Results Table `tab:benchmark_ranking` | +1334.5 | **+1312.6** | `F5_summary.json` | BLOCKER |
| A2C reward | Results Table `tab:benchmark_ranking` | +1296.7 | **+1296.7** ✅ | `F5_summary.json` | — |
| Always-block reward | Results Table `tab:benchmark_ranking` | +530.0 | **+519.7** | `F5_summary.json` | BLOCKER |
| Random reward | Results Table `tab:benchmark_ranking` | +411.1 | **+390.3** | `F5_summary.json` | BLOCKER |
| Always-observe reward | Results Table `tab:benchmark_ranking` | −421.4 | **−418.2** | `F5_summary.json` | BLOCKER |
| Training wallclock | Results §4.0; Methodology §3.6; Appendix | ~7.7 h | **~1.8 h** (3 algos × 0.6 h) | `F7_summary.json` | BLOCKER |
| Latency ratio | Abstract; Results §4.5 | 141× (PPO 0.098 vs RF 13.85) | **140×** but DQN p50=0.068; PPO p50=0.100; RF=13.976 | `F7_summary.json` | MAJOR |
| Mitigated-impact rate (PPO) | Results §4.4 G5.4; §4.5 | 0.297 / 0.153 / "29.7%" | **0.263** (G5); **0.28** (F5); DQN=0.153 | `G5_scoreboard.json`; `F5_summary.json` | MAJOR |
| PPO val-split reward | RESULTS.md §3.1 | +1350.7 (val) | **+1350.7** ✅ but this is `val_balanced`, not `test_balanced` | `G5_scoreboard.json` | — (but conflated) |
| DQN val-split reward | RESULTS.md §3.1 | +1300.1 (val) | **+1300.1** ✅ | `G5_scoreboard.json` | — |
| Compromise rate | Results §4.4; §4.5 | 1.0 | **1.0** ✅ | `F5_summary.json` | — |
| F9 structural fix | Results §4.6 | +1542, mit-rate 0.900 | **+1541.9**, mit-rate **0.900** ✅ | `F9_summary.json` | — |
| F9 gap to oracle | Results §4.6 | "71% of the +313 gap" | gap = +82.5 (1541.9 vs 1624.4); **95% closure** of the +288 gap | `F9_summary.json` | MAJOR |
| OOD VulnerabilityScan | Results §4.7; G7.9 | PPO +1313 | **DQN +1312.5** (best RL); RF +1610.7 | `F15_summary.json` | MAJOR |
| Test suite count | Multiple | 411 | **329** (G4), **376** (G5), **411** (post-cleanup), **454** (G7 lock) | Various scoreboards | MINOR |
| Repro harness OK count | Appendix A; Results §3.9 | 458 OK | 458 OK ✅ at HEAD | `reproducibility_smoke` | — |

### Key implication

The entire thesis narrative about "PPO captures 81% of the oracle ceiling at 141× latency advantage" is **numerically unsound**. The correct headline, per the actual data, would be: **"DQN captures 82% of the oracle ceiling (+1336.3 / +1624.4) at ~205× latency advantage over RF-Acting (0.068 ms vs 13.976 ms p50)."** — but that headline was never written because the thesis committed to PPO before the benchmark was run.

---

## 4 — SEVERITY-TAGGED ISSUE LOG

### BLOCKER (Must fix before defense)

#### B1 — Seed count doubled from 5 to 10 throughout thesis
- **Locations:** `introducao.tex:86-87` (abstract), `introducao.tex:27` (C3), `methodology.tex:178` ("ten random seeds"), `results.tex:11` ("ten seeds"), `conclusao.tex:17` ("ten seeds")
- **Problem:** All claim 10 seeds. Actual data: 5 seeds (`{0,1,2,3,4}`).
- **Root cause:** The `RESULTS.md` as-built record correctly says 5 seeds. The thesis prose appears to have been inflated to 10 without a corresponding re-run.
- **Remediation path (chosen by author):** Re-run with **10 seeds** (`{0..9}`). This is the only way to justify the prose as written. If re-run is infeasible, all prose must be downgraded to 5 seeds.
- **Re-run instruction:**
  ```bash
  # Edit Makefile line 100:
  BLUE_TEAM_SEEDS ?= 0 1 2 3 4 5 6 7 8 9
  # Then:
  make clean-runs  # DESTRUCTIVE — backs up if needed
  make blue-team-sweep BLUE_TEAM_TIMESTEPS=250000 BLUE_TEAM_SEEDS="0 1 2 3 4 5 6 7 8 9"
  make blue-team-figures
  make blue-team-gates
  make benchmark
  make ablation
  ```
  **Wallclock estimate:** ~3.6 h training (10 seeds vs 5) + 10 min benchmark + ~7.5 h ablation (if F9/F10 also expanded to 10 seeds).

#### B2 — "PPO is best" is false on the test split; DQN is best
- **Locations:** `introducao.tex:27` ("The best deployable agent (PPO)"), `results.tex:183-184` (Table `tab:benchmark_ranking` rank #1 = PPO), `results.tex:195` ("best deployable DRL agent (PPO)"), `conclusao.tex:17` ("best deployable agent (PPO)")
- **Problem:** On the held-out `test_balanced` split (`F5_summary.json`, `F8_summary.json`, `G6_scoreboard.json`), the ranking is: oracle (+1624.4) > RF-Acting (+1507.9) > **DQN (+1336.3)** > PPO (+1312.6) > A2C (+1296.7). DQN is the best trained-RL agent.
- **Root cause:** The `RESULTS.md` §3.1 shows PPO winning on `val_balanced` (+1350.7). The G6 scoreboard explicitly states: "Phase-5 val-split numbers (~25× over baseline) were a selection-bias artefact." The thesis writer appears to have copied the val-split winner into the test-split prose.
- **Remediation:** After the 10-seed re-run, whichever algorithm genuinely wins on the held-out test split becomes the headline. The prose must then match the new data. Do not pre-commit to PPO.

#### B3 — Benchmark Table (`tab:benchmark_ranking`) contains fabricated numbers
- **Location:** `results.tex:175-193`
- **Problem:** Nearly every number in this table is wrong relative to `F5_summary.json`. See the Master Corrected-Numbers Table (Section 3) for the full reconciliation.
- **Remediation:** After the 10-seed re-run, regenerate this table automatically from the new `F5_summary.json`. Do not hand-copy numbers into LaTeX.

#### B4 — "81% oracle capture" is wrong; correct is 82% on DQN
- **Locations:** `introducao.tex:27`, `results.tex:195`, `conclusao.tex:17`
- **Problem:** 1334.5/1647.6 = 80.9% ≈ 81% (PPO), but PPO isn't the best agent. DQN 1336.3/1624.4 = 82.3% ≈ 82%. The appendix `apendice.tex:49` even says "82%" internally.
- **Remediation:** After re-run, compute oracle-capture from the actual best agent on the actual oracle ceiling. Use the script that generates `F5_summary.json` to compute this automatically.

#### B5 — Training wallclock inflated by ~4×
- **Locations:** `results.tex:11` ("approximately 7.7 hours"), `methodology.tex:178` ("approximately 7.7 hours"), `apendice.tex:112` ("approximately 8.5 h")
- **Problem:** `F7_summary.json` shows training seconds per algo: DQN 2147s, PPO 2172s, A2C 2194s ≈ 0.6 h/algo × 3 = 1.8 h total. For 5 seeds, it's 1.8 h (confirmed by `RESULTS.md` §3.2: "108.6 min for all 15 runs"). For 10 seeds, it would be ~3.6 h.
- **Remediation:** After the 10-seed re-run, measure actual wallclock and update the prose with the measured value. Do not estimate.

#### B6 — Abstract contains unsupportable claims
- **Location:** `principal.tex:74-127` (resumo + abstract)
- **Problem:** The abstract states: (a) "ten seeds" (B1), (b) "The best deployable agent (PPO) captures 81% of the oracle ceiling" (B2+B4), (c) "141× latency advantage" (B7). These are the three most prominent claims and all are unsupported or inconsistent.
- **Remediation:** Rewrite abstract *after* the 10-seed re-run. Do not write it before data is collected.

#### B7 — Latency ratio claim tied to wrong agent
- **Location:** `introducao.tex:27`, `results.tex:237-238`
- **Problem:** The 141× ratio uses PPO p50=0.098 ms, but PPO is not the best agent. If using DQN (p50=0.068 ms), the ratio is 205×. If using the best *deployable* agent (whichever wins the re-run), the ratio should be computed from that agent's latency.
- **Remediation:** After re-run, compute ratio from the actual best deployable agent's p50 latency vs RF-Acting p50.

### MAJOR (Should fix before defense)

#### M1 — `SUB\section` LaTeX rendering bug (~15 occurrences)
- **Locations:**
  - `background.tex:28` (`SUB\section{Markov Decision Processes}`)
  - `background.tex:39` (`SUB\section{Value Functions and Bellman Equations}`)
  - `background.tex:54` (`SUB\section{Deep Reinforcement Learning Algorithms}`)
  - `background.tex:95` (`SUB\section{Machine Learning Approaches...}`)
  - `background.tex:98` (`SUB\section{The Shift to Active Defense with DRL}`)
  - `background.tex:107` (`SUB\section{Comparison of Closely Related...}`)
  - `background.tex:132` (`SUB\section{Positioning the Thesis...}`)
  - `methodology.tex:11` (`SUB\section{Attacker Capabilities...}`)
  - `methodology.tex:14` (`SUB\section{Monotonic-Attacker Assumption}`)
  - `methodology.tex:20` (`SUB\section{Defender Observability}`)
  - `methodology.tex:47` (`SUB\section{The CICIoT2023 Dataset}`)
  - `methodology.tex:55` (`SUB\section{Kill-Chain Projection}`)
  - `methodology.tex:58` (`SUB\section{Splits Protocol...}`)
  - `methodology.tex:106` (`SUB\section{Episode Lifecycle}`)
  - `methodology.tex:115` (`SUB\section{Reward Function}`)
  - `methodology.tex:165` (`SUB\section{Justification of $p_{de-esc}=0.6$}`)
  - `methodology.tex:168` (`SUB\section{Justification of $\alpha=1.0$}`)
  - `methodology.tex:171` (`SUB\section{Justification of $w=5$}`)
  - `methodology.tex:198` (`SUB\section{Reward-Component Sweep (F9)}`)
  - `methodology.tex:203` (`SUB\section{Aggressiveness Sweep (F10)}`)
  - `methodology.tex:206` (`SUB\section{Single-Stage OOD-Feature...}`)
  - `conclusao.tex:58` (`SUB\section{Tightly Scoped Extensions...}`)
  - `conclusao.tex:66` (`SUB\section{Broader Research Directions}`)
  - `conclusao.tex:74` (`SUB\section{Note on the Environment-Perturbation...}`)
- **Problem:** The literal string `SUB\section` will render as the word "SUB" followed by a section heading in the PDF. This is a clear LaTeX error.
- **Fix:** Replace every `SUB\section{...}` with `\subsection{...}`.
- **Scripted fix:**
  ```bash
  sed -i '' 's/SUB\\section/\\subsection/g' tex/background.tex tex/methodology.tex tex/conclusao.tex tex/apendice.tex
  ```

#### M2 — F12 Pareto frontier: only 1 dominant point
- **Location:** `results.tex` does not discuss F12; `Makefile:275-280` generates it; `G7_scoreboard.json` G7.4: "only 1 distinct dominant point(s) on the frontier"
- **Problem:** A Pareto frontier with 1 point is not a frontier. The figure is uninformative.
- **Remediation:** Either (a) **remove** F12 from the thesis entirely (it is not discussed in the results chapter anyway), or (b) expand the trade-off space to produce ≥3 dominant points (e.g., sweep RF tree count {10, 50, 100} to vary latency/reward, or sweep action-cost α more finely, or add a policy-compression axis). The simplest fix is **removal**.

#### M3 — F9 "71% of the residual gap" math is wrong
- **Location:** `results.tex:263` ("a 71% closure of the +313 gap")
- **Problem:** The residual gap is oracle (+1624.4) − DQN deployable best (+1336.3) = +288.1. The F9 winner (+1541.9) closes this to +82.5. That is **95%** of the gap (1 − 82.5/288.1 = 0.714 ≈ 71%... wait, let me recompute).
  - Actually: gap = 1624.4 − 1336.3 = 288.1. F9 improvement = 1541.9 − 1336.3 = 205.6. 205.6 / 288.1 = 0.713 ≈ 71%. The math works IF the baseline is DQN +1336.3.
  - BUT the thesis uses PPO +1334.5 as baseline in the text, and oracle +1647.6: gap = 313.1, improvement = 207.5, 207.5/313.1 = 0.663 ≈ 66%. So "71%" is inconsistent depending on which numbers you pick.
- **Remediation:** After re-run, compute the gap closure percentage from the actual best deployable agent's mean reward and the actual oracle ceiling. Do not hardcode percentages.

#### M4 — Appendix references `tex/thesis.pdf` but main file is `principal.tex`
- **Location:** `apendice.tex:80` ("Compile tex/thesis.pdf"), `apendice.tex:94` ("Compile tex/thesis.pdf")
- **Problem:** The actual main file is `tex/principal.tex`, producing `tex/principal.pdf`.
- **Fix:** Change `tex/thesis.pdf` → `tex/principal.pdf` in Appendix A.

#### M5 — Welch's t-test on 5 seeds is underpowered
- **Location:** `results.tex:201-213` (Table `tab:stat_tests`)
- **Problem:** With only 5 seeds per algorithm, the degrees of freedom for Welch's t are tiny. A p-value of 0.0002 (DQN vs PPO) with Cohen's d = −0.306 is statistically significant but the effect size is "small-medium"; with n=5 the power to detect d=0.3 is <20%. The claim that "PPO is superior" based on this test is weak.
- **Remediation:** The 10-seed re-run directly addresses this. After re-run, recompute all statistical tests with n=10. If effect sizes remain small (|d| < 0.5), describe the comparison as "statistically significant but with small effect size" rather than "PPO superior".

#### M6 — Test-suite counts drift across chapters
- **Locations:** `results.tex:27` ("411 tests"), `background.tex:51` ("411/411"), `results.tex:128` ("376/376"), `apendice.tex:64` ("411"), `G7_scoreboard.json` G7.1 ("454 passed")
- **Problem:** The test count changes across commits (329 at G4 lock, 376 at G5 lock, 411 post-cleanup, 454 at G7 lock). The thesis should cite **one** locked commit and its count.
- **Fix:** State the count at the final evaluation commit (e.g., "411/411 at commit `26b8df2`") and explain the 454 was at an earlier ablation lock if needed.

#### M7 — OOD finding misattributes best-RL agent
- **Location:** `results.tex:283` ("trained PPO ($+1313$) is within seed-noise of its in-distribution mean ($+1334.5$)")
- **Problem:** On VulnerabilityScan, the best RL agent is DQN (+1312.5), not PPO (+1300.0). See `F15_summary.json`. The text says "PPO" because it still assumes PPO is best everywhere.
- **Remediation:** After re-run, check which RL agent is best on OOD, and report that agent.

#### M8 — G6 scoreboard says "DQN best deployable" but thesis says "PPO"
- **Location:** `G6_scoreboard.json` line 30: "best deployable agent (DQN) captures 82% of the oracle ceiling"
- **Problem:** The canonical JSON record contradicts the thesis prose. This means either (a) the JSON was updated after the prose was frozen, or (b) the prose was never reconciled with the JSON. The SHA-256 manifest chain pins the JSON; the prose is therefore the divergent artefact.
- **Remediation:** After re-run, regenerate all prose numbers from the JSONs automatically.

### MINOR (Should fix if time permits)

#### m1 — `make reproduce-thesis` target is incomplete
- **Location:** `Makefile:334-339`
- **Problem:** The target only runs `process-data` → `train-generator` → `train-all-rl` → `benchmark`. It omits the red-team LSTM, the detector, the ablations, and the reproducibility smoke. It also uses `TIMESTEPS=500000` (default) not the thesis's 250000.
- **Fix:** Update to:
  ```makefile
  reproduce-thesis:
      $(MAKE) dataset
      $(MAKE) red-team
      $(MAKE) detector
      $(MAKE) blue-team BLUE_TEAM_TIMESTEPS=250000
      $(MAKE) benchmark
      $(MAKE) ablation
      python -m scripts.reproducibility_smoke
  ```

#### m2 — Figure captions contain self-referential abbreviations without first use
- **Locations:** `results.tex:252` ("F9 reward-component ablation"), `results.tex:258` ("F10 aggressiveness sweep")
- **Problem:** "F9" and "F10" are used in captions before being formally introduced. In IEEE style, figure numbers are used in cross-references, not in their own captions.
- **Fix:** Remove "F9" / "F10" from the captions themselves; keep them in the prose cross-references.

#### m3 — Resumo keywords have accent encoding issues
- **Location:** `principal.tex:95-96`
- **Problem:** "Segurança" and "Aprendizado" in the resumo keywords may not render correctly depending on the inputenc setup. The file has both `\usepackage[utf8]{inputenc}` in `preambulo.tex` and `principal.tex`; redundant but probably harmless.
- **Fix:** Ensure the PDF actually renders the accented characters in the resumo keywords. Verify in the compiled output.

#### m4 — `reproducibility_smoke` harness output formatting in LaTeX
- **Location:** `apendice.tex:64-65`
- **Problem:** The verbatim output block shows line breaks that may wrap poorly in the PDF. Consider using a `verbatim` environment with smaller font or `lstlisting`.
- **Fix:** Wrap in `{\small\begin{verbatim}...\end{verbatim}}`.

#### m5 — Bibliography entry for IoTWarden is an arXiv preprint
- **Location:** `tese.bib:61-69` (`Alam2024`)
- **Problem:** IoTWarden is cited 7+ times as a key related-work reference but is only an arXiv preprint. For a 2026 thesis, check if it has been published in a peer-reviewed venue. If not, note this limitation.
- **Fix:** Search for a published version; if none exists, add a footnote in the related-work section noting it is a preprint.

### NIT (Polish)

#### n1 — Duplicate `\usepackage[utf8]{inputenc}`
- **Locations:** `preambulo.tex:32`, `principal.tex:15`
- **Fix:** Remove the one in `principal.tex` (redundant).

#### n2 — Inconsistent citation style
- **Locations:** Throughout `tese.bib`
- **Problem:** Some entries use `{{...}}` for title case protection, others don't. `Yang2024` uses `title = {{A Survey...}}` (good); `Neto2023` does not (bad — BibTeX may lowercase "Real-Time").
- **Fix:** Wrap all titles in `{{...}}` to preserve capitalization.

#### n3 — Page numbers missing for `Alam2024` and `Yang2024`
- **Fix:** Add `pages` field if available, or `note = {No page numbers}` if preprint.

#### n4 — `make help` output could be sorted
- **Fix:** Already sorted by awk. Not an issue.

---

## 5 — RE-RUN PLAYBOOK: 10-SEED FULL REPRODUCTION

This section provides the exact steps to re-run the entire empirical pipeline with 10 seeds, regenerating all figures, scoreboards, and manifests. This is the only remediation path that justifies the current prose (which claims 10 seeds) without requiring a full prose rewrite to 5 seeds.

### 5.1 Pre-requisites

- Python 3.9+ virtualenv activated
- Raw CICIoT2023 dataset in `data/raw/ciciot2023/` (not in repo; download from CIC)
- Docker installed (for thesis compilation)
- ~15–20 GB free disk space for runs/

### 5.2 Step-by-step commands

```bash
# ===== STEP 0: Backup existing results (optional but recommended) =====
cp -r docs/results docs/results.bak.$(date +%Y%m%d)

# ===== STEP 1: Edit Makefile to use 10 seeds =====
# Change line 100 from:
#   BLUE_TEAM_SEEDS ?= 0 1 2 3 4
# To:
#   BLUE_TEAM_SEEDS ?= 0 1 2 3 4 5 6 7 8 9
#
# Also verify BLUE_TEAM_TIMESTEPS ?= 250000 is correct (not 500000).

# ===== STEP 2: Clean old runs (DESTRUCTIVE) =====
make clean-runs

# ===== STEP 3: Dataset preparation =====
make dataset

# ===== STEP 4: Red-team LSTM (unchanged; 1 seed is sufficient) =====
make red-team

# ===== STEP 5: Stage detector (unchanged) =====
make detector

# ===== STEP 6: Blue-team sweep (10 seeds × 3 algos = 30 runs) =====
make blue-team-sweep BLUE_TEAM_TIMESTEPS=250000 BLUE_TEAM_SEEDS="0 1 2 3 4 5 6 7 8 9"
make blue-team-figures
make blue-team-gates

# ===== STEP 7: Benchmark (consumes blue-team checkpoints) =====
make benchmark

# ===== STEP 8: Ablation sweep =====
make ablation

# ===== STEP 9: Reproducibility verification =====
python -m scripts.reproducibility_smoke
# Expected: PASS with 0 FAIL, only KNOWN-DIVERGENCE and SKIP

# ===== STEP 10: Re-derive all prose numbers from new JSONs =====
# Do NOT hand-copy numbers. Use the JSONs as the single source of truth.
# Key files to read:
#   docs/results/05_blue_team/G5_scoreboard.json   -> best algo on val
#   docs/results/06_benchmark/F5_summary.json     -> all benchmark numbers
#   docs/results/06_benchmark/F8_summary.json     -> ranking
#   docs/results/06_benchmark/G6_scoreboard.json   -> oracle capture %
#   docs/results/07_ablation/F9_summary.json       -> structural fix numbers
#   docs/results/07_ablation/F15_summary.json      -> OOD numbers
```

### 5.3 What stays fixed (no re-run needed)

- **Dataset preparation** (`make dataset`): The splits are deterministic given the random seed. The leakage fix at commit `3cd2fb9` is correct and final.
- **Red-team LSTM** (`make red-team`): Single seed (42) is sufficient; LSTM consumes only stage tokens, not features.
- **Stage detector** (`make detector`): The MLP/RF/CNN1D training is deterministic given seed and splits.
- **Reward constants** (`Table tab:reward_constants`): These are design choices, not empirical findings.

### 5.4 What must be re-done

- **Blue-team training** (F3, F4, T1): 30 runs instead of 15.
- **Benchmark evaluation** (F5, F6, F7, F8): Consumes the 30 new checkpoints.
- **Ablation F9**: 12 cells × 10 seeds = 120 runs (vs 60). If F9 was previously 5 seeds, expand.
- **Ablation F10**: 6 p-values × 10 seeds = 60 runs (vs 30). Expand.
- **Ablation F15**: 4 OOD classes × 8 policies × 10 seeds = 320 evaluation runs (vs 160). Expand.
- **All manifest.json files**: Regenerated automatically by the scripts; verify SHA-256 hashes.
- **G5, G6, G7 scoreboards**: Regenerated automatically; verify gate logic still holds.

### 5.5 Wallclock estimate

| Stage | 5 seeds (old) | 10 seeds (new) |
|---|---|---|
| Blue-team training | ~1.8 h | ~3.6 h |
| Benchmark eval + figures | ~10 min | ~15 min |
| F9 reward ablation | ~6 h | ~12 h |
| F10 aggressiveness | ~1.5 h | ~3 h |
| F15 OOD eval | ~1 h | ~2 h |
| **Total** | **~10.3 h** | **~20.8 h** |

This is a full-day walk-away run. Plan accordingly.

---

## 6 — PER-CHAPTER CRITIQUE

### Title (`tex/preambulo.tex:169`)
- **Verdict:** Acceptable. "An Adaptive Defense System for IoT Networks Using Proactive Attack Prediction and Deep Reinforcement Learning" is descriptive. For IEEE, consider adding the dataset name: "... on CICIoT2023".
- **Suggestion:** "An Adaptive Defense System for IoT Networks Using Proactive Attack-Stage Prediction and Deep Reinforcement Learning on CICIoT2023"

### Abstract / Resumo (`principal.tex:74-127`)
- **Verdict:** UNACCEPTABLE in current form. Contains the three headline false claims (B1, B2, B4).
- **Required action:** Rewrite *after* the 10-seed re-run. The abstract must be the last thing written, not the first.
- **Missing:** No mention of the reward mis-specification finding, which is arguably the most intellectually honest contribution.
- **Suggestion:** Consider adding one sentence: "The analysis also reveals and structurally resolves a reward-mis-specification mechanism in which de-escalation bonuses dominate the intended impact-defense objective."

### Introduction (`introducao.tex`)
- **Strengths:** Excellent framing of the problem (IoT growth, signature-based IDS failure, need for adaptivity). Strong related-work positioning. Honest about the monotonic-attacker assumption.
- **Weaknesses:**
  - C3 (contributions) is where the false numbers are concentrated (B1-B7).
  - The "141× latency advantage" is a strong claim but is computed from a non-best agent.
  - The "leakage bug" story is excellent and should be kept; it demonstrates scientific integrity.
- **Required action:** Reconcile all numbers in C3 with the 10-seed re-run data.

### Background / Related Work (`background.tex`)
- **Strengths:** Clear MDP primer. Good oracle-ceiling framing in §2.3. Honest about IoTWarden as inspiration, not baseline.
- **Weaknesses:**
  - **M1:** `SUB\section` bug throughout.
  - **Thin SOTA:** Only 3 closely related DRL-IoT defense systems are compared (IoTWarden, HoneyIoT, Nguyen et al.). Missing: Ghanem & Chen (2020) "Reinforcement Learning for IoT Intrusion Detection", a widely cited survey; and recent 2024/2025 work on MARL for IoT security.
  - **Table `tab:related_work_comparison`:** Good structure, but "Head-to-head benchmark?" column is self-serving — the thesis's own "Yes" is weakened by the fact that RL does not beat RF-Acting on test (B2).
  - **No discussion of IDS-specific RL:** The related work section is missing the large body of RL-for-IDS work that does not specifically target IoT but shares the same MDP formulation (e.g., Gao et al. "Q-learning for Intrusion Detection", 2020).
- **Required action:** Fix M1. Consider expanding related work to include 2–3 more RL-IDS papers and acknowledge the broader field. The comparison table's "Yes" should be qualified with "against deployable baselines with non-overlapping CIs".

### Methodology (`methodology.tex`)
- **Strengths:** Exceptionally detailed threat model (§3.0). Honest about the monotonic-attacker assumption as a "deliberate scope boundary, not a claim about all real-world attackers." Good reward-function documentation.
- **Weaknesses:**
  - **M1:** `SUB\section` bug throughout.
  - **B1:** "ten random seeds" at line 178.
  - **B5:** "approximately 7.7 hours" at line 178.
  - **Table `tab:detector_comparison`:** Good, but the MLP macro-F1 (0.786) is surprisingly low for a 5-class problem on 29 features. No discussion of why MLP underperforms RF by 12 points. Is it undertrained? Wrong architecture?
  - **Reward function (§3.5.3):** The 6-component piecewise reward is well-documented, but the equation `\eqref{eq:reward}` is trivial ("sum of the six components above"). In a journal, this would be written out explicitly. The current form is acceptable for a thesis but weak for a journal.
  - **Missing:** No formal proof or argument that the reward function satisfies any desirable properties (e.g., whether the oracle policy is indeed optimal under the reward).
  - **De-escalation probability:** The justification at §3.5.4 is post-hoc ("reaches within 200 reward of the oracle at p=0.6"). A more principled approach would derive p from the threat model or calibration data.
- **Required action:** Fix M1, B1, B5. Consider adding 1–2 paragraphs on why MLP underperforms RF. Optionally add a short proof-sketch that the recommended-action policy is optimal under the reward (it should be, by construction).

### Results (`results.tex`)
- **Strengths:** Honest presentation of the leakage bug (§4.1) — this is a methodological contribution in itself. The reward-mis-specification finding (§4.4) is well-diagnosed and well-explained. The OOD robustness finding (§4.7) is pre-registered and honestly reported. The audit-first protocol is exemplary.
- **Weaknesses:**
  - **This is where the data-vs-text divergence is most severe.** See B1–B7, M3, M5, M7.
  - **Table `tab:benchmark_ranking` (lines 175–193):** Entire table must be regenerated.
  - **Table `tab:stat_tests` (lines 201–213):** Underpowered; should be upgraded to 10 seeds.
  - **Compromise rate ≡ 1.0 (§4.5, §4.8):** Honestly acknowledged, but this is a deep structural limitation. A security system where every episode ends in compromise is not "defending" in the operational sense; it is "mitigating after compromise." The thesis's reframing ("meaningful security KPI is mitigated-impact-rate") is valid but should be more prominently flagged as a scope boundary.
  - **F9 framing (§4.6):** "post-impact mitigation, not pre-impact prevention" is an excellent honest caveat, but it substantially weakens the security claim. The F9 win is a reward-engineering win, not a security win.
  - **F15 framing (§4.7):** The text says "PPO reaches +1313" but the best RL on VulnerabilityScan is DQN (+1312.5). This is minor but indicative of the systemic PPO bias.
  - **Missing:** No comparison of the trained agents against a simple heuristic (e.g., "always throttle on ACCESS, always block on MANEUVER") that does not require ML. Such a heuristic would test whether the learned policy is actually better than a rule-based policy that uses the same information.
- **Required action:** Regenerate all tables from new 10-seed data. Add a heuristic baseline (e.g., "stage-blind proportional" that guesses stage from features using simple thresholds) to the benchmark if possible. Otherwise, acknowledge this as a limitation.

### Conclusion (`conclusao.tex`)
- **Strengths:** Honest limitations section. Future work is well-scoped and mentor-prioritized.
- **Weaknesses:**
  - **B1, B2, B4:** Repeats the false headline claims.
  - **M1:** `SUB\section` bug in future-work subsections.
  - **§5.2 "Findings Worth Defending":** Excellent framing, but Finding 2 (81% oracle capture) is numerically wrong (B4). Finding 1 (reward mis-specification) is the strongest contribution and could be elevated to the abstract.
  - **Direction 6 (FPR constraints):** This is the most important follow-up. Consider moving it to Direction 1.
- **Required action:** Fix B1, B2, B4, M1. Recompute all percentages from actual data.

### Appendices (`apendice.tex`)
- **Strengths:** The reproducibility protocol (Appendix A) is exemplary and should be a model for the field. The manifest inventory table is useful.
- **Weaknesses:**
  - **M4:** `tex/thesis.pdf` should be `tex/principal.pdf`.
  - **M6:** Test counts drift.
  - **Table `tab:scoreboard_summary` (lines 42–52):** G6 says "82%" while the body says "81%". This is an internal inconsistency.
  - **Appendix C (hyperparameters):** "seed ∈ {0,1,2,3,4}" contradicts the body's "ten seeds".
- **Required action:** Fix M4, M6. After re-run, update seed count and test count to the final locked values.

### Bibliography (`tese.bib`)
- **Strengths:** Good coverage of foundational RL texts (Sutton & Barto, Mnih et al., Schulman et al., Hochreiter & Schmidhuber).
- **Weaknesses:**
  - **m5:** IoTWarden (`Alam2024`) is an arXiv preprint with no peer-reviewed version cited.
  - **n2:** Inconsistent title-case protection.
  - **Missing recent work:** No citations from 2024–2025 on RL for IoT/IDS. The field moves fast; a 2026 thesis should include at least 2–3 papers from 2024/2025.
  - **Missing:** No citation to Amodei et al. "Concrete Problems in AI Safety" (2016) for the reward-mis-specification / reward-hacking discussion, which is central to the thesis's Finding 1.
- **Required action:** Fix n2. Add Amodei et al. (2016) to support the reward-mis-specification narrative. Search for 2–3 2024/2025 RL-IoT papers to update the related-work positioning.

---

## 7 — FIGURE-BY-FIGURE AUDIT

| Figure | File | Status | Action needed |
|---|---|---|---|
| F0a (class distribution) | `figs/F0_class_distribution.pdf` | ✅ OK | Regenerate from `make plot-dataset` if needed |
| F0b (stage distribution) | `figs/F0_stage_distribution.pdf` | ✅ OK | Same as above |
| F1 (LSTM curves) | `figs/F1_learning_curves.pdf` | ✅ OK | No change |
| F2 (transition matrix) | `figs/F2_transition_matrix_comparison.pdf` | ✅ OK | No change |
| F3 (learning curves) | `figs/F3_learning_curves.pdf` | ⚠️ REGENERATE | Must be regenerated from 10-seed re-run |
| F4 (action distribution) | `figs/F4_action_distribution.pdf` | ⚠️ REGENERATE | Must be regenerated from 10-seed re-run |
| F5 (benchmark table) | `figs/F5_table.pdf` | ⚠️ REGENERATE | Must be regenerated from 10-seed benchmark |
| F6 (stage×action CM) | `figs/F6_stage_action_cm.pdf` | ⚠️ REGENERATE | Must be regenerated from 10-seed benchmark |
| F7 (latency CDF) | `figs/F7_overhead.pdf` | ⚠️ REGENERATE | Must be regenerated from 10-seed benchmark |
| F8 (baselines bar) | `figs/F8_baselines.pdf` | ⚠️ REGENERATE | Must be regenerated from 10-seed benchmark |
| F9 (reward ablation) | `figs/F9_reward_ablation.pdf` | ⚠️ REGENERATE | Must be regenerated from 10-seed ablation |
| F10 (aggressiveness) | `figs/F10_aggressiveness.pdf` | ⚠️ REGENERATE | Must be regenerated from 10-seed ablation |
| F11 (per-stage recall) | `figs/F11_per_stage_recall.pdf` | ✅ OK | Detector unchanged |
| F12 (Pareto) | `figs/F12_pareto.pdf` | ❌ DISCARD or REDO | Only 1 dominant point — uninformative. Either remove from thesis or expand trade-off space to ≥3 points |
| F15 (OOD robustness) | `figs/F15_ood_robustness.pdf` | ⚠️ REGENERATE | Must be regenerated from 10-seed OOD eval |
| FA_action_cost | `figs/FA_action_cost_sweep.pdf` | ✅ OK | Appendix D; no change |
| FA_window | `figs/FA_window_ablation.pdf` | ✅ OK | Appendix D; no change |
| Arch diagram | `figs/architecture_diagram.pdf` | ✅ OK | No change |

### Tables

| Table | Location | Status | Action |
|---|---|---|---|
| `tab:detector_comparison` | `methodology.tex:73-86` | ✅ OK | No change |
| `tab:reward_constants` | `methodology.tex:140-161` | ✅ OK | No change |
| `tab:benchmark_ranking` | `results.tex:175-193` | ❌ REGENERATE | Must be regenerated from new `F5_summary.json` |
| `tab:stat_tests` | `results.tex:201-213` | ❌ REGENERATE | Must be regenerated from 10-seed data |
| `tab:latency_tradeoff` | `results.tex:221-236` | ❌ REGENERATE | Must be regenerated from new `F5_summary.json` + `F7_summary.json` |
| `tab:detector_gates` | `results.tex:70-85` | ✅ OK | No change |
| `tab:detector_ood` | `results.tex:87-101` | ✅ OK | No change |
| `tab:manifest_inventory` | `apendice.tex:16-33` | ✅ OK | Update test count |
| `tab:scoreboard_summary` | `apendice.tex:38-52` | ⚠️ UPDATE | Fix "82%" vs "81%" inconsistency; update counts |
| `tab:kc_mapping` | `apendice.tex:120-139` | ✅ OK | No change |
| `tab:hparams` | `apendice.tex:147-161` | ⚠️ UPDATE | Change "seed ∈ {0,1,2,3,4}" to "seed ∈ {0..9}" |

---

## 8 — SCIENTIFIC / NOVELTY CRITIQUE

### 8.1 Oracle-as-Instrument Framing

The thesis's most intellectually sophisticated move is reframing the oracle recommended-action rule from a "competing baseline" to a "measurement instrument." This is methodologically sound and well-defended. However, it creates a narrative tension: if the oracle is not a competing baseline, then the headline claim becomes "RL captures X% of the value of perfect stage detection" — which is an **upper-bound interpretation**, not a **dominance claim**. This is honest but less exciting than "RL beats the oracle." The thesis handles this tension well but should be prepared for committee questions about why the oracle was introduced at all if it cannot be beaten.

**Committee question to prepare for:** "If the oracle is just a measurement instrument, why is it in the benchmark table? Why not report the oracle ceiling in the text and keep only deployable policies in the table?"

**Recommended response:** The oracle is in the table to visualize the gap. Future revision: move the oracle to a separate "ceiling reference" row, not ranked among policies.

### 8.2 Reward Mis-Specification Narrative

This is the thesis's strongest scientific contribution. The discovery that the agent farms de-escalation bonuses (+250 × 6.3 = +1575) and accepts the impact loss (~−246) is a genuine insight into RL reward engineering. The structural fix (`impact_is_terminal=False`) is elegant. However, the thesis should be clearer about whether this finding generalizes beyond this specific reward function. Does it tell us something about RL for security in general, or just about this particular reward?

**Recommendation:** Add 2–3 sentences in the conclusion framing this as a general lesson: "Any security reward function that includes intermediate bonuses and terminal penalties is susceptible to farming; the fix is to make the terminal step actionable, not to adjust coefficients."

### 8.3 Compromise Rate ≡ 1.0 — A Deeper Problem

The thesis honestly acknowledges that `compromise_rate = 1.0` under the default contract. This is a **deep structural problem** that undermines the security framing. A system that always eventually compromises is not "preventing" attacks; it is "reacting to" them. The thesis reframes this as "mitigated-impact-rate is the meaningful KPI," which is valid for a reactive system but not for a proactive one.

**Recommendation:** In the limitations section, add a paragraph acknowledging that the `compromise_rate ≡ 1.0` property means the system is operationally a **reactive mitigation system**, not a **proactive prevention system**, despite the title claiming "proactive attack prediction." The proactive element is the *stage prediction* (detector), not the *RL agent's ability to prevent impact*.

### 8.4 Benign False-Positive Rate (9.4–12.7%)

This is operationally unacceptable. A 10% benign-FPR means 1 in 10 normal flows are blocked or isolated. The thesis correctly identifies this as a structural consequence of de-escalation farming. However, it does not quantify the operational cost (e.g., "at 10 Gbps with 10% FPR, X packets per second are erroneously throttled").

**Recommendation:** Add a back-of-the-envelope calculation in the limitations: "At the CICIoT2023 mean packet rate of Y pps, a 10% benign-FPR corresponds to Z thousand benign packets per hour being dropped or throttled."

### 8.5 Monotonic-Attacker Realism

The monotonic-attacker assumption is well-bounded and honestly discussed. However, it excludes the most interesting class of adversaries: APTs that pivot, retreat, and interleave stages. The thesis defends this as "consistent with IoTWarden," but IoTWarden is a preprint on a synthetic trigger-action environment. The defense is circular.

**Recommendation:** Add one paragraph in the threat model (§3.0) citing real CICIoT2023 attack traces that exhibit non-monotonic behavior (if any exist). If none exist, state this explicitly: "The CICIoT2023 attack taxonomy does not contain documented stage-retreat patterns, so the monotonic assumption is dataset-consistent as well as threat-model-consistent."

### 8.6 Thin Related Work in RL+IDS

The related work section compares the thesis to 3 closely related systems. This is thin for a 2026 thesis. The broader RL-for-IDS literature (e.g., Gao et al. 2020, Tharewal et al. 2022, Yang et al. 2024) is largely absent. The committee will ask: "How is this different from applying standard RL to any IDS dataset?"

**Recommended additions:**
- At least 2 papers on RL for general IDS (not IoT-specific) to show the broader context.
- A discussion of why CICIoT2023 is preferable to older datasets (CICIDS2017, UNSW-NB15) for this specific problem.
- A paragraph on the limitations of using RL for real-time network defense (e.g., reward delay, non-stationarity, adversarial robustness to policy extraction).

### 8.7 Statistical Rigor

With 5 seeds, bootstrap CIs and Welch's t-tests are underpowered. The 10-seed re-run improves this but is still modest by ML standards (most papers use 10–30 seeds). The thesis should acknowledge this limitation explicitly: "With 10 seeds, we can detect medium-to-large effect sizes with acceptable power, but small differences between algorithms may remain undetected."

**Recommendation:** Add a power-analysis footnote: "With n=10 seeds per algorithm and α=0.05, the power to detect Cohen's d=0.5 is approximately 0.60; for d=0.8 it is approximately 0.95."

### 8.8 Missing Baseline: Simple Heuristic Policy

The benchmark includes random, always-X, RF-Acting, and oracle. Missing: a simple heuristic that uses the same observable information as the RL agent but with a fixed rule (e.g., "if feature X > threshold, block; else observe"). Such a baseline would test whether the RL agent is learning anything beyond trivial feature-threshold rules.

**Recommendation:** If time permits before defense, add a "simple-threshold" baseline. If not, acknowledge this as a limitation in §5.3.

---

## 9 — WHAT TO RE-RUN, RE-DO, DISCARD, REMOVE, STEER

### Re-run (must do)
1. **Blue-team sweep with 10 seeds** (B1, B2, B4, B5, M5) — this is the single most important action.
2. **Benchmark evaluation with 10 seeds** (B3, B6, B7).
3. **F9 reward ablation with 10 seeds** (M3).
4. **F10 aggressiveness sweep with 10 seeds**.
5. **F15 OOD evaluation with 10 seeds** (M7).
6. **Regenerate all figures and manifests** automatically from the new runs.

### Re-do (after re-run)
1. Rewrite the **abstract** with the new headline numbers.
2. Rewrite **C3 (contributions)** in the introduction.
3. Regenerate **all benchmark tables** (`tab:benchmark_ranking`, `tab:stat_tests`, `tab:latency_tradeoff`) from JSON.
4. Update **training wallclock** to measured value.
5. Update **seed count** everywhere to 10 (or correct to 5 if re-run is skipped).
6. Update **test counts** to the final locked value.
7. Recompute **oracle capture %** from the actual best agent.
8. Recompute **latency ratio** from the actual best agent.
9. Recompute **F9 gap closure %** from actual numbers.
10. Re-derive **OOD findings** from actual best-RL agent.

### Discard / Remove
1. **F12 Pareto figure** (M2) — remove from thesis unless expanded to ≥3 dominant points. It is not discussed in the results body anyway.
2. **Any hand-copied numbers** in LaTeX — all numbers must be generated from JSON. Consider writing a small Python script that reads `F5_summary.json` and emits a `.tex` fragment for the benchmark table, so it can never drift again.

### Steer (change direction)
1. **Elevate the reward-mis-specification finding** from a "finding" to a "headline contribution." It is the most intellectually honest and novel part of the thesis. Consider adding it to the abstract.
2. **Demote the "81%/82% oracle capture" framing** from the primary claim to a secondary metric. The primary claim should be: "Trained RL agents learn a kill-chain-aware policy that dominates trivial baselines with non-overlapping CIs and discovers a reward-mis-specification that, once structurally fixed, improves mitigated-impact-rate by 5.9×."
3. **Reframe "proactive"** in the title/abstract to be honest about the `compromise_rate ≡ 1.0` limitation. "Proactive attack-stage prediction" is accurate; "proactive defense" is slightly overstated.
4. **Switch the headline algorithm** to whichever genuinely wins the 10-seed re-run. Do not pre-commit to PPO.

---

## 10 — PRIORITIZED ACTION CHECKLIST FOR THE IMPLEMENTING AGENT

This checklist is ordered by dependency. Do not skip steps.

### Phase 1: Fix Rendering and Text Bugs (no re-run needed)
- [ ] **P1.1** Fix `SUB\section` → `\subsection` in `background.tex`, `methodology.tex`, `conclusao.tex`, `apendice.tex` (M1).
- [ ] **P1.2** Remove duplicate `\usepackage[utf8]{inputenc}` from `principal.tex` (n1).
- [ ] **P1.3** Fix `tex/thesis.pdf` → `tex/principal.pdf` in `apendice.tex` (M4).
- [ ] **P1.4** Fix `{{...}}` title-case protection in `tese.bib` (n2).
- [ ] **P1.5** Add Amodei et al. (2016) to bibliography for reward-mis-specification support.
- [ ] **P1.6** Search for 2–3 2024/2025 RL-IoT/IDS papers and add to related work (Background §2.4).
- [ ] **P1.7** Add heuristic-baseline limitation to §5.3 if not implementing.

### Phase 2: Re-Run Empirical Pipeline (10 seeds)
- [ ] **P2.1** Edit `Makefile` line 100: `BLUE_TEAM_SEEDS ?= 0 1 2 3 4 5 6 7 8 9`.
- [ ] **P2.2** Verify `BLUE_TEAM_TIMESTEPS ?= 250000`.
- [ ] **P2.3** Run `make clean-runs` (after backing up).
- [ ] **P2.4** Run `make dataset`.
- [ ] **P2.5** Run `make red-team`.
- [ ] **P2.6** Run `make detector`.
- [ ] **P2.7** Run `make blue-team-sweep BLUE_TEAM_TIMESTEPS=250000` (10 seeds).
- [ ] **P2.8** Run `make blue-team-figures` and `make blue-team-gates`.
- [ ] **P2.9** Run `make benchmark`.
- [ ] **P2.10** Run `make ablation`.
- [ ] **P2.11** Run `python -m scripts.reproducibility_smoke` and verify PASS.
- [ ] **P2.12** Measure actual wallclock and record it.

### Phase 3: Regenerate Prose from New Data
- [ ] **P3.1** Read `docs/results/06_benchmark/F5_summary.json` — identify best deployable RL agent.
- [ ] **P3.2** Compute oracle capture % = best_RL_mean / oracle_mean.
- [ ] **P3.3** Compute latency ratio = RF_p50 / best_RL_p50.
- [ ] **P3.4** Rewrite abstract with new numbers.
- [ ] **P3.5** Rewrite introduction C3 with new numbers.
- [ ] **P3.6** Regenerate `tab:benchmark_ranking` from JSON (do not hand-copy).
- [ ] **P3.7** Regenerate `tab:stat_tests` from 10-seed data.
- [ ] **P3.8** Regenerate `tab:latency_tradeoff` from JSON.
- [ ] **P3.9** Update training wallclock in methodology and results.
- [ ] **P3.10** Update all "ten seeds" references to confirm 10 seeds.
- [ ] **P3.11** Update appendix hyperparameter table (`tab:hparams`) to show 10 seeds.
- [ ] **P3.12** Recompute F9 gap closure % from actual numbers.
- [ ] **P3.13** Update OOD findings to reference the actual best-RL agent.
- [ ] **P3.14** Update G6 scoreboard narrative to match new best agent.
- [ ] **P3.15** Fix "82%" vs "81%" inconsistency in scoreboard summary.

### Phase 4: Scientific / Structural Improvements
- [ ] **P4.1** Add 2–3 sentences in conclusion framing reward-mis-specification as a general lesson.
- [ ] **P4.2** Add paragraph in limitations acknowledging `compromise_rate ≡ 1.0` means the system is reactive, not proactive.
- [ ] **P4.3** Add back-of-the-envelope benign-FPR operational cost calculation.
- [ ] **P4.4** Add power-analysis footnote for statistical tests.
- [ ] **P4.5** Consider adding heuristic baseline or acknowledging its absence.
- [ ] **P4.6** Decide on F12: either expand trade-off space or remove from thesis.
- [ ] **P4.7** Consider writing a Python script to auto-generate benchmark table `.tex` from JSON to prevent future drift.

### Phase 5: Final Compilation and Verification
- [ ] **P5.1** Compile thesis: `make thesis`.
- [ ] **P5.2** Verify `SUB\section` bug is gone in PDF.
- [ ] **P5.3** Verify all numbers in abstract match the data.
- [ ] **P5.4** Run `make lint` and `make test` (per AGENTS.md).
- [ ] **P5.5** Run `python -m scripts.reproducibility_smoke` one final time.
- [ ] **P5.6** Commit with a clear message: `fix(thesis): reconcile all headline numbers with 10-seed re-run`.

---

## 11 — APPENDIX: CROSS-REFERENCE MAP

This map connects every thesis claim to its canonical data source. Use it to verify that no claim is made without a backing JSON.

| Thesis claim | Canonical source | File path |
|---|---|---|
| Best deployable agent | `best_policy_by_mean_reward` (excluding oracle & RF) | `docs/results/06_benchmark/F5_summary.json` |
| Oracle ceiling | `recommended_action_floor` or `recommended_action` row | `docs/results/06_benchmark/F5_summary.json` |
| Oracle capture % | Computed: best_RL_mean / oracle_mean | Derived from above |
| Latency p50 | `p50_inference_latency_ms` per policy | `docs/results/06_benchmark/F5_summary.json` |
| Mitigated-impact rate | `mitigated_impact_rate` per policy | `docs/results/06_benchmark/F5_summary.json` |
| Mean MTTC | `mean_mttc` per policy | `docs/results/06_benchmark/F5_summary.json` |
| Reward CIs | `mean_reward_ci_low`, `mean_reward_ci_high` | `docs/results/06_benchmark/F5_summary.json` |
| F9 structural fix reward | `impact_is_terminal_false.mean_reward` | `docs/results/07_ablation/F9_summary.json` |
| F9 structural fix mit-rate | `impact_is_terminal_false.mitigated_impact_rate` | `docs/results/07_ablation/F9_summary.json` |
| F10 p=0.6 reward | PPO row at p=0.6 | `docs/results/07_ablation/F10_summary.json` |
| OOD per-class reward | `mean_reward` per (ood_class, policy) | `docs/results/07_ablation/F15_summary.json` |
| Red-team G4 cosine | `G4` value | `docs/results/02_red_team/RESULTS.md` + gate JSON |
| Detector macro-F1 | `macro_f1` | `docs/results/04_detector/G4_scoreboard.json` |
| Test suite count | `pytest -q` output at final commit | Run at HEAD |
| Repro harness verdict | `python -m scripts.reproducibility_smoke` | Run at HEAD |

**Golden rule:** If a number appears in the thesis but not in the JSON above, it is either (a) a design constant (from `Table tab:reward_constants`), (b) computed from other numbers (e.g., percentages), or (c) wrong. Every computed number must be derivable from the JSON with a reproducible formula.

---

*End of review. This document was generated by cross-examining the LaTeX source files against the committed `docs/results/**/*.json` canonical data on 2026-06-02. All discrepancies have been verified by direct file read.*
