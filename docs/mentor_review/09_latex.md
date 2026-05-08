# Step 9 — LaTeX Framing & Thesis Prose: Mentor Memo

**Closed:** 2026-05-08 ~08:45 BRT
**Author (agent):** mentor agent (Step 9)
**Reviewed phase / scope:** Step 9 — full LaTeX rebuild (abstract,
introduction, background, methodology, results, conclusions,
appendices) against the now-stable per-phase RESULTS.md files.
**Status:** **completed — VERDICT PASS** at HEAD `892eb59` on
`mentor-review/step-9-latex`. pytest 411 / 411; R1 harness
VERDICT PASS (458 OK / 0 FAIL / 2 KNOWN-DIVERGENCE / 6 SKIP).
Awaiting candidate sign-off before G2 merge.

---

## 1. Verdict

**PASS** — every chapter rewrites cleanly against the as-built
per-phase RESULTS.md files; every Step-7 reframe (82 %-of-ceiling,
D7.9.1 OOD-robustness, Q7 `compromise_rate=1.0`, Q8 Phase-8-skipped)
lands in LaTeX prose; the R2 reproducibility appendix ships at
two pages with the model-card pattern (Mitchell et al. 2019); the
Step-3 F4 MTTC IMPACT-clamp footnote and the Step-4 F5
Phase-4-Finding-3 → RL-OOD cross-link both propagate. No
findings opened in Step 9; no test changes; no model retraining.

---

## 2. Verdict line by line

| Acceptance criterion (08_HANDOFF.md §5 "Acceptance criterion for Step 9 PASS") | Status | Evidence |
|---|:---:|---|
| Every chapter cites the as-built RESULTS files faithfully | **PASS** | 6 LaTeX commits e1202dc..892eb59; numerical sources cited per commit body |
| The 82 %-of-ceiling reframe lands in `tex/` | **PASS** | abstract `tex/thesis.tex` L40; intro `tex/introduction.tex` contribution #3; results `tex/results.tex` §4.5 + Table 4.5 + dedicated paragraph; conclusions `tex/conclusions.tex` §5.1 contrib #3 + §5.2 Finding 2 |
| The D7.9.1 OOD reframe lands in `tex/` | **PASS** | abstract L40; intro contribution #4; results `tex/results.tex` §4.7 (full D7.9.1 quote-block); conclusions §5.2 Finding 3 |
| The Q7 `compromise_rate=1.0` paragraph is authored in LaTeX (not just cited) | **PASS** | `tex/results.tex` §4.5 dedicated paragraph "The compromise-rate $=$ $1.0$ caveat"; cross-cited in §4.6 F12 caveat + §4.8 threats-to-validity #1 + conclusions §5.3 limitation #1 |
| The R2 reproducibility appendix lists every artefact + R1 harness recipe | **PASS** | `tex/appendices.tex` Appendix A (~2 pages): 5 subsections (manifest pattern, scoreboards summary, harness verdict buckets, fresh-checkout recipe, hardware) |
| pytest 411 / 411 + R1 harness PASS at every commit | **PASS** | verified at HEAD: `pytest -q` → 411 passed; `python -m scripts.reproducibility_smoke` → VERDICT PASS (458 / 0 / 2 / 6) |
| Step-3 F4 MTTC IMPACT-clamp footnote propagates | **PASS** | `tex/methodology.tex` §3.5.3 footnote on the MTTC paragraph + `tex/results.tex` §4.4 footnote on G5.3 + `tex/conclusions.tex` §5.3 limitation #2; all three cite `docs/results/03_env/RESULTS.md §7 R2` |
| Step-4 F5 Phase-4-Finding-3 → RL-OOD cross-link propagates | **PASS** | forward-link in `tex/methodology.tex` §3.4 (closing paragraph); double-cited in `tex/results.tex` §4.3 (closing paragraph after Table 4.4) and §4.7 (closing paragraph "closes the loop with the Phase-4 OOD-asymmetry finding"); finally in conclusions §5.2 Finding 3 |
| F6 / F13 / F14 reframed as post-thesis future work (Q8 Phase-8-skipped) | **PASS** | `tex/conclusions.tex` §5.4 directions 1–3 + the "Note on Phase 8 of the audit protocol" subsubsection authoring the Q8 paragraph |
| `09_latex.md` + `09_HANDOFF.md` shipped | **PASS** | this commit (8/8) |

---

## 3. Findings

**None opened in Step 9.** Step 9 is the prose-rewrite step; it
produces no new gates and no new findings. Every prior-phase
finding referenced in the LaTeX is cited verbatim from the
producing RESULTS.md and scoreboard.

---

## 4. Actions taken in this session

### Branches & commits (in order)

- **Phase G2 of Step 8 + Phase G1 of Step 9** (the start-of-session
  housekeeping owed by `08_HANDOFF.md §5` pre-flight):
  - `git merge --no-ff mentor-review/step-8-cleanup` → `main =
    26b8df2` (`/tmp/merge-step-8.txt` body).
  - `mentor-review/step-8-cleanup` deleted local + remote.
  - `mentor-review/step-9-latex` cut from `26b8df2`.
  - `pytest -q` 411 passed; `python -m scripts.reproducibility_smoke`
    VERDICT PASS.

- **`e1202dc`** —
  `docs(thesis,abstract): retire preliminary-results hedge; surface Step-6 / Step-7 reframes`.
  3-paragraph rewrite of `tex/thesis.tex` L33–41. Surfaces the
  82 %-of-ceiling reframe + the F9 `impact_is_terminal=False`
  +1542 / 5.9× mit-rate result + the D7.9.1 OOD reframe in the
  abstract for the first time. Removes the duplicate
  "implements and benchmarks three prominent DRL algorithms…"
  copy-paste artefact. Updates the keywords to add CICIoT2023 +
  Reproducibility + Cyber Kill Chain.
- **`98d36fa`** —
  `docs(thesis,introduction): refresh contributions; close mid-sentence cliff; surface kill-chain framing`.
  Closes the L32 mid-sentence cliff with a chapter-roadmap
  paragraph; replaces the 4-item contribution list with a 5-item
  list that names the kill-chain projection, the deployable-
  baseline taxonomy + oracle reference, the F9 / F15 ablation
  results (verbatim numbers), the OOD-leakage-bug honesty point,
  and the audit-first reproducibility protocol.
- **`be927ad`** —
  `docs(thesis,background): fix MDP action set to 5 actions; add kill-chain + baselines-taxonomy primers`.
  Three targeted fixes: (a) MDP action set L16 from 4-tuple to
  the as-built 5-tuple `{OBSERVE, LOG, THROTTLE, BLOCK, ISOLATE}`,
  ordered by escalating defensive force; (b) new §2.1 kill-chain
  primer (Mavroeidis2023 + Alam2024); (c) new §2.3 deployable-
  baselines + oracle-as-instrument primer (locking the audit-AF2
  framing for Chapter 4). Modernises §2.4 related work with the
  IoTWarden-as-inspiration framing.
- **`3574adf`** —
  `docs(thesis,methodology): rewrite §3.4-3.5 reward + state/action against as-built; add §3.4 detector / §3.7 baselines / §3.8 ablations / §3.9 reproducibility`.
  Heaviest single-chapter rewrite. New §3.3 red-team, §3.4
  detector, §3.5 RL environment (full as-built reward function
  with eq:reward + eq:reward_terminal + Table 3.1 of 11 reward
  constants), §3.7 baselines, §3.8 ablations, §3.9 reproducibility
  framework. Step-3 F4 MTTC IMPACT-clamp footnote ships in
  §3.5.3. Step-4 F5 cross-link forward-links in §3.4 closing
  paragraph.
- **`ba9206e`** —
  `docs(thesis,results): ground-up rewrite for Phase-1..7; author Q7 compromise_rate=1.0 paragraph; cross-link Phase-4 Finding 3 to RL OOD claim (Step-4 F5)`.
  Heaviest content commit. 9-subsection ground-up rewrite of the
  results chapter against every Phase-1..7 RESULTS.md. Authors
  the Q7 `compromise_rate=1.0` paragraph in §4.5. Closes the
  Step-4 F5 / Step-7 D7.9.1 cross-link loop in §4.7. Includes 15
  thesis-blocking figures (F0a..F15) copied from
  `docs/results/<phase>/` to `tex/figs/` for LaTeX compilation;
  the canonical hash-pinned source-of-truth files remain under
  `docs/results/<phase>/`.
- **`470e622`** —
  `docs(thesis,conclusions): replace pre-Step-3 Next-Steps list with as-built findings; reframe F6/F13/F14 + Phase-8-skipped as post-thesis future work`.
  Replaces the stale 2025-Q4 "Next Steps" list with §5.1
  Contributions / §5.2 Findings Worth Defending / §5.3 Limitations
  / §5.4 Future Work. The Q8 "Note on Phase 8 of the audit
  protocol" paragraph is authored in §5.4. Direction 1
  (MANEUVER-stage de-escalation farming, F6 inspection at 58 %
  ISOLATE) + Direction 2 (F13 noise/drift) + Direction 3 (F14
  OOD-augmentation) are reframed as honest post-thesis future
  work.
- **`892eb59`** —
  `docs(thesis,appendices): replace stale Gantt with R2 reproducibility appendix + kill-chain mapping + hyperparameters`.
  Replaces the stale 2025-Q4 Gantt with three appendices:
  Appendix A (R2 reproducibility, 2 pages, model-card pattern,
  citing Mitchell2019 added to references.bib in the same
  commit), Appendix B (CICIoT2023 → kill-chain table), Appendix C
  (Phase-5 hyperparameters T1).
- **`<this commit>`** —
  `docs(mentor-review,step-9): Step 9 LaTeX framing memo + HANDOFF`.

### Files added / changed (counted across all Step-9 commits)

- **+15 NEW figures** copied to `tex/figs/`:
  `F0_class_distribution.png`, `F0_stage_distribution.png`,
  `F1_learning_curves.png`, `F2_transition_matrix_comparison.png`,
  `F11_per_stage_recall.png`, `F3_learning_curves.png`,
  `F4_action_distribution.png`, `F5_table.png`,
  `F6_stage_action_cm.png`, `F7_overhead.png`,
  `F8_baselines.png`, `F9_reward_ablation.png`,
  `F10_aggressiveness.png`, `F12_pareto.png`,
  `F15_ood_robustness.png`. Source-of-truth files remain in
  `docs/results/<phase>/`; these are display copies for LaTeX
  compilation only and are NOT part of the audit hash chain (R1
  harness does not check `tex/figs/`).
- **6 LaTeX chapters rewritten in full**:
  `tex/thesis.tex` (abstract only), `tex/introduction.tex`,
  `tex/background.tex`, `tex/methodology.tex`, `tex/results.tex`,
  `tex/conclusions.tex`, `tex/appendices.tex`.
- **+1 new bib entry** (`Mitchell2019`) in `tex/references.bib`.
- **+2 NEW mentor-review docs**:
  `docs/mentor_review/09_latex.md` (this file),
  `docs/mentor_review/09_HANDOFF.md`.
- **0 deletions** of locked artefacts. The 7 stale pre-Step-3 PNGs
  in `tex/figs/` (`eda.png`, the 3× `lstm_*.png`,
  `performance_comparison.png`, `reward_distributions.png`,
  `iot_rl_defense_system_diagram_mermaid.png`) are left in place
  for now: only the diagram is still cited by `tex/methodology.tex`
  §3.1; the other six are unreferenced. Step 10 hygiene cleanup
  may delete the unreferenced six.
- **0 module changes**: no Python, no producer scripts, no
  manifests, no scoreboards. Step 9 is documentation-only.

### Tests & invariants

- `pytest -q` 411 / 411 passed at every commit; verified
  end-to-end at HEAD `892eb59`.
- `python -m scripts.reproducibility_smoke` VERDICT PASS at HEAD
  (458 OK / 0 FAIL / 2 KNOWN-DIVERGENCE / 6 SKIP) — the two
  KNOWN-DIVERGENCE rows are the documented pre-leakage-fix
  splits-manifest SHAs in Phase 1 and Phase 2 manifests; the six
  SKIP rows are gitignored Phase-4 input arrays (`features.npy`,
  `stages.npy`, four `splits/*.idx.npy` files).

### Git phases

- **G1 of Step 9** completed at session start (executed at the
  same time as G2 of Step 8): `mentor-review/step-9-latex` cut
  from `main = 26b8df2`. Tags remain empty (mentor-loop policy).
- **G2 of Step 9** owed at the next session start (Step 10):
  merge `mentor-review/step-9-latex` → `main` with
  `--no-ff -F /tmp/merge-step-9.txt`, push, delete branch local +
  remote. **Pause for candidate sign-off** before merging — Step 9
  prose decisions (the contributions list, the future-work
  directions, the abstract framing) are the candidate's call.

---

## 5. Outstanding actions for Step 10

Step 10 is the **release-tag** step (likely `v1.0.0` per
`08_HANDOFF.md §8 Q3`). The work is mostly housekeeping; nothing
substantive in the LaTeX should change from this point forward.

### Pre-flight (Phase G1 of Step 10)

- [ ] Verify candidate sign-off on Step 9 (commit, comment, or
      out-of-band).
- [ ] If sign-off given before merge: execute Phase G2 of Step 9
      ourselves. Write `/tmp/merge-step-9.txt` via `write_to_file`,
      then merge → `main`, push, delete branch.
- [ ] Verify state at the new `main`: `pytest -q` → 411 passed;
      `python -m scripts.reproducibility_smoke` → VERDICT PASS;
      `git tag -l` → empty (still no tags during the loop).
- [ ] Cut `mentor-review/step-10-release` off the new `main`.

### Step 10 review checklist (release-tag)

#### A. v0.1.0 disposition (08_HANDOFF.md §8 Q3 — preserve, per Step-9 decision)

- [ ] Confirm the existing `v0.1.0` tag is still attached to the
      pre-mentor-review commit (verify with
      `git show v0.1.0 --no-patch --pretty=oneline`).
- [ ] Push `v0.1.0` to `origin` if not already there
      (`git push origin v0.1.0`).
- [ ] Document `v0.1.0` in `CHANGELOG.md` as the historical
      anchor for the parallel-Phase-10 chain that Step-7 Q8
      surfaced.

#### B. v1.0.0 release tag

- [ ] Tag `main` HEAD as `v1.0.0` with an annotated message
      summarising the eight-phase audit chain + the audit-first
      protocol + the headline empirical findings (82 %-of-ceiling,
      D7.9.1 OOD, F9 5.9× mit-rate). Use `git tag -a v1.0.0
      -F /tmp/tag-v1.0.0.txt` (NOT inline heredoc).
- [ ] Push `v1.0.0` to `origin`.
- [ ] Verify `git tag -l` shows `v0.1.0` and `v1.0.0` only.

#### C. CHANGELOG.md final pass

- [ ] Add a top-level `## [1.0.0] — 2026-05-XX` block summarising
      the Step-1..Step-9 mentor-review work and citing the merge
      commits.
- [ ] Add a backfill `## [0.1.0]` block describing the
      pre-mentor-review historical anchor.

#### D. Optional (documentation hygiene cleanup)

- [ ] Delete the 6 unreferenced pre-Step-3 PNGs from `tex/figs/`
      (`eda.png`, `lstm_train_accuracy_and_loss.png`,
      `lstm_validation_acc_and_loss.png`,
      `lstm_confusion_matrix.png`, `performance_comparison.png`,
      `reward_distributions.png`). The architecture diagram
      (`iot_rl_defense_system_diagram_mermaid.png`) is still cited
      by `tex/methodology.tex` §3.1 and stays.
- [ ] Verify the LaTeX compiles cleanly with `latexmk -pdf
      thesis.tex` from `tex/`. (Step 9 did not pin a compile
      verification because no LaTeX toolchain is required by the
      audit protocol; Step 10 may verify before tagging.)

#### E. Pytest + R1 invariants

- [ ] `pytest -q` → 411 passed unchanged.
- [ ] `python -m scripts.reproducibility_smoke` → VERDICT PASS
      unchanged.

### Step 10 outputs (deliverables)

- [ ] `docs/mentor_review/10_release.md` — full mentor memo for
      the release-tag step.
- [ ] `docs/mentor_review/10_HANDOFF.md` — final handoff (the
      review loop closes here; the next "step" is the candidate's
      defense itself).
- [ ] Annotated `v1.0.0` tag on `main`.

### Acceptance criterion for Step 10 PASS

- `git tag -l` shows `v0.1.0` (preserved historical) and
  `v1.0.0` (release).
- The `v1.0.0` tag is on `main` and reachable from origin.
- pytest 411 / 411 + R1 harness PASS at HEAD.
- `CHANGELOG.md` documents the Step 1..9 work and both tags.

---

## 6. How to resume

```bash
# Re-open the project
cd /Users/felipe.santos/Projects/rl-iot-defense-system

# Activate the environment
source .venv/bin/activate

# Verify the project is in the state this handoff claims
git rev-parse HEAD                 # expect: HEAD of mentor-review/step-9-latex
                                   #         (the docs(mentor-review,step-9) memo + HANDOFF commit)
git --no-pager log --oneline -10   # expect: docs(mentor-review,step-9) on top of
                                   #         892eb59 (appendices + Mitchell2019)
                                   #         470e622 (conclusions)
                                   #         ba9206e (results + 15 figs)
                                   #         3574adf (methodology)
                                   #         be927ad (background)
                                   #         98d36fa (introduction)
                                   #         e1202dc (abstract)
                                   #         26b8df2 main = Step-8 merge
git status                         # expect: clean
git tag -l                         # expect: EMPTY (mentor-loop policy still)
pytest -q                          # expect: 411 passed
python -m scripts.reproducibility_smoke  # expect: VERDICT PASS
ls docs/mentor_review/             # expect: 00–09 present
```

If any of those expectations fail, **stop** and surface the
divergence before continuing.

**Phase-G2 of Step 9** is owed at the next session start
(after candidate sign-off): merge `mentor-review/step-9-latex`
→ `main` with `--no-ff -F /tmp/merge-step-9.txt`, push, delete
branch local + remote, cut `mentor-review/step-10-release` off
the new `main`.

---

## 7. Context-loading recipe for a fresh agent

Read in this order, in full, before doing any work in Step 10:

1. `docs/mentor_review/README.md` — directory purpose & conventions.
2. `docs/mentor_review/00_framing.md` …
   `08_HANDOFF.md` — the eight predecessor steps. Skim all;
   re-read the §5 ("Outstanding actions") and §8 ("Open
   questions") of each `_HANDOFF.md` to confirm what Step 10
   inherits.
3. `docs/mentor_review/09_latex.md` — Step-9 full memo.
4. `docs/mentor_review/09_HANDOFF.md` — the resume point.
5. `tex/*.tex` — read end-to-end at least once; Step 10 may need
   to verify LaTeX compilation as part of the release-tag work.
6. `CHANGELOG.md` — the file Step 10 will append the v1.0.0 entry
   to.

Skim these for reference if needed (do not read in full):

- The four scoreboards (`docs/results/0[4-7]_*/G[4-7]_scoreboard.json`)
  for the v1.0.0 tag-message numbers.
- `scripts/reproducibility_smoke.py` for the verdict-format
  reference.

---

## 8. Open questions for the user

These do not block Step-9 sign-off; all surface in Step 10 unless
resolved sooner.

1. **Step-10 release-tag scope.** Should `v1.0.0` be a simple
   annotated Git tag, or a full GitHub Release with a release-
   notes asset (the v1.0.0 tag-message body) and an attached
   PDF of the compiled thesis? Mentor-recommendation: Git tag
   only for now (no GitHub Release until the candidate has
   defended). The PDF is not part of the audit chain; binding it
   to a Release would create a moving target if the LaTeX is
   re-compiled.

2. **Optional `tex/figs/` hygiene cleanup.** Should the six
   unreferenced pre-Step-3 PNGs in `tex/figs/` be deleted as part
   of Step 10, or left in place? Mentor-recommendation: delete in
   Step 10 (one extra commit, zero LaTeX impact since they are
   unreferenced). Keeping them costs ~1.5 MB of repo bloat and
   creates a small ambiguity for any future contributor who
   wonders why the thesis ships with eight figures it never
   cites.

3. **`docs/HANDOFF.md` deprecation.** The pre-mentor-review
   `docs/HANDOFF.md` (the historical Phase-7 → Phase-10 closeout
   document) was preserved with a STATUS banner pointing readers
   to `docs/mentor_review/`. Should it be retired in Step 10
   (e.g.\ moved to a `docs/archive/` subdirectory), or left in
   place as a historical anchor? Mentor-recommendation: leave in
   place (the STATUS banner already does the redirect; moving it
   would break any external link).

---

## 9. Risks introduced or noticed (Step 9)

- **Risk: Mitchell2019 bib entry may need a DOI / venue style
  cleanup if the candidate uses a non-standard bibliography
  style.** Likelihood: low (the entry uses the standard
  `inproceedings` template). Impact: cosmetic only. Mitigation:
  the entry is self-contained; if the bibliography style flags it,
  a one-line fix in the bib.

- **Risk: `tex/figs/` display copies could drift out of sync if
  the `docs/results/<phase>/` PNGs are regenerated in Step 10 or
  later.** Likelihood: low (no Step 10 work touches the figure
  pipeline). Impact: medium (the LaTeX would render the older PNG
  while the audit chain points at the newer one). Mitigation:
  documented in this memo (the `tex/figs/` copies are display
  copies; canonical files-of-record stay under
  `docs/results/<phase>/`); Step 10 should re-copy the 15 PNGs if
  any of them is regenerated.

- **Risk: LaTeX compile-verification was skipped in Step 9.** The
  audit protocol does not require a working LaTeX toolchain (the
  per-phase RESULTS.md files + the producer scripts are the
  authoritative numerical record), and Step 9 is documentation-
  only. Likelihood: low that the LaTeX has compile errors (the
  prose was authored against the existing chapter scaffolding +
  `subfig` / standard amsmath / standard graphicx). Impact: low
  (a compile error would surface immediately when the candidate
  builds the PDF for the defense). Mitigation: Step 10 §D.2 in
  this memo explicitly lists `latexmk -pdf` as an optional but
  recommended step before tagging `v1.0.0`.

---

## 10. Sign-off

Step 9 is **complete pending candidate sign-off**. The verdict
is **PASS** at HEAD `892eb59` on `mentor-review/step-9-latex`:
all eight commits landed; pytest 411 / 411; R1 harness PASS;
every Step-7 reframe + Step-3 F4 + Step-4 F5 + Step-7 §8 Q7 + Q8
deliverable propagated into LaTeX prose with file:line citations
in every commit body.

When the candidate types "go" / "Step 10":

1. Phase G2 of Step 9: merge `mentor-review/step-9-latex` →
   `main` `--no-ff -F /tmp/merge-step-9.txt`, push, delete branch
   local + remote.
2. Phase G1 of Step 10: cut `mentor-review/step-10-release` off
   the new `main`.
3. Begin Step 10: `v1.0.0` annotated tag + CHANGELOG.md final
   pass + (optional) `tex/figs/` hygiene cleanup, per §5 above.

— mentor-review agent, 2026-05-08
