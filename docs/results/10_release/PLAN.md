# Phase 10 — Open-Source Hygiene + Release: Plan

> **Locked PLAN.** Same protocol as Phases 2–7: this document is committed
> *before* any implementation. Subsequent `fix(phase-10,§N)` and
> `docs(phase-10,§N)` commits cite §-numbers from this file. Section 8 is
> the locked design-decisions ledger; revisions get explicit `D10.X.1`
> follow-up entries with date + rationale.
>
> Phase 10 ships **no new science**. It is a code-cleanup + documentation
> + release-tag phase whose purpose is to make the public GitHub repo
> match the thesis chapter that cites it. The phase is self-contained and
> reversible (no irreversible model retraining, no overwriting of frozen
> RESULTS docs).

## 1 — Why Phase 10 exists

The thesis chapter is locked at commit `396f827` (Phase 7 closeout) /
`8d5dd67` (HANDOFF rewrite). The repo HEAD that the thesis bibliography
will cite is therefore in a state where:

1. The root `README.md` is **pre-restart**: it claims "179+ unit tests"
   (actual: 454), describes the project as a generic Red/Blue training
   loop, and does not mention any of the eight closed phases (0–7), the
   F-figure series, the `make phase-N` reproduction recipes, or the
   manifest hash-chain reproducibility property that the rest of the
   thesis leans on. Anyone landing on the GitHub URL printed in the
   thesis will read a description that disagrees with every chapter.
2. Three pre-restart **dead source modules** (`src/benchmarking/` —
   note the `g`; the live package is `src/benchmark/`) and three
   pre-restart **dead scripts** are still on disk and consume non-trivial
   amounts of test-suite time and reviewer attention. They are not
   referenced by any Phase 1–7 deliverable.
3. `main.py`'s `--mode evaluate` and `--mode train-all` paths import
   the dead `src/benchmarking/` package and would fail loudly if a
   reader tried to follow them. They are not the canonical eval path
   (which is `make phase-6-eval` / `scripts/benchmark/run_test_eval.py`).
4. `CITATION.cff` exists but contains no thesis-specific `Person` block.
5. There is no `v0.1.0` (or equivalent) git tag pinning the
   thesis-cited state of the repo. A reader cloning at a future date
   has no canonical commit to check out.

Phase 10's job is to fix exactly those five drifts and tag the result.
Anything beyond those five is out of scope (see §6 — what we are NOT
doing).

## 2 — Audit findings (verified at PLAN-commit time)

### 2.1 — Dead `src/benchmarking/` package

```text
src/benchmarking/
├── benchmark_analyzer.py      28358 B  (last touched 2025-03-09)
├── benchmark_runner.py        26937 B  (last touched 2026-04-28; cosmetic only)
└── metrics_collector.py       24735 B  (last touched 2025-03-09)
```

**Consumers (verified 2026-05-02 via `grep -rn "src\.benchmarking" --include='*.py' .`):**

| File | Lines | Consumer status |
|---|---|---|
| `tests/test_benchmark_runner.py` | 359 | Exclusive consumer. Patches `src.benchmarking.benchmark_runner.{ALGORITHM_CLASSES, BenchmarkRunner._create_env, load_model, DummyVecEnv, Monitor}`. |
| `tests/test_metrics_collector.py` | 420 | Exclusive consumer. Imports `src.benchmarking.metrics_collector`. |
| `main.py` | 667–668 | `run_evaluate()` and `run_train_all_rl()` paths. **Not** part of any `make phase-N` workflow; not used by Phase 5/6/7 sweeps. |

**Test-collection count for the two test files:** `pytest --collect-only -q tests/test_benchmark_runner.py tests/test_metrics_collector.py` → **43 tests collected**.

**Live equivalent:** `src/benchmark/{baseline_policies,eval_runner,latency}.py`,
introduced Phase 6, used by Phase 6/7 pipelines. The dead package's
`metrics_collector` overlaps in spirit with `src/benchmark/eval_runner.py` —
the live module is the cleaner, smaller, and tested-by-Phase-6 replacement.

**Documentation references** (will be addressed in C5/C7):

- `docs/metrics-glossary.md:33-34` cites `src/benchmarking/metrics_collector.py`
  and `src/benchmarking/benchmark_runner.py` as authoritative.
- `docs/benchmarking-results.md:9-11, 75-77` cites all three modules.

Both docs are **pre-restart** (predate Phase 1), do not appear in the
Phase 5/6/7 RESULTS chains, and are themselves Phase-10 cleanup
candidates. Decision in §8 D10.2.

### 2.2 — Dead pre-restart scripts

Three scripts under `scripts/` (not under any `scripts/<phase>/`
subdirectory) are pre-restart artefacts:

```text
scripts/evaluate_generator.py
scripts/measure_improved_targets.py
scripts/separability_analysis.py
```

**Consumer search** (verified 2026-05-02):

```bash
grep -rn 'evaluate_generator\|measure_improved_targets\|separability_analysis' \
  --include='*.py' --include='Makefile' --include='*.md' --include='*.yml' \
  --include='*.toml' .
```

→ Only `docs/HANDOFF.md` (lines 121–122 and 252–255) mentions them, and
exclusively in the *context of recommending their deletion in Phase 10*.
Zero functional consumers.

### 2.3 — `main.py` benchmarking imports

`main.py` lines 666–728 (`run_evaluate`) is the only first-party
non-test code that imports `src.benchmarking.*`. The function is also
reachable from `run_train_all` (which calls `run_evaluate` for the
"benchmark all algorithms" CLI mode).

**Decision (D10.1, §8):** retire the import by replacing the function
body with a deprecation pointer to the canonical Phase-6 evaluation
path. Do **not** delete the function or the CLI flag — that would be a
breaking-change to the CLI surface that we'd rather flag as deprecated
for this release.

### 2.4 — `README.md` is pre-restart

`README.md` (root) was last meaningfully edited before the Phase 1
restart. Verified pre-restart markers:

- Line 9: claims "Comprehensive Testing: 179+ unit tests" (actual:
  454; will be 411 post-C3).
- No mention of: Phase 0/1/2/3/4/5/6/7, F-figures (F0..F15), `make
  phase-N` targets, the manifest SHA-256 hash chain, the IoTWarden
  paper extension framing, or `docs/results/` as the canonical thesis
  artefact directory.
- Line 181 + line 191: code blocks that invoke `python main.py --mode
  evaluate` (the about-to-be-deprecated path).

### 2.5 — `CITATION.cff` Person

`CITATION.cff` exists but contains no `Person` author block specific
to the thesis. Will be filled in C6.

### 2.6 — Git tag

`git tag -l` shows no `v0.1.0` (or any version tag). C7 creates one.

## 3 — Deliverables

| ID | What | Where it lands |
|---|---|---|
| **D10.1** | Retire `main.py` benchmarking imports (deprecation pointer) | `main.py::run_evaluate` body |
| **D10.2** | Delete dead `src/benchmarking/` package + its 2 test files | repo |
| **D10.3** | Delete 3 pre-restart orphan scripts | repo |
| **D10.4** | Rewrite root `README.md` | `README.md` |
| **D10.5** | Add thesis Person to `CITATION.cff` | `CITATION.cff` |
| **D10.6** | Phase-10 RESULTS doc + G10 scoreboard | `docs/results/10_release/{RESULTS.md,G10_scoreboard.json}` |
| **D10.7** | CHANGELOG `[Unreleased] — Phase 10` block | `CHANGELOG.md` |
| **D10.8** | Annotate or retire `docs/{benchmarking-results,metrics-glossary}.md` | both files |
| **D10.9** | Add `10_release/` row to `docs/results/README.md` layout block | `docs/results/README.md` |
| **D10.10** | Git tag `v0.1.0` annotated to commit-the-closeout | `refs/tags/v0.1.0` |

## 4 — Exit gates (G10)

Each gate has a deterministic, machine-checkable threshold. The full
bar is `7 PASS / 0 FAIL` to declare Phase 10 closed.

| Gate | What it checks | Threshold |
|---|---|---|
| **G10.1** | Test suite passes after package deletion | `pytest -q` reports `411 passed`, 0 errors, 0 failed, 0 new skips. (454 − 43 = 411.) |
| **G10.2** | No first-party consumer of `src.benchmarking` remains | `grep -rn "from src.benchmarking\|import src\.benchmarking\|src\.benchmarking" --include='*.py' .` returns empty. |
| **G10.3** | Pre-restart scripts removed | `ls scripts/evaluate_generator.py scripts/measure_improved_targets.py scripts/separability_analysis.py 2>&1` reports all three as `No such file or directory`. |
| **G10.4** | README mentions all 8 phases + reproducibility property | `grep -E "Phase 0\|Phase 1\|Phase 2\|Phase 3\|Phase 4\|Phase 5\|Phase 6\|Phase 7" README.md` finds ≥ 8 distinct matches AND `grep -E "make phase-" README.md` finds ≥ 1 match AND `grep -iE "manifest\|sha-?256\|reproducib" README.md` finds ≥ 1 match. |
| **G10.5** | Phase 0–7 frozen test-coverage preserved | The 411 surviving tests include every test that was already green pre-Phase-10 minus exactly the 43 from `test_benchmark_runner.py` + `test_metrics_collector.py`. Verified by diffing `pytest --collect-only -q` before vs. after C3/C4. |
| **G10.6** | Release tag exists and points at the closeout commit | `git tag -l v0.1.0` non-empty; `git rev-parse v0.1.0` resolves to the C7 commit (or later). |
| **G10.7** | RESULTS scoreboard committed with G10 verdicts | `docs/results/10_release/G10_scoreboard.json` exists, has 7 entries, every `passes` field is `true`. |

## 5 — Sequencing

| # | Commit message | Tests | What it touches |
|---|---|---|---|
| **C1** | `docs(phase-10,§1-§8): audit & PLAN` | 454 | This document. **Lock decisions before code.** |
| **C2** | `fix(phase-10,§3.1): retire main.py benchmarking imports (D10.1)` | 454 | `main.py` only. `run_evaluate` body becomes a deprecation pointer to `make phase-6-eval`. Imports removed; `run_train_all_rl`'s call to `run_evaluate` left intact (it'll surface the deprecation message). |
| **C3** | `fix(phase-10,§3.2): delete dead src/benchmarking/ package (D10.2)` | **411** | `rm -r src/benchmarking/` + `rm tests/test_benchmark_runner.py tests/test_metrics_collector.py`. Pytest re-run; baseline 411 confirmed. |
| **C4** | `fix(phase-10,§3.3): delete pre-restart orphan scripts (D10.3)` | 411 | `rm scripts/{evaluate_generator,measure_improved_targets,separability_analysis}.py`. |
| **C5** | `docs(phase-10,§4): rewrite README.md + retire pre-restart docs (D10.4, D10.8)` | 411 | New README. Annotate (front-matter) `docs/benchmarking-results.md` and `docs/metrics-glossary.md` as `STATUS: pre-restart, retained for historical reference; superseded by docs/results/06_benchmark/RESULTS.md`. Add `10_release/` row to `docs/results/README.md`. |
| **C6** | `docs(phase-10,§5): CITATION.cff thesis Person (D10.5)` | 411 | `CITATION.cff` author block. |
| **C7** | `docs(phase-10,§6): close — RESULTS + CHANGELOG + tag v0.1.0 (D10.6, D10.7, D10.10)` | 411 | `docs/results/10_release/{RESULTS.md, G10_scoreboard.json}`, prepend `[Unreleased] — Phase 10` to `CHANGELOG.md`, run `git tag -a v0.1.0 -m "Phase-10 closeout — open-source release"`. |
| **C8** | (manual, by user) | — | `git push origin feature/reward-shaping --tags`. |

## 6 — What we are NOT doing in Phase 10

These items are explicitly out of scope. Do **not** add them mid-phase.

- **No Phase 8** (F13 obs-noise / F14 train-time augmentation). That is
  a separate decision (D2 in HANDOFF), already deferred.
- **No re-running of any sweep.** Phase 5/6/7 numbers stay frozen.
  `runs/phase{5,6,7}/` are not touched. Manifests in
  `docs/results/0[5-7]_*/` are not regenerated.
- **No edits to `docs/results/0[2-7]_*/RESULTS.md`.** Those chapters are
  locked. New cross-references go in `10_release/RESULTS.md` only.
- **No `src/` restructure** outside the `src/benchmarking/` deletion.
  `src/benchmark/`, `src/blue_team/`, `src/environment/`, etc. are
  unchanged.
- **No new tests.** C2's deprecation-pointer change is too thin to
  justify a test (a one-line `print` + `return False`). Test count
  goes 454 → 411 monotonically.
- **No CLI changes** beyond the `--mode evaluate` deprecation pointer.
  `--mode evaluate` itself stays as a CLI flag (deprecated, prints
  pointer, exits non-zero); `--mode train-all-rl` is unchanged in
  surface (its internal call to `run_evaluate` will surface the
  deprecation as part of the user-visible output).
- **No version bump** beyond `v0.1.0`. Future releases get later tags.
- **No public-release-readiness audit beyond what's listed.** Things
  like LICENSE-header sweeps, dependency pinning, Docker images,
  ReadTheDocs setup are out of scope; can be a Phase 11 if ever.

## 7 — Risks tracked

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| **R10.1** | An import of `src.benchmarking` exists outside the three known consumers and gets stranded by C3, surfacing as a `ModuleNotFoundError` at pytest time. | Low | Medium | Pre-C3 grep is part of the audit (§2.1) and was clean. C3 itself runs `pytest -q` and fails the commit if the count is not exactly 411. |
| **R10.2** | A user-facing doc references `--mode evaluate` and would be misleading after C2's deprecation. | Confirmed (README.md:181, 191) | Low | Both occurrences are inside the README that is being rewritten in C5; no other doc cites the flag (verified `grep -rn "mode evaluate"` 2026-05-02). |
| **R10.3** | The README rewrite breaks an internal cross-link from `docs/` to `README.md`. | Low | Low | `grep -rn "README\.md" docs/` (2026-05-02) finds only HANDOFF + `docs/results/README.md` references; both are about *this* phase or about a different README (`docs/results/README.md`). |
| **R10.4** | The git tag `v0.1.0` is created against an unintended commit (e.g. detached HEAD or local-only branch). | Low | Medium | C7 explicitly verifies `git rev-parse HEAD` matches the closeout commit before tagging. The user pushes with `--tags` in C8. |
| **R10.5** | Pytest count drift between machines (e.g., a platform-conditional skip not present on user's macOS) makes G10.1's exact-411 threshold brittle. | Low | Low | Threshold is `411 passed, 0 errors, 0 failed`; new skips are forbidden. Existing platform-conditional tests in the suite are already stable across the eight closed phases (HANDOFF: "454 passed, 0 skipped"). |

## 8 — Locked design decisions

These are locked at PLAN-commit time. Subsequent `fix(phase-10,…)`
or `docs(phase-10,…)` commits implementing them must cite the
relevant `D10.X` ID. Any deviation requires a new dated `D10.X.1`
entry below the original (AF3 protocol-continuity precedent set by
D5.4.1 / D6.2.1 / D7.1.1 / D7.9.1).

### D10.1 — `main.py --mode evaluate` becomes a deprecation pointer

**Decision:** Replace the body of `main.py::run_evaluate` with a
short message directing the user to `make phase-6-eval` /
`scripts/benchmark/run_test_eval.py`, returning `False` so the
process exit-code is non-zero. Keep the CLI flag itself (so existing
shell scripts that call it surface a clear, actionable error
message rather than a `KeyError: 'evaluate'`).

**Rationale:** `src/benchmarking/` is dead (§2.1). The canonical
evaluation path that actually produced the Phase 6 RESULTS is
`scripts/benchmark/run_test_eval.py`, gated through `make
phase-6-eval`. Deleting the CLI flag would be a breaking-change
without sufficient warning; deprecating it preserves the surface
for one release and points readers at the live tooling.

**What this does NOT mean:** it is not a promise of indefinite
support. Future phases (Phase 11+) may delete the flag entirely.

### D10.2 — Pre-restart docs `benchmarking-results.md` + `metrics-glossary.md` annotate, do not delete

**Decision:** Both files are annotated with a top-of-file front-matter
banner declaring them pre-restart and superseded by
`docs/results/06_benchmark/RESULTS.md`. They are **not** deleted.

**Rationale:** They contain narrative text that may have value for
future readers tracing the project's history (the metrics glossary
in particular has definitions worth keeping). Annotating costs
nothing and preserves audit-honesty (HANDOFF rule 4). Deletion
loses information; future phases can revisit if cleanup is needed.

### D10.3 — `v0.1.0` is the canonical thesis-cited tag

**Decision:** The phase closes with `git tag -a v0.1.0 -m "Phase 10
closeout — eight phases, F0..F15 + T1, 411 tests, manifest
hash-chain reproducibility"`.

**Rationale:** Single source of truth for "the version of the repo
the thesis bibliography points at." `0.1.0` rather than `1.0.0`
because Phase 8 (and beyond) is still possible; this is a research
release, not a product release. SemVer guidance: 0.x.y for
"public API may change."

### D10.4 — README is a thesis-aware re-write, not a patch

**Decision:** C5 produces a fully-replaced `README.md` rather than
an in-place patch. Voice: structured around the eight phases. New
sections, in order: hero / quick claim / what's in this repo /
phases-as-chapters (each with its headline F-figure inline, its
`make phase-N` recipe, and its key gate result) / quick start /
reproducibility / dataset citation / inspiring paper citation /
contributing / license / how to cite this work.

**Rationale:** The pre-restart README is so out of date that
in-place patches would leave fossils everywhere. A clean rewrite
is faster and produces a less embarrassing artefact.

### D10.5 — Test count goes 454 → 411 monotonically

**Decision:** No new tests added in Phase 10. The 43-test drop
(C3) is the only test-count change. G10.1 enforces `411 passed`
exactly.

**Rationale:** Phase 10 ships no new behaviour worth testing. The
deprecation pointer (D10.1) is one print + one return — if it
breaks, every CLI invocation will surface the breakage instantly;
a unit test would not catch what an end-user wouldn't. (Contrast:
the close_phase7 parsers in 7537493 *did* warrant 12 new tests
because the parser logic was a bug-prone non-trivial computation;
this is not that.)

## 9 — Time budget

- **C1**: ~1 h human (this PLAN). Done at PLAN-commit.
- **C2**: ~30 min human, 0 h CPU.
- **C3**: ~10 min human, 1 min CPU (pytest).
- **C4**: ~5 min human.
- **C5**: ~3–4 h human (the actual README rewrite is the bulk of
  Phase 10's work). 0 h CPU.
- **C6**: ~10 min human.
- **C7**: ~1 h human (RESULTS scaffold + CHANGELOG block + tag).
  0 h CPU.
- **C8**: trivial, manual.

**Total: ~6–8 h human, ~1 min CPU.** Phase 10 is bounded by writing
quality, not compute.

---

*PLAN locked at C1-commit time. Any subsequent change to §3
deliverables, §4 gate thresholds, §6 scope boundaries, or §8
decisions requires a new dated `D10.X.1` entry in §8.*
