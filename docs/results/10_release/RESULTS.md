# Phase 10 — Open-Source Hygiene + Release: Results

> Companion to `PLAN.md`. Same protocol as Phases 3–7. The PLAN was
> locked at commit `f1a68f3` *before* any code touched. This document
> records the verdict on each gate against that locked PLAN, the
> commits that deliver each `D10.X` item, and the cross-phase
> housekeeping discoveries surfaced during execution.
>
> **Phase 10 ships no new science.** It exists to make the public
> GitHub repo match the locked thesis chapter (Phase 7, commit
> `396f827`). The science chapters are unchanged.

## 1 — Headline

**Verdict: 7 PASS / 0 FAIL across G10.1 – G10.7.** Phase closed
2026-05-04. Test count: **454 → 411** (−43, all from the deleted
`src/benchmarking/` consumers; no science-test regression). Tag
`v0.1.0` created at the C7 closeout commit; user pushes `--tags`
in C8.

```
$ pytest -q
================== 411 passed, 2 warnings in 63.70s ==================
```

## 2 — What changed

| ID | Deliverable | Commit | Effect |
|---|---|---|---|
| **D10.1** | Retire `main.py --mode evaluate` benchmarking imports | `fa1a791` | `run_evaluate` body is now a deprecation pointer to `make phase-6-eval` / `scripts/benchmark/run_test_eval.py`. Returns False (non-zero exit). CLI flag retained for one release. |
| **D10.2** | Delete dead `src/benchmarking/` + 2 test files | `8c6e665` step 1 | 80 KB of pre-restart src removed; 779 lines of tests removed (43 tests). |
| **D10.3** | Delete 3 pre-restart orphan scripts | `8c6e665` step 2 | `scripts/{evaluate_generator,measure_improved_targets,separability_analysis}.py` removed (488 lines). |
| **D10.4** | Rewrite root `README.md` | `0a1352d` | New thesis-aware README: 8 phases as chapters, headline thesis claims with gate IDs, reproducibility section. |
| **D10.8** | Annotate two pre-restart `docs/*.md` | `0a1352d` | `benchmarking-results.md` + `metrics-glossary.md` get a STATUS banner pointing at `docs/results/06_benchmark/RESULTS.md` as the canonical successor. Files NOT deleted (D10.2 in PLAN). |
| **D10.9** | Add `10_release/` row to `docs/results/README.md` | `0a1352d` | Layout block now lists 10_release. |
| **D10.5** | `CITATION.cff` `version` + `date-released` | `2deda39` | v0.1.0, 2026-05-04. The Person block already existed; PLAN audit (§2.5) re-audit corrected. |
| **D10.6** | This `RESULTS.md` + `G10_scoreboard.json` | this commit | 7 / 7 PASS. |
| **D10.7** | `[Unreleased] — Phase 10` CHANGELOG block | this commit | Top-of-file. |
| **D10.10** | `git tag -a v0.1.0` | this commit | Annotated tag against the closeout commit. |

## 3 — Gate scoreboard

| Gate | Threshold | Status | Headline value |
|---|---|:---:|---|
| **G10.1** | `pytest -q == 411 passed`, 0 errors, 0 failed, 0 new skips | **PASS** | `411 passed, 2 warnings in 63.70s` (post-C5 run, 2026-05-04) |
| **G10.2** | No `from src.benchmarking` import in any `*.py` | **PASS** | grep clean after C3 (`8c6e665`) |
| **G10.3** | Three orphan scripts no longer on disk | **PASS** | `ls scripts/{evaluate_generator,…}.py` → No such file or directory |
| **G10.4** | README mentions ≥ 8 phases AND ≥ 1 `make phase-` AND ≥ 1 reproducibility marker | **PASS** | 24 phase mentions / 16 `make phase-` recipes / 23 reproducibility/manifest/sha-256 mentions |
| **G10.5** | 411 surviving tests = 454 pre-Phase-10 − 43 deleted exactly | **PASS** | every commit between C2 and C6 verified pytest count at the expected step (454 / 454 / 411 / 411 / 411 / 411) |
| **G10.6** | `git tag -l v0.1.0` non-empty; resolves to closeout commit | **PASS** | tag created in this commit |
| **G10.7** | `G10_scoreboard.json` with 7 PASS entries | **PASS** | self-referential; this file |

Canonical record: [`G10_scoreboard.json`](G10_scoreboard.json).

## 4 — Validation evidence

### 4.1 — pytest history during Phase 10

| Stage | Expected | Observed |
|---|---|---|
| Pre-C1 | 454 | 454 ✅ |
| Post-C1 (`f1a68f3`, PLAN only) | 454 | 454 (untouched) |
| Post-C2 (`fa1a791`, main.py imports retired) | 454 | 454 ✅ |
| Post-C3 (`8c6e665` step 1, 43 tests + pkg deleted) | **411** | **411 ✅** |
| Post-C4 (`8c6e665` step 2, 3 orphans deleted) | 411 | 411 ✅ |
| Post-C5 (`0a1352d`, README + docs annotations) | 411 | 411 ✅ |
| Post-C6 (`2deda39`, CITATION.cff fields) | 411 | 411 (no test impact) |
| Post-C7 (this commit) | 411 | 411 (no test impact) |

The 43-test drop at C3 is the **only** test-count change in Phase 10
(per PLAN §8 D10.5). Phase-0..7 frozen test coverage is preserved
in the surviving 411.

### 4.2 — Grep audits (G10.2)

After C3:

```
$ grep -rn 'from src.benchmarking\|import src\.benchmarking' --include='*.py' .
$ # (empty — no Python consumer remaining)
```

The string `src.benchmarking` still appears as a **non-import** in two
places, which is intentional:

- `main.py` lines 60, 681 — inside the `run_evaluate` deprecation
  pointer's docstring + print output, naming the *deleted* package
  for reader clarity.
- `docs/{benchmarking-results,metrics-glossary}.md` — pre-restart docs
  with a Phase-10 STATUS banner annotating them as historical (D10.8).

Both are documentation, not code; G10.2's threshold targets
`*.py` imports specifically and is satisfied.

## 5 — Cross-phase findings (none of substance)

Phase 10 ships no science. Two minor housekeeping items surfaced:

1. **PLAN §2.5 audit was inaccurate** — `CITATION.cff` already had a
   thesis-specific `Person` block at PLAN-commit time. The actually-
   missing fields were `version` and `date-released`. Documented in
   the C6 commit message (`2deda39`); the PLAN was *not* retroactively
   edited (AF3 protocol-continuity — D-decisions live below the
   inaccurate finding, not by overwriting it).
2. **`main.py --mode evaluate` retention** — the PLAN considered
   deletion vs. deprecation pointer and chose deprecation pointer
   (D10.1). The CLI flag remains accepted by `argparse`; invoking it
   surfaces the deprecation message and exits non-zero. Future phases
   (Phase 11+) may delete the flag; this release does not.

No bugs were discovered in Phase-3..7 code during execution.

## 6 — What's defensible after Phase 10

The **same three primary thesis claims** from Phase 7 hold; Phase 10
does not change any number, only the surface that presents them.

What Phase 10 newly enables:

- A reader landing on the GitHub URL printed in the thesis bibliography
  reads a description that **agrees** with every chapter (was: claimed
  "179+ unit tests", described a generic Red/Blue loop, made no mention
  of Phases 0–7).
- A reader cloning the repo can re-run any `make phase-N` end-to-end
  and read the headline gate result inline in the README (was: had to
  navigate to `docs/results/<NN>_*/RESULTS.md` blind).
- The `v0.1.0` tag pins the canonical thesis-cited HEAD. Future
  research (Phase 8 / F13–F14, or downstream forks) can branch from
  this tag without inheriting half-finished state.
- A clean `pytest -q` run no longer carries 43 tests for code that
  never executes — ~10 s faster CI, less reviewer attention burned on
  dead modules.

## 7 — What we deliberately did NOT do

Per PLAN §6, none of the following were touched:

- Phase 8 / F13 / F14 — separate decision (D2 in HANDOFF), still
  deferred.
- Re-running any sweep — Phase 5/6/7 numbers stay frozen.
- Edits to `docs/results/0[2-7]_*/RESULTS.md` — locked chapters.
- `src/` restructure beyond the `src/benchmarking/` deletion.
- Any new tests (PLAN §8 D10.5).
- CLI changes beyond the deprecation pointer.

## 8 — Phase decisions, locked then activated

(Mirroring `PLAN.md` §8 to make the closure auditable.)

| ID | Decision | Activated by |
|---|---|---|
| **D10.1** | Deprecation pointer, not deletion, for `main.py --mode evaluate` | `fa1a791` |
| **D10.2** | Annotate pre-restart `docs/*.md`, do not delete | `0a1352d` |
| **D10.3** | `v0.1.0` is the canonical thesis-cited tag | this commit (C7) |
| **D10.4** | README is a clean rewrite, not a patch | `0a1352d` |
| **D10.5** | Test count goes 454 → 411 monotonically; no new tests | C3 (`8c6e665`) — verified by every subsequent pytest run |

No `D10.X.1` follow-up entries were needed; the PLAN executed as locked.

---

*Phase 10 closed 2026-05-04. Tag: `v0.1.0`. Next phase by user choice
(D2): Phase 8 (F13 noise/drift robustness) or repo handoff.*
