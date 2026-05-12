# docs/archive/

This directory holds pre-mentor-review historical documents that have been
superseded by the eight-step mentor-review walkthrough in
`docs/mentor_review/`. They are **preserved intact** (not edited) because
their prose is referenced from locked audit artefacts:

| File | Original path | Why preserved |
|---|---|---|
| `HANDOFF.md` | `docs/HANDOFF.md` | Audit narrative in §"Audit-fix cycle on 2026-05-01" is referenced from `docs/results/07_ablation/RESULTS.md §5` and from commit messages `7537493` / `396f827`. |
| `benchmarking-results.md` | `docs/benchmarking-results.md` | Pre-restart benchmarking summary; referenced in Phase-10 PLAN.md §3 and the `[v0.1.0]` CHANGELOG entry. Metric definitions remain useful as historical reference. |
| `metrics-glossary.md` | `docs/metrics-glossary.md` | Metric definitions for the pre-restart `src/benchmarking/` package; referenced alongside `benchmarking-results.md`. |

**Do not edit these files.** Per `docs/mentor_review/README.md §"Authoring
conventions"`, historical documents are immutable. If a correction is
needed, issue it as a note in a new mentor-review memo and link back here.

**For the current project state**, read:
- `docs/mentor_review/10_HANDOFF.md` — the final mentor-review handoff
  (mentor-review loop closed at Step 10).
- Root `README.md` — project overview and reproduction recipes.
