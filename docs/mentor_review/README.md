# Mentor Review

This directory is the **single source of truth** for the thesis-mentor
walkthrough that finalises the project for the MSc defense at
Unicamp/FEEC.

It exists in addition to (not replacing) the per-phase
`docs/results/<NN>_<name>/RESULTS.md` chapters. Those are the
*scientific* record of what each phase produced; this directory is the
*pedagogical* record of how a fresh pair of eyes evaluated and signed
off on those results, end-to-end, before the dissertation is bound.

## Naming convention

For each step in the review walkthrough we ship a pair of files:

| File | Purpose |
|---|---|
| `<NN>_<step>.md` | **Mentor memo.** What was reviewed, the verdict (pass / pass-with-fixes / fail), the findings, the actions taken or recommended. |
| `<NN>_HANDOFF.md` | **Resume-handoff.** A self-contained file that lets a fresh agent (or future-you, after a context-window reset) pick up the walkthrough at the next step without re-discovering anything. |

`<NN>` is two digits matching the phase number, except for `00_*` which
covers framing / scope / chapter outline (no phase). Step numbers and
phase numbers thus align: step 1 reviews Phase 1, step 6 reviews Phase
6, etc.

## Walkthrough plan

| # | Step | Reviews | Key artifacts |
|---|---|---|---|
| 0  | Framing & scope     | Audience, claims, chapter outline   | `00_framing.md`, `00_HANDOFF.md` |
| 1  | Phase 0–1 dataset   | F0a, F0b, splits, kill-chain map    | `01_dataset.md` |
| 2  | Phase 2 red team    | F1, F2, LSTM convergence            | `02_red_team.md` |
| 3  | Phase 3 environment | MDP correctness, reward, gates      | `03_env.md` |
| 4  | Phase 4 detector    | F11, realism                        | `04_detector.md` |
| 5  | Phase 5 blue team   | F3, F4, T1, G5                      | `05_blue_team.md` |
| 6  | Phase 6 benchmarks  | F5, F6, F7, F8, G6                  | `06_benchmark.md` |
| 7  | Phase 7 ablations   | F9, F10, F12, F15, G7 + refactor    | `07_ablation.md` |
| 8  | Cross-cutting audit | Stat tests, oracle, threats         | `08_audits.md` |
| 9  | LaTeX rebuild       | `tex/*` aligned with real results   | `09_latex.md` |
| 10 | Release polish      | README, CITATION, repo metadata     | `10_release.md` |

## How to resume after a context-window reset

A fresh agent should follow this exact recipe:

1. Read **this README** to understand the directory's purpose.
2. Read **the most recent `<NN>_HANDOFF.md`** (highest `<NN>` on disk).
3. Follow that file's *"Context-loading recipe for a fresh agent"*
   section verbatim.
4. Continue with the *"Outstanding actions for next session"* checklist.

The handoff files are the canonical resume points. The mentor memos are
reference material. If a memo and a handoff disagree, trust the
**later** of the two, because handoffs are written last in each step.

## Authoring conventions

- **Mentor memos** are written in present tense, lead with the verdict,
  cite figures by their canonical ID (F0a, F1, F11, etc.), and keep
  numerical claims linkable to the underlying JSON or PNG.
- **Handoffs** are written for an agent that has zero context, in
  imperative voice. Every command is copy-pasteable. Every file path is
  absolute or repo-relative-from-root.
- Neither file ever silently rewrites history. If a verdict in step N
  is later overturned in step M (M > N), step M issues a *correction*
  in its memo and links the original step N memo for traceability —
  nothing in step N is edited.

## Relationship to other docs

| If you want to know… | Read… |
|---|---|
| What the project *is* and how to run it | root `README.md` |
| What each phase *did* (locked, scientific) | `docs/results/<NN>_<name>/RESULTS.md` |
| What was *planned* before each phase ran | `docs/results/<NN>_<name>/PLAN.md` |
| Numerical gate verdicts | `docs/results/<NN>_<name>/G<N>_scoreboard.json` |
| Reproducibility / hash chain | `docs/reproducibility.md` |
| **How a mentor evaluated the thesis end-to-end** | **this directory** |
| Where each figure lives in the thesis | `docs/thesis_results_map.md` |

## Status

Started 2026-05-05. Owner: thesis-mentor agent (handed off via
`<NN>_HANDOFF.md` files between sessions). Audience: the candidate
(Felipe Santos), the advisor (Prof. Denis Fantinato, FEEC/UNICAMP),
and the defense committee.
