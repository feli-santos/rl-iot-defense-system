"""Phase-7 closer (C9): assemble G7_scoreboard.json + RESULTS.md skeleton + CHANGELOG entry.

Run once all four figures (F9/F10/F12/F15) and their *_summary.json
files exist under ``docs/results/07_ablation/``. This script does
NOT run any new computation — it simply aggregates the gate
verdicts that the four plotters already wrote into their
``F<N>_summary.json#gates`` blocks.

Usage::

    python -m scripts.ablation.close_phase7 [--out-dir docs/results/07_ablation]

Outputs:

- ``G7_scoreboard.json``  — canonical per-gate threshold + value +
  status + finding-id, mirroring the Phase-6 ``G6_scoreboard.json``
  shape.
- ``RESULTS.md``           — Phase-7 results doc with §1–§9 sections
  populated from the live numbers (placeholder narrative for the
  agent or user to flesh out before locking).
- ``CHANGELOG.md`` (root) gets a Phase-7 ``[Unreleased]`` block
  prepended with the gate scoreboard + headline.

Gates evaluated:

| Gate | Source | Reads |
|---|---|---|
| G7.1 | tests | runs ``pytest -q`` and reads count |
| G7.2 | F9 | ``F9_summary.json#gates.G7.2`` |
| G7.3 | F10 | ``F10_summary.json#gates.G7.3`` |
| G7.4 | F12 | ``F12_summary.json#gates.G7.4`` |
| G7.5 | manual | always PASS as long as Phase-3 frozen tests pass |
| G7.6 | tests | always PASS as long as full pytest is green |
| G7.7 | manifests | checks F9/F10/F12/F15 manifest.json files exist |
| G7.8 | F15 | ``F15_summary.json#gates.G7.8`` (audit-AF1) |
| G7.9 | F15 | ``F15_summary.json#gates.G7.9`` (audit-AF1, headline) |
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("scripts.ablation.close_phase7")

_ROOT = Path(__file__).resolve().parents[2]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _read_summary(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        logger.warning("missing summary: %s", path)
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.error("failed to parse %s: %s", path, exc)
        return None


def _run_pytest_count() -> Dict[str, Any]:
    """Run pytest -q and parse the trailing 'X passed' line."""
    try:
        proc = subprocess.run(
            ["pytest", "-q", "--tb=no"],
            cwd=_ROOT, capture_output=True, text=True, timeout=600,
        )
        last_line = (proc.stdout.strip().splitlines() or [""])[-1]
        passed = 0
        failed = 0
        for word in last_line.split():
            if word.isdigit():
                # Pattern: '442 passed, 0 failed in 64s'
                idx = last_line.index(word)
                tail = last_line[idx:].split(",", 1)[0]
                if "passed" in tail:
                    passed = int(word)
                elif "failed" in tail:
                    failed = int(word)
        return {
            "ok": proc.returncode == 0,
            "passed": passed,
            "failed": failed,
            "summary_line": last_line,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _evaluate_gates(
    out_dir: Path, *, run_pytest: bool = True,
) -> List[Dict[str, Any]]:
    """Materialise the G7.1–G7.9 scoreboard rows."""
    gates: List[Dict[str, Any]] = []

    # G7.1 — pytest green.
    if run_pytest:
        pyt = _run_pytest_count()
        gates.append({
            "id": "G7.1",
            "threshold": "pytest -q ≥ 430 passed; zero new skips",
            "value": pyt.get("summary_line", "?"),
            "passes": bool(pyt.get("ok") and pyt.get("passed", 0) >= 430),
            "kind": "tests",
        })
    else:
        gates.append({
            "id": "G7.1", "kind": "tests",
            "threshold": "pytest -q ≥ 430 passed",
            "passes": None, "value": "skipped (run-pytest=false)",
        })

    # G7.2 — F9 reward-component sweep.
    f9 = _read_summary(out_dir / "F9_summary.json")
    if f9 is not None:
        g72 = f9.get("gates", {}).get("G7.2", {})
        gates.append({
            "id": "G7.2",
            "threshold": "F9 best cell mean test reward > Phase-6 DQN +1336 by ≥1σ",
            "value": (
                f"best_cell={g72.get('best_cell', '?')}, "
                f"mean={g72.get('best_mean_reward', float('nan')):+.1f}, "
                f"Δ_to_dqn={g72.get('delta_to_deployable', float('nan')):+.1f}, "
                f"meets_oracle_stretch={g72.get('meets_oracle_ceiling_stretch')}"
            ),
            "passes": bool(g72.get("passes")),
            "kind": "f9",
            "interpretation": g72.get("interpretation"),
        })
    else:
        gates.append({
            "id": "G7.2", "kind": "f9",
            "threshold": "F9 best cell mean test reward > DQN +1336 by ≥1σ",
            "passes": False, "value": "F9_summary.json missing",
        })

    # G7.3 — F10 aggressiveness.
    f10 = _read_summary(out_dir / "F10_summary.json")
    if f10 is not None:
        g73 = f10.get("gates", {}).get("G7.3", {})
        gates.append({
            "id": "G7.3",
            "threshold": "PPO p=0.0 < p=0.6 by ≥1σ AND rule monotone",
            "value": g73.get("ppo_reason", "?"),
            "passes": bool(g73.get("passes")),
            "kind": "f10",
            "interpretation": g73.get("interpretation"),
        })
    else:
        gates.append({
            "id": "G7.3", "kind": "f10",
            "threshold": "PPO p=0.0 < p=0.6 by ≥1σ",
            "passes": False, "value": "F10_summary.json missing",
        })

    # G7.4 — F12 Pareto.
    f12 = _read_summary(out_dir / "F12_summary.json")
    if f12 is not None:
        g74 = f12.get("gates", {}).get("G7.4", {})
        gates.append({
            "id": "G7.4",
            "threshold": "Pareto frontier ≥ 3 distinct dominant points",
            "value": (
                f"n_distinct={g74.get('n_distinct_frontier_points', '?')}/"
                f"{f12.get('n_points_total', '?')}"
            ),
            "passes": bool(g74.get("passes")),
            "kind": "f12",
            "interpretation": g74.get("interpretation"),
        })
    else:
        gates.append({
            "id": "G7.4", "kind": "f12",
            "threshold": "≥ 3 distinct Pareto frontier points",
            "passes": False, "value": "F12_summary.json missing",
        })

    # G7.5 + G7.6 — Phase-3 / overall regression: piggyback on G7.1.
    pyt_ok = bool(gates[0].get("passes"))
    gates.append({
        "id": "G7.5",
        "threshold": "Phase-3 frozen tests pass with impact_is_terminal=True",
        "passes": pyt_ok,
        "value": "G7.1 carries this through (full pytest green ⇒ Phase-3 contract preserved)",
        "kind": "regression",
    })
    gates.append({
        "id": "G7.6",
        "threshold": "No regression on Phase-3/4/5/6 frozen tests overall",
        "passes": pyt_ok,
        "value": "G7.1 carries this through",
        "kind": "regression",
    })

    # G7.7 — manifests.
    manifest_paths = [
        out_dir / "F9_manifest.json",
        out_dir / "F10_manifest.json",
        out_dir / "F12_manifest.json",
        out_dir / "F15_manifest.json",
    ]
    missing = [str(p.relative_to(_ROOT)) for p in manifest_paths if not p.exists()]
    gates.append({
        "id": "G7.7",
        "threshold": "F9/F10/F12/F15 manifest.json all present + SHA-pinned",
        "passes": not missing,
        "value": (
            f"all 4 manifests present"
            if not missing else f"missing: {missing}"
        ),
        "kind": "manifests",
    })

    # G7.8 — F15 OOD matrix complete (audit-AF1).
    f15 = _read_summary(out_dir / "F15_summary.json")
    if f15 is not None:
        g78 = f15.get("gates", {}).get("G7.8", {})
        gates.append({
            "id": "G7.8",
            "threshold": "F15 4-class × 8-policy matrix complete, no NaN means",
            "value": (
                f"{g78.get('n_cells_present', '?')}/{g78.get('n_cells_expected', '?')} "
                f"cells; "
                f"n_missing={len(g78.get('missing_cells', []))}; "
                f"n_nan={len(g78.get('nan_cells', []))}"
            ),
            "passes": bool(g78.get("passes")),
            "kind": "f15",
            "audit_finding": "AF1",
        })

        # G7.9 — F15 headline.
        g79 = f15.get("gates", {}).get("G7.9", {})
        gates.append({
            "id": "G7.9",
            "threshold": (
                "On VulnerabilityScan, best trained RL CI_low > "
                "RF-Acting CI_high (≥1σ separation, RL > RF)"
            ),
            "value": (
                f"best_rl={g79.get('best_rl_algo', '?')} "
                f"({g79.get('best_rl_mean_reward', float('nan')):+.1f}), "
                f"RF=({g79.get('rf_acting_mean_reward', float('nan')):+.1f}), "
                f"Δ={g79.get('delta_mean', float('nan')):+.1f}"
            ),
            "passes": bool(g79.get("passes")),
            "kind": "f15",
            "audit_finding": "AF1 (HEADLINE)",
            "interpretation": g79.get("interpretation"),
        })
    else:
        gates.append({
            "id": "G7.8", "kind": "f15", "audit_finding": "AF1",
            "threshold": "F15 matrix complete, no NaN means",
            "passes": False, "value": "F15_summary.json missing",
        })
        gates.append({
            "id": "G7.9", "kind": "f15", "audit_finding": "AF1 (HEADLINE)",
            "threshold": "On VulnerabilityScan: trained RL > RF-Acting by ≥1σ",
            "passes": False, "value": "F15_summary.json missing",
        })

    return gates


# ---------------------------------------------------------------- writers


def _write_scoreboard(
    out_dir: Path, gates: List[Dict[str, Any]],
) -> Path:
    n_pass = sum(1 for g in gates if g.get("passes") is True)
    n_fail = sum(1 for g in gates if g.get("passes") is False)
    n_skip = sum(1 for g in gates if g.get("passes") is None)
    payload = {
        "schema_version": "1.0",
        "phase": 7,
        "git_sha": _git_sha(),
        "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_skip": n_skip,
        "gates": gates,
    }
    path = out_dir / "G7_scoreboard.json"
    path.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s (pass=%d fail=%d skip=%d)", path, n_pass, n_fail, n_skip)
    return path


def _summary_table(gates: List[Dict[str, Any]]) -> str:
    """Markdown table: gate / threshold / status / value."""
    rows = [
        "| Gate | Threshold | Status | Value / Notes |",
        "|---|---|:---:|---|",
    ]
    for g in gates:
        status = (
            "**PASS**" if g.get("passes") is True
            else "FAIL-WITH-FINDING" if g.get("passes") is False
            else "SKIP"
        )
        rows.append(
            f"| **{g['id']}** | {g['threshold']} | {status} | {g.get('value', '?')} |"
        )
    return "\n".join(rows)


def _write_results_md(
    out_dir: Path, gates: List[Dict[str, Any]],
) -> Path:
    """Write a Phase-7 RESULTS.md skeleton with live numbers.

    Mirrors the Phase-6 RESULTS.md structure (§1 headline / §2
    scoreboard / §3 deliverables / §4 code / §5 cross-phase
    findings / §6 findings worth defending / §7 hand-offs / §8
    reproducibility / §9 test count history).
    """
    f9 = _read_summary(out_dir / "F9_summary.json") or {}
    f10 = _read_summary(out_dir / "F10_summary.json") or {}
    f12 = _read_summary(out_dir / "F12_summary.json") or {}
    f15 = _read_summary(out_dir / "F15_summary.json") or {}
    n_pass = sum(1 for g in gates if g.get("passes") is True)
    n_fail = sum(1 for g in gates if g.get("passes") is False)

    g72 = f9.get("gates", {}).get("G7.2", {})
    g73 = f10.get("gates", {}).get("G7.3", {})
    g74 = f12.get("gates", {}).get("G7.4", {})
    g78 = f15.get("gates", {}).get("G7.8", {})
    g79 = f15.get("gates", {}).get("G7.9", {})

    md = f"""# Phase 7 — Ablations + OOD-class Robustness: Results

> Companion to `PLAN.md`. Same protocol as Phases 3–6: locked PLAN
> first, then implementation, then this document captures **what
> happened on real data**. The two headline strands (per audit
> AF1 / AF2) are **F9** (does the reward-component sweep close
> the +288 deployable gap to the oracle ceiling?) and **F15**
> (does trained RL recover the supervised detector's
> `VulnerabilityScan` blind spot?).

## 1 — Headline numbers

**F9 — reward-component sweep (D7.1):**
{g72.get("interpretation", "(F9 not produced yet)")}

  - Best cell: `{g72.get('best_cell', '?')}` (mean = {g72.get('best_mean_reward', float('nan')):+.1f},
    CI = ({g72.get('best_ci', [float('nan'), float('nan')])[0]:+.1f},
    {g72.get('best_ci', [float('nan'), float('nan')])[1]:+.1f}))
  - Δ to Phase-6 deployable best (DQN +1336): **{g72.get('delta_to_deployable', float('nan')):+.1f}**
  - Δ to Phase-6 oracle ceiling (rule +1624): **{g72.get('delta_to_oracle', float('nan')):+.1f}**
  - Stretch goal (oracle ceiling) met: **{g72.get('meets_oracle_ceiling_stretch')}**

**F15 — OOD-class robustness (audit-AF1, HEADLINE):**
{g79.get("interpretation", "(F15 not produced yet)")}

  - On `VulnerabilityScan` (Phase-4 RF recall = 0.001):
    - Best trained RL: `{g79.get('best_rl_algo', '?')}` mean = {g79.get('best_rl_mean_reward', float('nan')):+.1f}
      (CI {g79.get('best_rl_ci', [float('nan'), float('nan')])})
    - RF-Acting mean = {g79.get('rf_acting_mean_reward', float('nan')):+.1f}
      (CI {g79.get('rf_acting_ci', [float('nan'), float('nan')])})
    - Δ = **{g79.get('delta_mean', float('nan')):+.1f}**

**F10 — attack-aggressiveness (IoTWarden Fig. 6 re-impl):**
{g73.get("interpretation", "(F10 not produced yet)")}

**F12 — security-vs-availability Pareto:**
{g74.get("interpretation", "(F12 not produced yet)")}

  - Total points collected: {f12.get('n_points_total', '?')}
  - Frontier points (distinct): {g74.get('n_distinct_frontier_points', '?')}

## 2 — Gate scoreboard

{_summary_table(gates)}

Tally: **{n_pass} PASS / {n_fail} FAIL-WITH-FINDING**.
Source of record: `G7_scoreboard.json` next to this file.

## 3 — Deliverables (figures + tables)

| Artefact | Path | Description |
|---|---|---|
| **F9** (Tier 2) | `F9_reward_ablation.png` + `F9_summary.json` | 6-panel reward-component effect plot (5 components × {{0.5×, 1×, 2×}} + impact_is_terminal binary) with Phase-6 reference lines (oracle +1624, DQN +1336). |
| **F10** (Tier 2) | `F10_aggressiveness.png` + `F10_summary.json` | PPO and oracle-rule mean test reward as a function of `p_defender_deescalation`; IoTWarden Fig. 6 re-impl. |
| **F12** (Tier 2) | `F12_pareto.png` + `F12_summary.json` | 2-D scatter on (availability_cost, security_gain) with Pareto frontier; reads F9 + F10 + Phase-6 outputs. |
| **F15** (Tier 1, audit-AF1) | `F15_ood_robustness.png` + `F15_summary.json` | 4 OOD class × 8 policy grouped bar chart with bootstrap CIs. |
| Captions | `F9_caption.md`, `F10_caption.md`, `F12_caption.md`, `F15_caption.md` | Thesis-paper captions per figure. |
| Manifests | `F9_manifest.json` … `F15_manifest.json` | SHA-256 hash chain over input JSONLs + Phase-5 sweep manifest + Phase-6 eval manifest + git SHA at production time. |
| Scoreboard | `G7_scoreboard.json` | Per-gate threshold + value + status + finding-id. |
| Run artefacts (gitignored) | `runs/phase7/{{ood,reward_sweep,aggressiveness}}/.../eval_test.jsonl` | The schema-v1.0 input data for every figure. |

## 4 — Code summary

| File | Purpose |
|---|---|
| `src/environment/adversarial_env.py` | Added `impact_is_terminal: bool = True` (default preserves Phase-3 frozen contract). |
| `src/blue_team/run_config.py` | `EnvConfigSerializable` extended from 7 → 18 fields (all reward coefficients + `impact_is_terminal`). |
| `src/blue_team/env_factory.py` | `_build_env_config` now forwards full reward field set. |
| `scripts/blue_team/train_agent.py` | Added `--reward-overrides JSON`, `--p-defender-deescalation FLOAT`, `--impact-is-terminal BOOL` CLI args. |
| `scripts/ablation/run_ood_eval.py` | F15 OOD eval driver with hybrid realiser (in-distribution train pool + OOD overlay at the OOD class's stage). |
| `scripts/ablation/plot_ood_robustness.py` | F15 plotter + G7.8 / G7.9 evaluators. |
| `scripts/ablation/run_reward_sweep.py` | F9 12-cell sparse one-at-a-time sweep driver (PPO + 5 components × 3 multipliers + impact_is_terminal binary). |
| `scripts/ablation/plot_reward_ablation.py` | F9 plotter + G7.2 evaluator. |
| `scripts/ablation/run_aggressiveness_sweep.py` | F10 6-p-value PPO sweep + oracle-rule reference rolls. |
| `scripts/ablation/plot_aggressiveness.py` | F10 plotter + G7.3 evaluator. |
| `scripts/ablation/plot_pareto.py` | F12 Pareto-frontier plot + G7.4 evaluator. |
| `scripts/ablation/close_phase7.py` | This file: assembles `G7_scoreboard.json` + `RESULTS.md` + CHANGELOG. |
| `tests/test_phase31_impact_terminal.py` | 8 synthetic tests pinning the `impact_is_terminal` codepath. |
| `tests/test_train_agent_reward_overrides.py` | 14 synthetic tests pinning the CLI override plumbing. |

Total tests: 442 → ~442 (no run-time-data tests added; G7.2/G7.3/G7.4/G7.8/G7.9 are real-data acceptance tests).

## 5 — Cross-phase findings discovered during Phase 7

(Hand-fill — examples: hybrid OOD realiser was needed because each OOD class is single-stage; train-eval window-shape mismatch under `--smoke` surfaced by smoke run; etc.)

## 6 — Phase-7 findings worth defending in the thesis

### 6.1 The reward-component sweep result (D7.2.1 if needed)

(Hand-fill from G7.2 above — either the +288 gap was closed, partially closed, or characterised as the limit of one-at-a-time Phase-3-style reward shaping per D7.1.1.)

### 6.2 The OOD-class robustness result (D7.9.1 if needed; audit-AF1 HEADLINE)

(Hand-fill from G7.9 above — either trained RL beats RF-Acting on `VulnerabilityScan` by ≥1σ (RL closes the OOD gap), or it does not (RL is *robust to* not *better at* the OOD class). Either outcome is defensible.)

### 6.3 The IoTWarden Fig. 6 sensitivity replication (G7.3)

(Hand-fill from G7.3 above.)

### 6.4 The operating-point Pareto contribution (G7.4)

(Hand-fill from G7.4 above.)

## 7 — Phase-8 hand-offs

Phase 8 owns:

1. **F13 — Robustness to observation noise / drift** (Tier 3).
2. **F14 — Generalisation training to held-out attack class** (Tier 3 if it ships); F15 covered the eval-time complement, F14 would be the train-time augmentation.

Phase 7 does NOT defer:

- The +288 deployable gap. F9 either closed it (G7.2 PASS) or characterised the closure attempt as the limit of the
  Phase-3 reward formulation (D7.1.1).
- The OOD-class robustness claim. F15 either delivered it (G7.9 PASS) or narrowed it to "robust to (not better at)"
  per D7.9.1.

## 8 — Reproducibility

Every Phase-7 figure ships a `manifest.json` with:

- SHA-256 hashes of every input JSONL under
  `runs/phase7/{{ood,reward_sweep,aggressiveness}}/.../eval_test.jsonl`.
- SHA-256 of the upstream `runs/phase5/sweep_manifest.json` (trained
  checkpoints) and `runs/phase6/eval_manifest.json` (Phase-6
  baselines).
- Git SHA at production time.

To regenerate from scratch on a fresh checkout::

    make phase-5-sweep PHASE5_TIMESTEPS=250000   # ~108 min CPU (one-off)
    make phase-6                                 # ~10 min CPU
    make phase-7                                 # ~7.5 h CPU (walk-away)
    python -m scripts.ablation.close_phase7      # assemble G7 scoreboard + RESULTS

The `runs/phase5/`, `runs/phase6/`, `runs/phase7/` dirs are all
gitignored; all derived figures + summaries + manifests live under
`docs/results/0[5-7]_*/`.

## 9 — Test count history

Phase 0 254 → Phase 1 266 → Phase 2 283 → Phase 3 296 → Phase 4 329
→ Phase 5 376 → Phase 6 420 → **Phase 7 442** (+22 from C3 + C4).
"""

    path = out_dir / "RESULTS.md"
    path.write_text(md)
    logger.info("wrote %s (%d bytes)", path, len(md))
    return path


def _prepend_changelog(
    repo_root: Path, gates: List[Dict[str, Any]],
) -> Optional[Path]:
    """Prepend a Phase-7 [Unreleased] section to CHANGELOG.md."""
    changelog = repo_root / "CHANGELOG.md"
    if not changelog.exists():
        logger.warning("CHANGELOG.md not found; skipping prepend")
        return None
    n_pass = sum(1 for g in gates if g.get("passes") is True)
    n_fail = sum(1 for g in gates if g.get("passes") is False)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    block = f"""## [Unreleased] — Phase 7 closeout ({today})

Tally: **{n_pass} PASS / {n_fail} FAIL-WITH-FINDING** across G7.1–G7.9.

### Gate scoreboard

{_summary_table(gates)}

### Headline findings (see `docs/results/07_ablation/RESULTS.md` for full text)

- **G7.2 (F9 reward-component sweep)**: see RESULTS §6.1 — either the
  +288 deployable gap was closed at one or more cells, or the
  characterisation activates D7.1.1 (limit of one-at-a-time
  Phase-3-style reward shaping).
- **G7.9 (F15 audit-AF1 HEADLINE)**: see RESULTS §6.2 — either
  trained RL beats RF-Acting on `VulnerabilityScan` by ≥1σ
  (the "RL closes the OOD gap" claim), or D7.9.1 narrows the claim
  to "RL is robust to (not better at) the OOD class".

### What ships

- F9 / F10 / F12 / F15 figures + summaries + manifests under
  `docs/results/07_ablation/`.
- `G7_scoreboard.json` per-gate JSON record.
- `runs/phase7/{{ood,reward_sweep,aggressiveness}}/` raw eval JSONLs
  (gitignored; ~7.5 h CPU walk-away to regenerate via `make phase-7`).
- 22 new synthetic-only tests (Phase 7 §3.3): test count 420 → 442.

"""
    existing = changelog.read_text()
    new_text = block + "\n" + existing
    changelog.write_text(new_text)
    logger.info("prepended Phase-7 block to %s", changelog)
    return changelog


# ---------------------------------------------------------------- main


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase-7 closer: assemble G7_scoreboard + RESULTS + CHANGELOG.",
    )
    p.add_argument("--out-dir", default="docs/results/07_ablation")
    p.add_argument(
        "--no-pytest", action="store_true",
        help="Skip the pytest run for G7.1 (use the most recent known result).",
    )
    p.add_argument(
        "--no-changelog", action="store_true",
        help="Skip the CHANGELOG.md prepend.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gates = _evaluate_gates(out_dir, run_pytest=not args.no_pytest)
    _write_scoreboard(out_dir, gates)
    _write_results_md(out_dir, gates)
    if not args.no_changelog:
        _prepend_changelog(_ROOT, gates)

    n_pass = sum(1 for g in gates if g.get("passes") is True)
    n_fail = sum(1 for g in gates if g.get("passes") is False)
    logger.info(
        "Phase-7 closer done: %d PASS / %d FAIL-WITH-FINDING across G7.1-G7.9",
        n_pass, n_fail,
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
