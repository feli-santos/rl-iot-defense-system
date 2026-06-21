#!/usr/bin/env python3
"""JSON -> LaTeX fragment generator (anti-drift tooling).

Reads the canonical redesign summary JSONs under ``docs/results/`` and emits:
  - ``tex/generated/numbers.tex``   : ``\\newcommand`` macros for headline numbers
  - ``tex/generated/tables.tex``    : ``\\input{}``-able table bodies

Source of truth for the partially-observable redesign:
  - ``docs/results/ablation/Falpha_summary.json``    : observation-aliasing sweep
  - ``docs/results/ablation/Fcoupling_summary.json`` : coupled-vs-outcome ablation

Intended to be re-run after every data regeneration so prose numbers are always
mechanically derived from the canonical JSON source of truth. LaTeX control
sequences must be all-letters after the backslash, so alpha levels are spelled
out (``\\AlphaZeroPPO`` etc.).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (canonical data)
# ---------------------------------------------------------------------------

FALPHA = Path("docs/results/ablation/Falpha_summary.json")
FCOUPLING = Path("docs/results/ablation/Fcoupling_summary.json")
TEST_COUNT = Path("docs/results/test_count.json")

# Spelled-out alpha keys (LaTeX macro names cannot contain digits or dots).
_ALPHA_WORD = {
    "0.0": "Zero",
    "0.2": "Two",
    "0.4": "Four",
    "0.6": "Six",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Canonical JSON not found: {path}")
    return json.loads(path.read_text())


def _newcmd(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


# ---------------------------------------------------------------------------
# Macro rendering
# ---------------------------------------------------------------------------


def _render_numbers() -> str:
    fa = _load(FALPHA)
    fc = _load(FCOUPLING)
    per_alpha = fa["per_alpha"]
    crossover = {c["alpha"]: c for c in fa["crossover"]["per_alpha"]}

    lines = [
        f"% Auto-generated from {FALPHA} and {FCOUPLING} -- do not hand-edit.",
        "% Regenerate with: PYTHONPATH=. .venv/bin/python -m scripts.thesis.render_tables",
    ]

    # Headline agent + seed count (PPO is the sole headline RL agent).
    n_seeds = per_alpha["0.0"]["ppo"].get("n_seeds", 10)
    lines.append(_newcmd("BestAgentName", "PPO"))
    lines.append(_newcmd("NumSeeds", f"{n_seeds:d}"))

    # Per-alpha reward macros for PPO (headline), tuned RF-Acting, and the oracle
    # ceiling, plus the PPO-minus-RF crossover gap and its significance verdict.
    for akey, word in _ALPHA_WORD.items():
        cell = per_alpha[akey]
        ppo = cell["ppo"]
        rf = cell["rf_acting"]
        orc = cell["recommended_action"]
        lines.append(_newcmd(f"Alpha{word}PPO", f"{ppo['mean']:+0.1f}"))
        lines.append(_newcmd(f"Alpha{word}PPOCILow", f"{ppo['ci_low']:+0.1f}"))
        lines.append(_newcmd(f"Alpha{word}PPOCIHigh", f"{ppo['ci_high']:+0.1f}"))
        lines.append(_newcmd(f"Alpha{word}RF", f"{rf['mean']:+0.1f}"))
        lines.append(_newcmd(f"Alpha{word}Oracle", f"{orc['mean']:+0.1f}"))
        cr = crossover[float(akey)]
        lines.append(_newcmd(f"Alpha{word}Gap", f"{cr['ppo_minus_rf']:+0.1f}"))
        sig = "significant" if cr["verdict"] == "ppo_significant" else "overlapping"
        lines.append(_newcmd(f"Alpha{word}Verdict", sig))

    # Convenience aliases for the headline anchor (alpha=0) and operating point
    # (alpha=0.4, where the PPO advantage first becomes significant).
    lines.append(_newcmd("HeadlineAlpha", "0.4"))
    lines.append(_newcmd("AnchorPPO", f"{per_alpha['0.0']['ppo']['mean']:+0.1f}"))
    lines.append(_newcmd("AnchorRF", f"{per_alpha['0.0']['rf_acting']['mean']:+0.1f}"))
    lines.append(
        _newcmd("OracleCeiling", f"{per_alpha['0.0']['recommended_action']['mean']:+0.1f}")
    )

    # Coupled-vs-outcome ablation gaps (RF-Acting minus best RL; negative => RL wins).
    gc = fc["gap_coupled"]
    go = fc["gap_outcome"]
    lines.append(_newcmd("CouplingGapCoupled", f"{gc:+0.1f}"))
    lines.append(_newcmd("CouplingGapOutcome", f"{go:+0.1f}"))
    lines.append(_newcmd("CouplingBestCoupled", fc["per_mode"]["coupled"]["best_algo"].upper()))
    lines.append(_newcmd("CouplingBestOutcome", fc["per_mode"]["outcome"]["best_algo"].upper()))
    lines.append(
        _newcmd(
            "CouplingDQNCoupled",
            f"{fc['per_mode']['coupled']['per_algo']['dqn']['mean_reward']:+0.1f}",
        )
    )
    lines.append(
        _newcmd(
            "CouplingDQNOutcome",
            f"{fc['per_mode']['outcome']['per_algo']['dqn']['mean_reward']:+0.1f}",
        )
    )

    # Test count (sidecar JSON; canonical pytest count).
    if TEST_COUNT.exists():
        num_tests = json.loads(TEST_COUNT.read_text()).get("num_tests", 446)
    else:
        num_tests = 446
    lines.append(_newcmd("NumTests", f"{num_tests:d}"))

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def _render_alpha_table() -> str:
    """Reward vs. observation-aliasing rate, one row per alpha level.

    Columns: alpha & PPO [CI] & DQN & A2C & RF-Acting [CI] & Oracle.
    """
    fa = _load(FALPHA)
    per_alpha = fa["per_alpha"]
    body: list[str] = []
    for akey in ("0.0", "0.2", "0.4", "0.6"):
        c = per_alpha[akey]
        ppo, dqn, a2c = c["ppo"], c["dqn"], c["a2c"]
        rf, orc = c["rf_acting"], c["recommended_action"]
        body.append(
            f"  {akey} & ${ppo['mean']:+.1f}$ $[{ppo['ci_low']:+.1f}, {ppo['ci_high']:+.1f}]$ & "
            f"${dqn['mean']:+.1f}$ & ${a2c['mean']:+.1f}$ & "
            f"${rf['mean']:+.1f}$ $[{rf['ci_low']:+.1f}, {rf['ci_high']:+.1f}]$ & "
            f"${orc['mean']:+.1f}$ \\\\"
        )
    return "\n".join(body)


def _render_coupling_table() -> str:
    """Coupled-vs-outcome reward ablation: best RL vs. RF-Acting per mode."""
    fc = _load(FCOUPLING)
    body: list[str] = []
    for mode in ("coupled", "outcome"):
        m = fc["per_mode"][mode]
        best = m["best_algo"].upper()
        body.append(
            f"  {mode.capitalize()} & {best} & ${m['best_rl_reward']:+.1f}$ & "
            f"${m['rf_acting_reward']:+.1f}$ "
            f"$[{m['rf_acting_ci_low']:+.1f}, {m['rf_acting_ci_high']:+.1f}]$ & "
            f"${m['rf_minus_rl_gap']:+.1f}$ \\\\"
        )
    return "\n".join(body)


def _render_tables() -> str:
    lines = [
        r"% Auto-generated from docs/results/ablation/*.json -- do not hand-edit.",
        r"\newcommand{\AlphaCurveTableBody}{%",
        _render_alpha_table(),
        r"}%",
        r"\newcommand{\CouplingTableBody}{%",
        _render_coupling_table(),
        r"}%",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    out_dir = Path("tex/generated")
    out_dir.mkdir(parents=True, exist_ok=True)

    numbers_tex = _render_numbers()
    tables_tex = _render_tables()

    (out_dir / "numbers.tex").write_text(numbers_tex)
    (out_dir / "tables.tex").write_text(tables_tex)

    print(f"Wrote {out_dir / 'numbers.tex'} ({len(numbers_tex)} chars)")
    print(f"Wrote {out_dir / 'tables.tex'} ({len(tables_tex)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
