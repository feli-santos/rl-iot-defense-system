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
FTEN = Path("docs/results/ablation/F10_summary.json")
FSEVENTEEN = Path("docs/results/ablation/F17_summary.json")
FELEVEN = Path("docs/results/stage-detector/F11_summary.json")
TEST_COUNT = Path("docs/results/test_count.json")

# Spelled-out alpha keys (LaTeX macro names cannot contain digits or dots).
_ALPHA_WORD = {
    "0.0": "Zero",
    "0.2": "Two",
    "0.4": "Four",
    "0.6": "Six",
}

# Spelled-out p_down sweep points (LaTeX macro names cannot contain digits/dots).
_PDOWN_WORD = {
    0.0: "Zero",
    0.2: "Two",
    0.4: "Four",
    0.6: "Six",
    0.8: "Eight",
    1.0: "One",
}

# Spelled-out evasion sweep points.
_EVASION_WORD = {
    0.0: "Zero",
    0.25: "TwentyFive",
    0.5: "Fifty",
    0.75: "SeventyFive",
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


def _render_aggressiveness_numbers() -> list[str]:
    """F10 environment-difficulty sweep: PPO reward + CI at each p_down point.

    Emits ``\\Ften<Word>{PPO,PPOCILow,PPOCIHigh}`` for every swept p_down so the
    Section 4.6 prose (results.tex) can cite the harshest, an intermediate, and
    the easiest setting mechanically instead of by hand.
    """
    if not FTEN.exists():
        return []
    f10 = _load(FTEN)
    lines: list[str] = ["% --- F10 environment-difficulty sweep (PPO) ---"]
    ppo_by_p = {round(float(r["p"]), 2): r for r in f10.get("ppo_rows", [])}
    for p, word in _PDOWN_WORD.items():
        row = ppo_by_p.get(round(p, 2))
        if row is None:
            continue
        lines.append(_newcmd(f"Ften{word}PPO", f"{row['mean_reward']:+0.1f}"))
        lines.append(_newcmd(f"Ften{word}PPOCILow", f"{row['ci_low']:+0.1f}"))
        lines.append(_newcmd(f"Ften{word}PPOCIHigh", f"{row['ci_high']:+0.1f}"))
    return lines


def _render_evasion_numbers() -> list[str]:
    """F17 evasion-before-commit sweep: PPO reward + compromise at each evasion.

    Emits ``\\Fseventeen<Word>{Reward,Compromise}`` per evasion point plus the
    headline degradation macros the Section 4.6 prose cites (reward at the
    reference vs. the most evasive attacker, and the lower-CI degradation).
    """
    if not FSEVENTEEN.exists():
        return []
    f17 = _load(FSEVENTEEN)
    lines: list[str] = ["% --- F17 evasion-before-commit sweep (PPO) ---"]
    rows_by_e = {round(float(r["evasion_prob"]), 2): r for r in f17.get("rows", [])}
    for e, word in _EVASION_WORD.items():
        row = rows_by_e.get(round(e, 2))
        if row is None:
            continue
        lines.append(_newcmd(f"Fseventeen{word}Reward", f"{row['mean_reward']:+0.1f}"))
        lines.append(_newcmd(f"Fseventeen{word}Compromise", f"{row['compromise_rate']:0.3f}"))
    # Lower-CI degradation between the reference (evasion=0) and the most evasive
    # attacker, as reported by the G7.10 robustness gate.
    gate = f17.get("gates", {}).get("G7.10", {})
    deg = gate.get("ci_low_degradation")
    if deg is not None:
        lines.append(_newcmd("FseventeenCILowDegradation", f"{abs(float(deg)):0.1f}"))
    return lines


def _render_detector_numbers() -> list[str]:
    """F11 stage-detector macro-F1 macros, split-tagged.

    Emits the production MLP and tuned-RandomForest macro-F1 on both the
    ``test_balanced`` split (the one Section 4.3 prose and Figure 4.3 report) and
    the full ``test`` split, so the detector numbers are mechanically derived
    from ``F11_summary.json`` instead of hand-typed. The split tag is part of the
    macro name to make split-mix-ups (e.g.\\ quoting the full-test 0.925 as a
    ``test_balanced`` number) impossible.
    """
    if not FELEVEN.exists():
        return []
    f11 = _load(FELEVEN)
    models = f11["models"]
    lines: list[str] = ["% --- F11 stage-detector macro-F1 (split-tagged) ---"]
    for macro, model, split in (
        ("DetectorMlpFoneBalanced", "StageDetector", "test_balanced"),
        ("DetectorMlpFoneFull", "StageDetector", "test"),
        ("DetectorRfFoneBalanced", "RandomForest", "test_balanced"),
        ("DetectorRfFoneFull", "RandomForest", "test"),
    ):
        f1 = models[model][split]["macro_f1"]
        lines.append(_newcmd(macro, f"{f1:0.3f}"))
    return lines


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

    # Environment-difficulty (F10) and evasion (F17) sweep macros, emitted only
    # when their canonical summaries exist (regenerated after the sweeps re-run).
    lines.extend(_render_aggressiveness_numbers())
    lines.extend(_render_evasion_numbers())
    lines.extend(_render_detector_numbers())

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
