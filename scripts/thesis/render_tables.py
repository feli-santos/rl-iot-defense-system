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
FFIFTEEN = Path("docs/results/ablation/F15_summary.json")
FELEVEN = Path("docs/results/stage-detector/F11_summary.json")
FTUNEDRF = Path("docs/results/stage-detector/tuned_rf_stage_detection.json")
TEST_COUNT = Path("docs/results/test_count.json")

# Spelled-out alpha keys (LaTeX macro names cannot contain digits or dots).
_ALPHA_WORD = {
    "0.0": "Zero",
    "0.2": "Two",
    "0.4": "Four",
    "0.6": "Six",
    "0.8": "Eight",
    "1.0": "One",
}

# Spelled-out algorithm tokens for macro names. LaTeX control sequences must
# be all-letters after the backslash, so "A2C" (which contains the digit 2)
# is emitted as "ATwoC". "PPO"/"DQN" carry no digits and are kept verbatim.
# NB: this mapping is only for MACRO NAMES; rendered table columns use the
# literal algorithm labels ("A2C") via \texttt in the table body.
_MACRO_ALGO = {
    "ppo": "PPO",
    "dqn": "DQN",
    "a2c": "ATwoC",
}

# Human-readable algorithm labels for use in macro *bodies* (typeset in prose),
# where the digit in "A2C" is fine. Distinct from _MACRO_ALGO, which is only for
# building all-letter macro NAMES.
_DISPLAY_ALGO = {
    "ppo": "PPO",
    "dqn": "DQN",
    "a2c": "A2C",
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
    """F10 environment-difficulty sweep: per-algo reward + CI at each p_down point.

    Emits ``\\Ften<Word><Algo>{,,CILow,CIHigh}`` for every swept p_down and every
    fixed defender (PPO/A2C/DQN) so the Section 4.6 prose (results.tex) can cite
    the harshest, an intermediate, and the easiest setting mechanically instead of
    by hand, and can contrast the three algorithms' off-distribution behaviour.
    Also emits the per-algo G7.3 monotonicity findings (``\\FtenGSevenThree<Algo>``).
    """
    if not FTEN.exists():
        return []
    f10 = _load(FTEN)
    lines: list[str] = ["% --- F10 environment-difficulty sweep (PPO/A2C/DQN) ---"]
    # New multi-algo source (algo_rows dict keyed by ppo/a2c/dqn); fall back to
    # the legacy top-level ppo_rows list so older summaries still render PPO.
    algo_rows = f10.get("algo_rows") or {"ppo": f10.get("ppo_rows", [])}
    for algo, macro_algo in _MACRO_ALGO.items():
        rows = algo_rows.get(algo)
        if not rows:
            continue
        rows_by_p = {round(float(r["p"]), 2): r for r in rows}
        for p, word in _PDOWN_WORD.items():
            row = rows_by_p.get(round(p, 2))
            if row is None:
                continue
            lines.append(_newcmd(f"Ften{word}{macro_algo}", f"{row['mean_reward']:+0.1f}"))
            lines.append(_newcmd(f"Ften{word}{macro_algo}CILow", f"{row['ci_low']:+0.1f}"))
            lines.append(_newcmd(f"Ften{word}{macro_algo}CIHigh", f"{row['ci_high']:+0.1f}"))
    # Per-algo G7.3 finding (Yes/No pass of the monotone-difficulty gate).
    per_algo = f10.get("gates", {}).get("G7.3", {}).get("per_algo", {})
    for algo, macro_algo in _MACRO_ALGO.items():
        cell = per_algo.get(algo)
        if not cell:
            continue
        lines.append(_newcmd(f"FtenGSevenThree{macro_algo}", "Yes" if cell.get("passes") else "No"))
    return lines


def _render_evasion_numbers() -> list[str]:
    """F17 evasion-before-commit sweep: per-algo reward + compromise at each evasion.

    Emits ``\\Fseventeen<Word><Algo>{Reward,Compromise}`` per evasion point for
    every fixed defender (PPO/A2C/DQN) plus the headline degradation macros the
    Section 4.6 prose cites (reward at the reference vs. the most evasive attacker,
    and the per-algo lower-CI degradation / robustness-gate finding). The legacy
    ``\\Fseventeen<Word>{Reward,Compromise}`` (PPO, un-suffixed) and
    ``\\FseventeenCILowDegradation`` macros are retained for back-compatibility.
    """
    if not FSEVENTEEN.exists():
        return []
    f17 = _load(FSEVENTEEN)
    lines: list[str] = ["% --- F17 evasion-before-commit sweep (PPO/A2C/DQN) ---"]

    # Legacy PPO-only, un-suffixed macros (kept so existing prose keeps compiling).
    rows_by_e = {round(float(r["evasion_prob"]), 2): r for r in f17.get("rows", [])}
    for e, word in _EVASION_WORD.items():
        row = rows_by_e.get(round(e, 2))
        if row is None:
            continue
        lines.append(_newcmd(f"Fseventeen{word}Reward", f"{row['mean_reward']:+0.1f}"))
        lines.append(_newcmd(f"Fseventeen{word}Compromise", f"{row['compromise_rate']:0.3f}"))

    # Multi-algo, algo-suffixed macros (algo_rows dict keyed by ppo/a2c/dqn).
    algo_rows = f17.get("algo_rows") or {"ppo": f17.get("rows", [])}
    for algo, macro_algo in _MACRO_ALGO.items():
        rows = algo_rows.get(algo)
        if not rows:
            continue
        by_e = {round(float(r["evasion_prob"]), 2): r for r in rows}
        for e, word in _EVASION_WORD.items():
            row = by_e.get(round(e, 2))
            if row is None:
                continue
            lines.append(
                _newcmd(f"Fseventeen{word}{macro_algo}Reward", f"{row['mean_reward']:+0.1f}")
            )
            lines.append(
                _newcmd(
                    f"Fseventeen{word}{macro_algo}Compromise",
                    f"{row['compromise_rate']:0.3f}",
                )
            )

    # Lower-CI degradation + pass finding between the reference (evasion=0) and the
    # most evasive attacker, as reported by the G7.10 robustness gate.
    gate = f17.get("gates", {}).get("G7.10", {})
    deg = gate.get("ci_low_degradation")
    if deg is not None:
        lines.append(_newcmd("FseventeenCILowDegradation", f"{abs(float(deg)):0.1f}"))
    per_algo = gate.get("per_algo", {})
    for algo, macro_algo in _MACRO_ALGO.items():
        cell = per_algo.get(algo)
        if not cell:
            continue
        cell_deg = cell.get("ci_low_degradation")
        if cell_deg is not None:
            lines.append(
                _newcmd(
                    f"Fseventeen{macro_algo}CILowDegradation",
                    f"{abs(float(cell_deg)):0.1f}",
                )
            )
        lines.append(
            _newcmd(
                f"FseventeenGSevenTen{macro_algo}",
                "Yes" if cell.get("passes") else "No",
            )
        )
    return lines


def _render_ood_numbers() -> list[str]:
    """F15 OOD-robustness macros: prevention ranges, advantage, detector-independence stats.

    Every OOD number the thesis prose cites (Sections 1, 3.5, 4.6, conclusion) is
    emitted here so the fragile hand-typed literals become mechanically derived
    from ``F15_summary.json``. LaTeX macro names cannot contain digits, so the
    F15 macros are prefixed ``\\Ffifteen``.

    Emits (all from the canonical F15 summary):
      - ``\\FfifteenPPOPreventionLow/High``   PPO prevention-rate range across the
        ten held-out OOD classes.
      - ``\\FfifteenRFPreventionLow/High``    tuned RF-Acting prevention-rate range.
      - ``\\FfifteenAdvantageLow/High``       best-RL-minus-RF prevention advantage range.
      - ``\\FfifteenSpearmanRho/P``,          detector-independence rank-correlation,
        ``\\FfifteenPearsonR/P``,             linear correlation, and
        ``\\FfifteenOLSSlopeCILow/High``      bootstrap OLS-slope CI (spans zero).
      - per-class exemplars cited in prose: ``\\FfifteenSynFloodRecall/Advantage``
        (high-recall class) and ``\\FfifteenVulnScanRecall/Advantage`` plus
        ``\\FfifteenDNSSpoofingRecall`` (low-recall blind spots).
    """
    if not FFIFTEEN.exists():
        return []
    f15 = _load(FFIFTEEN)
    di = f15.get("detector_independence", {})
    points = di.get("points", [])
    if not points:
        return []
    lines: list[str] = ["% --- F15 OOD-robustness (detector-independence) ---"]

    # Prevention ranges across the ten OOD classes (best-RL vs. tuned RF-Acting).
    # ``best_rl_metric`` is max{dqn, ppo, a2c} prevention per class, so the macros
    # report the *best* windowed RL agent's range, not PPO's alone. Historically
    # this was PPO (the only trained on-policy agent worth quoting); after the
    # A2C n_steps fix the best-RL agent at the headline regime is A2C. The legacy
    # ``FfifteenPPOPreventionLow/High`` macro names are kept alongside the new
    # ``FfifteenBestRLPreventionLow/High`` so existing prose continues to compile.
    rl_prev = [p["best_rl_metric"] for p in points]
    rf_prev = [p["rf_metric"] for p in points]
    adv = [p["advantage"] for p in points]
    lines.append(_newcmd("FfifteenBestRLPreventionLow", f"{min(rl_prev):0.2f}"))
    lines.append(_newcmd("FfifteenBestRLPreventionHigh", f"{max(rl_prev):0.2f}"))
    lines.append(_newcmd("FfifteenPPOPreventionLow", f"{min(rl_prev):0.2f}"))
    lines.append(_newcmd("FfifteenPPOPreventionHigh", f"{max(rl_prev):0.2f}"))
    lines.append(_newcmd("FfifteenRFPreventionLow", f"{min(rf_prev):0.2f}"))
    lines.append(_newcmd("FfifteenRFPreventionHigh", f"{max(rf_prev):0.2f}"))
    lines.append(_newcmd("FfifteenAdvantageLow", f"{min(adv):+0.2f}"))
    lines.append(_newcmd("FfifteenAdvantageHigh", f"{max(adv):+0.2f}"))

    # Detector-independence statistics (Spearman non-significant, OLS CI spans zero).
    stats = di.get("stats", {})
    if stats:
        lines.append(_newcmd("FfifteenSpearmanRho", f"{stats['spearman_rho']:0.2f}"))
        lines.append(_newcmd("FfifteenSpearmanP", f"{stats['spearman_p']:0.2f}"))
        lines.append(_newcmd("FfifteenPearsonR", f"{stats['pearson_r']:0.2f}"))
        lines.append(_newcmd("FfifteenPearsonP", f"{stats['pearson_p']:0.2f}"))
        lines.append(_newcmd("FfifteenOLSSlopeCILow", f"{stats['ols_slope_ci_low']:+0.2f}"))
        lines.append(_newcmd("FfifteenOLSSlopeCIHigh", f"{stats['ols_slope_ci_high']:+0.2f}"))

    # Per-class exemplars cited in prose (high-recall vs. blind-spot classes).
    by_class = {p["ood_class"]: p for p in points}
    syn = by_class.get("DoS-SYN_Flood")
    if syn is not None:
        lines.append(_newcmd("FfifteenSynFloodRecall", f"{syn['rf_recall']:0.3f}"))
        lines.append(_newcmd("FfifteenSynFloodAdvantage", f"{syn['advantage']:+0.2f}"))
    vuln = by_class.get("VulnerabilityScan")
    if vuln is not None:
        lines.append(_newcmd("FfifteenVulnScanRecall", f"{vuln['rf_recall']:0.3f}"))
        lines.append(_newcmd("FfifteenVulnScanAdvantage", f"{vuln['advantage']:+0.2f}"))
    dns = by_class.get("DNS_Spoofing")
    if dns is not None:
        lines.append(_newcmd("FfifteenDNSSpoofingRecall", f"{dns['rf_recall']:0.3f}"))
    return lines


def _render_detector_numbers() -> list[str]:
    """F11 stage-detector macro-F1 macros, split-tagged.

    Emits the tuned-RandomForest (RF-Acting baseline) macro-F1 on the
    ``test_balanced`` split, sourced from ``tuned_rf_stage_detection.json`` so the
    detector number quoted in prose (Section 3.5 and Section 4.1) is mechanically
    the same tuned model the RF-Acting policy deploys, not the earlier untuned
    reference RF. The dropped supervised-MLP detector is no longer emitted.
    """
    if not FTUNEDRF.exists():
        return []
    trf = _load(FTUNEDRF)
    lines: list[str] = ["% --- tuned RF-Acting stage-detector macro-F1 ---"]
    lines.append(_newcmd("DetectorRfFoneBalanced", f"{trf['macro_f1']:0.3f}"))
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

    # Seed count (PPO is the pinned headline defender for F10/F17; A2C carries
    # the highest outcome-reward policy at the headline aliasing rate).
    n_seeds = per_alpha["0.0"]["ppo"].get("n_seeds", 10)
    lines.append(_newcmd("NumSeeds", f"{n_seeds:d}"))

    # Per-alpha reward macros for PPO (headline), A2C, DQN, tuned RF-Acting, and
    # the oracle ceiling, plus the PPO-minus-RF crossover gap and its significance
    # verdict. A2C/DQN means and CIs are emitted alongside PPO so the prose can
    # cite all three learned agents' headline numbers mechanically from the JSON.
    for akey, word in _ALPHA_WORD.items():
        cell = per_alpha[akey]
        ppo = cell["ppo"]
        dqn = cell.get("dqn", {})
        a2c = cell.get("a2c", {})
        rf = cell["rf_acting"]
        orc = cell["recommended_action"]
        lines.append(_newcmd(f"Alpha{word}PPO", f"{ppo['mean']:+0.1f}"))
        lines.append(_newcmd(f"Alpha{word}PPOCILow", f"{ppo['ci_low']:+0.1f}"))
        lines.append(_newcmd(f"Alpha{word}PPOCIHigh", f"{ppo['ci_high']:+0.1f}"))
        if dqn:
            lines.append(_newcmd(f"Alpha{word}DQN", f"{dqn['mean']:+0.1f}"))
            lines.append(_newcmd(f"Alpha{word}DQNCILow", f"{dqn['ci_low']:+0.1f}"))
            lines.append(_newcmd(f"Alpha{word}DQNCIHigh", f"{dqn['ci_high']:+0.1f}"))
        if a2c:
            macro_a2c = _MACRO_ALGO["a2c"]
            lines.append(_newcmd(f"Alpha{word}{macro_a2c}", f"{a2c['mean']:+0.1f}"))
            lines.append(_newcmd(f"Alpha{word}{macro_a2c}CILow", f"{a2c['ci_low']:+0.1f}"))
            lines.append(_newcmd(f"Alpha{word}{macro_a2c}CIHigh", f"{a2c['ci_high']:+0.1f}"))
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
    lines.append(
        _newcmd("CouplingBestCoupled", _DISPLAY_ALGO[fc["per_mode"]["coupled"]["best_algo"]])
    )
    lines.append(
        _newcmd("CouplingBestOutcome", _DISPLAY_ALGO[fc["per_mode"]["outcome"]["best_algo"]])
    )
    # Emit every (mode, algo) reward so prose can cite each learned agent's
    # performance under both reward contracts, not only the historical DQN
    # coupled / DQN and PPO outcome cells.
    for mode in ("coupled", "outcome"):
        m = fc["per_mode"][mode]
        for algo, cell in m["per_algo"].items():
            macro_algo = _MACRO_ALGO[algo]
            lines.append(
                _newcmd(
                    f"Coupling{macro_algo}{mode.capitalize()}",
                    f"{cell['mean_reward']:+0.1f}",
                )
            )

    # Environment-difficulty (F10) and evasion (F17) sweep macros, emitted only
    # when their canonical summaries exist (regenerated after the sweeps re-run).
    lines.extend(_render_aggressiveness_numbers())
    lines.extend(_render_evasion_numbers())
    lines.extend(_render_ood_numbers())
    lines.extend(_render_detector_numbers())

    # Test count (sidecar JSON; canonical pytest count).
    if TEST_COUNT.exists():
        num_tests = json.loads(TEST_COUNT.read_text()).get("num_tests", 455)
    else:
        num_tests = 455
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
    for akey in ("0.0", "0.2", "0.4", "0.6", "0.8", "1.0"):
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
