#!/usr/bin/env python3
"""JSON → LaTeX fragment generator (anti-drift tooling).

Reads canonical summary JSONs under ``docs/results/`` and emits:
  - ``tex/generated/numbers.tex``   : ``\newcommand`` macros for headline numbers
  - ``tex/generated/tables.tex``    : ``\input{}``-able table bodies

Intended to be re-run after every data regeneration so prose numbers are
always mechanically derived from the canonical JSON source of truth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (canonical data)
# ---------------------------------------------------------------------------

F5 = Path("docs/results/06_benchmark/F5_summary.json")
F7 = Path("docs/results/06_benchmark/F7_summary.json")
F9 = Path("docs/results/07_ablation/F9_summary.json")
G6 = Path("docs/results/06_benchmark/G6_scoreboard.json")
BENIGN_FPR = Path("docs/results/06_benchmark/benign_fpr.json")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Canonical JSON not found: {path}")
    return json.loads(path.read_text())


def _find_row(rows: list[dict], policy: str) -> dict:
    for r in rows:
        if r["policy"] == policy:
            return r
    raise KeyError(f"Policy '{policy}' not found in benchmark summary")


def _best_deployable_rl(rows: list[dict]) -> dict:
    exclude = {
        "recommended_action",
        "rf_acting",
        "random",
        "always_block",
        "always_observe",
    }
    candidates = [r for r in rows if r["policy"] not in exclude]
    if not candidates:
        raise RuntimeError("No deployable RL agents found in F5 summary")
    return max(candidates, key=lambda r: r["mean_reward"])


# ---------------------------------------------------------------------------
# Macro rendering
# ---------------------------------------------------------------------------


def _render_numbers() -> str:
    f5 = _load(F5)
    rows = f5["rows"]
    best = _best_deployable_rl(rows)
    oracle = _find_row(rows, "recommended_action")
    rf = _find_row(rows, "rf_acting")

    capture_pct = best["mean_reward"] / oracle["mean_reward"] * 100
    latency_ratio = rf["p50_inference_latency_ms"] / best["p50_inference_latency_ms"]

    lines = [
        f"% Auto-generated from {F5} — do not hand-edit",
        r"\newcommand{\BestAgentName}{" + best["policy"].upper() + "}",
        r"\newcommand{\BestAgentReward}{%+0.1f}" % best["mean_reward"],
        r"\newcommand{\BestAgentCILow}{%+0.1f}" % best["mean_reward_ci_low"],
        r"\newcommand{\BestAgentCIHigh}{%+0.1f}" % best["mean_reward_ci_high"],
        r"\newcommand{\OracleCeiling}{%+0.1f}" % oracle["mean_reward"],
        r"\newcommand{\OracleCILow}{%+0.1f}" % oracle["mean_reward_ci_low"],
        r"\newcommand{\OracleCIHigh}{%+0.1f}" % oracle["mean_reward_ci_high"],
        r"\newcommand{\OracleCapturePct}{%0.1f}" % capture_pct,
        r"\newcommand{\LatencyRatio}{%0.1f}" % latency_ratio,
        r"\newcommand{\RFReward}{%+0.1f}" % rf["mean_reward"],
        r"\newcommand{\RFLatency}{%0.3f}" % rf["p50_inference_latency_ms"],
        r"\newcommand{\BestAgentLatency}{%0.3f}" % best["p50_inference_latency_ms"],
    ]

    # Seed count (read from n_seeds of the best agent)
    lines.append(
        r"\newcommand{\NumSeeds}{%d}" % best.get("n_seeds", 5)
    )

    # FPR numbers
    if BENIGN_FPR.exists():
        fpr = _load(BENIGN_FPR)
        for policy, val in fpr.items():
            safe = policy.replace("-", "")
            lines.append(
                r"\newcommand{\FPR" + safe + "}{%0.3f}" % val
            )

    # F9 structural fix
    if F9.exists():
        f9 = _load(F9)
        # F9_summary.json has a list of cells; find impact_is_terminal=False
        structural = None
        for cell in f9.get("cells", []):
            if cell.get("impact_is_terminal") is False:
                structural = cell
                break
        if structural is None:
            # Fallback: look for a cell dict keyed by name
            structural = f9.get("impact_is_terminal_false")
        if structural:
            lines.append(
                r"\newcommand{\F9StructuralReward}{%+0.1f}" % structural["mean_reward"]
            )
            lines.append(
                r"\newcommand{\F9StructuralMitRate}{%0.3f}" % structural.get("mitigated_impact_rate", 0.0)
            )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def _fmt_reward(r: float) -> str:
    return f"${r:+.1f}$"


def _fmt_ci(low: float, high: float) -> str:
    return f"$[{low:+.1f}, {high:+.1f}]$"


def _render_benchmark_table() -> str:
    f5 = _load(F5)
    rows = f5["rows"]
    # Order: oracle, RF, best RL, other RL, random, always-block, always-observe
    order = [
        "recommended_action",
        "rf_acting",
        "dqn",
        "ppo",
        "a2c",
        "random",
        "always_block",
        "always_observe",
    ]
    body: list[str] = []
    for policy in order:
        try:
            r = _find_row(rows, policy)
        except KeyError:
            continue
        name = {
            "recommended_action": "Recommended-Action (oracle)",
            "rf_acting": "RF-Acting",
            "dqn": "DQN",
            "ppo": "PPO",
            "a2c": "A2C",
            "random": "Random",
            "always_block": "Always-BLOCK",
            "always_observe": "Always-OBSERVE",
        }.get(policy, policy)
        body.append(
            f"  {name} & {_fmt_reward(r['mean_reward'])} & "
            f"{_fmt_ci(r['mean_reward_ci_low'], r['mean_reward_ci_high'])} & "
            f"{r['n_episodes']} & {r['mitigated_impact_rate']:.3f} \\\\"
        )
    return "\n".join(body)


def _render_latency_table() -> str:
    f5 = _load(F5)
    rows = f5["rows"]
    body: list[str] = []
    for r in rows:
        name = {
            "recommended_action": "Oracle (ref.)",
            "rf_acting": "RF-Acting",
            "dqn": "DQN",
            "ppo": "PPO",
            "a2c": "A2C",
            "random": "Random",
            "always_block": "Always-BLOCK",
            "always_observe": "Always-OBSERVE",
        }.get(r["policy"], r["policy"])
        body.append(
            f"  {name} & {_fmt_reward(r['mean_reward'])} & "
            f"{r['p50_inference_latency_ms']:.3f} & "
            f"{r['p95_inference_latency_ms']:.3f} \\\\"
        )
    return "\n".join(body)


def _render_tables() -> str:
    lines = [
        r"% Auto-generated from docs/results/**/*.json — do not hand-edit",
        r"\newcommand{\BenchmarkTableBody}{%",
        _render_benchmark_table(),
        r"}%",
        r"\newcommand{\LatencyTableBody}{%",
        _render_latency_table(),
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
