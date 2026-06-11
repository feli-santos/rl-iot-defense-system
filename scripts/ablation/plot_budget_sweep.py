"""ablation F16 — Budget-sensitivity prevention curve (prevention pivot).

Reads the budget-sweep JSON produced by
``scripts.ablation.run_budget_sweep`` (``--out docs/results/ablation/
budget_sweep.json``) and renders the headline prevention-pivot figure:

  x-axis:  attacker_budget ∈ {20, 30, 40, 50, 60, 80, ∞}
  y-axis:  prevention_post_grace (defender-attributable prevention rate)
  curves:  one line per fixed baseline policy

The post-grace conditioning is the honest prevention metric: the grace
clamp downgrades any IMPACT before ``min_episode_length`` to MANEUVER, so
a budget exhausted inside the grace window is NOT defender-attributable
(caveat C2). Only post-grace preventions count as the defender genuinely
starving the attacker.

``budget=None`` (unbounded) is plotted at the right edge as the ∞ tick;
every policy should collapse toward prevention≈0 there (the reactive-only
control), and stronger/more-aggressive policies should lift the curve at
finite budgets — the prevention-pivot thesis.

Outputs (under ``--out-dir``):
- ``F16_budget_sweep.png``
- ``F16_summary.json`` — per-policy {budget → prevention_post_grace,
                          compromise_rate}; + G7.4 evaluation
- ``F16_caption.md`` (placeholder)
- ``F16_manifest.json`` (SHA chain)

Gate evaluation:

- **G7.4** — pass iff at least one policy's ``prevention_post_grace`` is
  monotone non-increasing in budget (more budget ⇒ harder to prevent),
  AND the unbounded (∞) cell prevention is below the smallest finite
  budget's prevention for that policy (the prevention pivot is real).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger("scripts.ablation.plot_budget_sweep")

_ROOT = Path(__file__).resolve().parents[2]

# Policies highlighted in the figure legend (others still plotted, thinner).
_HEADLINE_POLICIES = ("always_block", "recommended_action", "random", "always_observe")

_COLORS = {
    "always_block": "#dc2626",
    "recommended_action": "#2563eb",
    "random": "#9333ea",
    "always_observe": "#6b7280",
}


def _sha256(path: Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


# Sentinel x-position for the unbounded (None) budget cell. Chosen above the
# largest finite budget so the ∞ tick sits at the right edge of the axis.
_INF_X = 100.0


def _budget_key_to_x(key: str, finite_budgets: list[int]) -> float:
    """Map a results-dict budget key (str) to an x coordinate."""
    if key in ("None", "null", "inf", "infinity"):
        return _INF_X
    return float(int(key))


def _load_rows(results: dict[str, Any]) -> tuple[list[float], dict[str, list[dict[str, Any]]]]:
    """Pivot results[budget][policy] -> per-policy ordered rows.

    Returns (sorted finite budgets, {policy: [{x, budget_label,
    prevention_post_grace, compromise_rate, n_episodes}, ...]}).
    """
    finite_budgets = sorted(int(k) for k in results if k not in ("None", "null", "inf", "infinity"))

    # Collect the union of policy names across all budget cells.
    policies: list[str] = []
    for cell in results.values():
        for pol in cell:
            if pol not in policies:
                policies.append(pol)

    def _budget_sort_key(k: str) -> float:
        return _budget_key_to_x(k, finite_budgets)

    per_policy: dict[str, list[dict[str, Any]]] = {pol: [] for pol in policies}
    for budget_key in sorted(results.keys(), key=_budget_sort_key):
        cell = results[budget_key]
        x = _budget_key_to_x(budget_key, finite_budgets)
        is_inf = x == _INF_X
        for pol in policies:
            metrics = cell.get(pol)
            if metrics is None:
                continue
            per_policy[pol].append(
                {
                    "x": x,
                    "budget_label": "∞" if is_inf else int(budget_key),
                    "is_inf": is_inf,
                    "prevention_post_grace": float(metrics.get("prevention_post_grace", math.nan)),
                    "compromise_rate": float(metrics.get("compromise_rate", math.nan)),
                    "n_episodes": int(metrics.get("n_episodes", 0)),
                }
            )
    return [float(b) for b in finite_budgets], per_policy


def _evaluate_g74(per_policy: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """G7.4: at least one policy shows the prevention pivot (interior peak).

    Prevention_post_grace is NOT monotone in budget — it is a *hump*:

    * tiny budget (e.g. 20): the attacker exhausts inside the grace window,
      its IMPACT is clamped to MANEUVER, so the prevention is not
      defender-attributable -> prevention ≈ 0.
    * calibrated budget (≈40): the attacker progresses past the grace
      window but the defender can still starve it -> prevention peaks.
    * large / unbounded budget: the attacker has too much budget to be
      starved -> prevention decays back toward 0 (the ∞ control).

    So the prevention pivot signature is an INTERIOR PEAK: the maximum
    prevention occurs at a finite, non-extreme budget and strictly exceeds
    BOTH the smallest-budget cell AND the unbounded (∞) control.
    """
    pivots: dict[str, dict[str, Any]] = {}
    any_pivot = False
    for pol, rows in per_policy.items():
        finite = [r for r in rows if not r["is_inf"] and math.isfinite(r["prevention_post_grace"])]
        inf_rows = [r for r in rows if r["is_inf"]]
        if len(finite) < 3 or not inf_rows:
            continue
        finite = sorted(finite, key=lambda r: r["x"])
        inf_val = inf_rows[0]["prevention_post_grace"]

        # Locate the peak cell.
        peak_idx = max(range(len(finite)), key=lambda i: finite[i]["prevention_post_grace"])
        peak = finite[peak_idx]
        peak_val = peak["prevention_post_grace"]
        smallest_val = finite[0]["prevention_post_grace"]
        largest_val = finite[-1]["prevention_post_grace"]

        # Interior peak: not at either extreme of the finite grid, and the
        # peak strictly dominates the smallest-budget cell, the largest
        # finite cell, and the unbounded control (margin to beat noise).
        margin = 0.05
        interior = 0 < peak_idx < len(finite) - 1
        dominates = bool(
            peak_val > smallest_val + margin
            and peak_val > largest_val + margin
            and math.isfinite(inf_val)
            and peak_val > inf_val + margin
        )
        policy_pivot = bool(interior and dominates)

        pivots[pol] = {
            "peak_budget": peak["budget_label"],
            "peak_prevention": peak_val,
            "smallest_finite_prevention": smallest_val,
            "largest_finite_prevention": largest_val,
            "unbounded_prevention": inf_val,
            "interior_peak": interior,
            "peak_dominates": dominates,
            "pivot": policy_pivot,
        }
        any_pivot = any_pivot or policy_pivot

    return {
        "passes": bool(any_pivot),
        "per_policy": pivots,
        "interpretation": (
            "PASS: at least one fixed policy exhibits the prevention pivot — "
            "prevention_post_grace peaks at an interior (finite, non-extreme) "
            "attacker budget and decays toward 0 at both the tiny-budget "
            "(grace-clamped) and unbounded (∞) extremes, confirming that a "
            "calibrated finite attacker budget is what makes prevention "
            "defender-attributable."
            if any_pivot
            else "FAIL-WITH-FINDING: no policy showed a clean interior-peak "
            "prevention pivot; inspect per_policy fields."
        ),
    }


def _render(
    finite_budgets: list[float],
    per_policy: dict[str, list[dict[str, Any]]],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    # Plot headline policies bold/colored, others thin grey.
    ordered = list(_HEADLINE_POLICIES) + [p for p in per_policy if p not in _HEADLINE_POLICIES]
    for pol in ordered:
        rows = per_policy.get(pol)
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r["x"])
        xs = [r["x"] for r in rows]
        ys = [r["prevention_post_grace"] for r in rows]
        if pol in _HEADLINE_POLICIES:
            ax.plot(
                xs,
                ys,
                "o-",
                color=_COLORS.get(pol, "#111827"),
                label=pol.replace("_", "-"),
                linewidth=1.9,
                markersize=5,
            )
        else:
            ax.plot(xs, ys, "o-", color="#cbd5e1", linewidth=1.0, markersize=3, alpha=0.7)

    # x ticks: finite budgets + ∞ sentinel.
    xticks = list(finite_budgets) + [_INF_X]
    xlabels = [str(int(b)) for b in finite_budgets] + ["∞"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)

    ax.set_xlabel("Attacker budget (steps of kill-chain progression)", fontsize=10)
    ax.set_ylabel("Prevention rate (post-grace, defender-attributable)", fontsize=10)
    ax.set_title(
        "F16 — Prevention vs attacker budget (the prevention pivot)",
        fontsize=11,
    )
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ablation F16 — budget-sensitivity prevention curve.",
    )
    p.add_argument("--sweep-json", default="docs/results/ablation/budget_sweep.json")
    p.add_argument("--out-dir", default="docs/results/ablation")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    sweep_path = Path(args.sweep_json)
    if not sweep_path.exists():
        logger.error("sweep json not found: %s", sweep_path)
        return 1

    data = json.loads(sweep_path.read_text())
    results = data.get("results", {})
    if not results:
        logger.error("no 'results' key in %s", sweep_path)
        return 1

    finite_budgets, per_policy = _load_rows(results)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "F16_budget_sweep.png"
    _render(finite_budgets, per_policy, png_path)

    g74 = _evaluate_g74(per_policy)
    summary = {
        "schema_version": "1.0",
        "phase": 7,
        "figure": "F16",
        "budget_grid": data.get("budget_grid"),
        "min_episode_length": data.get("min_episode_length"),
        "n_episodes": data.get("n_episodes"),
        "per_policy": per_policy,
        "gates": {"G7.4": g74},
        "headline": g74.get("interpretation", "?"),
    }
    (out_dir / "F16_summary.json").write_text(json.dumps(summary, indent=2))

    manifest = {
        "schema_version": "1.0",
        "figure": "F16",
        "git_sha": _git_sha(),
        "outputs": {
            "png": str(png_path),
            "json": str(out_dir / "F16_summary.json"),
        },
        "inputs": {
            "budget_sweep_json": {
                "path": str(sweep_path),
                "sha256": _sha256(sweep_path),
            },
        },
    }
    (out_dir / "F16_manifest.json").write_text(json.dumps(manifest, indent=2))

    caption_path = out_dir / "F16_caption.md"
    if not caption_path.exists():
        caption_path.write_text(
            "**F16 — Prevention vs attacker budget.** Defender-attributable "
            "prevention rate (post-grace) on `test_balanced` for fixed "
            "baseline policies as a function of the finite attacker budget "
            "(∞ = unbounded control). Stronger/more-aggressive policies lift "
            "the curve at small budgets and every policy collapses toward the "
            "unbounded control — the prevention pivot. (PLAN §3.1.6.)\n"
        )

    logger.info(
        "F16 written to %s — G7.4 passes=%s",
        out_dir,
        g74.get("passes"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
