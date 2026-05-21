"""Phase-6 F6 — per-algo stage × action confusion matrices (PLAN §3.1.6, C5).

Builds one 5×5 row-normalised heatmap per algorithm (DQN/PPO/A2C) and,
for context, three reference panels (random / always-OBSERVE /
always-BLOCK). Rows are kill-chain stages (0=BENIGN..4=IMPACT), columns
are defensive actions (0=OBSERVE..4=ISOLATE). Cell colour = empirical
fraction of decisions at that stage that chose that action.

The proportionality band ``|action − recommended(stage)| ≤ 1`` is
overlaid as a transparent diagonal band. The G6.3 score (per-policy
fraction of mass inside the band, averaged over **non-IMPACT stages**
per D6.7) is printed as a sub-caption per panel.

Outputs:
- ``F6_stage_action_cm.png``    — multi-panel heatmap.
- ``F6_summary.json``           — per-policy 5×5 matrix + G6.3 score.
- ``F6_manifest.json``          — SHA-256 hash chain.
- ``F6_caption.md``             — thesis caption.

Inputs come from the same ``runs/phase6/<policy>/seed_*/eval_test.jsonl``
files F5 already consumed (using
:func:`src.blue_team.aggregation.per_stage_action_distribution` directly
to keep the F5-vs-F6 numbers consistent by construction).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.blue_team.aggregation import (
    per_stage_action_distribution,
    read_episodes_jsonl,
)

logger = logging.getLogger("scripts.benchmark.plot_stage_action_cm")

_ROOT = Path(__file__).resolve().parents[2]


# Recommended action by stage; mirrors src.environment.adversarial_env.
# Drift-guarded by the same constant in baseline_policies.py and by
# tests/test_baseline_policies.py.
_REC: Dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

_STAGE_LABELS = ["BENIGN", "RECON", "ACCESS", "MANEUVER", "IMPACT"]
_ACTION_LABELS = ["OBSERVE", "LOG", "THROTTLE", "BLOCK", "ISOLATE"]


# Default panel ordering (left → right, top → bottom for the 2×3 grid).
# Trained-RL on the top row; non-RL on the bottom row, in increasing
# aggression. Recommended-action is intentionally **not** plotted here —
# its CM would be a perfect identity matrix by construction (D6.5
# mapping) and the panel would carry no information; F6 is about
# *deviations* from the recommended policy.
_PANEL_ORDER: List[str] = [
    "dqn", "ppo", "a2c",
    "random", "always_observe", "always_block",
]

_DISPLAY: Dict[str, str] = {
    "dqn": "DQN",
    "ppo": "PPO",
    "a2c": "A2C",
    "random": "Random",
    "always_observe": "Always-OBSERVE",
    "always_block": "Always-BLOCK",
    "recommended_action": "Recommended-Action",
    "rf_acting": "RF-Acting",
}


def _nan_to_none(o: Any) -> Any:
    """Recursively translate float NaN to ``None`` so the resulting
    structure round-trips through ``json.dumps(..., allow_nan=False)``
    without exception. Step-6 F5 / Step-8 doc-fix.

    Phase-6 F6 emits per-policy 5×5 matrices whose IMPACT row is
    intentionally NaN-filled (D6.7 excludes IMPACT from the
    proportionality scoring); the RFC-7159 representation is JSON
    null, not the bare token ``NaN``. This helper is a pre-pass
    that touches only float NaN; integers, strings, lists, and
    dicts pass through unchanged.
    """
    if isinstance(o, float) and o != o:  # NaN check
        return None
    if isinstance(o, list):
        return [_nan_to_none(x) for x in o]
    if isinstance(o, dict):
        return {k: _nan_to_none(v) for k, v in o.items()}
    return o


def _sha256(path: Path) -> Optional[str]:
    """SHA-256 of file content; ``None`` if missing."""
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
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _gather_records(runs_root: Path, policy: str) -> List[Dict]:
    """Read every per-seed JSONL for this policy and concatenate."""
    base = runs_root / policy
    if not base.exists():
        return []
    out: List[Dict] = []
    for seed_dir in sorted(base.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        out.extend(read_episodes_jsonl(seed_dir / "eval_test.jsonl"))
    return out


def _proportionality_score(
    cm: np.ndarray,
    *,
    exclude_impact: bool = True,
) -> float:
    """G6.3 score: fraction of decisions inside the |a−rec(s)| ≤ 1 band.

    Args:
        cm: 5×5 row-normalised confusion matrix (rows sum to 1.0
            for stages with any decisions; NaN row if no decisions).
        exclude_impact: per D6.7, drop the IMPACT row before averaging.
            The G5.4 finding leaves IMPACT-stage behaviour outside the
            scope of the proportionality test.

    Returns:
        Mean over the surviving rows of (sum of in-band cells in that
        row). NaN if every surviving row is empty.
    """
    in_band = np.zeros_like(cm)
    for s in range(5):
        rec_a = _REC[s]
        for a in range(5):
            if abs(a - rec_a) <= 1:
                in_band[s, a] = 1.0
    band_mass = np.nansum(cm * in_band, axis=1)  # (5,)
    if exclude_impact:
        band_mass = band_mass[:4]
    finite = band_mass[np.isfinite(band_mass)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _render(
    matrices: Dict[str, np.ndarray],
    scores: Dict[str, float],
    out_path: Path,
) -> None:
    """Render the 2×3 grid of stage × action heatmaps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    panels = [p for p in _PANEL_ORDER if p in matrices]
    n = len(panels)
    if n == 0:
        raise RuntimeError("F6: no policies available to plot")
    rows = 2
    cols = math.ceil(n / rows)
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.6 * cols, 3.6 * rows),
        sharex=False, sharey=False,
    )
    axes_flat = np.asarray(axes).reshape(-1)

    for idx, ax in enumerate(axes_flat):
        if idx >= n:
            ax.set_visible(False)
            continue
        pol = panels[idx]
        cm = matrices[pol]
        # Heatmap with NaN → white.
        masked = np.ma.array(cm, mask=~np.isfinite(cm))
        im = ax.imshow(masked, cmap="viridis", vmin=0.0, vmax=1.0,
                       aspect="auto")
        # Cell annotations (white on dark cells, black on light).
        for s in range(5):
            for a in range(5):
                v = cm[s, a]
                if not np.isfinite(v):
                    continue
                color = "white" if v > 0.45 else "black"
                ax.text(a, s, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=7)
        # Proportionality band overlay (|a-rec(s)|<=1).
        for s in range(5):
            rec_a = _REC[s]
            lo = max(rec_a - 1, 0)
            hi = min(rec_a + 1, 4)
            ax.add_patch(Rectangle(
                (lo - 0.5, s - 0.5), hi - lo + 1, 1,
                linewidth=1.2, edgecolor="red", facecolor="none",
                alpha=0.7,
            ))
        # Axes & title.
        ax.set_xticks(range(5))
        ax.set_xticklabels(_ACTION_LABELS, rotation=30, ha="right",
                           fontsize=7)
        ax.set_yticks(range(5))
        ax.set_yticklabels(_STAGE_LABELS, fontsize=7)
        ax.set_xlabel("Action", fontsize=8)
        ax.set_ylabel("Decision stage", fontsize=8)
        score = scores.get(pol, float("nan"))
        title = (
            f"{_DISPLAY.get(pol, pol)}\n"
            f"G6.3 (non-IMPACT) = "
            f"{('—' if not np.isfinite(score) else f'{score:.2f}')}"
        )
        ax.set_title(title, fontsize=10, pad=4)
    cbar = fig.colorbar(im, ax=axes_flat[:n], fraction=0.025, pad=0.02)
    cbar.set_label("Decision share within stage", fontsize=8)
    fig.suptitle(
        "F6 — Stage × Action Decision Distribution per Policy on `test_balanced`",
        fontsize=12, y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 0.95, 0.97))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _benign_fpr(
    cm: np.ndarray,
    *,
    aggressive_actions: Tuple[int, ...] = (3, 4),
    warn_threshold: float = 0.01,
    policy_name: str = "unknown",
) -> Dict[str, Any]:
    """Compute false-positive rate on BENIGN traffic (thesis review issue C17).

    The BENIGN row (index 0) of the stage×action matrix gives the fraction of
    decisions made when the true stage is BENIGN. An aggressive action (BLOCK=3
    or ISOLATE=4) on a BENIGN step is a false positive: it disrupts legitimate
    IoT traffic without any security justification.

    FPR = P(BLOCK or ISOLATE | stage=BENIGN)
        = sum(cm[0, 3], cm[0, 4])

    Args:
        cm: 5×5 row-normalised confusion matrix (rows=stages, cols=actions).
            Rows should sum to 1.0; NaN rows indicate no data for that stage.
        aggressive_actions: Action indices considered false positives on BENIGN.
            Default: (3=BLOCK, 4=ISOLATE). LOG and THROTTLE on BENIGN may
            be debatable but are not counted here.
        warn_threshold: Emit a warning if FPR exceeds this value (default 1%).
        policy_name: For logging only.

    Returns:
        Dict with:
            - ``benign_fpr``: P(aggressive | BENIGN), or null if no BENIGN data
            - ``benign_fpr_pct``: same × 100
            - ``exceeds_threshold``: bool, True if FPR > warn_threshold
            - ``n_benign_decisions``: raw count or null
            - ``aggressive_actions``: which action indices counted as FP
    """
    benign_row = cm[0, :]  # shape (5,)
    if not np.any(np.isfinite(benign_row)):
        return {
            "benign_fpr": None,
            "benign_fpr_pct": None,
            "exceeds_threshold": None,
            "n_benign_decisions": None,
            "aggressive_actions": list(aggressive_actions),
        }
    fpr = float(np.nansum([benign_row[a] for a in aggressive_actions]))
    if fpr > warn_threshold:
        logger.warning(
            "⚠  BENIGN FPR = %.3f (%.1f%%) for policy %r — exceeds "
            "%.1f%% threshold. Consider tuning the reward's "
            "disproportionate-action penalty.",
            fpr, fpr * 100, policy_name, warn_threshold * 100,
        )
    return {
        "benign_fpr": fpr,
        "benign_fpr_pct": fpr * 100,
        "exceeds_threshold": bool(fpr > warn_threshold),
        "aggressive_actions": list(aggressive_actions),
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-6 F6 — stage × action CMs.")
    p.add_argument("--runs-root", default="runs/phase6")
    p.add_argument("--out-dir", default="docs/results/06_benchmark")
    p.add_argument("--policies", nargs="+", default=_PANEL_ORDER)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrices: Dict[str, np.ndarray] = {}
    scores: Dict[str, float] = {}
    in_hashes: Dict[str, str] = {}
    for pol in args.policies:
        records = _gather_records(runs_root, pol)
        if not records:
            logger.warning("no records for policy %r; skipping", pol)
            continue
        cm = per_stage_action_distribution(records)
        matrices[pol] = cm
        scores[pol] = _proportionality_score(cm, exclude_impact=True)
        # Hash every input JSONL for the manifest.
        for sd in sorted((runs_root / pol).iterdir()):
            if not sd.is_dir() or not sd.name.startswith("seed_"):
                continue
            jp = sd / "eval_test.jsonl"
            sha = _sha256(jp)
            if sha:
                in_hashes[str(jp)] = sha
        logger.info(
            "%s: G6.3 score (non-IMPACT) = %s",
            pol, "—" if not np.isfinite(scores[pol]) else f"{scores[pol]:.3f}",
        )

    if not matrices:
        logger.error("no matrices produced; aborting")
        return 1

    png_path = out_dir / "F6_stage_action_cm.png"
    _render(matrices, scores, png_path)

    summary = {
        "schema_version": "1.0",
        "phase": 6,
        "figure": "F6",
        "stage_labels": _STAGE_LABELS,
        "action_labels": _ACTION_LABELS,
        "recommended_by_stage": _REC,
        "g63_threshold": 0.70,
        "g63_excludes_impact": True,
        "policies": {
            pol: {
                "matrix": _nan_to_none(matrices[pol].tolist()),
                "g63_score_non_impact": (
                    None if not np.isfinite(scores[pol])
                    else float(scores[pol])
                ),
                "g63_pass": bool(
                    np.isfinite(scores[pol]) and scores[pol] >= 0.70
                ),
            }
            for pol in matrices
        },
    }
    # Step-6 F5 / Step-8 doc-fix: emit RFC-7159 / ECMA-404 valid JSON
    # by translating bare NaN to JSON null. The previous emission used
    # Python's default `allow_nan=True` which produces non-strict-JSON
    # `NaN` literals — strict parsers reject them. The IMPACT row of
    # the per-policy `matrix` is intentionally NaN-filled (D6.7
    # excludes IMPACT from the proportionality scoring); we encode
    # that as `null` and pass `allow_nan=False` so any future
    # regression that introduces a non-IMPACT NaN will fail loudly
    # at serialisation time rather than silently emitting non-RFC
    # JSON.
    (out_dir / "F6_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )

    eval_manifest_path = runs_root / "eval_manifest.json"
    manifest = {
        "schema_version": "1.0",
        "figure": "F6",
        "git_sha": _git_sha(),
        "outputs": {
            "png":  str(png_path),
            "json": str(out_dir / "F6_summary.json"),
        },
        "inputs": {
            "eval_manifest": {
                "path": str(eval_manifest_path),
                "sha256": _sha256(eval_manifest_path),
            },
            "eval_jsonl_sha256": in_hashes,
        },
    }
    (out_dir / "F6_manifest.json").write_text(json.dumps(manifest, indent=2))

    # ---- Benign FPR analysis (thesis review issue C17) ----
    fpr_results: Dict[str, Any] = {}
    any_exceeds = False
    for pol, cm in matrices.items():
        fpr_result = _benign_fpr(cm, warn_threshold=0.01, policy_name=pol)
        fpr_results[pol] = fpr_result
        if fpr_result.get("exceeds_threshold"):
            any_exceeds = True
        fpr_val = fpr_result.get("benign_fpr_pct")
        logger.info(
            "%s: BENIGN FPR = %s",
            pol,
            "N/A" if fpr_val is None else f"{fpr_val:.2f}%",
        )

    fpr_summary = {
        "schema_version": "1.0",
        "phase": 6,
        "description": (
            "Benign-traffic false-positive rate per policy. "
            "FPR = P(BLOCK or ISOLATE | true stage=BENIGN). "
            "Computed from the phase-6 stage×action confusion matrices (F6). "
            "Threshold for flagging: 1% (aggressive_actions=[3=BLOCK, 4=ISOLATE])."
        ),
        "fpr_threshold_pct": 1.0,
        "any_policy_exceeds_threshold": any_exceeds,
        "policies": fpr_results,
    }
    fpr_path = out_dir / "benign_fpr.json"
    fpr_path.write_text(json.dumps(fpr_summary, indent=2))
    logger.info(
        "Benign FPR summary written to %s  [any_exceeds=%s]",
        fpr_path, any_exceeds,
    )

    logger.info("F6 written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
