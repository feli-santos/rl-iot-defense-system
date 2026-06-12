"""Measured RF-Acting inference-latency study (F7b / D6.8 deep-dive).

Motivation
----------
The Held-Out Benchmark reports an RF-Acting per-step inference latency of
~16.5 ms p50, against ~0.06--0.09 ms for the trained RL agents (a ~175x
gap). A reviewer will rightly ask: *is that 16.5 ms a fundamental cost of
a supervised-classifier-plus-rules defender, or an artifact of an
unoptimised 100-tree, unbounded-depth scikit-learn forest queried one
sample at a time?* The original prose answered this with a line-rate
plausibility argument; this study replaces that with **measurement**.

What it measures
----------------
Using the same ``measure_inference_latency`` harness that produced the F7
numbers, on a pool of **real** CICIoT2023 ``test_balanced`` feature rows,
we time:

1. **Production RF** (100 trees, ``max_depth=None``) per-sample
   ``predict`` wrapped exactly as :class:`RFActingPolicy` queries it.
   This reproduces the F5/F7 16.5 ms baseline end-to-end.

2. **scikit-learn-native optimisation levers** (no new dependencies, so
   the pinned reproduction environment is untouched), each retrained on
   the same training split and re-measured:
     - ``n_estimators`` sweep: 100 -> 50 -> 25 -> 10 (inference is
       linear in tree count);
     - ``max_depth`` cap: None -> 12 -> 8 (shallower trees, fewer node
       visits per sample);
     - a single ``DecisionTreeClassifier`` (the depth-bounded lower
       bound of the family).

   Each variant also reports its **macro-F1 on test_balanced** so the
   latency saving is paired with its accuracy cost (an optimisation that
   destroys detection quality is not a real win).

3. **Batched vs per-sample** throughput for the production RF: the F5
   contract is strictly per-step (one obs at a time), but a deployment
   that buffers flows can amortise the Python/NumPy call overhead. We
   report amortised per-sample latency at batch sizes {1, 32, 256} to
   separate fixed call overhead from irreducible tree-traversal cost.

Outputs
-------
- ``docs/results/benchmark/F7b_rf_latency_study.json`` — every variant's
  latency quantiles (p50/p95/p99/mean, ns->ms), tree/depth/leaf stats,
  macro-F1, and the per-sample-vs-batched amortisation curve, plus the
  canonical F7 RL/oracle p50 numbers for context.
- ``docs/results/benchmark/F7b_manifest.json`` — SHA-256 hash chain over
  the production RF artifact, the feature/label/idx inputs, and this
  script, with the git SHA at production time.

This is a standalone study: it does NOT touch ``runs/`` or the F5/F7
pipeline, and is safe to run independently of the main sweeps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.benchmark.baseline_policies import RFActingPolicy
from src.benchmark.latency import measure_inference_latency
from src.detector.evaluation import macro_f1
from src.detector.random_forest import (
    RandomForestConfig,
    load_random_forest,
    train_random_forest,
)

logger = logging.getLogger("scripts.benchmark.run_rf_latency_study")

_ROOT = Path(__file__).resolve().parents[2]

# Canonical F7 per-step p50 latencies (ms) for context in the JSON. These
# are read from F7_summary.json at runtime; the literals here are only a
# documented fallback if that file is absent on a fresh checkout.
_F7_FALLBACK_P50_MS: dict[str, float] = {
    "dqn": 0.062542,
    "ppo": 0.093833,
    "a2c": 0.093667,
    "recommended_action": 0.000583,
    "rf_acting": 16.505417,
}

_NUM_FEATURES = 29
_WINDOW_SIZE = 5
_INCLUDE_DELTAS = True


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


def _quantiles_ms(durations_ns: np.ndarray) -> dict[str, float]:
    """ns sample array -> ms quantile summary."""
    ms = durations_ns.astype(np.float64) / 1e6
    return {
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "mean_ms": float(np.mean(ms)),
        "n_samples": int(ms.size),
    }


def _build_obs_pool(
    features: np.ndarray,
    idx: np.ndarray,
    *,
    pool_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Build a pool of flattened env-style observations from real rows.

    :class:`RFActingPolicy` slices the *last* window row's raw features
    out of a flattened ``(window_size * num_features * 2,)`` obs vector
    (deltas appended). We therefore construct each obs by tiling a real
    sampled feature row across the window and zero-filling the delta
    block, so the policy's ``_extract_latest_features`` recovers the real
    row. Latency depends only on obs shape/dtype and the sampled row, not
    on temporal structure, so this faithfully reproduces the F5 query.
    """
    per_row = _NUM_FEATURES * (2 if _INCLUDE_DELTAS else 1)
    obs_dim = per_row * _WINDOW_SIZE
    sel = rng.choice(idx, size=pool_size, replace=pool_size > idx.size)
    pool: list[np.ndarray] = []
    for row_idx in sel:
        row = np.asarray(features[int(row_idx)], dtype=np.float32)
        obs = np.zeros(obs_dim, dtype=np.float32)
        # Place the real row in the raw-feature slot of every window row;
        # the policy only reads the last row's first F entries.
        for w in range(_WINDOW_SIZE):
            start = w * per_row
            obs[start : start + _NUM_FEATURES] = row
        pool.append(obs)
    return pool


def _measure_policy_latency(
    rf_model: Any,
    obs_pool: list[np.ndarray],
    *,
    n_warmup: int,
    n_measure: int,
) -> dict[str, float]:
    """Wrap an RF in RFActingPolicy and time it exactly as F5 does."""
    policy = RFActingPolicy(
        rf_model,
        num_features=_NUM_FEATURES,
        window_size=_WINDOW_SIZE,
        include_deltas=_INCLUDE_DELTAS,
    )
    durations = measure_inference_latency(
        policy,
        obs_pool,
        info_pool=None,
        n_warmup=n_warmup,
        n_measure=n_measure,
    )
    return _quantiles_ms(durations)


def _tree_stats(model: Any) -> dict[str, float]:
    """Mean tree depth / leaf count — the structural latency drivers."""
    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        # single DecisionTree
        t = model.tree_
        return {
            "n_trees": 1,
            "mean_depth": float(t.max_depth),
            "mean_leaves": float(t.n_leaves),
            "total_nodes": int(t.node_count),
        }
    depths = [est.tree_.max_depth for est in estimators]
    leaves = [est.tree_.n_leaves for est in estimators]
    nodes = [est.tree_.node_count for est in estimators]
    return {
        "n_trees": int(len(estimators)),
        "mean_depth": float(np.mean(depths)),
        "mean_leaves": float(np.mean(leaves)),
        "total_nodes": int(np.sum(nodes)),
    }


def _macro_f1_on(model: Any, X: np.ndarray, y: np.ndarray) -> float:
    y_pred = model.predict(X)
    return float(macro_f1(y, y_pred, num_classes=5))


def _batched_amortised(
    model: Any,
    X: np.ndarray,
    *,
    batch_sizes: tuple[int, ...],
    n_warmup: int,
    n_measure: int,
) -> dict[str, dict[str, float]]:
    """Amortised per-sample latency of bare ``predict`` at each batch size.

    Separates fixed Python/NumPy call overhead (dominant at batch=1) from
    irreducible tree-traversal cost (revealed as batch grows).
    """
    out: dict[str, dict[str, float]] = {}
    n = X.shape[0]
    for bs in batch_sizes:
        # warmup
        for i in range(n_warmup):
            start = (i * bs) % max(1, n - bs)
            model.predict(X[start : start + bs])
        per_sample_ns: list[float] = []
        total_ns: list[int] = []
        for i in range(n_measure):
            start = (i * bs) % max(1, n - bs)
            chunk = X[start : start + bs]
            t0 = time.perf_counter_ns()
            model.predict(chunk)
            t1 = time.perf_counter_ns()
            total_ns.append(t1 - t0)
            per_sample_ns.append((t1 - t0) / bs)
        out[str(bs)] = {
            "batch_p50_ms": float(np.percentile(np.asarray(total_ns) / 1e6, 50)),
            "amortised_per_sample_p50_ms": float(
                np.percentile(np.asarray(per_sample_ns) / 1e6, 50)
            ),
            "n_measure": int(n_measure),
        }
    return out


def _load_f7_p50() -> dict[str, float]:
    p = _ROOT / "docs/results/benchmark/F7_summary.json"
    if not p.exists():
        logger.warning("F7_summary.json missing; using documented fallback p50s")
        return dict(_F7_FALLBACK_P50_MS)
    try:
        data = json.loads(p.read_text())
        policies = data.get("policies", data)
        out: dict[str, float] = {}
        for name, rec in policies.items():
            if isinstance(rec, dict) and "p50_ms" in rec:
                out[name] = float(rec["p50_ms"])
        return out or dict(_F7_FALLBACK_P50_MS)
    except Exception:  # noqa: BLE001
        logger.warning("Could not parse F7_summary.json; using fallback p50s")
        return dict(_F7_FALLBACK_P50_MS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rf-path",
        default="artifacts/detector/random_forest.joblib",
        help="Production RF (the F5 RF-Acting backbone).",
    )
    parser.add_argument(
        "--features",
        default="data/processed/ciciot2023/features.npy",
    )
    parser.add_argument(
        "--stages",
        default="data/processed/ciciot2023/stages.npy",
    )
    parser.add_argument(
        "--splits-dir",
        default="data/processed/ciciot2023/splits",
    )
    parser.add_argument("--pool-size", type=int, default=512)
    parser.add_argument("--n-warmup", type=int, default=200)
    parser.add_argument("--n-measure", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default="docs/results/benchmark/F7b_rf_latency_study.json",
    )
    parser.add_argument("--smoke", action="store_true", help="Tiny fast run.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.smoke:
        args.pool_size = 32
        args.n_warmup = 10
        args.n_measure = 50

    rf_path = Path(args.rf_path)
    features_path = Path(args.features)
    stages_path = Path(args.stages)
    splits_dir = Path(args.splits_dir)

    rng = np.random.default_rng(args.seed)

    logger.info("Loading production RF from %s ...", rf_path)
    prod_rf = load_random_forest(rf_path)

    logger.info("Loading features / stages / splits ...")
    features = np.load(features_path, mmap_mode="r")
    stages = np.load(stages_path)
    train_idx = np.load(splits_dir / "train.idx.npy")
    test_idx = np.load(splits_dir / "test_balanced.idx.npy")

    # Materialise the train/test matrices the variants are (re)fit / scored on.
    X_train = np.asarray(features[train_idx], dtype=np.float32)
    y_train = stages[train_idx].astype(int)
    X_test = np.asarray(features[test_idx], dtype=np.float32)
    y_test = stages[test_idx].astype(int)

    if args.smoke:
        # subsample train for speed
        sub = rng.choice(X_train.shape[0], size=min(5000, X_train.shape[0]), replace=False)
        X_train, y_train = X_train[sub], y_train[sub]

    obs_pool = _build_obs_pool(features, test_idx, pool_size=args.pool_size, rng=rng)

    # ---- 1. production RF, per-sample, via RFActingPolicy (F5 reproduction) ----
    logger.info("Measuring production RF (100 trees, depth=None) ...")
    prod_latency = _measure_policy_latency(
        prod_rf, obs_pool, n_warmup=args.n_warmup, n_measure=args.n_measure
    )
    prod_stats = _tree_stats(prod_rf)
    prod_f1 = _macro_f1_on(prod_rf, X_test, y_test)
    logger.info(
        "  production: p50=%.3f ms  macro-F1=%.4f  mean_depth=%.1f  n_trees=%d",
        prod_latency["p50_ms"],
        prod_f1,
        prod_stats["mean_depth"],
        prod_stats["n_trees"],
    )

    # ---- 2. sklearn-native optimisation variants ----
    # Each entry: (label, RandomForestConfig | "decision_tree")
    variant_specs: list[tuple[str, Any]] = [
        ("rf_n50", RandomForestConfig(n_estimators=50)),
        ("rf_n25", RandomForestConfig(n_estimators=25)),
        ("rf_n10", RandomForestConfig(n_estimators=10)),
        ("rf_depth12", RandomForestConfig(n_estimators=100, max_depth=12)),
        ("rf_depth8", RandomForestConfig(n_estimators=100, max_depth=8)),
        ("rf_n25_depth12", RandomForestConfig(n_estimators=25, max_depth=12)),
        ("decision_tree", "decision_tree"),
    ]
    if args.smoke:
        variant_specs = variant_specs[:2]

    variants: dict[str, dict[str, Any]] = {}
    for label, spec in variant_specs:
        logger.info("Training + measuring variant %s ...", label)
        if spec == "decision_tree":
            from sklearn.tree import DecisionTreeClassifier

            model: Any = DecisionTreeClassifier(
                max_depth=12, class_weight="balanced", random_state=args.seed
            )
            model.fit(X_train, y_train)
        else:
            model = train_random_forest(X_train, y_train, seed=args.seed, config=spec)
        lat = _measure_policy_latency(
            model, obs_pool, n_warmup=args.n_warmup, n_measure=args.n_measure
        )
        f1 = _macro_f1_on(model, X_test, y_test)
        tstats = _tree_stats(model)
        speedup = prod_latency["p50_ms"] / lat["p50_ms"] if lat["p50_ms"] > 0 else float("inf")
        variants[label] = {
            "latency": lat,
            "macro_f1_test_balanced": f1,
            "tree_stats": tstats,
            "speedup_vs_production": float(speedup),
            "f1_delta_vs_production": float(f1 - prod_f1),
            "config": (
                {"n_estimators": spec.n_estimators, "max_depth": spec.max_depth}
                if spec != "decision_tree"
                else {"model": "DecisionTreeClassifier", "max_depth": 12}
            ),
        }
        logger.info(
            "  %s: p50=%.3f ms (%.1fx)  macro-F1=%.4f (Δ%.4f)",
            label,
            lat["p50_ms"],
            speedup,
            f1,
            f1 - prod_f1,
        )

    # ---- 3. batched / amortised throughput for the production RF ----
    logger.info("Measuring batched amortisation curve (production RF) ...")
    batch_sizes = (1,) if args.smoke else (1, 32, 256)
    amort_n_measure = 50 if args.smoke else 500
    batched = _batched_amortised(
        prod_rf,
        X_test,
        batch_sizes=batch_sizes,
        n_warmup=20 if args.smoke else 100,
        n_measure=amort_n_measure,
    )

    f7_p50 = _load_f7_p50()
    rl_best_p50 = min(f7_p50.get(a, _F7_FALLBACK_P50_MS[a]) for a in ("dqn", "ppo", "a2c"))

    summary: dict[str, Any] = {
        "figure": "F7b",
        "study": "rf_acting_measured_latency",
        "description": (
            "Measured RF-Acting inference latency and scikit-learn-native "
            "optimisation sweep, replacing the line-rate plausibility argument. "
            "Answers whether the ~16.5 ms RF-Acting p50 is fundamental or an "
            "unoptimised-forest artifact."
        ),
        "params": {
            "pool_size": args.pool_size,
            "n_warmup": args.n_warmup,
            "n_measure": args.n_measure,
            "seed": args.seed,
            "num_features": _NUM_FEATURES,
            "window_size": _WINDOW_SIZE,
            "include_deltas": _INCLUDE_DELTAS,
        },
        "production_rf": {
            "latency": prod_latency,
            "macro_f1_test_balanced": prod_f1,
            "tree_stats": prod_stats,
            "config": {"n_estimators": 100, "max_depth": None},
        },
        "variants": variants,
        "batched_amortisation_production_rf": batched,
        "context_f7_p50_ms": f7_p50,
        "rl_best_p50_ms": rl_best_p50,
        "rf_latency_budget_ms": 3.0,
        "interpretation": _interpretation(prod_latency, variants, batched, rl_best_p50),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", out_path)

    # ---- manifest ----
    manifest = {
        "figure": "F7b",
        "generated_by": "scripts/benchmark/run_rf_latency_study.py",
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "inputs": {
            "rf_path": {"path": str(rf_path), "sha256": _sha256(rf_path)},
            "features": {"path": str(features_path), "sha256": _sha256(features_path)},
            "stages": {"path": str(stages_path), "sha256": _sha256(stages_path)},
            "test_balanced_idx": {
                "path": str(splits_dir / "test_balanced.idx.npy"),
                "sha256": _sha256(splits_dir / "test_balanced.idx.npy"),
            },
            "script": {
                "path": "scripts/benchmark/run_rf_latency_study.py",
                "sha256": _sha256(Path(__file__)),
            },
        },
        "outputs": {
            "summary": {"path": str(out_path), "sha256": _sha256(out_path)},
        },
    }
    manifest_path = out_path.parent / "F7b_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote %s", manifest_path)

    return 0


def _interpretation(
    prod_latency: dict[str, float],
    variants: dict[str, dict[str, Any]],
    batched: dict[str, dict[str, float]],
    rl_best_p50: float,
) -> dict[str, Any]:
    """Machine-readable headline conclusions for the prose macros."""
    prod_p50 = prod_latency["p50_ms"]
    # Best variant that stays within 0.02 macro-F1 of production.
    quality_floor = -0.02
    admissible = {k: v for k, v in variants.items() if v["f1_delta_vs_production"] >= quality_floor}
    best_admissible = None
    if admissible:
        best_admissible = min(admissible.items(), key=lambda kv: kv[1]["latency"]["p50_ms"])
    note = {
        "production_p50_ms": prod_p50,
        "rl_best_p50_ms": rl_best_p50,
        "production_vs_rl_ratio": (
            float(prod_p50 / rl_best_p50) if rl_best_p50 > 0 else float("inf")
        ),
        "best_quality_preserving_variant": (best_admissible[0] if best_admissible else None),
        "best_quality_preserving_p50_ms": (
            best_admissible[1]["latency"]["p50_ms"] if best_admissible else None
        ),
        "best_quality_preserving_ratio_vs_rl": (
            float(best_admissible[1]["latency"]["p50_ms"] / rl_best_p50)
            if best_admissible and rl_best_p50 > 0
            else None
        ),
        "batched_amortised_p50_ms": (batched.get("256", {}).get("amortised_per_sample_p50_ms")),
        "quality_floor_f1_delta": quality_floor,
    }
    return note


if __name__ == "__main__":
    raise SystemExit(main())
