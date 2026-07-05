"""Tune the Random Forest detector baseline (fairness re-run).

Motivation
----------
The RF-Acting baseline (``artifacts/detector/random_forest.joblib``) was
trained with a *fixed* configuration (``n_estimators=100``, default depth),
never hyperparameter-searched. For a fair head-to-head against the trained
RL agents, every benchmarked policy must be tuned, RF included; otherwise a
reviewer can dismiss "windowed RL beats RF" as an under-tuned-baseline
artefact.

This script runs an explicit grid search over the RF configuration, selects
the configuration that maximises **macro-F1 on the held-out balanced
validation split** (``val_balanced``, the same selection signal the MLP
detector uses), refits on the training split with that configuration, and
reports its accuracy and per-class out-of-distribution recall on the ten
held-out zero-day classes.

Honesty guarantees
------------------
- Selection uses ``val_balanced`` only; ``test_balanced`` / ``test`` /
  the OOD splits are *reported* but never used for model selection.
- Trains on the SAME ``splits/train`` indices the RL agents and the
  original detector use (the 10-class-excluded split); disjointness from
  val/test/ood is verified before training (re-using the detector's check).
- Operates on the raw ``features.npy`` matrix exactly as the production RF
  and the benchmark ``RFActingPolicy`` do (no scaler transform).

Safety
------
By default the script is non-destructive: it writes the tuned model and a
provenance JSON to ``--out-dir`` (default ``artifacts/detector/tuned``) and
PRINTS the comparison, but does NOT overwrite the in-use
``artifacts/detector/random_forest.joblib``. Pass ``--commit`` to promote the
tuned model into place (the previous model is backed up alongside it).

Parallelism
-----------
Each RF fit already uses ``n_jobs=-1``. Grid points are evaluated
sequentially (one all-core fit at a time) to avoid CPU oversubscription;
DO NOT run this concurrently with a ``--parallel`` training sweep.

Usage
-----
    .venv/bin/python -m scripts.detector.tune_random_forest          # dry run
    .venv/bin/python -m scripts.detector.tune_random_forest --commit # promote
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import itertools
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from scripts.detector.train_detector import (
    _OOD_EXPECTED_STAGE,
    _git_sha,
    _load_ood_split,
    _load_split,
    _sha256,
    _verify_disjoint,
)
from src.detector.random_forest import RandomForestConfig, save_random_forest, train_random_forest

logger = logging.getLogger("tune_random_forest")


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
# Grid kept deliberately compact (3 x 3 x 3 x 2 = 54 fits). Each fit is a
# full-core RandomForest on ~235k x 29; a single fit is seconds-to-minutes,
# so the whole grid is a small number of minutes on the dev box.
_GRID_N_ESTIMATORS: tuple[int, ...] = (200, 400, 800)
_GRID_MAX_DEPTH: tuple[int | None, ...] = (None, 20, 40)
_GRID_MIN_SAMPLES_LEAF: tuple[int, ...] = (1, 2, 4)
_GRID_CLASS_WEIGHT: tuple[str, ...] = ("balanced", "balanced_subsample")


@dataclasses.dataclass
class _GridResult:
    config: dict
    val_macro_f1: float
    test_balanced_macro_f1: float
    train_time_seconds: float


def _macro_f1(model, X: np.ndarray, y: np.ndarray) -> float:
    pred = model.predict(X)
    return float(f1_score(y, pred, average="macro"))


def _ood_recall(model, X: np.ndarray, idx: np.ndarray, expected_stage: int) -> float:
    """Fraction of an OOD class's rows the RF maps to its canonical stage.

    Predicts on the RAW feature rows directly (no scaler) - matching the
    production RF and the benchmark ``RFActingPolicy``.
    """
    if idx.size == 0:
        return float("nan")
    rows = np.ascontiguousarray(X[idx], dtype=np.float32)
    pred = model.predict(rows)
    return float(np.mean(pred == expected_stage))


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Grid-tune the RF detector baseline.")
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed/ciciot2023"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/detector/tuned"))
    p.add_argument(
        "--rf-path",
        type=Path,
        default=Path("artifacts/detector/random_forest.joblib"),
        help="In-use RF model; promoted to this path when --commit is set.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--commit",
        action="store_true",
        help="Promote the tuned model to --rf-path (backs up the previous one).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load features + stages + splits (mirror train_detector.py).
    logger.info("Loading features and stages from %s", args.processed_dir)
    X = np.load(args.processed_dir / "features.npy", mmap_mode="r")
    y = np.load(args.processed_dir / "stages.npy", mmap_mode="r").astype(np.int64)

    splits = {
        name: _load_split(args.processed_dir, name)
        for name in ("train", "val_balanced", "test_balanced")
    }
    ood_class_to_idx = {
        cls: _load_ood_split(args.processed_dir, cls) for cls in _OOD_EXPECTED_STAGE
    }
    _verify_disjoint({**splits, **{f"ood:{c}": idx for c, idx in ood_class_to_idx.items()}})

    X_train = np.ascontiguousarray(X[splits["train"]], dtype=np.float32)
    y_train = y[splits["train"]].astype(np.int64)
    X_val = np.ascontiguousarray(X[splits["val_balanced"]], dtype=np.float32)
    y_val = y[splits["val_balanced"]].astype(np.int64)
    X_tb = np.ascontiguousarray(X[splits["test_balanced"]], dtype=np.float32)
    y_tb = y[splits["test_balanced"]].astype(np.int64)

    logger.info(
        "train=%d val_balanced=%d test_balanced=%d  num_features=%d",
        X_train.shape[0],
        X_val.shape[0],
        X_tb.shape[0],
        X_train.shape[1],
    )

    # ---- Grid search (select on val_balanced macro-F1).
    grid = list(
        itertools.product(
            _GRID_N_ESTIMATORS,
            _GRID_MAX_DEPTH,
            _GRID_MIN_SAMPLES_LEAF,
            _GRID_CLASS_WEIGHT,
        )
    )
    logger.info("Grid search over %d RF configurations...", len(grid))
    results: list[_GridResult] = []
    best_model = None
    best_result: _GridResult | None = None
    t_grid = time.perf_counter()
    for i, (n_est, max_depth, min_leaf, cls_w) in enumerate(grid, start=1):
        cfg = RandomForestConfig(
            n_estimators=n_est,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            class_weight=cls_w,
        )
        model = train_random_forest(X_train, y_train, seed=args.seed, config=cfg)
        val_f1 = _macro_f1(model, X_val, y_val)
        tb_f1 = _macro_f1(model, X_tb, y_tb)
        res = _GridResult(
            config=dataclasses.asdict(cfg),
            val_macro_f1=val_f1,
            test_balanced_macro_f1=tb_f1,
            train_time_seconds=float(model.run_info.train_time_seconds),
        )
        results.append(res)
        logger.info(
            "  [%2d/%2d] n_est=%-3d depth=%-4s leaf=%d cw=%-18s  "
            "val_macroF1=%.4f  tb_macroF1=%.4f  (%.1fs)",
            i,
            len(grid),
            n_est,
            str(max_depth),
            min_leaf,
            cls_w,
            val_f1,
            tb_f1,
            res.train_time_seconds,
        )
        if best_result is None or val_f1 > best_result.val_macro_f1:
            best_result = res
            best_model = model
    grid_seconds = time.perf_counter() - t_grid
    assert best_model is not None and best_result is not None

    # ---- Per-class OOD recall for the selected (best) model.
    ood_recall = {
        cls: _ood_recall(best_model, X, ood_class_to_idx[cls], stage)
        for cls, stage in _OOD_EXPECTED_STAGE.items()
    }

    logger.info("=" * 70)
    logger.info(
        "BEST RF config (selected on val_balanced macro-F1=%.4f): %s",
        best_result.val_macro_f1,
        best_result.config,
    )
    logger.info("  test_balanced macro-F1 = %.4f", best_result.test_balanced_macro_f1)
    logger.info("  per-class OOD recall (tuned RF):")
    for cls in _OOD_EXPECTED_STAGE:
        logger.info("    %-26s %.4f", cls, ood_recall[cls])

    # ---- Persist tuned model + provenance.
    tuned_path = args.out_dir / "random_forest_tuned.joblib"
    save_random_forest(best_model, tuned_path)

    provenance = {
        "kind": "rf_detector_tuning",
        "git_sha": _git_sha(),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "seed": args.seed,
        "selection_metric": "val_balanced macro-F1",
        "grid_size": len(grid),
        "grid_search_seconds": round(grid_seconds, 2),
        "search_space": {
            "n_estimators": list(_GRID_N_ESTIMATORS),
            "max_depth": list(_GRID_MAX_DEPTH),
            "min_samples_leaf": list(_GRID_MIN_SAMPLES_LEAF),
            "class_weight": list(_GRID_CLASS_WEIGHT),
        },
        "best_config": best_result.config,
        "best_val_macro_f1": best_result.val_macro_f1,
        "best_test_balanced_macro_f1": best_result.test_balanced_macro_f1,
        "ood_rf_recall_tuned": ood_recall,
        "n_train": int(X_train.shape[0]),
        "n_val_balanced": int(X_val.shape[0]),
        "n_test_balanced": int(X_tb.shape[0]),
        "num_features": int(X_train.shape[1]),
        "tuned_model_path": str(tuned_path),
        "tuned_model_sha256": _sha256(tuned_path),
        "all_results": [dataclasses.asdict(r) for r in results],
        "input_hashes": {
            "features.npy": _sha256(args.processed_dir / "features.npy"),
            "stages.npy": _sha256(args.processed_dir / "stages.npy"),
            "splits/train.idx.npy": _sha256(args.processed_dir / "splits" / "train.idx.npy"),
            "splits/val_balanced.idx.npy": _sha256(
                args.processed_dir / "splits" / "val_balanced.idx.npy"
            ),
        },
    }
    prov_path = args.out_dir / "rf_tuning_summary.json"
    prov_path.write_text(json.dumps(provenance, indent=2, default=str))
    logger.info("Wrote tuned model -> %s", tuned_path)
    logger.info("Wrote provenance  -> %s", prov_path)

    # ---- Optional promotion (non-destructive backup).
    if args.commit:
        rf_path = args.rf_path
        if rf_path.exists():
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = rf_path.with_name(f"{rf_path.stem}.pretune_{ts}.joblib")
            shutil.copy2(rf_path, backup)
            logger.info("Backed up previous RF -> %s", backup)
        shutil.copy2(tuned_path, rf_path)
        logger.info("Promoted tuned RF -> %s (in use by RF-Acting)", rf_path)
    else:
        logger.info(
            "DRY RUN: tuned model NOT promoted. Re-run with --commit to " "replace %s.",
            args.rf_path,
        )


if __name__ == "__main__":
    main()
