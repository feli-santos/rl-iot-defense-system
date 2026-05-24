"""Random Forest baseline for detector.

Wraps ``sklearn.ensemble.RandomForestClassifier`` with the canonical
configuration locked in PLAN §8.D3:

    n_estimators=100
    class_weight="balanced"
    random_state=<seed>
    n_jobs=-1

Why a thin wrapper rather than calling sklearn directly?

- Locks the configuration (no ad-hoc tweaks across runs).
- Provides a uniform ``train_random_forest(...) -> RandomForestClassifier``
  factory matching the other detector baselines.
- Saves a tiny amount of book-keeping (training time, feature importances)
  inside an attribute on the returned model so the entrypoint script can
  ship them in the summary JSON.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


@dataclass
class RandomForestConfig:
    """Locked configuration for the detector RF baseline (PLAN §8.D3)."""

    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_leaf: int = 1
    class_weight: str = "balanced"
    n_jobs: int = -1


@dataclass
class RandomForestRunInfo:
    """Sidecar metadata attached to the returned classifier."""

    train_time_seconds: float
    n_train: int
    n_features: int
    feature_importances: np.ndarray = field(default_factory=lambda: np.zeros(0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    config: RandomForestConfig | None = None,
) -> RandomForestClassifier:
    """Fit a RandomForest with the locked detector configuration.

    The fitted classifier has a ``run_info`` attribute (a
    :class:`RandomForestRunInfo`) with training metadata that the
    entrypoint script will surface in the summary JSON.

    Args:
        X_train: ``(n, num_features)`` float array.
        y_train: ``(n,)`` int array of stage labels (0..4).
        seed: Random seed (reproducibility).
        config: Optional override; defaults to :class:`RandomForestConfig()`.

    Returns:
        Fitted ``RandomForestClassifier`` with a ``run_info`` attribute.
    """
    cfg = config or RandomForestConfig()
    clf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        class_weight=cfg.class_weight,
        random_state=seed,
        n_jobs=cfg.n_jobs,
    )
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    clf.run_info = RandomForestRunInfo(  # type: ignore[attr-defined]
        train_time_seconds=elapsed,
        n_train=int(X_train.shape[0]),
        n_features=int(X_train.shape[1]),
        feature_importances=np.asarray(clf.feature_importances_, dtype=np.float64),
    )
    return clf


def save_random_forest(model: RandomForestClassifier, path: Path) -> None:
    """Persist a fitted RF classifier to disk (joblib)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_random_forest(path: Path) -> RandomForestClassifier:
    """Load a previously-saved RF classifier."""
    return joblib.load(Path(path))
