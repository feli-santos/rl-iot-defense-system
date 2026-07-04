"""Stage Detector ``src/detector/`` regression tests.

One test file rather than four because the four modules are intentionally
small and tightly coupled. The synthetic-toy fixtures (29-D Gaussian
clusters, one per stage) are shared across model tests so all three
baselines train on the *same* problem and the comparison is meaningful.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.detector import (
    DetectorEvaluation,
    RandomForestConfig,
    confusion_matrix,
    macro_f1,
    per_class_f1,
    per_stage_recall,
    summarize_run,
    train_random_forest,
)
from src.detector.evaluation import NUM_STAGES, evaluate_ood_class

# ---------------------------------------------------------------------------
# Shared toy dataset: 29-D Gaussian clusters, one per stage. Linearly
# separable enough that both baselines should achieve near-perfect F1.
# ---------------------------------------------------------------------------


# Cluster centres are computed ONCE (fixed seed) so train/val/test draw
# from the same underlying generative model — otherwise the test set
# would live in a totally different region of feature space and *no*
# model could possibly generalise.
_TOY_CENTRES: np.ndarray = (
    np.random.default_rng(12345).standard_normal((NUM_STAGES, 29)) * 5.0
).astype(np.float32)


def _toy_dataset(
    *, n_per_class: int = 600, num_features: int = 29, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    if num_features != _TOY_CENTRES.shape[1]:
        raise ValueError("This fixture is hard-wired to 29 features.")
    rng = np.random.default_rng(seed)
    X_parts, y_parts = [], []
    for stage in range(NUM_STAGES):
        X_parts.append(_TOY_CENTRES[stage] + rng.standard_normal((n_per_class, num_features)))
        y_parts.append(np.full(n_per_class, stage, dtype=np.int64))
    X = np.vstack(X_parts).astype(np.float32)
    y = np.concatenate(y_parts)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


@pytest.fixture(scope="module")
def toy_train_val_test() -> tuple:
    X_tr, y_tr = _toy_dataset(n_per_class=400, seed=0)
    X_val, y_val = _toy_dataset(n_per_class=80, seed=1)
    X_te, y_te = _toy_dataset(n_per_class=160, seed=2)
    return X_tr, y_tr, X_val, y_val, X_te, y_te


# ---------------------------------------------------------------------------
# evaluation.py
# ---------------------------------------------------------------------------


class TestEvaluationModule:
    def test_confusion_matrix_shape_and_sum(self) -> None:
        y_true = np.array([0, 1, 2, 3, 4, 0, 1])
        y_pred = np.array([0, 1, 2, 4, 4, 0, 0])
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (NUM_STAGES, NUM_STAGES)
        assert cm.sum() == y_true.shape[0]
        assert cm[0, 0] == 2  # both BENIGNs predicted correctly
        assert cm[3, 4] == 1  # one MANEUVER predicted as IMPACT

    def test_per_stage_recall_matches_sklearn(self) -> None:
        from sklearn.metrics import recall_score

        rng = np.random.default_rng(0)
        y_true = rng.integers(0, NUM_STAGES, size=300)
        y_pred = rng.integers(0, NUM_STAGES, size=300)
        ours = per_stage_recall(y_true, y_pred)
        theirs = recall_score(
            y_true, y_pred, labels=list(range(NUM_STAGES)), average=None, zero_division=0
        )
        np.testing.assert_allclose(ours, theirs, atol=1e-12)

    def test_per_class_f1_matches_sklearn(self) -> None:
        from sklearn.metrics import f1_score

        rng = np.random.default_rng(1)
        y_true = rng.integers(0, NUM_STAGES, size=400)
        y_pred = rng.integers(0, NUM_STAGES, size=400)
        ours = per_class_f1(y_true, y_pred)
        theirs = f1_score(
            y_true, y_pred, labels=list(range(NUM_STAGES)), average=None, zero_division=0
        )
        np.testing.assert_allclose(ours, theirs, atol=1e-12)

    def test_macro_f1_matches_sklearn(self) -> None:
        from sklearn.metrics import f1_score

        rng = np.random.default_rng(2)
        y_true = rng.integers(0, NUM_STAGES, size=500)
        y_pred = rng.integers(0, NUM_STAGES, size=500)
        ours = macro_f1(y_true, y_pred)
        theirs = f1_score(
            y_true, y_pred, labels=list(range(NUM_STAGES)), average="macro", zero_division=0
        )
        assert abs(ours - theirs) < 1e-12

    def test_summarize_run_returns_evaluation_dataclass(self) -> None:
        y_true = np.array([0, 1, 2, 3, 4] * 20)
        y_pred = y_true.copy()
        evaluation = summarize_run("ToyModel", "test_balanced", y_true, y_pred)
        assert isinstance(evaluation, DetectorEvaluation)
        assert evaluation.macro_f1 == 1.0
        assert evaluation.n_samples == 100
        d = evaluation.to_dict()
        assert d["macro_f1"] == 1.0
        assert "BENIGN" in d["per_stage_recall"]

    def test_recall_with_empty_class_does_not_crash(self) -> None:
        y_true = np.array([0, 1, 2, 3])  # no IMPACT (4)
        y_pred = np.array([0, 1, 2, 3])
        rec = per_stage_recall(y_true, y_pred)
        assert rec.shape == (NUM_STAGES,)
        assert rec[4] == 0.0  # zero-support class -> 0.0, no NaN

    def test_evaluate_ood_class_recall(self) -> None:
        # 100 OOD-class predictions, 30 of which fall on the expected stage.
        y_pred = np.concatenate([np.full(30, 2), np.full(70, 0)])
        ood = evaluate_ood_class(attack_class="ToyOOD", expected_stage=2, y_pred=y_pred)
        assert ood.recall == pytest.approx(0.3)
        assert ood.n_samples == 100
        assert ood.predicted_stage_distribution["ACCESS"] == 30
        assert ood.predicted_stage_distribution["BENIGN"] == 70


# ---------------------------------------------------------------------------
# random_forest.py
# ---------------------------------------------------------------------------


class TestRandomForest:
    def test_default_config(self) -> None:
        cfg = RandomForestConfig()
        assert cfg.n_estimators == 100
        assert cfg.class_weight == "balanced"

    def test_fit_predicts_separable_clusters(self, toy_train_val_test) -> None:
        X_tr, y_tr, _, _, X_te, y_te = toy_train_val_test
        # Smaller forest for speed in tests.
        cfg = RandomForestConfig(n_estimators=20)
        rf = train_random_forest(X_tr, y_tr, seed=0, config=cfg)
        y_pred = rf.predict(X_te)
        assert y_pred.shape == (X_te.shape[0],)
        f1 = macro_f1(y_te, y_pred.astype(np.int64))
        assert f1 > 0.90, f"RF toy macro-F1 = {f1:.3f}, expected > 0.90"

    def test_predict_proba_shape_and_sum(self, toy_train_val_test) -> None:
        X_tr, y_tr, _, _, X_te, _ = toy_train_val_test
        cfg = RandomForestConfig(n_estimators=10)
        rf = train_random_forest(X_tr, y_tr, seed=0, config=cfg)
        proba = rf.predict_proba(X_te)
        assert proba.shape == (X_te.shape[0], 5)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)

    def test_run_info_attached(self, toy_train_val_test) -> None:
        X_tr, y_tr, *_ = toy_train_val_test
        rf = train_random_forest(X_tr, y_tr, seed=0, config=RandomForestConfig(n_estimators=8))
        assert hasattr(rf, "run_info")
        info = rf.run_info  # type: ignore[attr-defined]
        assert info.n_train == X_tr.shape[0]
        assert info.n_features == X_tr.shape[1]
        assert info.feature_importances.shape == (29,)
        assert info.train_time_seconds > 0


class TestRandomForestConfig:
    """Test RandomForestConfig parameterisation (review 2.4.2)."""

    def test_default_n_estimators(self) -> None:
        cfg = RandomForestConfig()
        assert cfg.n_estimators == 100

    def test_custom_n_estimators(self, toy_train_val_test) -> None:
        X_tr, y_tr, *_ = toy_train_val_test
        cfg = RandomForestConfig(n_estimators=10)
        rf = train_random_forest(X_tr, y_tr, seed=0, config=cfg)
        assert rf.n_estimators == 10
        assert rf.run_info.n_train == X_tr.shape[0]
