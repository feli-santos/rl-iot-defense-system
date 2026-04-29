"""Phase-4 ``src/detector/`` regression tests.

One test file rather than four because the four modules are intentionally
small and tightly coupled. The synthetic-toy fixtures (29-D Gaussian
clusters, one per stage) are shared across model tests so all three
baselines train on the *same* problem and the comparison is meaningful.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.detector import (
    CNN1D,
    CNN1DConfig,
    DetectorEvaluation,
    RandomForestConfig,
    StageDetector,
    StageDetectorConfig,
    confusion_matrix,
    macro_f1,
    per_class_f1,
    per_stage_recall,
    summarize_run,
    train_cnn1d,
    train_random_forest,
)
from src.detector.evaluation import NUM_STAGES, evaluate_ood_class

# ---------------------------------------------------------------------------
# Shared toy dataset: 29-D Gaussian clusters, one per stage. Linearly
# separable enough that all three baselines should achieve near-perfect F1.
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
        X_parts.append(
            _TOY_CENTRES[stage] + rng.standard_normal((n_per_class, num_features))
        )
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
# stage_detector.py
# ---------------------------------------------------------------------------


class TestStageDetector:
    def test_default_config(self) -> None:
        cfg = StageDetectorConfig()
        assert cfg.num_features == 29
        assert cfg.num_classes == 5
        assert cfg.hidden_sizes == (64, 32)

    def test_predict_before_fit_raises(self) -> None:
        det = StageDetector()
        with pytest.raises(RuntimeError, match="before fit"):
            det.predict_proba(np.zeros((1, 29), dtype=np.float32))

    def test_fit_predicts_separable_clusters(self, toy_train_val_test) -> None:
        X_tr, y_tr, X_val, y_val, X_te, y_te = toy_train_val_test
        cfg = StageDetectorConfig(max_epochs=8, patience=3)
        det = StageDetector(cfg).fit(X_tr, y_tr, X_val, y_val, seed=0, verbose=False)
        y_pred = det.predict(X_te)
        f1 = macro_f1(y_te, y_pred)
        assert f1 > 0.90, f"toy macro-F1 = {f1:.3f}, expected > 0.90"

    def test_predict_proba_sums_to_one(self, toy_train_val_test) -> None:
        X_tr, y_tr, X_val, y_val, X_te, _ = toy_train_val_test
        cfg = StageDetectorConfig(max_epochs=3, patience=3)
        det = StageDetector(cfg).fit(X_tr, y_tr, X_val, y_val, seed=0, verbose=False)
        proba = det.predict_proba(X_te)
        assert proba.shape == (X_te.shape[0], 5)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_save_and_load_round_trip(self, tmp_path: Path, toy_train_val_test) -> None:
        X_tr, y_tr, X_val, y_val, X_te, _ = toy_train_val_test
        cfg = StageDetectorConfig(max_epochs=3, patience=3)
        det = StageDetector(cfg).fit(X_tr, y_tr, X_val, y_val, seed=0, verbose=False)
        ckpt = tmp_path / "stage_det.pt"
        det.save(ckpt)
        # Sidecar JSON exists.
        assert (ckpt.with_suffix(".run_info.json")).exists()
        loaded = StageDetector.from_checkpoint(ckpt)
        np.testing.assert_array_equal(det.predict(X_te), loaded.predict(X_te))

    def test_run_info_populated(self, toy_train_val_test) -> None:
        X_tr, y_tr, X_val, y_val, _, _ = toy_train_val_test
        cfg = StageDetectorConfig(max_epochs=4, patience=3)
        det = StageDetector(cfg).fit(X_tr, y_tr, X_val, y_val, seed=0, verbose=False)
        assert len(det.run_info.train_loss_history) >= 1
        assert det.run_info.best_epoch >= 1
        assert 0.0 <= det.run_info.best_val_macro_f1 <= 1.0
        assert det.run_info.train_time_seconds > 0

    def test_inference_latency_under_one_ms(self, toy_train_val_test) -> None:
        """Per-sample inference must be under ≈ 1 ms (PLAN G4.5).

        We measure on a single 29-D vector, repeated 1 000 times. A
        generous 5-ms budget here lets the test pass on slow CI runners
        while still being orders of magnitude away from a real bug
        (e.g., accidentally using a window-of-features architecture).
        """
        X_tr, y_tr, X_val, y_val, _, _ = toy_train_val_test
        cfg = StageDetectorConfig(max_epochs=2, patience=3)
        det = StageDetector(cfg).fit(X_tr, y_tr, X_val, y_val, seed=0, verbose=False)
        x = np.random.RandomState(0).standard_normal((1, 29)).astype(np.float32)

        # Warm-up
        for _ in range(20):
            det.predict_proba(x)

        import time
        n_iter = 1000
        t0 = time.perf_counter()
        for _ in range(n_iter):
            det.predict_proba(x)
        elapsed_ms = (time.perf_counter() - t0) * 1000 / n_iter
        assert elapsed_ms < 5.0, f"per-sample inference {elapsed_ms:.2f} ms is too slow"


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


# ---------------------------------------------------------------------------
# cnn1d.py
# ---------------------------------------------------------------------------


class TestCNN1D:
    def test_default_config(self) -> None:
        cfg = CNN1DConfig()
        assert cfg.num_features == 29
        assert cfg.num_classes == 5

    def test_forward_shape(self) -> None:
        cfg = CNN1DConfig()
        cnn = CNN1D(cfg)
        # Need to fit at least once to construct the inner _ConvNet, but we
        # can also directly poke forward via the model to test the shape.
        from src.detector.cnn1d import _ConvNet

        net = _ConvNet(cfg)
        x = torch.randn(7, 29)
        out = net(x)
        assert out.shape == (7, 5)
        # Also accept (N, 1, F).
        x2 = torch.randn(7, 1, 29)
        out2 = net(x2)
        assert out2.shape == (7, 5)

    def test_fit_reduces_loss_on_toy(self, toy_train_val_test) -> None:
        X_tr, y_tr, X_val, y_val, X_te, y_te = toy_train_val_test
        cfg = CNN1DConfig(max_epochs=8, patience=3)
        cnn = train_cnn1d(X_tr, y_tr, X_val, y_val, seed=0, config=cfg, verbose=False)
        # First-epoch loss should be greater than last-epoch loss.
        history = cnn.run_info.train_loss_history
        assert len(history) >= 2
        assert history[0] > history[-1], (
            f"loss did not decrease: {history[0]:.4f} -> {history[-1]:.4f}"
        )
        # And toy macro-F1 should be > 0.85 (CNN is a bit weaker than MLP at
        # this problem size, hence the slightly lower bar).
        f1 = macro_f1(y_te, cnn.predict(X_te))
        assert f1 > 0.85, f"CNN toy macro-F1 = {f1:.3f}, expected > 0.85"

    def test_predict_proba_shape_and_sum(self, toy_train_val_test) -> None:
        X_tr, y_tr, X_val, y_val, X_te, _ = toy_train_val_test
        cfg = CNN1DConfig(max_epochs=3, patience=3)
        cnn = train_cnn1d(X_tr, y_tr, X_val, y_val, seed=0, config=cfg, verbose=False)
        proba = cnn.predict_proba(X_te)
        assert proba.shape == (X_te.shape[0], 5)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_save_and_load_round_trip(self, tmp_path: Path, toy_train_val_test) -> None:
        X_tr, y_tr, X_val, y_val, X_te, _ = toy_train_val_test
        cfg = CNN1DConfig(max_epochs=3, patience=3)
        cnn = train_cnn1d(X_tr, y_tr, X_val, y_val, seed=0, config=cfg, verbose=False)
        ckpt = tmp_path / "cnn1d.pt"
        cnn.save(ckpt)
        assert (ckpt.with_suffix(".run_info.json")).exists()
        loaded = CNN1D.from_checkpoint(ckpt)
        np.testing.assert_array_equal(cnn.predict(X_te), loaded.predict(X_te))
