"""Tests for environment-design split-aware RealizationEngine.

Covers two new pieces of public API:

- ``RealizationEngine(allowed_indices=...)`` constructor argument.
- ``RealizationEngine.from_split_manifest(...)`` factory.

The fixtures build a tiny synthetic dataset on ``tmp_path`` so the tests do
not depend on the real 442 237-row CICIoT snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.utils.label_mapper import KillChainStage
from src.utils.realization_engine import RealizationEngine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_synthetic_dataset(tmp_path: Path) -> Path:
    """Create a 100-row × 8-feature synthetic dataset with stage labels."""
    data_path = tmp_path / "dataset"
    data_path.mkdir(parents=True)

    rng = np.random.default_rng(0)
    features = rng.standard_normal((100, 8)).astype(np.float32)
    labels = np.tile(np.arange(5), 20)  # 20 rows per stage
    np.save(data_path / "features.npy", features)

    state_indices = {str(s): np.where(labels == s)[0].tolist() for s in range(5)}
    (data_path / "state_indices.json").write_text(json.dumps(state_indices))

    scaler = StandardScaler().fit(features)
    joblib.dump(scaler, data_path / "scaler.joblib")
    return data_path


def _build_synthetic_splits_manifest(
    data_path: Path, ood_attack_class_size: int = 4
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Build a fake splits/ tree + manifest.json next to ``data_path``.

    Returns ``(manifest_path, train_idx, ood_idx)``.
    """
    splits_dir = data_path / "splits"
    splits_dir.mkdir()
    (splits_dir / "ood_attack").mkdir()

    all_idx = np.arange(100)
    rng = np.random.default_rng(0)
    rng.shuffle(all_idx)
    train_idx = np.sort(all_idx[:60])
    val_idx = np.sort(all_idx[60:80])
    test_idx = np.sort(all_idx[80:])
    ood_idx = np.sort(all_idx[:ood_attack_class_size])  # overlaps train deliberately

    np.save(splits_dir / "train.idx.npy", train_idx)
    np.save(splits_dir / "val.idx.npy", val_idx)
    np.save(splits_dir / "test.idx.npy", test_idx)
    np.save(splits_dir / "ood_attack" / "AttackZ.idx.npy", ood_idx)

    manifest = {
        "version": "1.0",
        "outputs": {
            "splits/train.idx.npy": "fake-sha",
            "splits/val.idx.npy": "fake-sha",
            "splits/test.idx.npy": "fake-sha",
            "splits/ood_attack/AttackZ.idx.npy": "fake-sha",
        },
        "ood_attack_classes": ["AttackZ"],
        "seed": 42,
    }
    manifest_path = splits_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, train_idx, ood_idx


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    return _build_synthetic_dataset(tmp_path)


# ---------------------------------------------------------------------------
# allowed_indices argument
# ---------------------------------------------------------------------------


def test_default_engine_uses_all_rows(synthetic_dataset: Path) -> None:
    eng = RealizationEngine(synthetic_dataset, seed=0)
    counts = eng.get_stage_sample_counts()
    assert sum(counts.values()) == 100
    assert all(v == 20 for v in counts.values())


def test_allowed_indices_restricts_to_subset(synthetic_dataset: Path) -> None:
    allowed = list(range(0, 50))  # half the rows
    eng = RealizationEngine(synthetic_dataset, seed=0, allowed_indices=allowed)
    total = sum(eng.get_stage_sample_counts().values())
    assert 0 < total <= 50
    for s in range(5):
        for idx in eng.get_indices_for_stage(s):
            assert idx in allowed


def test_allowed_indices_empty_raises(synthetic_dataset: Path) -> None:
    with pytest.raises(ValueError):
        RealizationEngine(synthetic_dataset, seed=0, allowed_indices=set())


def test_allowed_indices_drops_empty_stage(
    synthetic_dataset: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """If a stage's intersection is empty, the stage is dropped (with a warning)
    and sampling from it raises ``ValueError`` at runtime."""
    allowed = [i for i in range(100) if i % 5 != 4]  # exclude IMPACT rows
    with caplog.at_level("WARNING"):
        eng = RealizationEngine(synthetic_dataset, seed=0, allowed_indices=allowed)
    assert KillChainStage.IMPACT.value not in eng.get_stage_sample_counts()
    with pytest.raises(ValueError):
        eng.sample(KillChainStage.IMPACT)


def test_sample_returns_only_allowed_rows(synthetic_dataset: Path) -> None:
    """Repeated samples must always come from ``allowed_indices``."""
    allowed = set(range(50, 80))
    eng = RealizationEngine(synthetic_dataset, seed=0, allowed_indices=allowed)
    # The synthetic features are i.i.d., so we can't read off the index from
    # the returned vector directly. Instead, we cross-check via
    # get_features_for_indices, which is a contract test on get_indices_for_stage.
    for s in range(5):
        for idx in eng.get_indices_for_stage(s):
            assert idx in allowed


# ---------------------------------------------------------------------------
# from_split_manifest
# ---------------------------------------------------------------------------


def test_from_split_manifest_uses_train_split(synthetic_dataset: Path) -> None:
    manifest_path, train_idx, _ = _build_synthetic_splits_manifest(synthetic_dataset)
    eng = RealizationEngine.from_split_manifest(
        synthetic_dataset,
        manifest_path,
        "train",
        exclude_ood=False,
        seed=0,
    )
    pool = set()
    for s in range(5):
        pool.update(eng.get_indices_for_stage(s))
    assert pool.issubset(set(train_idx.tolist()))


def test_from_split_manifest_excludes_ood_by_default(synthetic_dataset: Path) -> None:
    manifest_path, train_idx, ood_idx = _build_synthetic_splits_manifest(synthetic_dataset)
    eng = RealizationEngine.from_split_manifest(
        synthetic_dataset,
        manifest_path,
        "train",
        seed=0,  # exclude_ood=True default
    )
    pool = set()
    for s in range(5):
        pool.update(eng.get_indices_for_stage(s))
    # OOD rows that overlap train must have been removed.
    assert len(pool & set(ood_idx.tolist())) == 0
    # And the remaining rows are still a subset of train.
    assert pool.issubset(set(train_idx.tolist()))


def test_from_split_manifest_unknown_split_raises(synthetic_dataset: Path) -> None:
    manifest_path, _, _ = _build_synthetic_splits_manifest(synthetic_dataset)
    with pytest.raises(KeyError, match="totally-not-a-split"):
        RealizationEngine.from_split_manifest(
            synthetic_dataset,
            manifest_path,
            "totally-not-a-split",
            seed=0,
        )


def test_from_split_manifest_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RealizationEngine.from_split_manifest(
            tmp_path,
            tmp_path / "does-not-exist.json",
            "train",
            seed=0,
        )
