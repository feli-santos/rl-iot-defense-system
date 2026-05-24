"""Tests for ``scripts.data.build_split_indices``.

These tests run on tiny synthetic data (no dependence on
``data/processed/ciciot2023/``) and exercise the core invariants of the
split builder: determinism, exhaustivity, disjointness, balanced subsetting,
and OOD-class extraction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.data.build_split_indices import (
    BuilderConfig,
    _balanced_subset,
    _ood_attack_indices,
    _stratified_split,
    _string_to_stage_ids,
    build_splits,
)
from src.utils.label_mapper import AbstractStateLabelMapper


# ---------------------------------------------------------------------------
# Synthetic-data helpers
# ---------------------------------------------------------------------------

# A small set of real CICIoT2023 labels covering all 5 Kill Chain stages.
_LABELS_COVERING_ALL_STAGES: list[str] = [
    "BenignTraffic",        # stage 0
    "Recon-PortScan",       # stage 1
    "VulnerabilityScan",    # stage 1
    "SqlInjection",         # stage 2
    "DictionaryBruteForce", # stage 2
    "MITM-ArpSpoofing",     # stage 3
    "Mirai-udpplain",       # stage 3
    "DDoS-ICMP_Flood",      # stage 4
    "DDoS-HTTP_Flood",      # stage 4
]


def _build_synthetic_processed_dir(tmp_path: Path, *, n_per_label: int = 50) -> Path:
    """Materialize a tiny ``processed/ciciot2023``-style directory."""
    processed = tmp_path / "ciciot2023"
    processed.mkdir(parents=True)

    rng = np.random.default_rng(0)
    string_labels: list[str] = []
    for lbl in _LABELS_COVERING_ALL_STAGES:
        string_labels.extend([lbl] * n_per_label)
    rng.shuffle(string_labels)
    string_labels_arr = np.asarray(string_labels)
    n = string_labels_arr.size

    features = rng.normal(size=(n, 4)).astype(np.float32)
    np.save(processed / "features.npy", features)
    np.save(processed / "labels.npy", string_labels_arr)

    metadata = {
        "num_samples": n,
        "num_features": 4,
        "num_stages": 5,
        "feature_columns": [f"f{i}" for i in range(4)],
    }
    (processed / "metadata.json").write_text(json.dumps(metadata))
    return processed


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestStringToStageIds:
    def test_maps_known_labels(self) -> None:
        labels = np.asarray(["BenignTraffic", "DDoS-ICMP_Flood", "Recon-PortScan"])
        out = _string_to_stage_ids(labels, AbstractStateLabelMapper())
        np.testing.assert_array_equal(out, [0, 4, 1])

    def test_raises_on_unknown_label(self) -> None:
        labels = np.asarray(["BenignTraffic", "TotallyNewAttack"])
        with pytest.raises(KeyError, match="not in the AbstractStateLabelMapper"):
            _string_to_stage_ids(labels, AbstractStateLabelMapper())


class TestStratifiedSplit:
    def test_partition_is_exhaustive_and_disjoint(self) -> None:
        labels = np.repeat(np.arange(5), 100)  # 500 samples, 100/stage
        rng = np.random.default_rng(42)
        tr, va, te = _stratified_split(labels, (0.7, 0.1, 0.2), rng)
        assert tr.size + va.size + te.size == labels.size
        union = np.concatenate([tr, va, te])
        assert np.unique(union).size == labels.size, "splits overlap"

    def test_per_stage_ratios_approx(self) -> None:
        labels = np.repeat(np.arange(5), 1000)
        rng = np.random.default_rng(42)
        tr, va, te = _stratified_split(labels, (0.7, 0.1, 0.2), rng)
        for stage in range(5):
            assert (labels[tr] == stage).sum() == 700
            assert (labels[va] == stage).sum() == 100
            assert (labels[te] == stage).sum() == 200

    def test_seed_determinism(self) -> None:
        labels = np.repeat(np.arange(5), 100)
        a = _stratified_split(labels, (0.7, 0.1, 0.2), np.random.default_rng(42))
        b = _stratified_split(labels, (0.7, 0.1, 0.2), np.random.default_rng(42))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_invalid_ratios(self) -> None:
        with pytest.raises(ValueError, match="must sum to 1.0"):
            _stratified_split(
                np.array([0, 1, 2]),
                (0.5, 0.5, 0.5),
                np.random.default_rng(0),
            )


class TestBalancedSubset:
    def test_caps_each_stage(self) -> None:
        labels = np.repeat(np.arange(5), 100)
        pool = np.arange(labels.size)
        out = _balanced_subset(labels, pool, per_stage=10, rng=np.random.default_rng(1))
        for stage in range(5):
            assert (labels[out] == stage).sum() == 10

    def test_handles_undersized_stage(self) -> None:
        labels = np.array([0] * 5 + [1] * 100)
        pool = np.arange(labels.size)
        out = _balanced_subset(labels, pool, per_stage=20, rng=np.random.default_rng(0))
        # Stage 0 has only 5 samples; should keep them all.
        assert (labels[out] == 0).sum() == 5
        assert (labels[out] == 1).sum() == 20


class TestOODAttackIndices:
    def test_indices_are_label_exact(self) -> None:
        labels = np.asarray(
            ["BenignTraffic", "DDoS-HTTP_Flood", "Recon-PortScan", "DDoS-HTTP_Flood"]
        )
        out = _ood_attack_indices(labels, ("DDoS-HTTP_Flood",))
        np.testing.assert_array_equal(out["DDoS-HTTP_Flood"], [1, 3])

    def test_missing_class_logs_and_skips(self) -> None:
        labels = np.asarray(["BenignTraffic"])
        out = _ood_attack_indices(labels, ("NeverSeenAttack",))
        assert out == {}


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------


class TestBuildSplitsEndToEnd:
    def test_run_on_synthetic_dataset(self, tmp_path: Path) -> None:
        processed = _build_synthetic_processed_dir(tmp_path, n_per_label=50)
        cfg = BuilderConfig(
            processed_dir=processed,
            seed=42,
            val_balanced_per_stage=5,
            test_balanced_per_stage=10,
            ood_attack_classes=("DDoS-HTTP_Flood",),
        )
        manifest = build_splits(cfg)

        splits_dir = processed / "splits"
        assert (splits_dir / "manifest.json").exists()
        for fname in ("train.idx.npy", "val.idx.npy", "test.idx.npy"):
            assert (splits_dir / fname).exists()

        tr = np.load(splits_dir / "train.idx.npy")
        va = np.load(splits_dir / "val.idx.npy")
        te = np.load(splits_dir / "test.idx.npy")
        n_total = manifest["num_samples"]
        # OOD rows are excluded from train/val/test (detector leakage fix).
        ood_path = splits_dir / "ood_attack" / "DDoS-HTTP_Flood.idx.npy"
        assert ood_path.exists()
        ood_idx = np.load(ood_path)
        assert ood_idx.size == 50
        assert tr.size + va.size + te.size + ood_idx.size == n_total

        # train/val/test partition the in-distribution rows exhaustively.
        union = np.concatenate([tr, va, te])
        assert np.unique(union).size == tr.size + va.size + te.size

        # **Disjointness with OOD** — the regression test that locks the
        # detector leakage fix in place.
        ood_set = set(ood_idx.tolist())
        for split_name, split_idx in (("train", tr), ("val", va), ("test", te)):
            overlap = ood_set.intersection(split_idx.tolist())
            assert not overlap, (
                f"OOD class leaked into {split_name}: {len(overlap)} rows. "
                "If this fires, the dataset-prep build script regressed on the fix."
            )

        # Stage-balanced subsets ⊆ val/test pools
        vb = np.load(splits_dir / "val_balanced.idx.npy")
        tb = np.load(splits_dir / "test_balanced.idx.npy")
        assert np.isin(vb, va).all()
        assert np.isin(tb, te).all()

        # Manifest hashes are deterministic in the inputs
        assert {"features.npy", "labels.npy", "metadata.json"} <= set(
            manifest["inputs"].keys()
        )
        assert "splits/train.idx.npy" in manifest["outputs"]

    def test_determinism_across_runs(self, tmp_path: Path) -> None:
        processed = _build_synthetic_processed_dir(tmp_path, n_per_label=30)
        cfg = BuilderConfig(processed_dir=processed, seed=42, ood_attack_classes=())
        m1 = build_splits(cfg)
        first_train = np.load(processed / "splits" / "train.idx.npy")

        # Wipe outputs and rebuild — same seed → identical content.
        for p in (processed / "splits").iterdir():
            if p.is_file():
                p.unlink()
        m2 = build_splits(cfg)
        second_train = np.load(processed / "splits" / "train.idx.npy")
        np.testing.assert_array_equal(first_train, second_train)
        assert m1["outputs"] == m2["outputs"]
