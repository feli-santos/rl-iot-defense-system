"""Tests for ``scripts.data.derive_stage_labels``.

Three layers of coverage:

1. Synthetic happy path on a 100-row toy dataset (always runs).
2. Error paths (missing inputs, non-exhaustive, double-assignment).
3. Real-data regression test that the derived ``stages.npy`` agrees with
   ``state_indices.json`` on every row. Skipped in environments where the
   442 K-row processed snapshot is not available (CI, tests on a fresh
   clone).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.data.derive_stage_labels import (
    derive_stage_labels,
    write_outputs,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_dataset(
    tmp_path: Path,
    *,
    num_rows: int = 100,
    num_features: int = 8,
    seed: int = 0,
) -> Path:
    """Tiny exhaustive 5-stage dataset for happy-path tests."""
    data_path = tmp_path / "ds"
    data_path.mkdir()
    rng = np.random.default_rng(seed)
    np.save(
        data_path / "features.npy", rng.standard_normal((num_rows, num_features)).astype(np.float32)
    )

    # Round-robin assignment: row i -> stage i % 5.  Exhaustive and disjoint.
    state_indices: dict[str, list[int]] = {str(s): [] for s in range(5)}
    for i in range(num_rows):
        state_indices[str(i % 5)].append(i)
    (data_path / "state_indices.json").write_text(json.dumps(state_indices))
    return data_path


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_returns_int8_array(tmp_path: Path) -> None:
    data_path = _make_synthetic_dataset(tmp_path)
    stages = derive_stage_labels(data_path)
    assert stages.dtype == np.int8
    assert stages.shape == (100,)
    assert set(np.unique(stages).tolist()) == {0, 1, 2, 3, 4}


def test_happy_path_round_robin_assignment(tmp_path: Path) -> None:
    data_path = _make_synthetic_dataset(tmp_path)
    stages = derive_stage_labels(data_path)
    expected = np.array([i % 5 for i in range(100)], dtype=np.int8)
    np.testing.assert_array_equal(stages, expected)


def test_write_outputs_creates_npy_and_manifest(tmp_path: Path) -> None:
    data_path = _make_synthetic_dataset(tmp_path)
    stages = derive_stage_labels(data_path)
    manifest_path = write_outputs(data_path, stages)
    assert manifest_path.exists()
    assert (data_path / "stages.npy").exists()
    manifest = json.loads(manifest_path.read_text())
    for k in (
        "version",
        "generated_at",
        "inputs",
        "outputs",
        "num_rows",
        "num_stages",
        "stage_counts",
        "stage_dtype",
    ):
        assert k in manifest, f"missing manifest key: {k}"
    assert manifest["num_rows"] == 100
    assert manifest["num_stages"] == 5


def test_write_outputs_is_idempotent(tmp_path: Path) -> None:
    """Running derive + write twice must produce byte-identical files."""
    data_path = _make_synthetic_dataset(tmp_path)
    stages_1 = derive_stage_labels(data_path)
    write_outputs(data_path, stages_1)
    npy_1 = (data_path / "stages.npy").read_bytes()

    # Reload and rewrite — bytes must match.
    stages_2 = derive_stage_labels(data_path)
    write_outputs(data_path, stages_2)
    npy_2 = (data_path / "stages.npy").read_bytes()
    assert npy_1 == npy_2


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_missing_features_raises(tmp_path: Path) -> None:
    data_path = _make_synthetic_dataset(tmp_path)
    (data_path / "features.npy").unlink()
    with pytest.raises(FileNotFoundError):
        derive_stage_labels(data_path)


def test_missing_state_indices_raises(tmp_path: Path) -> None:
    data_path = _make_synthetic_dataset(tmp_path)
    (data_path / "state_indices.json").unlink()
    with pytest.raises(FileNotFoundError):
        derive_stage_labels(data_path)


def test_non_exhaustive_state_indices_raises(tmp_path: Path) -> None:
    data_path = _make_synthetic_dataset(tmp_path)
    # Drop row 0 from stage 0; nothing else picks it up.
    si = {str(s): [] for s in range(5)}
    for i in range(1, 100):
        si[str(i % 5)].append(i)
    (data_path / "state_indices.json").write_text(json.dumps(si))
    with pytest.raises(ValueError, match="not exhaustive"):
        derive_stage_labels(data_path)


def test_double_assignment_state_indices_raises(tmp_path: Path) -> None:
    data_path = _make_synthetic_dataset(tmp_path)
    # Round-robin places row 7 naturally in stage 7 % 5 == 2.  Re-add it
    # under stage 4 so the second pass tries to overwrite a previous label.
    si = json.loads((data_path / "state_indices.json").read_text())
    assert 7 in si["2"], "fixture invariant: row 7 should be in stage 2"
    si["4"].append(7)
    (data_path / "state_indices.json").write_text(json.dumps(si))
    with pytest.raises(ValueError, match="re-assigns"):
        derive_stage_labels(data_path)


def test_wrong_stage_keys_raises(tmp_path: Path) -> None:
    data_path = _make_synthetic_dataset(tmp_path)
    si = json.loads((data_path / "state_indices.json").read_text())
    si["5"] = []  # extra stage id outside 0..4
    (data_path / "state_indices.json").write_text(json.dumps(si))
    with pytest.raises(ValueError, match="must have exactly stages"):
        derive_stage_labels(data_path)


# ---------------------------------------------------------------------------
# Real-data regression (skipped if the snapshot is absent)
# ---------------------------------------------------------------------------


_REAL_DATA = Path("data/processed/ciciot2023")


@pytest.mark.skipif(
    not (_REAL_DATA / "features.npy").exists() or not (_REAL_DATA / "state_indices.json").exists(),
    reason="Real CICIoT processed snapshot not present.",
)
def test_real_dataset_round_trip() -> None:
    """Derived stages.npy must agree with state_indices.json on every row."""
    stages = derive_stage_labels(_REAL_DATA)

    state_indices = {
        int(k): v for k, v in json.loads((_REAL_DATA / "state_indices.json").read_text()).items()
    }
    for stage_id, idx_list in state_indices.items():
        idx = np.asarray(idx_list, dtype=np.int64)
        assert (
            stages[idx] == stage_id
        ).all(), f"Stage {stage_id} disagrees on {(stages[idx] != stage_id).sum()} rows."

    # And the array is exhaustive.
    counts = {int(s): int((stages == s).sum()) for s in range(5)}
    assert sum(counts.values()) == stages.shape[0]
