"""Derive a frozen ``stages.npy`` from ``state_indices.json``.

detector step 4.2 (see ``docs/results/stage-detector/PLAN.md`` §A2).

The dataset-prep dataset already groups every row by Kill Chain stage in
``state_indices.json``. For detector we want a flat ``(N,)`` int8 array of
stage labels so both supervised baselines (StageDetector MLP and RF)
train on the *same* per-row labels with O(1) lookup. Building it
once and hash-pinning the output prevents a class of "did we use the
same labels?" bugs from ever showing up downstream.

Usage
-----
    python -m scripts.data.derive_stage_labels [--data-path PATH]

Outputs (next to ``features.npy``):
    stages.npy            — int8 array of shape (N,), values in {0..4}
    stages.manifest.json  — provenance: SHA-256 of stages.npy + inputs.

The script is **idempotent**: running it twice produces byte-identical
files (the array is built deterministically from the JSON, no RNG).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def derive_stage_labels(data_path: Path) -> np.ndarray:
    """Build the per-row stage label vector from ``state_indices.json``.

    Args:
        data_path: Directory containing ``features.npy`` and
            ``state_indices.json``.

    Returns:
        Int8 NumPy array of shape ``(num_rows,)`` with values in ``{0..4}``.

    Raises:
        FileNotFoundError: If either input is missing.
        ValueError: If the per-stage index lists are not exhaustive (some
            row not assigned) or not disjoint (some row assigned twice).
    """
    data_path = Path(data_path)
    features_path = data_path / "features.npy"
    indices_path = data_path / "state_indices.json"

    if not features_path.exists():
        raise FileNotFoundError(features_path)
    if not indices_path.exists():
        raise FileNotFoundError(indices_path)

    # We don't need to load the full features array — just its row count.
    # ``np.load`` with mmap is O(1) and avoids reading 442 K * 29 floats.
    num_rows = int(np.load(features_path, mmap_mode="r").shape[0])

    state_indices: dict[int, list[int]] = {
        int(k): v for k, v in json.loads(indices_path.read_text()).items()
    }
    expected_stages = {0, 1, 2, 3, 4}
    if set(state_indices.keys()) != expected_stages:
        raise ValueError(
            f"state_indices.json must have exactly stages {sorted(expected_stages)}, "
            f"got {sorted(state_indices.keys())}"
        )

    # Sentinel value -1 lets us detect both unassigned (still -1) and
    # double-assigned (overwrite to a different stage on a second pass) rows.
    stages = np.full(num_rows, fill_value=-1, dtype=np.int8)
    for stage_id in sorted(state_indices.keys()):
        idx = np.asarray(state_indices[stage_id], dtype=np.int64)
        if idx.size == 0:
            logger.warning("Stage %d has zero rows — empty per-stage list.", stage_id)
            continue
        already_assigned = stages[idx] != -1
        if already_assigned.any():
            offenders = idx[already_assigned][:5].tolist()
            raise ValueError(
                f"Stage {stage_id} re-assigns rows already labeled in another "
                f"stage. First offenders: {offenders}"
            )
        stages[idx] = stage_id

    if (stages == -1).any():
        unassigned = int((stages == -1).sum())
        first = np.where(stages == -1)[0][:5].tolist()
        raise ValueError(
            f"state_indices.json is not exhaustive: {unassigned} rows have no "
            f"stage assignment. First offenders: {first}"
        )

    return stages


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_outputs(data_path: Path, stages: np.ndarray) -> Path:
    """Write ``stages.npy`` + ``stages.manifest.json`` next to the dataset.

    Returns the path to the manifest.
    """
    out_npy = data_path / "stages.npy"
    out_manifest = data_path / "stages.manifest.json"

    np.save(out_npy, stages, allow_pickle=False)
    counts: dict[int, int] = {
        int(s): int((stages == s).sum()) for s in sorted(np.unique(stages))
    }

    manifest = {
        "version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "features.npy": _sha256(data_path / "features.npy"),
            "state_indices.json": _sha256(data_path / "state_indices.json"),
        },
        "outputs": {
            "stages.npy": _sha256(out_npy),
        },
        "num_rows": int(stages.shape[0]),
        "num_stages": int(len(counts)),
        "stage_counts": counts,
        "stage_dtype": str(stages.dtype),
    }
    out_manifest.write_text(json.dumps(manifest, indent=2))
    return out_manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/ciciot2023"),
        help="Directory containing features.npy and state_indices.json.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging."
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    stages = derive_stage_labels(args.data_path)
    manifest_path = write_outputs(args.data_path, stages)

    counts = {int(s): int((stages == s).sum()) for s in sorted(np.unique(stages))}
    total = sum(counts.values())
    logger.info("Wrote %s (%d rows)", args.data_path / "stages.npy", total)
    logger.info("Per-stage counts: %s", counts)
    logger.info("Manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
