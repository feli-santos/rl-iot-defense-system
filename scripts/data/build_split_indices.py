"""
Build immutable split indices and a hash manifest for the processed CICIoT2023
dataset.

This script does NOT re-process the raw 47 M-row dataset. It operates on
``data/processed/ciciot2023/`` (produced by ``main.py --mode process-data``)
and writes:

- ``data/processed/ciciot2023/splits/train.idx.npy``
- ``data/processed/ciciot2023/splits/val.idx.npy``
- ``data/processed/ciciot2023/splits/test.idx.npy``
- ``data/processed/ciciot2023/splits/val_balanced.idx.npy``
- ``data/processed/ciciot2023/splits/test_balanced.idx.npy``
- ``data/processed/ciciot2023/splits/ood_attack/<class>.idx.npy``
- ``data/processed/ciciot2023/splits/manifest.json``

Splits are stratified by Kill Chain stage with a fixed seed (42) so they are
deterministic. Balanced splits draw a fixed number of samples per stage with
the same seed.

The hash manifest records SHA-256 digests for every input artifact and every
output index file, anchoring downstream phases (Red Team training, RL
training, benchmarking) to an immutable data state.

Usage
-----
    python -m scripts.data.build_split_indices \
        --processed-dir data/processed/ciciot2023 \
        --seed 42

Or via the Makefile (added later):
    make build-split-indices
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# Make ``src`` importable when this script is invoked via ``python -m`` or
# directly. The Makefile target sets PYTHONPATH=., but we add a fallback for
# direct invocation.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.label_mapper import AbstractStateLabelMapper  # noqa: E402

# -----------------------------------------------------------------------------
# Configuration & logging
# -----------------------------------------------------------------------------

LOG = logging.getLogger("build_split_indices")
NUM_STAGES = 5
DEFAULT_RATIOS = (0.7, 0.1, 0.2)  # train, val, test (stratified)
DEFAULT_VAL_BALANCED_PER_STAGE = 200
DEFAULT_TEST_BALANCED_PER_STAGE = 1000
# Held-out CICIoT2023 classes for zero-day (OOD) evaluation. We hold out two
# classes per upper kill-chain stage (RECON, ACCESS, MANEUVER) and four for
# IMPACT, spanning the detector's recall spectrum from near-perfect to the
# VulnerabilityScan structural blind spot (recall ~0.001). These rows are
# removed from the data BEFORE the stratified train/val/test split, so they
# never appear in training, the detector, or the in-distribution benchmark --
# the correct zero-day protocol. Each upper stage retains >=3 in-distribution
# classes and >=26k training rows to avoid starving any stage.
DEFAULT_OOD_CLASSES = (
    # RECON
    "VulnerabilityScan",        # RECON    (RF detector blind spot, recall ~0.001)
    "Recon-OSScan",             # RECON
    # ACCESS
    "XSS",                      # ACCESS
    "SqlInjection",             # ACCESS
    # MANEUVER
    "Mirai-udpplain",           # MANEUVER
    "DNS_Spoofing",             # MANEUVER
    # IMPACT
    "DDoS-HTTP_Flood",          # IMPACT
    "DoS-SYN_Flood",            # IMPACT
    "DDoS-SlowLoris",           # IMPACT
    "DDoS-ACK_Fragmentation",   # IMPACT
)


@dataclass(frozen=True)
class BuilderConfig:
    """Configuration for the split-index builder."""

    processed_dir: Path
    seed: int = 42
    train_val_test_ratios: tuple[float, float, float] = DEFAULT_RATIOS
    val_balanced_per_stage: int = DEFAULT_VAL_BALANCED_PER_STAGE
    test_balanced_per_stage: int = DEFAULT_TEST_BALANCED_PER_STAGE
    ood_attack_classes: tuple[str, ...] = DEFAULT_OOD_CLASSES


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _stratified_split(
    labels: np.ndarray,
    ratios: tuple[float, float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test split by integer label.

    Indices within each label class are shuffled deterministically using *rng*
    and then divided according to *ratios*. The split is exhaustive and
    disjoint.
    """
    train_ratio, val_ratio, test_ratio = ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("ratios must sum to 1.0")

    train_idx: list[np.ndarray] = []
    val_idx: list[np.ndarray] = []
    test_idx: list[np.ndarray] = []

    for stage in range(NUM_STAGES):
        stage_indices = np.flatnonzero(labels == stage)
        if stage_indices.size == 0:
            LOG.warning("Stage %d has no samples; skipping in stratified split", stage)
            continue
        rng.shuffle(stage_indices)
        n = stage_indices.size
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Anything left goes to test (avoid rounding-induced gaps).
        train_idx.append(stage_indices[:n_train])
        val_idx.append(stage_indices[n_train : n_train + n_val])
        test_idx.append(stage_indices[n_train + n_val :])

    return (
        np.concatenate(train_idx).astype(np.int64),
        np.concatenate(val_idx).astype(np.int64),
        np.concatenate(test_idx).astype(np.int64),
    )


def _balanced_subset(
    labels: np.ndarray,
    pool_indices: np.ndarray,
    per_stage: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample *per_stage* indices for each Kill-Chain stage from *pool_indices*."""
    chosen: list[np.ndarray] = []
    pool_labels = labels[pool_indices]
    for stage in range(NUM_STAGES):
        stage_pool = pool_indices[pool_labels == stage]
        if stage_pool.size == 0:
            LOG.warning("Stage %d unrepresented in pool; balanced subset will skip it", stage)
            continue
        if stage_pool.size <= per_stage:
            chosen.append(stage_pool)
            continue
        picks = rng.choice(stage_pool, size=per_stage, replace=False)
        chosen.append(np.sort(picks))
    return np.concatenate(chosen).astype(np.int64)


def _ood_attack_indices(
    string_labels: np.ndarray,
    ood_classes: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Return ``{class_name: indices_in_processed_array}`` for each OOD class.

    Resolved directly from the string-label array (``labels.npy`` stores the
    original CICIoT2023 attack label, e.g. ``"DDoS-HTTP_Flood"``). This is
    label-exact and stable: it does not depend on the processor's internal
    ordering.
    """
    out: dict[str, np.ndarray] = {}
    for cls in ood_classes:
        idx = np.flatnonzero(string_labels == cls).astype(np.int64)
        if idx.size == 0:
            LOG.warning("OOD class %s not present in labels.npy; skipping", cls)
            continue
        out[cls] = idx
        LOG.info("OOD class %s -> %d rows", cls, idx.size)
    return out


def _string_to_stage_ids(
    string_labels: np.ndarray, mapper: AbstractStateLabelMapper
) -> np.ndarray:
    """Vectorize-map string CICIoT labels to integer Kill Chain stage IDs."""
    # Build a lookup over the *unique* label vocabulary present in the data.
    unique = np.unique(string_labels)
    table: dict[str, int] = {}
    missing: list[str] = []
    for lbl in unique:
        try:
            table[str(lbl)] = mapper.get_stage_id(str(lbl))
        except KeyError:
            missing.append(str(lbl))
    if missing:
        raise KeyError(
            f"{len(missing)} label(s) in labels.npy are not in the "
            f"AbstractStateLabelMapper: {missing}. "
            "Update src/utils/label_mapper.py if these are new attack classes."
        )
    # np.vectorize on the lookup is O(N) but creates a temporary unicode
    # array; for our 442k rows this takes <0.5 s, so it is fine.
    out = np.empty(string_labels.shape[0], dtype=np.int64)
    for lbl, stage in table.items():
        out[string_labels == lbl] = stage
    return out


# -----------------------------------------------------------------------------
# Main builder
# -----------------------------------------------------------------------------

def build_splits(cfg: BuilderConfig) -> dict:
    """Build all split indices and the hash manifest."""
    processed = cfg.processed_dir
    if not processed.is_dir():
        raise FileNotFoundError(f"processed dir not found: {processed}")

    features_path = processed / "features.npy"
    labels_path = processed / "labels.npy"
    metadata_path = processed / "metadata.json"
    scaler_path = processed / "scaler.joblib"
    state_indices_path = processed / "state_indices.json"

    for required in (features_path, labels_path, metadata_path):
        if not required.exists():
            raise FileNotFoundError(f"missing required artifact: {required}")

    LOG.info("Loading metadata from %s", metadata_path)
    with metadata_path.open("r") as fp:
        metadata = json.load(fp)
    metadata["__state_indices_path"] = str(state_indices_path)

    LOG.info("Loading labels from %s", labels_path)
    string_labels = np.asarray(np.load(labels_path, allow_pickle=False))
    if string_labels.ndim != 1:
        string_labels = string_labels.ravel()
    n_total = string_labels.shape[0]
    LOG.info("Total samples: %d", n_total)

    # Map original CICIoT string labels -> integer Kill Chain stage IDs.
    mapper = AbstractStateLabelMapper()
    stage_ids = _string_to_stage_ids(string_labels, mapper)
    stage_dist = Counter(int(x) for x in stage_ids.tolist())
    LOG.info(
        "Stage distribution: %s",
        {s: stage_dist.get(s, 0) for s in range(NUM_STAGES)},
    )

    # ---- OOD-attack indices: computed FIRST so we can exclude them from
    # train/val/test before splitting. The held-out attack classes must
    # never appear in any in-distribution split — otherwise the detector
    # silently trains on them and ablation OOD evaluation is meaningless.
    # (Bug discovered in detector step 4.5 by the train_detector.py
    # leakage check; documented in `docs/results/stage-detector/RESULTS.md`.)
    ood_indices = _ood_attack_indices(
        string_labels=string_labels, ood_classes=cfg.ood_attack_classes
    )
    ood_mask = np.zeros(n_total, dtype=bool)
    for cls, idx in ood_indices.items():
        ood_mask[idx] = True
    LOG.info(
        "OOD held-out: %d rows across %d classes %s",
        int(ood_mask.sum()),
        len(ood_indices),
        list(ood_indices.keys()),
    )

    # In-distribution stage labels: -1 sentinel for OOD rows so the
    # stratified splitter ignores them. We then materialise an index
    # array of *only* the in-distribution rows.
    in_dist_idx = np.flatnonzero(~ood_mask).astype(np.int64)
    in_dist_stage_ids = stage_ids[in_dist_idx]
    LOG.info(
        "In-distribution rows for splitting: %d (was %d before OOD removal)",
        in_dist_idx.size,
        n_total,
    )

    # Deterministic stratified split on the in-distribution rows only.
    rng_main = np.random.default_rng(cfg.seed)
    rel_train, rel_val, rel_test = _stratified_split(
        labels=in_dist_stage_ids,
        ratios=cfg.train_val_test_ratios,
        rng=rng_main,
    )
    # Map the relative indices back into absolute row indices.
    train_idx = in_dist_idx[rel_train]
    val_idx = in_dist_idx[rel_val]
    test_idx = in_dist_idx[rel_test]
    LOG.info(
        "Stratified split: train=%d, val=%d, test=%d (sum=%d)",
        train_idx.size, val_idx.size, test_idx.size,
        train_idx.size + val_idx.size + test_idx.size,
    )

    # Balanced subsets drawn from the (already disjoint and OOD-free)
    # val / test pools.
    rng_bal = np.random.default_rng(cfg.seed + 1)
    val_balanced = _balanced_subset(
        labels=stage_ids,
        pool_indices=val_idx,
        per_stage=cfg.val_balanced_per_stage,
        rng=rng_bal,
    )
    test_balanced = _balanced_subset(
        labels=stage_ids,
        pool_indices=test_idx,
        per_stage=cfg.test_balanced_per_stage,
        rng=rng_bal,
    )

    # -------------------------------------------------------------------------
    # Persist outputs
    # -------------------------------------------------------------------------

    out_dir = processed / "splits"
    ood_dir = out_dir / "ood_attack"
    out_dir.mkdir(parents=True, exist_ok=True)
    ood_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "train.idx.npy", train_idx)
    np.save(out_dir / "val.idx.npy", val_idx)
    np.save(out_dir / "test.idx.npy", test_idx)
    np.save(out_dir / "val_balanced.idx.npy", val_balanced)
    np.save(out_dir / "test_balanced.idx.npy", test_balanced)
    for cls, idx in ood_indices.items():
        np.save(ood_dir / f"{cls}.idx.npy", idx)

    # -------------------------------------------------------------------------
    # Hash manifest
    # -------------------------------------------------------------------------

    def _per_stage_counts(idx: np.ndarray) -> dict[int, int]:
        c = Counter(int(x) for x in stage_ids[idx].tolist())
        return {s: int(c.get(s, 0)) for s in range(NUM_STAGES)}

    def _per_class_counts(idx: np.ndarray) -> dict[str, int]:
        c = Counter(string_labels[idx].tolist())
        return {str(k): int(v) for k, v in sorted(c.items())}

    manifest: dict = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": cfg.seed,
        "ratios": list(cfg.train_val_test_ratios),
        "num_samples": int(n_total),
        "num_stages": NUM_STAGES,
        "inputs": {
            "features.npy": _sha256(features_path),
            "labels.npy": _sha256(labels_path),
            "metadata.json": _sha256(metadata_path),
        },
        "outputs": {},
        "stage_counts": {
            "all": _per_stage_counts(np.arange(n_total)),
            "train": _per_stage_counts(train_idx),
            "val": _per_stage_counts(val_idx),
            "test": _per_stage_counts(test_idx),
            "val_balanced": _per_stage_counts(val_balanced),
            "test_balanced": _per_stage_counts(test_balanced),
        },
        "ood_attack_classes": list(ood_indices.keys()),
        "ood_attack_sizes": {cls: int(idx.size) for cls, idx in ood_indices.items()},
    }
    if scaler_path.exists():
        manifest["inputs"]["scaler.joblib"] = _sha256(scaler_path)
    if state_indices_path.exists():
        manifest["inputs"]["state_indices.json"] = _sha256(state_indices_path)

    for split_name in ("train", "val", "test", "val_balanced", "test_balanced"):
        path = out_dir / f"{split_name}.idx.npy"
        manifest["outputs"][f"splits/{split_name}.idx.npy"] = _sha256(path)
    for cls in ood_indices:
        path = ood_dir / f"{cls}.idx.npy"
        manifest["outputs"][f"splits/ood_attack/{cls}.idx.npy"] = _sha256(path)

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w") as fp:
        json.dump(manifest, fp, indent=2, sort_keys=True)
    LOG.info("Wrote manifest -> %s", manifest_path)

    return manifest


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/ciciot2023"),
        help="Directory containing features.npy/labels.npy/metadata.json.",
    )
    p.add_argument("--seed", type=int, default=42, help="Master seed (default: 42).")
    p.add_argument(
        "--val-balanced-per-stage",
        type=int,
        default=DEFAULT_VAL_BALANCED_PER_STAGE,
        help=f"Samples per stage in the balanced val split (default: {DEFAULT_VAL_BALANCED_PER_STAGE}).",
    )
    p.add_argument(
        "--test-balanced-per-stage",
        type=int,
        default=DEFAULT_TEST_BALANCED_PER_STAGE,
        help=f"Samples per stage in the balanced test split (default: {DEFAULT_TEST_BALANCED_PER_STAGE}).",
    )
    p.add_argument(
        "--ood-classes",
        nargs="*",
        default=list(DEFAULT_OOD_CLASSES),
        help="Original CICIoT2023 attack classes to hold out for OOD evaluation.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )
    cfg = BuilderConfig(
        processed_dir=args.processed_dir,
        seed=args.seed,
        val_balanced_per_stage=args.val_balanced_per_stage,
        test_balanced_per_stage=args.test_balanced_per_stage,
        ood_attack_classes=tuple(args.ood_classes),
    )
    manifest = build_splits(cfg)
    print(json.dumps({k: v for k, v in manifest.items() if k != "outputs"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
