#!/usr/bin/env python3
"""Compute the deployable model footprint (policy vs. RF detector) as canonical JSON.

Motivation
----------
The IoT-edge argument in the paper claims the windowed DRL policy is small enough
to deploy on a resource-constrained node while the tuned Random-Forest stage
detector is not. That claim must be *measured*, not asserted. This script loads
the canonical headline checkpoints and the tuned RF and writes a canonical
summary + hash-chain manifest so the numbers flow into the thesis/paper macros
via ``scripts/thesis/render_tables.py`` (never hand-typed).

Outputs
-------
- ``docs/results/benchmark/model_footprint.json`` : canonical footprint summary
- ``docs/results/benchmark/model_footprint_manifest.json`` : git SHA + input/output
  SHA-256 hash chain (reproducibility contract)

The "deployable" policy footprint counts only the inference pathway (drops the
PPO/A2C value head and the DQN target network); see
``src.benchmark.model_stats.count_deployable_params``. A 2x64 ReLU MLP over a
290-dim observation with 5 actions has a closed-form forward cost, reported as
multiply-accumulate operations (MACs).

Usage::

    PYTHONPATH=. .venv/bin/python -m scripts.benchmark.compute_model_footprint
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import joblib
from stable_baselines3 import A2C, DQN, PPO

from src.benchmark.model_stats import count_deployable_params, get_file_stats

# ---------------------------------------------------------------------------
# Canonical inputs (headline deterministic-5M regime, alpha=0.4, seed 0)
# ---------------------------------------------------------------------------

OBS_DIM = 290
NUM_ACTIONS = 5
HIDDEN = (64, 64)

_ALGO_CLASSES = {"ppo": PPO, "a2c": A2C, "dqn": DQN}

CHECKPOINTS = {
    "ppo": Path("runs/redesign_5M_det/alpha_04/ppo/seed_0/best_model.zip"),
    "a2c": Path("runs/redesign_5M_det/alpha_04/a2c/seed_0/best_model.zip"),
    "dqn": Path("runs/redesign_5M_det/alpha_04/dqn/seed_0/best_model.zip"),
}
RF_PATH = Path("artifacts/detector/tuned/random_forest_tuned.joblib")

OUT_JSON = Path("docs/results/benchmark/model_footprint.json")
OUT_MANIFEST = Path("docs/results/benchmark/model_footprint_manifest.json")

FP32_BYTES = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _mlp_forward_macs() -> int:
    """Closed-form multiply-accumulates for one forward pass of the 2x64 policy MLP.

    Linear layer MACs = in_features * out_features (bias adds negligible adds).
    Layers: 290->64, 64->64, 64->5 (action logits). ReLU is MAC-free.
    """
    dims = [OBS_DIM, *HIDDEN, NUM_ACTIONS]
    return sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))


def _rf_node_stats(rf_obj: Any) -> dict[str, int]:
    """Total decision-tree nodes/leaves/internal-splits across the forest."""
    est = rf_obj
    try:
        from sklearn.pipeline import Pipeline

        if isinstance(rf_obj, Pipeline):
            est = rf_obj.steps[-1][1]
    except Exception:  # noqa: BLE001
        pass
    trees = est.estimators_
    total_nodes = int(sum(t.tree_.node_count for t in trees))
    total_leaves = int(sum(t.tree_.n_leaves for t in trees))
    return {
        "n_trees": int(len(trees)),
        "total_nodes": total_nodes,
        "total_leaves": total_leaves,
        "internal_nodes": total_nodes - total_leaves,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_footprint() -> dict[str, Any]:
    # --- Deployable DRL policy (identical arch across algos; verify per algo) ---
    per_algo: dict[str, Any] = {}
    for algo, path in CHECKPOINTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        model = _ALGO_CLASSES[algo].load(str(path), env=None, device="cpu")
        deployable = count_deployable_params(model, algo)
        per_algo[algo] = {
            "deployable_params": deployable,
            "deployable_kb_fp32": round(deployable * FP32_BYTES / 1024, 1),
            "checkpoint_zip_mb": round(get_file_stats(path)["size_mb"], 2),
        }

    # The three share one architecture; report the common footprint as headline.
    deployable_params = per_algo["ppo"]["deployable_params"]
    if not all(v["deployable_params"] == deployable_params for v in per_algo.values()):
        # Not fatal, but the paper quotes a single number; surface the spread.
        deployable_params = max(v["deployable_params"] for v in per_algo.values())

    policy = {
        "architecture": f"{HIDDEN[0]}x{HIDDEN[1]} ReLU MLP",
        "obs_dim": OBS_DIM,
        "num_actions": NUM_ACTIONS,
        "deployable_params": deployable_params,
        "deployable_kb_fp32": round(deployable_params * FP32_BYTES / 1024, 1),
        "forward_macs": _mlp_forward_macs(),
        "per_algo": per_algo,
    }

    # --- Tuned RF stage detector ---
    if not RF_PATH.exists():
        raise FileNotFoundError(f"RF detector not found: {RF_PATH}")
    rf_obj = joblib.load(RF_PATH)
    rf_stats = _rf_node_stats(rf_obj)
    rf_size_mb = round(get_file_stats(RF_PATH)["size_mb"], 1)
    rf = {
        "size_mb_on_disk": rf_size_mb,
        **rf_stats,
    }

    # --- Contrast (edge-deployment argument) ---
    policy_bytes = deployable_params * FP32_BYTES
    rf_bytes = RF_PATH.stat().st_size
    ratio = int(round(rf_bytes / policy_bytes))

    return {
        "schema_version": "1.0",
        "kind": "model_footprint",
        "description": (
            "Deployable inference footprint of the headline DRL policy "
            "(alpha=0.4, seed 0) vs. the tuned Random-Forest stage detector."
        ),
        "policy": policy,
        "rf_detector": rf,
        "contrast": {
            "policy_kb_fp32": policy["deployable_kb_fp32"],
            "rf_mb_on_disk": rf_size_mb,
            "rf_over_policy_size_ratio": ratio,
        },
    }


def main() -> int:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    summary = build_footprint()
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    manifest = {
        "schema_version": "1.0",
        "kind": "model_footprint_manifest",
        "git_sha": _git_sha(),
        "inputs": {
            **{f"checkpoint_{algo}": _sha256(p) for algo, p in CHECKPOINTS.items()},
            "rf_detector": _sha256(RF_PATH),
        },
        "outputs": {"summary_json": _sha256(OUT_JSON)},
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MANIFEST}")
    print(
        f"  policy: {summary['policy']['deployable_params']} params "
        f"= {summary['policy']['deployable_kb_fp32']} KB (fp32); "
        f"RF: {summary['rf_detector']['size_mb_on_disk']} MB; "
        f"ratio ~{summary['contrast']['rf_over_policy_size_ratio']}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
