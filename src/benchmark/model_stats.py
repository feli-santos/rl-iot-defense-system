"""Compute FLOPs, MACs, parameter count, and on-disk size for SB3 policy networks.

This module addresses thesis review issue C13: the IoT-resource motivation
requires reporting model efficiency metrics (FLOPs/MACs, memory footprint)
alongside inference latency. It is wired into Phase-6 evaluation to emit
``results/06_benchmark/model_stats.json``.

Usage::

    from src.benchmark.model_stats import get_model_stats, get_detector_stats
    stats = get_model_stats(model, obs_dim=290)
    # -> {"algo": "dqn", "params": 6789, "macs": 12345, "size_mb": 0.08}

Dependencies:
    - ``thop`` (optional, graceful degradation if missing): pip install thop
    - ``torch``: already a project dependency.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _try_thop(
    module: torch.nn.Module,
    dummy_input: torch.Tensor,
) -> tuple[Optional[float], Optional[float]]:
    """Attempt thop FLOPs/MACs count; return (macs, flops) or (None, None)."""
    try:
        from thop import profile as thop_profile  # type: ignore[import]

        macs, params = thop_profile(module, inputs=(dummy_input,), verbose=False)
        return float(macs), float(params)
    except ImportError:
        logger.debug(
            "thop not installed; skipping MACs count. "
            "Install with: pip install thop"
        )
        return None, None
    except Exception as exc:  # noqa: BLE001
        logger.debug("thop profiling failed: %s", exc)
        return None, None


def _count_params(module: torch.nn.Module) -> int:
    """Return total trainable parameter count."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _disk_size_mb(model: Any) -> Optional[float]:
    """Save the SB3 model to a temp file and return the zip size in MB."""
    try:
        with tempfile.NamedTemporaryFile(suffix="", delete=False) as f:
            tmp_base = f.name
        model.save(tmp_base)
        zip_path = tmp_base + ".zip"
        # SB3 appends .zip automatically
        target = zip_path if Path(zip_path).exists() else tmp_base
        size = os.path.getsize(target) / 1e6
        for p in [tmp_base, zip_path]:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
        return size
    except Exception as exc:  # noqa: BLE001
        logger.debug("disk size computation failed: %s", exc)
        return None


def get_model_stats(
    model: Any,
    obs_dim: int,
    algo: str = "unknown",
) -> Dict[str, Any]:
    """Compute efficiency metrics for an SB3 policy network.

    Args:
        model: A loaded SB3 model (DQN, PPO, or A2C).
        obs_dim: Observation dimension of the environment
            (e.g., 290 = window_size × num_features × 2 for the
            Phase-3 production env with deltas).
        algo: Algorithm name for labelling in the output dict.

    Returns:
        Dict with keys:
            - ``algo``: algorithm name
            - ``params``: number of trainable parameters
            - ``macs``: multiply-accumulate operations (forward pass),
              or ``null`` if thop is unavailable
            - ``size_mb``: on-disk model size in megabytes (SB3 zip),
              or ``null`` if save failed
            - ``obs_dim``: observation dimension used for profiling
    """
    policy_net = model.policy
    policy_net.eval()

    params = _count_params(policy_net)

    # For MACs we only profile the actor/q_net — the part that runs at
    # inference time. For DQN it's q_net; for PPO/A2C it's mlp_extractor
    # + action_net. Use the full policy module for a conservative upper bound.
    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)
    with torch.no_grad():
        macs, _ = _try_thop(policy_net, dummy_obs)

    size_mb = _disk_size_mb(model)

    return {
        "algo": algo,
        "obs_dim": obs_dim,
        "params": params,
        "macs": macs,
        "size_mb": size_mb,
    }


def get_file_stats(path: Path) -> Dict[str, Any]:
    """Return disk size stats for a model saved as a plain file (e.g., joblib).

    Used for the RF / 1D-CNN detectors which are not SB3 models.
    """
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "size_mb": None, "exists": False}
    return {
        "path": str(p),
        "size_mb": p.stat().st_size / 1e6,
        "exists": True,
    }


__all__ = ["get_model_stats", "get_file_stats"]
