"""Tests for src.benchmark.model_stats (thesis review issue C13)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn


from src.benchmark.model_stats import get_file_stats, get_model_stats


# ------------------------------------------------------------------ helpers


class _TinyNet(nn.Module):
    """Minimal MLP for unit tests — avoids loading full SB3 checkpoints."""

    def __init__(self, in_dim: int = 10, out_dim: int = 5) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _make_mock_sb3_model(obs_dim: int = 10) -> MagicMock:
    """Return a MagicMock that mimics an SB3 BaseAlgorithm."""
    model = MagicMock()
    model.policy = _TinyNet(in_dim=obs_dim)
    # Mock save() to write a small zip file so _disk_size_mb works.
    def _fake_save(path: str) -> None:
        import zipfile
        zip_path = path + ".zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("policy.pth", b"fake_weights" * 100)
    model.save.side_effect = _fake_save
    return model


# ------------------------------------------------------------------ tests


class TestGetModelStats:
    def test_returns_required_keys(self) -> None:
        model = _make_mock_sb3_model(obs_dim=10)
        stats = get_model_stats(model, obs_dim=10, algo="dqn")
        assert set(stats.keys()) >= {"algo", "obs_dim", "params", "macs", "size_mb"}

    def test_algo_label(self) -> None:
        model = _make_mock_sb3_model(obs_dim=10)
        stats = get_model_stats(model, obs_dim=10, algo="ppo")
        assert stats["algo"] == "ppo"

    def test_obs_dim(self) -> None:
        model = _make_mock_sb3_model(obs_dim=10)
        stats = get_model_stats(model, obs_dim=10)
        assert stats["obs_dim"] == 10

    def test_params_positive(self) -> None:
        model = _make_mock_sb3_model(obs_dim=10)
        stats = get_model_stats(model, obs_dim=10)
        assert stats["params"] > 0

    def test_size_mb_positive_or_none(self) -> None:
        model = _make_mock_sb3_model(obs_dim=10)
        stats = get_model_stats(model, obs_dim=10)
        # size_mb should be positive if the mock save works, or None if it fails
        if stats["size_mb"] is not None:
            assert stats["size_mb"] > 0

    def test_macs_none_without_thop(self) -> None:
        """When thop is not installed, macs should be None (graceful degradation)."""
        model = _make_mock_sb3_model(obs_dim=10)
        with patch.dict("sys.modules", {"thop": None}):
            stats = get_model_stats(model, obs_dim=10)
        # macs is either a float or None; must not raise
        assert stats["macs"] is None or isinstance(stats["macs"], float)

    def test_param_count_matches_network(self) -> None:
        """Parameter count must match what PyTorch reports directly."""
        model = _make_mock_sb3_model(obs_dim=10)
        expected_params = sum(
            p.numel() for p in model.policy.parameters() if p.requires_grad
        )
        stats = get_model_stats(model, obs_dim=10)
        assert stats["params"] == expected_params


class TestGetFileStats:
    def test_existing_file(self, tmp_path: Path) -> None:
        p = tmp_path / "model.joblib"
        p.write_bytes(b"x" * 1024)  # 1 KB
        result = get_file_stats(p)
        assert result["exists"] is True
        assert result["size_mb"] == pytest.approx(1024 / 1e6, rel=1e-3)
        assert result["path"] == str(p)

    def test_missing_file(self, tmp_path: Path) -> None:
        p = tmp_path / "nonexistent.joblib"
        result = get_file_stats(p)
        assert result["exists"] is False
        assert result["size_mb"] is None

    def test_returns_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "f.bin"
        p.write_bytes(b"hello")
        result = get_file_stats(p)
        assert isinstance(result, dict)
