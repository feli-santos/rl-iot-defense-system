"""Tests for the stateless helpers added to ``src.generator.episode_generator``.

These exercise the new public functions (``episodes_to_training_sequences``,
``episodes_to_numpy``, ``stage_distribution_from_split_manifest``) without
constructing an :class:`EpisodeGenerator`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.generator.episode_generator import (
    episodes_to_numpy,
    episodes_to_training_sequences,
    stage_distribution_from_split_manifest,
)


class TestEpisodesToTrainingSequences:
    def test_basic_sliding_window(self) -> None:
        episodes = [[0, 1, 2, 3, 4]]
        seqs, tgts = episodes_to_training_sequences(episodes, sequence_length=2)
        assert seqs == [[0, 1], [1, 2], [2, 3]]
        assert tgts == [2, 3, 4]

    def test_skips_too_short_episodes(self) -> None:
        episodes = [[0, 1], [0, 1, 2, 3]]
        seqs, tgts = episodes_to_training_sequences(episodes, sequence_length=3)
        # First episode skipped (length 2 <= 3); second yields exactly one window.
        assert seqs == [[0, 1, 2]]
        assert tgts == [3]

    def test_empty_inputs(self) -> None:
        assert episodes_to_training_sequences([], sequence_length=3) == ([], [])
        assert episodes_to_training_sequences([[0]], sequence_length=3) == ([], [])


class TestEpisodesToNumpy:
    def test_returns_int64_arrays(self) -> None:
        X, y = episodes_to_numpy([[0, 1, 2, 3]], sequence_length=2)
        assert X.dtype == np.int64
        assert y.dtype == np.int64
        np.testing.assert_array_equal(X, [[0, 1], [1, 2]])
        np.testing.assert_array_equal(y, [2, 3])

    def test_empty_input_returns_zero_shaped_arrays(self) -> None:
        X, y = episodes_to_numpy([], sequence_length=4)
        assert X.shape == (0, 4)
        assert y.shape == (0,)
        assert X.dtype == np.int64
        assert y.dtype == np.int64


class TestStageDistributionFromSplitManifest:
    def _write_manifest(self, tmp_path: Path) -> Path:
        manifest = {
            "version": 1,
            "stage_counts": {
                "all": {"0": 1000, "1": 500, "2": 300, "3": 600, "4": 1900},
                "train": {"0": 700, "1": 350, "2": 210, "3": 420, "4": 1330},
                "val": {"0": 100, "1": 50, "2": 30, "3": 60, "4": 190},
                "test": {"0": 200, "1": 100, "2": 60, "3": 120, "4": 380},
            },
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path

    def test_loads_train_split(self, tmp_path: Path) -> None:
        path = self._write_manifest(tmp_path)
        dist = stage_distribution_from_split_manifest(path, split_name="train")
        assert dist == {0: 700, 1: 350, 2: 210, 3: 420, 4: 1330}

    def test_loads_other_splits(self, tmp_path: Path) -> None:
        path = self._write_manifest(tmp_path)
        for split in ("all", "val", "test"):
            dist = stage_distribution_from_split_manifest(path, split_name=split)
            assert set(dist.keys()) == {0, 1, 2, 3, 4}
            assert all(isinstance(v, int) and v > 0 for v in dist.values())

    def test_missing_split_raises(self, tmp_path: Path) -> None:
        path = self._write_manifest(tmp_path)
        with pytest.raises(KeyError, match="not found in stage_counts"):
            stage_distribution_from_split_manifest(path, split_name="holdout")
