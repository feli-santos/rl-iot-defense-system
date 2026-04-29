"""Tests for src/blue_team/env_factory.py (3.2.3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.blue_team.env_factory import make_eval_env, make_train_env
from src.blue_team.run_config import EnvConfigSerializable
from src.environment.adversarial_env import AdversarialIoTEnv
from src.generator.attack_sequence_generator import (
    AttackSequenceGenerator,
    AttackSequenceGeneratorConfig,
)


# --------------------------------------------------------------- fixtures


@pytest.fixture
def synthetic_paths(tmp_path: Path) -> Tuple[Path, Path]:
    """Create a synthetic generator + tiny dataset, no Phase-1 manifest."""
    # Generator
    gen_dir = tmp_path / "generator"
    gen_dir.mkdir(parents=True)
    cfg = AttackSequenceGeneratorConfig(
        num_stages=5, embedding_dim=8, hidden_size=16, num_layers=1
    )
    gen = AttackSequenceGenerator(config=cfg)
    gen.save(gen_dir / "attack_sequence_generator.pth", save_config=True)

    # Dataset
    ds_dir = tmp_path / "dataset"
    ds_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    features = rng.standard_normal((100, 29)).astype(np.float32)
    np.save(ds_dir / "features.npy", features)
    labels = rng.integers(0, 5, size=100)
    np.save(ds_dir / "labels.npy", labels)
    state_indices = {str(i): [] for i in range(5)}
    for idx, lab in enumerate(labels):
        state_indices[str(int(lab))].append(int(idx))
    (ds_dir / "state_indices.json").write_text(json.dumps(state_indices))
    scaler = StandardScaler().fit(features)
    joblib.dump(scaler, ds_dir / "scaler.joblib")

    return gen_dir, ds_dir


@pytest.fixture
def synthetic_manifest(synthetic_paths: Tuple[Path, Path]) -> Path:
    """Build a Phase-1-shape splits manifest pointing at the synthetic dataset.

    The manifest has the same on-disk layout the production
    ``build_split_indices.py`` produces: ``splits/<name>.idx.npy`` and
    ``splits/manifest.json``. We split the 100 rows 70/15/15 and reserve
    a small OOD slice.
    """
    _, ds_dir = synthetic_paths
    splits = ds_dir / "splits"
    splits.mkdir(parents=True)

    rng = np.random.default_rng(1)
    perm = rng.permutation(100)
    train_idx = perm[:70]
    val_idx = perm[70:85]
    test_idx = perm[85:95]
    ood_idx = perm[95:100]

    np.save(splits / "train.idx.npy", train_idx.astype(np.int64))
    np.save(splits / "val.idx.npy", val_idx.astype(np.int64))
    np.save(splits / "val_balanced.idx.npy", val_idx.astype(np.int64))
    np.save(splits / "test.idx.npy", test_idx.astype(np.int64))

    ood_dir = splits / "ood_attack"
    ood_dir.mkdir(parents=True)
    np.save(ood_dir / "synthetic_ood.idx.npy", ood_idx.astype(np.int64))

    manifest_outputs = {
        "splits/train.idx.npy": "x" * 64,
        "splits/val.idx.npy": "x" * 64,
        "splits/val_balanced.idx.npy": "x" * 64,
        "splits/test.idx.npy": "x" * 64,
        "splits/ood_attack/synthetic_ood.idx.npy": "x" * 64,
    }
    manifest_path = splits / "manifest.json"
    manifest_path.write_text(
        json.dumps({"version": "1.0", "outputs": manifest_outputs})
    )
    return manifest_path


# --------------------------------------------------------------- tests


class TestMakeTrainEnv:
    def test_returns_dummy_vec_env_with_correct_obs_space(
        self, synthetic_paths: Tuple[Path, Path]
    ) -> None:
        gen_dir, ds_dir = synthetic_paths
        spec = EnvConfigSerializable(
            split="train", exclude_ood=False,
            min_episode_length=5, max_steps=20,
            window_size=4, include_deltas=True,
        )
        vec = make_train_env(
            spec=spec, generator_path=gen_dir, dataset_path=ds_dir,
            splits_manifest=None, seed=0,
        )
        assert vec.num_envs == 1
        # 4 (window) * 29 (features) * 2 (deltas) = 232
        assert vec.observation_space.shape == (4 * 29 * 2,)
        assert vec.action_space.n == 5

    def test_obs_space_no_deltas(
        self, synthetic_paths: Tuple[Path, Path]
    ) -> None:
        gen_dir, ds_dir = synthetic_paths
        spec = EnvConfigSerializable(
            split="train", include_deltas=False, window_size=3,
        )
        vec = make_train_env(
            spec=spec, generator_path=gen_dir, dataset_path=ds_dir,
            splits_manifest=None, seed=0,
        )
        # 3 * 29 * 1 = 87
        assert vec.observation_space.shape == (3 * 29,)

    def test_step_with_random_action(
        self, synthetic_paths: Tuple[Path, Path]
    ) -> None:
        gen_dir, ds_dir = synthetic_paths
        spec = EnvConfigSerializable(
            split="train", min_episode_length=5, max_steps=10,
        )
        vec = make_train_env(
            spec=spec, generator_path=gen_dir, dataset_path=ds_dir,
            splits_manifest=None, seed=0,
        )
        obs = vec.reset()
        for _ in range(3):
            actions = np.array([vec.action_space.sample()])
            obs, rewards, dones, infos = vec.step(actions)
            assert obs.shape[0] == 1
            assert rewards.shape == (1,)


class TestSplitAwareManifest:
    def test_train_engine_only_sees_in_distribution_indices(
        self,
        synthetic_paths: Tuple[Path, Path],
        synthetic_manifest: Path,
    ) -> None:
        gen_dir, ds_dir = synthetic_paths
        spec = EnvConfigSerializable(split="train", exclude_ood=True,
                                     window_size=4, max_steps=10,
                                     min_episode_length=4)
        vec = make_train_env(
            spec=spec, generator_path=gen_dir, dataset_path=ds_dir,
            splits_manifest=synthetic_manifest, seed=0,
        )
        # Reach into the underlying env to verify the engine got
        # restricted. DummyVecEnv.envs[0] is the Monitor wrapper;
        # Monitor.env is the AdversarialIoTEnv.
        wrapped = vec.envs[0].env
        assert isinstance(wrapped, AdversarialIoTEnv)
        engine = wrapped._realization_engine
        train_idx = set(np.load(
            synthetic_manifest.parent / "train.idx.npy"
        ).tolist())
        ood_idx = set(np.load(
            synthetic_manifest.parent / "ood_attack" / "synthetic_ood.idx.npy"
        ).tolist())
        # Every index the engine still considers must be in train and
        # NOT in ood.
        for stage_id, idx_list in engine._state_indices.items():
            for idx in idx_list:
                assert idx in train_idx
                assert idx not in ood_idx

    def test_eval_split_is_val_balanced(
        self,
        synthetic_paths: Tuple[Path, Path],
        synthetic_manifest: Path,
    ) -> None:
        gen_dir, ds_dir = synthetic_paths
        spec = EnvConfigSerializable(split="val_balanced", exclude_ood=True,
                                     window_size=4, max_steps=10,
                                     min_episode_length=4)
        vec = make_eval_env(
            spec=spec, generator_path=gen_dir, dataset_path=ds_dir,
            splits_manifest=synthetic_manifest, seed=0,
        )
        wrapped = vec.envs[0].env
        engine = wrapped._realization_engine
        val_idx = set(np.load(
            synthetic_manifest.parent / "val_balanced.idx.npy"
        ).tolist())
        for idx_list in engine._state_indices.values():
            for idx in idx_list:
                assert idx in val_idx

    def test_train_eval_pools_disjoint(
        self,
        synthetic_paths: Tuple[Path, Path],
        synthetic_manifest: Path,
    ) -> None:
        """The Phase-3 R2 invariant: training never sees eval rows."""
        gen_dir, ds_dir = synthetic_paths
        train_vec = make_train_env(
            spec=EnvConfigSerializable(split="train", exclude_ood=True,
                                       window_size=4, max_steps=10,
                                       min_episode_length=4),
            generator_path=gen_dir, dataset_path=ds_dir,
            splits_manifest=synthetic_manifest, seed=0,
        )
        eval_vec = make_eval_env(
            spec=EnvConfigSerializable(split="val_balanced", exclude_ood=True,
                                       window_size=4, max_steps=10,
                                       min_episode_length=4),
            generator_path=gen_dir, dataset_path=ds_dir,
            splits_manifest=synthetic_manifest, seed=0,
        )
        train_pool = set()
        for v in train_vec.envs[0].env._realization_engine._state_indices.values():
            train_pool.update(v)
        eval_pool = set()
        for v in eval_vec.envs[0].env._realization_engine._state_indices.values():
            eval_pool.update(v)
        assert train_pool.isdisjoint(eval_pool)
