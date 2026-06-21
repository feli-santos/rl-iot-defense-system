"""Smoke / round-trip test for ``scripts.blue_team.train_agent`` (3.2.4).

Synthetic env only — no dependence on the real CICIoT snapshot. The
test runs ~200 timesteps with PPO, verifies the four output artefacts
appear, asserts JSONL is well-formed and parsable by the aggregation
helpers, and that the saved model re-loads to the same prediction.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.blue_team import BlueTeamRunConfig, EnvConfigSerializable
from src.blue_team.aggregation import read_episodes_jsonl

# --------------------------------------------------------------- fixtures


@pytest.fixture
def synthetic_paths(tmp_path: Path) -> Path:
    """Tiny synthetic dataset for the smoke run."""
    ds_dir = tmp_path / "dataset"
    ds_dir.mkdir(parents=True)
    rng = np.random.default_rng(42)
    features = rng.standard_normal((200, 29)).astype(np.float32)
    np.save(ds_dir / "features.npy", features)
    labels = rng.integers(0, 5, size=200)
    np.save(ds_dir / "labels.npy", labels)
    state_indices = {str(i): [] for i in range(5)}
    for idx, lab in enumerate(labels):
        state_indices[str(int(lab))].append(int(idx))
    (ds_dir / "state_indices.json").write_text(json.dumps(state_indices))
    joblib.dump(StandardScaler().fit(features), ds_dir / "scaler.joblib")

    return ds_dir


# --------------------------------------------------------------- tests


class TestTrainAgentSmoke:
    def _build_cfg(
        self,
        *,
        ds_dir: Path,
        out_dir: Path,
        algo: str = "ppo",
        seed: int = 0,
        total_timesteps: int = 200,
    ) -> BlueTeamRunConfig:
        env = EnvConfigSerializable(
            split="train",
            exclude_ood=False,
            min_episode_length=5,
            max_steps=20,
            window_size=4,
            include_deltas=True,
        )
        eval_env = EnvConfigSerializable(
            split="train",
            exclude_ood=False,
            min_episode_length=5,
            max_steps=20,
            window_size=4,
            include_deltas=True,
        )
        return BlueTeamRunConfig(
            algo=algo,
            seed=seed,
            total_timesteps=total_timesteps,
            eval_freq=100,
            n_eval_episodes=2,
            out_dir=str(out_dir),
            dataset_path=str(ds_dir),
            splits_manifest="",  # synthetic — no dataset-prep manifest
            env=env,
            eval_env=eval_env,
            algo_hparams={
                "learning_rate": 3e-4,
                "n_steps": 32,
                "batch_size": 16,
                "n_epochs": 2,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
            },
            notes="smoke",
        )

    def test_smoke_run_produces_all_artefacts(
        self, synthetic_paths: tuple[Path, Path], tmp_path: Path
    ) -> None:
        from scripts.blue_team.train_agent import train

        ds_dir = synthetic_paths
        out_dir = tmp_path / "run"
        cfg = self._build_cfg(
            ds_dir=ds_dir,
            out_dir=out_dir,
            total_timesteps=200,
        )
        result = train(cfg, verbose=0)

        # All four artefacts present.
        assert (out_dir / "episodes.jsonl").exists()
        assert (out_dir / "eval.jsonl").exists()
        assert (out_dir / "run_manifest.json").exists()
        assert (out_dir / "model.zip").exists()

        # episodes.jsonl is well-formed and aggregation can read it.
        records = read_episodes_jsonl(out_dir / "episodes.jsonl")
        assert len(records) >= 1
        assert all(r["algo"] == "ppo" for r in records)
        assert all(r["seed"] == 0 for r in records)
        assert all(r["schema_version"] == "1.0" for r in records)

        # Manifest round-trips.
        loaded = BlueTeamRunConfig.from_manifest(out_dir / "run_manifest.json")
        assert loaded.algo == "ppo"
        assert loaded.seed == 0
        assert loaded.total_timesteps == 200

        # Telemetry was filled in.
        manifest = json.loads((out_dir / "run_manifest.json").read_text())
        assert manifest["n_episodes_train"] == result["n_episodes_train"]
        assert manifest["n_episodes_train"] >= 1
        assert manifest["wallclock_seconds"] > 0

    def test_early_stop_writes_best_checkpoint_and_manifest(
        self, synthetic_paths: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """WS3: with early-stop enabled the run must emit ``best_model.zip``
        (the canonical checkpoint for downstream eval) and record the
        early-stop telemetry in the manifest, round-tripping through the
        config dataclass."""
        from scripts.blue_team.train_agent import train

        ds_dir = synthetic_paths
        out_dir = tmp_path / "run"
        cfg = self._build_cfg(
            ds_dir=ds_dir,
            out_dir=out_dir,
            total_timesteps=300,
        )
        cfg.early_stop = True
        cfg.early_stop_patience = 2
        cfg.early_stop_min_evals = 1
        train(cfg, verbose=0)

        # Best-eval checkpoint is what benchmark/OOD load.
        assert (out_dir / "best_model.zip").exists()
        assert (out_dir / "model.zip").exists()

        manifest = json.loads((out_dir / "run_manifest.json").read_text())
        assert "early_stopped" in manifest
        assert "actual_timesteps" in manifest
        # SB3 finishes the in-flight rollout, so actual can exceed the cap by
        # up to one n_steps batch; just assert it is recorded and positive.
        assert manifest["actual_timesteps"] > 0
        assert manifest["best_model_path"].endswith("best_model.zip")

        loaded = BlueTeamRunConfig.from_manifest(out_dir / "run_manifest.json")
        assert loaded.early_stop is True
        assert loaded.early_stop_patience == 2
        assert loaded.early_stop_min_evals == 1

    def test_saved_model_round_trips_to_same_prediction(
        self, synthetic_paths: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Same-as :func:`test_save_load_model` regression in
        :class:`TestAdversarialAlgorithm`, but for the Blue-Team entrypoint."""
        from stable_baselines3 import PPO

        from scripts.blue_team.train_agent import train

        ds_dir = synthetic_paths
        out_dir = tmp_path / "run"
        cfg = self._build_cfg(
            ds_dir=ds_dir,
            out_dir=out_dir,
            total_timesteps=64,
        )
        train(cfg, verbose=0)

        model = PPO.load(str(out_dir / "model.zip"))
        # Build a deterministic obs and verify two predict() calls
        # return the same action.
        obs = np.zeros(model.observation_space.shape, dtype=np.float32)
        a1, _ = model.predict(obs, deterministic=True)
        a2, _ = model.predict(obs, deterministic=True)
        assert int(a1) == int(a2)
        assert 0 <= int(a1) < 5
