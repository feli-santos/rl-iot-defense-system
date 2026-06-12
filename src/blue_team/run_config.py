"""blue-team run configuration dataclass.

A :class:`BlueTeamRunConfig` binds one ``(algo, seed, total_timesteps,
…)`` tuple together with the env / eval config so a single run can be
serialised to ``run_manifest.json`` and replayed verbatim. Every blue-team
training script materialises a ``BlueTeamRunConfig`` first, then
materialises the env and the model from it.

The manifest written at training time has this shape::

    {
      "schema_version": "1.0",
      "git_sha": "9b70d7d…",
      "run_id": "ppo_seed_3",
      "algo": "ppo",
      "seed": 3,
      "total_timesteps": 500000,
      "eval_freq": 25000,
      "n_eval_episodes": 30,
      "env": {"split": "train", "exclude_ood": true,
              "min_episode_length": 20, "max_steps": 100, "window_size": 5,
              "include_deltas": true, "p_defender_deescalation": 0.6},
      "eval_env": {"split": "val_balanced", "exclude_ood": true,
                   "min_episode_length": 20, "max_steps": 100, ...},
      "algo_hparams": {...},
      "paths": {"generator": "...", "dataset": "...",
                "splits_manifest": "...",
                "out_dir": "runs/ppo/seed_3"},
      "completed_at": "2026-04-30T15:42:11Z",
      "wallclock_seconds": 4321.7,
      "n_episodes_train": 12380,
      "n_episodes_eval": 600
    }
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_SCHEMA_VERSION = "1.0"


@dataclass
class EnvConfigSerializable:
    """Subset of :class:`AdversarialEnvConfig` that we serialise.

    blue-team originally serialised only the lifecycle + sampling levers
    (``min_episode_length``, ``max_steps``, ``window_size``,
    ``include_deltas``, ``p_defender_deescalation``) because the
    reward-shaping coefficients were *frozen* by the environment-design contract
    and not blue-team levers.

    ablation (audit AF1 / D7.3 / PLAN §3.1.2) extends this to the full
    set of :class:`AdversarialEnvConfig` reward fields so that the F9
    reward-component sweep can override individual coefficients
    per-cell via ``train_agent.py --reward-overrides``. Every new field
    has a default that matches environment-design's frozen value, so existing
    blue-team manifests deserialise unchanged and the default training
    behaviour is byte-for-byte identical to blue-team.
    """

    # Lifecycle + sampling (original blue-team fields)
    split: str = "train"
    exclude_ood: bool = True
    min_episode_length: int = 20
    max_steps: int = 100
    window_size: int = 5
    include_deltas: bool = True
    p_defender_deescalation: float = 0.6
    # ablation D7.3: explicit IMPACT-row decision step toggle. Default
    # ``True`` preserves the environment-design/4/5/6 frozen contract.
    impact_is_terminal: bool = True

    # Stage-prediction ablation (review 2.4.1)
    stage_detector_path: Optional[str] = None
    include_stage_pred: bool = False

    # Non-monotonic attacker stress-test (review 2.4.3)
    retreat_prob: float = 0.0

    # Evasion-before-commit reactive attacker (defender-action-coupled stall).
    evasion_prob: float = 0.0

    # Finite attacker budget (prevention model). ``attacker_budget=None``
    # preserves the unbounded contract (compromise_rate == 1.0); a finite
    # budget drains by ``budget_step_cost`` per active progression step and
    # ``budget_reset_cost`` per defender de-escalation, and an attacker that
    # exhausts its budget before IMPACT is prevented.
    attacker_budget: Optional[int] = None
    budget_step_cost: int = 1
    budget_reset_cost: int = 2
    budget_cost_model: str = "hybrid"
    prevention_bonus: float = 50.0

    # Tug-of-war dynamics (headline contract). The defender's signed force
    # difference d = action - recommended(stage) governs the attacker:
    # d <= -1 (under-force) escalates w.p. ``p_up``; d == 0 (proportional)
    # de-escalates w.p. ``p_down`` (``p_down_isolate`` for ISOLATE); d >= 1
    # (over-force) holds. BENIGN has an autonomous multi-rung onset
    # (``p_onset`` -> RECON, ``p_onset_access`` -> ACCESS) independent of the
    # defender. ``tug_of_war=False`` recovers the legacy autonomous-Markov +
    # ``_maybe_defender_deescalation`` path (retained for the
    # reward-mis-specification ablation strand). Defaults mirror
    # :class:`AdversarialEnvConfig`.
    tug_of_war: bool = True
    p_onset: float = 0.35
    p_onset_access: float = 0.10
    p_down: float = 0.90
    p_up: float = 0.90
    p_down_isolate: float = 0.98

    # Reward shaping — ablation F9 axes (defaults from environment-design RESULTS §3)
    action_cost_scale: float = 1.0
    reward_proportional: float = 5.0
    penalty_disproportionate: float = 5.0
    # Per-episode caps that close reward-farming loopholes (None disables a cap,
    # used as a reward-mis-specification ablation cell). Routine de-escalations
    # earn ``reward_deescalation`` (small), decoupled from the
    # ``defense_success_bonus`` reserved for surviving a terminal IMPACT step.
    proportional_bonus_cap: Optional[float] = 100.0
    reward_deescalation: float = 15.0
    deescalation_bonus_cap: Optional[float] = 150.0
    # Reward mode ablation: "proportional" (default) or "outcome_only".
    reward_mode: str = "proportional"
    impact_penalty: float = 200.0
    penalty_missed_impact: float = 150.0
    defense_success_bonus: float = 250.0
    reward_benign_passive: float = 10.0
    penalty_overreact_benign: float = 50.0
    penalty_block_benign: float = 100.0
    penalty_block_recon: float = 50.0

    # Lagrangian FPR penalty (review 2.2 / Direction 6)
    fpr_penalty_beta: float = 0.0


@dataclass
class BlueTeamRunConfig:
    """Frozen configuration for one blue-team (algo, seed) run."""

    algo: str
    seed: int
    total_timesteps: int = 500_000
    eval_freq: int = 25_000
    n_eval_episodes: int = 30
    out_dir: str = "runs/ppo/seed_0"
    generator_path: str = "artifacts/generator/phase2"
    dataset_path: str = "data/processed/ciciot2023"
    splits_manifest: str = "data/processed/ciciot2023/splits/manifest.json"
    env: EnvConfigSerializable = field(default_factory=EnvConfigSerializable)
    eval_env: EnvConfigSerializable = field(
        default_factory=lambda: EnvConfigSerializable(split="val_balanced")
    )
    algo_hparams: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def run_id(self) -> str:
        """Stable identifier ``"<algo>_seed_<seed>"`` used in JSONL/manifest."""
        return f"{self.algo}_seed_{self.seed}"

    # ------------------------------------------------------------------ I/O

    def to_dict(self, *, completed: bool = False, **extra: Any) -> dict[str, Any]:
        """Render this config to a JSON-serialisable dict.

        When ``completed=True``, fills the post-run telemetry fields
        (``completed_at``, ``wallclock_seconds``, ``n_episodes_*``)
        from ``extra``.
        """
        d: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "run_id": self.run_id,
            "algo": self.algo,
            "seed": self.seed,
            "total_timesteps": self.total_timesteps,
            "eval_freq": self.eval_freq,
            "n_eval_episodes": self.n_eval_episodes,
            "env": asdict(self.env),
            "eval_env": asdict(self.eval_env),
            "algo_hparams": dict(self.algo_hparams),
            "paths": {
                "generator": self.generator_path,
                "dataset": self.dataset_path,
                "splits_manifest": self.splits_manifest,
                "out_dir": self.out_dir,
            },
            "notes": self.notes,
        }
        if completed:
            d["completed_at"] = extra.get(
                "completed_at",
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            for k in (
                "wallclock_seconds",
                "n_episodes_train",
                "n_episodes_eval",
                "git_sha",
                "final_eval_reward",
                "final_eval_mttc",
                "final_eval_compromise_rate",
            ):
                if k in extra:
                    d[k] = extra[k]
        return d

    def write_manifest(self, path: Path | str, **extra: Any) -> Path:
        """Atomic-write ``run_manifest.json`` next to the JSONL files."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(completed=True, **extra), indent=2))
        tmp.replace(p)
        return p

    # ------------------------------------------------------------------ factories

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BlueTeamRunConfig:
        """Round-trip from the JSON manifest written by :meth:`write_manifest`.

        Strict on schema version so tests can lock the wire format.
        """
        v = d.get("schema_version")
        if v != _SCHEMA_VERSION:
            raise ValueError(
                f"BlueTeamRunConfig: unsupported schema_version {v!r}; expected {_SCHEMA_VERSION!r}"
            )
        env = EnvConfigSerializable(**d.get("env", {}))
        eval_env = EnvConfigSerializable(**d.get("eval_env", {}))
        paths = d.get("paths", {})
        return cls(
            algo=d["algo"],
            seed=int(d["seed"]),
            total_timesteps=int(d.get("total_timesteps", 500_000)),
            eval_freq=int(d.get("eval_freq", 25_000)),
            n_eval_episodes=int(d.get("n_eval_episodes", 30)),
            out_dir=str(paths.get("out_dir", "")),
            generator_path=str(paths.get("generator", "")),
            dataset_path=str(paths.get("dataset", "")),
            splits_manifest=str(paths.get("splits_manifest", "")),
            env=env,
            eval_env=eval_env,
            algo_hparams=dict(d.get("algo_hparams", {})),
            notes=str(d.get("notes", "")),
        )

    @classmethod
    def from_manifest(cls, path: Path | str) -> BlueTeamRunConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))


__all__ = ["BlueTeamRunConfig", "EnvConfigSerializable"]
