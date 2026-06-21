"""Blue-Team run configuration dataclass.

A :class:`BlueTeamRunConfig` binds one ``(algo, seed, total_timesteps,
…)`` tuple together with the env / eval config so a single run can be
serialised to ``run_manifest.json`` and replayed verbatim. Every Blue-Team
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
      "paths": {"dataset": "...",
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

    Blue-Team originally serialised only the lifecycle + sampling levers
    (``min_episode_length``, ``max_steps``, ``window_size``,
    ``include_deltas``, ``p_defender_deescalation``) because the
    reward-shaping coefficients were *frozen* by the environment-design contract
    and not Blue-Team levers.

    ablation (audit AF1 / D7.3 / PLAN §3.1.2) extends this to the full
    set of :class:`AdversarialEnvConfig` reward fields so that the F9
    reward-component sweep can override individual coefficients
    per-cell via ``train_agent.py --reward-overrides``. Every new field
    has a default that matches environment-design's frozen value, so existing
    Blue-Team manifests deserialise unchanged and the default training
    behaviour is byte-for-byte identical to Blue-Team.
    """

    # Lifecycle + sampling (original Blue-Team fields)
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

    # Terminal reward for a prevented attack (attacker held below IMPACT for
    # the entire horizon under proximity-coupled escalation). Must match
    # between train and eval.
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
    # Reward mode. Canonical: "coupled" (kill-chain-aware shaping, the
    # reward-shaping ablation cell) or "outcome" (outcome-only, the primary
    # deployment contract). Legacy aliases "proportional"/"outcome_only" are
    # accepted and normalised in __post_init__ so serialised manifests are
    # canonical (keeps the train/eval parity check alias-insensitive).
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

    # Partial-observability redesign (sequential POMDP). ``aliasing_rate`` (alpha)
    # is the probability that a step emits a feature row drawn from an adjacent
    # kill-chain stage instead of the true stage, applied identically to every
    # policy so the supervised baseline and the RL agents see the same ambiguous
    # observation stream. ``session_coherent`` draws contiguous without-replacement
    # runs of same-stage rows (a session proxy). ``no_post_transition_leak`` emits
    # the refreshed observation from the pre-transition stage so the just-occurred
    # transition is not revealed one step early. ``proximity_coupled`` replaces the
    # finite intrusion budget with a proximity-coupled escalation/tolerance rule
    # (prevention is awarded for holding the attacker below IMPACT for the horizon,
    # not for draining a counter); ``proximity_min_escalation`` floors the
    # proximity-scaled under-force escalation probability. Defaults reproduce the
    # legacy fully-observable, budget-based contract byte-for-byte. Must match
    # between train and eval.
    aliasing_rate: float = 0.0
    session_coherent: bool = False
    no_post_transition_leak: bool = False
    proximity_coupled: bool = False
    proximity_min_escalation: float = 0.4

    def __post_init__(self) -> None:
        # Normalise reward-mode aliases to the canonical token so manifests are
        # consistent regardless of which spelling the caller passed.
        _aliases = {
            "proportional": "coupled",
            "coupled": "coupled",
            "outcome_only": "outcome",
            "outcome": "outcome",
        }
        canonical = _aliases.get(self.reward_mode)
        if canonical is None:
            raise ValueError(
                f"Unknown reward_mode {self.reward_mode!r}; "
                f"expected one of {sorted(set(_aliases))}."
            )
        self.reward_mode = canonical


@dataclass
class BlueTeamRunConfig:
    """Frozen configuration for one Blue-Team (algo, seed) run."""

    algo: str
    seed: int
    total_timesteps: int = 500_000
    eval_freq: int = 25_000
    n_eval_episodes: int = 30
    # Early-stopping on the eval-reward plateau. ``total_timesteps`` is the
    # generous cap; training stops early when the best-so-far eval reward has
    # not improved for ``early_stop_patience`` consecutive evaluations, but
    # never before ``early_stop_min_evals`` evaluations have happened. The
    # best-eval checkpoint (``best_model.zip``) is what downstream eval loads,
    # so per-algorithm convergence speed does not bias the comparison.
    early_stop: bool = True
    early_stop_patience: int = 10
    early_stop_min_evals: int = 10
    out_dir: str = "runs/ppo/seed_0"
    dataset_path: str = "data/processed/ciciot2023"
    splits_manifest: str = "data/processed/ciciot2023/splits/manifest.json"
    env: EnvConfigSerializable = field(default_factory=EnvConfigSerializable)
    eval_env: EnvConfigSerializable = field(
        default_factory=lambda: EnvConfigSerializable(split="val_balanced")
    )
    algo_hparams: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    # Env fields that define the *task contract*. Train and eval must agree on
    # every one of these or the reported eval number measures a different MDP
    # than the one the agent trained on. ``split`` and the eval-only sampling
    # knobs are intentionally excluded (train uses ``train`` / eval uses a
    # held-out balanced split by design).
    _PARITY_FIELDS = (
        "exclude_ood",
        "impact_is_terminal",
        "reward_mode",
        "tug_of_war",
        "p_onset",
        "p_onset_access",
        "p_down",
        "p_up",
        "p_down_isolate",
        "p_defender_deescalation",
        "retreat_prob",
        "evasion_prob",
        "action_cost_scale",
        "reward_proportional",
        "penalty_disproportionate",
        "proportional_bonus_cap",
        "reward_deescalation",
        "deescalation_bonus_cap",
        "impact_penalty",
        "penalty_missed_impact",
        "defense_success_bonus",
        "reward_benign_passive",
        "penalty_overreact_benign",
        "penalty_block_benign",
        "penalty_block_recon",
        "prevention_bonus",
        "fpr_penalty_beta",
        "min_episode_length",
        "max_steps",
        "window_size",
        "include_deltas",
        "include_stage_pred",
        "aliasing_rate",
        "session_coherent",
        "no_post_transition_leak",
        "proximity_coupled",
        "proximity_min_escalation",
    )

    def __post_init__(self) -> None:
        self.assert_train_eval_parity()

    def assert_train_eval_parity(self) -> None:
        """Fail loudly if train/eval disagree on any task-contract field.

        Without this, a silent mismatch (e.g. training with
        ``proximity_coupled=True`` but evaluating with
        ``proximity_coupled=False``, or training ``coupled`` and evaluating
        ``outcome``) would not error — the eval number would simply measure a
        different MDP. The eval manifest records these values but nothing else
        enforces that they match the training contract.
        """
        mismatches = []
        for name in self._PARITY_FIELDS:
            train_val = getattr(self.env, name)
            eval_val = getattr(self.eval_env, name)
            if train_val != eval_val:
                mismatches.append(f"  {name}: train={train_val!r} eval={eval_val!r}")
        if mismatches:
            raise ValueError(
                "BlueTeamRunConfig: train/eval env contract mismatch — the eval "
                "number would measure a different MDP than training:\n" + "\n".join(mismatches)
            )

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
            "early_stop": self.early_stop,
            "early_stop_patience": self.early_stop_patience,
            "early_stop_min_evals": self.early_stop_min_evals,
            "env": asdict(self.env),
            "eval_env": asdict(self.eval_env),
            "algo_hparams": dict(self.algo_hparams),
            "paths": {
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
                "early_stopped",
                "actual_timesteps",
                "best_model_path",
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
            early_stop=bool(d.get("early_stop", True)),
            early_stop_patience=int(d.get("early_stop_patience", 10)),
            early_stop_min_evals=int(d.get("early_stop_min_evals", 10)),
            out_dir=str(paths.get("out_dir", "")),
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
