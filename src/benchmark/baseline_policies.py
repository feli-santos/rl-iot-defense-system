"""Non-RL baseline policies for Phase 6 (PLAN §3.1.1).

Every policy is callable with the same signature::

    policy(obs: np.ndarray, info: Dict[str, Any]) -> int

so :func:`src.benchmark.eval_runner.run_policy` can drive any of them
(or an SB3 trained model wrapped in :class:`SB3PolicyAdapter`) through
the same rollout loop. The signature is:

- ``obs``: the env's flattened observation. Shape
  ``(window_size * num_features,)`` (no deltas) or
  ``(window_size * num_features * 2,)`` (with deltas, the Phase-5
  default). Most baselines ignore it; :class:`RFActingPolicy` slices
  the **last step's raw features** out of it.
- ``info``: the env's per-step ``info`` dict, including
  ``info["recommended_action"]`` (Phase-3 contract).

Returns an integer action in ``[0, 4]``.

The baselines covered here implement the four "non-RL" rows of F5 / F8:

================================  ===================================================
Policy                            What it does
================================  ===================================================
``random_policy(rng=...)``        Uniform-random over ``[0,4]`` (seeded).
``always_observe``                Constant 0.
``always_block``                  Constant 3.
``recommended_action_policy``     Returns ``info["recommended_action"]`` —
                                  the IoTWarden hand-crafted rule baseline.
:class:`RFActingPolicy`           ``recommended_action(rf.predict(features))`` —
                                  supervised classifier + rules (D6.5).
================================  ===================================================

The fifth comparator in F5 / F8 is the trained RL trio (DQN / PPO / A2C),
which arrives via :class:`SB3PolicyAdapter`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np


class Policy(Protocol):
    """Structural type for everything :func:`run_policy` can roll.

    Implementing classes/functions just need a ``__call__`` with the
    signature below. We keep this as a Protocol (not an ABC) so
    plain functions and lambdas work without inheritance ceremony.
    """

    def __call__(self, obs: np.ndarray, info: dict[str, Any]) -> int: ...


# ---------------------------------------------------------------------- random


def random_policy(
    obs: np.ndarray,  # noqa: ARG001 — uniform-random ignores obs/info
    info: dict[str, Any],  # noqa: ARG001
    *,
    rng: np.random.Generator | None = None,
) -> int:
    """Uniform-random action in ``[0, 4]``.

    Pass an explicit ``rng`` (``np.random.default_rng(seed)``) for
    reproducibility — the rollout harness threads one RNG through the
    whole episode batch so seeds remain isolated.

    Args:
        obs: ignored.
        info: ignored.
        rng: numpy ``Generator``. ``None`` falls back to a fresh
            unseeded generator (NOT recommended for reported numbers;
            tests must always pass an explicit RNG).
    """
    g = rng if rng is not None else np.random.default_rng()
    return int(g.integers(0, 5))


# ---------------------------------------------------------------------- constants


def always_observe(
    obs: np.ndarray,  # noqa: ARG001
    info: dict[str, Any],  # noqa: ARG001
) -> int:
    """Always pick OBSERVE (action 0) — the most permissive baseline.

    Useful as a lower bound for both compromise rate (worst case: never
    intervene) and for action cost (best case: zero cost per step).
    """
    return 0


def always_block(
    obs: np.ndarray,  # noqa: ARG001
    info: dict[str, Any],  # noqa: ARG001
) -> int:
    """Always pick BLOCK (action 3) — the most aggressive non-isolating baseline.

    Useful as the upper bound on (security cost + action cost) trade-off:
    blocks compromises but pays the BLOCK penalty on every benign step.
    """
    return 3


# ---------------------------------------------------------------- recommended


def recommended_action_policy(
    obs: np.ndarray,  # noqa: ARG001
    info: dict[str, Any],
) -> int:
    """The IoTWarden hand-crafted rule baseline.

    Reads ``info["recommended_action"]`` from the Phase-3 env, which is
    the locked per-stage proportional mapping
    ``{BENIGN→OBSERVE, RECON→LOG, ACCESS→THROTTLE, MANEUVER→BLOCK,
    IMPACT→ISOLATE}``. This is exactly the rule-based comparator that
    Phase-5 G5.2 measured the trained RL trio against (the floor was
    +50; trained agents reach +1300..+1350).

    Raises:
        KeyError: if ``info`` is missing ``recommended_action``. The
            Phase-3 env always emits it; absence indicates a stub env
            and should fail loudly.
    """
    return int(info["recommended_action"])


# ------------------------------------------------------------------- RF baseline


# The Phase-3 recommended_action mapping, replicated here so RFActingPolicy
# does NOT need a live env info dict — it works off the predicted stage
# alone. Kept in lock-step with src/environment/adversarial_env.py's
# `_recommended_action`; if that mapping ever changes, this constant must
# move with it (and a Phase-3.1 test will catch the drift).
_RECOMMENDED_BY_STAGE: dict[int, int] = {
    0: 0,  # BENIGN   → OBSERVE
    1: 1,  # RECON    → LOG
    2: 2,  # ACCESS   → THROTTLE
    3: 3,  # MANEUVER → BLOCK
    4: 4,  # IMPACT   → ISOLATE
}


class RFActingPolicy:
    """Supervised stage classifier + recommended-action mapping (D6.5).

    Tests the thesis claim that *learned* RL beats *supervised stage
    classifier composed with rules*: at each step we extract the raw
    features of the **most recent observation row** from the env's flat
    obs vector, ask the Phase-4 RandomForest to predict the kill-chain
    stage, and return the recommended action for that stage.

    Args:
        rf: A fitted ``RandomForestClassifier`` (the Phase-4
            ``artifacts/detector/random_forest.joblib``) **or** a path
            to one. The model must expose ``predict(X)`` returning
            integer stage labels in ``[0, 4]``.
        num_features: The Phase-3 env's number of raw features per step
            (``29`` for the production CICIoT2023 split).
        window_size: The Phase-3 env's window size (``5`` by default).
        include_deltas: Whether the obs vector also carries first-order
            deltas appended along the feature axis. ``True`` is the
            Phase-5 default and matches the production env spec.

    Notes on the obs slicing (must mirror the Phase-3 env):

        ``_build_observation`` stacks the window into shape ``(W, F)``
        and, when ``include_deltas`` is True, concatenates a ``(W, F)``
        delta block along axis=1 to get ``(W, 2F)``. It then ``.flatten()``s
        in C-order, so row ``i`` lives at indices ``[i*K, (i+1)*K)``
        where ``K = F`` (no deltas) or ``K = 2F`` (with deltas), and
        within each row the **first F values are the raw features**
        (deltas come after). The latest step is row ``W-1``; we therefore
        slice ``obs[(W-1)*K : (W-1)*K + F]``.
    """

    def __init__(
        self,
        rf: Any,
        *,
        num_features: int = 29,
        window_size: int = 5,
        include_deltas: bool = True,
    ) -> None:
        if num_features < 1:
            raise ValueError(f"num_features must be >= 1, got {num_features}")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self._rf = self._coerce_rf(rf)
        self._num_features = int(num_features)
        self._window_size = int(window_size)
        self._include_deltas = bool(include_deltas)

    # ------------------------------------------------------------------ public

    @property
    def rf(self) -> Any:
        """Underlying classifier; exposed for hash-pinning manifests."""
        return self._rf

    def __call__(self, obs: np.ndarray, info: dict[str, Any]) -> int:  # noqa: ARG002
        features = self._extract_latest_features(obs)
        stage = int(self._rf.predict(features.reshape(1, -1))[0])
        if not (0 <= stage <= 4):
            raise ValueError(
                f"RFActingPolicy: classifier returned out-of-range stage {stage} (expected 0..4)"
            )
        return _RECOMMENDED_BY_STAGE[stage]

    # ------------------------------------------------------------------ helpers

    def _extract_latest_features(self, obs: np.ndarray) -> np.ndarray:
        """Return the raw-feature slice of the *most recent* env-window row."""
        flat = np.asarray(obs).reshape(-1)
        per_row = self._num_features * (2 if self._include_deltas else 1)
        expected = per_row * self._window_size
        if flat.size != expected:
            raise ValueError(
                f"RFActingPolicy: expected obs of size {expected} "
                f"(window={self._window_size}, F={self._num_features}, "
                f"include_deltas={self._include_deltas}), got {flat.size}"
            )
        last_row_start = (self._window_size - 1) * per_row
        return flat[last_row_start : last_row_start + self._num_features]

    @staticmethod
    def _coerce_rf(rf: Any) -> Any:
        """Accept either a fitted model or a path-like to a joblib dump."""
        if isinstance(rf, (str, Path)):
            # Local import avoids pulling joblib into module-level imports
            # for synthetic-only test consumers.
            from src.detector.random_forest import load_random_forest

            return load_random_forest(Path(rf))
        if not hasattr(rf, "predict"):
            raise TypeError(
                f"RFActingPolicy: rf must be a sklearn-like classifier "
                f"with .predict(X) or a path to a joblib dump; got {type(rf)!r}"
            )
        return rf


# -------------------------------------------------------------- SB3 adapter


class SB3PolicyAdapter:
    """Wrap an SB3 ``BaseAlgorithm`` so it satisfies the :class:`Policy` protocol.

    SB3 models expect a batched obs (shape ``(1, obs_dim)`` for our
    DummyVecEnv) and return a tuple ``(action_array, state)``; we hide
    that and return a plain Python ``int`` so the rollout harness can
    log it directly into the action histogram.

    Args:
        model: A loaded SB3 model (e.g., ``DQN.load(path)``).
        deterministic: Forwarded to ``model.predict``. Set ``True`` for
            the Phase-6 eval rollouts (D6.3) so seed-to-seed variance
            comes from the env, not the policy's exploration noise.
    """

    def __init__(self, model: Any, *, deterministic: bool = True) -> None:
        if not hasattr(model, "predict"):
            raise TypeError(
                f"SB3PolicyAdapter: model must expose .predict(obs); got {type(model)!r}"
            )
        self._model = model
        self._deterministic = bool(deterministic)

    @property
    def model(self) -> Any:
        return self._model

    def __call__(
        self,
        obs: np.ndarray,
        info: dict[str, Any],  # noqa: ARG002 — SB3 doesn't read info
    ) -> int:
        # SB3.predict expects a leading batch dim; our caller (eval_runner)
        # passes the raw vec-env obs which is already batched. Make this
        # robust to either: if obs is 1-D add the batch dim ourselves.
        x = np.asarray(obs)
        if x.ndim == 1:
            x = x[None, :]
        action_arr, _state = self._model.predict(x, deterministic=self._deterministic)
        # action_arr shape: (1,) for Discrete spaces.
        return int(np.asarray(action_arr).reshape(-1)[0])


__all__ = [
    "Policy",
    "RFActingPolicy",
    "SB3PolicyAdapter",
    "always_block",
    "always_observe",
    "random_policy",
    "recommended_action_policy",
]
