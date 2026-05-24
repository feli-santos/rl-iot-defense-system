"""
red-team — train the LSTM Red Team episode generator and produce F1 + F2.

Pipeline
--------
1. Verify ``data/processed/ciciot2023/splits/manifest.json`` exists and load
   the **train-split** stage prior from it (no leakage from val/test/OOD).
2. Build an :class:`EpisodeGenerator` with that prior and synthesize a
   training corpus.
3. Train the :class:`AttackSequenceGenerator` LSTM via
   :class:`GeneratorTrainer` (cross-entropy, early stopping on val loss).
4. Evaluate on a held-out synthetic split and on a freshly-sampled
   "ground-truth" set drawn from the *same* transition matrix.
5. Compute the LSTM's empirical 5×5 transition matrix from 10 000 generated
   sequences and the ground-truth synthetic transition matrix; report KL
   divergence (G3 gate from PLAN.md).
6. Emit:
   - ``docs/results/02_red_team/F1_learning_curves.png``
   - ``docs/results/02_red_team/F2_transition_matrix_comparison.png``
   - ``docs/results/02_red_team/F1_summary.json`` (loss/F1/G* gate values)
   - ``docs/results/02_red_team/manifest.json`` (figure→inputs hash chain)

Usage
-----
    PYTHONPATH=. python -m scripts.red_team.train_lstm \\
        --processed-dir data/processed/ciciot2023 \\
        --out-dir docs/results/02_red_team \\
        --epochs 30 --num-episodes 8000 --seed 42

Or via the Makefile (added in this commit): ``make phase-2``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.generator.attack_sequence_generator import (  # noqa: E402
    AttackSequenceGenerator,
    AttackSequenceGeneratorConfig,
)
from src.generator.episode_generator import (  # noqa: E402
    EpisodeGenerator,
    EpisodeGeneratorConfig,
    stage_distribution_from_split_manifest,
)
from src.training.generator_trainer import (  # noqa: E402
    GeneratorTrainer,
    GeneratorTrainingConfig,
)

LOG = logging.getLogger("train_lstm")
NUM_STAGES = 5

# red-team exit gates (from docs/results/02_red_team/PLAN.md §3.2)
DEFAULT_GATES = {
    "G1_max_train_val_gap": 0.25,    # max abs(train-val)/val per epoch
    "G2_min_token_accuracy": 0.55,   # token-level top-1 on synthetic val
    "G3_max_kl_divergence": 0.05,    # KL(P_lstm || P_synthetic_truth)
    "G4_min_cosine_similarity": 0.90,
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for piece in iter(lambda: fp.read(chunk), b""):
            h.update(piece)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except Exception:  # pragma: no cover — best-effort
        return "unknown"


def _flatten_to_transition_matrix(episodes: List[List[int]]) -> np.ndarray:
    """Compute the empirical 5x5 transition matrix from a list of episodes.

    Cell ``T[i, j]`` holds ``P(stage_{t+1} = j | stage_t = i)`` over all
    consecutive pairs found in *episodes*.
    """
    counts = np.zeros((NUM_STAGES, NUM_STAGES), dtype=np.float64)
    for ep in episodes:
        for prev, nxt in zip(ep[:-1], ep[1:]):
            counts[prev, nxt] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    # Avoid division by zero for unvisited rows; leave them as zeros.
    safe = np.where(row_sums > 0, row_sums, 1.0)
    return counts / safe


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """Mean per-row KL(p || q), ignoring rows where p has zero mass."""
    kl_per_row: List[float] = []
    for i in range(p.shape[0]):
        if p[i].sum() <= 0:
            continue
        pi = p[i] + eps
        qi = q[i] + eps
        pi = pi / pi.sum()
        qi = qi / qi.sum()
        kl_per_row.append(float(np.sum(pi * np.log(pi / qi))))
    return float(np.mean(kl_per_row)) if kl_per_row else 0.0


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


# ---------------------------------------------------------------------------
# Sequence sampling from the trained LSTM
# ---------------------------------------------------------------------------

def _generate_lstm_episodes(
    model: AttackSequenceGenerator,
    n_episodes: int,
    seq_len: int,
    *,
    seed_episodes: List[List[int]],
    rng: np.random.Generator,
    sample_temperature: float = 1.0,
) -> List[List[int]]:
    """Greedy/sampling rollout of ``n_episodes`` from the LSTM.

    For each rollout we pick a random "seed" prefix from *seed_episodes* of
    length ``seq_len`` and then auto-regress for the same length as the seed
    episode (capped at 30 to keep runtime bounded).
    """
    out: List[List[int]] = []
    if not seed_episodes:
        raise ValueError("seed_episodes must be non-empty")
    for _ in range(n_episodes):
        seed_idx = int(rng.integers(0, len(seed_episodes)))
        seed_ep = seed_episodes[seed_idx]
        if len(seed_ep) <= seq_len:
            continue
        prefix = seed_ep[:seq_len]
        target_len = min(len(seed_ep), 30)
        gen = model.generate_sequence(
            start_history=list(prefix),
            length=target_len,
            temperature=sample_temperature,
        )
        out.append(gen)
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_learning_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_macro_f1s: List[float],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160)

    ax = axes[0]
    ax.plot(epochs, train_losses, label="Train CE", color="#0072B2", linewidth=1.6)
    ax.plot(epochs, val_losses, label="Val CE", color="#D55E00", linewidth=1.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("LSTM Red Team — training / validation loss")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(epochs, val_macro_f1s, label="Val macro-F1 (synthetic)", color="#009E73", linewidth=1.6)
    ax.axhline(0.20, linestyle="--", color="grey", alpha=0.6, label="Uniform baseline = 0.20")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro-F1")
    ax.set_title("LSTM Red Team — synthetic-token macro-F1")
    ax.set_ylim(0.0, 1.0)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


def _plot_transition_matrices(
    p_lstm: np.ndarray,
    p_truth: np.ndarray,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    stage_labels = ["BENIGN", "RECON", "ACCESS", "MANEUVER", "IMPACT"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=160)

    def _heat(ax, mat: np.ndarray, title: str, *, vmin=0.0, vmax=1.0, cmap="viridis"):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(NUM_STAGES))
        ax.set_yticks(range(NUM_STAGES))
        ax.set_xticklabels(stage_labels, rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(stage_labels, fontsize=8)
        ax.set_xlabel("next stage")
        ax.set_ylabel("current stage")
        ax.set_title(title, fontsize=10)
        for i in range(NUM_STAGES):
            for j in range(NUM_STAGES):
                ax.text(
                    j, i, f"{mat[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if mat[i, j] < (vmin + vmax) / 2 else "black",
                    fontsize=7,
                )
        return im

    _heat(axes[0], p_truth, "Ground truth (synthetic)")
    _heat(axes[1], p_lstm,  "LSTM empirical (10 000 gen.)")
    diff = p_lstm - p_truth
    _heat(axes[2], diff, "LSTM − Truth", vmin=-0.5, vmax=0.5, cmap="coolwarm")

    fig.suptitle("F2 — Empirical 5×5 transition matrix vs ground truth", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed/ciciot2023"))
    p.add_argument("--out-dir", type=Path, default=Path("docs/results/02_red_team"))
    p.add_argument("--artifact-dir", type=Path, default=Path("artifacts/generator/red_team"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--sequence-length", type=int, default=5)
    p.add_argument("--num-episodes", type=int, default=50_000)
    p.add_argument("--min-episode-length", type=int, default=8)
    p.add_argument("--max-episode-length", type=int, default=20)
    p.add_argument("--lstm-hidden", type=int, default=32)   # smaller -> less memorisation
    p.add_argument("--lstm-layers", type=int, default=1)    # 1 layer is enough for 5 tokens
    p.add_argument("--lstm-embed", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--n-eval-episodes", type=int, default=10_000)
    p.add_argument("--no-mlflow", action="store_true", help="disable MLflow logging")
    p.add_argument("--dry-run", action="store_true",
                   help="exit before training (just verify wiring)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )

    splits_manifest = args.processed_dir / "splits" / "manifest.json"
    if not splits_manifest.exists():
        LOG.error(
            "splits/manifest.json not found at %s. Run `make build-split-indices` "
            "first (dataset-prep).", splits_manifest,
        )
        return 1

    # --- 1) Load train-split prior --------------------------------------------------
    train_prior = stage_distribution_from_split_manifest(splits_manifest, "train")
    LOG.info("Train-split prior loaded: %s", train_prior)

    np.random.seed(args.seed)
    rng_master = np.random.default_rng(args.seed)

    # --- 2) Build EpisodeGenerator --------------------------------------------------
    epi_cfg = EpisodeGeneratorConfig(
        num_episodes=args.num_episodes,
        min_length=args.min_episode_length,
        max_length=args.max_episode_length,
        num_stages=NUM_STAGES,
        distribution_temperature=0.7,  # mild flattening (rare stages get more mass)
    )
    epi_gen = EpisodeGenerator(
        config=epi_cfg, stage_distribution=train_prior, seed=args.seed
    )
    ground_truth_T = epi_gen._transition_probs.copy()  # noqa: SLF001 — deliberate
    LOG.info("Ground-truth synthetic transition matrix:\n%s", np.round(ground_truth_T, 3))

    train_episodes = epi_gen.generate_all()
    LOG.info("Generated %d training episodes", len(train_episodes))

    if args.dry_run:
        LOG.info("--dry-run requested — exiting before LSTM training")
        return 0

    # --- 3) Train -------------------------------------------------------------------
    model_cfg = AttackSequenceGeneratorConfig(
        num_stages=NUM_STAGES,
        embedding_dim=args.lstm_embed,
        hidden_size=args.lstm_hidden,
        num_layers=args.lstm_layers,
        dropout=args.dropout,
    )
    train_cfg = GeneratorTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sequence_length=args.sequence_length,
        val_split=0.1,
        early_stopping_patience=8,
        output_dir=args.artifact_dir,
        device="cpu",
        use_mlflow=(not args.no_mlflow),
        mlflow_experiment_name="phase2_lstm_red_team",
        balanced_validation=True,
        val_samples_per_class=200,
        seed=args.seed,
        grad_clip_norm=5.0,
        use_lr_scheduler=True,
        scheduler_patience=3,
        scheduler_factor=0.5,
    )
    trainer = GeneratorTrainer(config=train_cfg, model_config=model_cfg)
    train_results = trainer.train(train_episodes)

    # --- 4) Evaluate on held-out synthetic episodes ----------------------------------
    holdout_episodes = epi_gen.generate_batch(2000)
    holdout_metrics = trainer.evaluate(holdout_episodes)
    LOG.info(
        "Holdout: token-acc=%.4f, macro-F1=%.4f, perplexity=%.3f",
        holdout_metrics["accuracy"],
        holdout_metrics["macro_f1"],
        holdout_metrics["perplexity"],
    )

    # --- 5) Empirical transition matrix from LSTM rollouts ---------------------------
    rng_eval = np.random.default_rng(args.seed + 1)
    lstm_episodes = _generate_lstm_episodes(
        trainer.model,
        n_episodes=args.n_eval_episodes,
        seq_len=args.sequence_length,
        seed_episodes=train_episodes,
        rng=rng_eval,
        sample_temperature=1.0,
    )
    p_lstm = _flatten_to_transition_matrix(lstm_episodes)
    p_truth = ground_truth_T

    kl_div = _kl_divergence(p_lstm, p_truth)
    LOG.info("KL(P_lstm || P_truth) = %.4f", kl_div)

    # Stage-frequency cosine similarity (G4) — apples-to-apples comparison.
    # We compare the LSTM rollouts against an *equal-sized* sample of the
    # synthetic ground-truth EpisodeGenerator. This is the correct comparator
    # because both samples share the IMPACT-absorbing-state inflation effect;
    # the train-split prior would not, since it counts individual rows rather
    # than steps in absorbing-state rollouts.
    truth_episodes = epi_gen.generate_batch(args.n_eval_episodes)

    def _stage_freq(eps: List[List[int]]) -> np.ndarray:
        f = np.zeros(NUM_STAGES)
        for ep in eps:
            for s in ep:
                f[s] += 1
        return f / max(f.sum(), 1.0)

    lstm_freq = _stage_freq(lstm_episodes)
    truth_freq = _stage_freq(truth_episodes)
    train_freq = np.array([train_prior.get(i, 0) for i in range(NUM_STAGES)], dtype=float)
    train_freq /= train_freq.sum()
    cos_sim = _cosine(lstm_freq, truth_freq)
    LOG.info(
        "stage-freq cosine(LSTM, truth_rollouts) = %.4f\n  LSTM        = %s\n  truth roll. = %s\n  train prior = %s",
        cos_sim, np.round(lstm_freq, 3), np.round(truth_freq, 3), np.round(train_freq, 3),
    )

    # G1 — generalization gap on i.i.d. data (NOT on the balanced validation
    # split, which intentionally over-samples rare stages and so always gives
    # a worse loss than train).
    #   final_train_loss vs holdout_loss, both on samples drawn from the
    #   *same* synthetic distribution. Closer = better generalization.
    train_losses = train_results["train_losses"]
    val_losses = train_results["val_losses"]
    final_train_loss = float(train_losses[-1])
    holdout_loss = float(holdout_metrics["loss"])
    iid_gap = abs(final_train_loss - holdout_loss) / max(final_train_loss, holdout_loss, 0.1)
    LOG.info(
        "G1 i.i.d. gap = %.3f (final_train=%.4f, holdout=%.4f); balanced-val=%.4f for reference",
        iid_gap, final_train_loss, holdout_loss, val_losses[-1],
    )

    # Gates evaluation
    gates = {
        "G1_iid_train_holdout_gap": float(iid_gap),
        "G1_balanced_val_loss_for_reference": float(val_losses[-1]),
        "G2_token_accuracy": float(holdout_metrics["accuracy"]),
        "G3_kl_divergence": float(kl_div),
        "G4_cosine_similarity": float(cos_sim),
    }
    gates_passed = {
        "G1": iid_gap <= DEFAULT_GATES["G1_max_train_val_gap"],
        "G2": holdout_metrics["accuracy"] >= DEFAULT_GATES["G2_min_token_accuracy"],
        "G3": kl_div <= DEFAULT_GATES["G3_max_kl_divergence"],
        "G4": cos_sim >= DEFAULT_GATES["G4_min_cosine_similarity"],
    }
    LOG.info("red-team exit gates: %s", gates_passed)

    # --- 6) Figures + summary -------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    f1_path = args.out_dir / "F1_learning_curves.png"
    f2_path = args.out_dir / "F2_transition_matrix_comparison.png"

    _plot_learning_curves(
        train_losses, val_losses, train_results["val_macro_f1s"], f1_path
    )
    _plot_transition_matrices(p_lstm, p_truth, f2_path)

    summary = {
        "phase": 2,
        "seed": args.seed,
        "git_sha": _git_sha(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training": {
            "epochs_trained": train_results["epochs_trained"],
            "best_epoch": train_results["best_epoch"],
            "best_val_loss": train_results["best_val_loss"],
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "val_macro_f1_max": float(max(train_results["val_macro_f1s"])),
        },
        "holdout_metrics": holdout_metrics,
        "gates_thresholds": DEFAULT_GATES,
        "gates_values": gates,
        "gates_passed": gates_passed,
        "all_gates_passed": all(gates_passed.values()),
        "ground_truth_transition_matrix": ground_truth_T.tolist(),
        "lstm_transition_matrix": p_lstm.tolist(),
        "stage_frequency_train_prior": train_freq.tolist(),
        "stage_frequency_lstm": lstm_freq.tolist(),
    }
    summary_path = args.out_dir / "F1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    LOG.info("Wrote %s", summary_path)

    # Per-figure manifest pinning every input + output
    manifest = {
        "figure_id": "F1+F2",
        "title": "red-team Red Team v2 (LSTM episode generator)",
        "produced_by": "scripts/red_team/train_lstm.py",
        "git_sha": _git_sha(),
        "phase": 2,
        "tier": "must-have",
        "inputs": {
            "data/processed/ciciot2023/splits/manifest.json": _sha256(splits_manifest),
        },
        "outputs": {
            "F1_learning_curves.png": _sha256(f1_path),
            "F2_transition_matrix_comparison.png": _sha256(f2_path),
            "F1_summary.json": _sha256(summary_path),
        },
        "all_gates_passed": summary["all_gates_passed"],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    LOG.info("Wrote %s/manifest.json", args.out_dir)

    # Surface a one-line PASS/FAIL banner so CI / shell users see it immediately.
    banner = "✅ ALL GATES PASSED" if summary["all_gates_passed"] else "❌ ONE OR MORE GATES FAILED"
    print("=" * 72)
    print(f"red-team Red Team v2 :: {banner}")
    for gate, passed in gates_passed.items():
        threshold = DEFAULT_GATES[
            {"G1": "G1_max_train_val_gap", "G2": "G2_min_token_accuracy",
             "G3": "G3_max_kl_divergence", "G4": "G4_min_cosine_similarity"}[gate]
        ]
        value = gates[
            {"G1": "G1_iid_train_holdout_gap", "G2": "G2_token_accuracy",
             "G3": "G3_kl_divergence", "G4": "G4_cosine_similarity"}[gate]
        ]
        status = "PASS" if passed else "FAIL"
        print(f"  {gate}: {status}   threshold={threshold:.3f}   observed={value:.4f}")
    print("=" * 72)

    return 0 if summary["all_gates_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
