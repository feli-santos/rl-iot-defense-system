"""Shared evaluation helpers for the detector stage detector + baselines.

Every helper takes raw 1-D ``y_true`` / ``y_pred`` arrays of integer stage
labels and returns NumPy arrays (or floats / dicts of floats). No
randomness, no model dependencies — deliberately sklearn-compatible so
existing tests can cross-check results with ``sklearn.metrics``.

The ``summarize_run`` aggregator is what each detector baseline returns
to the entrypoint script in ``scripts/detector/train_detector.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Number of Kill Chain stages = 5 (BENIGN .. IMPACT). Hard-coded because it
# is also hard-coded in src/utils/label_mapper.KillChainStage and
# src/environment/adversarial_env.
NUM_STAGES: int = 5
STAGE_NAMES: list[str] = ["BENIGN", "RECON", "ACCESS", "MANEUVER", "IMPACT"]


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int = NUM_STAGES
) -> np.ndarray:
    """Integer confusion matrix of shape ``(num_classes, num_classes)``.

    Rows are true classes, columns are predicted classes — same convention
    as ``sklearn.metrics.confusion_matrix``.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true {y_true.shape} and y_pred {y_pred.shape} must match")
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_stage_recall(
    y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int = NUM_STAGES
) -> np.ndarray:
    """Per-class recall vector of shape ``(num_classes,)``.

    Recall_i = TP_i / (TP_i + FN_i). A class with zero true rows yields
    ``0.0`` (rather than NaN) — matches ``sklearn.metrics.recall_score(
    average=None, zero_division=0)``.
    """
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    support = cm.sum(axis=1).astype(np.float64)
    tp = np.diag(cm).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.where(support > 0, tp / np.maximum(support, 1.0), 0.0)
    return rec.astype(np.float64)


def per_class_f1(
    y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int = NUM_STAGES
) -> np.ndarray:
    """Per-class F1 vector of shape ``(num_classes,)``."""
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1.0), 0.0)
        recall = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1.0), 0.0)
        f1 = np.where(
            precision + recall > 0,
            2 * precision * recall / np.maximum(precision + recall, 1e-12),
            0.0,
        )
    return f1.astype(np.float64)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int = NUM_STAGES) -> float:
    """Unweighted mean of per-class F1 scores."""
    return float(per_class_f1(y_true, y_pred, num_classes=num_classes).mean())


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------


@dataclass
class DetectorEvaluation:
    """Single-run evaluation summary for one (model, split) pair.

    Attributes:
        model_name: Human-readable model id (e.g. ``"RandomForest"``).
        split_name: Split this evaluation was computed on
            (``"test_balanced"`` / ``"test"`` / ``"ood:DDoS-HTTP_Flood"`` …).
        macro_f1: Macro F1 over the ``NUM_STAGES`` classes.
        per_class_f1: Per-class F1, indexed by stage id.
        per_stage_recall: Per-class recall, indexed by stage id.
        confusion_matrix: ``(NUM_STAGES, NUM_STAGES)`` matrix.
        n_samples: Number of evaluated rows.
    """

    model_name: str
    split_name: str
    macro_f1: float
    per_class_f1: np.ndarray
    per_stage_recall: np.ndarray
    confusion_matrix: np.ndarray
    n_samples: int

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "split_name": self.split_name,
            "macro_f1": round(float(self.macro_f1), 6),
            "per_class_f1": {
                STAGE_NAMES[i]: round(float(v), 6) for i, v in enumerate(self.per_class_f1.tolist())
            },
            "per_stage_recall": {
                STAGE_NAMES[i]: round(float(v), 6)
                for i, v in enumerate(self.per_stage_recall.tolist())
            },
            "confusion_matrix": self.confusion_matrix.tolist(),
            "n_samples": int(self.n_samples),
        }


def summarize_run(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int = NUM_STAGES,
) -> DetectorEvaluation:
    """One-call factory: build a :class:`DetectorEvaluation` from raw labels."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    return DetectorEvaluation(
        model_name=model_name,
        split_name=split_name,
        macro_f1=macro_f1(y_true, y_pred, num_classes=num_classes),
        per_class_f1=per_class_f1(y_true, y_pred, num_classes=num_classes),
        per_stage_recall=per_stage_recall(y_true, y_pred, num_classes=num_classes),
        confusion_matrix=cm,
        n_samples=int(y_true.shape[0]),
    )


# ---------------------------------------------------------------------------
# OOD eval helpers (detector G4.4)
# ---------------------------------------------------------------------------


@dataclass
class OODEvaluation:
    """Recall on a single held-out attack class.

    Because every held-out class has a single ground-truth Kill Chain stage,
    ``recall`` here is *binary*: fraction of OOD-class rows that the
    detector predicted into the correct stage. The thesis G4.4 gate
    *wants* this to be low (≤ 0.30 ideally).
    """

    attack_class: str
    expected_stage: int
    n_samples: int
    recall: float
    predicted_stage_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "attack_class": self.attack_class,
            "expected_stage": STAGE_NAMES[self.expected_stage],
            "n_samples": int(self.n_samples),
            "recall": round(float(self.recall), 6),
            "predicted_stage_distribution": self.predicted_stage_distribution,
        }


def evaluate_ood_class(
    *,
    attack_class: str,
    expected_stage: int,
    y_pred: np.ndarray,
) -> OODEvaluation:
    """Build an :class:`OODEvaluation` for one held-out attack class."""
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = int(y_pred.shape[0])
    if n == 0:
        return OODEvaluation(
            attack_class=attack_class,
            expected_stage=expected_stage,
            n_samples=0,
            recall=0.0,
            predicted_stage_distribution=dict.fromkeys(STAGE_NAMES, 0),
        )
    correct = int((y_pred == expected_stage).sum())
    dist = {STAGE_NAMES[i]: int((y_pred == i).sum()) for i in range(NUM_STAGES)}
    return OODEvaluation(
        attack_class=attack_class,
        expected_stage=expected_stage,
        n_samples=n,
        recall=correct / n,
        predicted_stage_distribution=dist,
    )
