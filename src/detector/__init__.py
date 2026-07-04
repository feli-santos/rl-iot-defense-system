"""Stage Detector + supervised baseline.

detector module. Public API:

    train_random_forest      — sklearn RandomForestClassifier wrapper
                               (the tuned RF-Acting stage detector).
    per_stage_recall, ...    — shared evaluation helpers.

See ``docs/results/stage-detector/PLAN.md`` §A3 for the rationale.
"""

from src.detector.evaluation import (
    DetectorEvaluation,
    confusion_matrix,
    macro_f1,
    per_class_f1,
    per_stage_recall,
    summarize_run,
)
from src.detector.random_forest import RandomForestConfig, train_random_forest

__all__ = [
    # Models
    "RandomForestConfig",
    "train_random_forest",
    # Evaluation
    "DetectorEvaluation",
    "per_stage_recall",
    "per_class_f1",
    "macro_f1",
    "confusion_matrix",
    "summarize_run",
]
