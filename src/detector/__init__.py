"""Stage Detector + supervised baselines.

detector module. Public API:

    StageDetector            — production MLP head used by the RL agent.
    train_random_forest      — sklearn RandomForestClassifier wrapper.
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
from src.detector.stage_detector import StageDetector, StageDetectorConfig

__all__ = [
    # Models
    "StageDetector",
    "StageDetectorConfig",
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
