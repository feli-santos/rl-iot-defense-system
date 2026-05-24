"""Stage Detector + supervised baselines.

detector module. Public API:

    StageDetector            — production MLP head used by the RL agent.
    train_random_forest      — sklearn RandomForestClassifier wrapper.
    CNN1D, train_cnn1d       — 1-D conv baseline (Tharewal-style).
    per_stage_recall, ...    — shared evaluation helpers.

See ``docs/results/04_detector/PLAN.md`` §A3 for the rationale.
"""

from src.detector.cnn1d import CNN1D, CNN1DConfig, train_cnn1d
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
    "CNN1D",
    "CNN1DConfig",
    "train_cnn1d",
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
