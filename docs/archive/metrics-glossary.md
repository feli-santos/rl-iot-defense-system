# Metrics Glossary

> **STATUS — pre-restart, retained for historical reference (Phase 10, D10.2).**
> The metric *definitions* below are still useful, but the file
> references the dead `src/benchmarking/` package, which was removed in
> Phase 10 (2026-05-04). The canonical glossary used by the Phase 5/6/7
> RESULTS chapters is inlined per-figure in the corresponding
> `RESULTS.md` and `<F<N>>_summary.json`. Metrics are computed by the
> live `src/benchmark/eval_runner.py`, not by the dead modules cited here.

## Generator metrics

- **Loss**: Cross-entropy loss for next-token prediction.
- **Accuracy**: Fraction of correct next-stage predictions.
- **Perplexity**: $\exp\left(-\frac{1}{N}\sum\log p\right)$; lower is better.
- **Macro-F1**: Average F1 across 5 stages; prevents BENIGN dominance.
- **Transition Accuracy**: Accuracy of stage transitions in episode context.

## RL training metrics

- **Episode reward**: Sum of step rewards per episode.
- **Success rate**: Fraction of positive-reward episodes.
- **Action entropy**: Diversity of action choices.
- **FPS / step time**: Performance indicators.

## Security metrics (benchmarking)

- **Attack Mitigation Rate**: $1 - \frac{\text{episodes reaching IMPACT}}{\text{total episodes}}$
- **False Positive Rate**: $\frac{\text{active actions on BENIGN}}{\text{BENIGN steps}}$
- **Mean Time to Contain**: Mean steps from attack start to BENIGN reset.
- **Availability Score**: $\frac{1}{1 + \sum \text{action cost}}$

## Detection metrics

RL actions are projected to stage predictions for stage recall and macro-F1. This is a *proxy* diagnostic, not a real classifier.

## Key files

- `src/training/generator_trainer.py`
- `src/training/training_manager.py`
- `src/benchmarking/metrics_collector.py`
- `src/benchmarking/benchmark_runner.py`
