"""Held-Out Benchmark CLI scripts.

Each module under this package is a standalone entrypoint:

- :mod:`scripts.benchmark.run_test_eval`        — produces eval JSONLs.
- :mod:`scripts.benchmark.build_summary_table`  — F5.
- :mod:`scripts.benchmark.plot_stage_action_cm` — F6.
- :mod:`scripts.benchmark.plot_overhead`        — F7.
- :mod:`scripts.benchmark.plot_baselines`       — F8.

The package itself exposes nothing — invoke each module via
``python -m scripts.benchmark.<module>``.
"""
