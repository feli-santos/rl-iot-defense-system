"""Tests for the JSON→LaTeX render_tables generator."""

from scripts.thesis.render_tables import (
    F5,
    _best_deployable_rl,
    _find_row,
    _load,
    _render_numbers,
    _render_tables,
)


class TestRenderTables:
    def test_load_existing_json(self):
        f5 = _load(F5)
        assert "rows" in f5

    def test_find_row(self):
        f5 = _load(F5)
        row = _find_row(f5["rows"], "ppo")
        assert row["policy"] == "ppo"

    def test_best_deployable_rl(self):
        f5 = _load(F5)
        best = _best_deployable_rl(f5["rows"])
        assert best["policy"] in {"dqn", "ppo", "a2c"}

    def test_render_numbers_not_empty(self):
        tex = _render_numbers()
        assert r"\newcommand{\BestAgentName}" in tex
        assert r"\newcommand{\OracleCapturePct}" in tex

    def test_render_tables_not_empty(self):
        tex = _render_tables()
        assert r"\newcommand{\BenchmarkTableBody}" in tex
        assert r"\newcommand{\LatencyTableBody}" in tex

    def test_benchmark_table_uses_compromise_and_prevention_rates(self):
        tex = _render_tables()
        assert "DQN & $+267.8$" in tex
        assert "300 & 0.463 & 0.537" in tex
        assert "300 & 0.127" not in tex
