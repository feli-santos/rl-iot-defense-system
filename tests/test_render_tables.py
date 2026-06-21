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
        # Derive expectations from the live canonical JSON so the test verifies
        # the *structure* of the rendered table (reward, n_episodes, compromise,
        # prevention columns) rather than pinning stale literal numbers that
        # shift on every data regeneration.
        f5 = _load(F5)
        dqn = _find_row(f5["rows"], "dqn")
        tex = _render_tables()
        # The DQN row renders with its current reward from the JSON.
        assert f"DQN & ${dqn['mean_reward']:+.1f}$" in tex
        # The benchmark table carries compromise_rate + prevention_rate columns
        # (the retired mitigated_impact_rate must not appear as a trailing pair).
        assert (
            f"{dqn['n_episodes']} & {dqn['compromise_rate']:.3f} & "
            f"{dqn['prevention_rate']:.3f}"
        ) in tex
