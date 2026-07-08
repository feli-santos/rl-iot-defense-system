"""Tests for the JSON->LaTeX render_tables generator."""

import re

from scripts.thesis.render_tables import (
    FALPHA,
    FCOUPLING,
    _load,
    _render_alpha_table,
    _render_coupling_table,
    _render_numbers,
    _render_tables,
)


class TestRenderTables:
    def test_load_alpha_json(self):
        fa = _load(FALPHA)
        assert "per_alpha" in fa
        assert set(fa["per_alpha"]) >= {"0.0", "0.2", "0.4", "0.6"}

    def test_load_coupling_json(self):
        fc = _load(FCOUPLING)
        assert "per_mode" in fc
        assert set(fc["per_mode"]) >= {"coupled", "outcome"}

    def test_render_numbers_headline_macros(self):
        tex = _render_numbers()
        assert r"\newcommand{\NumSeeds}" in tex
        assert r"\newcommand{\OracleCeiling}" in tex
        assert r"\newcommand{\NumTests}" in tex

    def test_render_numbers_per_alpha_macros(self):
        # Spelled-out alpha words (LaTeX macros cannot contain digits).
        tex = _render_numbers()
        for word in ("Zero", "Two", "Four", "Six"):
            assert rf"\newcommand{{\Alpha{word}PPO}}" in tex
            assert rf"\newcommand{{\Alpha{word}RF}}" in tex
            assert rf"\newcommand{{\Alpha{word}Gap}}" in tex
            assert rf"\newcommand{{\Alpha{word}Verdict}}" in tex

    def test_render_numbers_anchor_matches_json(self):
        # Numbers are mechanically derived from the canonical JSON, not pinned
        # to stale literals: verify the anchor (alpha=0) PPO mean round-trips.
        fa = _load(FALPHA)
        anchor = fa["per_alpha"]["0.0"]["ppo"]["mean"]
        tex = _render_numbers()
        assert _newcmd_value(tex, "AnchorPPO") == f"{anchor:+.1f}"

    def test_render_numbers_coupling_gaps_match_json(self):
        fc = _load(FCOUPLING)
        tex = _render_numbers()
        assert _newcmd_value(tex, "CouplingGapCoupled") == f"{fc['gap_coupled']:+.1f}"
        assert _newcmd_value(tex, "CouplingGapOutcome") == f"{fc['gap_outcome']:+.1f}"

    def test_render_tables_wraps_bodies(self):
        tex = _render_tables()
        assert r"\newcommand{\AlphaCurveTableBody}" in tex
        assert r"\newcommand{\CouplingTableBody}" in tex

    def test_alpha_table_one_row_per_alpha(self):
        # The body emits one LaTeX row (\\ terminator) per alpha level, with the
        # alpha key leading each row, derived live from the JSON.
        fa = _load(FALPHA)
        body = _render_alpha_table()
        for akey in ("0.0", "0.2", "0.4", "0.6"):
            assert f"  {akey} &" in body
        assert body.count(r"\\") == len(fa["per_alpha"])

    def test_coupling_table_rows_match_json(self):
        fc = _load(FCOUPLING)
        body = _render_coupling_table()
        for mode in ("coupled", "outcome"):
            best = fc["per_mode"][mode]["best_algo"].upper()
            assert f"  {mode.capitalize()} & {best} &" in body

    def test_no_macro_name_contains_digit(self):
        """LaTeX control sequences must be all-letters after the backslash; a
        digit ends the name and turns the rest into literal text (e.g.
        ``\\AlphaFourA2C`` parses as ``\\AlphaFourA`` followed by ``2C``),
        which silently corrupts every macro that names the A2C algorithm.
        Regression for the silent-LaTeX-failure bug observed when
        ``render_tables`` first learned to emit A2C per-alpha macros.
        """
        tex = _render_numbers()
        macro_names = re.findall(r"\\newcommand\{\\(\w+)\}", tex)
        assert macro_names, "expected at least one \\newcommand macro"
        bad = [n for n in macro_names if any(ch.isdigit() for ch in n)]
        assert not bad, f"macro names with digits are invalid LaTeX: {bad}"

    def test_a2c_macronames_spelled_out(self):
        """The A2C algorithm token is spelled out as ATwoC in macro *names*
        (the ``2`` would end the LaTeX control sequence), but macro *bodies*
        that are typeset into prose must carry the human-readable label
        ``A2C``.  Names and bodies are distinct concerns: a name is a control
        sequence (all-letters), a body is displayed text."""
        tex = _render_numbers()
        # Macro NAMES: spelled out (no digits).
        assert r"\newcommand{\AlphaFourATwoC}" in tex
        assert r"\newcommand{\AlphaFourATwoCCILow}" in tex
        assert r"\newcommand{\AlphaFourATwoCCIHigh}" in tex
        assert r"\newcommand{\CouplingATwoCOutcome}" in tex
        assert r"\newcommand{\AlphaFourA2C}" not in tex
        assert r"\newcommand{\CouplingA2C" not in tex

    def test_display_label_macros_use_readable_a2c(self):
        """The ``\\CouplingBest*`` macros expand to a label that is typeset in
        prose (e.g. "the best learned agent now A2C"), so their *body* must be
        the readable ``A2C`` -- never the macro-name spelling ``ATwoC``.
        Regression for the ``ATwoC`` prose-leak bug where the macro-name
        spelling escaped into the compiled body text."""
        tex = _render_numbers()
        assert r"\newcommand{\CouplingBestOutcome}{A2C}" in tex
        assert r"\newcommand{\CouplingBestCoupled}{DQN}" in tex
        assert r"\newcommand{\CouplingBestOutcome}{ATwoC}" not in tex


def _newcmd_value(tex: str, name: str) -> str:
    """Extract the body of ``\\newcommand{\\<name>}{<value>}`` from rendered tex."""
    marker = rf"\newcommand{{\{name}}}{{"
    start = tex.index(marker) + len(marker)
    end = tex.index("}", start)
    return tex[start:end]
