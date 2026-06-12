"""Unit tests for the first-order Markov attacker."""

import numpy as np
import pytest

from src.generator.episode_generator import EpisodeGenerator, EpisodeGeneratorConfig
from src.generator.markov_attacker import MarkovAttacker

NUM_STAGES = 5
IMPACT = NUM_STAGES - 1


class TestMarkovAttackerMatrix:
    def test_matrix_shape(self):
        attacker = MarkovAttacker()
        assert attacker.transition_matrix.shape == (NUM_STAGES, NUM_STAGES)

    def test_rows_sum_to_one(self):
        attacker = MarkovAttacker()
        rows = attacker.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(rows, np.ones(NUM_STAGES), rtol=1e-9, atol=1e-9)

    def test_impact_is_absorbing(self):
        attacker = MarkovAttacker()
        trans = attacker.transition_matrix
        assert trans[IMPACT, IMPACT] == 1.0
        assert trans[IMPACT, :IMPACT].sum() == 0.0

    def test_attack_rows_are_upper_triangular_no_regression(self):
        attacker = MarkovAttacker()
        trans = attacker.transition_matrix
        for i in range(1, NUM_STAGES):
            # No probability of moving to a strictly lower stage.
            assert trans[i, :i].sum() == 0.0

    def test_benign_row_weighted_by_distribution(self):
        # BENIGN-row distribution weighting only applies to the skip-capable
        # ablation (skip_weight > 0); the strict-sequential headline onset is
        # RECON-only. Heavily weight ACCESS (stage 2); its onset prob should
        # then dominate the other attack-onset entries in the BENIGN row.
        dist = {1: 0.01, 2: 1.0, 3: 0.01, 4: 0.01}
        attacker = MarkovAttacker(stage_distribution=dist, skip_weight=0.2)
        benign_row = attacker.transition_matrix[0]
        attack_onsets = benign_row[1:]
        assert np.argmax(attack_onsets) == 1  # index 1 within [1:] == stage 2

    def test_missing_distribution_defaults_to_quarter(self):
        # Skip-capable ablation: with no distribution, all attack-onset entries
        # share the same prior.
        attacker = MarkovAttacker(skip_weight=0.2)
        benign_row = attacker.transition_matrix[0]
        np.testing.assert_allclose(
            benign_row[1:], np.full(NUM_STAGES - 1, benign_row[1]), rtol=1e-9
        )

    def test_strict_sequential_headline_onset_is_recon_only(self):
        # Headline (skip_weight=0): BENIGN can only begin an attack at RECON.
        attacker = MarkovAttacker()
        benign_row = attacker.transition_matrix[0]
        assert benign_row[1] == pytest.approx(0.6)
        assert benign_row[2:].sum() == pytest.approx(0.0)


class TestMarkovAttackerSampling:
    def test_determinism_under_seeded_rng(self):
        a1 = MarkovAttacker()
        a2 = MarkovAttacker()
        seq1 = self._roll(a1, np.random.default_rng(123))
        seq2 = self._roll(a2, np.random.default_rng(123))
        assert seq1 == seq2

    def test_different_seeds_diverge(self):
        attacker = MarkovAttacker()
        seq1 = self._roll(attacker, np.random.default_rng(1))
        seq2 = self._roll(attacker, np.random.default_rng(2))
        assert seq1 != seq2

    def test_no_regression_in_samples(self):
        attacker = MarkovAttacker()
        rng = np.random.default_rng(7)
        stage = 1
        for _ in range(200):
            nxt = attacker.sample_next(stage, rng)
            assert nxt >= stage  # never regresses from an attack stage
            stage = nxt
            if stage == IMPACT:
                assert attacker.sample_next(stage, rng) == IMPACT  # absorbing
                break

    def test_invalid_stage_raises(self):
        attacker = MarkovAttacker()
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            attacker.sample_next(NUM_STAGES, rng)
        with pytest.raises(ValueError):
            attacker.sample_next(-1, rng)

    @staticmethod
    def _roll(attacker, rng, n=30):
        stage = 0
        out = []
        for _ in range(n):
            stage = attacker.sample_next(stage, rng)
            out.append(stage)
        return out


class TestParityWithEpisodeGenerator:
    def test_matrix_matches_episode_generator(self):
        """MarkovAttacker must reproduce EpisodeGenerator's matrix exactly.

        EpisodeGenerator applies Laplace smoothing + temperature to the raw
        stage counts before building the matrix, so we feed MarkovAttacker the
        SAME smoothed probability distribution to isolate matrix-building parity.
        """
        raw_counts = {0: 1000, 1: 300, 2: 250, 3: 200, 4: 250}
        cfg = EpisodeGeneratorConfig(num_stages=NUM_STAGES)
        gen = EpisodeGenerator(config=cfg, stage_distribution=raw_counts)
        # Use the generator's internal (smoothed) distribution for a fair match.
        smoothed = gen._stage_distribution
        attacker = MarkovAttacker(
            num_stages=NUM_STAGES,
            stage_distribution=smoothed,
            persistence_weight=cfg.persistence_weight,
            progression_weight=cfg.progression_weight,
            skip_weight=cfg.skip_weight,
        )
        np.testing.assert_allclose(
            attacker.transition_matrix,
            gen._build_transition_matrix(),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_invalid_num_stages_raises(self):
        with pytest.raises(ValueError):
            MarkovAttacker(num_stages=1)
