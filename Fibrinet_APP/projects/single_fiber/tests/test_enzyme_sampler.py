"""
Tests for enzyme cleavage sampler.

Phase 4: Strain-Enzyme Coupling Lab

These tests verify:
- Poisson process sampling is correct
- Seeds provide reproducibility
- Rupture probability computation
- Multi-segment sampling
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from projects.single_fiber.src.single_fiber.enzyme_models.sampler import (
    EnzymeCleavageSampler,
    CleavageEvent,
    compute_survival_probability,
    compute_mean_rupture_time,
)


class TestEnzymeCleavageSampler:
    """Tests for the main sampler class."""

    def test_seed_reproducibility(self):
        """Same seed produces identical results."""
        sampler1 = EnzymeCleavageSampler(seed=42)
        sampler2 = EnzymeCleavageSampler(seed=42)

        results1 = [sampler1.sample_rupture(0.5, 0.1) for _ in range(100)]
        results2 = [sampler2.sample_rupture(0.5, 0.1) for _ in range(100)]

        assert results1 == results2

    def test_different_seeds_differ(self):
        """Different seeds produce different results."""
        sampler1 = EnzymeCleavageSampler(seed=42)
        sampler2 = EnzymeCleavageSampler(seed=123)

        results1 = [sampler1.sample_rupture(0.5, 0.1) for _ in range(100)]
        results2 = [sampler2.sample_rupture(0.5, 0.1) for _ in range(100)]

        # Highly unlikely to be equal with different seeds
        assert results1 != results2

    def test_zero_hazard_no_rupture(self):
        """Zero hazard rate never causes rupture."""
        sampler = EnzymeCleavageSampler(seed=42)
        for _ in range(1000):
            assert sampler.sample_rupture(hazard_rate=0.0, dt_us=1.0) is False

    def test_high_hazard_always_ruptures(self):
        """Very high hazard rate almost always causes rupture."""
        sampler = EnzymeCleavageSampler(seed=42)
        rupture_count = sum(
            sampler.sample_rupture(hazard_rate=1e6, dt_us=1.0)
            for _ in range(100)
        )
        # With λ*dt = 1e6, P = 1 - exp(-1e6) ≈ 1
        assert rupture_count == 100

    def test_statistical_distribution(self):
        """Rupture frequency matches expected Poisson distribution."""
        sampler = EnzymeCleavageSampler(seed=42)
        hazard_rate = 1.0  # 1/µs
        dt_us = 0.1  # µs
        n_samples = 10000

        # Expected probability: P = 1 - exp(-λ*dt) = 1 - exp(-0.1) ≈ 0.0952
        expected_prob = 1.0 - np.exp(-hazard_rate * dt_us)

        rupture_count = sum(
            sampler.sample_rupture(hazard_rate, dt_us)
            for _ in range(n_samples)
        )
        observed_prob = rupture_count / n_samples

        # Should be within 3 standard errors (99.7% confidence)
        std_err = np.sqrt(expected_prob * (1 - expected_prob) / n_samples)
        assert abs(observed_prob - expected_prob) < 3 * std_err


class TestMultiSegmentSampling:
    """Tests for multi-segment chain sampling."""

    def test_sample_segments_returns_list_of_events(self):
        """sample_segments returns list of CleavageEvent."""
        sampler = EnzymeCleavageSampler(seed=42)
        hazard_rates = [1.0, 1.0, 1.0]  # High rates for some events
        events = sampler.sample_segments(hazard_rates, dt_us=0.1, current_time_us=0.0)
        assert isinstance(events, list)
        for event in events:
            assert isinstance(event, CleavageEvent)

    def test_sample_segments_only_intact_rupture(self):
        """Only intact segments can rupture."""
        sampler = EnzymeCleavageSampler(seed=42)
        hazard_rates = [1e6, 1e6, 1e6]  # All high rates
        intact_mask = [True, False, True]  # Middle already ruptured
        events = sampler.sample_segments(
            hazard_rates, dt_us=1.0, current_time_us=0.0, intact_mask=intact_mask
        )
        # Middle segment (idx 1) should never appear in events
        for event in events:
            assert event.segment_idx != 1

    def test_sample_segments_records_time(self):
        """CleavageEvent records the current time."""
        sampler = EnzymeCleavageSampler(seed=42)
        hazard_rates = [1e6]  # Guaranteed rupture
        events = sampler.sample_segments(hazard_rates, dt_us=1.0, current_time_us=5.5)
        assert len(events) == 1
        assert events[0].time_us == 5.5


class TestCleavageEvent:
    """Tests for CleavageEvent dataclass."""

    def test_cleavage_event_creation(self):
        """CleavageEvent can be created with all fields."""
        event = CleavageEvent(
            segment_idx=2,
            time_us=10.5,
            hazard_rate=0.05
        )
        assert event.segment_idx == 2
        assert event.time_us == 10.5
        assert event.hazard_rate == 0.05
        assert event.cause == "enzyme"  # Default value

    def test_cleavage_event_custom_cause(self):
        """CleavageEvent can have custom cause."""
        event = CleavageEvent(
            segment_idx=0,
            time_us=5.0,
            hazard_rate=0.1,
            cause="mechanical"
        )
        assert event.cause == "mechanical"


class TestSurvivalProbability:
    """Tests for survival probability computation."""

    def test_zero_hazard_full_survival(self):
        """Zero hazard rate means 100% survival."""
        prob = compute_survival_probability(hazard_rate=0.0, time_us=100.0)
        assert prob == pytest.approx(1.0)

    def test_survival_decreases_with_time(self):
        """Survival probability decreases with time."""
        hazard_rate = 0.1
        prob1 = compute_survival_probability(hazard_rate, time_us=1.0)
        prob2 = compute_survival_probability(hazard_rate, time_us=10.0)
        prob3 = compute_survival_probability(hazard_rate, time_us=100.0)
        assert prob1 > prob2 > prob3

    def test_survival_decreases_with_hazard(self):
        """Survival probability decreases with hazard rate."""
        time_us = 10.0
        prob1 = compute_survival_probability(hazard_rate=0.01, time_us=time_us)
        prob2 = compute_survival_probability(hazard_rate=0.1, time_us=time_us)
        prob3 = compute_survival_probability(hazard_rate=1.0, time_us=time_us)
        assert prob1 > prob2 > prob3

    def test_survival_formula(self):
        """Survival matches exponential decay: S = exp(-λ*t)."""
        hazard_rate = 0.5
        time_us = 2.0
        expected = np.exp(-hazard_rate * time_us)
        actual = compute_survival_probability(hazard_rate, time_us)
        assert actual == pytest.approx(expected)


class TestMeanRuptureTime:
    """Tests for mean rupture time computation."""

    def test_mean_rupture_time_formula(self):
        """Mean rupture time is 1/λ for constant hazard."""
        hazard_rate = 0.1  # 1/µs
        expected = 1.0 / hazard_rate  # 10 µs
        actual = compute_mean_rupture_time(hazard_rate)
        assert actual == pytest.approx(expected)

    def test_zero_hazard_infinite_mean(self):
        """Zero hazard rate gives infinite mean rupture time."""
        mean_time = compute_mean_rupture_time(hazard_rate=0.0)
        assert mean_time == float('inf')

    def test_high_hazard_short_mean(self):
        """High hazard rate gives short mean rupture time."""
        mean_time = compute_mean_rupture_time(hazard_rate=100.0)
        assert mean_time == pytest.approx(0.01)


class TestReset:
    """Tests for sampler reset functionality."""

    def test_reset_reproduces_sequence(self):
        """Resetting sampler reproduces same random sequence."""
        sampler = EnzymeCleavageSampler(seed=42)

        # Generate first sequence
        results1 = [sampler.sample_rupture(0.3, 0.1) for _ in range(50)]

        # Reset and regenerate
        sampler.reset()
        results2 = [sampler.sample_rupture(0.3, 0.1) for _ in range(50)]

        assert results1 == results2

    def test_reset_with_new_seed(self):
        """Reset with new seed produces different sequence."""
        # Use high hazard rate to get varied True/False results
        sampler1 = EnzymeCleavageSampler(seed=42)
        sampler2 = EnzymeCleavageSampler(seed=999)

        results1 = [sampler1.sample_rupture(0.5, 1.0) for _ in range(100)]
        results2 = [sampler2.sample_rupture(0.5, 1.0) for _ in range(100)]

        # With high probability (P ~= 0.39), sequences should differ statistically
        # Count differences
        differences = sum(r1 != r2 for r1, r2 in zip(results1, results2))
        assert differences > 0, "Different seeds should produce different sequences"
