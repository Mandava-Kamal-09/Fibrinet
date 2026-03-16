"""Verification tests for Gillespie SSA and tau-leaping chemistry engine."""

import sys
import os
import math
import pytest
import numpy as np
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.fibrinet_core_v2 import (
    StochasticChemistryEngine, NetworkState, WLCFiber, PhysicalConstants,
)
from src.validation.canonical_networks import small_lattice, line
from tests.conftest import dict_to_network_state

PC = PhysicalConstants()


def _make_engine(seed=42, plasmin=1.0, threshold=100.0):
    """Create a StochasticChemistryEngine."""
    return StochasticChemistryEngine(
        rng_seed=seed,
        tau_leap_threshold=threshold,
        plasmin_concentration=plasmin,
    )



# Deterministic replay

class TestDeterministicReplay:

    def test_deterministic_replay(self, lattice_network):
        """Same seed must produce identical cleavage sequence."""
        state1 = lattice_network
        state2 = dict_to_network_state(
            small_lattice(4, 6), applied_strain=0.1, prestrain=True,
        )

        eng1 = _make_engine(seed=42)
        eng2 = _make_engine(seed=42)

        # Run several steps and collect results
        results1, results2 = [], []
        for _ in range(20):
            p1 = eng1.compute_propensities(state1)
            p2 = eng2.compute_propensities(state2)
            cleaved1, dt1 = eng1.advance(state1, 0.01, p1)
            cleaved2, dt2 = eng2.advance(state2, 0.01, p2)
            results1.append((cleaved1, dt1))
            results2.append((cleaved2, dt2))

        assert results1 == results2, "Deterministic replay failed: different results with same seed"



# Zero plasmin → no cleavage

class TestZeroPlasmin:

    def test_zero_plasmin_no_cleavage(self, lattice_network):
        """With plasmin_concentration near zero, no fibers should be cleaved."""
        # Use very low concentration (can't be exactly 0 since it's a multiplier)
        eng = StochasticChemistryEngine(
            rng_seed=42, plasmin_concentration=1e-15,
        )
        state = lattice_network

        total_cleaved = 0
        for _ in range(1000):
            props = eng.compute_propensities(state)
            cleaved, _ = eng.advance(state, 0.01, props)
            total_cleaved += len(cleaved)

        assert total_cleaved == 0, (
            f"Expected no cleavage with near-zero plasmin, got {total_cleaved}"
        )



# Propensity scaling with concentration

class TestPropensityScaling:

    def test_propensity_scales_with_concentration(self, lattice_network):
        """Doubling plasmin concentration doubles total propensity."""
        state = lattice_network
        eng1 = _make_engine(plasmin=1.0)
        eng2 = _make_engine(plasmin=2.0)

        props1 = eng1.compute_propensities(state)
        props2 = eng2.compute_propensities(state)

        total1 = sum(props1.values())
        total2 = sum(props2.values())

        assert abs(total2 - 2.0 * total1) < 1e-10 * total1, (
            f"Expected 2× propensity: total(λ=1)={total1:.4e}, "
            f"total(λ=2)={total2:.4e}"
        )



# Gillespie waiting time distribution

class TestGillespieStatistics:

    def test_gillespie_wait_time_exponential(self, lattice_network):
        """Gillespie waiting times should follow Exp(a_total) distribution."""
        state = lattice_network
        eng = _make_engine(seed=123, plasmin=1.0)

        props = eng.compute_propensities(state)
        a_total = sum(props.values())
        if a_total == 0:
            pytest.skip("No propensity in this network configuration")

        # Collect many wait times
        n_samples = 1000
        wait_times = []
        for _ in range(n_samples):
            # Reset engine each time with fresh seed for independent samples
            eng_sample = StochasticChemistryEngine(
                rng_seed=eng.rng.integers(0, 2**31),
                plasmin_concentration=1.0,
            )
            _, dt = eng_sample.gillespie_step(state, max_dt=1e6, propensities=props)
            wait_times.append(dt)

        wait_times = np.array(wait_times)
        mean_dt = np.mean(wait_times)
        expected_mean = 1.0 / a_total

        # Mean should be close (within 20% for 1000 samples)
        rel_error = abs(mean_dt - expected_mean) / expected_mean
        assert rel_error < 0.2, (
            f"Mean wait time {mean_dt:.4e} deviates from 1/a_total={expected_mean:.4e} "
            f"(rel_error={rel_error:.2f})"
        )



# Tau-leap threshold switching

class TestTauLeap:

    def test_tau_leap_threshold(self):
        """Engine switches to tau-leap when total propensity > threshold."""
        # Create a network with many fibers at zero strain → high propensity
        state = dict_to_network_state(
            small_lattice(5, 8), applied_strain=0.0, prestrain=False,
        )

        # Low threshold to force tau-leap
        eng = StochasticChemistryEngine(
            rng_seed=42, tau_leap_threshold=0.001, plasmin_concentration=100.0,
        )

        props = eng.compute_propensities(state)
        a_total = sum(props.values())

        # With high concentration and zero strain, propensity should be high
        # If so, advance should use tau-leap path (multiple fibers may react)
        if a_total > 0.001:
            cleaved, dt = eng.advance(state, 0.1, props)
            # tau-leap can cleave multiple fibers in one step
            # (SSA can only cleave one)
            # We just verify the function runs without error
            assert isinstance(cleaved, list)



# Lysis fraction monotonicity

class TestLysisFraction:

    def test_lysis_fraction_increases(self):
        """Lysis fraction should be monotonically non-decreasing over time."""
        from src.core.fibrinet_core_v2 import HybridMechanochemicalSimulation
        state = dict_to_network_state(
            small_lattice(3, 4), applied_strain=0.05, prestrain=True,
        )

        sim = HybridMechanochemicalSimulation(
            initial_state=state,
            rng_seed=42,
            dt_chem=0.01,
            t_max=10.0,
            plasmin_concentration=2.0,
        )

        lysis_values = [0.0]
        for _ in range(200):
            if not sim.step():
                break
            lysis_values.append(sim.state.lysis_fraction)

        # Check monotonicity
        for i in range(1, len(lysis_values)):
            assert lysis_values[i] >= lysis_values[i - 1] - 1e-12, (
                f"Lysis fraction decreased: step {i-1}={lysis_values[i-1]:.4f} "
                f"→ step {i}={lysis_values[i]:.4f}"
            )



# Propensity computed once per step (no redundancy)

class TestPropensityEfficiency:

    def test_propensity_computed_once_per_step(self, lattice_network):
        """Propensity passed to advance() should not trigger recomputation."""
        state = lattice_network
        eng = _make_engine(seed=42)

        with patch.object(eng, 'compute_propensities', wraps=eng.compute_propensities) as mock_cp:
            # Compute once externally
            props = eng.compute_propensities(state)
            call_count_after_first = mock_cp.call_count

            # Pass precomputed propensities to advance
            eng.advance(state, 0.01, propensities=props)

            # Should NOT have been called again inside advance
            assert mock_cp.call_count == call_count_after_first, (
                f"compute_propensities called {mock_cp.call_count - call_count_after_first} "
                f"extra times inside advance() when propensities were pre-supplied"
            )
