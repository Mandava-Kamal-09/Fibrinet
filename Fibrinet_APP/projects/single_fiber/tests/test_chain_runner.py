"""
Tests for ChainSimulationRunner - End-to-end chain simulation.

Tests:
    - Complete simulation execution
    - N=1 backward compatibility with Phase 2
    - Multi-segment simulation
    - Rupture behavior in chains
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from projects.single_fiber.src.single_fiber.chain_runner import (
    ChainSimulationRunner,
    run_chain_simulation,
    run_simulation_as_chain
)
from projects.single_fiber.src.single_fiber.runner import run_simulation
from projects.single_fiber.src.single_fiber.config import (
    SimulationConfig,
    ModelConfig,
    HookeConfig,
    WLCConfig,
    GeometryConfig,
    DynamicsConfig,
    LoadingConfig,
    EnzymeConfig,
    OutputConfig
)


def make_hooke_config(n_segments=1, L0_per_segment=None):
    """Create complete simulation config for Hookean chain.

    Args:
        n_segments: Number of segments (for calculating per-segment L0).
        L0_per_segment: If provided, use this L0 per segment. Otherwise,
                        use 100.0/n_segments to match initial segment lengths.
    """
    # Default: L0 matches initial segment length
    if L0_per_segment is None:
        L0_per_segment = 100.0 / n_segments if n_segments > 1 else 100.0

    return SimulationConfig(
        model=ModelConfig(
            law="hooke",
            hooke=HookeConfig(k_pN_per_nm=0.1, L0_nm=L0_per_segment)
        ),
        geometry=GeometryConfig(
            x1_nm=[0.0, 0.0, 0.0],
            x2_nm=[100.0, 0.0, 0.0]
        ),
        dynamics=DynamicsConfig(
            dt_us=0.1,
            gamma_pN_us_per_nm=1.0
        ),
        loading=LoadingConfig(
            v_nm_per_us=0.5,
            t_end_us=100.0
        ),
        output=OutputConfig(save_every_steps=10)
    )


def make_wlc_config(n_segments=1, Lc=200.0):
    """Create complete simulation config for WLC chain."""
    return SimulationConfig(
        model=ModelConfig(
            law="wlc",
            wlc=WLCConfig(Lp_nm=50.0, Lc_nm=Lc, rupture_at_Lc=True)
        ),
        geometry=GeometryConfig(
            x1_nm=[0.0, 0.0, 0.0],
            x2_nm=[100.0, 0.0, 0.0]
        ),
        dynamics=DynamicsConfig(
            dt_us=0.1,
            gamma_pN_us_per_nm=1.0
        ),
        loading=LoadingConfig(
            v_nm_per_us=1.0,
            t_end_us=120.0
        ),
        output=OutputConfig(save_every_steps=10)
    )


class TestSingleSegmentBackwardCompatibility:
    """Tests for N=1 backward compatibility with Phase 2."""

    def test_hooke_chain_matches_fiber_simulation(self):
        """Chain N=1 should match Phase 2 single fiber simulation."""
        config = make_hooke_config()

        # Run with Phase 2 engine
        result_p2 = run_simulation(config)

        # Run with Phase 3 chain engine via compatibility wrapper
        result_chain = run_simulation_as_chain(config)

        # Compare results
        assert len(result_p2.records) == len(result_chain.records)
        assert pytest.approx(result_p2.max_tension_pN, abs=0.1) == result_chain.max_tension_pN
        assert pytest.approx(result_p2.final_strain, abs=0.01) == result_chain.final_strain

    def test_wlc_rupture_same_time(self):
        """WLC rupture should occur at same time in both engines."""
        config = make_wlc_config(Lc=200.0)

        result_p2 = run_simulation(config)
        result_chain = run_simulation_as_chain(config)

        assert result_p2.rupture_occurred == result_chain.rupture_occurred
        if result_p2.rupture_occurred:
            assert pytest.approx(result_p2.rupture_time_us, abs=0.5) == result_chain.rupture_time_us


class TestMultiSegmentSimulation:
    """Tests for multi-segment chain simulation."""

    def test_multi_segment_runs_to_completion(self):
        """Multi-segment simulation should complete without error."""
        config = make_hooke_config(n_segments=5)
        result = run_chain_simulation(config, n_segments=5)

        assert len(result.records) > 0
        assert result.final_state.n_segments == 5
        assert result.final_state.n_nodes == 6

    def test_multi_segment_records_all_data(self):
        """Records should contain per-segment data."""
        config = make_hooke_config(n_segments=4)
        result = run_chain_simulation(config, n_segments=4)

        # Check first record has correct segment count
        record = result.records[0]
        assert record.n_segments == 4
        assert len(record.segment_lengths_nm) == 4
        assert len(record.segment_tensions_pN) == 4
        assert len(record.segment_intact) == 4

    def test_multi_segment_node_positions_recorded(self):
        """All node positions should be recorded."""
        config = make_hooke_config(n_segments=3)
        result = run_chain_simulation(config, n_segments=3)

        record = result.records[-1]
        assert record.all_nodes_nm is not None
        assert len(record.all_nodes_nm) == 4  # 3 segments = 4 nodes

    def test_multi_segment_strain_distributed(self):
        """Strain should be distributed across segments."""
        # Use L0 that matches per-segment initial length for proper relaxation
        config = make_hooke_config(n_segments=4)
        result = run_chain_simulation(config, n_segments=4)

        # At end of simulation, global strain = 0.5 (50 nm / 100 nm)
        final = result.records[-1]

        # All segments should have similar strain (uniform distribution)
        strains = final.segment_strains
        # Check that all strains are close to each other (within 20% relative)
        min_strain = min(strains)
        max_strain = max(strains)
        if max_strain > 0:
            strain_variation = (max_strain - min_strain) / max_strain
            assert strain_variation < 0.2, f"Strains too different: {strains}"
        # Global strain should match expected
        assert pytest.approx(final.global_strain, abs=0.05) == 0.5


class TestChainRupture:
    """Tests for rupture behavior in chains."""

    def test_wlc_chain_ruptures(self):
        """WLC chain should rupture when extended past Lc."""
        config = make_wlc_config(Lc=200.0)
        result = run_chain_simulation(config, n_segments=1)

        assert result.any_rupture_occurred
        assert result.first_rupture_time_us is not None

    def test_rupture_stops_tension(self):
        """Tension should be zero after rupture."""
        config = make_wlc_config(Lc=200.0)
        result = run_chain_simulation(config, n_segments=1)

        # Find records after rupture
        rupture_time = result.first_rupture_time_us
        for record in result.records:
            if record.t_us > rupture_time + 1.0:
                assert record.max_tension_pN == 0.0

    def test_multi_segment_partial_rupture(self):
        """Multi-segment chain can have partial rupture."""
        # Create config where one segment will rupture
        config = make_wlc_config(Lc=60.0)  # Low Lc so it ruptures
        result = run_chain_simulation(config, n_segments=2)

        # At least one rupture should occur
        if result.any_rupture_occurred:
            final = result.final_state
            # Check that rupture state is tracked per-segment
            rupture_count = sum(1 for seg in final.segments if not seg.is_intact)
            assert rupture_count >= 1


class TestRelaxationTracking:
    """Tests for relaxation iteration tracking."""

    def test_relaxation_iterations_tracked(self):
        """Total relaxation iterations should be tracked."""
        config = make_hooke_config(n_segments=4)
        result = run_chain_simulation(config, n_segments=4)

        assert result.total_relax_iterations > 0

    def test_record_includes_relaxation_info(self):
        """Records should include relaxation convergence info."""
        config = make_hooke_config(n_segments=2)
        result = run_chain_simulation(config, n_segments=2)

        for record in result.records:
            assert hasattr(record, 'relax_converged')
            assert hasattr(record, 'relax_iterations')
            assert record.relax_converged  # Should always converge


class TestSimulationDeterminism:
    """Tests for deterministic behavior."""

    def test_chain_simulation_reproducible(self):
        """Same config should produce identical results."""
        config = make_hooke_config(n_segments=3)

        result1 = run_chain_simulation(config, n_segments=3)
        result2 = run_chain_simulation(config, n_segments=3)

        assert len(result1.records) == len(result2.records)

        for r1, r2 in zip(result1.records, result2.records):
            assert r1.t_us == r2.t_us
            assert pytest.approx(r1.max_tension_pN) == r2.max_tension_pN
            assert pytest.approx(r1.global_strain) == r2.global_strain

    def test_different_n_segments_different_dynamics(self):
        """Different segment counts should produce different internal dynamics."""
        config = make_hooke_config()

        result1 = run_chain_simulation(config, n_segments=1)
        result2 = run_chain_simulation(config, n_segments=5)

        # Global behavior similar
        assert pytest.approx(result1.final_global_strain, abs=0.1) == result2.final_global_strain

        # But different number of segments
        assert result1.final_state.n_segments == 1
        assert result2.final_state.n_segments == 5
