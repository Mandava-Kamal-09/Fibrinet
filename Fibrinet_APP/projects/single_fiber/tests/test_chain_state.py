"""
Tests for ChainState - N-segment chain state representation.

Tests:
    - Chain creation and properties
    - Segment length/strain calculations
    - Backward compatibility with N=1
    - Rupture state management
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from projects.single_fiber.src.single_fiber.chain_state import (
    ChainState,
    SegmentState,
    chain_to_fiber_state,
    fiber_state_to_chain
)
from projects.single_fiber.src.single_fiber.state import FiberState


class TestChainStateCreation:
    """Tests for ChainState creation and basic properties."""

    def test_from_endpoints_single_segment(self):
        """Single segment chain should have 2 nodes."""
        x1 = np.array([0.0, 0.0, 0.0])
        x2 = np.array([100.0, 0.0, 0.0])
        chain = ChainState.from_endpoints(x1, x2, n_segments=1)

        assert chain.n_nodes == 2
        assert chain.n_segments == 1
        np.testing.assert_array_almost_equal(chain.nodes_nm[0], x1)
        np.testing.assert_array_almost_equal(chain.nodes_nm[1], x2)

    def test_from_endpoints_multi_segment(self):
        """Multi-segment chain should have uniformly spaced nodes."""
        x1 = np.array([0.0, 0.0, 0.0])
        x2 = np.array([100.0, 0.0, 0.0])
        chain = ChainState.from_endpoints(x1, x2, n_segments=5)

        assert chain.n_nodes == 6
        assert chain.n_segments == 5

        # Check uniform spacing
        for i in range(6):
            expected_x = 20.0 * i
            assert pytest.approx(chain.nodes_nm[i, 0], abs=1e-10) == expected_x
            assert pytest.approx(chain.nodes_nm[i, 1], abs=1e-10) == 0.0
            assert pytest.approx(chain.nodes_nm[i, 2], abs=1e-10) == 0.0

    def test_segment_lengths_uniform(self):
        """All segments should have equal initial length."""
        x1 = np.array([0.0, 0.0, 0.0])
        x2 = np.array([100.0, 0.0, 0.0])
        chain = ChainState.from_endpoints(x1, x2, n_segments=5)

        expected_L = 20.0  # 100 / 5
        for i in range(5):
            assert pytest.approx(chain.segment_length(i), abs=1e-10) == expected_L
            assert pytest.approx(chain.L_initial_nm[i], abs=1e-10) == expected_L

    def test_all_segments_initially_intact(self):
        """All segments should be intact initially."""
        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([100.0, 0.0, 0.0]),
            n_segments=3
        )
        assert chain.all_intact()
        assert not chain.any_ruptured()
        for seg in chain.segments:
            assert seg.is_intact
            assert seg.rupture_time_us is None


class TestChainStateProperties:
    """Tests for ChainState computed properties."""

    def test_segment_strain_at_rest(self):
        """Strain should be zero at rest length."""
        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([100.0, 0.0, 0.0]),
            n_segments=4
        )
        for i in range(4):
            assert pytest.approx(chain.segment_strain(i), abs=1e-10) == 0.0

    def test_segment_strain_under_extension(self):
        """Strain should be positive under extension."""
        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([100.0, 0.0, 0.0]),
            n_segments=2
        )
        # Extend end node
        chain.nodes_nm[-1] = np.array([120.0, 0.0, 0.0])

        # Only last segment is stretched
        assert pytest.approx(chain.segment_strain(0), abs=1e-10) == 0.0
        # Segment 1: L = 70, L0 = 50, strain = (70-50)/50 = 0.4
        assert pytest.approx(chain.segment_strain(1), abs=1e-10) == 0.4

    def test_global_strain(self):
        """Global strain based on end-to-end distance."""
        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([100.0, 0.0, 0.0]),
            n_segments=4
        )
        # Extend end node by 20%
        chain.nodes_nm[-1] = np.array([120.0, 0.0, 0.0])
        assert pytest.approx(chain.global_strain(), abs=1e-10) == 0.2

    def test_total_contour_length(self):
        """Total contour should equal sum of initial segments."""
        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([100.0, 0.0, 0.0]),
            n_segments=5
        )
        assert pytest.approx(chain.total_contour_length(), abs=1e-10) == 100.0

    def test_segment_direction(self):
        """Segment direction should be unit vector."""
        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([100.0, 0.0, 0.0]),
            n_segments=1
        )
        direction = chain.segment_direction(0)
        np.testing.assert_array_almost_equal(direction, [1.0, 0.0, 0.0])
        assert pytest.approx(np.linalg.norm(direction)) == 1.0


class TestChainStateRupture:
    """Tests for rupture state management."""

    def test_mark_segment_ruptured(self):
        """Marking segment as ruptured should update state."""
        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([100.0, 0.0, 0.0]),
            n_segments=3
        )
        chain.segments[1].mark_ruptured(50.0)

        assert not chain.all_intact()
        assert chain.any_ruptured()
        assert not chain.segments[1].is_intact
        assert chain.segments[1].rupture_time_us == 50.0
        # Other segments still intact
        assert chain.segments[0].is_intact
        assert chain.segments[2].is_intact

    def test_rupture_is_latched(self):
        """Rupture should be permanent - cannot heal."""
        seg = SegmentState()
        seg.mark_ruptured(10.0)
        assert not seg.is_intact
        assert seg.rupture_time_us == 10.0

        # Try to mark again - should not change
        seg.mark_ruptured(20.0)
        assert seg.rupture_time_us == 10.0  # Still first time

    def test_first_rupture_time(self):
        """First rupture time should be earliest rupture."""
        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([100.0, 0.0, 0.0]),
            n_segments=3
        )
        chain.segments[2].mark_ruptured(100.0)
        chain.segments[0].mark_ruptured(50.0)

        assert chain.first_rupture_time() == 50.0


class TestChainStateCopy:
    """Tests for deep copy functionality."""

    def test_copy_is_independent(self):
        """Copy should be independent of original."""
        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([100.0, 0.0, 0.0]),
            n_segments=2
        )
        copy = chain.copy()

        # Modify original
        chain.nodes_nm[1] = np.array([60.0, 0.0, 0.0])
        chain.segments[0].mark_ruptured(10.0)

        # Copy should be unchanged
        assert pytest.approx(copy.nodes_nm[1, 0]) == 50.0
        assert copy.segments[0].is_intact


class TestBackwardCompatibility:
    """Tests for backward compatibility with Phase 2 FiberState."""

    def test_chain_to_fiber_state(self):
        """Single-segment chain should convert to FiberState."""
        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([100.0, 0.0, 0.0]),
            n_segments=1
        )
        chain.t_us = 25.0

        fiber = chain_to_fiber_state(chain)

        np.testing.assert_array_almost_equal(fiber.x1_nm, [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(fiber.x2_nm, [100.0, 0.0, 0.0])
        assert fiber.t_us == 25.0
        assert fiber.is_intact
        assert pytest.approx(fiber.L_initial_nm) == 100.0

    def test_chain_to_fiber_rejects_multi_segment(self):
        """Converting multi-segment chain to FiberState should fail."""
        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([100.0, 0.0, 0.0]),
            n_segments=2
        )
        with pytest.raises(ValueError, match="single-segment"):
            chain_to_fiber_state(chain)

    def test_fiber_state_to_chain(self):
        """FiberState should convert to single-segment chain."""
        fiber = FiberState(
            x1_nm=np.array([0.0, 0.0, 0.0]),
            x2_nm=np.array([100.0, 0.0, 0.0]),
            t_us=15.0
        )
        chain = fiber_state_to_chain(fiber)

        assert chain.n_segments == 1
        assert chain.t_us == 15.0
        np.testing.assert_array_almost_equal(chain.nodes_nm[0], fiber.x1_nm)
        np.testing.assert_array_almost_equal(chain.nodes_nm[1], fiber.x2_nm)
