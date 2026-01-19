"""
Tests for ChainModel - Force computation for N-segment chains.

Tests:
    - Force computation for single and multi-segment chains
    - Nodal force balance
    - Rupture detection
    - Backward compatibility with FiberModel
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from projects.single_fiber.src.single_fiber.chain_state import ChainState
from projects.single_fiber.src.single_fiber.chain_model import ChainModel, ChainForceOutput
from projects.single_fiber.src.single_fiber.config import ModelConfig, HookeConfig, WLCConfig


def make_hooke_config(k=0.1, L0=100.0):
    """Create Hookean model config."""
    return ModelConfig(
        law="hooke",
        hooke=HookeConfig(k_pN_per_nm=k, L0_nm=L0)
    )


def make_wlc_config(Lp=50.0, Lc=200.0):
    """Create WLC model config."""
    return ModelConfig(
        law="wlc",
        wlc=WLCConfig(Lp_nm=Lp, Lc_nm=Lc, rupture_at_Lc=True)
    )


class TestHookeanChainForces:
    """Tests for Hookean spring chains."""

    def test_single_segment_at_rest(self):
        """Single segment at rest length should have zero tension."""
        config = make_hooke_config(k=0.1, L0=100.0)
        model = ChainModel(config)

        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([100.0, 0.0, 0.0]),
            n_segments=1
        )

        forces = model.compute_forces(chain)

        assert pytest.approx(forces.max_tension_pN, abs=1e-10) == 0.0
        assert not forces.any_should_rupture

    def test_single_segment_stretched(self):
        """Stretched segment should have positive tension."""
        config = make_hooke_config(k=0.1, L0=100.0)
        model = ChainModel(config)

        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([150.0, 0.0, 0.0]),  # Stretched by 50 nm
            n_segments=1
        )
        chain.L_initial_nm = [100.0]  # Force initial length

        forces = model.compute_forces(chain)

        # T = k * (L - L0) = 0.1 * 50 = 5 pN
        assert pytest.approx(forces.max_tension_pN, abs=1e-10) == 5.0

    def test_multi_segment_uniform_stretch(self):
        """Uniformly stretched chain should have equal tensions."""
        config = make_hooke_config(k=0.1, L0=50.0)  # L0 per segment
        model = ChainModel(config)

        # Create 4-segment chain, each segment stretched equally
        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([240.0, 0.0, 0.0]),  # 60 nm per segment
            n_segments=4
        )

        forces = model.compute_forces(chain)

        # Each segment: L = 60, L0 = 50, T = 0.1 * 10 = 1 pN
        for sf in forces.segment_forces:
            assert pytest.approx(sf.tension_pN, abs=0.1) == 1.0

    def test_nodal_force_balance(self):
        """Internal nodes should have balanced forces (sum ≈ 0)."""
        config = make_hooke_config(k=0.1, L0=50.0)
        model = ChainModel(config)

        # Uniform chain at rest
        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([200.0, 0.0, 0.0]),
            n_segments=4
        )

        forces = model.compute_forces(chain)

        # Internal nodes (1, 2, 3) should have zero net force
        for i in range(1, 4):
            f_mag = np.linalg.norm(forces.node_forces_pN[i])
            assert pytest.approx(f_mag, abs=1e-10) == 0.0


class TestWLCChainForces:
    """Tests for WLC chains."""

    def test_wlc_tension_increases_with_extension(self):
        """WLC tension should increase with extension."""
        config = make_wlc_config(Lp=50.0, Lc=100.0)
        model = ChainModel(config)

        tensions = []
        for extension in [0.5, 0.7, 0.9]:
            L = 100.0 * extension  # fraction of Lc
            chain = ChainState.from_endpoints(
                np.array([0.0, 0.0, 0.0]),
                np.array([L, 0.0, 0.0]),
                n_segments=1
            )
            forces = model.compute_forces(chain)
            tensions.append(forces.max_tension_pN)

        # Tension should increase monotonically
        assert tensions[0] < tensions[1] < tensions[2]

    def test_wlc_rupture_at_contour_length(self):
        """WLC should rupture when L >= Lc."""
        config = make_wlc_config(Lp=50.0, Lc=100.0)
        model = ChainModel(config)

        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([105.0, 0.0, 0.0]),  # L > Lc
            n_segments=1
        )

        forces = model.compute_forces(chain)

        assert forces.any_should_rupture
        assert 0 in forces.rupture_indices


class TestRupturedSegments:
    """Tests for ruptured segment handling."""

    def test_ruptured_segment_zero_tension(self):
        """Ruptured segment should have zero tension."""
        config = make_hooke_config()
        model = ChainModel(config)

        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([150.0, 0.0, 0.0]),
            n_segments=1
        )
        chain.segments[0].mark_ruptured(10.0)

        forces = model.compute_forces(chain)

        assert pytest.approx(forces.segment_forces[0].tension_pN) == 0.0
        assert forces.segment_forces[0].reason == "already_ruptured"

    def test_partial_chain_rupture(self):
        """One ruptured segment should not affect others."""
        config = make_hooke_config(k=0.1, L0=50.0)
        model = ChainModel(config)

        # Create 3-segment chain
        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([180.0, 0.0, 0.0]),  # Each segment 60 nm
            n_segments=3
        )
        # Rupture middle segment
        chain.segments[1].mark_ruptured(10.0)

        forces = model.compute_forces(chain)

        # Segment 0 and 2 should have tension, segment 1 should not
        assert forces.segment_forces[0].tension_pN > 0
        assert forces.segment_forces[1].tension_pN == 0
        assert forces.segment_forces[2].tension_pN > 0


class TestForceOutputStructure:
    """Tests for ChainForceOutput data structure."""

    def test_output_has_correct_shape(self):
        """Force output should have correct dimensions."""
        config = make_hooke_config()
        model = ChainModel(config)

        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([100.0, 0.0, 0.0]),
            n_segments=5
        )

        forces = model.compute_forces(chain)

        assert len(forces.segment_forces) == 5
        assert forces.node_forces_pN.shape == (6, 3)

    def test_max_tension_correct(self):
        """Max tension should be maximum across all segments."""
        config = make_hooke_config(k=0.1, L0=100.0)
        model = ChainModel(config)

        # Create non-uniform chain
        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([300.0, 0.0, 0.0]),
            n_segments=2
        )
        # Move middle node to create different segment lengths
        chain.nodes_nm[1] = np.array([100.0, 0.0, 0.0])  # Seg 0: 100nm, Seg 1: 200nm

        forces = model.compute_forces(chain)

        # Segment 1 has larger extension (200-100=100) vs segment 0 (100-100=0)
        expected_max = 0.1 * 100  # k * (L - L0) for segment 1
        assert pytest.approx(forces.max_tension_pN, abs=0.1) == expected_max
