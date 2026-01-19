"""
Tests for ChainIntegrator - Quasi-static relaxation solver.

Tests:
    - Relaxation to equilibrium
    - Force convergence
    - Multi-node relaxation
    - Interactive displacement
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from projects.single_fiber.src.single_fiber.chain_state import ChainState
from projects.single_fiber.src.single_fiber.chain_model import ChainModel
from projects.single_fiber.src.single_fiber.chain_integrator import (
    ChainIntegrator,
    ChainLoadingController,
    RelaxationResult
)
from projects.single_fiber.src.single_fiber.config import (
    DynamicsConfig,
    LoadingConfig,
    ModelConfig,
    HookeConfig,
    WLCConfig
)


def make_dynamics_config(dt=0.1, gamma=1.0):
    """Create dynamics configuration."""
    return DynamicsConfig(
        dt_us=dt,
        gamma_pN_us_per_nm=gamma
    )


def make_hooke_model(k=0.1, L0=50.0):
    """Create Hookean chain model."""
    return ChainModel(ModelConfig(
        law="hooke",
        hooke=HookeConfig(k_pN_per_nm=k, L0_nm=L0)
    ))


class TestRelaxationToEquilibrium:
    """Tests for relaxation convergence."""

    def test_uniform_chain_already_at_equilibrium(self):
        """Uniform chain at rest should converge immediately."""
        integrator = ChainIntegrator(make_dynamics_config())
        model = make_hooke_model(k=0.1, L0=50.0)

        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([200.0, 0.0, 0.0]),
            n_segments=4
        )

        result = integrator.relax_to_equilibrium(chain, model, [0, 4])

        assert result.converged
        assert result.iterations < 10  # Should converge quickly
        assert result.max_force_pN < 1e-6

    def test_relaxation_restores_uniform_spacing(self):
        """Perturbed chain should relax to uniform spacing."""
        integrator = ChainIntegrator(make_dynamics_config(dt=0.5, gamma=1.0))
        model = make_hooke_model(k=1.0, L0=50.0)

        # Create chain with perturbed middle nodes
        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([200.0, 0.0, 0.0]),
            n_segments=4
        )
        # Perturb internal nodes
        chain.nodes_nm[1] = np.array([30.0, 0.0, 0.0])  # Should be 50
        chain.nodes_nm[2] = np.array([120.0, 0.0, 0.0])  # Should be 100
        chain.nodes_nm[3] = np.array([140.0, 0.0, 0.0])  # Should be 150

        result = integrator.relax_to_equilibrium(
            chain, model, [0, 4],
            max_iterations=2000,
            tol_pN=1e-6
        )

        assert result.converged
        # Check nodes return to uniform spacing
        for i in range(5):
            expected_x = 50.0 * i
            assert pytest.approx(result.state.nodes_nm[i, 0], abs=0.1) == expected_x

    def test_stretched_chain_relaxes_internal_nodes(self):
        """Stretched chain should distribute strain uniformly."""
        # Use higher mobility for faster convergence
        integrator = ChainIntegrator(make_dynamics_config(dt=0.5, gamma=1.0))
        model = make_hooke_model(k=0.1, L0=50.0)

        # Create 4-segment chain, then stretch end
        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([200.0, 0.0, 0.0]),
            n_segments=4
        )
        # Stretch end node by 40 nm (10 per segment)
        chain.nodes_nm[4] = np.array([240.0, 0.0, 0.0])

        result = integrator.relax_to_equilibrium(
            chain, model, [0, 4],
            max_iterations=5000,
            tol_pN=1e-4  # Relaxed tolerance
        )

        assert result.converged
        # Each segment should be 60 nm
        for i in range(4):
            L = result.state.segment_length(i)
            assert pytest.approx(L, abs=0.5) == 60.0


class TestStepWithRelaxation:
    """Tests for the main stepping function."""

    def test_step_applies_displacement(self):
        """Step should apply displacement to end node."""
        integrator = ChainIntegrator(make_dynamics_config())
        model = make_hooke_model(k=0.1, L0=50.0)

        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([200.0, 0.0, 0.0]),
            n_segments=4
        )

        target = np.array([220.0, 0.0, 0.0])
        new_state, forces, relax = integrator.step_with_relaxation(
            chain, model, target, t_new_us=1.0
        )

        # End node should be at target
        np.testing.assert_array_almost_equal(new_state.nodes_nm[-1], target)
        # Time should be updated
        assert new_state.t_us == 1.0

    def test_step_relaxes_internal_nodes(self):
        """Step should relax internal nodes after displacement."""
        # Use higher mobility and more iterations for better convergence
        dynamics = make_dynamics_config(dt=0.5, gamma=1.0)
        integrator = ChainIntegrator(dynamics)
        integrator.max_relax_iterations = 5000
        integrator.force_tol_pN = 1e-4
        model = make_hooke_model(k=0.1, L0=50.0)

        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([200.0, 0.0, 0.0]),
            n_segments=4
        )

        # Apply 40 nm extension
        target = np.array([240.0, 0.0, 0.0])
        new_state, forces, relax = integrator.step_with_relaxation(
            chain, model, target, t_new_us=1.0
        )

        assert relax.converged
        # Internal nodes should be uniformly spaced
        for i in range(1, 4):
            expected_x = 60.0 * i  # 240/4 = 60 per segment
            assert pytest.approx(new_state.nodes_nm[i, 0], abs=1.0) == expected_x


class TestInteractiveDisplacement:
    """Tests for GUI interactive dragging."""

    def test_drag_internal_node(self):
        """Dragging internal node should relax neighbors."""
        integrator = ChainIntegrator(make_dynamics_config(dt=0.1))
        model = make_hooke_model(k=0.1, L0=50.0)

        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([200.0, 0.0, 0.0]),
            n_segments=4
        )

        # Drag node 2 (middle) sideways
        new_pos = np.array([100.0, 20.0, 0.0])
        new_state, forces, relax = integrator.apply_interactive_displacement(
            chain, model, node_idx=2, new_position=new_pos,
            fixed_nodes=[4]  # Keep end node fixed too
        )

        # Node 2 should be at new position
        np.testing.assert_array_almost_equal(new_state.nodes_nm[2], new_pos)
        # Node 0 should be fixed
        np.testing.assert_array_almost_equal(new_state.nodes_nm[0], [0.0, 0.0, 0.0])
        # Node 4 should be fixed
        np.testing.assert_array_almost_equal(new_state.nodes_nm[4], [200.0, 0.0, 0.0])


class TestLoadingController:
    """Tests for ChainLoadingController."""

    def test_target_position_at_time_zero(self):
        """Target at t=0 should be start position."""
        config = LoadingConfig(
            v_nm_per_us=1.0,
            t_end_us=100.0,
            axis=[1.0, 0.0, 0.0]
        )
        x_start = np.array([100.0, 0.0, 0.0])
        controller = ChainLoadingController(config, x_start)

        target = controller.target_position(0.0)
        np.testing.assert_array_almost_equal(target, x_start)

    def test_target_position_linear_ramp(self):
        """Target should move linearly with time."""
        config = LoadingConfig(
            v_nm_per_us=2.0,
            t_end_us=100.0,
            axis=[1.0, 0.0, 0.0]
        )
        x_start = np.array([100.0, 0.0, 0.0])
        controller = ChainLoadingController(config, x_start)

        # At t=50, displacement = 2.0 * 50 = 100 nm
        target = controller.target_position(50.0)
        expected = np.array([200.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(target, expected)

    def test_target_clamped_at_end(self):
        """Target should not exceed final position."""
        config = LoadingConfig(
            v_nm_per_us=1.0,
            t_end_us=50.0,
            axis=[1.0, 0.0, 0.0]
        )
        x_start = np.array([100.0, 0.0, 0.0])
        controller = ChainLoadingController(config, x_start)

        # At t=100 (past end), should be clamped to t=50
        target = controller.target_position(100.0)
        expected = np.array([150.0, 0.0, 0.0])  # 100 + 1*50
        np.testing.assert_array_almost_equal(target, expected)

    def test_is_complete(self):
        """is_complete should return True when past t_end."""
        config = LoadingConfig(
            v_nm_per_us=1.0,
            t_end_us=50.0
        )
        controller = ChainLoadingController(config, np.array([0.0, 0.0, 0.0]))

        assert not controller.is_complete(25.0)
        assert controller.is_complete(50.0)
        assert controller.is_complete(60.0)


class TestRuptureInRelaxation:
    """Tests for rupture detection during relaxation."""

    def test_rupture_detected_during_relaxation(self):
        """Relaxation should detect and mark ruptures."""
        integrator = ChainIntegrator(make_dynamics_config())

        # WLC model that ruptures at Lc=100
        model = ChainModel(ModelConfig(
            law="wlc",
            wlc=WLCConfig(Lp_nm=50.0, Lc_nm=100.0, rupture_at_Lc=True)
        ))

        # Create chain with segment about to rupture
        chain = ChainState.from_endpoints(
            np.array([0.0, 0.0, 0.0]),
            np.array([105.0, 0.0, 0.0]),  # L > Lc
            n_segments=1
        )

        result = integrator.relax_to_equilibrium(chain, model, [0, 1])

        # Segment should be ruptured
        assert result.state.any_ruptured()
        assert not result.state.segments[0].is_intact
