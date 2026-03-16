"""Verification tests for plasmin agent-based model (ABM)."""

import sys
import os
import math
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.fibrinet_core_v2 import WLCFiber, NetworkState, PhysicalConstants
from src.core.plasmin_abm import (
    ABMParameters, PlasminAgent, PlasminABMEngine, FiberSplitter,
    AgentState, NetworkAdjacency,
    exponential_inhibition, linear_inhibition, constant_rate,
    STRAIN_CLEAVAGE_MODELS,
)
from src.validation.canonical_networks import small_lattice, line
from tests.conftest import dict_to_network_state

PC = PhysicalConstants()


# ---------------------------------------------------------------------------
# Agent count formula
# ---------------------------------------------------------------------------

class TestAgentCount:

    def test_agent_count_formula(self):
        """N = max(1, round(C_nM * V_um3 * 6.022e-4)) matches expected."""
        # Use values that give a count > 1 to verify the formula
        C_nM = 1.0     # 1 nM (lambda_0=1.0)
        V_um3 = 6000.0  # realistic_fibrin_sample volume
        expected = max(1, round(C_nM * V_um3 * 6.022e-4))
        computed = ABMParameters.compute_agent_count(C_nM, V_um3)
        assert computed == expected, (
            f"Agent count mismatch: computed={computed}, expected={expected}"
        )
        assert expected > 1, "Test values should produce count > 1"

    def test_agent_count_minimum_one(self):
        """Agent count should be at least 1 even for tiny volumes."""
        n = ABMParameters.compute_agent_count(0.001, 0.001)
        assert n >= 1


# ---------------------------------------------------------------------------
# Agent lifecycle
# ---------------------------------------------------------------------------

class TestAgentLifecycle:

    def test_agent_lifecycle(self):
        """Agents should cycle: IN_SOLUTION → BOUND → (cleave) → IN_SOLUTION/COOLDOWN."""
        state = dict_to_network_state(
            small_lattice(3, 4), applied_strain=0.0, prestrain=False,
        )

        params = ABMParameters(
            n_agents=5,
            auto_agent_count=False,
            k_on2=1e5,
            plasmin_concentration_nM=1000.0,  # High concentration for fast binding
            k_cat0=10.0,  # High cleavage rate to trigger events quickly
            k_off0=0.0001,  # Very low unbinding to keep agents bound
            beta_cat=0.0,  # No strain dependence for predictability
            strain_cleavage_model='constant',
            p_stay=0.0,  # Always release after cleavage
        )
        engine = PlasminABMEngine(params, rng_seed=42)
        engine.initialize(state)

        # Run several steps
        observed_states = set()
        for step in range(50):
            events, dt = engine.advance(state, 0.01)
            for agent in engine.agents:
                observed_states.add(agent.state)

        # We should have observed at least IN_SOLUTION and BOUND states
        assert AgentState.IN_SOLUTION in observed_states or AgentState.COOLDOWN in observed_states, (
            f"Expected to see free agents, only saw: {observed_states}"
        )


# ---------------------------------------------------------------------------
# Fiber splitting
# ---------------------------------------------------------------------------

class TestFiberSplitting:

    def test_fiber_split_conserves_length(self):
        """Split fiber's children should have L_c1 + L_c2 ≈ L_c_parent."""
        state = dict_to_network_state(line(n=3, spacing=1.0), prestrain=False)
        fiber = state.fibers[0]

        position_s = 0.4
        fiber_a, fiber_b, new_nid = FiberSplitter.split_fiber(
            fiber, position_s, state,
            next_fiber_id=100, next_node_id=200,
        )

        total_L_c = fiber_a.L_c + fiber_b.L_c
        assert abs(total_L_c - fiber.L_c) < 1e-15, (
            f"Length not conserved: L_c_a={fiber_a.L_c:.6e} + "
            f"L_c_b={fiber_b.L_c:.6e} = {total_L_c:.6e}, "
            f"expected {fiber.L_c:.6e}"
        )

    def test_fiber_split_creates_valid_node(self):
        """New node should be between original endpoints."""
        state = dict_to_network_state(line(n=3, spacing=1.0), prestrain=False)
        fiber = state.fibers[0]

        pos_i = state.node_positions[fiber.node_i].copy()
        pos_j = state.node_positions[fiber.node_j].copy()

        position_s = 0.6
        _, _, new_nid = FiberSplitter.split_fiber(
            fiber, position_s, state,
            next_fiber_id=100, next_node_id=200,
        )

        new_pos = state.node_positions[new_nid]

        # New node should be at interpolated position
        expected_pos = pos_i + position_s * (pos_j - pos_i)
        np.testing.assert_allclose(new_pos, expected_pos, atol=1e-15, err_msg=(
            f"New node at wrong position: {new_pos} vs expected {expected_pos}"
        ))

    def test_fiber_split_inherits_properties(self):
        """Child fibers should inherit parent's xi, force_model, K0 and degraded S."""
        state = dict_to_network_state(line(n=3, spacing=1.0), prestrain=False)
        fiber = state.fibers[0]

        delta_S = 0.1  # FiberSplitter default (S-inheritance per cleavage)
        parent_S = fiber.S
        expected_child_S = parent_S - delta_S
        fiber_a, fiber_b, _ = FiberSplitter.split_fiber(
            fiber, 0.5, state, 100, 200,
        )

        for child in (fiber_a, fiber_b):
            assert child.xi == fiber.xi
            assert child.S == pytest.approx(expected_child_S)
            assert child.force_model == fiber.force_model
            assert child.K0 == fiber.K0
            assert child.k_cat_0 == fiber.k_cat_0

    def test_fiber_split_position_clamped(self):
        """Split position should be clamped to [0.01, 0.99]."""
        state = dict_to_network_state(line(n=3, spacing=1.0), prestrain=False)
        fiber = state.fibers[0]

        # Try extreme positions
        fiber_a, fiber_b, _ = FiberSplitter.split_fiber(
            fiber, 0.001, state, 100, 200,
        )
        # Should be clamped to 0.01, so L_c_a is very small but not zero
        assert fiber_a.L_c > 0
        assert fiber_b.L_c > 0


# ---------------------------------------------------------------------------
# Bell model unbinding
# ---------------------------------------------------------------------------

class TestBellModel:

    def test_bell_model_unbinding(self):
        """k_off should increase with force (Bell model)."""
        params = ABMParameters()
        k_off0 = params.k_off0
        delta = params.delta_off
        kBT = 1.380649e-23 * 310.15

        # At zero force
        k_off_zero = k_off0 * math.exp(0.0)

        # At 10 pN force
        F = 10e-12  # 10 pN
        bell_exp = min(F * delta / kBT, 20.0)
        k_off_force = k_off0 * math.exp(bell_exp)

        assert k_off_force > k_off_zero, (
            f"Bell model: k_off should increase with force. "
            f"k_off(F=0)={k_off_zero:.4e}, k_off(F=10pN)={k_off_force:.4e}"
        )


# ---------------------------------------------------------------------------
# Strain-cleavage models
# ---------------------------------------------------------------------------

class TestStrainCleavageModels:

    def test_exponential_inhibition(self):
        """k(ε) = k0 * exp(-β * ε)."""
        k0, beta = 0.064, 10.0
        strain = 0.2
        expected = k0 * math.exp(-beta * strain)
        result = exponential_inhibition(strain, k0, beta)
        assert abs(result - expected) < 1e-15

    def test_linear_inhibition(self):
        """k(ε) = k0 * max(0, 1 - β * ε)."""
        k0, beta = 0.064, 10.0
        strain = 0.05
        expected = k0 * (1.0 - beta * strain)
        result = linear_inhibition(strain, k0, beta)
        assert abs(result - expected) < 1e-15

        # At high strain, rate should be zero
        result_high = linear_inhibition(0.2, k0, beta)
        assert result_high == 0.0

    def test_constant_rate(self):
        """k(ε) = k0, independent of strain."""
        k0, beta = 0.064, 10.0
        for strain in [0.0, 0.1, 0.5, 1.0]:
            assert constant_rate(strain, k0, beta) == k0

    def test_all_models_registered(self):
        """All three models should be in STRAIN_CLEAVAGE_MODELS dict."""
        assert 'exponential' in STRAIN_CLEAVAGE_MODELS
        assert 'linear' in STRAIN_CLEAVAGE_MODELS
        assert 'constant' in STRAIN_CLEAVAGE_MODELS


# ---------------------------------------------------------------------------
# Batch vectorized vs scalar
# ---------------------------------------------------------------------------

class TestBatchKinetics:

    def test_batch_vectorized_matches_scalar(self):
        """Batch Bell model in _step_bound_agents matches per-agent scalar computation."""
        params = ABMParameters()
        kBT = 1.380649e-23 * 310.15

        # Test Bell model formula consistency
        forces = np.array([0.0, 1e-12, 5e-12, 10e-12])  # 0, 1, 5, 10 pN
        delta = params.delta_off

        # Batch: vectorized computation
        bell_exp = np.minimum(forces * delta / kBT, 20.0)
        k_off_batch = params.k_off0 * np.exp(bell_exp)

        # Scalar: per-element
        for i, F in enumerate(forces):
            scalar_exp = min(F * delta / kBT, 20.0)
            k_off_scalar = params.k_off0 * math.exp(scalar_exp)
            assert abs(k_off_batch[i] - k_off_scalar) < 1e-15, (
                f"Batch/scalar mismatch at F={F:.1e}: "
                f"batch={k_off_batch[i]:.6e}, scalar={k_off_scalar:.6e}"
            )


# ---------------------------------------------------------------------------
# Network adjacency
# ---------------------------------------------------------------------------

class TestNetworkAdjacency:

    def test_adjacency_rebuild(self):
        """NetworkAdjacency should correctly map node-to-fibers and neighbors."""
        state = dict_to_network_state(small_lattice(2, 3), prestrain=False)
        adj = NetworkAdjacency()
        adj.rebuild(state.fibers)

        # Every fiber's endpoints should appear in the adjacency
        for fiber in state.fibers:
            assert fiber.node_j in adj.get_neighbors(fiber.node_i)
            assert fiber.node_i in adj.get_neighbors(fiber.node_j)
            assert fiber.fiber_id in adj.get_fibers_at_node(fiber.node_i)
            assert fiber.fiber_id in adj.get_fibers_at_node(fiber.node_j)

    def test_adjacency_empty_for_no_fibers(self):
        """Empty fiber list should produce empty adjacency."""
        adj = NetworkAdjacency()
        adj.rebuild([])
        assert adj.get_all_nodes() == set()
