"""Verification tests for L-BFGS-B energy minimization solver."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.fibrinet_core_v2 import (
    WLCFiber, NetworkState, EnergyMinimizationSolver,
    HybridMechanochemicalSimulation, PhysicalConstants,
)
from src.validation.canonical_networks import small_lattice, line, square
from tests.conftest import dict_to_network_state

PC = PhysicalConstants()


def _build_solver(state):
    """Create EnergyMinimizationSolver from NetworkState."""
    active = [f for f in state.fibers if f.S > 0]
    active_nids = set()
    for f in active:
        active_nids.add(f.node_i)
        active_nids.add(f.node_j)

    fixed = {nid: pos for nid, pos in state.fixed_nodes.items() if nid in active_nids}
    partial = {nid: x for nid, x in state.partial_fixed_x.items() if nid in active_nids}
    return EnergyMinimizationSolver(active, fixed, partial)


def _make_sim(state, plasmin=0.001):
    """Create a HybridMechanochemicalSimulation (low plasmin → pure mechanics)."""
    return HybridMechanochemicalSimulation(
        initial_state=state,
        rng_seed=42,
        dt_chem=0.01,
        t_max=1.0,
        plasmin_concentration=plasmin,
    )


# ---------------------------------------------------------------------------
# Convergence (via simulation's relax_network)
# ---------------------------------------------------------------------------

class TestMinimizerConvergence:

    def test_minimizer_converges(self, lattice_network):
        """relax_network() should produce finite, non-negative energy."""
        sim = _make_sim(lattice_network)
        sim.relax_network()

        assert np.isfinite(lattice_network.energy), (
            f"Energy is not finite: {lattice_network.energy}"
        )
        assert lattice_network.energy >= 0, (
            f"Energy is negative: {lattice_network.energy}"
        )

    def test_energy_finite_positive(self):
        """After relaxation, energy should be finite and non-negative."""
        state = dict_to_network_state(
            small_lattice(3, 4), applied_strain=0.15, prestrain=True,
        )
        sim = _make_sim(state)
        sim.relax_network()

        assert np.isfinite(state.energy), f"Energy not finite: {state.energy}"
        assert state.energy >= 0, f"Energy negative: {state.energy}"

        # Energy should be in a physically reasonable range
        # For ~30 fibers at ~15% strain: total energy ~ O(1e-20) to O(1e-18) J
        assert state.energy < 1e-15, (
            f"Energy {state.energy:.3e} J seems unreasonably large"
        )


# ---------------------------------------------------------------------------
# Force balance at equilibrium
# ---------------------------------------------------------------------------

class TestEquilibriumForces:

    def test_equilibrium_forces_balance(self, lattice_network):
        """After relaxation, fiber forces should be in a physically reasonable range."""
        sim = _make_sim(lattice_network)
        sim.relax_network()

        forces = sim.compute_forces()
        max_force = max(abs(f) for f in forces.values()) if forces else 0.0

        # Forces should be finite and bounded by F_MAX
        assert max_force <= PC.F_MAX, (
            f"Max force {max_force:.3e} exceeds F_MAX {PC.F_MAX:.3e}"
        )

        # Forces should be in a physically reasonable range for µm-scale fibers
        # k_BT/xi ~ 4.28e-15 N, so piconewton range is expected
        assert max_force > 0, "All forces are zero after relaxation with applied strain"
        assert max_force < 1e-9, (
            f"Max force {max_force:.3e} N seems too large for µm-scale fibers"
        )


# ---------------------------------------------------------------------------
# Boundary constraints
# ---------------------------------------------------------------------------

class TestBoundaryConstraints:

    def test_boundary_nodes_fixed(self, lattice_network):
        """Left boundary fully fixed, right boundary X unchanged after relaxation."""
        state = lattice_network

        left_positions_before = {
            nid: state.node_positions[nid].copy()
            for nid in state.left_boundary_nodes
        }
        right_x_before = {
            nid: state.node_positions[nid][0]
            for nid in state.right_boundary_nodes
        }

        sim = _make_sim(state)
        sim.relax_network()

        # Left boundary: fully fixed (both X and Y)
        for nid in state.left_boundary_nodes:
            np.testing.assert_allclose(
                state.node_positions[nid], left_positions_before[nid], atol=1e-15,
                err_msg=f"Left boundary node {nid} moved!"
            )

        # Right boundary: X fixed (Y may move for Poisson contraction)
        for nid in state.right_boundary_nodes:
            assert abs(state.node_positions[nid][0] - right_x_before[nid]) < 1e-15, (
                f"Right boundary node {nid} X changed: "
                f"{right_x_before[nid]:.6e} → {state.node_positions[nid][0]:.6e}"
            )


# ---------------------------------------------------------------------------
# Gradient verification (analytical vs numerical)
# ---------------------------------------------------------------------------

class TestGradientAccuracy:

    def test_gradient_matches_numerical(self):
        """Analytical gradient should match finite-difference gradient."""
        state = dict_to_network_state(
            square(with_diagonals=True), applied_strain=0.05, prestrain=True,
        )
        solver = _build_solver(state)
        x0 = solver.pack_free_coords(state.node_positions)

        # Skip if no free variables
        if len(x0) == 0:
            pytest.skip("No free variables in square network")

        grad_analytical = solver.compute_gradient(x0, solver.fixed_coords)

        # Numerical gradient via central difference
        eps = 1e-8
        grad_numerical = np.zeros_like(x0)
        for i in range(len(x0)):
            x_plus = x0.copy()
            x_minus = x0.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            E_plus = solver.compute_total_energy(x_plus, solver.fixed_coords)
            E_minus = solver.compute_total_energy(x_minus, solver.fixed_coords)
            grad_numerical[i] = (E_plus - E_minus) / (2 * eps)

        # Allow relative error up to 1e-3 per component
        for i in range(len(x0)):
            denom = max(abs(grad_analytical[i]), abs(grad_numerical[i]), 1e-20)
            rel_err = abs(grad_analytical[i] - grad_numerical[i]) / denom
            assert rel_err < 1e-3, (
                f"Gradient component {i}: analytical={grad_analytical[i]:.6e}, "
                f"numerical={grad_numerical[i]:.6e}, rel_error={rel_err:.2e}"
            )


# ---------------------------------------------------------------------------
# Batch vs scalar consistency
# ---------------------------------------------------------------------------

class TestBatchScalar:

    def test_batch_wlc_matches_scalar(self, lattice_network):
        """Batch vectorized energy/force should match per-fiber scalar compute."""
        state = lattice_network
        solver = _build_solver(state)

        # Compute fiber lengths
        lengths = []
        for f in solver.fibers:
            pos_i = state.node_positions[f.node_i]
            pos_j = state.node_positions[f.node_j]
            lengths.append(float(np.linalg.norm(pos_j - pos_i)))
        lengths = np.array(lengths)

        # Batch compute
        batch_energies = solver._batch_wlc_energy(lengths)
        batch_forces = solver._batch_wlc_force(lengths)

        # Scalar compute
        for i, f in enumerate(solver.fibers):
            scalar_energy = f.compute_energy(lengths[i])
            scalar_force = f.compute_force(lengths[i])

            rel_err_e = abs(batch_energies[i] - scalar_energy) / max(abs(scalar_energy), 1e-30)
            assert rel_err_e < 1e-10, (
                f"Fiber {f.fiber_id} energy mismatch: "
                f"batch={batch_energies[i]:.6e}, scalar={scalar_energy:.6e}"
            )

            rel_err_f = abs(batch_forces[i] - scalar_force) / max(abs(scalar_force), 1e-30)
            assert rel_err_f < 1e-10, (
                f"Fiber {f.fiber_id} force mismatch: "
                f"batch={batch_forces[i]:.6e}, scalar={scalar_force:.6e}"
            )


# ---------------------------------------------------------------------------
# Prestrain verification
# ---------------------------------------------------------------------------

class TestPrestrain:

    def test_prestrain_shortens_rest_length(self):
        """Prestrained fibers have L_c = L_geometric / (1 + 0.23)."""
        state = dict_to_network_state(line(n=3, spacing=1.0), prestrain=True)
        fiber = state.fibers[0]

        # Geometric length between nodes 0 and 1 at spacing=1.0, scale=1e-6
        L_geom = 1.0 * 1e-6  # 1 µm

        expected_L_c = L_geom / (1.0 + PC.PRESTRAIN)
        assert abs(fiber.L_c - expected_L_c) < 1e-18, (
            f"L_c={fiber.L_c:.6e}, expected {expected_L_c:.6e} "
            f"(L_geom / 1.23)"
        )
