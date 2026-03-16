"""Tests for the three-regime constitutive model (sigmoid blend)."""

import pytest
import numpy as np
import math
from dataclasses import replace

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.fibrinet_core_v2 import (
    WLCFiber, PhysicalConstants, PC, EnergyMinimizationSolver,
    HybridMechanochemicalSimulation, NetworkState,
    check_left_right_connectivity,
)


@pytest.fixture(autouse=True)
def restore_pc():
    """Restore PC state after each test."""
    orig = {attr: getattr(PC, attr) for attr in [
        'THREE_REGIME_ENABLED', 'MAX_STRAIN', 'F_MAX',
        'SIGMOID_EPSILON_MID', 'SIGMOID_DELTA_EPSILON', 'Y_A_STRONG', 'RUPTURE_STRAIN',
    ]}
    yield
    for attr, val in orig.items():
        setattr(PC, attr, val)


def make_fiber(L_c=5e-6, diameter_nm=130.0, force_model='wlc', S=1.0, fiber_id=0):
    """Create a WLCFiber with given parameters."""
    return WLCFiber(fiber_id=fiber_id, node_i=0, node_j=1, L_c=L_c,
                    diameter_nm=diameter_nm, force_model=force_model, S=S)


class TestKillSwitch:
    """THREE_REGIME_ENABLED=False must produce identical behavior to original WLC."""

    def test_force_identical_when_disabled(self):
        """Kill-switch off: force at all strains ≤ 0.99 must be unchanged."""
        PC.THREE_REGIME_ENABLED = False
        fiber = make_fiber()
        for eps in [0.0, 0.1, 0.23, 0.5, 0.8, 0.95, 0.99]:
            x = fiber.L_c * (1 + eps)
            F = fiber.compute_force(x)
            # Verify against manual WLC calculation
            strain = min(eps, PC.MAX_STRAIN)
            ome = 1.0 - strain
            F_expected = fiber.S * (PC.k_B_T / fiber.xi) * (
                1.0 / (4.0 * ome**2) - 0.25 + strain)
            F_expected = min(F_expected, PC.F_MAX)
            assert abs(F - F_expected) < 1e-15, f"Mismatch at eps={eps}"

    def test_energy_identical_when_disabled(self):
        """Kill-switch off: energy at all strains ≤ 0.99 must be unchanged."""
        PC.THREE_REGIME_ENABLED = False
        fiber = make_fiber()
        for eps in [0.0, 0.1, 0.23, 0.5, 0.8, 0.95]:
            x = fiber.L_c * (1 + eps)
            U = fiber.compute_energy(x)
            strain = min(eps, PC.MAX_STRAIN)
            ome = 1.0 - strain
            U_expected = fiber.S * (PC.k_B_T * fiber.L_c / fiber.xi) * (
                1.0 / (4.0 * ome) - 0.25 - strain / 4.0 + strain**2 / 2.0)
            assert abs(U - U_expected) < 1e-30, f"Mismatch at eps={eps}"

    def test_safe_strain_capped_at_max_strain_when_disabled(self):
        """Kill-switch off: _safe_strain caps at MAX_STRAIN (0.99)."""
        PC.THREE_REGIME_ENABLED = False
        fiber = make_fiber()
        x = fiber.L_c * 3.0  # eps = 2.0
        assert fiber._safe_strain(x) == PC.MAX_STRAIN


class TestSigmoidWeight:
    """Sigmoid blend weight w(eps) = 0.5*(1+tanh((eps-eps_mid)/delta_eps))."""

    def test_weight_low_at_half_strain(self):
        """At eps=0.5, w < 0.01 (WLC-dominated)."""
        w = 0.5 * (1.0 + math.tanh(
            (0.5 - PC.SIGMOID_EPSILON_MID) / PC.SIGMOID_DELTA_EPSILON))
        assert w < 0.01, f"w(0.5) = {w}, expected < 0.01"

    def test_weight_half_at_midpoint(self):
        """At eps=SIGMOID_EPSILON_MID, w = 0.5 +/- 0.001."""
        w = 0.5 * (1.0 + math.tanh(
            (PC.SIGMOID_EPSILON_MID - PC.SIGMOID_EPSILON_MID) / PC.SIGMOID_DELTA_EPSILON))
        assert abs(w - 0.5) < 0.001, f"w(eps_mid) = {w}, expected 0.5"

    def test_weight_high_at_double_strain(self):
        """At eps=2.0, w > 0.99 (backbone-dominated)."""
        w = 0.5 * (1.0 + math.tanh(
            (2.0 - PC.SIGMOID_EPSILON_MID) / PC.SIGMOID_DELTA_EPSILON))
        assert w > 0.99, f"w(2.0) = {w}, expected > 0.99"


class TestSigmoidForce:
    """Sigmoid blend force at physiological and high strain."""

    def test_wlc_dominates_at_physiological_strain(self):
        """At eps=0.23 (physiological), sigmoid force within 1% of pure WLC."""
        PC.THREE_REGIME_ENABLED = True
        fiber = make_fiber()
        eps = 0.23
        x = fiber.L_c * (1 + eps)
        F_sigmoid = fiber.compute_force(x)

        # Pure WLC reference
        PC.THREE_REGIME_ENABLED = False
        F_wlc = fiber.compute_force(x)

        rel_err = abs(F_sigmoid - F_wlc) / abs(F_wlc)
        assert rel_err < 0.01, (
            f"Sigmoid not transparent at eps=0.23: F_sig={F_sigmoid:.6e}, "
            f"F_wlc={F_wlc:.6e}, rel_err={rel_err:.4f}")

    def test_backbone_dominates_at_high_strain(self):
        """At eps=2.0, sigmoid force within 5% of pure backbone K_bb*eps."""
        PC.THREE_REGIME_ENABLED = True
        fiber = make_fiber()
        eps = 2.0
        x = fiber.L_c * (1 + eps)
        F = fiber.compute_force(x)

        # Pure backbone reference
        d_m = fiber.diameter_nm * 1e-9
        A = math.pi * (d_m / 2.0) ** 2
        K_bb = PC.Y_A_STRONG * A / fiber.L_c
        F_backbone = K_bb * eps

        rel_err = abs(F - F_backbone) / F_backbone
        assert rel_err < 0.05, (
            f"Backbone not dominant at eps=2.0: F={F:.6e}, "
            f"F_bb={F_backbone:.6e}, rel_err={rel_err:.4f}")


class TestContinuity:
    """Force curve must be C0 continuous within each regime."""

    def test_wlc_regime_continuous(self):
        """Below MAX_STRAIN: force is C0 continuous (pure WLC path)."""
        PC.THREE_REGIME_ENABLED = True
        fiber = make_fiber()
        delta = 1e-11
        for eps in [0.1, 0.3, 0.5, 0.8, 0.95]:
            x = fiber.L_c * (1 + eps)
            F_center = fiber.compute_force(x)
            F_plus = fiber.compute_force(x + delta)
            F_minus = fiber.compute_force(x - delta)
            if F_center > 0:
                rel_plus = abs(F_plus - F_center) / F_center
                rel_minus = abs(F_minus - F_center) / F_center
                assert rel_plus < 0.001, (
                    f"WLC discontinuity at eps={eps}: rel_plus={rel_plus:.6f}")
                assert rel_minus < 0.001, (
                    f"WLC discontinuity at eps={eps}: rel_minus={rel_minus:.6f}")

    def test_sigmoid_regime_continuous(self):
        """Above MAX_STRAIN: sigmoid blend is C0 continuous (no jumps)."""
        PC.THREE_REGIME_ENABLED = True
        fiber = make_fiber()
        delta = 1e-11
        for eps in [1.0, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5]:
            x = fiber.L_c * (1 + eps)
            F_center = fiber.compute_force(x)
            F_plus = fiber.compute_force(x + delta)
            F_minus = fiber.compute_force(x - delta)
            if F_center > 0:
                rel_plus = abs(F_plus - F_center) / F_center
                rel_minus = abs(F_minus - F_center) / F_center
                assert rel_plus < 0.001, (
                    f"Sigmoid discontinuity at eps={eps}: rel_plus={rel_plus:.6f}")
                assert rel_minus < 0.001, (
                    f"Sigmoid discontinuity at eps={eps}: rel_minus={rel_minus:.6f}")


class TestMonotonicity:
    """Force must be monotonically increasing across all regimes."""

    def test_force_monotonically_increasing(self):
        """Sample 20 points from ε=0 to ε=2.7: each must be > previous."""
        PC.THREE_REGIME_ENABLED = True
        fiber = make_fiber()
        strains = np.linspace(0.0, 2.7, 20)
        forces = [fiber.compute_force(fiber.L_c * (1 + eps)) for eps in strains]
        for i in range(1, len(forces)):
            assert forces[i] > forces[i-1], (
                f"Not monotonic at ε={strains[i]:.3f}: F={forces[i]:.6e} <= F={forces[i-1]:.6e}")


class TestEnergyConsistency:
    """dU/dε ≈ F × L_c (finite difference check)."""

    def test_energy_force_consistency_all_regimes(self):
        """Finite difference dU/dx ≈ F at 5 strain values across all regimes."""
        PC.THREE_REGIME_ENABLED = True
        fiber = make_fiber()
        dx = 1e-10  # Small perturbation in meters

        for eps in [0.3, 0.8, 1.2, 1.8, 2.5]:
            x = fiber.L_c * (1 + eps)
            F = fiber.compute_force(x)
            U_plus = fiber.compute_energy(x + dx)
            U_minus = fiber.compute_energy(x - dx)
            dU_dx = (U_plus - U_minus) / (2 * dx)

            if abs(F) > 1e-20:
                rel_err = abs(dU_dx - F) / abs(F)
                assert rel_err < 0.01, (
                    f"Energy-force mismatch at ε={eps}: dU/dx={dU_dx:.6e}, F={F:.6e}, rel_err={rel_err:.4f}")


class TestRupture:
    """Fiber at ε ≥ REGIME3_RUPTURE_STRAIN should return 0 force."""

    def test_rupture_signal(self):
        """compute_force returns 0.0 at rupture strain."""
        PC.THREE_REGIME_ENABLED = True
        fiber = make_fiber()
        x = fiber.L_c * (1 + PC.RUPTURE_STRAIN)
        assert fiber.compute_force(x) == 0.0

    def test_rupture_energy_zero(self):
        """compute_energy returns 0.0 at rupture strain."""
        PC.THREE_REGIME_ENABLED = True
        fiber = make_fiber()
        x = fiber.L_c * (1 + PC.RUPTURE_STRAIN)
        assert fiber.compute_energy(x) == 0.0


class TestDiameterScaling:
    """Larger diameter → higher backbone force (proportional to d²)."""

    def test_diameter_squared_scaling(self):
        """Doubling diameter should quadruple K_bb = Y_A * A / L_c."""
        d1 = 130e-9
        d2 = 260e-9
        L_c = 5e-6
        A1 = math.pi * (d1 / 2.0) ** 2
        A2 = math.pi * (d2 / 2.0) ** 2
        K_bb_130 = PC.Y_A_STRONG * A1 / L_c
        K_bb_260 = PC.Y_A_STRONG * A2 / L_c
        ratio = K_bb_260 / K_bb_130
        assert abs(ratio - 4.0) < 1e-10, f"Expected 4x scaling, got {ratio:.4f}"

    def test_larger_diameter_higher_force(self):
        """At ε=1.25, 260nm fiber has higher force than 130nm fiber."""
        PC.THREE_REGIME_ENABLED = True
        eps = 1.25
        fiber_small = make_fiber(diameter_nm=130.0)
        fiber_large = make_fiber(diameter_nm=260.0)
        x = fiber_small.L_c * (1 + eps)
        F_small = fiber_small.compute_force(x)
        F_large = fiber_large.compute_force(x)
        assert F_large > F_small


class TestBatchConsistency:
    """Batch vectorized methods must match scalar compute_force/energy."""

    def _make_solver_and_lengths(self, strains):
        """Create a solver with fibers at various strains."""
        PC.THREE_REGIME_ENABLED = True
        n = len(strains)
        fibers = []
        fixed = {0: np.array([0.0, 0.0])}
        for i, eps in enumerate(strains):
            L_c = 5e-6
            x = L_c * (1 + eps)
            fib = WLCFiber(fiber_id=i, node_i=0, node_j=i+1, L_c=L_c,
                           diameter_nm=130.0 + i*10.0)
            fibers.append(fib)
            fixed[i+1] = np.array([x, 0.0])

        solver = EnergyMinimizationSolver(fibers, fixed)
        lengths = np.array([fibers[i].L_c * (1 + strains[i]) for i in range(n)])
        return solver, fibers, lengths

    def test_batch_force_matches_scalar(self):
        """_batch_wlc_force matches compute_force for fibers across all regimes."""
        strains = [0.1, 0.5, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2, 2.6, 2.79]
        solver, fibers, lengths = self._make_solver_and_lengths(strains)

        batch_forces = solver._batch_wlc_force(lengths)
        for i, fib in enumerate(fibers):
            scalar_F = fib.compute_force(lengths[i])
            assert abs(batch_forces[i] - scalar_F) < 1e-12, (
                f"Batch/scalar force mismatch at ε={strains[i]}: "
                f"batch={batch_forces[i]:.6e}, scalar={scalar_F:.6e}")

    def test_batch_energy_matches_scalar(self):
        """_batch_wlc_energy matches compute_energy for fibers across all regimes."""
        strains = [0.1, 0.5, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2, 2.6, 2.79]
        solver, fibers, lengths = self._make_solver_and_lengths(strains)

        batch_energies = solver._batch_wlc_energy(lengths)
        for i, fib in enumerate(fibers):
            scalar_U = fib.compute_energy(lengths[i])
            assert abs(batch_energies[i] - scalar_U) < 1e-12, (
                f"Batch/scalar energy mismatch at ε={strains[i]}: "
                f"batch={batch_energies[i]:.6e}, scalar={scalar_U:.6e}")


class TestIntegration:
    """Integration: small network runs with THREE_REGIME_ENABLED."""

    def _make_small_network(self):
        """Create a minimal 3-node, 2-fiber network."""
        fibers = [
            WLCFiber(fiber_id=0, node_i=0, node_j=1, L_c=4e-6, diameter_nm=130.0),
            WLCFiber(fiber_id=1, node_i=1, node_j=2, L_c=4e-6, diameter_nm=130.0),
        ]
        node_positions = {
            0: np.array([0.0, 0.0]),
            1: np.array([5e-6, 0.0]),
            2: np.array([10e-6, 0.0]),
        }
        state = NetworkState(
            time=0.0,
            fibers=fibers,
            node_positions=node_positions,
            fixed_nodes={0: np.array([0.0, 0.0])},
            partial_fixed_x={2: 10e-6},
            left_boundary_nodes={0},
            right_boundary_nodes={2},
        )
        return state

    def test_simulation_runs_with_three_regime(self):
        """Simulation with THREE_REGIME_ENABLED=True runs without error."""
        PC.THREE_REGIME_ENABLED = True
        state = self._make_small_network()
        sim = HybridMechanochemicalSimulation(
            initial_state=state, rng_seed=42, dt_chem=0.1, t_max=5.0,
            plasmin_concentration=10.0)
        # Run a few steps
        for _ in range(10):
            if not sim.step():
                break

    def test_killswitch_disabled_matches_wlc(self):
        """Kill-switch off: force/energy at multiple strains must match pure WLC exactly."""
        PC.THREE_REGIME_ENABLED = False
        fiber = make_fiber(L_c=4e-6, diameter_nm=130.0)
        for eps in [0.0, 0.1, 0.25, 0.5, 0.8, 0.95]:
            x = fiber.L_c * (1 + eps)
            F = fiber.compute_force(x)
            U = fiber.compute_energy(x)
            # Manual WLC calculation
            strain = min(eps, PC.MAX_STRAIN)
            ome = 1.0 - strain
            F_expected = fiber.S * (PC.k_B_T / fiber.xi) * (
                1.0 / (4.0 * ome**2) - 0.25 + strain)
            F_expected = min(F_expected, PC.F_MAX)
            U_expected = fiber.S * (PC.k_B_T * fiber.L_c / fiber.xi) * (
                1.0 / (4.0 * ome) - 0.25 - strain / 4.0 + strain**2 / 2.0)
            assert abs(F - F_expected) < 1e-15, f"Force mismatch at eps={eps}"
            assert abs(U - U_expected) < 1e-30, f"Energy mismatch at eps={eps}"


class TestApplyCleavageForceRupture:
    """Test the force_rupture parameter on apply_cleavage."""

    def _make_sim(self):
        fibers = [
            WLCFiber(fiber_id=0, node_i=0, node_j=1, L_c=4e-6),
            WLCFiber(fiber_id=1, node_i=1, node_j=2, L_c=4e-6),
        ]
        node_positions = {
            0: np.array([0.0, 0.0]),
            1: np.array([5e-6, 0.0]),
            2: np.array([10e-6, 0.0]),
        }
        state = NetworkState(
            time=0.0,
            fibers=fibers, node_positions=node_positions,
            fixed_nodes={0: np.array([0.0, 0.0])},
            partial_fixed_x={2: 10e-6},
            left_boundary_nodes={0}, right_boundary_nodes={2},
        )
        return HybridMechanochemicalSimulation(
            initial_state=state, rng_seed=0, dt_chem=0.1, t_max=10.0)

    def test_force_rupture_sets_s_to_zero(self):
        """force_rupture=True sets S directly to 0."""
        sim = self._make_sim()
        sim.apply_cleavage(0, force_rupture=True)
        _, fiber = sim.state.get_fiber(0)
        assert fiber.S == 0.0

    def test_normal_cleavage_decrements(self):
        """force_rupture=False decrements by delta_S."""
        sim = self._make_sim()
        sim.apply_cleavage(0, force_rupture=False)
        _, fiber = sim.state.get_fiber(0)
        assert abs(fiber.S - 0.9) < 1e-10

    def test_force_rupture_logged(self):
        """force_rupture=True is logged in degradation history."""
        sim = self._make_sim()
        sim.apply_cleavage(0, force_rupture=True)
        entry = sim.state.degradation_history[-1]
        assert entry['three_regime_rupture'] is True

    def test_normal_cleavage_not_flagged(self):
        """Normal cleavage has three_regime_rupture=False."""
        sim = self._make_sim()
        sim.apply_cleavage(0, force_rupture=False)
        entry = sim.state.degradation_history[-1]
        assert entry['three_regime_rupture'] is False


class TestSafeStrain:
    """_safe_strain respects THREE_REGIME_ENABLED."""

    def test_safe_strain_caps_at_rupture_when_enabled(self):
        PC.THREE_REGIME_ENABLED = True
        fiber = make_fiber()
        x = fiber.L_c * 5.0  # eps = 4.0
        assert fiber._safe_strain(x) == PC.RUPTURE_STRAIN

    def test_safe_strain_caps_at_max_strain_when_disabled(self):
        PC.THREE_REGIME_ENABLED = False
        fiber = make_fiber()
        x = fiber.L_c * 5.0
        assert fiber._safe_strain(x) == PC.MAX_STRAIN


