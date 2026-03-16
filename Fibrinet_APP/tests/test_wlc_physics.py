"""Verification tests for WLC/eWLC force laws, energy, and cleavage rates."""

import sys
import os
import math
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.fibrinet_core_v2 import WLCFiber, PhysicalConstants
from src.config.physics_constants import (
    validate_constants, get_thermal_energy, compute_strain_inhibited_rate,
    compute_prestrained_contour_length,
)

PC = PhysicalConstants()


# Force law basics

class TestWLCForce:

    def test_wlc_zero_force_at_rest_length(self, single_fiber):
        """WLC force should be zero when fiber length equals contour length."""
        F = single_fiber.compute_force(single_fiber.L_c)
        # At x = L_c, strain = 0: F_wlc = kBT/xi * (1/4 - 1/4 + 0) = 0
        assert abs(F) < 1e-15, (
            f"Expected F(L_c) = 0, got {F:.3e} N"
        )

    def test_wlc_positive_force_under_extension(self, single_fiber):
        """Extending a fiber beyond L_c produces positive (tensile) force."""
        x = 1.2 * single_fiber.L_c  # 20% extension
        F = single_fiber.compute_force(x)
        assert F > 0, f"Expected positive force under extension, got {F:.3e} N"

    def test_wlc_force_diverges_near_contour(self, single_fiber):
        """Force increases sharply as strain approaches the 0.99 limit."""
        F_low = single_fiber.compute_force(1.1 * single_fiber.L_c)
        F_high = single_fiber.compute_force(1.95 * single_fiber.L_c)
        assert F_high > 10 * F_low, (
            f"Expected force divergence near strain limit: "
            f"F(10%)={F_low:.3e} N, F(95%)={F_high:.3e} N"
        )

    def test_wlc_compression_repulsive(self, single_fiber):
        """Compressed fiber (x < L_c) produces negative (repulsive) force."""
        x = 0.8 * single_fiber.L_c  # 20% compression
        F = single_fiber.compute_force(x)
        assert F < 0, (
            f"Expected repulsive (negative) force under compression, got {F:.3e} N"
        )

    def test_ewlc_adds_extensibility(self, single_fiber, single_fiber_ewlc):
        """eWLC force exceeds WLC force at same extension (K0 > 0 adds stiffness)."""
        x = 1.3 * single_fiber.L_c
        F_wlc = single_fiber.compute_force(x)
        F_ewlc = single_fiber_ewlc.compute_force(x)
        assert F_ewlc > F_wlc, (
            f"eWLC should be stiffer than WLC: F_wlc={F_wlc:.3e}, F_ewlc={F_ewlc:.3e}"
        )

    def test_force_clamped_at_fmax(self, single_fiber):
        """Force must never exceed F_MAX = 1e-6 N."""
        # Push strain to the maximum
        x = single_fiber.L_c * (1.0 + PC.MAX_STRAIN * 0.999)
        F = single_fiber.compute_force(x)
        assert F <= PC.F_MAX, (
            f"Force {F:.3e} exceeds F_MAX={PC.F_MAX:.3e} N"
        )

    def test_strain_clamped_at_max(self, single_fiber):
        """Strain is internally capped at MAX_STRAIN = 0.99."""
        # Extend beyond 100% — force should not be infinite
        x = 3.0 * single_fiber.L_c  # 200% strain
        F = single_fiber.compute_force(x)
        assert np.isfinite(F), f"Force is not finite at extreme extension: {F}"
        assert F > 0, f"Expected positive force, got {F:.3e}"


# Energy–force consistency

class TestEnergyForceConsistency:

    def test_energy_force_consistency(self, single_fiber):
        """dU/dx should approximately equal F(x) (central difference)."""
        x = 1.2 * single_fiber.L_c
        dx = 1e-12  # 1 pm step for numerical derivative
        U_plus = single_fiber.compute_energy(x + dx)
        U_minus = single_fiber.compute_energy(x - dx)
        dUdx_numerical = (U_plus - U_minus) / (2 * dx)

        F_analytical = single_fiber.compute_force(x)
        rel_error = abs(dUdx_numerical - F_analytical) / max(abs(F_analytical), 1e-30)
        assert rel_error < 1e-4, (
            f"Energy-force inconsistency: dU/dx={dUdx_numerical:.6e}, "
            f"F(x)={F_analytical:.6e}, relative error={rel_error:.2e}"
        )

    def test_energy_force_consistency_ewlc(self, single_fiber_ewlc):
        """dU/dx ≈ F(x) for eWLC model too."""
        fiber = single_fiber_ewlc
        x = 1.15 * fiber.L_c
        dx = 1e-12
        dUdx = (fiber.compute_energy(x + dx) - fiber.compute_energy(x - dx)) / (2 * dx)
        F = fiber.compute_force(x)
        rel_error = abs(dUdx - F) / max(abs(F), 1e-30)
        assert rel_error < 1e-4, (
            f"eWLC energy-force mismatch: rel_error={rel_error:.2e}"
        )


# Integrity scaling

class TestIntegrityScaling:

    def test_integrity_scales_force(self):
        """F(x, S=0.5) = 0.5 × F(x, S=1.0)."""
        L_c = 1e-6
        fiber_full = WLCFiber(fiber_id=0, node_i=0, node_j=1, L_c=L_c, S=1.0)
        fiber_half = WLCFiber(fiber_id=1, node_i=0, node_j=1, L_c=L_c, S=0.5)
        x = 1.15 * L_c
        F_full = fiber_full.compute_force(x)
        F_half = fiber_half.compute_force(x)
        assert abs(F_half - 0.5 * F_full) < 1e-18, (
            f"Integrity scaling failed: F(S=1)={F_full:.3e}, "
            f"F(S=0.5)={F_half:.3e}, expected {0.5*F_full:.3e}"
        )

    def test_integrity_scales_energy(self):
        """U(x, S=0.5) = 0.5 × U(x, S=1.0)."""
        L_c = 1e-6
        f1 = WLCFiber(fiber_id=0, node_i=0, node_j=1, L_c=L_c, S=1.0)
        f2 = WLCFiber(fiber_id=1, node_i=0, node_j=1, L_c=L_c, S=0.5)
        x = 1.15 * L_c
        assert abs(f2.compute_energy(x) - 0.5 * f1.compute_energy(x)) < 1e-30


# Cleavage rate model

class TestCleavageRate:

    def test_cleavage_rate_strain_inhibited(self, single_fiber):
        """Higher strain should produce lower cleavage rate (inhibition)."""
        x_zero_strain = single_fiber.L_c  # strain = 0
        x_high_strain = 1.5 * single_fiber.L_c  # strain = 0.5
        k_zero = single_fiber.compute_cleavage_rate(x_zero_strain)
        k_high = single_fiber.compute_cleavage_rate(x_high_strain)
        assert k_high < k_zero, (
            f"Strain should INHIBIT cleavage: k(ε=0)={k_zero:.4e}, "
            f"k(ε=0.5)={k_high:.4e}"
        )

    def test_cleavage_rate_exponential_model(self, single_fiber):
        """k(ε) = k0 × exp(-β × ε), verify against hand-calculated values."""
        # At ε=0: k = k0 = 0.1
        k_at_zero = single_fiber.compute_cleavage_rate(single_fiber.L_c)
        assert abs(k_at_zero - PC.k_cat_0) < 1e-12, (
            f"k(ε=0) should be k0={PC.k_cat_0}, got {k_at_zero}"
        )

        # At ε=0.23 (prestrain): k = 0.1 * exp(-10 * 0.23) = 0.1 * exp(-2.3) ≈ 0.01003
        strain = 0.23
        x = single_fiber.L_c * (1.0 + strain)
        k_prestrain = single_fiber.compute_cleavage_rate(x)
        k_expected = PC.k_cat_0 * math.exp(-PC.beta_strain * strain)
        assert abs(k_prestrain - k_expected) < 1e-12, (
            f"k(ε=0.23) expected {k_expected:.6e}, got {k_prestrain:.6e}"
        )

    def test_cleavage_rate_compression_equals_baseline(self, single_fiber):
        """Compressed fiber (x < L_c) has strain clamped at 0 → k = k0."""
        x = 0.8 * single_fiber.L_c
        k = single_fiber.compute_cleavage_rate(x)
        assert abs(k - PC.k_cat_0) < 1e-12, (
            f"Compressed fiber should have baseline rate {PC.k_cat_0}, got {k}"
        )


# Physics constants validation

class TestPhysicsConstants:

    def test_validate_constants_passes(self):
        """validate_constants() should pass without errors."""
        assert validate_constants() is True

    def test_kbt_value(self):
        """k_BT ≈ 4.28e-21 J at 37°C."""
        kbt = get_thermal_energy()
        expected = 1.380649e-23 * 310.15
        assert abs(kbt - expected) < 1e-28, (
            f"k_BT = {kbt:.6e} J, expected {expected:.6e} J"
        )
        # Sanity check: should be in the right ballpark (~4.28e-21)
        assert 4.2e-21 < kbt < 4.4e-21

    def test_prestrain_contour_length(self):
        """L_c = L / (1 + 0.23) for prestrained fibers."""
        L_geom = 1.0e-6  # 1 µm
        L_c = compute_prestrained_contour_length(L_geom)
        expected = L_geom / 1.23
        assert abs(L_c - expected) < 1e-18

    def test_wlc_force_prefactor_units(self):
        """k_BT / ξ should have units of N (force)."""
        from src.config.physics_constants import get_wlc_force_prefactor
        prefactor = get_wlc_force_prefactor()
        # kBT ~ 4.28e-21 J, xi = 1e-6 m → prefactor ~ 4.28e-15 N
        assert 4.0e-15 < prefactor < 5.0e-15, (
            f"Force prefactor = {prefactor:.3e} N, expected ~4.28e-15 N"
        )
