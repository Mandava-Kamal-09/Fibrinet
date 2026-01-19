"""
Tests for force law implementations.

Run with: pytest tests/test_force_laws.py -v
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.force_laws import (
    hooke_tension, HookeanParams,
    wlc_tension_marko_siggia, WLCParams,
    wlc_low_strain_stiffness, KBT_PN_NM
)


class TestHookean:
    """Tests for Hookean spring force law."""

    def test_zero_extension(self):
        """T(L0) = 0 for extension-only spring."""
        params = HookeanParams(k_pN_per_nm=0.1, L0_nm=100.0)
        result = hooke_tension(100.0, params)
        assert result.is_valid
        assert result.tension_pN == 0.0

    def test_positive_extension(self):
        """T(L0 + delta) = k * delta."""
        k = 0.1  # pN/nm
        L0 = 100.0  # nm
        delta = 10.0  # nm
        params = HookeanParams(k_pN_per_nm=k, L0_nm=L0)
        result = hooke_tension(L0 + delta, params)
        assert result.is_valid
        assert abs(result.tension_pN - k * delta) < 1e-10

    def test_compression_extension_only(self):
        """Compression returns 0 in extension_only mode."""
        params = HookeanParams(k_pN_per_nm=0.1, L0_nm=100.0, extension_only=True)
        result = hooke_tension(90.0, params)
        assert result.is_valid
        assert result.tension_pN == 0.0

    def test_compression_bidirectional(self):
        """Compression returns negative force in bidirectional mode."""
        k = 0.1
        L0 = 100.0
        L = 90.0  # 10 nm compression
        params = HookeanParams(k_pN_per_nm=k, L0_nm=L0, extension_only=False)
        result = hooke_tension(L, params)
        assert result.is_valid
        assert abs(result.tension_pN - k * (L - L0)) < 1e-10

    def test_invalid_length(self):
        """Negative or zero length is invalid."""
        params = HookeanParams(k_pN_per_nm=0.1, L0_nm=100.0)
        result = hooke_tension(0.0, params)
        assert not result.is_valid
        assert "invalid_length" in result.reason

        result = hooke_tension(-10.0, params)
        assert not result.is_valid

    def test_invalid_params(self):
        """Invalid parameters are rejected."""
        params = HookeanParams(k_pN_per_nm=-0.1, L0_nm=100.0)
        result = hooke_tension(100.0, params)
        assert not result.is_valid
        assert "invalid_params" in result.reason


class TestWLC:
    """Tests for WLC Marko-Siggia force law."""

    def test_rupture_at_contour_length(self):
        """Invalid on rupture when L >= Lc."""
        params = WLCParams(Lp_nm=50.0, Lc_nm=100.0, rupture_at_Lc=True)

        # At contour length
        result = wlc_tension_marko_siggia(100.0, params)
        assert not result.is_valid
        assert result.reason == "rupture"

        # Beyond contour length
        result = wlc_tension_marko_siggia(110.0, params)
        assert not result.is_valid
        assert result.reason == "rupture"

    def test_monotone_increasing(self):
        """Tension increases monotonically over a range well below Lc."""
        params = WLCParams(Lp_nm=50.0, Lc_nm=100.0)

        lengths = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        tensions = []
        for L in lengths:
            result = wlc_tension_marko_siggia(L, params)
            assert result.is_valid, f"Invalid at L={L}"
            tensions.append(result.tension_pN)

        # Check strictly increasing
        for i in range(1, len(tensions)):
            assert tensions[i] > tensions[i-1], f"Not monotone at index {i}"

    def test_low_strain_equivalence(self):
        """
        Low-strain slope matches theoretical value.

        dT/dL at x=0 ≈ (3/2) * (kBT / (Lp * Lc))
        Use finite difference and allow ~15% tolerance.
        """
        Lp = 50.0  # nm
        Lc = 100.0  # nm
        kBT = KBT_PN_NM
        params = WLCParams(Lp_nm=Lp, Lc_nm=Lc, kBT_pN_nm=kBT)

        # Theoretical low-strain stiffness
        theoretical_slope = 1.5 * kBT / (Lp * Lc)

        # Finite difference at small x
        L1 = 1.0  # nm (x = 0.01)
        L2 = 2.0  # nm (x = 0.02)

        T1 = wlc_tension_marko_siggia(L1, params).tension_pN
        T2 = wlc_tension_marko_siggia(L2, params).tension_pN

        numeric_slope = (T2 - T1) / (L2 - L1)

        # Allow 15% relative tolerance
        rel_error = abs(numeric_slope - theoretical_slope) / theoretical_slope
        assert rel_error < 0.15, f"Slope mismatch: numeric={numeric_slope:.6f}, theoretical={theoretical_slope:.6f}, error={rel_error:.2%}"

    def test_low_strain_stiffness_function(self):
        """Test the low_strain_stiffness helper function."""
        Lp = 50.0
        Lc = 100.0
        kBT = KBT_PN_NM

        expected = 1.5 * kBT / (Lp * Lc)
        computed = wlc_low_strain_stiffness(Lp, Lc, kBT)

        assert abs(computed - expected) < 1e-10

    def test_invalid_length(self):
        """Negative or zero length is invalid."""
        params = WLCParams(Lp_nm=50.0, Lc_nm=100.0)

        result = wlc_tension_marko_siggia(0.0, params)
        assert not result.is_valid

        result = wlc_tension_marko_siggia(-10.0, params)
        assert not result.is_valid

    def test_invalid_params(self):
        """Invalid parameters are rejected."""
        params = WLCParams(Lp_nm=-50.0, Lc_nm=100.0)
        result = wlc_tension_marko_siggia(50.0, params)
        assert not result.is_valid

    def test_tension_only(self):
        """WLC should return non-negative tension for valid inputs."""
        params = WLCParams(Lp_nm=50.0, Lc_nm=100.0)

        # Very small extension
        result = wlc_tension_marko_siggia(0.1, params)
        assert result.is_valid
        assert result.tension_pN >= 0.0

    def test_near_contour_length_stability(self):
        """Numerical stability near (but below) contour length."""
        params = WLCParams(Lp_nm=50.0, Lc_nm=100.0)

        # Very close to Lc but below
        result = wlc_tension_marko_siggia(99.999, params)
        assert result.is_valid
        assert result.tension_pN > 0  # Should be a large but finite value
        assert result.tension_pN < 1e10  # Should not be infinite


class TestUnits:
    """Tests for unit consistency."""

    def test_kbt_value(self):
        """kBT at 300K should be approximately 4.114 pN·nm."""
        assert abs(KBT_PN_NM - 4.114) < 0.01

    def test_hooke_wlc_equivalence_at_low_strain(self):
        """
        At low strain, WLC should behave like Hookean with k_eff.
        """
        Lp = 50.0  # nm
        Lc = 100.0  # nm
        kBT = KBT_PN_NM

        # WLC effective stiffness
        k_eff = wlc_low_strain_stiffness(Lp, Lc, kBT)

        # Compare tensions at small extension
        L = 5.0  # nm (x = 0.05)

        wlc_params = WLCParams(Lp_nm=Lp, Lc_nm=Lc, kBT_pN_nm=kBT)
        wlc_result = wlc_tension_marko_siggia(L, wlc_params)

        # Hookean with equivalent stiffness (rest length = 0 for direct comparison)
        hooke_tension_expected = k_eff * L

        # Allow 20% tolerance at x=0.05
        rel_error = abs(wlc_result.tension_pN - hooke_tension_expected) / hooke_tension_expected
        assert rel_error < 0.20, f"WLC-Hooke mismatch at low strain: WLC={wlc_result.tension_pN:.4f}, Hooke={hooke_tension_expected:.4f}"


class TestDataclasses:
    """Tests for parameter dataclasses."""

    def test_hookean_params_validation(self):
        """HookeanParams.validate() catches invalid values."""
        valid = HookeanParams(k_pN_per_nm=0.1, L0_nm=100.0)
        is_valid, _ = valid.validate()
        assert is_valid

        invalid_k = HookeanParams(k_pN_per_nm=0.0, L0_nm=100.0)
        is_valid, msg = invalid_k.validate()
        assert not is_valid
        assert "k" in msg.lower()

    def test_wlc_params_validation(self):
        """WLCParams.validate() catches invalid values."""
        valid = WLCParams(Lp_nm=50.0, Lc_nm=100.0)
        is_valid, _ = valid.validate()
        assert is_valid

        # Lp > Lc is invalid
        invalid = WLCParams(Lp_nm=150.0, Lc_nm=100.0)
        is_valid, msg = invalid.validate()
        assert not is_valid

    def test_wlc_params_low_strain_property(self):
        """WLCParams.low_strain_stiffness property."""
        params = WLCParams(Lp_nm=50.0, Lc_nm=100.0, kBT_pN_nm=KBT_PN_NM)
        expected = 1.5 * KBT_PN_NM / (50.0 * 100.0)
        assert abs(params.low_strain_stiffness - expected) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
