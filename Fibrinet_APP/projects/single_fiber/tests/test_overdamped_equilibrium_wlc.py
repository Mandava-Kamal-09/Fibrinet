"""
Tests for WLC spring equilibrium and behavior.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from single_fiber.config import (
    SimulationConfig, ModelConfig, WLCConfig,
    GeometryConfig, DynamicsConfig, LoadingConfig
)
from single_fiber.runner import run_simulation


class TestWLCEquilibrium:
    """Tests for WLC spring behavior."""

    def test_tension_increases_monotonically(self):
        """Tension increases monotonically with extension."""
        Lp = 50.0
        Lc = 200.0

        config = SimulationConfig(
            model=ModelConfig(
                law="wlc",
                wlc=WLCConfig(Lp_nm=Lp, Lc_nm=Lc)
            ),
            geometry=GeometryConfig(
                x1_nm=[0, 0, 0],
                x2_nm=[50, 0, 0]  # Start at x=0.25
            ),
            dynamics=DynamicsConfig(dt_us=0.1, gamma_pN_us_per_nm=1.0),
            loading=LoadingConfig(
                v_nm_per_us=1.0,
                t_end_us=100.0,  # Will reach L=150 (x=0.75)
                constraint="hard"
            )
        )
        config.output.save_every_steps = 1

        result = run_simulation(config)

        # Check monotonic increase
        prev_T = -1.0
        for rec in result.records:
            if rec.intact:  # Only check while intact
                assert rec.tension_pN >= prev_T, \
                    f"Non-monotonic: T={rec.tension_pN} < prev={prev_T}"
                prev_T = rec.tension_pN

    def test_tension_stiffens_at_high_extension(self):
        """WLC stiffens dramatically near contour length."""
        Lp = 50.0
        Lc = 200.0

        config = SimulationConfig(
            model=ModelConfig(
                law="wlc",
                wlc=WLCConfig(Lp_nm=Lp, Lc_nm=Lc)
            ),
            geometry=GeometryConfig(
                x1_nm=[0, 0, 0],
                x2_nm=[100, 0, 0]
            ),
            dynamics=DynamicsConfig(dt_us=0.5, gamma_pN_us_per_nm=1.0),
            loading=LoadingConfig(
                v_nm_per_us=1.0,
                t_end_us=90.0,  # Will reach L=190 (x=0.95)
                constraint="hard"
            )
        )
        config.output.save_every_steps = 1

        result = run_simulation(config)

        # Get tensions at low and high extension
        low_ext_records = [r for r in result.records if r.L_nm < 120]
        high_ext_records = [r for r in result.records if r.L_nm > 180 and r.intact]

        if low_ext_records and high_ext_records:
            low_T = low_ext_records[-1].tension_pN
            high_T = high_ext_records[-1].tension_pN

            # High extension tension should be much larger
            assert high_T > low_T * 5, \
                f"Expected strain stiffening: high_T={high_T}, low_T={low_T}"

    def test_wlc_reduces_to_hooke_at_low_strain(self):
        """At low extension, WLC ~ Hookean with k_eff = 3kBT/(2*Lp*Lc)."""
        Lp = 50.0
        Lc = 200.0
        kBT = 4.114

        # Theoretical low-strain stiffness
        k_eff = 1.5 * kBT / (Lp * Lc)

        config = SimulationConfig(
            model=ModelConfig(
                law="wlc",
                wlc=WLCConfig(Lp_nm=Lp, Lc_nm=Lc, kBT_pN_nm=kBT)
            ),
            geometry=GeometryConfig(
                x1_nm=[0, 0, 0],
                x2_nm=[10, 0, 0]  # Small L for low x
            ),
            dynamics=DynamicsConfig(dt_us=0.1, gamma_pN_us_per_nm=1.0),
            loading=LoadingConfig(
                v_nm_per_us=0.1,
                t_end_us=50.0,  # Go to L=15 (x=0.075)
                constraint="hard"
            )
        )
        config.output.save_every_steps = 5

        result = run_simulation(config)

        # Check that slope is approximately k_eff
        # T ~ k_eff * L for small L
        for rec in result.records:
            if rec.L_nm > 5 and rec.L_nm < 20:
                expected_T = k_eff * rec.L_nm
                rel_error = abs(rec.tension_pN - expected_T) / expected_T
                # Allow 30% error at these small strains
                assert rel_error < 0.30, \
                    f"At L={rec.L_nm}: T={rec.tension_pN}, expected~{expected_T}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
