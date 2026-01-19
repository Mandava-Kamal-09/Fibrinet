"""
Tests for WLC rupture behavior.
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


class TestWLCRupture:
    """Tests for WLC rupture at contour length."""

    def test_rupture_when_L_exceeds_Lc(self):
        """Fiber ruptures when L >= Lc."""
        Lp = 50.0
        Lc = 150.0
        L0 = 100.0

        config = SimulationConfig(
            model=ModelConfig(
                law="wlc",
                wlc=WLCConfig(Lp_nm=Lp, Lc_nm=Lc, rupture_at_Lc=True)
            ),
            geometry=GeometryConfig(
                x1_nm=[0, 0, 0],
                x2_nm=[L0, 0, 0]
            ),
            dynamics=DynamicsConfig(dt_us=0.1, gamma_pN_us_per_nm=1.0),
            loading=LoadingConfig(
                v_nm_per_us=1.0,
                t_end_us=100.0,  # Will exceed Lc
                constraint="hard"
            )
        )

        result = run_simulation(config)

        # Should have ruptured
        assert result.rupture_occurred, "Expected rupture"
        assert result.rupture_time_us is not None

        # Rupture should occur around t = 50 μs (when L = 150 = Lc)
        expected_rupture_time = Lc - L0  # = 50 μs at v=1 nm/μs
        assert abs(result.rupture_time_us - expected_rupture_time) < 1.0, \
            f"Expected rupture at ~{expected_rupture_time} μs, got {result.rupture_time_us}"

    def test_tension_zero_after_rupture(self):
        """After rupture, tension stays exactly zero."""
        Lp = 50.0
        Lc = 120.0
        L0 = 100.0

        config = SimulationConfig(
            model=ModelConfig(
                law="wlc",
                wlc=WLCConfig(Lp_nm=Lp, Lc_nm=Lc, rupture_at_Lc=True)
            ),
            geometry=GeometryConfig(
                x1_nm=[0, 0, 0],
                x2_nm=[L0, 0, 0]
            ),
            dynamics=DynamicsConfig(dt_us=0.5, gamma_pN_us_per_nm=1.0),
            loading=LoadingConfig(
                v_nm_per_us=1.0,
                t_end_us=50.0,
                constraint="hard"
            )
        )
        config.output.save_every_steps = 1

        result = run_simulation(config)

        # Find rupture point
        rupture_idx = None
        for i, rec in enumerate(result.records):
            if not rec.intact:
                rupture_idx = i
                break

        assert rupture_idx is not None, "Expected rupture"

        # All records after rupture should have zero tension
        for rec in result.records[rupture_idx:]:
            assert rec.tension_pN == 0.0, \
                f"Expected T=0 after rupture, got {rec.tension_pN}"
            assert not rec.intact

    def test_rupture_time_recorded(self):
        """Rupture time is correctly recorded in state and records."""
        Lp = 50.0
        Lc = 110.0
        L0 = 100.0

        config = SimulationConfig(
            model=ModelConfig(
                law="wlc",
                wlc=WLCConfig(Lp_nm=Lp, Lc_nm=Lc, rupture_at_Lc=True)
            ),
            geometry=GeometryConfig(
                x1_nm=[0, 0, 0],
                x2_nm=[L0, 0, 0]
            ),
            dynamics=DynamicsConfig(dt_us=0.1, gamma_pN_us_per_nm=1.0),
            loading=LoadingConfig(
                v_nm_per_us=1.0,
                t_end_us=20.0,
                constraint="hard"
            )
        )
        config.output.save_every_steps = 1

        result = run_simulation(config)

        # Check rupture time consistency
        assert result.rupture_occurred
        rupture_t = result.rupture_time_us

        # Find first ruptured record
        for rec in result.records:
            if not rec.intact:
                assert rec.rupture_time_us == rupture_t
                break

    def test_no_rupture_below_Lc(self):
        """No rupture if we stay below Lc."""
        Lp = 50.0
        Lc = 200.0
        L0 = 100.0

        config = SimulationConfig(
            model=ModelConfig(
                law="wlc",
                wlc=WLCConfig(Lp_nm=Lp, Lc_nm=Lc, rupture_at_Lc=True)
            ),
            geometry=GeometryConfig(
                x1_nm=[0, 0, 0],
                x2_nm=[L0, 0, 0]
            ),
            dynamics=DynamicsConfig(dt_us=0.1, gamma_pN_us_per_nm=1.0),
            loading=LoadingConfig(
                v_nm_per_us=0.5,
                t_end_us=100.0,  # Will reach L=150 < Lc
                constraint="hard"
            )
        )

        result = run_simulation(config)

        # Should NOT rupture
        assert not result.rupture_occurred
        assert result.final_state.is_intact
        assert all(rec.intact for rec in result.records)

    def test_rupture_disabled_option(self):
        """When rupture_at_Lc=False, no rupture even at L >= Lc."""
        Lp = 50.0
        Lc = 110.0
        L0 = 100.0

        config = SimulationConfig(
            model=ModelConfig(
                law="wlc",
                wlc=WLCConfig(Lp_nm=Lp, Lc_nm=Lc, rupture_at_Lc=False)
            ),
            geometry=GeometryConfig(
                x1_nm=[0, 0, 0],
                x2_nm=[L0, 0, 0]
            ),
            dynamics=DynamicsConfig(dt_us=0.1, gamma_pN_us_per_nm=1.0),
            loading=LoadingConfig(
                v_nm_per_us=1.0,
                t_end_us=15.0,  # Would exceed Lc
                constraint="hard"
            )
        )

        result = run_simulation(config)

        # Should NOT rupture (rupture disabled)
        # But simulation will use clamped values near singularity
        assert result.final_state.is_intact


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
