"""
Tests for Hookean spring equilibrium under displacement control.
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
    SimulationConfig, ModelConfig, HookeConfig,
    GeometryConfig, DynamicsConfig, LoadingConfig
)
from single_fiber.state import FiberState
from single_fiber.model import FiberModel
from single_fiber.runner import run_simulation


class TestHookeanEquilibrium:
    """Tests for Hookean spring tension under hard displacement."""

    def test_tension_equals_k_times_delta(self):
        """
        Under hard displacement, tension = k * (L - L0).

        Setup:
        - k = 0.1 pN/nm
        - L0 = 100 nm
        - Pull to L = 110 nm
        - Expected tension: 0.1 * 10 = 1.0 pN
        """
        k = 0.1
        L0 = 100.0
        delta = 10.0
        expected_tension = k * delta

        config = SimulationConfig(
            model=ModelConfig(
                law="hooke",
                hooke=HookeConfig(k_pN_per_nm=k, L0_nm=L0, extension_only=True)
            ),
            geometry=GeometryConfig(
                x1_nm=[0, 0, 0],
                x2_nm=[L0, 0, 0]  # Start at rest length
            ),
            dynamics=DynamicsConfig(
                dt_us=0.1,
                gamma_pN_us_per_nm=1.0
            ),
            loading=LoadingConfig(
                axis=[1, 0, 0],
                v_nm_per_us=1.0,  # 1 nm/μs
                t_end_us=delta,   # Pull for 10 μs to add 10 nm
                constraint="hard"
            )
        )

        result = run_simulation(config)

        # Check final state
        final_L = result.final_state.length_nm
        assert abs(final_L - (L0 + delta)) < 1e-6, f"Expected L={L0+delta}, got {final_L}"

        # Check tension in last record
        final_tension = result.records[-1].tension_pN
        assert abs(final_tension - expected_tension) < 1e-6, \
            f"Expected T={expected_tension}, got {final_tension}"

    def test_tension_zero_at_rest_length(self):
        """Tension is zero when L = L0."""
        k = 0.1
        L0 = 100.0

        config = SimulationConfig(
            model=ModelConfig(
                law="hooke",
                hooke=HookeConfig(k_pN_per_nm=k, L0_nm=L0)
            ),
            geometry=GeometryConfig(
                x1_nm=[0, 0, 0],
                x2_nm=[L0, 0, 0]
            ),
            dynamics=DynamicsConfig(dt_us=0.1, gamma_pN_us_per_nm=1.0),
            loading=LoadingConfig(
                v_nm_per_us=0.0,  # No displacement
                t_end_us=1.0
            )
        )

        result = run_simulation(config)
        assert result.records[0].tension_pN == 0.0

    def test_extension_only_no_compression_force(self):
        """In extension_only mode, compression gives zero tension."""
        k = 0.1
        L0 = 100.0

        config = SimulationConfig(
            model=ModelConfig(
                law="hooke",
                hooke=HookeConfig(k_pN_per_nm=k, L0_nm=L0, extension_only=True)
            ),
            geometry=GeometryConfig(
                x1_nm=[0, 0, 0],
                x2_nm=[90, 0, 0]  # Start compressed (L < L0)
            ),
            dynamics=DynamicsConfig(dt_us=0.1, gamma_pN_us_per_nm=1.0),
            loading=LoadingConfig(
                v_nm_per_us=0.0,
                t_end_us=1.0
            )
        )

        result = run_simulation(config)
        # Tension should be zero (extension_only mode)
        assert result.records[0].tension_pN == 0.0

    def test_linear_tension_extension_relationship(self):
        """Tension increases linearly with extension."""
        k = 0.1
        L0 = 100.0
        v = 1.0
        t_end = 50.0

        config = SimulationConfig(
            model=ModelConfig(
                law="hooke",
                hooke=HookeConfig(k_pN_per_nm=k, L0_nm=L0)
            ),
            geometry=GeometryConfig(
                x1_nm=[0, 0, 0],
                x2_nm=[L0, 0, 0]
            ),
            dynamics=DynamicsConfig(dt_us=1.0, gamma_pN_us_per_nm=1.0),
            loading=LoadingConfig(
                v_nm_per_us=v,
                t_end_us=t_end,
                constraint="hard"
            )
        )
        config.output.save_every_steps = 1

        result = run_simulation(config)

        # Check linearity
        for rec in result.records:
            L = rec.L_nm
            expected_T = k * max(0, L - L0)
            assert abs(rec.tension_pN - expected_T) < 1e-6, \
                f"At L={L}: expected T={expected_T}, got {rec.tension_pN}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
