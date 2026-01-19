"""
Tests for simulation reproducibility.
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
    SimulationConfig, ModelConfig, HookeConfig, WLCConfig,
    GeometryConfig, DynamicsConfig, LoadingConfig, EnzymeConfig
)
from single_fiber.runner import run_simulation


class TestReproducibility:
    """Tests for deterministic reproducibility."""

    def test_hooke_deterministic(self):
        """Same config produces identical results for Hooke."""
        config = SimulationConfig(
            model=ModelConfig(
                law="hooke",
                hooke=HookeConfig(k_pN_per_nm=0.1, L0_nm=100.0)
            ),
            geometry=GeometryConfig(
                x1_nm=[0, 0, 0],
                x2_nm=[100, 0, 0]
            ),
            dynamics=DynamicsConfig(dt_us=0.1, gamma_pN_us_per_nm=1.0),
            loading=LoadingConfig(
                v_nm_per_us=1.0,
                t_end_us=50.0
            )
        )
        config.output.save_every_steps = 1

        result1 = run_simulation(config)
        result2 = run_simulation(config)

        # Identical number of records
        assert len(result1.records) == len(result2.records)

        # Identical values
        for r1, r2 in zip(result1.records, result2.records):
            assert r1.t_us == r2.t_us
            assert r1.L_nm == r2.L_nm
            assert r1.tension_pN == r2.tension_pN
            assert r1.strain == r2.strain
            assert np.allclose(r1.x1_nm, r2.x1_nm)
            assert np.allclose(r1.x2_nm, r2.x2_nm)

    def test_wlc_deterministic(self):
        """Same config produces identical results for WLC."""
        config = SimulationConfig(
            model=ModelConfig(
                law="wlc",
                wlc=WLCConfig(Lp_nm=50.0, Lc_nm=200.0)
            ),
            geometry=GeometryConfig(
                x1_nm=[0, 0, 0],
                x2_nm=[100, 0, 0]
            ),
            dynamics=DynamicsConfig(dt_us=0.1, gamma_pN_us_per_nm=1.0),
            loading=LoadingConfig(
                v_nm_per_us=0.5,
                t_end_us=50.0
            )
        )
        config.output.save_every_steps = 1

        result1 = run_simulation(config)
        result2 = run_simulation(config)

        assert len(result1.records) == len(result2.records)
        for r1, r2 in zip(result1.records, result2.records):
            assert abs(r1.tension_pN - r2.tension_pN) < 1e-12

    def test_wlc_rupture_time_reproducible(self):
        """Rupture time is identical across runs."""
        config = SimulationConfig(
            model=ModelConfig(
                law="wlc",
                wlc=WLCConfig(Lp_nm=50.0, Lc_nm=150.0, rupture_at_Lc=True)
            ),
            geometry=GeometryConfig(
                x1_nm=[0, 0, 0],
                x2_nm=[100, 0, 0]
            ),
            dynamics=DynamicsConfig(dt_us=0.1, gamma_pN_us_per_nm=1.0),
            loading=LoadingConfig(
                v_nm_per_us=1.0,
                t_end_us=100.0
            )
        )

        result1 = run_simulation(config)
        result2 = run_simulation(config)

        assert result1.rupture_occurred == result2.rupture_occurred
        assert result1.rupture_time_us == result2.rupture_time_us

    def test_enzyme_reproducible_with_same_seed(self):
        """Enzyme cleavage is reproducible with same seed."""
        config = SimulationConfig(
            model=ModelConfig(
                law="hooke",
                hooke=HookeConfig(k_pN_per_nm=0.1, L0_nm=100.0)
            ),
            geometry=GeometryConfig(
                x1_nm=[0, 0, 0],
                x2_nm=[100, 0, 0]
            ),
            dynamics=DynamicsConfig(dt_us=0.1, gamma_pN_us_per_nm=1.0),
            loading=LoadingConfig(
                v_nm_per_us=1.0,
                t_end_us=100.0
            ),
            enzyme=EnzymeConfig(
                enabled=True,
                seed=12345,
                baseline_lambda_per_us=0.01  # Low rate
            )
        )

        result1 = run_simulation(config)
        result2 = run_simulation(config)

        # Same outcome (rupture or not)
        assert result1.rupture_occurred == result2.rupture_occurred

        # If both ruptured, same time
        if result1.rupture_occurred:
            assert result1.rupture_time_us == result2.rupture_time_us

    def test_different_seeds_different_outcomes(self):
        """Different seeds can produce different enzyme outcomes."""
        # Use high cleavage rate to increase chance of difference
        base_config = dict(
            model=ModelConfig(
                law="hooke",
                hooke=HookeConfig(k_pN_per_nm=0.1, L0_nm=100.0)
            ),
            geometry=GeometryConfig(
                x1_nm=[0, 0, 0],
                x2_nm=[100, 0, 0]
            ),
            dynamics=DynamicsConfig(dt_us=1.0, gamma_pN_us_per_nm=1.0),
            loading=LoadingConfig(
                v_nm_per_us=1.0,
                t_end_us=100.0
            )
        )

        # Run with many different seeds
        results = []
        for seed in range(10):
            config = SimulationConfig(
                **base_config,
                enzyme=EnzymeConfig(
                    enabled=True,
                    seed=seed,
                    baseline_lambda_per_us=0.05  # High rate
                )
            )
            results.append(run_simulation(config))

        # At least some should have different outcomes
        rupture_times = [r.rupture_time_us for r in results if r.rupture_occurred]
        # With high rate, most should rupture at different times
        if len(rupture_times) > 2:
            unique_times = len(set(rupture_times))
            assert unique_times > 1, "Expected variation in rupture times with different seeds"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
