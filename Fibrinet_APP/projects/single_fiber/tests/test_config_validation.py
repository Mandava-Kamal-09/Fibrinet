"""
Tests for configuration validation.
"""

import pytest
import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from single_fiber.config import (
    HookeConfig, WLCConfig, ModelConfig,
    GeometryConfig, DynamicsConfig, LoadingConfig,
    EnzymeConfig, OutputConfig, SimulationConfig
)


class TestHookeConfigValidation:
    """Tests for Hookean config validation."""

    def test_valid_config(self):
        cfg = HookeConfig(k_pN_per_nm=0.1, L0_nm=100.0)
        is_valid, err = cfg.validate()
        assert is_valid
        assert err is None

    def test_negative_k_rejected(self):
        cfg = HookeConfig(k_pN_per_nm=-0.1, L0_nm=100.0)
        is_valid, err = cfg.validate()
        assert not is_valid
        assert "k_pN_per_nm" in err

    def test_zero_k_rejected(self):
        cfg = HookeConfig(k_pN_per_nm=0.0, L0_nm=100.0)
        is_valid, err = cfg.validate()
        assert not is_valid

    def test_negative_L0_rejected(self):
        cfg = HookeConfig(k_pN_per_nm=0.1, L0_nm=-10.0)
        is_valid, err = cfg.validate()
        assert not is_valid
        assert "L0_nm" in err


class TestWLCConfigValidation:
    """Tests for WLC config validation."""

    def test_valid_config(self):
        cfg = WLCConfig(Lp_nm=50.0, Lc_nm=100.0)
        is_valid, err = cfg.validate()
        assert is_valid

    def test_negative_Lp_rejected(self):
        cfg = WLCConfig(Lp_nm=-50.0, Lc_nm=100.0)
        is_valid, err = cfg.validate()
        assert not is_valid
        assert "Lp_nm" in err

    def test_negative_Lc_rejected(self):
        cfg = WLCConfig(Lp_nm=50.0, Lc_nm=-100.0)
        is_valid, err = cfg.validate()
        assert not is_valid


class TestGeometryConfigValidation:
    """Tests for geometry config validation."""

    def test_valid_config(self):
        cfg = GeometryConfig(x1_nm=[0, 0, 0], x2_nm=[100, 0, 0])
        is_valid, err = cfg.validate()
        assert is_valid

    def test_wrong_dimension_x1(self):
        cfg = GeometryConfig(x1_nm=[0, 0], x2_nm=[100, 0, 0])
        is_valid, err = cfg.validate()
        assert not is_valid
        assert "x1_nm" in err

    def test_wrong_dimension_x2(self):
        cfg = GeometryConfig(x1_nm=[0, 0, 0], x2_nm=[100, 0])
        is_valid, err = cfg.validate()
        assert not is_valid
        assert "x2_nm" in err

    def test_zero_length_rejected(self):
        cfg = GeometryConfig(x1_nm=[0, 0, 0], x2_nm=[0, 0, 0])
        is_valid, err = cfg.validate()
        assert not is_valid
        assert "length" in err.lower()

    def test_initial_length_property(self):
        cfg = GeometryConfig(x1_nm=[0, 0, 0], x2_nm=[100, 0, 0])
        assert abs(cfg.initial_length_nm - 100.0) < 1e-10


class TestDynamicsConfigValidation:
    """Tests for dynamics config validation."""

    def test_valid_config(self):
        cfg = DynamicsConfig(dt_us=0.01, gamma_pN_us_per_nm=1.0)
        is_valid, err = cfg.validate()
        assert is_valid

    def test_negative_dt_rejected(self):
        cfg = DynamicsConfig(dt_us=-0.01, gamma_pN_us_per_nm=1.0)
        is_valid, err = cfg.validate()
        assert not is_valid

    def test_zero_gamma_rejected(self):
        cfg = DynamicsConfig(dt_us=0.01, gamma_pN_us_per_nm=0.0)
        is_valid, err = cfg.validate()
        assert not is_valid


class TestLoadingConfigValidation:
    """Tests for loading config validation."""

    def test_valid_config(self):
        cfg = LoadingConfig()
        is_valid, err = cfg.validate()
        assert is_valid

    def test_zero_axis_rejected(self):
        cfg = LoadingConfig(axis=[0, 0, 0])
        is_valid, err = cfg.validate()
        assert not is_valid
        assert "axis" in err

    def test_axis_unit_normalized(self):
        cfg = LoadingConfig(axis=[2, 0, 0])
        u = cfg.axis_unit
        assert abs(u[0] - 1.0) < 1e-10
        assert abs(u[1]) < 1e-10
        assert abs(u[2]) < 1e-10


class TestModelConfigValidation:
    """Tests for model config validation."""

    def test_hooke_requires_hooke_config(self):
        cfg = ModelConfig(law="hooke", hooke=None)
        is_valid, err = cfg.validate()
        assert not is_valid
        assert "hooke config required" in err

    def test_wlc_requires_wlc_config(self):
        cfg = ModelConfig(law="wlc", wlc=None)
        is_valid, err = cfg.validate()
        assert not is_valid
        assert "wlc config required" in err

    def test_valid_hooke_model(self):
        cfg = ModelConfig(
            law="hooke",
            hooke=HookeConfig(k_pN_per_nm=0.1, L0_nm=100.0)
        )
        is_valid, err = cfg.validate()
        assert is_valid

    def test_valid_wlc_model(self):
        cfg = ModelConfig(
            law="wlc",
            wlc=WLCConfig(Lp_nm=50.0, Lc_nm=100.0)
        )
        is_valid, err = cfg.validate()
        assert is_valid


class TestSimulationConfigValidation:
    """Tests for complete simulation config validation."""

    def test_valid_complete_config(self):
        cfg = SimulationConfig(
            model=ModelConfig(
                law="hooke",
                hooke=HookeConfig(k_pN_per_nm=0.1, L0_nm=100.0)
            ),
            geometry=GeometryConfig(x1_nm=[0, 0, 0], x2_nm=[100, 0, 0]),
            dynamics=DynamicsConfig(dt_us=0.01, gamma_pN_us_per_nm=1.0),
            loading=LoadingConfig()
        )
        is_valid, err = cfg.validate()
        assert is_valid

    def test_invalid_model_propagates(self):
        cfg = SimulationConfig(
            model=ModelConfig(
                law="hooke",
                hooke=HookeConfig(k_pN_per_nm=-0.1, L0_nm=100.0)  # Invalid
            ),
            geometry=GeometryConfig(x1_nm=[0, 0, 0], x2_nm=[100, 0, 0]),
            dynamics=DynamicsConfig(dt_us=0.01, gamma_pN_us_per_nm=1.0),
            loading=LoadingConfig()
        )
        is_valid, err = cfg.validate()
        assert not is_valid
        assert "model" in err


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
