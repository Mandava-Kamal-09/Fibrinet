"""
Tests for export CSV schema compliance.
"""

import pytest
import csv
import json
import tempfile
import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from single_fiber.config import (
    SimulationConfig, ModelConfig, HookeConfig, WLCConfig,
    GeometryConfig, DynamicsConfig, LoadingConfig
)
from single_fiber.runner import run_simulation
from single_fiber.exporters import export_results, CSV_COLUMNS


class TestExportSchema:
    """Tests for CSV export schema compliance."""

    def test_csv_columns_exact_match(self):
        """CSV columns match specification exactly."""
        expected_columns = [
            "t_us",
            "x1_x_nm", "x1_y_nm", "x1_z_nm",
            "x2_x_nm", "x2_y_nm", "x2_z_nm",
            "L_nm",
            "strain",
            "tension_pN",
            "law_name",
            "intact",
            "rupture_time_us",
            "hazard_lambda_per_us",
            "hazard_H"
        ]

        assert CSV_COLUMNS == expected_columns

    def test_csv_export_has_correct_columns(self):
        """Exported CSV has exactly the specified columns."""
        config = SimulationConfig(
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
                t_end_us=10.0
            )
        )

        result = run_simulation(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_results(result, Path(tmpdir), "test_run")

            # Read CSV and check columns
            with open(paths["csv"], 'r') as f:
                reader = csv.reader(f)
                header = next(reader)

            assert header == CSV_COLUMNS

    def test_csv_data_types(self):
        """CSV data has correct types."""
        config = SimulationConfig(
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
                t_end_us=5.0
            )
        )

        result = run_simulation(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_results(result, Path(tmpdir), "test_run")

            with open(paths["csv"], 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Numeric columns should be parseable
                    float(row["t_us"])
                    float(row["x1_x_nm"])
                    float(row["L_nm"])
                    float(row["strain"])
                    float(row["tension_pN"])

                    # intact should be 0 or 1
                    assert row["intact"] in ("0", "1")

                    # law_name should be string
                    assert row["law_name"] in ("hooke", "wlc")

    def test_metadata_json_structure(self):
        """Metadata JSON has required fields."""
        config = SimulationConfig(
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
                t_end_us=5.0
            )
        )

        result = run_simulation(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_results(result, Path(tmpdir), "test_run")

            with open(paths["metadata"], 'r') as f:
                meta = json.load(f)

            # Check required fields
            assert "timestamp" in meta
            assert "config" in meta
            assert "summary" in meta

            # Check summary fields
            summary = meta["summary"]
            assert "n_steps" in summary
            assert "max_tension_pN" in summary
            assert "final_strain" in summary
            assert "rupture_occurred" in summary

            # Check config fields
            cfg = meta["config"]
            assert "model" in cfg
            assert "geometry" in cfg
            assert "dynamics" in cfg
            assert "loading" in cfg

    def test_intact_column_binary(self):
        """intact column is 1 when intact, 0 when ruptured."""
        config = SimulationConfig(
            model=ModelConfig(
                law="wlc",
                wlc=WLCConfig(Lp_nm=50.0, Lc_nm=120.0, rupture_at_Lc=True)
            ),
            geometry=GeometryConfig(
                x1_nm=[0, 0, 0],
                x2_nm=[100, 0, 0]
            ),
            dynamics=DynamicsConfig(dt_us=1.0, gamma_pN_us_per_nm=1.0),
            loading=LoadingConfig(
                v_nm_per_us=1.0,
                t_end_us=30.0
            )
        )
        config.output.save_every_steps = 1

        result = run_simulation(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_results(result, Path(tmpdir), "test_run")

            with open(paths["csv"], 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Should have some intact and some ruptured
            intact_values = [row["intact"] for row in rows]
            assert "1" in intact_values, "Should have intact records"
            assert "0" in intact_values, "Should have ruptured records"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
