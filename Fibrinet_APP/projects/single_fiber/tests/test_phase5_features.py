"""
Tests for Phase 5 features: Presets, Benchmarks, Protocols.

These tests verify:
- All presets load correctly and produce valid configs
- Benchmark script runs without error
- Protocol YAML files are valid
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))


class TestPresets:
    """Tests for preset configurations."""

    def test_list_presets_returns_six(self):
        """Should have exactly 6 presets."""
        from projects.single_fiber.src.single_fiber.gui.presets import list_presets
        presets = list_presets()
        assert len(presets) == 6

    def test_all_presets_have_names(self):
        """All presets should have expected names."""
        from projects.single_fiber.src.single_fiber.gui.presets import list_presets
        expected = [
            "hooke_baseline",
            "wlc_baseline",
            "hooke_constant_hazard",
            "hooke_exp_strain",
            "wlc_bell_slip",
            "catch_slip_demo",
        ]
        presets = list_presets()
        for name in expected:
            assert name in presets, f"Missing preset: {name}"

    def test_preset_configs_validate(self):
        """All preset configs should pass validation."""
        from projects.single_fiber.src.single_fiber.gui.presets import (
            list_presets, get_preset
        )
        for name in list_presets():
            preset = get_preset(name)
            is_valid, error = preset.config.validate()
            assert is_valid, f"Preset '{name}' config invalid: {error}"

    def test_preset_can_run_one_step(self):
        """Preset configs can initialize and run one physics step."""
        import numpy as np
        from projects.single_fiber.src.single_fiber.gui.presets import (
            list_presets, get_preset
        )
        from projects.single_fiber.src.single_fiber.chain_state import ChainState
        from projects.single_fiber.src.single_fiber.chain_model import ChainModel
        from projects.single_fiber.src.single_fiber.chain_integrator import (
            ChainIntegrator, ChainLoadingController
        )

        for name in list_presets():
            preset = get_preset(name)
            config = preset.config

            x1 = np.array(config.geometry.x1_nm)
            x2 = np.array(config.geometry.x2_nm)
            state = ChainState.from_endpoints(x1, x2, preset.n_segments)

            model = ChainModel(config.model)
            integrator = ChainIntegrator(config.dynamics)

            end_pos = state.nodes_nm[-1].copy()
            loading = ChainLoadingController(config.loading, end_pos)

            t = config.dynamics.dt_us
            target = loading.target_position(t)

            # Should complete without error
            state, forces, relax = integrator.step_with_relaxation(
                state, model, target, t, fixed_boundary_node=0
            )

            assert state is not None
            assert forces is not None

    def test_preset_display_names_exist(self):
        """All presets have human-readable display names."""
        from projects.single_fiber.src.single_fiber.gui.presets import (
            get_preset_display_names
        )
        names = get_preset_display_names()
        assert len(names) == 6
        for name, display in names.items():
            assert len(display) > 0
            assert display != name  # Should be human-readable

    def test_preset_enzyme_params_valid(self):
        """Presets with enzyme models have valid hazard params."""
        from projects.single_fiber.src.single_fiber.gui.presets import (
            list_presets, get_preset
        )

        enzyme_presets = ["hooke_exp_strain", "wlc_bell_slip", "catch_slip_demo"]

        for name in enzyme_presets:
            preset = get_preset(name)
            assert preset.enzyme_model is not None
            assert preset.enzyme_params is not None
            assert len(preset.enzyme_params) > 0


class TestBenchmark:
    """Tests for benchmark script."""

    def test_benchmark_imports(self):
        """Benchmark script imports successfully."""
        from projects.single_fiber.benchmarks import benchmark_performance
        assert hasattr(benchmark_performance, 'run_benchmark_suite')
        assert hasattr(benchmark_performance, 'benchmark_physics_steps')

    def test_benchmark_single_run(self):
        """Can run a single benchmark measurement."""
        from projects.single_fiber.benchmarks.benchmark_performance import (
            create_hooke_config, benchmark_physics_steps
        )

        config = create_hooke_config(n_segments=5)
        result = benchmark_physics_steps(config, n_segments=5, n_steps=10, warmup_steps=5)

        assert "steps_per_sec" in result
        assert "microseconds_per_step" in result
        assert result["steps_per_sec"] > 0
        assert result["n_steps"] == 10

    def test_benchmark_scales_with_segments(self):
        """More segments should (generally) take more time."""
        from projects.single_fiber.benchmarks.benchmark_performance import (
            create_hooke_config, benchmark_physics_steps
        )

        result_5 = benchmark_physics_steps(
            create_hooke_config(5), n_segments=5, n_steps=50, warmup_steps=10
        )
        result_20 = benchmark_physics_steps(
            create_hooke_config(20), n_segments=20, n_steps=50, warmup_steps=10
        )

        # More segments should be slower (lower steps/sec)
        # Allow some tolerance for measurement noise
        assert result_20["microseconds_per_step"] >= result_5["microseconds_per_step"] * 0.5


class TestProtocols:
    """Tests for protocol YAML templates."""

    def test_protocol_files_exist(self):
        """Protocol YAML files exist."""
        protocols_dir = _project_root / "projects" / "single_fiber" / "protocols"

        expected_files = [
            "hazard_comparison.yaml",
            "bell_beta_sweep.yaml",
            "catch_slip_lifetime_sweep.yaml",
            "replicate_convergence.yaml",
        ]

        for fname in expected_files:
            path = protocols_dir / fname
            assert path.exists(), f"Missing protocol file: {fname}"

    def test_protocol_files_are_valid_yaml(self):
        """Protocol files parse as valid YAML."""
        import yaml

        protocols_dir = _project_root / "projects" / "single_fiber" / "protocols"

        for yaml_file in protocols_dir.glob("*.yaml"):
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)

            assert data is not None
            assert "hazard_model" in data
            assert "parameter_grid" in data
            assert "n_replicates" in data

    def test_protocol_has_adequate_replicates(self):
        """Protocol files specify adequate replicate counts."""
        import yaml

        protocols_dir = _project_root / "projects" / "single_fiber" / "protocols"

        for yaml_file in protocols_dir.glob("*.yaml"):
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)

            n_rep = data.get("n_replicates", 0)
            # All protocols should have >= 20 replicates
            assert n_rep >= 20, f"{yaml_file.name} has only {n_rep} replicates"


class TestSweepOutputSchema:
    """Tests for sweep output consistency."""

    def test_sweep_result_dataclass_fields(self):
        """SweepResult has expected fields."""
        from projects.single_fiber.src.single_fiber.enzyme_models.sweep_runner import (
            SweepResult
        )
        from dataclasses import fields

        field_names = {f.name for f in fields(SweepResult)}

        expected = {
            "params", "replicate", "seed", "rupture_time_us",
            "did_rupture", "final_strain", "final_tension_pN", "csv_path"
        }

        assert field_names == expected

    def test_sweep_config_dataclass_fields(self):
        """SweepConfig has expected fields."""
        from projects.single_fiber.src.single_fiber.enzyme_models.sweep_runner import (
            SweepConfig
        )
        from dataclasses import fields

        field_names = {f.name for f in fields(SweepConfig)}

        expected = {
            "base_config_path", "hazard_model", "parameter_grid",
            "n_replicates", "base_seed", "output_dir", "description"
        }

        assert field_names == expected
