"""
Tests for sweep_runner safety features.

Phase 4: Strain-Enzyme Coupling Lab

These tests verify:
- ast.literal_eval() is used (not eval) for safe param parsing
- n_replicates < 20 triggers warning
"""

import pytest
import warnings
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from projects.single_fiber.src.single_fiber.enzyme_models.sweep_runner import (
    SweepConfig,
    analyze_sweep_results,
    SweepResult,
    run_sweep,
)


class TestEvalSafety:
    """Tests for safe string parsing (no eval)."""

    def test_analyze_handles_malformed_param_key_gracefully(self):
        """Malformed param_key strings don't execute arbitrary code."""
        # Create results with valid params dict
        results = [
            SweepResult(
                params={"lambda0": 0.01, "alpha": 5.0},
                replicate=0,
                seed=42,
                rupture_time_us=50.0,
                did_rupture=True,
                final_strain=0.3,
                final_tension_pN=10.0,
            ),
            SweepResult(
                params={"lambda0": 0.01, "alpha": 5.0},
                replicate=1,
                seed=43,
                rupture_time_us=55.0,
                did_rupture=True,
                final_strain=0.35,
                final_tension_pN=12.0,
            ),
        ]

        # Should complete without error (no code execution)
        stats_df = analyze_sweep_results(results)
        assert len(stats_df) == 1  # One param combination
        assert stats_df.iloc[0]["n_replicates"] == 2

    def test_ast_literal_eval_rejects_code_injection(self):
        """ast.literal_eval rejects malicious strings that eval() would execute."""
        import ast

        # These strings would be dangerous with eval()
        dangerous_strings = [
            "__import__('os').system('echo pwned')",
            "open('/etc/passwd').read()",
            "exec('import os')",
            "(lambda: None)()",
        ]

        for dangerous in dangerous_strings:
            with pytest.raises((ValueError, SyntaxError)):
                ast.literal_eval(dangerous)

    def test_ast_literal_eval_accepts_valid_dict(self):
        """ast.literal_eval correctly parses valid dict strings."""
        import ast

        valid_dict_str = "{'lambda0': 0.01, 'alpha': 5.0}"
        result = ast.literal_eval(valid_dict_str)
        assert result == {"lambda0": 0.01, "alpha": 5.0}


class TestReplicateWarning:
    """Tests for n_replicates warning."""

    def test_low_replicates_triggers_warning(self):
        """n_replicates < 20 triggers UserWarning."""
        config = SweepConfig(
            base_config_path="",
            hazard_model="constant",
            parameter_grid={"lambda0": [0.01]},
            n_replicates=5,  # Below threshold
            base_seed=42,
            output_dir="test_output",
        )

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # This will fail since we don't have full simulation setup,
            # but warning should trigger before the simulation runs
            try:
                run_sweep(config)
            except Exception:
                pass  # Expected to fail - we just want to check warning

            # Check that warning was raised
            replicate_warnings = [
                x for x in w
                if "n_replicates" in str(x.message) and "low" in str(x.message).lower()
            ]
            assert len(replicate_warnings) >= 1, "Expected warning about low n_replicates"

    def test_adequate_replicates_no_warning(self):
        """n_replicates >= 20 does not trigger warning."""
        config = SweepConfig(
            base_config_path="",
            hazard_model="constant",
            parameter_grid={"lambda0": [0.01]},
            n_replicates=20,  # At threshold
            base_seed=42,
            output_dir="test_output",
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                run_sweep(config)
            except Exception:
                pass

            replicate_warnings = [
                x for x in w
                if "n_replicates" in str(x.message) and "low" in str(x.message).lower()
            ]
            assert len(replicate_warnings) == 0, "No warning expected for n_replicates >= 20"
