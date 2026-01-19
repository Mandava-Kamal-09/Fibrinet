"""
Parameter sweep runner for enzyme coupling experiments.

Runs batch simulations across parameter grids with full reproducibility.
Output is structured for downstream analysis.

Usage:
    python -m single_fiber.enzyme_models.sweep_runner sweep_config.yaml

Sweep Config Format (YAML):
    base_config: path/to/simulation_config.yaml
    hazard_model: exponential_strain
    parameter_grid:
      lambda0: [0.001, 0.01, 0.1]
      alpha: [1.0, 5.0, 10.0]
    n_replicates: 10
    base_seed: 42
    output_dir: sweeps/exp_strain_sweep
"""

import ast
import itertools
import json
import warnings
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime


@dataclass
class SweepConfig:
    """Configuration for a parameter sweep."""
    base_config_path: str
    hazard_model: str
    parameter_grid: Dict[str, List[float]]
    n_replicates: int = 10
    base_seed: int = 42
    output_dir: str = "sweeps/default"
    description: str = ""


@dataclass
class SweepResult:
    """Result of a single sweep run."""
    params: Dict[str, float]
    replicate: int
    seed: int
    rupture_time_us: float
    did_rupture: bool
    final_strain: float
    final_tension_pN: float
    csv_path: Optional[str] = None


def load_sweep_config(yaml_path: Path) -> SweepConfig:
    """
    Load sweep configuration from YAML file.

    Args:
        yaml_path: Path to YAML config

    Returns:
        SweepConfig instance
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    return SweepConfig(
        base_config_path=data.get("base_config", ""),
        hazard_model=data["hazard_model"],
        parameter_grid=data["parameter_grid"],
        n_replicates=data.get("n_replicates", 10),
        base_seed=data.get("base_seed", 42),
        output_dir=data.get("output_dir", "sweeps/default"),
        description=data.get("description", "")
    )


def generate_parameter_combinations(
    parameter_grid: Dict[str, List[float]]
) -> List[Dict[str, float]]:
    """
    Generate all combinations from parameter grid.

    Args:
        parameter_grid: Dict of param_name -> list of values

    Returns:
        List of dicts, each representing one parameter combination
    """
    keys = list(parameter_grid.keys())
    values = list(parameter_grid.values())

    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def run_single_simulation(
    base_config: Dict[str, Any],
    hazard_model: str,
    hazard_params: Dict[str, float],
    seed: int,
    output_path: Optional[Path] = None
) -> SweepResult:
    """
    Run a single simulation with given hazard parameters.

    Args:
        base_config: Base simulation configuration dict
        hazard_model: Name of hazard model
        hazard_params: Parameters for hazard model
        seed: Random seed
        output_path: Optional path to save CSV output

    Returns:
        SweepResult with rupture info
    """
    # Import here to avoid circular dependency
    from ..config import SimulationConfig
    from ..runner import run_simulation

    # Create modified config with enzyme settings
    config_dict = base_config.copy()

    # Enable enzyme with hazard model
    if "enzyme" not in config_dict:
        config_dict["enzyme"] = {}

    config_dict["enzyme"]["enabled"] = True
    config_dict["enzyme"]["seed"] = seed
    config_dict["enzyme"]["hazard_model"] = hazard_model
    config_dict["enzyme"]["hazard_params"] = hazard_params

    # Parse and validate config
    config = SimulationConfig(**config_dict)

    # Run simulation
    result = run_simulation(config)

    # Extract results
    rupture_time = result.rupture_time_us if result.rupture_occurred else float('inf')
    final_record = result.records[-1] if result.records else None

    sweep_result = SweepResult(
        params=hazard_params.copy(),
        replicate=0,  # Set by caller
        seed=seed,
        rupture_time_us=rupture_time,
        did_rupture=result.rupture_occurred,
        final_strain=final_record.strain if final_record else 0.0,
        final_tension_pN=final_record.tension_pN if final_record else 0.0,
        csv_path=str(output_path) if output_path else None
    )

    # Save CSV if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([asdict(r) for r in result.records])
        df.to_csv(output_path, index=False)

    return sweep_result


def run_sweep(
    config: SweepConfig,
    progress_callback=None
) -> List[SweepResult]:
    """
    Run full parameter sweep.

    Args:
        config: Sweep configuration
        progress_callback: Optional callback(completed, total) for progress

    Returns:
        List of SweepResult for all runs
    """
    # Warn about low replicate count for statistical power
    if config.n_replicates < 20:
        warnings.warn(
            f"n_replicates={config.n_replicates} is low for reliable statistics. "
            f"Consider n_replicates >= 20 for stable confidence intervals.",
            UserWarning
        )

    # Load base simulation config
    if config.base_config_path:
        with open(config.base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
    else:
        # Use default config
        base_config = {
            "model": {"law": "hooke", "hooke": {"k_pN_per_nm": 0.1, "L0_nm": 100.0}},
            "geometry": {"x1_nm": [0, 0, 0], "x2_nm": [100, 0, 0]},
            "dynamics": {"dt_us": 0.1, "gamma_pN_us_per_nm": 1.0},
            "loading": {"v_nm_per_us": 1.0, "t_end_us": 100.0}
        }

    # Generate parameter combinations
    param_combos = generate_parameter_combinations(config.parameter_grid)

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save sweep metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "hazard_model": config.hazard_model,
        "parameter_grid": config.parameter_grid,
        "n_replicates": config.n_replicates,
        "base_seed": config.base_seed,
        "n_combinations": len(param_combos),
        "total_runs": len(param_combos) * config.n_replicates,
        "description": config.description
    }
    with open(output_dir / "sweep_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Run all simulations
    results = []
    total_runs = len(param_combos) * config.n_replicates
    completed = 0

    for combo_idx, params in enumerate(param_combos):
        for rep in range(config.n_replicates):
            # Compute unique seed
            seed = config.base_seed + combo_idx * config.n_replicates + rep

            # Create output path
            param_str = "_".join(f"{k}{v:.4g}" for k, v in params.items())
            csv_name = f"combo{combo_idx:04d}_rep{rep:03d}_{param_str}.csv"
            csv_path = output_dir / "runs" / csv_name

            try:
                result = run_single_simulation(
                    base_config=base_config,
                    hazard_model=config.hazard_model,
                    hazard_params=params,
                    seed=seed,
                    output_path=csv_path
                )
                result.replicate = rep
                results.append(result)
            except Exception as e:
                print(f"Error in combo {combo_idx}, rep {rep}: {e}")
                # Record failed run
                results.append(SweepResult(
                    params=params,
                    replicate=rep,
                    seed=seed,
                    rupture_time_us=float('nan'),
                    did_rupture=False,
                    final_strain=float('nan'),
                    final_tension_pN=float('nan'),
                    csv_path=None
                ))

            completed += 1
            if progress_callback:
                progress_callback(completed, total_runs)

    # Save summary
    summary_df = pd.DataFrame([asdict(r) for r in results])
    summary_df.to_csv(output_dir / "sweep_summary.csv", index=False)

    return results


def analyze_sweep_results(
    results: List[SweepResult]
) -> pd.DataFrame:
    """
    Analyze sweep results: compute statistics per parameter combination.

    Args:
        results: List of SweepResult from run_sweep

    Returns:
        DataFrame with statistics per parameter combination
    """
    # Convert to DataFrame
    df = pd.DataFrame([asdict(r) for r in results])

    # Group by parameters (excluding replicate-specific columns)
    param_cols = [c for c in df.columns if c not in
                  ['replicate', 'seed', 'rupture_time_us', 'did_rupture',
                   'final_strain', 'final_tension_pN', 'csv_path']]

    # Create string key for grouping
    df['param_key'] = df[param_cols].apply(
        lambda row: str(dict(row)), axis=1
    )

    # Compute statistics
    stats = []
    for key, group in df.groupby('param_key'):
        params = ast.literal_eval(key)  # Safely convert string back to dict

        finite_times = group[np.isfinite(group['rupture_time_us'])]['rupture_time_us']

        stat = {
            **params,
            'n_replicates': len(group),
            'n_ruptured': group['did_rupture'].sum(),
            'fraction_ruptured': group['did_rupture'].mean(),
            'mean_rupture_time_us': finite_times.mean() if len(finite_times) > 0 else float('inf'),
            'std_rupture_time_us': finite_times.std() if len(finite_times) > 1 else 0.0,
            'median_rupture_time_us': finite_times.median() if len(finite_times) > 0 else float('inf'),
        }
        stats.append(stat)

    return pd.DataFrame(stats)


def create_example_sweep_config(output_path: Path) -> None:
    """Create an example sweep configuration file."""
    example = {
        "base_config": "examples/hooke_ramp.yaml",
        "hazard_model": "exponential_strain",
        "parameter_grid": {
            "lambda0": [0.001, 0.01, 0.1],
            "alpha": [1.0, 5.0, 10.0]
        },
        "n_replicates": 10,
        "base_seed": 42,
        "output_dir": "sweeps/example_sweep",
        "description": "Example parameter sweep for exponential strain hazard"
    }

    with open(output_path, 'w') as f:
        yaml.dump(example, f, default_flow_style=False)


__all__ = [
    "SweepConfig",
    "SweepResult",
    "load_sweep_config",
    "generate_parameter_combinations",
    "run_single_simulation",
    "run_sweep",
    "analyze_sweep_results",
    "create_example_sweep_config",
]
