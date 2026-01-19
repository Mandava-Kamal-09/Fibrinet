"""
Export simulation results to CSV and JSON.

CSV columns (exact schema):
    t_us, x1_x_nm, x1_y_nm, x1_z_nm, x2_x_nm, x2_y_nm, x2_z_nm,
    L_nm, strain, tension_pN, law_name, intact, rupture_time_us,
    hazard_lambda_per_us, hazard_H
"""

import csv
import json
import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime

from .runner import SimulationResult
from .config import SimulationConfig


CSV_COLUMNS = [
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


def export_csv(result: SimulationResult, path: Path) -> None:
    """
    Export simulation records to CSV.

    Args:
        result: Simulation result.
        path: Output CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)

        for rec in result.records:
            row = [
                rec.t_us,
                rec.x1_nm[0], rec.x1_nm[1], rec.x1_nm[2],
                rec.x2_nm[0], rec.x2_nm[1], rec.x2_nm[2],
                rec.L_nm,
                rec.strain,
                rec.tension_pN,
                rec.law_name,
                1 if rec.intact else 0,
                rec.rupture_time_us if rec.rupture_time_us is not None else "",
                rec.hazard_lambda_per_us if rec.hazard_lambda_per_us is not None else "",
                rec.hazard_H if rec.hazard_H is not None else ""
            ]
            writer.writerow(row)


def get_git_commit() -> Optional[str]:
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def config_to_dict(config: SimulationConfig) -> dict:
    """Convert config to serializable dict."""
    return {
        "model": {
            "law": config.model.law,
            "hooke": {
                "k_pN_per_nm": config.model.hooke.k_pN_per_nm,
                "L0_nm": config.model.hooke.L0_nm,
                "extension_only": config.model.hooke.extension_only
            } if config.model.hooke else None,
            "wlc": {
                "Lp_nm": config.model.wlc.Lp_nm,
                "Lc_nm": config.model.wlc.Lc_nm,
                "kBT_pN_nm": config.model.wlc.kBT_pN_nm,
                "rupture_at_Lc": config.model.wlc.rupture_at_Lc
            } if config.model.wlc else None
        },
        "geometry": {
            "x1_nm": config.geometry.x1_nm,
            "x2_nm": config.geometry.x2_nm,
            "initial_length_nm": config.geometry.initial_length_nm
        },
        "dynamics": {
            "dt_us": config.dynamics.dt_us,
            "gamma_pN_us_per_nm": config.dynamics.gamma_pN_us_per_nm,
            "relax_steps_per_increment": config.dynamics.relax_steps_per_increment
        },
        "loading": {
            "mode": config.loading.mode,
            "axis": config.loading.axis,
            "v_nm_per_us": config.loading.v_nm_per_us,
            "t_end_us": config.loading.t_end_us,
            "constraint": config.loading.constraint
        },
        "enzyme": {
            "enabled": config.enzyme.enabled,
            "seed": config.enzyme.seed,
            "baseline_lambda_per_us": config.enzyme.baseline_lambda_per_us
        },
        "output": {
            "out_dir": config.output.out_dir,
            "run_name": config.output.run_name,
            "save_every_steps": config.output.save_every_steps
        }
    }


def export_metadata(result: SimulationResult, path: Path) -> None:
    """
    Export metadata JSON with config and summary.

    Args:
        result: Simulation result.
        path: Output JSON path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "config": config_to_dict(result.config),
        "summary": {
            "n_steps": len(result.records),
            "max_tension_pN": result.max_tension_pN,
            "final_strain": result.final_strain,
            "rupture_occurred": result.rupture_occurred,
            "rupture_time_us": result.rupture_time_us,
            "enzyme_cleaved": result.enzyme_cleaved
        }
    }

    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)


def export_results(result: SimulationResult, out_dir: Path, run_name: str) -> dict:
    """
    Export all results to output directory.

    Args:
        result: Simulation result.
        out_dir: Output directory.
        run_name: Base name for output files.

    Returns:
        Dict with paths to exported files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{run_name}.csv"
    json_path = out_dir / f"{run_name}_metadata.json"

    export_csv(result, csv_path)
    export_metadata(result, json_path)

    return {
        "csv": str(csv_path),
        "metadata": str(json_path)
    }
