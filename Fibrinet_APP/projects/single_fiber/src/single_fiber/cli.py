"""
Command-line interface for single fiber simulation.

Usage:
    python -m single_fiber.cli --config examples/hooke_ramp.yaml --out output/
"""

import argparse
import sys
from pathlib import Path

from .config import load_config
from .runner import run_simulation
from .exporters import export_results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Single fiber simulation with Hookean or WLC force laws"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Run name (overrides config)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output except errors"
    )

    args = parser.parse_args()

    # Load and validate config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Override output settings if specified
    out_dir = args.out or Path(config.output.out_dir)
    run_name = args.name or config.output.run_name

    # Run simulation
    if not args.quiet:
        print(f"Running single fiber simulation...")
        print(f"  Law: {config.model.law}")
        print(f"  Initial length: {config.geometry.initial_length_nm:.2f} nm")
        print(f"  Loading: {config.loading.v_nm_per_us:.4f} nm/us for {config.loading.t_end_us:.1f} us")

    result = run_simulation(config)

    # Export results
    paths = export_results(result, out_dir, run_name)

    # Print summary
    if not args.quiet:
        print()
        print("=" * 50)
        print("SIMULATION COMPLETE")
        print("=" * 50)
        print(f"  Law: {config.model.law}")
        print(f"  Final strain: {result.final_strain:.4f} ({result.final_strain*100:.1f}%)")
        print(f"  Max tension: {result.max_tension_pN:.4f} pN")
        print(f"  Rupture: {'Yes' if result.rupture_occurred else 'No'}")
        if result.rupture_occurred:
            reason = "enzyme" if result.enzyme_cleaved else "mechanical"
            print(f"    Time: {result.rupture_time_us:.4f} us ({reason})")
        print()
        print("Output files:")
        print(f"  CSV: {paths['csv']}")
        print(f"  Metadata: {paths['metadata']}")

    sys.exit(0)


if __name__ == "__main__":
    main()
