"""
GUI command-line interface entry point.

Usage:
    python -m single_fiber.gui_cli [-c config.yaml] [-n 5]
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Single Fiber Simulation GUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default 5-segment Hookean chain
    python -m single_fiber.gui_cli

    # Run with 10 segments
    python -m single_fiber.gui_cli -n 10

    # Run with custom config
    python -m single_fiber.gui_cli -c examples/wlc_ramp.yaml -n 3
"""
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to YAML config file (optional)"
    )
    parser.add_argument(
        "-n", "--segments",
        type=int,
        default=5,
        help="Number of segments in chain (default: 5)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Single Fiber Simulation",
        help="Window title"
    )

    args = parser.parse_args()

    # Import here to avoid DearPyGui import if just checking help
    from .gui.app import run_gui

    try:
        run_gui(args.config, args.segments)
    except KeyboardInterrupt:
        print("\nGUI closed.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
