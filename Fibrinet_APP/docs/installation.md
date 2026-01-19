# Installation

## Requirements

- Python 3.10+
- pip

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Verify Installation

Run the test suite:

```bash
# Network tests
pytest test/ -v

# Single fiber tests
pytest projects/single_fiber/tests -v

# All tests
pytest test/ tests/ projects/single_fiber/tests/ -v
```

## Quick Start

### Network Simulator (GUI)

```bash
python FibriNet.py
```

### Network Simulator (CLI)

```bash
python scripts/cli/cli_main.py
```

### Collapse Analyzer

```bash
python scripts/cli/analyze_collapse_cli.py <path-to-xlsx> --constrain-center --iterate --out-dir exports/
```

### Single Fiber Simulator

```bash
# GUI
python -m projects.single_fiber.src.single_fiber.gui.app

# CLI
python -m projects.single_fiber.src.single_fiber.cli \
    -c projects/single_fiber/examples/hooke_ramp.yaml \
    -o output/
```

## Input Data Format

Network input files use Excel format (.xlsx) with three tables separated by blank rows:

1. **Nodes:** `n_id`, `n_x`, `n_y`
2. **Edges:** `e_id`, `n_from`, `n_to`
3. **Meta:** `meta_key`, `meta_value` (must include `spring_stiffness_constant`)

Example files are in `test/input_data/`.
