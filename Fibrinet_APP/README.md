# FibriNet

Simulation tools for studying fibrin mechanics.

## Components

### Fibrin Network Simulator

GUI and CLI tools for modeling fibrin-like networks under tension.

- Worm-Like Chain (WLC) mechanics using Marko-Siggia approximation
- Strain-inhibited enzymatic cleavage
- Stochastic chemistry (Gillespie SSA + tau-leaping)
- Network visualization
- Collapse analysis with stepwise iteration

**Run:**
```bash
# GUI
python FibriNet.py

# CLI
python scripts/cli/cli_main.py

# Collapse analyzer
python scripts/cli/analyze_collapse_cli.py <path-to-xlsx> --constrain-center --iterate --out-dir exports/
```

### Single Fiber Simulator

CLI-first simulator for individual fibrin fibers with enzyme coupling.

- Hookean and WLC force laws
- Overdamped dynamics
- Multiple enzyme hazard models
- Interactive visualization (DearPyGui)
- Parameter sweeps with reproducible seeds

**Run:**
```bash
# GUI
python -m projects.single_fiber.src.single_fiber.gui.app

# CLI
python -m projects.single_fiber.src.single_fiber.cli -c <config.yaml> -o <output_dir>
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

## Documentation

- [Overview](docs/overview.md)
- [Installation](docs/installation.md)
- [CLI Usage](docs/usage_cli.md)
- [Collapse Analyzer](docs/collapse_analyzer.md)
- [Single Fiber Guide](docs/single_fiber/usage.md)
- [Testing Guide](docs/TESTING_GUIDE.md)
- [Formula Reference](docs/QUICK_REFERENCE_FORMULA_SHEET.md)

See [docs/](docs/) for full documentation.

## Input Format

Network input files use Excel format (.xlsx) with three tables separated by blank rows:

1. **Nodes:** `n_id`, `n_x`, `n_y`
2. **Edges:** `e_id`, `n_from`, `n_to`
3. **Meta:** `meta_key`, `meta_value` (must include `spring_stiffness_constant`)

Example files are in `test/input_data/`.

## Testing

```bash
pytest test/ tests/ projects/single_fiber/tests/ -v
```

## References

- Marko, J.F. & Siggia, E.D. (1995) *Macromolecules* 28:8759-8770
- Li et al. (2017) *Biomacromolecules* 18:2074-2087
- Adhikari et al. (2012) *JACS* 134:13259-13265
