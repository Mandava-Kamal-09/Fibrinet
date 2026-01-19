# FibriNet Overview

FibriNet contains two related simulation tools for studying fibrin mechanics:

## 1. Fibrin Network Simulator

A GUI and CLI tool for modeling and analyzing fibrin-like networks under tension.

**Features:**
- Worm-Like Chain (WLC) mechanics using Marko-Siggia approximation
- Strain-inhibited enzymatic cleavage
- Stochastic chemistry (Gillespie SSA + tau-leaping)
- Real-time network visualization
- Collapse analysis with stepwise iteration

**Entry points:**
- GUI: `python FibriNet.py`
- CLI: `python cli_main.py`
- Collapse analyzer: `python analyze_collapse_cli.py`

## 2. Single Fiber Simulator

A CLI-first simulator for individual fibrin fibers with enzyme coupling.

**Features:**
- Hookean and WLC force laws
- Overdamped dynamics
- Five enzyme hazard models
- Interactive DearPyGui visualization
- Parameter sweeps with reproducible seeds

**Entry points:**
- GUI: `python -m projects.single_fiber.src.single_fiber.gui.app`
- CLI: `python -m projects.single_fiber.src.single_fiber.cli -c <config.yaml> -o <output_dir>`

See [projects/single_fiber/README.md](../projects/single_fiber/README.md) for detailed documentation.

## Documentation Index

- [Installation](installation.md)
- [CLI Usage](usage_cli.md)
- [Collapse Analyzer](collapse_analyzer.md)
- [Testing Guide](TESTING_GUIDE.md)
- [Formula Reference](QUICK_REFERENCE_FORMULA_SHEET.md)
- [Single Fiber Documentation](../projects/single_fiber/README.md)
