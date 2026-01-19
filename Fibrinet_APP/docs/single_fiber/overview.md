# Single Fiber Simulator

The single fiber simulator is a CLI-first tool for modeling individual fibrin fibers with optional enzyme coupling.

For complete documentation, see: [projects/single_fiber/README.md](../../projects/single_fiber/README.md)

## Quick Start

### GUI Mode

```bash
python -m projects.single_fiber.src.single_fiber.gui.app
```

### CLI Mode

```bash
python -m projects.single_fiber.src.single_fiber.cli \
    -c projects/single_fiber/examples/hooke_ramp.yaml \
    -o output/
```

## Key Features

- **Force Laws:** Hookean (linear) and WLC (Marko-Siggia)
- **Dynamics:** Overdamped, quasi-static
- **Enzyme Models:** Constant, linear, exponential, Bell slip, catch-slip hazards
- **GUI:** DearPyGui-based interactive visualization
- **Sweeps:** Batch parameter exploration with reproducible seeds

## Documentation

- [Usage Guide](usage.md) - Step-by-step tutorial
- [Validation](validation.md) - Validation approach and status
- [Full README](../../projects/single_fiber/README.md) - Complete reference
