# Single Fiber Simulation

CLI-first, publication-grade single fibrin fiber simulator with Hookean and WLC (Worm-Like Chain) force laws.

## Features

- **Two force laws**: Hookean (linear) and WLC (Marko-Siggia) with shared implementation
- **Overdamped dynamics**: Appropriate for low Reynolds number polymer systems
- **Displacement-controlled loading**: Constant-velocity ramp with hard or soft constraints
- **WLC rupture**: Fiber breaks when extension reaches contour length
- **Enzyme scaffold**: Interface for future strain-dependent cleavage (Bell model ready)
- **Deterministic**: Identical outputs given identical config + seed
- **Publication-ready exports**: CSV data and JSON metadata

## Quick Start

```bash
# From Fibrinet_APP directory

# Run Hookean example
python -m projects.single_fiber.src.single_fiber.cli -c projects/single_fiber/examples/hooke_ramp.yaml -o output/

# Run WLC example
python -m projects.single_fiber.src.single_fiber.cli -c projects/single_fiber/examples/wlc_ramp.yaml -o output/
```

## Units (Canonical)

| Quantity | Unit | Symbol |
|----------|------|--------|
| Length | nanometers | nm |
| Force | piconewtons | pN |
| Time | microseconds | μs |
| Energy | pN·nm | pN·nm |
| Thermal energy (300K) | 4.114 pN·nm | kBT |

## Configuration (YAML)

```yaml
model:
  law: hooke | wlc
  hooke:
    k_pN_per_nm: 0.1      # Spring constant
    L0_nm: 100.0          # Rest length
    extension_only: true  # No compression forces
  wlc:
    Lp_nm: 50.0           # Persistence length
    Lc_nm: 200.0          # Contour length (rupture threshold)
    kBT_pN_nm: 4.114      # Thermal energy
    rupture_at_Lc: true   # Enable rupture

geometry:
  x1_nm: [0, 0, 0]        # Fixed node position
  x2_nm: [100, 0, 0]      # Free node initial position

dynamics:
  dt_us: 0.1              # Time step
  gamma_pN_us_per_nm: 1.0 # Drag coefficient

loading:
  mode: displacement_ramp
  axis: [1, 0, 0]         # Pull direction
  v_nm_per_us: 1.0        # Velocity
  t_end_us: 100.0         # Duration
  constraint: hard        # hard | soft

enzyme:
  enabled: false
  seed: 42
  baseline_lambda_per_us: 0.01  # Optional constant rate

output:
  out_dir: output
  run_name: my_run
  save_every_steps: 10
```

## Output Files

### CSV (`<run_name>.csv`)

| Column | Description |
|--------|-------------|
| t_us | Time (μs) |
| x1_x_nm, x1_y_nm, x1_z_nm | Node 1 position (nm) |
| x2_x_nm, x2_y_nm, x2_z_nm | Node 2 position (nm) |
| L_nm | Segment length (nm) |
| strain | Engineering strain (L-L0)/L0 |
| tension_pN | Tension (pN) |
| law_name | Force law (hooke/wlc) |
| intact | 1 if intact, 0 if ruptured |
| rupture_time_us | Time of rupture (if occurred) |
| hazard_lambda_per_us | Enzyme hazard rate (if enabled) |
| hazard_H | Integrated hazard (if enabled) |

### Metadata JSON (`<run_name>_metadata.json`)

Contains full configuration, git commit hash, and run summary.

## Physics

### Hookean Spring

```
T = k × max(0, L - L₀)
```

Linear spring with optional tension-only mode.

### WLC (Marko-Siggia)

```
x = L / Lc
T = (kBT / Lp) × [1/(4(1-x)²) - 1/4 + x]
```

Entropic spring that diverges as L → Lc. Fiber ruptures at L ≥ Lc.

### Overdamped Dynamics

```
x_new = x_old + (dt / γ) × F_total
```

Inertia neglected (low Reynolds number limit).

## Running Tests

```bash
# From Fibrinet_APP directory
pytest projects/single_fiber/tests -v
```

## Project Structure

```
projects/single_fiber/
├── README.md
├── pyproject.toml
├── src/single_fiber/
│   ├── __init__.py
│   ├── config.py         # YAML config loading/validation
│   ├── state.py          # Fiber state representation
│   ├── model.py          # Force computation (uses shared force laws)
│   ├── loading.py        # Displacement-controlled loading
│   ├── integrator.py     # Overdamped dynamics
│   ├── enzyme_interface.py # Enzyme cleavage scaffold
│   ├── runner.py         # Main simulation runner
│   ├── exporters.py      # CSV/JSON export
│   └── cli.py            # Command-line interface
├── tests/
│   ├── test_config_validation.py
│   ├── test_overdamped_equilibrium_hooke.py
│   ├── test_overdamped_equilibrium_wlc.py
│   ├── test_displacement_ramp_reproducible.py
│   ├── test_rupture_behavior_wlc.py
│   └── test_export_schema.py
└── examples/
    ├── hooke_ramp.yaml
    └── wlc_ramp.yaml
```

## Dependencies

- numpy
- pyyaml
- pytest (for tests)

All force laws are imported from `src/core/force_laws/` (shared with network simulation).
