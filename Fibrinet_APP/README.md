# FibriNet: Dynamical Degradation of Fibrin Networks

A computational research tool for simulating the mechanochemical degradation of fibrin fiber networks. FibriNet models the interplay between mechanical strain and enzymatic (plasmin-mediated) cleavage to study how fibrin clots dissolve under physiological conditions.

## Physics Model

FibriNet implements a hybrid mechanochemical simulation coupling continuum polymer mechanics with stochastic enzyme kinetics:

### Worm-Like Chain (WLC) Mechanics
Each fibrin fiber is modeled as a semiflexible polymer using the Marko-Siggia interpolation formula:

```
F(e) = (k_B T / xi) * [1/(4(1-e)^2) - 1/4 + e]
```

where `e` is the fractional extension, `k_B T` is the thermal energy at 37 C (4.28e-21 J), and `xi` is the persistence length (1.0 um).

### Strain-Inhibited Enzymatic Cleavage
Plasmin cleavage rates depend on local fiber strain via an exponential inhibition law:

```
k(e) = k_0 * exp(-beta * e)
```

where `k_0 = 0.1 s^-1` is the baseline cleavage rate and `beta = 10` is the strain mechanosensitivity parameter (Adhikari et al., 2012). Fibers under high tension are protected from cleavage.

### Mechanical Equilibrium
Network relaxation is computed via L-BFGS-B energy minimization with analytical Jacobian, providing vectorized O(N) gradient evaluation per iteration.

### Stochastic Chemistry
Fiber cleavage events are sampled using a hybrid Gillespie SSA / tau-leaping algorithm that automatically selects the appropriate method based on total propensity.

### Percolation Detection
BFS-based left-to-right connectivity checking determines when the network loses structural integrity (clot clearance).

### Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Persistence length (xi) | 1.0 um | Collet et al. (2005) |
| Polymerization prestrain | 23% | Cone et al. (2020) |
| Baseline cleavage rate (k_0) | 0.1 s^-1 | Weisel & Litvinov (2017) |
| Strain sensitivity (beta) | 10.0 | Adhikari et al. (2012) |
| Temperature | 310.15 K (37 C) | Physiological |
| WLC strain cap | 0.99 | Singularity guard |

## Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

### Dependencies

- NumPy, SciPy (numerical computing and optimization)
- Matplotlib (real-time visualization and plotting)
- Pandas, OpenPyXL (network data I/O)
- Pillow (image export)
- Tkinter (GUI framework, included with Python)

## Usage

### GUI Application

```bash
python FibriNet.py
```

The GUI provides:
- Network loading from Excel files
- Interactive parameter configuration (strain, plasmin concentration, timestep)
- Strain mode selection: **Boundary Only** (legacy) or **All Fibers (Affine)**
- Real-time visualization with tension-based edge coloring
- Live lysis and tension plots
- Diagnostics export (JSON)
- Degradation history export (CSV)

### Network Input Format

Network files use Excel format (.xlsx) with stacked tables separated by blank rows:

1. **Nodes table**: `n_id`, `n_x`, `n_y`, `is_left_boundary`, `is_right_boundary`
2. **Edges table**: `e_id`, `n_from`, `n_to`, `thickness`
3. **Meta table** (optional): `meta_key`, `meta_value`

Example network files are provided in `data/input_networks/`.

## Repository Structure

```
FibriNet_APP/
|-- FibriNet.py                  # Application entry point
|-- requirements.txt             # Python dependencies
|-- README.md                    # This file
|-- data/
|   |-- input_networks/          # Example fibrin network files (.xlsx)
|-- src/
|   |-- config/                  # Physics constants, feature flags, config schema
|   |-- controllers/             # Application controller (MVC pattern)
|   |-- core/                    # Core V2 physics engine
|   |   |-- fibrinet_core_v2.py  # WLC mechanics, energy minimization, stochastic chemistry
|   |   |-- fibrinet_core_v2_adapter.py  # GUI adapter with unit conversions
|   |   |-- force_laws/          # Hookean and WLC force law implementations
|   |-- diagnostics/             # Simulation diagnostics (tension, strain, rupture analysis)
|   |-- events/                  # Event bus for decoupled UI updates
|   |-- logging/                 # Structured JSONL logging
|   |-- managers/                # Business logic managers
|   |   |-- network/             # Network graph models (nodes, edges, networks)
|   |   |-- export/              # Export strategies (Excel, PNG, CSV)
|   |   |-- input/               # Input parsing and validation
|   |-- models/                  # Data models and exceptions
|   |-- runners/                 # Headless simulation runner
|   |-- simulation/              # Simulation state machine, RNG, batch executor
|   |-- validation/              # Canonical network validation
|   |-- views/                   # GUI (Tkinter) and CLI views
|-- utils/                       # Logging utilities
```

## Strain Modes

FibriNet supports two strain application modes:

- **Boundary Only** (default): Applied strain displaces only right boundary nodes. Interior nodes reach equilibrium through energy minimization. This models a uniaxial tensile test with rigid grips.

- **All Fibers (Affine)**: Applied strain displaces all nodes proportionally to their x-position. Left boundary stays fixed, right boundary receives full displacement, and interior nodes are linearly interpolated. This provides a uniform initial strain field before mechanical relaxation.

## References

1. Marko, J.F. & Siggia, E.D. (1995). Stretching DNA. *Macromolecules*, 28:8759-8770.
2. Li, W. et al. (2017). Fibrin Fiber Stiffness Is Strongly Affected by Fiber Diameter, but Not by Fibrinogen Glycation. *Biophysical Journal*, 112:2017.
3. Adhikari, A.S. et al. (2012). Mechanical Load Induces a 100-Fold Increase in the Rate of Collagen Proteolysis by MMP-1. *JACS*, 134:13259-13265.
4. Cone, S.J. et al. (2020). Inherent Fiber Tension and Connectivity Regulate Fibrinolysis. *Biophysical Journal*, 118:963-980.
5. Bucay, I. et al. (2015). Physical Determinants of Fibrinolysis in Single Fibrin Fibers. *PLoS ONE*, 10:e0116350.
6. Weisel, J.W. & Litvinov, R.I. (2017). Fibrin Formation, Structure and Properties. *Subcellular Biochemistry*, 82:405-456.

## Author

Kamal Mandava, University of Central Oklahoma
