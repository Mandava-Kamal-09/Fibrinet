# FibriNet

Model and analyze fibrin-like networks under tension. Includes a Tkinter GUI with a **Research Simulation Tool** for mechanochemical fibrinolysis studies, plus a CLI for constrained collapse analysis.

## Features

### Research Simulation Tool (New)
Stochastic mechanochemical simulation engine for studying plasmin-mediated fibrinolysis under mechanical strain:
- **Worm-Like Chain (WLC) mechanics** using Marko-Siggia approximation
- **Strain-inhibited enzymatic cleavage** - stretched fibers resist degradation
- **Hybrid stochastic chemistry** (Gillespie SSA + tau-leaping)
- **Real-time visualization** of network degradation
- **Strain sweep experiments** to measure T50 clearance times

### Collapse Analyzer (CLI)
Physics-based network collapse analysis with stepwise outputs.

---

## Quick Start

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
```

### Run GUI (includes Research Simulation)

```bash
python Fibrinet_APP/FibriNet.py
```

### Run Collapse Analyzer (CLI)

```bash
python Fibrinet_APP/analyze_collapse_cli.py <path-to-xlsx> --constrain-center --iterate --out-dir Fibrinet_APP/exports
```

Optional: `--max-steps N` to limit iterations

---

## Input Format (.xlsx)

Single sheet with three tables separated by a blank row:

1. **Nodes:** `n_id`, `n_x`, `n_y`
2. **Edges:** `e_id`, `n_from`, `n_to`
3. **Meta:** `meta_key`, `meta_value` (must include `spring_stiffness_constant`)

See examples in `Fibrinet_APP/test/input_data/`.

---

## Research Simulation Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| Applied Strain | Network deformation | 0.0 - 0.5 |
| Plasmin Concentration | Enzyme density | 10 - 1000 nM |
| Cleavage Rate (k₀) | Base enzymatic rate | 0.01 - 1.0 s⁻¹ |
| Strain Sensitivity (β) | Mechanochemical coupling | 1.0 - 10.0 |

---

## Outputs

### Collapse Analyzer
- `initial_flush_region.png`
- `step_XXX.png` per removal
- `iteration_log.csv` (step, removed_edge_id, cut_size, cumulative_removed, etc.)

### Research Simulation
- Real-time network visualization
- Fiber integrity plots
- Simulation metadata (JSON)

---

## Documentation

Detailed documentation: [Google Drive folder](https://drive.google.com/drive/folders/1m1AaeAPe9KY9N34YW82rtmFUuHDx3FuP?usp=drive_link)

---

## References

- Marko, J.F. & Siggia, E.D. (1995) *Macromolecules* 28:8759-8770
- Li et al. (2017) *Biomacromolecules* 18:2074-2087
- Adhikari et al. (2012) *JACS* 134:13259-13265
