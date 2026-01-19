## Deterministic Lysis Analysis (Poster‑Ready Figures)

This folder contains **headless, deterministic** analysis utilities for exported Research Simulation outputs.

### Inputs

- **Experiment log**: exported `experiment_log` as **JSON** (preferred) or CSV.
  - Recommended: use the GUI **Export Experiment Log** button and choose `.json`.
- **Network snapshot**: exported snapshot `.json` from the GUI **Export Network Snapshot** button.

### Advisor-aligned metrics (important definitions)

- **Fiber lysis**: an edge is considered lysed when its integrity scalar \(S \le 0\) (this is what the simulator logs).
- **Stiffness-loss lysis fraction** (current default observable): the simulator reports a stiffness-loss proxy:

  \[
  \text{lysis\_fraction} = 1 - \frac{\sum_i k0_i\,S_i}{\sum_i k0_i}
  \]

  This is **not** the same as percolation, but it is related.

- **Percolation / load-path failure (connectivity)**: in analysis, we can also compute whether there exists an intact-edge path connecting any left boundary node to any right boundary node (boundary membership is taken from `snapshot.frozen_params`, not inferred geometrically). This directly addresses “are the poles still connected?”

### Outputs

All figures are written to:

- `analysis/figures/`

Expected filenames (with `--tag TAG`):

- `lysis_fraction_vs_time_TAG.png`
- `survival_curve_TAG.png`
- `lysis_time_vs_thickness_TAG.png`
- `compare_plasmin_modes_TAG.png` (only if `--compare-log` is provided)
- `connectivity_vs_time_TAG.png` (only if `--snapshots-dir` is provided)

### Reproduce figures

From the repository root:

```bash
python analysis/lysis_analysis.py --log path/to/experiment_log.json --snapshot path/to/network_snapshot.json --tag demo
```

Optional comparison (e.g., saturating vs limited plasmin):

```bash
python analysis/lysis_analysis.py --log path/to/saturating_log.json --compare-log path/to/limited_log.json --snapshot path/to/network_snapshot.json --tag plasmin_compare
```

Optional percolation/connectivity curve (requires a directory of time-stamped per-batch snapshots):

```bash
python analysis/lysis_analysis.py --log path/to/experiment_log.json --snapshot path/to/network_snapshot.json --snapshots-dir path/to/snapshots/ --tag with_connectivity
```

### Advisor-ready “same thickness, different tension” strain sweep (headless)

This runs the same network at multiple **applied_strain** values (tension changes via mechanics) and exports logs + per-batch snapshots per strain:

```bash
python analysis/run_strain_sweep.py --network path/to/network.xlsx --strains 0.00,0.02,0.04,0.06,0.08,0.10 --lambda0 1.0 --dt 0.1 --max-batches 200 --out analysis/runs/strain_sweep
```

Then generate figures for one strain folder (example):

```bash
python analysis/lysis_analysis.py --log analysis/runs/strain_sweep/strain_0.06/experiment_log.json --snapshot analysis/runs/strain_sweep/strain_0.06/final_snapshot.json --snapshots-dir analysis/runs/strain_sweep/strain_0.06/snapshots --tag strain_0p06
```

### Determinism guarantees

- The analysis script uses a **headless Matplotlib backend** (`Agg`) and a **fixed style** (DPI, fonts, layout).
- No randomness is used.
- All ordering is explicit (sorting by time/edge_id where applicable).
- Given identical inputs, outputs are intended to be **bit-for-bit reproducible**.


