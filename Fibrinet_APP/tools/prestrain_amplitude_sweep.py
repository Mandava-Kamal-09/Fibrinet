"""
Prestrain Amplitude Sweep — Calibrate PRESTRAIN_AMPLITUDE for Cone 2020
========================================================================

Sweeps PRESTRAIN_AMPLITUDE across [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
with 3 seeds each.  For each (amplitude, seed) pair the full Cone 2020
protocol is run (5 prestrains, lysis@15 s, normalized), yielding per-seed
R^2 and RMSE.  Error bars come from the seed-to-seed variance.

Usage:
    python -m tools.prestrain_amplitude_sweep

Output:
    results/prestrain_amplitude_sweep/sweep_results.csv
    results/prestrain_amplitude_sweep/sweep_report.txt
    results/figures/fig_08_prestrain_sweep.png
"""

import sys
import os
import csv
import time as wt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import src.core.fibrinet_core_v2 as core
from src.validation.experimental_comparison import (
    _run_headless_timed, CONE_2020, _r_squared, _rmse,
)


# Configuration

AMPLITUDES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
SEEDS = [42, 43, 44]

# Same simulation parameters as validate_cone_2020
SIM_PARAMS = dict(
    lam0=1.0,
    dt=1.0,
    t_max=1800.0,
    t_snapshot=15.0,
    delta_S=1.0,
    strain_mode='affine',
)

# Reference R^2 at amplitude=0.0 (production baseline)
PRODUCTION_R2 = -0.7416

OUTPUT_DIR = os.path.join(_ROOT, 'results', 'prestrain_amplitude_sweep')
FIG_DIR = os.path.join(_ROOT, 'results', 'figures')


# Single (amplitude, seed) runner

def run_cone_single_seed(amplitude: float, seed: int) -> dict:
    """Run the 5-prestrain Cone protocol for one (amplitude, seed) pair.

    Patches PhysicalConstants.PRESTRAIN_AMPLITUDE on the class, runs all
    five prestrains, normalizes lysis@15s, and returns R^2 / RMSE /
    within_2sig.  Restores amplitude to 0.0 in finally.
    """
    prestrains = CONE_2020['prestrain']
    exp_area = CONE_2020['area_cleared']
    exp_sem = CONE_2020['sem']

    original_amp = core.PhysicalConstants.PRESTRAIN_AMPLITUDE
    try:
        core.PhysicalConstants.PRESTRAIN_AMPLITUDE = amplitude

        lysis_values = []
        for ps in prestrains:
            res = _run_headless_timed(
                strain=ps, seed=seed,
                lam0=SIM_PARAMS['lam0'],
                dt=SIM_PARAMS['dt'],
                t_max=SIM_PARAMS['t_max'],
                t_snapshot=SIM_PARAMS['t_snapshot'],
                delta_S=SIM_PARAMS['delta_S'],
                strain_mode=SIM_PARAMS['strain_mode'],
            )
            lysis_values.append(res['lysis_at_snapshot'])
    finally:
        core.PhysicalConstants.PRESTRAIN_AMPLITUDE = original_amp

    lysis_arr = np.array(lysis_values)

    # Normalize to [0,1] by dividing by max (same as validate_cone_2020)
    ls_max = np.max(lysis_arr) if np.max(lysis_arr) > 0 else 1.0
    sim_norm = lysis_arr / ls_max

    r2 = _r_squared(exp_area, sim_norm)
    rmse = _rmse(exp_area, sim_norm)

    # Within-2-sigma check (same criterion as validate_cone_2020)
    within_2sig = True
    for i in range(len(prestrains)):
        if exp_sem[i] > 0 and abs(sim_norm[i] - exp_area[i]) > 2 * exp_sem[i]:
            within_2sig = False

    return {
        'amplitude': amplitude,
        'seed': seed,
        'r2': r2,
        'rmse': rmse,
        'within_2sig': within_2sig,
        'sim_norm': sim_norm.tolist(),
        'lysis_raw': lysis_arr.tolist(),
    }


# Full sweep

def run_sweep() -> list:
    """Run all (amplitude, seed) combinations."""
    total = len(AMPLITUDES) * len(SEEDS)
    results = []
    done = 0

    for amp in AMPLITUDES:
        for seed in SEEDS:
            done += 1
            t0 = wt.time()
            print(f"\n--- [{done}/{total}] amplitude={amp:.2f}  seed={seed} ---")
            try:
                res = run_cone_single_seed(amp, seed)
                wall = wt.time() - t0
                print(f"  R^2={res['r2']:.4f}  RMSE={res['rmse']:.4f}  "
                      f"2sig={'PASS' if res['within_2sig'] else 'FAIL'}  "
                      f"wall={wall:.0f}s")
                results.append(res)
            except Exception as e:
                wall = wt.time() - t0
                print(f"  ERROR: {type(e).__name__}: {e}  wall={wall:.0f}s")
                results.append({
                    'amplitude': amp,
                    'seed': seed,
                    'r2': float('nan'),
                    'rmse': float('nan'),
                    'within_2sig': False,
                    'sim_norm': [],
                    'lysis_raw': [],
                })

    return results


# Output: CSV

def save_csv(results: list, filepath: str):
    """Write per-(amplitude, seed) results to CSV."""
    fields = ['amplitude', 'seed', 'r2', 'rmse', 'within_2sig']
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    print(f"  CSV saved: {filepath}")


# Output: report

def save_report(results: list, filepath: str):
    """Write human-readable sweep report with aggregated statistics."""
    # Group by amplitude
    by_amp = {}
    for r in results:
        a = r['amplitude']
        by_amp.setdefault(a, []).append(r)

    lines = [
        "Prestrain Amplitude Sweep Report",
        "=" * 55,
        "",
        f"Amplitudes tested: {AMPLITUDES}",
        f"Seeds: {SEEDS}",
        f"Prestrains: {CONE_2020['prestrain'].tolist()}",
        f"Experimental area_cleared: {CONE_2020['area_cleared'].tolist()}",
        f"Production R^2 (amplitude=0.0): {PRODUCTION_R2}",
        "",
        "Aggregated Results (mean +/- std across seeds)",
        "-" * 55,
        f"{'Amplitude':>10}  {'R^2 mean':>10}  {'R^2 std':>9}  "
        f"{'RMSE mean':>10}  {'2sig':>5}",
    ]

    best_amp = None
    best_r2_mean = -np.inf
    improved_amps = []
    positive_amps = []

    for amp in AMPLITUDES:
        group = by_amp.get(amp, [])
        r2_vals = [r['r2'] for r in group if not np.isnan(r['r2'])]
        rmse_vals = [r['rmse'] for r in group if not np.isnan(r['rmse'])]
        n_pass = sum(1 for r in group if r['within_2sig'])

        r2_mean = np.mean(r2_vals) if r2_vals else float('nan')
        r2_std = np.std(r2_vals) if r2_vals else float('nan')
        rmse_mean = np.mean(rmse_vals) if rmse_vals else float('nan')
        pass_str = f"{n_pass}/{len(group)}"

        marker = ""
        if not np.isnan(r2_mean) and r2_mean > best_r2_mean:
            best_r2_mean = r2_mean
            best_amp = amp

        if not np.isnan(r2_mean) and r2_mean > PRODUCTION_R2:
            improved_amps.append(amp)
        if not np.isnan(r2_mean) and r2_mean > 0:
            positive_amps.append(amp)

        lines.append(
            f"  {amp:>8.2f}  {r2_mean:>10.4f}  {r2_std:>9.4f}  "
            f"{rmse_mean:>10.4f}  {pass_str:>5}"
        )

    # Mark best
    lines.append("")
    lines.append(f"  * Best R^2: amplitude={best_amp:.2f} "
                 f"(mean R^2={best_r2_mean:.4f})")

    lines += [
        "",
        "Flags",
        "-" * 55,
    ]

    if improved_amps:
        lines.append(f"  Amplitudes with mean R^2 > {PRODUCTION_R2} (production): "
                     f"{improved_amps}")
    else:
        lines.append(f"  No amplitude improved R^2 beyond production ({PRODUCTION_R2}).")

    if positive_amps:
        lines.append(f"  Amplitudes with mean R^2 > 0.0: {positive_amps}")
    else:
        lines.append("  No amplitude achieved positive R^2.")

    lines += [
        "",
        "Recommendation",
        "-" * 55,
    ]

    if best_r2_mean > PRODUCTION_R2:
        lines.append(
            f"  Set PRESTRAIN_AMPLITUDE = {best_amp:.2f} "
            f"(R^2 = {best_r2_mean:.4f}, improves over production {PRODUCTION_R2})."
        )
    elif best_r2_mean > 0:
        lines.append(
            f"  Best amplitude {best_amp:.2f} achieves positive R^2 = {best_r2_mean:.4f}."
        )
    else:
        lines.append(
            f"  Best R^2 is {best_r2_mean:.4f} (still negative). "
            "Spatial prestrain alone does not resolve the Cone 2020 gap. "
            "Consider combining with other parameter adjustments."
        )

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Report saved: {filepath}")


# Output: figure

def plot_sweep(results: list, filepath: str):
    """Two-panel figure: R^2 and RMSE vs amplitude with error bars."""
    by_amp = {}
    for r in results:
        a = r['amplitude']
        by_amp.setdefault(a, []).append(r)

    amps = []
    r2_means = []
    r2_stds = []
    rmse_means = []
    rmse_stds = []

    for amp in AMPLITUDES:
        group = by_amp.get(amp, [])
        r2_vals = [r['r2'] for r in group if not np.isnan(r['r2'])]
        rmse_vals = [r['rmse'] for r in group if not np.isnan(r['rmse'])]

        amps.append(amp)
        r2_means.append(np.mean(r2_vals) if r2_vals else float('nan'))
        r2_stds.append(np.std(r2_vals) if r2_vals else 0.0)
        rmse_means.append(np.mean(rmse_vals) if rmse_vals else float('nan'))
        rmse_stds.append(np.std(rmse_vals) if rmse_vals else 0.0)

    amps = np.array(amps)
    r2_means = np.array(r2_means)
    r2_stds = np.array(r2_stds)
    rmse_means = np.array(rmse_means)
    rmse_stds = np.array(rmse_stds)

    best_r2_idx = int(np.nanargmax(r2_means))
    best_rmse_idx = int(np.nanargmin(rmse_means))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: R^2 vs amplitude
    ax1.errorbar(amps, r2_means, yerr=r2_stds, fmt='o-', color='#2c3e50',
                 lw=1.8, markersize=7, capsize=5, capthick=1.2)
    ax1.axhline(PRODUCTION_R2, color='#e74c3c', ls='--', lw=1.0, alpha=0.7,
                label=f'Production R$^2$={PRODUCTION_R2}')
    ax1.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.6,
                label='R$^2$=0 target')
    ax1.plot(amps[best_r2_idx], r2_means[best_r2_idx], '*',
             color='#e67e22', markersize=18, zorder=5,
             label=f'Best R$^2$={r2_means[best_r2_idx]:.4f} '
                   f'at {amps[best_r2_idx]:.2f}')
    ax1.set_xlabel('PRESTRAIN_AMPLITUDE', fontsize=10)
    ax1.set_ylabel('R$^2$ (Cone 2020)', fontsize=10)
    ax1.set_title('R$^2$ vs Spatial Prestrain Amplitude')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right panel: RMSE vs amplitude
    ax2.errorbar(amps, rmse_means, yerr=rmse_stds, fmt='s-', color='#8e44ad',
                 lw=1.8, markersize=7, capsize=5, capthick=1.2)
    ax2.plot(amps[best_rmse_idx], rmse_means[best_rmse_idx], '*',
             color='#e67e22', markersize=18, zorder=5,
             label=f'Best RMSE={rmse_means[best_rmse_idx]:.4f} '
                   f'at {amps[best_rmse_idx]:.2f}')
    ax2.set_xlabel('PRESTRAIN_AMPLITUDE', fontsize=10)
    ax2.set_ylabel('RMSE (Cone 2020)', fontsize=10)
    ax2.set_title('RMSE vs Spatial Prestrain Amplitude')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Spatial Prestrain Amplitude Sweep \u2014 Cone 2020',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {filepath}")


# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 60)
    print("  Prestrain Amplitude Sweep — Cone 2020")
    print("=" * 60)
    print(f"  Amplitudes: {AMPLITUDES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total iterations: {len(AMPLITUDES) * len(SEEDS)} x 5 prestrains "
          f"= {len(AMPLITUDES) * len(SEEDS) * 5} simulations")

    t_start = wt.time()

    results = run_sweep()

    wall_total = wt.time() - t_start

    # Save outputs
    csv_path = os.path.join(OUTPUT_DIR, 'sweep_results.csv')
    report_path = os.path.join(OUTPUT_DIR, 'sweep_report.txt')
    fig_path = os.path.join(FIG_DIR, 'fig_08_prestrain_sweep.png')

    save_csv(results, csv_path)
    save_report(results, report_path)
    plot_sweep(results, fig_path)

    # Summary
    by_amp = {}
    for r in results:
        by_amp.setdefault(r['amplitude'], []).append(r)
    r2_by_amp = {a: np.mean([r['r2'] for r in g if not np.isnan(r['r2'])])
                 for a, g in by_amp.items()}
    best_amp = max(r2_by_amp, key=r2_by_amp.get)

    print("\n" + "=" * 60)
    print("  SWEEP COMPLETE")
    print("=" * 60)
    print(f"  Best amplitude:  {best_amp:.2f}")
    print(f"  Best mean R^2:   {r2_by_amp[best_amp]:.4f}")
    print(f"  Production R^2:  {PRODUCTION_R2}")
    print(f"  Total wall time: {wall_total / 60:.1f} min")

    # Verify reset
    current = core.PhysicalConstants.PRESTRAIN_AMPLITUDE
    assert current == 0.0, (
        f"PRESTRAIN_AMPLITUDE not reset! Got {current}, expected 0.0"
    )
    print(f"  PRESTRAIN_AMPLITUDE reset to {current} (verified)")


if __name__ == '__main__':
    main()
