"""
Cascade Threshold Sweep — Calibrate CASCADE_RUPTURE_THRESHOLD for Cone 2020
============================================================================

Sweeps CASCADE_RUPTURE_THRESHOLD across a range of values and evaluates
the Cone 2020 prestrain-vs-clearance validation at each point. Identifies
the threshold that maximizes R^2 (best quantitative match to experiment).

Usage:
    python -m tools.cascade_threshold_sweep            # full 13-point sweep
    python -m tools.cascade_threshold_sweep --smoke     # quick 3-point check

Output:
    results/cascade_threshold_sweep/sweep_results.csv
    results/cascade_threshold_sweep/sweep_report.txt
    results/figures/cascade_threshold_sweep.png
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

from src.core.fibrinet_core_v2 import PhysicalConstants as PC
from src.validation.experimental_comparison import (
    validate_cone_2020, CONE_2020, _r_squared, _rmse,
)


# Sweep configurations

THRESHOLDS_SMOKE = [0.40, 0.50, 0.70]
THRESHOLDS_FULL = [
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.90,
]
ORIGINAL_THRESHOLD = 0.50

OUTPUT_DIR = os.path.join(_ROOT, 'results', 'cascade_threshold_sweep')
FIG_DIR = os.path.join(_ROOT, 'results', 'figures')


# Core sweep function

def run_cone_at_threshold(threshold: float, n_seeds: int = 3) -> dict:
    """Run Cone 2020 validation with a specific cascade threshold.

    Temporarily overrides PC.CASCADE_RUPTURE_THRESHOLD, runs the
    validation, then restores the original value (even on error).
    """
    orig = PC.CASCADE_RUPTURE_THRESHOLD
    try:
        PC.CASCADE_RUPTURE_THRESHOLD = threshold
        result = validate_cone_2020(n_seeds=n_seeds)
    finally:
        PC.CASCADE_RUPTURE_THRESHOLD = orig

    # Trend direction: positive slope = more strain -> more lysis (correct)
    prestrains = CONE_2020['prestrain']
    slope = np.polyfit(prestrains, result['sim_means'], 1)[0]
    trend = 'positive' if slope > 0 else 'negative'

    return {
        'threshold': threshold,
        'r2': result['r2'],
        'rmse': result['rmse'],
        'within_2sig': result['within_2sig'],
        'trend_direction': trend,
        'trend_slope': slope,
        'sim_means': result['sim_means'].tolist(),
        'sim_stds': result['sim_stds'].tolist(),
    }


# Smoke test

def smoke_test():
    """Quick validation that the sweep machinery works (3 thresholds)."""
    print("=" * 60)
    print("  SMOKE TEST — 3 thresholds")
    print("=" * 60)

    passed = 0
    for thr in THRESHOLDS_SMOKE:
        try:
            res = run_cone_at_threshold(thr)
            if np.isnan(res['r2']):
                print(f"  FAIL: threshold={thr:.2f} produced NaN R^2")
            else:
                print(f"  OK:   threshold={thr:.2f}  R^2={res['r2']:.4f}")
                passed += 1
        except Exception as e:
            print(f"  FAIL: threshold={thr:.2f} raised {type(e).__name__}: {e}")

    print(f"\nSmoke test: {passed}/{len(THRESHOLDS_SMOKE)} passed")
    if passed < len(THRESHOLDS_SMOKE):
        print("Aborting — smoke test did not pass all thresholds.")
        sys.exit(1)
    print()


# Full sweep

def full_sweep(thresholds: list) -> list:
    """Run Cone 2020 validation for each threshold value."""
    print("=" * 60)
    print(f"  FULL SWEEP — {len(thresholds)} thresholds")
    print("=" * 60)

    results = []
    for i, thr in enumerate(thresholds, 1):
        t0 = wt.time()
        print(f"\n--- [{i}/{len(thresholds)}] threshold = {thr:.2f} ---")
        res = run_cone_at_threshold(thr)
        wall = wt.time() - t0
        print(f"  R^2={res['r2']:.4f}  RMSE={res['rmse']:.4f}  "
              f"trend={res['trend_direction']}  wall={wall:.0f}s")
        results.append(res)

    return results


# Output: CSV

def save_csv(results: list, filepath: str):
    """Write sweep results to CSV."""
    fields = ['threshold', 'r2', 'rmse', 'within_2sig',
              'trend_direction', 'trend_slope']
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    print(f"  CSV saved: {filepath}")


# Output: report

def save_report(results: list, filepath: str):
    """Write human-readable sweep report."""
    best = max(results, key=lambda r: r['r2'])
    lowest_rmse = min(results, key=lambda r: r['rmse'])

    lines = [
        "Cascade Threshold Sweep Report",
        "=" * 50,
        "",
        f"Thresholds tested: {len(results)}",
        f"Seeds per prestrain: 3",
        f"Prestrains: {CONE_2020['prestrain'] * 100}%",
        f"Experimental area_cleared: {CONE_2020['area_cleared']}",
        "",
        "Results",
        "-" * 50,
        f"{'Threshold':>10}  {'R^2':>8}  {'RMSE':>8}  {'2sig':>5}  {'Trend':>9}",
    ]

    for r in results:
        marker = " *" if r is best else ""
        lines.append(
            f"  {r['threshold']:>8.2f}  {r['r2']:>8.4f}  {r['rmse']:>8.4f}  "
            f"{'PASS' if r['within_2sig'] else 'FAIL':>5}  "
            f"{r['trend_direction']:>9}{marker}"
        )

    lines += [
        "",
        "Best R^2",
        "-" * 50,
        f"  Threshold: {best['threshold']:.2f}",
        f"  R^2:       {best['r2']:.4f}",
        f"  RMSE:      {best['rmse']:.4f}",
        f"  Sim means: {best['sim_means']}",
        "",
        "Lowest RMSE",
        "-" * 50,
        f"  Threshold: {lowest_rmse['threshold']:.2f}",
        f"  RMSE:      {lowest_rmse['rmse']:.4f}",
        f"  R^2:       {lowest_rmse['r2']:.4f}",
        "",
        "Recommendation",
        "-" * 50,
    ]

    if best['r2'] > 0:
        lines.append(f"  Set CASCADE_RUPTURE_THRESHOLD = {best['threshold']:.2f} "
                      f"(R^2 = {best['r2']:.4f}, positive).")
    else:
        lines.append(f"  Best R^2 is {best['r2']:.4f} (negative). Consider expanding "
                      "the sweep range or investigating other parameters.")

    lines += [
        "",
        "Note: cascade event counts are not available from validate_cone_2020() "
        "return values. A future enhancement could instrument the simulation "
        "loop to capture per-run cascade statistics.",
    ]

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Report saved: {filepath}")


# Output: plot

def plot_sweep(results: list, filepath: str):
    """Two-panel figure: R^2 and RMSE vs threshold."""
    thresholds = [r['threshold'] for r in results]
    r2_vals = [r['r2'] for r in results]
    rmse_vals = [r['rmse'] for r in results]

    best_r2_idx = int(np.argmax(r2_vals))
    best_rmse_idx = int(np.argmin(rmse_vals))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: R^2 vs threshold
    ax1.plot(thresholds, r2_vals, 'o-', color='#2c3e50', lw=1.8, markersize=7)
    ax1.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.6)
    ax1.axvline(ORIGINAL_THRESHOLD, color='#e74c3c', ls=':', lw=1.0,
                alpha=0.6, label=f'Original ({ORIGINAL_THRESHOLD})')
    ax1.plot(thresholds[best_r2_idx], r2_vals[best_r2_idx], '*',
             color='#e67e22', markersize=18, zorder=5,
             label=f'Best R^2={r2_vals[best_r2_idx]:.4f} at {thresholds[best_r2_idx]:.2f}')
    ax1.set_xlabel('CASCADE_RUPTURE_THRESHOLD')
    ax1.set_ylabel('R^2 (Cone 2020)')
    ax1.set_title('R^2 vs Cascade Threshold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: RMSE vs threshold
    ax2.plot(thresholds, rmse_vals, 's-', color='#8e44ad', lw=1.8, markersize=7)
    ax2.axvline(ORIGINAL_THRESHOLD, color='#e74c3c', ls=':', lw=1.0,
                alpha=0.6, label=f'Original ({ORIGINAL_THRESHOLD})')
    ax2.plot(thresholds[best_rmse_idx], rmse_vals[best_rmse_idx], '*',
             color='#e67e22', markersize=18, zorder=5,
             label=f'Best RMSE={rmse_vals[best_rmse_idx]:.4f} at {thresholds[best_rmse_idx]:.2f}')
    ax2.set_xlabel('CASCADE_RUPTURE_THRESHOLD')
    ax2.set_ylabel('RMSE (Cone 2020)')
    ax2.set_title('RMSE vs Cascade Threshold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Cascade Threshold Sweep — Cone 2020 Validation', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {filepath}")


# Main

def main():
    smoke_mode = '--smoke' in sys.argv

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    t_start = wt.time()

    # Always run smoke test first
    smoke_test()

    if smoke_mode:
        print("Smoke mode — skipping full sweep.")
        return

    # Full sweep
    results = full_sweep(THRESHOLDS_FULL)

    # Save outputs
    csv_path = os.path.join(OUTPUT_DIR, 'sweep_results.csv')
    report_path = os.path.join(OUTPUT_DIR, 'sweep_report.txt')
    fig_path = os.path.join(FIG_DIR, 'cascade_threshold_sweep.png')

    save_csv(results, csv_path)
    save_report(results, report_path)
    plot_sweep(results, fig_path)

    # Summary
    best = max(results, key=lambda r: r['r2'])
    wall_total = wt.time() - t_start

    print("\n" + "=" * 60)
    print("  SWEEP COMPLETE")
    print("=" * 60)
    print(f"  Best threshold: {best['threshold']:.2f}")
    print(f"  Best R^2:       {best['r2']:.4f}")
    print(f"  Best RMSE:      {best['rmse']:.4f}")
    print(f"  Total wall time: {wall_total / 60:.1f} min")

    # Verify reset
    assert PC.CASCADE_RUPTURE_THRESHOLD == ORIGINAL_THRESHOLD, (
        f"Threshold not reset! Got {PC.CASCADE_RUPTURE_THRESHOLD}, "
        f"expected {ORIGINAL_THRESHOLD}"
    )
    print(f"  CASCADE_RUPTURE_THRESHOLD reset to {PC.CASCADE_RUPTURE_THRESHOLD} (verified)")


if __name__ == '__main__':
    main()
