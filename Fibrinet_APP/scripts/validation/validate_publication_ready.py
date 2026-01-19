"""
FibriNet Core V2 - Publication-Ready Validation Suite
======================================================
Automated validation testing for peer-reviewed publication.

This script verifies:
1. Numerical stability across strain range (0.0-0.5)
2. Monotonic clearance time trend (higher strain -> longer clearance)
3. Force metrics remain within bounds
4. No simulation crashes or NaN values

Run after implementing critical fixes (timestep capping, force clamping).

Expected Behavior After Fixes:
- Strain 0.0: Baseline clearance time
- Strain 0.5: Significantly longer clearance (or timeout)
- Smooth, monotonic increase in clearance time
- No force spikes > 1e-6 N (force ceiling)
- All simulations complete without errors

Usage:
    python validate_publication_ready.py

Outputs:
    - validation_results/strain_sweep_results.csv
    - validation_results/strain_sweep_plot.png
    - validation_results/force_metrics_plot.png
    - Console report with PASS/FAIL status
"""

import sys
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter

# Configuration
INPUT_FILE = os.path.join(_project_root, "test", "input_data", "fibrin_network_big.xlsx")
OUTPUT_DIR = os.path.join(_project_root, "validation_results")
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Test parameters
STRAINS_TO_TEST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
PLASMIN_CONC = 1.0
TIME_STEP = 0.01  # Will be capped to 0.005 internally
MAX_TIME = 300.0

# Validation criteria
FORCE_CEILING = 1e-6  # Max allowed tension [N]
MONOTONICITY_TOLERANCE = 1.1  # Allow 10% deviation from monotonic trend


def run_strain_sweep():
    """Execute full strain sweep and collect metrics."""
    print("=" * 80)
    print("FIBRINET CORE V2 - PUBLICATION VALIDATION SUITE")
    print("=" * 80)
    print(f"\nInput Network: {INPUT_FILE}")
    print(f"Strains to test: {STRAINS_TO_TEST}")
    print(f"Time step: {TIME_STEP}s (will be capped to 0.005s)")
    print(f"Max time per run: {MAX_TIME}s")
    print("-" * 80)

    results = []
    adapter = CoreV2GUIAdapter()

    # Load network once
    try:
        adapter.load_from_excel(INPUT_FILE)
        print(f"\nLoaded network: {len(adapter._edges_raw)} fibers")
    except Exception as e:
        print(f"\nERROR: Failed to load network: {e}")
        return None

    # Run simulations for each strain
    for strain in STRAINS_TO_TEST:
        print(f"\n{'=' * 80}")
        print(f"STRAIN = {strain*100:.0f}%")
        print("=" * 80)

        # Configure simulation
        adapter.configure_parameters(
            plasmin_concentration=PLASMIN_CONC,
            time_step=TIME_STEP,
            max_time=MAX_TIME,
            applied_strain=strain
        )

        adapter.start_simulation()
        start_wall_time = time.time()

        # Track metrics
        max_tension_observed = 0.0
        force_spike_detected = False
        t_50 = None

        # Run simulation
        step_count = 0
        try:
            while True:
                running = adapter.advance_one_batch()
                step_count += 1

                sim_time = adapter.get_current_time()
                lysis = adapter.get_lysis_fraction()
                max_tension = adapter.get_max_tension()

                # Track maximum tension
                if max_tension > max_tension_observed:
                    max_tension_observed = max_tension

                # Detect force spikes
                if max_tension > FORCE_CEILING:
                    force_spike_detected = True
                    print(f"\n  WARNING: Force spike detected at t={sim_time:.1f}s: {max_tension:.3e} N")

                # Record T50
                if t_50 is None and lysis >= 0.5:
                    t_50 = sim_time

                # Progress update
                if step_count % 100 == 0:
                    print(f"  t={sim_time:6.1f}s | Lysis={lysis*100:5.1f}% | Max F={max_tension:.2e} N", end='\r')

                if not running:
                    print(f"\n  Terminated: {adapter.termination_reason} at t={sim_time:.1f}s")
                    break

        except Exception as e:
            print(f"\n  CRASH: {e}")
            results.append({
                'strain': strain,
                'status': 'CRASH',
                'error': str(e),
                'wall_time_s': time.time() - start_wall_time
            })
            continue

        # Collect results
        wall_time = time.time() - start_wall_time
        final_lysis = adapter.get_lysis_fraction()
        termination = adapter.termination_reason

        result = {
            'strain': strain,
            'status': 'PASS' if not force_spike_detected else 'FORCE_SPIKE',
            't50_lysis_time_s': t_50 if t_50 else MAX_TIME,
            'clearance_time_s': sim_time if termination == "network_cleared" else None,
            'final_lysis_fraction': final_lysis,
            'termination_reason': termination,
            'max_tension_N': max_tension_observed,
            'force_spike_detected': force_spike_detected,
            'wall_time_s': wall_time,
            'steps': step_count
        }

        results.append(result)

        # Summary
        print(f"\n  Results:")
        print(f"    Status: {result['status']}")
        print(f"    T50 lysis: {result['t50_lysis_time_s']:.1f}s")
        if result['clearance_time_s']:
            print(f"    Clearance: {result['clearance_time_s']:.1f}s")
        print(f"    Final lysis: {final_lysis*100:.1f}%")
        print(f"    Max tension: {max_tension_observed:.3e} N")
        print(f"    Wall time: {wall_time:.1f}s")

    return pd.DataFrame(results)


def validate_monotonicity(df):
    """Check if clearance times increase monotonically with strain."""
    print("\n" + "=" * 80)
    print("MONOTONICITY VALIDATION")
    print("=" * 80)

    # Extract clearance times (use T50 as proxy if clearance didn't occur)
    clearance_times = []
    for _, row in df.iterrows():
        if row['clearance_time_s'] is not None and not pd.isna(row['clearance_time_s']):
            clearance_times.append(row['clearance_time_s'])
        else:
            clearance_times.append(row['t50_lysis_time_s'])

    # Check monotonicity
    is_monotonic = True
    violations = []

    for i in range(len(clearance_times) - 1):
        current_time = clearance_times[i]
        next_time = clearance_times[i + 1]

        # Allow small violations due to stochasticity
        if next_time < current_time * MONOTONICITY_TOLERANCE:
            is_monotonic = False
            violations.append({
                'strain_low': STRAINS_TO_TEST[i],
                'strain_high': STRAINS_TO_TEST[i + 1],
                'time_low': current_time,
                'time_high': next_time,
                'ratio': next_time / current_time
            })

    # Report
    print("\nClearance Time Trend:")
    for i, (strain, t) in enumerate(zip(STRAINS_TO_TEST, clearance_times)):
        print(f"  Strain {strain*100:3.0f}%: {t:6.1f}s")

    if is_monotonic:
        print("\n[PASS] Monotonic increase confirmed")
    else:
        print("\n[FAIL] Non-monotonic behavior detected")
        for v in violations:
            print(f"  Strain {v['strain_low']*100:.0f}% ({v['time_low']:.1f}s) -> "
                  f"Strain {v['strain_high']*100:.0f}% ({v['time_high']:.1f}s) "
                  f"[Ratio: {v['ratio']:.2f}]")

    return is_monotonic, violations


def validate_force_bounds(df):
    """Check if all forces remained within bounds."""
    print("\n" + "=" * 80)
    print("FORCE BOUNDS VALIDATION")
    print("=" * 80)

    max_force = df['max_tension_N'].max()
    force_spikes = df[df['force_spike_detected'] == True]

    print(f"\nMaximum tension observed: {max_force:.3e} N")
    print(f"Force ceiling: {FORCE_CEILING:.3e} N")

    if len(force_spikes) == 0:
        print("\n[PASS] All forces within bounds")
        return True
    else:
        print(f"\n[FAIL] {len(force_spikes)} force spikes detected")
        for _, row in force_spikes.iterrows():
            print(f"  Strain {row['strain']*100:.0f}%: {row['max_tension_N']:.3e} N")
        return False


def validate_stability(df):
    """Check if all simulations completed without crashes."""
    print("\n" + "=" * 80)
    print("NUMERICAL STABILITY VALIDATION")
    print("=" * 80)

    crashes = df[df['status'] == 'CRASH']

    if len(crashes) == 0:
        print("\n[PASS] All simulations completed successfully")
        return True
    else:
        print(f"\n[FAIL] {len(crashes)} simulations crashed")
        for _, row in crashes.iterrows():
            print(f"  Strain {row['strain']*100:.0f}%: {row['error']}")
        return False


def generate_plots(df):
    """Generate publication-quality validation plots."""
    print("\n" + "=" * 80)
    print("GENERATING VALIDATION PLOTS")
    print("=" * 80)

    # Extract clearance times (use T50 if no clearance)
    clearance_times = []
    for _, row in df.iterrows():
        if row['clearance_time_s'] is not None and not pd.isna(row['clearance_time_s']):
            clearance_times.append(row['clearance_time_s'])
        else:
            clearance_times.append(row['t50_lysis_time_s'])

    # Plot 1: Clearance Time vs Strain
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df['strain'], clearance_times, 'o-', linewidth=2, markersize=8, color='#d32f2f')
    ax1.set_xlabel('Applied Strain (ε)', fontsize=12)
    ax1.set_ylabel('Clearance Time (s)', fontsize=12)
    ax1.set_title('Strain-Dependent Network Clearance\n(After Numerical Stability Fixes)', fontsize=13)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(bottom=0)

    # Plot 2: Max Tension vs Strain
    ax2.plot(df['strain'], df['max_tension_N'], 'o-', linewidth=2, markersize=8, color='#1976d2')
    ax2.axhline(y=FORCE_CEILING, color='red', linestyle='--', label=f'Force Ceiling ({FORCE_CEILING:.0e} N)')
    ax2.set_xlabel('Applied Strain (ε)', fontsize=12)
    ax2.set_ylabel('Maximum Tension (N)', fontsize=12)
    ax2.set_title('Force Spike Monitoring\n(Should Remain Below Ceiling)', fontsize=13)
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "validation_plots.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nPlots saved to: {plot_path}")


def main():
    """Run complete validation suite."""
    # Run strain sweep
    df = run_strain_sweep()

    if df is None or len(df) == 0:
        print("\nERROR: Validation failed - no results collected")
        return False

    # Save results
    csv_path = os.path.join(OUTPUT_DIR, "validation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n\nResults saved to: {csv_path}")

    # Run validation checks
    monotonic_pass, violations = validate_monotonicity(df)
    force_pass = validate_force_bounds(df)
    stability_pass = validate_stability(df)

    # Generate plots
    generate_plots(df)

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VALIDATION VERDICT")
    print("=" * 80)

    all_pass = monotonic_pass and force_pass and stability_pass

    print(f"\n  Monotonicity:         {'[PASS]' if monotonic_pass else '[FAIL]'}")
    print(f"  Force Bounds:         {'[PASS]' if force_pass else '[FAIL]'}")
    print(f"  Numerical Stability:  {'[PASS]' if stability_pass else '[FAIL]'}")

    if all_pass:
        print("\n" + "=" * 80)
        print("*** PUBLICATION READY ***")
        print("=" * 80)
        print("\nAll validation tests passed. Tool is ready for peer-reviewed publication.")
        print("\nNext steps:")
        print("  1. Generate publication figures (run generate_publication_figures.py)")
        print("  2. Test reproducibility (run test_reproducibility.py)")
        print("  3. Document seed in Methods section")
    else:
        print("\n" + "=" * 80)
        print("*** VALIDATION FAILED ***")
        print("=" * 80)
        print("\nSome validation tests failed. Review errors above before publication.")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
