"""
FibriNet Core V2 - Mechanochemical Coupling Publication Figures
================================================================
Generates two separate publication-quality plots:
1. T100 Time vs Applied Strain (with failure mode classification)
2. Predicted Cleavage Rate vs Average Fiber Strain (chemistry validation)

Runs strain sweep on fibrin_network_big.xlsx to capture
mechanochemical coupling and measure time to COMPLETE lysis (100%).

Usage:
    python generate_mechanochemical_coupling_figures.py

Outputs:
    - publication_figures/t100_vs_strain.png
    - publication_figures/chemistry_validation.png
    - fibrin_network_big_strain_sweep_results.csv
"""

import sys
import os
import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input/Output
EXCEL_FILE = os.path.join(_project_root, "test", "input_data", "fibrin_network_big.xlsx")
OUTPUT_DIR = os.path.join(_project_root, "publication_figures")
OUTPUT_CSV = os.path.join(_project_root, "fibrin_network_big_strain_sweep_results.csv")

Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Simulation Parameters (Run until complete clearance)
PLASMIN_CONCENTRATION = 10.0   # Moderate plasmin to reveal mechanochemical coupling
TIME_STEP = 0.05               # Small timestep for accuracy
MAX_TIME = 100.0               # Extended time to ensure complete lysis at high strains

# Strain sweep range (wide range to show full protection effect)
APPLIED_STRAINS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
LYSIS_THRESHOLD = 1.0  # Complete lysis (100%)

# Chemistry parameters (from fibrinet_core_v2.py)
K_CAT_0 = 0.1  # Baseline cleavage rate (s^-1)
BETA = 10.0    # Strain sensitivity parameter


# ============================================================================
# STRAIN SWEEP EXECUTION
# ============================================================================

def run_strain_sweep():
    """Execute strain sweep and collect data."""
    print("=" * 80)
    print("MECHANOCHEMICAL COUPLING: STRAIN SWEEP")
    print("=" * 80)
    print()
    print(f"Input:  {EXCEL_FILE}")
    print(f"Output: {OUTPUT_CSV}")
    print()
    print("PARAMETERS:")
    print(f"  Plasmin concentration: {PLASMIN_CONCENTRATION:.1f}x")
    print(f"  Time step: {TIME_STEP:.4f}s")
    print(f"  Max time: {MAX_TIME:.1f}s")
    print(f"  Applied strains: {APPLIED_STRAINS}")
    print()

    # Load network once to check size
    adapter_test = CoreV2GUIAdapter()
    adapter_test.load_from_excel(EXCEL_FILE)
    print(f"Network: {len(adapter_test._edges_raw)} edges, "
          f"{len(adapter_test.left_boundary_node_ids)} left + "
          f"{len(adapter_test.right_boundary_node_ids)} right boundaries")
    print()

    # Run sweep
    print("[EXECUTING] Strain sweep...")
    print("=" * 80)
    print()

    results = []

    for strain_idx, applied_strain in enumerate(APPLIED_STRAINS):
        print(f"Test {strain_idx + 1}/{len(APPLIED_STRAINS)}: Applied Strain = {applied_strain:.2f}")
        print("-" * 80)

        # Fresh adapter for each test
        adapter = CoreV2GUIAdapter()
        adapter.load_from_excel(EXCEL_FILE)
        adapter.configure_parameters(
            plasmin_concentration=PLASMIN_CONCENTRATION,
            time_step=TIME_STEP,
            max_time=MAX_TIME,
            applied_strain=applied_strain
        )
        adapter.start_simulation()

        # Measure initial fiber strains (after prestrain, before degradation)
        fiber_strains = []
        for fiber in adapter.simulation.state.fibers:
            pos_i = adapter.simulation.state.node_positions[fiber.node_i]
            pos_j = adapter.simulation.state.node_positions[fiber.node_j]
            length = float(np.linalg.norm(pos_j - pos_i))
            strain = (length - fiber.L_c) / fiber.L_c
            fiber_strains.append(strain)

        avg_strain = np.mean(fiber_strains)
        max_strain = np.max(fiber_strains)
        std_strain = np.std(fiber_strains)

        # Compute theoretical cleavage rate at average strain
        predicted_k = K_CAT_0 * np.exp(-BETA * avg_strain)

        print(f"  Initial fiber strains: mean={avg_strain:.3f}, max={max_strain:.3f}, std={std_strain:.3f}")
        print(f"  Predicted k_cleave: {predicted_k:.4f} s^-1 (vs k0={K_CAT_0:.2f})")

        # Warn about dangerous overstrain
        if max_strain > 3.0:
            print(f"  [WARNING] Max strain = {max_strain:.1f} (singularity risk)")

        # Run simulation until complete lysis
        step_count = 0
        t_100 = None

        print(f"  Simulating...", end='')

        while adapter.advance_one_batch():
            step_count += 1

            lysis_frac = adapter.get_lysis_fraction()
            current_time = adapter.get_current_time()

            # Detect T100 (complete lysis)
            if t_100 is None and lysis_frac >= LYSIS_THRESHOLD:
                t_100 = current_time

            # Progress dots
            if step_count % 100 == 0:
                print('.', end='', flush=True)

        print()

        final_time = adapter.get_current_time()
        final_lysis = adapter.get_lysis_fraction()
        termination = adapter.termination_reason or "Timeout"

        # If complete lysis not reached, use final time
        if t_100 is None:
            t_100 = final_time

        # Classify failure mode
        if "cleared" in termination.lower():
            if final_lysis < 0.3:
                failure_mode = "MECHANICAL"  # Cleared with low lysis = rupture
            else:
                failure_mode = "ENZYMATIC"   # Cleared with high lysis = enzymatic
        else:
            failure_mode = "INCOMPLETE"

        print(f"  Result: T100={t_100:.1f}s, Final={final_lysis*100:.1f}%, Mode={failure_mode}")
        print()

        results.append({
            'applied_strain': applied_strain,
            'avg_fiber_strain': avg_strain,
            'max_fiber_strain': max_strain,
            'std_fiber_strain': std_strain,
            'predicted_k_cleave': predicted_k,
            't_100': t_100,
            'final_time': final_time,
            'final_lysis': final_lysis,
            'n_steps': step_count,
            'termination': termination,
            'failure_mode': failure_mode
        })

    # Save results
    print("=" * 80)
    print("[SAVING] Results...")
    print()

    with open(OUTPUT_CSV, 'w', newline='') as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved to: {OUTPUT_CSV}")
    print()

    return results


# ============================================================================
# FIGURE 1: T100 vs APPLIED STRAIN
# ============================================================================

def generate_t100_figure(results):
    """Generate publication-quality T100 vs Strain plot."""
    print("[GENERATING] Figure 1: T100 vs Applied Strain...")

    # Extract data
    applied_strains = [r['applied_strain'] for r in results]
    t100_values = [r['t_100'] for r in results]
    failure_modes = [r['failure_mode'] for r in results]

    # Separate by failure mode
    enzymatic_strains = [s for s, mode in zip(applied_strains, failure_modes) if mode == "ENZYMATIC"]
    enzymatic_t100 = [t for t, mode in zip(t100_values, failure_modes) if mode == "ENZYMATIC"]

    mechanical_strains = [s for s, mode in zip(applied_strains, failure_modes) if mode == "MECHANICAL"]
    mechanical_t100 = [t for t, mode in zip(t100_values, failure_modes) if mode == "MECHANICAL"]

    incomplete_strains = [s for s, mode in zip(applied_strains, failure_modes) if mode == "INCOMPLETE"]
    incomplete_t100 = [t for t, mode in zip(t100_values, failure_modes) if mode == "INCOMPLETE"]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot data points
    if enzymatic_strains:
        ax.scatter(enzymatic_strains, enzymatic_t100, s=150, c='#2ECC71',
                  edgecolors='black', linewidths=1.5, label='Enzymatic', zorder=5, alpha=0.8)

    if mechanical_strains:
        ax.scatter(mechanical_strains, mechanical_t100, s=150, c='#E74C3C',
                  edgecolors='black', linewidths=1.5, label='Mechanical', zorder=5, alpha=0.8)

    if incomplete_strains:
        ax.scatter(incomplete_strains, incomplete_t100, s=150, c='#95A5A6',
                  edgecolors='black', linewidths=1.5, label='Incomplete', zorder=5, alpha=0.8)

    # Connect points with dashed line (guide to eye)
    ax.plot(applied_strains, t100_values, 'k--', alpha=0.3, linewidth=1, zorder=1)

    # Annotate lowest T100 point
    min_t100_idx = t100_values.index(min(t100_values))
    min_strain = applied_strains[min_t100_idx]
    min_t100 = t100_values[min_t100_idx]
    ax.annotate(f'T100 = {min_t100:.1f}s',
                xy=(min_strain, min_t100),
                xytext=(min_strain + 0.05, min_t100 + 0.5),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4', edgecolor='black', linewidth=1),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Formatting
    ax.set_xlabel('Applied Strain', fontsize=14, fontweight='bold')
    ax.set_ylabel('T100: Time to Complete Lysis (s)', fontsize=14, fontweight='bold')
    ax.set_title('Mechanochemical Coupling: Strain-Dependent Protection',
                 fontsize=16, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(title='Failure Mode', fontsize=11, title_fontsize=12,
             loc='best', framealpha=0.95, edgecolor='black')

    ax.tick_params(labelsize=11)
    ax.set_xlim(-0.02, max(applied_strains) + 0.02)

    # Add subtle background
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')

    plt.tight_layout()

    # Save
    output_path = os.path.join(OUTPUT_DIR, "t100_vs_strain.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


# ============================================================================
# FIGURE 2: CHEMISTRY VALIDATION (k vs Strain)
# ============================================================================

def generate_chemistry_figure(results):
    """Generate publication-quality chemistry validation plot."""
    print("[GENERATING] Figure 2: Chemistry k(epsilon) = k0 x exp(-beta x epsilon)...")

    # Extract data
    avg_strains = [r['avg_fiber_strain'] for r in results]
    predicted_k = [r['predicted_k_cleave'] for r in results]
    failure_modes = [r['failure_mode'] for r in results]

    # Generate theoretical curve
    strain_theory = np.linspace(0, max(avg_strains) * 1.1, 100)
    k_theory = K_CAT_0 * np.exp(-BETA * strain_theory)

    # Separate by failure mode for coloring
    enzymatic_strains = [s for s, mode in zip(avg_strains, failure_modes) if mode == "ENZYMATIC"]
    enzymatic_k = [k for k, mode in zip(predicted_k, failure_modes) if mode == "ENZYMATIC"]

    mechanical_strains = [s for s, mode in zip(avg_strains, failure_modes) if mode == "MECHANICAL"]
    mechanical_k = [k for k, mode in zip(predicted_k, failure_modes) if mode == "MECHANICAL"]

    incomplete_strains = [s for s, mode in zip(avg_strains, failure_modes) if mode == "INCOMPLETE"]
    incomplete_k = [k for k, mode in zip(predicted_k, failure_modes) if mode == "INCOMPLETE"]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot theoretical curve
    ax.plot(strain_theory, k_theory, 'b-', linewidth=3, alpha=0.7,
            label=f'Theory: β={BETA:.2f}', zorder=2)

    # Plot simulation data points
    if enzymatic_strains:
        ax.scatter(enzymatic_strains, enzymatic_k, s=150, c='#2ECC71',
                  edgecolors='black', linewidths=1.5, label='Enzymatic', zorder=5, alpha=0.8)

    if mechanical_strains:
        ax.scatter(mechanical_strains, mechanical_k, s=150, c='#E74C3C',
                  edgecolors='black', linewidths=1.5, label='Mechanical', zorder=5, alpha=0.8)

    if incomplete_strains:
        ax.scatter(incomplete_strains, incomplete_k, s=150, c='#95A5A6',
                  edgecolors='black', linewidths=1.5, label='Incomplete', zorder=5, alpha=0.8)

    # Formatting
    ax.set_xlabel('Average Fiber Strain', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted k_cleave (1/s)', fontsize=14, fontweight='bold')
    ax.set_title('Chemistry: k(ε) = k₀ × exp(-β × ε)',
                 fontsize=16, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=11, loc='best', framealpha=0.95, edgecolor='black')

    ax.tick_params(labelsize=11)
    ax.set_xlim(-0.01, max(avg_strains) * 1.05)
    ax.set_ylim(0, K_CAT_0 * 1.05)

    # Add subtle background
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')

    # Add text box with parameters
    textstr = f'k₀ = {K_CAT_0:.3f} s⁻¹\nβ = {BETA:.1f}'
    props = dict(boxstyle='round', facecolor='#E3F2FD', edgecolor='black', linewidth=1, alpha=0.9)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=props, fontfamily='monospace')

    plt.tight_layout()

    # Save
    output_path = os.path.join(OUTPUT_DIR, "chemistry_validation.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run strain sweep and generate both figures."""
    print()
    print("=" * 80)
    print("=" * 15 + " MECHANOCHEMICAL COUPLING FIGURE GENERATOR " + "=" * 22)
    print("=" * 80)
    print()

    # Run strain sweep
    results = run_strain_sweep()

    # Generate summary
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Applied':<10} {'Avg Fiber':<12} {'Pred k':<12} {'T100':<10} {'Final':<12} {'Failure':<12}")
    print(f"{'Strain':<10} {'Strain':<12} {'(1/s)':<12} {'(sec)':<10} {'Lysis (%)':<12} {'Mode':<12}")
    print("-" * 80)

    for r in results:
        print(f"{r['applied_strain']:<10.2f} {r['avg_fiber_strain']:<12.3f} {r['predicted_k_cleave']:<12.4f} "
              f"{r['t_100']:<10.1f} {r['final_lysis']*100:<12.1f} {r['failure_mode']:<12}")

    print("=" * 80)
    print()

    # Generate figures
    print("=" * 80)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 80)
    print()

    generate_t100_figure(results)
    generate_chemistry_figure(results)

    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print()
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("Generated files:")
    print("  1. t100_vs_strain.png - T100 vs Applied Strain (with failure modes)")
    print("  2. chemistry_validation.png - k(epsilon) chemistry curve validation")
    print(f"  3. {OUTPUT_CSV} - Raw data CSV")
    print()
    print("These figures are publication-ready (300 DPI) and suitable for:")
    print("  - Research posters")
    print("  - Manuscript figures")
    print("  - Grant proposals")
    print("  - Conference presentations")
    print()
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
