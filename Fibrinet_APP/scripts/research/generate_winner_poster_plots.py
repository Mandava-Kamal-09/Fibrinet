"""
Winner Poster - Three Critical Plots for Dynamics Days 2026
============================================================
Generates the three essential plots that tell the "Shield vs. Sword" story:

1. The Golden Curve: T100 vs Applied Strain (The Paradox)
2. The Shield: Cleavage Rate vs Fiber Strain (Local Physics)
3. The Sword: Connectivity vs Time (Topological Failure)

Usage:
    python generate_winner_poster_plots.py
"""

import sys
import os
import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter

# Output directory
OUTPUT_DIR = os.path.join(_project_root, "publication_figures")
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Chemistry parameters
K_CAT_0 = 0.1  # Baseline cleavage rate (s^-1)
BETA = 10.0    # Strain sensitivity parameter

# Input files
EXCEL_FILE = os.path.join(_project_root, "test", "input_data", "fibrin_network_big.xlsx")
RESULTS_CSV = "fibrin_network_big_strain_sweep_results.csv"


# ============================================================================
# PLOT 1: THE GOLDEN CURVE (T100 vs Applied Strain)
# ============================================================================

def generate_plot1_golden_curve():
    """
    Generate the centerpiece plot showing U-shaped mechanochemical transition.
    """
    print("[PLOT 1] Generating Golden Curve: T100 vs Applied Strain...")

    # Read existing results
    if not os.path.exists(RESULTS_CSV):
        print(f"  ERROR: {RESULTS_CSV} not found. Run generate_mechanochemical_coupling_figures.py first.")
        return

    results = []
    with open(RESULTS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'applied_strain': float(row['applied_strain']),
                't_100': float(row['t_100']),
                'failure_mode': row['failure_mode']
            })

    applied_strains = [r['applied_strain'] for r in results]
    t100_values = [r['t_100'] for r in results]
    failure_modes = [r['failure_mode'] for r in results]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Separate by failure mode for coloring
    enzymatic_strains = [s for s, mode in zip(applied_strains, failure_modes) if mode == "ENZYMATIC"]
    enzymatic_t100 = [t for t, mode in zip(t100_values, failure_modes) if mode == "ENZYMATIC"]

    mechanical_strains = [s for s, mode in zip(applied_strains, failure_modes) if mode == "MECHANICAL"]
    mechanical_t100 = [t for t, mode in zip(t100_values, failure_modes) if mode == "MECHANICAL"]

    # Plot the main curve with large markers
    if enzymatic_strains:
        ax.plot(enzymatic_strains, enzymatic_t100, 'o-',
                color='#2E86DE', markersize=14, linewidth=3,
                markeredgecolor='black', markeredgewidth=2,
                label='Enzymatic Clearance', zorder=5, alpha=0.9)

    if mechanical_strains:
        ax.scatter(mechanical_strains, mechanical_t100, s=250, c='#EE5A6F',
                  edgecolors='black', linewidths=2, marker='s',
                  label='Mechanical Rupture', zorder=6, alpha=0.9)

    # Find minimum (optimal strain)
    min_idx = t100_values.index(min(t100_values))
    optimal_strain = applied_strains[min_idx]
    min_t100 = t100_values[min_idx]

    # Add transition zone shading
    transition_start = 0.15
    transition_end = 0.25
    ax.axvspan(transition_start, transition_end, alpha=0.15, color='orange',
               label='Transition Zone', zorder=1)

    # Annotate optimal point
    ax.annotate(f'Optimal: e = {optimal_strain:.2f}\nT100 = {min_t100:.1f}s',
                xy=(optimal_strain, min_t100),
                xytext=(optimal_strain - 0.08, min_t100 - 1.2),
                fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF176',
                         edgecolor='black', linewidth=2),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='black',
                               connectionstyle='arc3,rad=0.3'))

    # Annotate regions
    ax.text(0.03, max(t100_values) * 0.95, 'SHIELD\nDominates',
            fontsize=12, fontweight='bold', color='#2E86DE',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='#2E86DE', linewidth=2, alpha=0.8))

    if max(applied_strains) > 0.25:
        ax.text(max(applied_strains) - 0.05, max(t100_values) * 0.95, 'Protection\nRegime',
                fontsize=12, fontweight='bold', color='#27AE60',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor='#27AE60', linewidth=2, alpha=0.8),
                ha='right')

    # Formatting
    ax.set_xlabel('Applied Strain (e)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Time to Network Clearance (s)', fontsize=16, fontweight='bold')
    ax.set_title('The Golden Curve: Mechanochemical Transition',
                 fontsize=18, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.legend(fontsize=12, loc='upper left', framealpha=0.95,
             edgecolor='black', shadow=True)

    ax.tick_params(labelsize=13)
    ax.set_xlim(-0.02, max(applied_strains) + 0.02)

    # Styling
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')

    plt.tight_layout()

    # Save
    output_path = os.path.join(OUTPUT_DIR, "plot1_golden_curve.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  [OK] Saved: {output_path}")
    plt.close()


# ============================================================================
# PLOT 2: THE SHIELD (Cleavage Rate vs Fiber Strain)
# ============================================================================

def generate_plot2_shield():
    """
    Generate chemistry validation plot showing exponential strain protection.
    """
    print("[PLOT 2] Generating Shield: Cleavage Rate vs Fiber Strain...")

    # Generate strain range
    strain_range = np.linspace(0, 1.0, 200)

    # Compute cleavage rates
    k_absolute = K_CAT_0 * np.exp(-BETA * strain_range)
    k_normalized = np.exp(-BETA * strain_range)

    # Create figure with dual y-axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # LEFT PLOT: Linear scale
    ax1.plot(strain_range, k_absolute, linewidth=4, color='#E91E63',
            label=f'k(e) = k0 exp(-beta*e)')
    ax1.axhline(y=K_CAT_0, color='gray', linestyle='--', linewidth=2,
               alpha=0.5, label=f'k0 = {K_CAT_0} /s')

    # Shade protection region
    ax1.fill_between(strain_range, 0, k_absolute, alpha=0.2, color='#E91E63')

    # Annotate 10x reduction point
    strain_10x = np.log(10) / BETA
    k_10x = K_CAT_0 / 10
    ax1.plot([strain_10x], [k_10x], 'o', markersize=12, color='black', zorder=5)
    ax1.annotate(f'10x Protection\ne = {strain_10x:.2f}',
                xy=(strain_10x, k_10x),
                xytext=(strain_10x + 0.15, k_10x + 0.02),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFE082',
                         edgecolor='black', linewidth=2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax1.set_xlabel('Fiber Strain (e)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cleavage Rate k(e) [/s]', fontsize=14, fontweight='bold')
    ax1.set_title('Linear Scale: Exponential Protection', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11, loc='upper right', framealpha=0.95, edgecolor='black')
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, K_CAT_0 * 1.1)
    ax1.tick_params(labelsize=11)

    # RIGHT PLOT: Log scale
    ax2.semilogy(strain_range, k_absolute, linewidth=4, color='#9C27B0')
    ax2.axhline(y=K_CAT_0, color='gray', linestyle='--', linewidth=2, alpha=0.5)

    # Annotate orders of magnitude
    ax2.text(0.5, K_CAT_0 / 100, '2 Orders\nof Magnitude',
            fontsize=11, fontweight='bold', color='#9C27B0',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='#9C27B0', linewidth=2, alpha=0.9))

    ax2.set_xlabel('Fiber Strain (e)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Cleavage Rate k(e) [/s] (log scale)', fontsize=14, fontweight='bold')
    ax2.set_title('Log Scale: Orders of Magnitude Suppression', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', which='both')
    ax2.set_xlim(0, 1.0)
    ax2.tick_params(labelsize=11)

    # Add parameter box
    textstr = f'Bell Model Parameters:\nk0 = {K_CAT_0} /s\nbeta = {BETA}'
    props = dict(boxstyle='round', facecolor='#E3F2FD', edgecolor='black', linewidth=2, alpha=0.95)
    ax2.text(0.97, 0.97, textstr, transform=ax2.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=props, fontfamily='monospace', fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = os.path.join(OUTPUT_DIR, "plot2_shield.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  [OK] Saved: {output_path}")
    plt.close()


# ============================================================================
# PLOT 3: THE SWORD (Connectivity vs Time)
# ============================================================================

def generate_plot3_sword():
    """
    Generate percolation connectivity plot showing gradual vs catastrophic failure.
    """
    print("[PLOT 3] Generating Sword: Connectivity vs Time...")
    print("  Running simulations for low strain (e=0.05) and high strain (e=0.25)...")

    # Run two simulations: low strain and high strain
    configs = [
        {'strain': 0.05, 'label': 'Low Strain (e=0.05): Gradual Enzymatic',
         'color': '#27AE60', 'style': '-'},
        {'strain': 0.25, 'label': 'High Strain (e=0.25): Protected -> Rupture',
         'color': '#E74C3C', 'style': '--'}
    ]

    fig, ax = plt.subplots(figsize=(10, 7))

    for config in configs:
        print(f"    Running e = {config['strain']:.2f}...")

        # Setup simulation
        adapter = CoreV2GUIAdapter()
        adapter.load_from_excel(EXCEL_FILE)
        adapter.configure_parameters(
            plasmin_concentration=10.0,
            time_step=0.05,
            max_time=20.0,
            applied_strain=config['strain']
        )
        adapter.start_simulation()

        # Track connectivity over time
        times = [0.0]
        connectivity = [1.0]  # Start at 100% connected

        step = 0
        while adapter.advance_one_batch():
            step += 1

            # Check connectivity every few steps
            if step % 10 == 0:
                current_time = adapter.get_current_time()

                # Check if network still percolates
                from src.core.fibrinet_core_v2 import check_left_right_connectivity
                is_connected = check_left_right_connectivity(adapter.simulation.state)

                times.append(current_time)
                connectivity.append(1.0 if is_connected else 0.0)

                # Stop after clearance
                if not is_connected:
                    break

        # Plot connectivity trace
        ax.plot(times, connectivity, config['style'], linewidth=4,
               color=config['color'], label=config['label'],
               marker='o', markersize=6, markevery=max(1, len(times)//15))

    # Formatting
    ax.set_xlabel('Time (s)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Network Connectivity', fontsize=16, fontweight='bold')
    ax.set_title('The Sword: Topological Failure Modes',
                 fontsize=18, fontweight='bold', pad=20)

    ax.set_ylim(-0.1, 1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Disconnected', 'Connected'], fontsize=13)

    # Add shaded regions
    ax.axhspan(0.9, 1.1, alpha=0.15, color='green', zorder=1)
    ax.axhspan(-0.1, 0.1, alpha=0.15, color='red', zorder=1)

    ax.text(0.5, 1.05, 'Percolating Network', fontsize=11,
           color='darkgreen', fontweight='bold', ha='center')
    ax.text(0.5, 0.05, 'Cleared (Percolation Lost)', fontsize=11,
           color='darkred', fontweight='bold', ha='center')

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.legend(fontsize=12, loc='center left', framealpha=0.95,
             edgecolor='black', shadow=True)

    ax.tick_params(labelsize=13)
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')

    plt.tight_layout()

    # Save
    output_path = os.path.join(OUTPUT_DIR, "plot3_sword.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  [OK] Saved: {output_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all three winner poster plots."""
    print()
    print("=" * 80)
    print(" " * 20 + "WINNER POSTER PLOT GENERATOR")
    print("=" * 80)
    print()
    print("Generating three critical plots for Dynamics Days 2026:")
    print("  1. The Golden Curve - T100 vs Applied Strain")
    print("  2. The Shield - Cleavage Rate vs Fiber Strain")
    print("  3. The Sword - Connectivity vs Time")
    print()
    print("=" * 80)
    print()

    # Generate plots
    generate_plot1_golden_curve()
    print()

    generate_plot2_shield()
    print()

    generate_plot3_sword()
    print()

    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print()
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("Generated files:")
    print("  1. plot1_golden_curve.png - The centerpiece U-shaped paradox")
    print("  2. plot2_shield.png - Exponential strain protection (dual scale)")
    print("  3. plot3_sword.png - Gradual vs catastrophic failure modes")
    print()
    print("All plots are publication-ready (300 DPI) for your poster!")
    print()
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
