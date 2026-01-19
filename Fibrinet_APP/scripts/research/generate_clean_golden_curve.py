"""
Clean Golden Curve Plot - NO ANNOTATIONS
=========================================
Generates a clean T100 vs Applied Strain plot with ONLY:
- X-axis and label
- Y-axis and label
- Data points with failure mode colors
- Connecting line
- Legend

NO text annotations, NO boxes, NO transition zones
"""

import os
import sys
import csv
import matplotlib.pyplot as plt

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

# Paths
OUTPUT_DIR = os.path.join(_project_root, "publication_figures")
RESULTS_CSV = os.path.join(_project_root, "fibrin_network_big_strain_sweep_results.csv")

def generate_clean_golden_curve():
    """Generate clean T100 vs Strain plot - axes and labels ONLY."""
    print("[GENERATING] Clean Golden Curve: T100 vs Applied Strain...")

    # Read results
    if not os.path.exists(RESULTS_CSV):
        print(f"  ERROR: {RESULTS_CSV} not found.")
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

    # Plot with large, clear markers
    if enzymatic_strains:
        ax.plot(enzymatic_strains, enzymatic_t100, 'o-',
                color='#2ECC71', markersize=16, linewidth=3.5,
                markeredgecolor='black', markeredgewidth=2,
                label='Enzymatic', zorder=5)

    if mechanical_strains:
        ax.scatter(mechanical_strains, mechanical_t100, s=300, c='#E74C3C',
                  edgecolors='black', linewidths=2.5, marker='s',
                  label='Mechanical', zorder=6)

    # CLEAN formatting - axes and labels ONLY
    ax.set_xlabel('Applied Strain', fontsize=18, fontweight='bold')
    ax.set_ylabel('Time to Network Clearance (s)', fontsize=18, fontweight='bold')

    # Grid for readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)

    # Legend
    ax.legend(title='Failure Mode', fontsize=14, title_fontsize=15,
             loc='upper left', framealpha=0.98, edgecolor='black', shadow=True)

    ax.tick_params(labelsize=15)
    ax.set_xlim(-0.05, max(applied_strains) + 0.05)
    ax.set_ylim(min(t100_values) - 1, max(t100_values) + 1)

    # Clean background
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')

    plt.tight_layout()

    # Save
    output_path = os.path.join(OUTPUT_DIR, "golden_curve_clean.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  [OK] Saved: {output_path}")
    plt.close()

    print()
    print(f"Clean golden curve saved to: {output_path}")
    print("Plot contains ONLY:")
    print("  - X-axis: Applied Strain")
    print("  - Y-axis: Time to Network Clearance (s)")
    print("  - Data points (enzymatic=green, mechanical=red)")
    print("  - Legend")
    print("  - NO annotations, NO text boxes, NO transition zones")


if __name__ == "__main__":
    generate_clean_golden_curve()
