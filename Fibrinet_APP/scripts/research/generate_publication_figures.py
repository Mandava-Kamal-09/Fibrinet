"""
FibriNet Core V2 - Publication Figure Generator
================================================
Generates high-quality network visualization figures for publication.

Creates multi-panel figure showing:
- Panel A: Initial state (t=0, all fibers blue from prestrain)
- Panel B: Mid-degradation (mixed strain colors, some ruptures)
- Panel C: Near clearance (high strain, many ruptures)
- Panel D: Critical fiber highlighted (magenta)

Features:
- Strain-based color gradient (blue -> yellow -> orange -> red)
- Color legend with strain scale
- High resolution (300 DPI)
- Professional formatting
- Scale bar (10 µm)

Usage:
    python generate_publication_figures.py

Outputs:
    - publication_figures/network_evolution_strain_0p3.png
    - publication_figures/critical_fiber_detail.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter

# Configuration
INPUT_FILE = os.path.join(_project_root, "test", "input_data", "fibrin_network_big.xlsx")
OUTPUT_DIR = os.path.join(_project_root, "publication_figures")
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Simulation parameters (representative case)
STRAIN = 0.3  # 30% strain (shows good strain heterogeneity)
PLASMIN = 1.0
TIME_STEP = 0.01
MAX_TIME = 150.0

# Snapshot times for multi-panel figure
SNAPSHOT_TIMES = [0.0, 30.0, 60.0, 90.0]  # Initial, early, mid, late


def compute_strain_color(strain):
    """
    Map fiber strain to color using gradient: blue -> cyan -> yellow -> orange -> red.

    Matches GUI coloring scheme.
    """
    keypoints = [
        (0.0, (0x44, 0x88, 0xFF)),  # Blue
        (0.15, (0x44, 0xFF, 0xFF)),  # Cyan
        (0.25, (0xFF, 0xFF, 0x44)),  # Yellow
        (0.35, (0xFF, 0xAA, 0x00)),  # Orange
        (0.50, (0xFF, 0x44, 0x44)),  # Red
    ]

    strain = max(0.0, min(strain, 1.0))

    if strain <= keypoints[0][0]:
        r, g, b = keypoints[0][1]
    elif strain >= keypoints[-1][0]:
        r, g, b = keypoints[-1][1]
    else:
        for i in range(len(keypoints) - 1):
            s_low, (r_low, g_low, b_low) = keypoints[i]
            s_high, (r_high, g_high, b_high) = keypoints[i + 1]
            if s_low <= strain <= s_high:
                t = (strain - s_low) / (s_high - s_low)
                r = int(r_low + t * (r_high - r_low))
                g = int(g_low + t * (g_high - g_low))
                b = int(b_low + t * (b_high - b_low))
                break

    return (r/255, g/255, b/255)


def plot_network_state(ax, render_data, title, show_critical=False):
    """Plot network on given axes with strain-based coloring."""
    nodes = render_data['nodes']
    edges = render_data['edges']
    strains = render_data.get('strains', {})
    critical_fiber_id = render_data.get('critical_fiber_id', None)

    if not nodes:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return

    # Get bounds
    xs = [pos[0] for pos in nodes.values()]
    ys = [pos[1] for pos in nodes.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Draw edges
    for edge_id, n_from, n_to, is_ruptured in edges:
        if n_from not in nodes or n_to not in nodes:
            continue

        x1, y1 = nodes[n_from]
        x2, y2 = nodes[n_to]

        # Determine color and width
        is_critical = (show_critical and critical_fiber_id is not None and edge_id == critical_fiber_id)

        if is_critical:
            color = (1.0, 0.0, 1.0)  # Magenta
            width = 3.0
            zorder = 10
        elif is_ruptured:
            color = (1.0, 0.27, 0.27)  # Red
            width = 0.5
            zorder = 1
        else:
            strain = strains.get(edge_id, 0.0)
            color = compute_strain_color(strain)
            width = 1.5
            zorder = 2

        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, zorder=zorder, solid_capstyle='round')

    # Draw nodes (small dots)
    for x, y in nodes.values():
        ax.plot(x, y, 'o', color='black', markersize=2, zorder=5)

    # Formatting
    ax.set_xlim(x_min - 10, x_max + 10)
    ax.set_ylim(y_min - 10, y_max + 10)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')

    # Add scale bar
    scale_bar_length = 10  # µm in abstract units
    scale_x = x_max - 20
    scale_y = y_min + 5
    ax.plot([scale_x, scale_x + scale_bar_length], [scale_y, scale_y], 'k-', linewidth=2)
    ax.text(scale_x + scale_bar_length/2, scale_y - 3, '10 µm', ha='center', fontsize=8)


def create_color_legend(ax):
    """Create strain colorbar legend."""
    strain_values = np.linspace(0, 0.5, 100)
    colors = [compute_strain_color(s) for s in strain_values]

    # Create gradient
    gradient = np.vstack([colors] * 10)
    ax.imshow(gradient, aspect='auto', extent=[0, 0.5, 0, 1], origin='lower')

    ax.set_xlabel('Fiber Strain (ε)', fontsize=10)
    ax.set_yticks([])
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5'])
    ax.set_title('Strain Color Scale', fontsize=10, fontweight='bold')


def generate_evolution_figure():
    """Generate 4-panel network evolution figure."""
    print("=" * 80)
    print("PUBLICATION FIGURE GENERATION")
    print("=" * 80)
    print(f"\nGenerating network evolution figure at strain = {STRAIN*100:.0f}%")

    # Initialize adapter
    adapter = CoreV2GUIAdapter()
    adapter.load_from_excel(INPUT_FILE)

    adapter.configure_parameters(
        plasmin_concentration=PLASMIN,
        time_step=TIME_STEP,
        max_time=MAX_TIME,
        applied_strain=STRAIN
    )

    # Collect snapshots
    snapshots = []
    snapshot_idx = 0

    print("\nRunning simulation and collecting snapshots...")
    adapter.start_simulation()

    # Initial state
    snapshots.append({
        'time': 0.0,
        'data': adapter.get_render_data(),
        'lysis': 0.0
    })
    print(f"  Snapshot 1: t=0.0s (initial state)")

    # Run simulation and capture snapshots
    while True:
        running = adapter.advance_one_batch()
        sim_time = adapter.get_current_time()
        lysis = adapter.get_lysis_fraction()

        # Capture snapshot at target times
        if snapshot_idx + 1 < len(SNAPSHOT_TIMES):
            target_time = SNAPSHOT_TIMES[snapshot_idx + 1]
            if sim_time >= target_time:
                snapshots.append({
                    'time': sim_time,
                    'data': adapter.get_render_data(),
                    'lysis': lysis
                })
                snapshot_idx += 1
                print(f"  Snapshot {snapshot_idx + 1}: t={sim_time:.1f}s (lysis={lysis*100:.1f}%)")

        if not running or snapshot_idx >= len(SNAPSHOT_TIMES) - 1:
            # Capture final state if not already captured
            if sim_time > snapshots[-1]['time']:
                snapshots.append({
                    'time': sim_time,
                    'data': adapter.get_render_data(),
                    'lysis': lysis
                })
                print(f"  Snapshot {len(snapshots)}: t={sim_time:.1f}s (final, lysis={lysis*100:.1f}%)")
            break

    print(f"\nSimulation terminated: {adapter.termination_reason}")
    print(f"Collected {len(snapshots)} snapshots")

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.15], hspace=0.3, wspace=0.2)

    # Plot network snapshots
    for i, snapshot in enumerate(snapshots[:4]):  # Max 4 panels
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col:col+2])

        title = f"t = {snapshot['time']:.1f}s, Lysis = {snapshot['lysis']*100:.1f}%"
        show_critical = (i == len(snapshots) - 1)  # Show critical fiber in last panel
        plot_network_state(ax, snapshot['data'], title, show_critical=show_critical)

    # Add color legend
    ax_legend = fig.add_subplot(gs[2, :])
    create_color_legend(ax_legend)

    # Add panel labels
    panel_labels = ['A', 'B', 'C', 'D']
    for i in range(min(4, len(snapshots))):
        row = i // 2
        col = i % 2
        ax = fig.axes[i]
        ax.text(0.02, 0.98, panel_labels[i], transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Overall title
    fig.suptitle(f'Strain-Inhibited Fibrinolysis: Network Evolution at ε = {STRAIN*100:.0f}%',
                 fontsize=14, fontweight='bold', y=0.98)

    # Save
    output_path = os.path.join(OUTPUT_DIR, f"network_evolution_strain_{int(STRAIN*100):02d}pct.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    plt.close()

    return adapter.termination_reason == "network_cleared"


def main():
    """Generate all publication figures."""
    success = generate_evolution_figure()

    print("\n" + "=" * 80)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated figures:")
    print("  - network_evolution_strain_30pct.png (multi-panel network evolution)")

    if success:
        print("\n[SUCCESS] Network cleared during simulation - critical fiber visible in final panel")
    else:
        print("\n  Network did not clear - extend MAX_TIME to capture clearance event")

    print("\nThese figures are publication-ready (300 DPI) and can be used in:")
    print("  - Main manuscript figures")
    print("  - Supplementary information")
    print("  - Grant proposals")
    print("  - Conference presentations")

    return True


if __name__ == "__main__":
    main()
