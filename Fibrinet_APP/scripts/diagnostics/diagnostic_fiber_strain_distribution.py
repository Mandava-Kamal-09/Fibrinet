"""
Deep Dive: Fiber Strain Distribution Analysis
==============================================

Investigates why all fibers show exactly 0.23 strain (prestrain value)
even when 30% strain is applied to boundaries.

This could indicate:
1. Network compliance (physical)
2. Geometry bug (all fibers aligned with strain direction)
3. Relaxation bug (solver not converging)
"""

import sys
import os
import numpy as np

# Try to import matplotlib (optional for visualization)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not installed. Skipping visualization.")
    print()

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter
from src.core.fibrinet_core_v2 import PhysicalConstants as PC


def analyze_fiber_strain_distribution(excel_path: str, applied_strain: float = 0.3):
    """
    Detailed analysis of fiber strain distribution.
    """
    print("=" * 80)
    print("FIBER STRAIN DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print(f"Network: {excel_path}")
    print(f"Applied Strain: {applied_strain}")
    print()

    # Load and configure
    adapter = CoreV2GUIAdapter()
    adapter.load_from_excel(excel_path)
    adapter.configure_parameters(
        plasmin_concentration=1.0,
        time_step=0.01,
        max_time=100.0,
        applied_strain=applied_strain
    )
    adapter.start_simulation()

    # Get fiber geometry
    fibers = adapter.simulation.state.fibers
    node_positions = adapter.simulation.state.node_positions

    print(f"Analyzing {len(fibers)} fibers...")
    print()

    # Collect fiber data
    fiber_data = []
    for fiber in fibers:
        pos_i = node_positions[fiber.node_i]
        pos_j = node_positions[fiber.node_j]

        # Compute fiber vector
        dx = pos_j[0] - pos_i[0]
        dy = pos_j[1] - pos_i[1]
        current_length = float(np.linalg.norm([dx, dy]))

        # Compute angle (relative to x-axis, which is strain direction)
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        # Compute strain
        strain = (current_length - fiber.L_c) / fiber.L_c

        # Compute force
        force = fiber.compute_force(current_length)

        fiber_data.append({
            'fiber_id': fiber.fiber_id,
            'L_c': fiber.L_c,
            'L_current': current_length,
            'strain': strain,
            'angle_deg': angle_deg,
            'dx': dx,
            'dy': dy,
            'force': force,
            'node_i': fiber.node_i,
            'node_j': fiber.node_j,
            'pos_i_x': pos_i[0],
            'pos_i_y': pos_i[1],
            'pos_j_x': pos_j[0],
            'pos_j_y': pos_j[1]
        })

    # Convert to arrays for analysis
    strains = np.array([f['strain'] for f in fiber_data])
    angles = np.array([f['angle_deg'] for f in fiber_data])
    forces = np.array([f['force'] for f in fiber_data])
    lengths_current = np.array([f['L_current'] for f in fiber_data])
    lengths_rest = np.array([f['L_c'] for f in fiber_data])

    # Statistical summary
    print("STRAIN STATISTICS:")
    print(f"  Mean strain: {np.mean(strains):.6f}")
    print(f"  Std dev: {np.std(strains):.6f}")
    print(f"  Min strain: {np.min(strains):.6f}")
    print(f"  Max strain: {np.max(strains):.6f}")
    print(f"  Range: {np.max(strains) - np.min(strains):.6f}")
    print()

    print("STRAIN DISTRIBUTION:")
    hist, bin_edges = np.histogram(strains, bins=10)
    for i in range(len(hist)):
        print(f"  [{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f}): {hist[i]} fibers")
    print()

    print("ANGLE STATISTICS (relative to x-axis = strain direction):")
    print(f"  Mean angle: {np.mean(angles):.1f} degrees")
    print(f"  Std dev: {np.std(angles):.1f} degrees")
    print()

    print("FORCE STATISTICS:")
    print(f"  Mean force: {np.mean(forces):.6e} N")
    print(f"  Std dev: {np.std(forces):.6e} N")
    print(f"  Min force: {np.min(forces):.6e} N")
    print(f"  Max force: {np.max(forces):.6e} N")
    print()

    # Check if all strains are identical (bug indicator)
    if np.std(strains) < 1e-9:
        print("WARNING: All fibers have IDENTICAL strain!")
        print("         This suggests a bug in the geometry or relaxation solver.")
        print(f"         All fibers have strain = {np.mean(strains):.6f}")
        print()

        # Check if all fibers have the same rest length
        if np.std(lengths_rest) < 1e-12:
            print("         All fibers have IDENTICAL rest length: {:.6e} m".format(np.mean(lengths_rest)))
            print("         This is highly unusual and likely a bug.")
        else:
            print("         Rest lengths vary: {:.6e} to {:.6e} m".format(np.min(lengths_rest), np.max(lengths_rest)))
            print("         But current lengths are scaling proportionally to maintain constant strain.")
            print("         This suggests the network is in a 'locked' state.")
    else:
        print("OK: Fibers have varying strains (expected for heterogeneous network)")
        print()

    # Check for affine deformation (all fibers scale equally)
    length_ratios = lengths_current / lengths_rest
    print("LENGTH RATIO STATISTICS (L_current / L_rest):")
    print(f"  Mean ratio: {np.mean(length_ratios):.6f}")
    print(f"  Std dev: {np.std(length_ratios):.6f}")
    print(f"  Expected for affine deformation: {1.0 + PC.PRESTRAIN:.6f}")
    print()

    if np.std(length_ratios) < 1e-9:
        print("WARNING: All fibers have IDENTICAL length ratio!")
        print("         This indicates AFFINE deformation (all points move proportionally).")
        print("         This is unphysical for a network with compliant fibers!")
        print("         Possible causes:")
        print("           1. Energy minimization solver not running")
        print("           2. All nodes are boundary nodes (over-constrained)")
        print("           3. Network has no free nodes to relax")
        print()

    # Detailed fiber-by-fiber report for first 10 fibers
    print("DETAILED FIBER REPORT (first 10 fibers):")
    print("-" * 120)
    print(f"{'ID':>5} {'L_c (m)':>12} {'L_current (m)':>15} {'Strain':>10} {'Angle':>8} {'Force (N)':>12} {'Node i':>8} {'Node j':>8}")
    print("-" * 120)
    for i, fd in enumerate(fiber_data[:10]):
        print(f"{fd['fiber_id']:5d} {fd['L_c']:12.6e} {fd['L_current']:15.6e} {fd['strain']:10.6f} {fd['angle_deg']:8.1f} {fd['force']:12.6e} {fd['node_i']:8d} {fd['node_j']:8d}")
    print("-" * 120)
    print()

    # Check boundary node count
    left_boundary = adapter.left_boundary_node_ids
    right_boundary = adapter.right_boundary_node_ids
    all_nodes = set(node_positions.keys())
    free_nodes = all_nodes - set(left_boundary) - set(right_boundary)

    print("NETWORK TOPOLOGY:")
    print(f"  Total nodes: {len(all_nodes)}")
    print(f"  Left boundary nodes: {len(left_boundary)}")
    print(f"  Right boundary nodes: {len(right_boundary)}")
    print(f"  Free nodes: {len(free_nodes)}")
    print(f"  Degrees of freedom: {len(free_nodes) * 2} (x, y for each free node)")
    print()

    if len(free_nodes) == 0:
        print("CRITICAL ERROR: No free nodes!")
        print("                All nodes are boundary-constrained.")
        print("                The network CANNOT relax - affine deformation is forced!")
        print()

    # Generate visualization
    if not HAS_MATPLOTLIB:
        print("Skipping visualization (matplotlib not installed)")
        print()
    else:
      try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Strain histogram
        axes[0, 0].hist(strains, bins=20, edgecolor='black')
        axes[0, 0].axvline(PC.PRESTRAIN, color='red', linestyle='--', label=f'Prestrain = {PC.PRESTRAIN}')
        axes[0, 0].set_xlabel('Fiber Strain')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Fiber Strain Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Strain vs Angle
        axes[0, 1].scatter(angles, strains, alpha=0.6)
        axes[0, 1].set_xlabel('Fiber Angle (degrees)')
        axes[0, 1].set_ylabel('Fiber Strain')
        axes[0, 1].set_title('Strain vs Fiber Orientation')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Network geometry
        for fd in fiber_data:
            x = [fd['pos_i_x'], fd['pos_j_x']]
            y = [fd['pos_i_y'], fd['pos_j_y']]
            strain_color = plt.cm.viridis(fd['strain'] / (np.max(strains) + 1e-9))
            axes[1, 0].plot(x, y, color=strain_color, linewidth=1, alpha=0.7)

        # Mark boundary nodes
        for nid in left_boundary:
            pos = node_positions[nid]
            axes[1, 0].scatter(pos[0], pos[1], color='red', s=50, marker='s', label='Left' if nid == left_boundary[0] else '')
        for nid in right_boundary:
            pos = node_positions[nid]
            axes[1, 0].scatter(pos[0], pos[1], color='blue', s=50, marker='s', label='Right' if nid == right_boundary[0] else '')

        axes[1, 0].set_xlabel('X position (m)')
        axes[1, 0].set_ylabel('Y position (m)')
        axes[1, 0].set_title('Network Geometry (colored by strain)')
        axes[1, 0].legend()
        axes[1, 0].set_aspect('equal')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Force distribution
        axes[1, 1].hist(forces, bins=20, edgecolor='black')
        axes[1, 1].set_xlabel('Force (N)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Fiber Force Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = 'diagnostic_fiber_strain_distribution.png'
        plt.savefig(output_path, dpi=150)
        print(f"Visualization saved to: {output_path}")
        print()
      except Exception as e:
        print(f"WARNING: Could not generate visualization: {e}")
        print()

    # DIAGNOSIS
    print("=" * 80)
    print("DIAGNOSIS:")
    print("=" * 80)

    if np.std(strains) < 1e-9 and len(free_nodes) == 0:
        print("ROOT CAUSE: OVER-CONSTRAINED NETWORK")
        print()
        print("Your network has NO free nodes! All nodes are boundary-constrained.")
        print("This forces AFFINE deformation (all points move proportionally).")
        print("The network cannot relax, so all fibers experience identical strain.")
        print()
        print("SOLUTION:")
        print("  Check your Excel file boundary flags (is_left_boundary, is_right_boundary)")
        print("  Ensure that ONLY edge nodes are marked as boundaries, not all nodes!")
    elif np.std(strains) < 1e-9:
        print("POSSIBLE BUG: All fibers have identical strain despite having free nodes.")
        print()
        print("This suggests:")
        print("  1. Network geometry is perfectly uniform (all fibers aligned)")
        print("  2. Energy minimization solver is not running")
        print("  3. Fibers have infinite stiffness (no relaxation possible)")
    elif np.abs(np.mean(strains) - PC.PRESTRAIN) < 0.01 and applied_strain > 0.2:
        print("PHYSICAL BEHAVIOR: Network compliance absorbs applied strain")
        print()
        print("The network IS relaxing, but the compliance is very high.")
        print("Free nodes adjust positions to minimize energy, which redistributes")
        print("strain unevenly. Some fibers relax while load-bearing fibers carry more.")
        print()
        print("This is EXPECTED for soft WLC networks with low connectivity.")
    else:
        print("Network appears to be functioning correctly.")
        print("Strain distribution shows expected heterogeneity.")

    print("=" * 80)


if __name__ == "__main__":
    test_network = "test/input_data/fibrin_network_big.xlsx"
    applied_strain = 0.3

    if len(sys.argv) > 1:
        test_network = sys.argv[1]
    if len(sys.argv) > 2:
        applied_strain = float(sys.argv[2])

    try:
        analyze_fiber_strain_distribution(test_network, applied_strain)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
