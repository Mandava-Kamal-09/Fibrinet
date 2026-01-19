"""
Diagnostic: Check if node positions are actually changing during simulation.

This will help us understand if the issue is:
1. Positions not updating (BUG)
2. Positions updating but deformation too small to see (PHYSICS)
"""

import sys
import os
import numpy as np

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter

# Load network using adapter (handles Excel format correctly)
print("Loading network...")
adapter = CoreV2GUIAdapter()
success = adapter.load_from_excel(os.path.join(_project_root, 'test/input_data/TestNetwork.xlsx'))
if not success:
    print("ERROR: Failed to load network")
    sys.exit(1)

adapter.configure_simulation_parameters(
    plasmin_nM=100.0,
    temperature_K=310.0,
    dt_chem_s=0.1,
    t_max_s=50.0,
    lysis_threshold=0.99,
    delta_S=0.1
)
adapter.set_strain(0.3)
initial_state = adapter.simulation.state

# Store initial positions for comparison
initial_positions = {nid: pos.copy() for nid, pos in initial_state.node_positions.items()}

# Run simulation
print(f"Initial nodes: {len(initial_positions)}")
print(f"Initial fibers: {len(initial_state.fibers)}")

# Start simulation
adapter.start_simulation()

# Run simulation until termination
print("\nRunning simulation...")
step_count = 0
while True:
    # Advance simulation by one batch (multiple chemistry steps)
    result = adapter.advance_one_batch()
    step_count += 1

    if step_count % 10 == 0:
        lysis = adapter.get_lysis_fraction()
        time = adapter.get_current_time()
        print(f"  Step {step_count}: t={time:.2f}s, lysis={lysis:.3f}")

    # Check termination
    if result == "network_cleared":
        print(f"\n✓ Network cleared at step {step_count}")
        break
    elif result == "time_limit":
        print(f"\n⚠ Time limit reached at step {step_count}")
        break
    elif result == "lysis_threshold":
        print(f"\n⚠ Lysis threshold reached at step {step_count}")
        break
    elif result == "complete_rupture":
        print(f"\n⚠ All fibers ruptured at step {step_count}")
        break

    if step_count > 1000:
        print("\n⚠ Max steps reached (safety limit)")
        break

# Analyze node displacement
print(f"\n{'='*70}")
print("NODE DISPLACEMENT ANALYSIS")
print(f"{'='*70}")

final_positions = adapter.simulation.state.node_positions
displacements = {}

for nid, final_pos in final_positions.items():
    initial_pos = initial_positions[nid]
    displacement = np.linalg.norm(final_pos - initial_pos)
    displacements[nid] = displacement

# Statistics
disp_values = list(displacements.values())
max_disp = max(disp_values)
mean_disp = np.mean(disp_values)
median_disp = np.median(disp_values)

print(f"\nDisplacement Statistics (in meters):")
print(f"  Max displacement:    {max_disp:.6e} m")
print(f"  Mean displacement:   {mean_disp:.6e} m")
print(f"  Median displacement: {median_disp:.6e} m")

# Show top 10 most displaced nodes
print(f"\nTop 10 Most Displaced Nodes:")
sorted_nodes = sorted(displacements.items(), key=lambda x: x[1], reverse=True)[:10]
for nid, disp in sorted_nodes:
    in_left = nid in initial_state.left_boundary_nodes
    in_right = nid in initial_state.right_boundary_nodes
    boundary_status = "LEFT BOUNDARY" if in_left else ("RIGHT BOUNDARY" if in_right else "FREE")
    print(f"  Node {nid:3d}: {disp:.6e} m  [{boundary_status}]")

# Check if positions are actually different
if max_disp < 1e-12:
    print("\n⚠ WARNING: Node positions have NOT changed!")
    print("   This is a BUG - positions should update during simulation.")
elif max_disp < 1e-6:
    print("\n⚠ ISSUE: Displacements are VERY SMALL (< 1 micron)")
    print("   Positions are updating, but deformation is too small to see visually.")
    print("   Recommendation: Add displacement amplification or use displacement vectors.")
else:
    print(f"\n✓ Node positions ARE changing (max displacement: {max_disp*1e6:.2f} microns)")
    print("   If visualization looks unchanged, this is a visualization scaling issue.")

# Compare initial vs final network span
initial_xs = [pos[0] for pos in initial_positions.values()]
final_xs = [pos[0] for pos in final_positions.values()]
initial_span = max(initial_xs) - min(initial_xs)
final_span = max(final_xs) - min(final_xs)

print(f"\nNetwork X-span:")
print(f"  Initial: {initial_span:.6e} m")
print(f"  Final:   {final_span:.6e} m")
print(f"  Change:  {(final_span - initial_span):.6e} m ({(final_span/initial_span - 1)*100:.2f}%)")

print(f"\n{'='*70}")
