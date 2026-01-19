"""
Diagnostic Strain Sweep with Plasmin Boost
Addresses: Ceiling Effect & Boundary Detection Verification

This script:
1. Verifies boundary nodes are detected from Excel
2. Shows actual fiber strains achieved at each applied strain level
3. Runs with HIGH plasmin concentration to break timeout ceiling
4. Generates clear comparison data

Run: python diagnostic_strain_sweep_boosted.py
"""

import sys
import os
import numpy as np

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter

# Configuration
EXCEL_FILE = os.path.join(_project_root, "test/input_data/fibrin_network_big.xlsx")  # Change to your test file
APPLIED_STRAINS = [0.0, 0.1, 0.3, 0.5]
PLASMIN_BOOST = 10.0  # 10x normal concentration (breaks ceiling)
DT = 0.01  # timestep
MAX_TIME = 300.0  # timeout
LYSIS_THRESHOLD = 0.5  # Track time to 50% lysis

print("=" * 80)
print("DIAGNOSTIC STRAIN SWEEP WITH PLASMIN BOOST")
print("=" * 80)
print()
print(f"Input file: {EXCEL_FILE}")
print(f"Plasmin concentration: {PLASMIN_BOOST:.1f}x normal (BOOSTED)")
print(f"Applied strains: {APPLIED_STRAINS}")
print(f"Max time: {MAX_TIME}s")
print()

# Load network ONCE to verify boundaries
print("[1/3] VERIFYING BOUNDARY NODE DETECTION")
print("-" * 80)

adapter_test = CoreV2GUIAdapter()
try:
    adapter_test.load_from_excel(EXCEL_FILE)
    print(f"[PASS] Network loaded successfully")
    print(f"  Total nodes: {len(adapter_test.node_coords_raw)}")
    print(f"  Total edges: {len(adapter_test._edges_raw)}")
    print(f"  Left boundary nodes: {len(adapter_test.left_boundary_node_ids)} nodes")
    print(f"    Node IDs: {sorted(adapter_test.left_boundary_node_ids)}")
    print(f"  Right boundary nodes: {len(adapter_test.right_boundary_node_ids)} nodes")
    print(f"    Node IDs: {sorted(adapter_test.right_boundary_node_ids)}")

    if len(adapter_test.left_boundary_node_ids) == 0 or len(adapter_test.right_boundary_node_ids) == 0:
        print()
        print("[FAIL] BOUNDARY NODES NOT DETECTED!")
        print("  This explains your flatline: no boundaries = no stretch")
        print("  Check Excel column names: 'is_left_boundary' and 'is_right_boundary'")
        sys.exit(1)

    print(f"  [PASS] Boundary nodes detected successfully")

except Exception as e:
    print(f"[FAIL] Could not load network: {e}")
    sys.exit(1)

print()

# Compute network span for strain calculations
x_coords = [pos[0] for pos in adapter_test.node_coords_raw.values()]
x_min, x_max = min(x_coords), max(x_coords)
x_span_raw = x_max - x_min
x_span_si = x_span_raw * adapter_test.coord_to_m

print(f"Network X-span: {x_span_raw:.1f} [abstract units] = {x_span_si:.6e} m")
print()

# Run strain sweep
print("[2/3] RUNNING STRAIN SWEEP (BOOSTED PLASMIN)")
print("-" * 80)

results = []

for applied_strain in APPLIED_STRAINS:
    print(f"\nTest: Applied Strain = {applied_strain:.1f}")
    print("  " + "-" * 60)

    # Create fresh adapter for this test
    adapter = CoreV2GUIAdapter()
    adapter.load_from_excel(EXCEL_FILE)

    # Configure with BOOSTED plasmin
    adapter.configure_parameters(
        plasmin_concentration=PLASMIN_BOOST,  # HIGH concentration
        time_step=DT,
        max_time=MAX_TIME,
        applied_strain=applied_strain
    )

    # Start simulation
    adapter.start_simulation()

    # Check actual fiber strains after boundary stretch
    fiber_strains = []
    for fiber in adapter.simulation.state.fibers:
        pos_i = adapter.simulation.state.node_positions[fiber.node_i]
        pos_j = adapter.simulation.state.node_positions[fiber.node_j]
        length = float(np.linalg.norm(pos_j - pos_i))
        strain = (length - fiber.L_c) / fiber.L_c
        fiber_strains.append(strain)

    avg_fiber_strain = np.mean(fiber_strains)
    min_fiber_strain = np.min(fiber_strains)
    max_fiber_strain = np.max(fiber_strains)

    print(f"  Initial fiber strains (after boundary stretch):")
    print(f"    Mean: {avg_fiber_strain:.3f}")
    print(f"    Min:  {min_fiber_strain:.3f}")
    print(f"    Max:  {max_fiber_strain:.3f}")

    # Run simulation
    step_count = 0
    t_50 = None

    while adapter.advance_one_batch():
        step_count += 1

        lysis_frac = adapter.get_lysis_fraction()
        current_time = adapter.get_current_time()

        # Check for 50% lysis threshold
        if t_50 is None and lysis_frac >= LYSIS_THRESHOLD:
            t_50 = current_time

        # Progress indicator every 50 steps
        if step_count % 50 == 0:
            print(f"  t={current_time:6.2f}s  Lysis={lysis_frac*100:5.1f}%  Steps={step_count}", end='\r')

    # Get final results
    final_time = adapter.get_current_time()
    final_lysis = adapter.get_lysis_fraction()
    termination = adapter.termination_reason or "Timeout"

    if t_50 is None:
        t_50 = final_time  # Did not reach 50%

    print(f"  t={final_time:6.2f}s  Lysis={final_lysis*100:5.1f}%  Steps={step_count}  ({termination})")

    results.append({
        'applied_strain': applied_strain,
        'avg_fiber_strain': avg_fiber_strain,
        'min_fiber_strain': min_fiber_strain,
        'max_fiber_strain': max_fiber_strain,
        't_50': t_50,
        'final_time': final_time,
        'final_lysis': final_lysis,
        'termination': termination
    })

print()
print()

# Summary table
print("[3/3] RESULTS SUMMARY")
print("=" * 80)
print(f"{'Applied':<12} {'Avg Fiber':<12} {'T50':<12} {'Final':<12} {'Final':<12} {'Status'}")
print(f"{'Strain':<12} {'Strain':<12} {'(seconds)':<12} {'Time (s)':<12} {'Lysis (%)':<12} {''}")
print("-" * 80)

for r in results:
    status = "TIMEOUT" if r['termination'] == "Timeout" else "CLEARED"
    t50_str = f"{r['t_50']:.1f}" if r['t_50'] < MAX_TIME else ">300"

    print(f"{r['applied_strain']:<12.1f} {r['avg_fiber_strain']:<12.3f} {t50_str:<12} "
          f"{r['final_time']:<12.1f} {r['final_lysis']*100:<12.1f} {status}")

print("=" * 80)
print()

# Diagnostic interpretation
print("INTERPRETATION:")
print("-" * 80)

strain_variation = max([r['avg_fiber_strain'] for r in results]) - min([r['avg_fiber_strain'] for r in results])
t50_variation = max([r['t_50'] for r in results]) - min([r['t_50'] for r in results])

print(f"Fiber strain variation: {strain_variation:.3f}")
print(f"T50 variation: {t50_variation:.1f} seconds")
print()

if strain_variation < 0.05:
    print("[FAIL] Fiber strains are NOT varying with applied strain!")
    print("  -> Boundary actuation is broken (geometry not stretching)")
    print("  -> All tests running at same effective strain")
elif t50_variation < 10.0:
    print("[FAIL] T50 times are NOT varying despite strain differences!")
    print("  -> Mechanochemical coupling may be broken")
    print("  -> OR plasmin concentration still too low (try 100x boost)")
else:
    print("[PASS] Strain-dependent lysis detected!")
    print(f"  -> T50 varies by {t50_variation:.1f}s across strain range")
    print("  -> Mechanochemical coupling is ACTIVE")
    print("  -> Your 'Golden Curve' exists!")

print()
print("NEXT STEPS:")
print("-" * 80)
if strain_variation < 0.05:
    print("1. Check boundary node coordinates (are left/right nodes spatially separated?)")
    print("2. Verify applied_strain is being passed to _create_core_v2_state()")
    print("3. Inspect node_positions after start_simulation() (did boundaries move?)")
elif t50_variation < 10.0:
    print("1. Increase plasmin boost to 100x (change PLASMIN_BOOST = 100.0)")
    print("2. Reduce MAX_TIME to 100s (forces early termination contrast)")
    print("3. Check if all simulations are timing out (increase max_time)")
else:
    print("1. Reduce plasmin back to 1.0-5.0x for final publication runs")
    print("2. Generate full strain sweep with more data points")
    print("3. Export degradation_history.csv for detailed analysis")

print("=" * 80)
