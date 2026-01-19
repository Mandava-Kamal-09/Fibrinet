"""
BOOSTED + STABILIZED STRAIN SWEEP
Implements the remediation protocol to reveal the Golden Curve

Parameters:
- Plasmin: 10.0x (breaks ceiling effect)
- Timestep: 0.002s (prevents numerical instability)
- Max time: 200s (creates timeout contrast)
"""

import sys
import os
import numpy as np
import csv
from datetime import datetime

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter

# ============================================================================
# CONFIGURATION (BOOSTED + STABILIZED)
# ============================================================================

EXCEL_FILE = os.path.join(_project_root, "test/input_data/fibrin_network_big.xlsx")
OUTPUT_CSV = os.path.join(_project_root, "boosted_strain_sweep_results.csv")

# REMEDIATION PARAMETERS
PLASMIN_CONCENTRATION = 10.0  # 10x boost (breaks ceiling)
TIME_STEP = 0.002              # 5x smaller (prevents instability)
MAX_TIME = 200.0               # Reduced timeout (creates contrast)

# Test matrix
APPLIED_STRAINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
LYSIS_THRESHOLD = 0.5  # Track time to 50% lysis

# ============================================================================
# EXECUTION
# ============================================================================

print("=" * 80)
print("BOOSTED + STABILIZED STRAIN SWEEP")
print("=" * 80)
print()
print("REMEDIATION PARAMETERS:")
print(f"  Plasmin concentration: {PLASMIN_CONCENTRATION:.1f}x (10x BOOST)")
print(f"  Time step: {TIME_STEP:.4f}s (5x SMALLER for stability)")
print(f"  Max time: {MAX_TIME:.1f}s (reduced timeout)")
print()
print(f"Input: {EXCEL_FILE}")
print(f"Output: {OUTPUT_CSV}")
print(f"Test strains: {APPLIED_STRAINS}")
print()

# Verify network file exists
if not os.path.exists(EXCEL_FILE):
    print(f"ERROR: Network file not found: {EXCEL_FILE}")
    print()
    print("Available test files:")
    if os.path.exists("test/input_data"):
        for f in os.listdir("test/input_data"):
            if f.endswith('.xlsx'):
                print(f"  - test/input_data/{f}")
    sys.exit(1)

# Pre-flight check: Load network to verify boundaries
print("[PRE-FLIGHT] Verifying network structure...")
print("-" * 80)

adapter_test = CoreV2GUIAdapter()
adapter_test.load_from_excel(EXCEL_FILE)

print(f"  Nodes: {len(adapter_test.node_coords_raw)}")
print(f"  Edges: {len(adapter_test._edges_raw)}")
print(f"  Left boundary: {len(adapter_test.left_boundary_node_ids)} nodes")
print(f"  Right boundary: {len(adapter_test.right_boundary_node_ids)} nodes")

if len(adapter_test.left_boundary_node_ids) == 0 or len(adapter_test.right_boundary_node_ids) == 0:
    print()
    print("ERROR: No boundary nodes detected!")
    sys.exit(1)

print(f"  [PASS] Network loaded successfully")
print()

# Run strain sweep
print("[EXECUTING] Boosted strain sweep...")
print("=" * 80)
print()

results = []

for strain_idx, applied_strain in enumerate(APPLIED_STRAINS):
    print(f"Test {strain_idx + 1}/{len(APPLIED_STRAINS)}: Applied Strain = {applied_strain:.1f}")
    print("-" * 80)

    # Create fresh adapter
    adapter = CoreV2GUIAdapter()
    adapter.load_from_excel(EXCEL_FILE)

    # Configure with BOOSTED + STABILIZED parameters
    adapter.configure_parameters(
        plasmin_concentration=PLASMIN_CONCENTRATION,
        time_step=TIME_STEP,
        max_time=MAX_TIME,
        applied_strain=applied_strain
    )

    # Start simulation
    adapter.start_simulation()

    # Measure actual fiber strains after boundary application
    fiber_strains = []
    for fiber in adapter.simulation.state.fibers:
        pos_i = adapter.simulation.state.node_positions[fiber.node_i]
        pos_j = adapter.simulation.state.node_positions[fiber.node_j]
        length = float(np.linalg.norm(pos_j - pos_i))
        strain = (length - fiber.L_c) / fiber.L_c
        fiber_strains.append(strain)

    avg_strain = np.mean(fiber_strains)
    min_strain = np.min(fiber_strains)
    max_strain = np.max(fiber_strains)

    print(f"  Fiber strains: mean={avg_strain:.3f}, min={min_strain:.3f}, max={max_strain:.3f}")

    # Run simulation
    step_count = 0
    t_50 = None
    t_90 = None

    print(f"  Simulating...", end='')

    while adapter.advance_one_batch():
        step_count += 1

        lysis_frac = adapter.get_lysis_fraction()
        current_time = adapter.get_current_time()

        # Track thresholds
        if t_50 is None and lysis_frac >= 0.5:
            t_50 = current_time
        if t_90 is None and lysis_frac >= 0.9:
            t_90 = current_time

        # Progress indicator
        if step_count % 100 == 0:
            print('.', end='', flush=True)

    print()  # Newline after dots

    # Final results
    final_time = adapter.get_current_time()
    final_lysis = adapter.get_lysis_fraction()
    termination = adapter.termination_reason or "Timeout"

    if t_50 is None:
        t_50 = final_time
    if t_90 is None:
        t_90 = final_time

    # Determine status
    if termination == "Network cleared":
        status = "CLEARED"
    elif final_lysis >= 0.5:
        status = "PARTIAL"
    else:
        status = "TIMEOUT"

    print(f"  Result: T50={t_50:.1f}s, Final={final_lysis*100:.1f}%, Status={status}")
    print()

    results.append({
        'applied_strain': applied_strain,
        'avg_fiber_strain': avg_strain,
        'min_fiber_strain': min_strain,
        'max_fiber_strain': max_strain,
        't_50': t_50,
        't_90': t_90,
        'final_time': final_time,
        'final_lysis': final_lysis,
        'n_steps': step_count,
        'termination': termination,
        'status': status
    })

# Save results to CSV
print("=" * 80)
print("[SAVING] Writing results to CSV...")
print()

with open(OUTPUT_CSV, 'w', newline='') as f:
    fieldnames = ['applied_strain', 'avg_fiber_strain', 'min_fiber_strain', 'max_fiber_strain',
                  't_50', 't_90', 'final_time', 'final_lysis', 'n_steps', 'termination', 'status']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"Results saved to: {OUTPUT_CSV}")
print()

# Display summary table
print("=" * 80)
print("RESULTS SUMMARY: THE GOLDEN CURVE")
print("=" * 80)
print()
print(f"{'Applied':<10} {'Avg Fiber':<12} {'T50':<10} {'Final':<10} {'Final':<12} {'Status':<10}")
print(f"{'Strain':<10} {'Strain':<12} {'(sec)':<10} {'Time':<10} {'Lysis (%)':<12} {'':<10}")
print("-" * 80)

for r in results:
    t50_str = f"{r['t_50']:.1f}" if r['t_50'] < MAX_TIME else f">{MAX_TIME:.0f}"
    lysis_pct = r['final_lysis'] * 100

    print(f"{r['applied_strain']:<10.1f} {r['avg_fiber_strain']:<12.3f} {t50_str:<10} "
          f"{r['final_time']:<10.1f} {lysis_pct:<12.1f} {r['status']:<10}")

print("=" * 80)
print()

# Analysis
print("DIAGNOSTIC ANALYSIS:")
print("-" * 80)

# Check if strains vary
strain_range = max([r['avg_fiber_strain'] for r in results]) - min([r['avg_fiber_strain'] for r in results])
print(f"Fiber strain variation: {strain_range:.3f}")

if strain_range < 0.05:
    print("  [FAIL] Strains not varying - boundary actuation broken")
else:
    print("  [PASS] Strains vary correctly with applied strain")

# Check if T50 varies
t50_values = [r['t_50'] for r in results]
t50_range = max(t50_values) - min(t50_values)
print(f"T50 variation: {t50_range:.1f} seconds")

if t50_range < 10.0:
    print("  [FAIL] T50 not varying - mechanochemical coupling inactive")
else:
    print("  [PASS] T50 varies with strain - coupling is ACTIVE!")

# Check for monotonic increase (Golden Curve signature)
is_monotonic = all(t50_values[i] <= t50_values[i+1] for i in range(len(t50_values)-1))

if is_monotonic:
    print(f"  [GOLDEN CURVE DETECTED] T50 increases monotonically with strain!")
    print(f"    Protection factor: {t50_values[-1] / t50_values[0]:.1f}x at strain {APPLIED_STRAINS[-1]}")
else:
    print(f"  [WARNING] T50 not monotonic - check for numerical instabilities")

# Check if Strain 0.3 stabilized
strain_03_result = next((r for r in results if r['applied_strain'] == 0.3), None)
if strain_03_result:
    if strain_03_result['status'] == "CLEARED" and strain_03_result['t_50'] < 50:
        print(f"  [WARNING] Strain 0.3 still shows fast rupture (T50={strain_03_result['t_50']:.1f}s)")
        print(f"    Try smaller timestep (dt=0.001) for better stability")
    else:
        print(f"  [PASS] Strain 0.3 stabilized (T50={strain_03_result['t_50']:.1f}s)")

print()
print("CONCLUSION:")
print("-" * 80)

if is_monotonic and t50_range > 50.0:
    print("SUCCESS! Your mechanochemical coupling is working.")
    print(f"The Golden Curve shows {t50_values[-1]/t50_values[0]:.1f}x protection at high strain.")
    print()
    print("NEXT STEPS:")
    print("  1. Generate publication figures from this data")
    print("  2. Export degradation_history.csv for detailed analysis")
    print("  3. Run higher-resolution sweep (more strain points)")
elif strain_range < 0.05:
    print("ISSUE: Boundary actuation failure")
    print("  -> Check Excel boundary node flags")
    print("  -> Verify nodes are spatially separated (left vs right)")
else:
    print("ISSUE: Ceiling effect persists or instability present")
    print("  -> Try 100x plasmin boost (plasmin_concentration=100.0)")
    print("  -> Try smaller timestep (time_step=0.001)")

print("=" * 80)
print()
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
