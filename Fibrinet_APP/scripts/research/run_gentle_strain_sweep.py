"""
GENTLE STRAIN SWEEP
Tests in regime where enzymatic lysis dominates over mechanical rupture

Strategy:
- Use SMALL applied strains (0.0 to 0.20 instead of 0.0 to 0.50)
- Keep high plasmin boost (10x) to see enzymatic effects
- Avoid stress concentration regime that causes mechanical failure
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
# CONFIGURATION (GENTLE STRAIN REGIME)
# ============================================================================

EXCEL_FILE = os.path.join(_project_root, "test/input_data/Hangman.xlsx")
OUTPUT_CSV = os.path.join(_project_root, "hangman_gentle_sweep_results.csv")

# GENTLE TEST PARAMETERS
PLASMIN_CONCENTRATION = 10.0  # Keep 10x boost
TIME_STEP = 0.002              # Keep small timestep
MAX_TIME = 100.0               # Shorter timeout (enzymatic should be fast at 10x)

# GENTLE strain range (avoid mechanical rupture)
APPLIED_STRAINS = [0.00, 0.05, 0.10, 0.15, 0.20]
LYSIS_THRESHOLD = 0.5

# ============================================================================
# EXECUTION
# ============================================================================

print("=" * 80)
print("GENTLE STRAIN SWEEP (Enzymatic Regime)")
print("=" * 80)
print()
print("STRATEGY: Test in gentle strain regime to avoid mechanical rupture")
print()
print("PARAMETERS:")
print(f"  Plasmin concentration: {PLASMIN_CONCENTRATION:.1f}x")
print(f"  Time step: {TIME_STEP:.4f}s")
print(f"  Max time: {MAX_TIME:.1f}s")
print(f"  Applied strains: {APPLIED_STRAINS} (GENTLE)")
print()
print(f"Input: {EXCEL_FILE}")
print(f"Output: {OUTPUT_CSV}")
print()

# Load network
adapter_test = CoreV2GUIAdapter()
adapter_test.load_from_excel(EXCEL_FILE)
print(f"Network loaded: {len(adapter_test._edges_raw)} edges, "
      f"{len(adapter_test.left_boundary_node_ids)} left + "
      f"{len(adapter_test.right_boundary_node_ids)} right boundaries")
print()

# Run gentle sweep
print("[EXECUTING] Gentle strain sweep...")
print("=" * 80)
print()

results = []

for strain_idx, applied_strain in enumerate(APPLIED_STRAINS):
    print(f"Test {strain_idx + 1}/{len(APPLIED_STRAINS)}: Applied Strain = {applied_strain:.2f}")
    print("-" * 80)

    adapter = CoreV2GUIAdapter()
    adapter.load_from_excel(EXCEL_FILE)
    adapter.configure_parameters(
        plasmin_concentration=PLASMIN_CONCENTRATION,
        time_step=TIME_STEP,
        max_time=MAX_TIME,
        applied_strain=applied_strain
    )
    adapter.start_simulation()

    # Measure fiber strains
    fiber_strains = []
    for fiber in adapter.simulation.state.fibers:
        pos_i = adapter.simulation.state.node_positions[fiber.node_i]
        pos_j = adapter.simulation.state.node_positions[fiber.node_j]
        length = float(np.linalg.norm(pos_j - pos_i))
        strain = (length - fiber.L_c) / fiber.L_c
        fiber_strains.append(strain)

    avg_strain = np.mean(fiber_strains)
    max_strain = np.max(fiber_strains)

    # Check for dangerous overstrain
    if max_strain > 3.0:
        print(f"  [WARNING] Max fiber strain = {max_strain:.1f} (approaching singularity)")

    print(f"  Fiber strains: mean={avg_strain:.3f}, max={max_strain:.3f}")

    # Run simulation
    step_count = 0
    t_50 = None

    print(f"  Simulating...", end='')

    while adapter.advance_one_batch():
        step_count += 1

        lysis_frac = adapter.get_lysis_fraction()
        current_time = adapter.get_current_time()

        if t_50 is None and lysis_frac >= 0.5:
            t_50 = current_time

        if step_count % 100 == 0:
            print('.', end='', flush=True)

    print()

    final_time = adapter.get_current_time()
    final_lysis = adapter.get_lysis_fraction()
    termination = adapter.termination_reason or "Timeout"

    if t_50 is None:
        t_50 = final_time

    # Classify failure mode
    if "cleared" in termination.lower():
        if final_lysis < 0.3:
            failure_mode = "MECHANICAL"  # Cleared with low lysis = mechanical rupture
        else:
            failure_mode = "ENZYMATIC"   # Cleared with high lysis = enzymatic
    else:
        failure_mode = "INCOMPLETE"

    print(f"  Result: T50={t_50:.1f}s, Final={final_lysis*100:.1f}%, Mode={failure_mode}")
    print()

    results.append({
        'applied_strain': applied_strain,
        'avg_fiber_strain': avg_strain,
        'max_fiber_strain': max_strain,
        't_50': t_50,
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

# Summary
print("=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print()
print(f"{'Applied':<10} {'Avg Fiber':<12} {'Max Fiber':<12} {'T50':<10} {'Final':<12} {'Failure':<12}")
print(f"{'Strain':<10} {'Strain':<12} {'Strain':<12} {'(sec)':<10} {'Lysis (%)':<12} {'Mode':<12}")
print("-" * 80)

for r in results:
    print(f"{r['applied_strain']:<10.2f} {r['avg_fiber_strain']:<12.3f} {r['max_fiber_strain']:<12.3f} "
          f"{r['t_50']:<10.1f} {r['final_lysis']*100:<12.1f} {r['failure_mode']:<12}")

print("=" * 80)
print()

# Analysis
print("DIAGNOSTIC ANALYSIS:")
print("-" * 80)

t50_values = [r['t_50'] for r in results]
is_monotonic = all(t50_values[i] <= t50_values[i+1] for i in range(len(t50_values)-1))

enzymatic_count = sum(1 for r in results if r['failure_mode'] == "ENZYMATIC")
mechanical_count = sum(1 for r in results if r['failure_mode'] == "MECHANICAL")

print(f"Failure modes: {enzymatic_count} enzymatic, {mechanical_count} mechanical")
print()

if mechanical_count > 0:
    print("[WARNING] Some tests still showing mechanical rupture")
    print("  -> Network topology creates stress concentrations")
    print("  -> Try even gentler strains (0.00, 0.02, 0.05, 0.08, 0.10)")
    print()

if is_monotonic and enzymatic_count >= len(results) - 1:
    protection_factor = t50_values[-1] / t50_values[0] if t50_values[0] > 0 else 0
    print(f"[SUCCESS] Golden Curve detected!")
    print(f"  T50 increases monotonically: {t50_values[0]:.1f}s -> {t50_values[-1]:.1f}s")
    print(f"  Protection factor: {protection_factor:.1f}x at {APPLIED_STRAINS[-1]:.2f} strain")
    print()
    print("Your mechanochemical coupling is WORKING!")
else:
    t50_range = max(t50_values) - min(t50_values)
    if t50_range < 5.0:
        print(f"[ISSUE] T50 variation too small ({t50_range:.1f}s)")
        print("  -> Increase plasmin to 50x or 100x to amplify differences")
    else:
        print(f"[ISSUE] T50 not monotonic (variation={t50_range:.1f}s)")
        print("  -> Mechanical rupture interfering with enzymatic trend")

print("=" * 80)
print()
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
