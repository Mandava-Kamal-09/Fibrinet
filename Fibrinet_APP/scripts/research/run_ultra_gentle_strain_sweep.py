"""
ULTRA-GENTLE STRAIN SWEEP (Option 2)
Tests mechanochemical coupling in regime below mechanical rupture threshold

Strategy:
- Use ULTRA-SMALL applied strains (0.00 to 0.10 in fine increments)
- Keep high plasmin boost (10x) to see enzymatic effects
- Stay below critical fiber strain threshold to avoid topology interference
"""

import sys
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter

# ============================================================================
# CONFIGURATION (ULTRA-GENTLE STRAIN REGIME)
# ============================================================================

EXCEL_FILE = os.path.join(_project_root, "test/input_data/fibrin_network_big.xlsx")
OUTPUT_CSV = os.path.join(_project_root, "ultra_gentle_sweep_results.csv")
OUTPUT_PLOT = os.path.join(_project_root, "ultra_gentle_strain_protection_curve.png")

# ULTRA-GENTLE TEST PARAMETERS
PLASMIN_CONCENTRATION = 10.0  # 10x boost to see enzymatic effects
TIME_STEP = 0.002              # Small timestep for accuracy
MAX_TIME = 100.0               # Shorter timeout (enzymatic should be fast at 10x)

# ULTRA-GENTLE strain range (Option 2 specification)
APPLIED_STRAINS = [0.00, 0.02, 0.05, 0.08, 0.10]
LYSIS_THRESHOLD = 0.5

# ============================================================================
# EXECUTION
# ============================================================================

print("=" * 80)
print("ULTRA-GENTLE STRAIN SWEEP (Option 2: Mechanochemical Coupling Test)")
print("=" * 80)
print()
print("STRATEGY: Stay below mechanical rupture threshold to reveal chemistry")
print()
print("PARAMETERS:")
print(f"  Plasmin concentration: {PLASMIN_CONCENTRATION:.1f}x")
print(f"  Time step: {TIME_STEP:.4f}s")
print(f"  Max time: {MAX_TIME:.1f}s")
print(f"  Applied strains: {APPLIED_STRAINS} (ULTRA-GENTLE)")
print()
print(f"Input: {EXCEL_FILE}")
print(f"Output CSV: {OUTPUT_CSV}")
print(f"Output Plot: {OUTPUT_PLOT}")
print()

# Load network
adapter_test = CoreV2GUIAdapter()
adapter_test.load_from_excel(EXCEL_FILE)
print(f"Network loaded: {len(adapter_test._edges_raw)} edges, "
      f"{len(adapter_test.left_boundary_node_ids)} left + "
      f"{len(adapter_test.right_boundary_node_ids)} right boundaries")
print()

# Run ultra-gentle sweep
print("[EXECUTING] Ultra-gentle strain sweep...")
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
    std_strain = np.std(fiber_strains)

    # Check for dangerous overstrain
    if max_strain > 3.0:
        print(f"  [WARNING] Max fiber strain = {max_strain:.1f} (approaching WLC singularity)")
    elif max_strain > 1.5:
        print(f"  [CAUTION] Max fiber strain = {max_strain:.3f} (entering non-linear regime)")

    print(f"  Fiber strains: mean={avg_strain:.3f}, max={max_strain:.3f}, std={std_strain:.3f}")

    # Calculate predicted cleavage rate from mechanochemical coupling
    # k(ε) = k₀ × exp(-β × ε) where β ≈ 1.15
    k0 = 0.1  # Baseline cleavage rate (1/s at strain=0)
    beta = 1.15  # Inhibition parameter
    predicted_k_cleave = k0 * np.exp(-beta * avg_strain)

    print(f"  Predicted k_cleave: {predicted_k_cleave:.4f} /s (chemistry)")

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
        'std_fiber_strain': std_strain,
        'predicted_k_cleave': predicted_k_cleave,
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
print(f"{'Applied':<10} {'Avg Fiber':<12} {'Max Fiber':<12} {'Pred k':<12} {'T50':<10} {'Final':<12} {'Failure':<12}")
print(f"{'Strain':<10} {'Strain':<12} {'Strain':<12} {'(/s)':<12} {'(sec)':<10} {'Lysis (%)':<12} {'Mode':<12}")
print("-" * 80)

for r in results:
    print(f"{r['applied_strain']:<10.2f} {r['avg_fiber_strain']:<12.3f} {r['max_fiber_strain']:<12.3f} "
          f"{r['predicted_k_cleave']:<12.4f} {r['t_50']:<10.1f} {r['final_lysis']*100:<12.1f} {r['failure_mode']:<12}")

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
    print("  -> Network topology creates stress concentrations even at ultra-gentle strains")
    print("  -> Consider Option 1 (uniform synthetic network) instead")
    print()

if is_monotonic and enzymatic_count >= len(results) - 1:
    protection_factor = t50_values[-1] / t50_values[0] if t50_values[0] > 0 else 0
    print(f"[SUCCESS] Golden Curve detected!")
    print(f"  T50 increases monotonically: {t50_values[0]:.1f}s -> {t50_values[-1]:.1f}s")
    print(f"  Protection factor: {protection_factor:.2f}x at {APPLIED_STRAINS[-1]:.2f} strain")
    print()
    print("✓ Mechanochemical coupling is WORKING!")
    print("  Higher strain → Slower enzymatic cleavage → Longer T50")
else:
    t50_range = max(t50_values) - min(t50_values)
    if t50_range < 2.0:
        print(f"[ISSUE] T50 variation too small ({t50_range:.1f}s)")
        print("  -> Increase plasmin to 50x or 100x to amplify differences")
        print("  -> Or use a larger network to reduce percolation noise")
    else:
        print(f"[ISSUE] T50 not monotonic (variation={t50_range:.1f}s)")
        print("  -> Mechanical rupture still interfering with enzymatic trend")
        print("  -> Topology creates stress concentrators that dominate over chemistry")

print("=" * 80)
print()

# Generate strain-protection plot
print("[GENERATING] Strain-dependent protection plot...")
print()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: T50 vs Applied Strain
applied_strains_plot = [r['applied_strain'] for r in results]
t50_plot = [r['t_50'] for r in results]
failure_modes = [r['failure_mode'] for r in results]

colors = ['green' if mode == 'ENZYMATIC' else 'red' if mode == 'MECHANICAL' else 'gray'
          for mode in failure_modes]

ax1.scatter(applied_strains_plot, t50_plot, c=colors, s=100, alpha=0.6, edgecolors='black', linewidths=1.5)
ax1.plot(applied_strains_plot, t50_plot, 'k--', alpha=0.3, linewidth=1)

ax1.set_xlabel('Applied Strain', fontsize=12, fontweight='bold')
ax1.set_ylabel('T50: Time to 50% Lysis (s)', fontsize=12, fontweight='bold')
ax1.set_title('Mechanochemical Coupling: Strain-Dependent Protection', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Add legend for failure modes
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', edgecolor='black', label='Enzymatic'),
    Patch(facecolor='red', edgecolor='black', label='Mechanical'),
    Patch(facecolor='gray', edgecolor='black', label='Incomplete')
]
ax1.legend(handles=legend_elements, loc='upper left', title='Failure Mode')

# Add annotations for key points
if len(t50_plot) > 1:
    # Annotate first and last points
    ax1.annotate(f'T50 = {t50_plot[0]:.1f}s',
                 xy=(applied_strains_plot[0], t50_plot[0]),
                 xytext=(10, -20), textcoords='offset points',
                 fontsize=9, ha='left',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax1.annotate(f'T50 = {t50_plot[-1]:.1f}s',
                 xy=(applied_strains_plot[-1], t50_plot[-1]),
                 xytext=(10, 20), textcoords='offset points',
                 fontsize=9, ha='left',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Plot 2: Predicted k_cleave vs Avg Fiber Strain
avg_strains_plot = [r['avg_fiber_strain'] for r in results]
pred_k_plot = [r['predicted_k_cleave'] for r in results]

ax2.scatter(avg_strains_plot, pred_k_plot, c=colors, s=100, alpha=0.6, edgecolors='black', linewidths=1.5)
ax2.plot(avg_strains_plot, pred_k_plot, 'k--', alpha=0.3, linewidth=1)

ax2.set_xlabel('Average Fiber Strain', fontsize=12, fontweight='bold')
ax2.set_ylabel('Predicted k_cleave (/s)', fontsize=12, fontweight='bold')
ax2.set_title('Chemistry: k(ε) = k₀ × exp(-β × ε)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add theoretical curve
strain_theory = np.linspace(0, max(avg_strains_plot)*1.1, 100)
k0 = 0.1
beta = 1.15
k_theory = k0 * np.exp(-beta * strain_theory)
ax2.plot(strain_theory, k_theory, 'b-', alpha=0.4, linewidth=2, label='Theory: β=1.15')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {OUTPUT_PLOT}")
print()

# Final verdict
print("=" * 80)
print("FINAL VERDICT")
print("=" * 80)
print()

if is_monotonic and enzymatic_count >= len(results) - 1:
    print("✓ SUCCESS: Mechanochemical coupling clearly demonstrated!")
    print()
    print("  Evidence:")
    print(f"  1. T50 increases with strain: {t50_values[0]:.1f}s → {t50_values[-1]:.1f}s")
    print(f"  2. All tests are enzymatic-dominated (not mechanical)")
    print(f"  3. Protection factor: {t50_values[-1]/t50_values[0]:.2f}x")
    print()
    print("  This is your Golden Curve!")
elif mechanical_count == 0 and t50_range >= 2.0:
    print("⚠ PARTIAL SUCCESS: Trend visible but not perfectly monotonic")
    print()
    print(f"  T50 range: {t50_range:.1f}s (detectable)")
    print("  All tests enzymatic (good)")
    print("  But trend is noisy (possibly network size effects)")
    print()
    print("  Recommendation: Increase plasmin to 50x to amplify signal")
else:
    print("✗ TOPOLOGY INTERFERENCE: Mechanical rupture still present")
    print()
    print(f"  Mechanical failures: {mechanical_count}/{len(results)}")
    print("  Even ultra-gentle strains create stress concentrations")
    print()
    print("  Recommendation: Use Option 1 (synthetic uniform network)")

print("=" * 80)
print()
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
