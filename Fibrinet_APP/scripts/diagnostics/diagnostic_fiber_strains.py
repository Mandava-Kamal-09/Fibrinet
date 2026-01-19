"""
Diagnostic: Check actual fiber strains at different applied_strain values
===============================================================
This script verifies that the prestrain fix is working correctly
by printing actual fiber strain distributions.
"""

import sys
import os
import numpy as np

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter

# Load network
input_file = os.path.join(_project_root, "test", "input_data", "fibrin_network_big.xlsx")
adapter = CoreV2GUIAdapter()
adapter.load_from_excel(input_file)

# Test different applied strains
applied_strains = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

print("=" * 70)
print("FIBER STRAIN DIAGNOSTIC")
print("=" * 70)

for applied_strain in applied_strains:
    print(f"\n>>> Applied Strain = {applied_strain*100:.0f}%")
    print("-" * 70)

    # Configure and create initial state
    adapter.configure_parameters(
        plasmin_concentration=1.0,
        time_step=0.05,
        max_time=300.0,
        applied_strain=applied_strain
    )

    # Create state (this applies prestrain + boundary strain)
    state = adapter._create_core_v2_state(applied_strain)

    # Calculate actual fiber strains
    fiber_strains = []
    for fiber in state.fibers:
        # Get current node positions
        pos_i = state.node_positions[fiber.node_i]
        pos_j = state.node_positions[fiber.node_j]

        # Current length
        current_length = float(np.linalg.norm(pos_j - pos_i))

        # Strain = (L - L_c) / L_c
        strain = (current_length - fiber.L_c) / fiber.L_c
        fiber_strains.append(strain)

    # Statistics
    fiber_strains = np.array(fiber_strains)
    print(f"  Fiber strain statistics:")
    print(f"    Min:    {fiber_strains.min():.4f}")
    print(f"    Mean:   {fiber_strains.mean():.4f}")
    print(f"    Median: {np.median(fiber_strains):.4f}")
    print(f"    Max:    {fiber_strains.max():.4f}")
    print(f"    Std:    {fiber_strains.std():.4f}")

    # Show if all fibers have ~23% strain (BUG) or if they vary (CORRECT)
    if fiber_strains.std() < 0.01:
        print(f"  WARNING: All fibers have nearly identical strain (bug!)")
    else:
        print(f"  OK: Fiber strains vary (expected)")

    # Show distribution
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    hist, _ = np.histogram(fiber_strains, bins=bins)
    print(f"  Distribution:")
    for i in range(len(bins)-1):
        pct = (hist[i] / len(fiber_strains)) * 100
        print(f"    {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]:2d} fibers ({pct:5.1f}%)")

print("\n" + "=" * 70)
print("EXPECTED BEHAVIOR:")
print("  - Higher applied_strain → higher mean fiber strain")
print("  - Fiber strain std should increase with applied_strain")
print("  - At applied_strain=0%, fibers should have ~23% strain (prestrain only)")
print("=" * 70)
