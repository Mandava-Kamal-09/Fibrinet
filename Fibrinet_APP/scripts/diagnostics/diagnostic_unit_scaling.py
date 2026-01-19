"""
High-Integrity Diagnostic: Unit Scaling Audit
Traces forces from geometry → WLC → chemistry without running full simulation
"""

import sys
import os
import numpy as np

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.core.fibrinet_core_v2 import WLCFiber, PhysicalConstants as PC

print("=" * 80)
print("PHASE 3 DIAGNOSTIC: UNIT SCALING AUDIT")
print("=" * 80)
print()

# Simulate typical network geometry
print("[1/4] Network Geometry (Typical Values)")
print("-" * 80)

coord_to_m = 1.0e-6  # Default: 1 unit = 1 µm
fiber_length_abstract = 50.0  # Typical fiber spans ~50 abstract units
fiber_length_si = fiber_length_abstract * coord_to_m  # Convert to meters

print(f"Coordinate scaling: {coord_to_m:.6e} m/unit")
print(f"Fiber geometric length: {fiber_length_abstract:.1f} [abstract units]")
print(f"Fiber geometric length: {fiber_length_si:.6e} m ({fiber_length_si * 1e6:.1f} µm)")
print()

# Compute rest length with prestrain
print("[2/4] Rest Length Calculation (with 23% prestrain)")
print("-" * 80)

L_c = fiber_length_si / (1.0 + PC.PRESTRAIN)  # Rest length
print(f"Prestrain factor: {PC.PRESTRAIN} (fibers born under 23% tension)")
print(f"Rest length L_c: {L_c:.6e} m ({L_c * 1e6:.1f} µm)")
print()

# Apply test strains
test_strains = [0.0, 0.1, 0.3, 0.5]

print("[3/4] WLC Force Calculation at Different Applied Strains")
print("-" * 80)
print(f"{'Applied':<10} {'Network':<10} {'Fiber':<12} {'Fiber':<12} {'WLC Force':<15} {'Order'}")
print(f"{'Strain':<10} {'Stretch':<10} {'Length (um)':<12} {'Strain e':<12} {'(Newtons)':<15} {'Magnitude'}")
print("-" * 80)

fiber = WLCFiber(
    fiber_id=1,
    node_i=0,
    node_j=1,
    L_c=L_c,
    xi=PC.xi,
    S=1.0
)

for applied_strain in test_strains:
    # Network stretch (applied to boundary)
    network_stretch = 1.0 + applied_strain

    # Fiber length after network stretch (assuming affine deformation)
    # Original fiber: fiber_length_si (already includes 23% prestrain)
    # After stretch: fiber_length_si * network_stretch
    current_length = fiber_length_si * network_stretch

    # Fiber strain: ε = (L - L_c) / L_c
    fiber_strain = (current_length - L_c) / L_c

    # WLC force
    force = fiber.compute_force(current_length)

    # Order of magnitude
    if force == 0:
        order = "ZERO"
    elif force < 1e-15:
        order = "atto-N"
    elif force < 1e-12:
        order = "femto-N [!]"
    elif force < 1e-9:
        order = "pico-N"
    elif force < 1e-6:
        order = "nano-N [OK]"
    else:
        order = "micro-N"

    print(f"{applied_strain:<10.1f} {network_stretch:<10.2f} {current_length*1e6:<12.2f} {fiber_strain:<12.3f} {force:<15.6e} {order}")

print()

# Chemical sensitivity check
print("[4/4] Chemical Inhibition (Strain-Based)")
print("-" * 80)
print(f"{'Applied':<10} {'Fiber':<12} {'k_cleave':<15} {'Inhibition':<15} {'Status'}")
print(f"{'Strain':<10} {'Strain e':<12} {'(1/s)':<15} {'Factor':<15} {''}")
print("-" * 80)

k_baseline = PC.k_cat_0  # 0.1 /s

for applied_strain in test_strains:
    network_stretch = 1.0 + applied_strain
    current_length = fiber_length_si * network_stretch

    # Cleavage rate with strain inhibition
    k_cleave = fiber.compute_cleavage_rate(current_length)

    # Inhibition factor
    inhibition = k_baseline / k_cleave if k_cleave > 0 else float('inf')

    fiber_strain = (current_length - L_c) / L_c

    if inhibition < 2:
        status = "Minimal protection"
    elif inhibition < 10:
        status = "Moderate protection [OK]"
    elif inhibition < 100:
        status = "Strong protection [OK]"
    else:
        status = "Extreme protection [OK]"

    print(f"{applied_strain:<10.1f} {fiber_strain:<12.3f} {k_cleave:<15.6e} {inhibition:<15.1f}x {status}")

print()
print("=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
print()
print("INTERPRETATION:")
print("-" * 80)
print("[OK] If forces are in NANO-NEWTON range: Physics is coupled to chemistry")
print("[!]  If forces are in FEMTO-NEWTON range: Forces too small (unit scaling issue)")
print()
print("[OK] If inhibition grows with strain: Mechanochemical coupling is ACTIVE")
print("[!]  If inhibition is flat: beta_strain might be zero or chemistry broken")
print()
print("Expected Results (Healthy System):")
print("  - Fiber strain should scale with applied strain (e ~ prestrain + applied)")
print("  - Forces should be ~1e-9 to 1e-7 N (nano-Newton range)")
print("  - Inhibition should grow exponentially: 10x at 23% strain, 100x at 46%")
print("=" * 80)
