"""
Test WLC Force Computation
===========================

Verify that the WLC force formula produces different forces for different strains.
"""

import sys
import os
import numpy as np

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.core.fibrinet_core_v2 import WLCFiber, PhysicalConstants as PC

print("=" * 80)
print("WLC FORCE FORMULA TEST")
print("=" * 80)
print()

# Create a test fiber
L_c = 10e-6  # 10 microns rest length
fiber = WLCFiber(
    fiber_id=1,
    node_i=0,
    node_j=1,
    L_c=L_c,
    xi=PC.xi,
    S=1.0,  # Intact
    x_bell=PC.x_bell,
    k_cat_0=PC.k_cat_0
)

print(f"Test Fiber:")
print(f"  L_c = {fiber.L_c:.6e} m")
print(f"  xi = {fiber.xi:.6e} m")
print(f"  S = {fiber.S}")
print(f"  k_B T / xi = {PC.k_B_T / PC.xi:.6e} N (force scale)")
print()

# Test different strains
test_strains = [0.0, 0.1, 0.23, 0.5, 1.0, 2.0, 5.0, 7.0]

print("FORCE vs STRAIN:")
print("-" * 60)
print(f"{'Strain':>10} {'Length (m)':>15} {'Force (N)':>15}")
print("-" * 60)

forces = []
for strain in test_strains:
    length = L_c * (1.0 + strain)
    force = fiber.compute_force(length)
    forces.append(force)
    print(f"{strain:10.2f} {length:15.6e} {force:15.6e}")

print("-" * 60)
print()

# Check if forces are varying
forces_array = np.array(forces)
print("FORCE STATISTICS:")
print(f"  Min force: {np.min(forces_array):.6e} N")
print(f"  Max force: {np.max(forces_array):.6e} N")
print(f"  Range: {np.max(forces_array) - np.min(forces_array):.6e} N")
print(f"  Std dev: {np.std(forces_array):.6e} N")
print()

if np.std(forces_array) < 1e-20:
    print("[FAIL] All forces are identical! WLC formula is broken!")
else:
    print("[PASS] Forces vary with strain as expected.")
    print()
    print("Force ratio (max/min):", np.max(forces_array) / np.min(forces_array))

print("=" * 80)
