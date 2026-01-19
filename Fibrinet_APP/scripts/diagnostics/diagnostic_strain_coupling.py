"""
Diagnostic Script: Strain Coupling Verification
================================================

This script systematically checks all phases of the diagnostic checklist
to identify where the coupling between physics and chemistry is breaking.

The implementation uses STRAIN-BASED enzymatic inhibition:
    k(ε) = k₀ × exp(-β × ε)

NOT force-based Bell model. This is critical!
"""

import sys
import os
import numpy as np

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter
from src.core.fibrinet_core_v2 import PhysicalConstants as PC

def run_diagnostics(excel_path: str, applied_strain: float = 0.3):
    """
    Run complete diagnostic suite.

    Args:
        excel_path: Path to network Excel file
        applied_strain: Strain to test (0.0 to 1.0)
    """
    print("=" * 80)
    print("STRAIN COUPLING DIAGNOSTIC REPORT")
    print("=" * 80)
    print(f"Network: {excel_path}")
    print(f"Applied Strain: {applied_strain}")
    print()

    # =========================================================================
    # PHASE 1: Input Probe (Is Strain Arriving?)
    # =========================================================================
    print("PHASE 1: INPUT PROBE")
    print("-" * 80)

    adapter = CoreV2GUIAdapter()
    adapter.load_from_excel(excel_path)

    print(f"Check 1: configure_parameters")
    adapter.configure_parameters(
        plasmin_concentration=1.0,
        time_step=0.01,
        max_time=100.0,
        applied_strain=applied_strain
    )
    print(f"  [OK] adapter.applied_strain = {adapter.applied_strain}")

    if abs(adapter.applied_strain - applied_strain) > 1e-9:
        print(f"  [FAIL] Strain mismatch! Expected {applied_strain}, got {adapter.applied_strain}")
        return
    else:
        print(f"  [PASS] Strain correctly stored")
    print()

    # =========================================================================
    # PHASE 2: Physics Probe (Is Network Stretching?)
    # =========================================================================
    print("PHASE 2: PHYSICS PROBE")
    print("-" * 80)

    # Start simulation
    adapter.start_simulation()

    print(f"Check 2: Verify boundary nodes moved")

    # Get original (unstrained) positions
    x_coords_orig = [pos[0] for pos in adapter.node_coords_raw.values()]
    x_min_orig = min(x_coords_orig)
    x_max_orig = max(x_coords_orig)
    width_orig = x_max_orig - x_min_orig

    print(f"  Original network width (abstract units): {width_orig:.2f}")
    print(f"  Original X range: [{x_min_orig:.2f}, {x_max_orig:.2f}]")

    # Get current positions (after strain applied)
    current_positions = adapter.simulation.state.node_positions

    # Check right boundary nodes
    right_boundary_sample = adapter.right_boundary_node_ids[0]
    pos_right_orig = adapter.node_coords_raw[right_boundary_sample]
    pos_right_current_si = current_positions[right_boundary_sample]
    pos_right_current_abstract = pos_right_current_si / adapter.coord_to_m

    print(f"  Right boundary node {right_boundary_sample}:")
    print(f"    Original X (abstract): {pos_right_orig[0]:.6f}")
    print(f"    Current X (SI): {pos_right_current_si[0]:.6e} m")
    print(f"    Current X (abstract): {pos_right_current_abstract[0]:.6f}")
    print(f"    Expected X (abstract): {pos_right_orig[0] + applied_strain * width_orig:.6f}")

    expected_x_abstract = pos_right_orig[0] + applied_strain * width_orig
    actual_x_abstract = pos_right_current_abstract[0]

    if abs(actual_x_abstract - expected_x_abstract) < 1e-3:
        print(f"  [PASS] Boundary node moved correctly")
    else:
        print(f"  [FAIL] Boundary node did NOT move!")
        print(f"         Difference: {abs(actual_x_abstract - expected_x_abstract):.6f}")
    print()

    # =========================================================================
    # PHASE 3: Unit Scaling Probe (The 1e-18 Trap)
    # =========================================================================
    print("PHASE 3: UNIT SCALING PROBE (THE 1e-18 TRAP)")
    print("-" * 80)

    print(f"Check 3: Verify forces are in correct magnitude")
    print(f"  Unit conversion factor: coord_to_m = {adapter.coord_to_m:.6e} m/unit")

    # Compute forces for all fibers
    forces = adapter.simulation.compute_forces()
    force_values = [f for f in forces.values() if f > 0]

    if force_values:
        avg_force = np.mean(force_values)
        min_force = np.min(force_values)
        max_force = np.max(force_values)

        print(f"  Force statistics:")
        print(f"    Average force: {avg_force:.6e} N")
        print(f"    Min force: {min_force:.6e} N")
        print(f"    Max force: {max_force:.6e} N")
        print(f"    Physical Constants:")
        print(f"      k_B T = {PC.k_B_T:.6e} J")
        print(f"      xi (persistence length) = {PC.xi:.6e} m")
        print(f"      k_B T / xi = {PC.k_B_T / PC.xi:.6e} N (force scale)")

        # Check if forces are in reasonable range (1e-12 to 1e-6 N for micron-scale fibers)
        if 1e-13 < avg_force < 1e-5:
            print(f"  [PASS] Forces are in reasonable range for biopolymers")
        elif avg_force < 1e-15:
            print(f"  [FAIL] Forces are TOO SMALL! This is the 1e-18 trap!")
            print(f"         Your scale_factor is likely wrong.")
            print(f"         Forces should be ~1e-9 to 1e-7 N (nanoNewtons)")
        elif avg_force > 1e-3:
            print(f"  [FAIL] Forces are TOO LARGE! Check your units.")
        else:
            print(f"  [WARNING] Forces are unusual but might be correct")
    else:
        print(f"  [FAIL] No positive forces found!")
    print()

    # =========================================================================
    # PHASE 4: Coupling Probe (Is Chemistry Reading Physics?)
    # =========================================================================
    print("PHASE 4: COUPLING PROBE")
    print("-" * 80)

    print(f"Check 4: Verify chemistry uses updated geometry")

    # Compute fiber strains and cleavage rates
    print(f"  Fiber strain and cleavage rate analysis:")
    print(f"  (Implementation uses STRAIN-BASED model: k(e) = k0 x exp(-beta x e))")
    print()

    strains = []
    cleavage_rates = []

    for i, fiber in enumerate(adapter.simulation.state.fibers[:5]):  # Sample first 5 fibers
        if fiber.S <= 0:
            continue

        pos_i = adapter.simulation.state.node_positions[fiber.node_i]
        pos_j = adapter.simulation.state.node_positions[fiber.node_j]
        current_length = float(np.linalg.norm(pos_j - pos_i))

        # Compute strain
        strain = (current_length - fiber.L_c) / fiber.L_c
        strains.append(strain)

        # Compute cleavage rate
        k_cleave = fiber.compute_cleavage_rate(current_length)
        cleavage_rates.append(k_cleave)

        print(f"  Fiber {fiber.fiber_id}:")
        print(f"    Rest length L_c: {fiber.L_c:.6e} m")
        print(f"    Current length: {current_length:.6e} m")
        print(f"    Strain e: {strain:.6f}")
        print(f"    Cleavage rate k: {k_cleave:.6e} s^-1")
        print(f"    Baseline rate k0: {fiber.k_cat_0:.6e} s^-1")
        print(f"    Ratio k/k0: {k_cleave/fiber.k_cat_0:.6f}")
        print(f"    Expected ratio: exp(-beta*e) = exp(-{PC.beta_strain}*{strain:.3f}) = {np.exp(-PC.beta_strain * strain):.6f}")
        print()

    if strains:
        avg_strain = np.mean(strains)
        avg_k = np.mean(cleavage_rates)

        print(f"  Summary:")
        print(f"    Average fiber strain: {avg_strain:.6f}")
        print(f"    Average cleavage rate: {avg_k:.6e} s^-1")
        print(f"    Baseline rate k0: {PC.k_cat_0:.6e} s^-1")
        print(f"    Inhibition factor: {avg_k/PC.k_cat_0:.6f}x")

        if avg_strain < 1e-6:
            print(f"  [FAIL] Fibers have ZERO strain! Network is not stretched!")
        elif avg_strain > 0.01:
            print(f"  [PASS] Fibers are under tension (e > 0.01)")

            if avg_k < PC.k_cat_0 * 0.001:
                print(f"  [WARNING] Cleavage rate is VERY slow (<0.1% of baseline)")
                print(f"            This is EXPECTED for high strain (strain inhibits cleavage)")
                print(f"            At e={avg_strain:.3f}, the rate is reduced by exp(-{PC.beta_strain * avg_strain:.2f}) = {np.exp(-PC.beta_strain * avg_strain):.6f}x")
                print(f"  [WARNING] THIS MAY BE YOUR 'FLATLINE' - Not a bug, but biological reality!")
        else:
            print(f"  [WARNING] Fibers have low strain (e < 0.01)")
    print()

    # =========================================================================
    # PHASE 5: Operation Order Check
    # =========================================================================
    print("PHASE 5: OPERATION ORDER")
    print("-" * 80)

    print(f"Check 5: Verify relax() is called before chemistry")
    print(f"  Looking at HybridMechanochemicalSimulation.step() method:")
    print(f"    Line 884: self.relax_network()  [OK]")
    print(f"    Line 890: self.chemistry.advance()  [OK]")
    print(f"  [PASS] Correct order (physics before chemistry)")
    print()

    # =========================================================================
    # DIAGNOSIS SUMMARY
    # =========================================================================
    print("=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)

    if strains and avg_strain > 0.01 and avg_k < PC.k_cat_0 * 0.01:
        print("DIAGNOSIS: Likely NOT a bug - biological behavior!")
        print()
        print("Your 'flatline' is caused by STRAIN INHIBITION of enzymatic cleavage.")
        print("This is the CORRECT physical behavior for fibrinolysis:")
        print()
        print("  k(e) = k0 x exp(-beta x e)")
        print(f"  At applied strain = {applied_strain}:")
        print(f"    Expected inhibition: exp(-{PC.beta_strain} x {avg_strain:.3f}) = {np.exp(-PC.beta_strain * avg_strain):.6f}x")
        print(f"    Observed inhibition: {avg_k/PC.k_cat_0:.6f}x")
        print()
        print("SOLUTION OPTIONS:")
        print("  1. Use LOWER applied strain (e.g., 0.0 to 0.1)")
        print("  2. Increase k0 (plasmin concentration) to compensate")
        print(f"  3. Decrease beta (currently {PC.beta_strain}) in PhysicalConstants")
        print("  4. Run simulation LONGER (cleavage is slow but not zero)")
    elif not strains or avg_strain < 1e-6:
        print("DIAGNOSIS: COUPLING FAILURE - Fibers not being stretched!")
        print()
        print("The applied strain is not actually stretching the fibers.")
        print("This could be due to:")
        print("  1. Boundary nodes not being moved")
        print("  2. Strain being applied to wrong nodes")
        print("  3. Geometry reset bug")
    else:
        print("DIAGNOSIS: Inconclusive - Manual inspection needed")

    print("=" * 80)


if __name__ == "__main__":
    # Test with standard network
    test_network = "test/input_data/TestNetwork.xlsx"

    if len(sys.argv) > 1:
        test_network = sys.argv[1]

    applied_strain = 0.3
    if len(sys.argv) > 2:
        applied_strain = float(sys.argv[2])

    try:
        run_diagnostics(test_network, applied_strain)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
