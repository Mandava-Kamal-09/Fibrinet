"""
FibriNet Core V2 - Reproducibility Test
========================================
Verifies deterministic simulation replay with fixed seed.

Critical for Publication:
- Peer reviewers require bit-exact reproducibility
- Same seed -> identical results (down to floating-point precision)
- Validates that RNG implementation is correct

This test:
1. Runs simulation A with seed=0 (default)
2. Runs simulation B with seed=0 (fresh adapter)
3. Compares degradation histories element-by-element
4. Verifies ALL fields match exactly

Expected Result:
- Degradation order: IDENTICAL
- Cleavage times: IDENTICAL (within floating-point tolerance 1e-12)
- Fiber IDs: IDENTICAL
- Strains: IDENTICAL
- Tensions: IDENTICAL

If test fails:
- Non-deterministic RNG detected
- Numerical instability causing divergence
- NOT publication-ready

Usage:
    python test_reproducibility.py

Outputs:
    - Console report: PASS or FAIL
    - If FAIL: Shows first differing event
"""

import sys
import os
import numpy as np

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter

# Configuration
INPUT_FILE = os.path.join(_project_root, "test", "input_data", "fibrin_network_big.xlsx")

# Test parameters (must use same seed)
SEED = 0  # Fixed seed for reproducibility
STRAIN = 0.2
PLASMIN = 1.0
TIME_STEP = 0.01
MAX_TIME = 100.0

FLOAT_TOLERANCE = 1e-12  # Floating-point comparison tolerance


def run_simulation(run_id):
    """Run simulation and return degradation history."""
    print(f"\n{'=' * 80}")
    print(f"SIMULATION RUN {run_id}")
    print("=" * 80)

    # Create fresh adapter
    adapter = CoreV2GUIAdapter()
    adapter.load_from_excel(INPUT_FILE)

    print(f"  Network loaded: {len(adapter._edges_raw)} fibers")
    print(f"  RNG seed: {SEED} (deterministic)")

    # Configure with IDENTICAL parameters
    adapter.configure_parameters(
        plasmin_concentration=PLASMIN,
        time_step=TIME_STEP,
        max_time=MAX_TIME,
        applied_strain=STRAIN
    )

    # Start simulation (uses seed=0 internally)
    adapter.start_simulation()

    # Run to completion
    step_count = 0
    while True:
        running = adapter.advance_one_batch()
        step_count += 1

        if step_count % 100 == 0:
            sim_time = adapter.get_current_time()
            lysis = adapter.get_lysis_fraction()
            print(f"  t={sim_time:6.1f}s | Lysis={lysis*100:5.1f}%", end='\r')

        if not running:
            break

    # Extract results
    sim_time = adapter.get_current_time()
    lysis = adapter.get_lysis_fraction()
    termination = adapter.termination_reason
    history = adapter.simulation.state.degradation_history.copy()

    print(f"\n  Terminated: {termination}")
    print(f"  Final time: {sim_time:.2f}s")
    print(f"  Final lysis: {lysis*100:.1f}%")
    print(f"  Fibers cleaved: {len(history)}")

    return {
        'history': history,
        'final_time': sim_time,
        'final_lysis': lysis,
        'termination': termination,
        'steps': step_count
    }


def compare_histories(hist_a, hist_b):
    """
    Compare two degradation histories element-by-element.

    Returns:
        (is_identical, differences)
    """
    differences = []

    # Check length
    if len(hist_a) != len(hist_b):
        differences.append({
            'field': 'length',
            'event': 'N/A',
            'value_a': len(hist_a),
            'value_b': len(hist_b)
        })
        return False, differences

    # Compare each event
    for i, (event_a, event_b) in enumerate(zip(hist_a, hist_b)):
        # Check all fields
        for field in ['fiber_id', 'order', 'node_i', 'node_j']:
            if event_a[field] != event_b[field]:
                differences.append({
                    'field': field,
                    'event': i + 1,
                    'value_a': event_a[field],
                    'value_b': event_b[field]
                })

        # Check floating-point fields with tolerance
        for field in ['time', 'length', 'strain', 'tension']:
            val_a = event_a.get(field, 0.0)
            val_b = event_b.get(field, 0.0)
            if abs(val_a - val_b) > FLOAT_TOLERANCE:
                differences.append({
                    'field': field,
                    'event': i + 1,
                    'value_a': val_a,
                    'value_b': val_b,
                    'diff': abs(val_a - val_b)
                })

    return len(differences) == 0, differences


def main():
    """Run reproducibility test."""
    print("=" * 80)
    print("FIBRINET CORE V2 - DETERMINISTIC REPRODUCIBILITY TEST")
    print("=" * 80)
    print("\nTest Parameters:")
    print(f"  RNG Seed: {SEED} (deterministic)")
    print(f"  Strain: {STRAIN*100:.0f}%")
    print(f"  Plasmin: {PLASMIN}")
    print(f"  Time step: {TIME_STEP}s")
    print(f"  Max time: {MAX_TIME}s")
    print(f"\nFloating-point tolerance: {FLOAT_TOLERANCE}")

    # Run simulation A
    results_a = run_simulation("A")

    # Run simulation B (fresh adapter, same seed)
    results_b = run_simulation("B")

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    # Compare final state
    print("\nFinal State:")
    print(f"  Termination A: {results_a['termination']}")
    print(f"  Termination B: {results_b['termination']}")
    print(f"  Time A: {results_a['final_time']:.6f}s")
    print(f"  Time B: {results_b['final_time']:.6f}s")
    print(f"  Lysis A: {results_a['final_lysis']:.6f}")
    print(f"  Lysis B: {results_b['final_lysis']:.6f}")

    # Compare degradation histories
    is_identical, differences = compare_histories(results_a['history'], results_b['history'])

    print("\nDegradation History Comparison:")
    print(f"  Events A: {len(results_a['history'])}")
    print(f"  Events B: {len(results_b['history'])}")

    # Final verdict
    print("\n" + "=" * 80)
    print("REPRODUCIBILITY VERDICT")
    print("=" * 80)

    if is_identical:
        print("\n*** PASS: PERFECT REPRODUCIBILITY ***")
        print("\nSimulations A and B produced IDENTICAL results:")
        print("  - Same degradation order")
        print("  - Same cleavage times (within floating-point precision)")
        print("  - Same fiber IDs")
        print("  - Same strains and tensions")
        print("\nDeterministic replay confirmed. Publication-ready.")
        return True
    else:
        print("\n*** FAIL: NON-DETERMINISTIC BEHAVIOR DETECTED ***")
        print(f"\nFound {len(differences)} differences between runs:")

        # Show first 10 differences
        for i, diff in enumerate(differences[:10]):
            if 'diff' in diff:
                print(f"\n  Event {diff['event']}, field '{diff['field']}':")
                print(f"    Run A: {diff['value_a']}")
                print(f"    Run B: {diff['value_b']}")
                print(f"    Difference: {diff['diff']}")
            else:
                print(f"\n  Event {diff['event']}, field '{diff['field']}':")
                print(f"    Run A: {diff['value_a']}")
                print(f"    Run B: {diff['value_b']}")

        if len(differences) > 10:
            print(f"\n  ... and {len(differences) - 10} more differences")

        print("\nPossible causes:")
        print("  - Non-deterministic RNG implementation")
        print("  - Numerical instability causing divergence")
        print("  - Time-dependent or system-dependent behavior")
        print("\nNOT publication-ready until reproducibility is achieved.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
