"""
Integration test for Core V2 with existing Excel files.

Tests:
1. Load network from Excel
2. Configure parameters
3. Start simulation
4. Advance a few steps
5. Export results in GUI format

Usage:
    python test_core_v2_integration.py
"""

import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.core.fibrinet_core_v2_adapter import create_adapter_from_excel


def test_excel_loading():
    """Test loading and simulating with existing Excel files."""

    # Find a test network
    test_files = [
        os.path.join(_project_root, "test/input_data/TestNetwork.xlsx"),
        os.path.join(_project_root, "test/input_data/Hangman.xlsx"),
        os.path.join(_project_root, "test/input_data/fibrin_network_big.xlsx")
    ]

    test_file = None
    for f in test_files:
        if os.path.exists(f):
            test_file = f
            break

    if test_file is None:
        print("ERROR: No test Excel files found")
        return False

    print(f"Testing with: {test_file}")
    print("=" * 60)

    try:
        # Create adapter
        adapter = create_adapter_from_excel(
            excel_path=test_file,
            plasmin_concentration=1.0,
            time_step=0.01,
            max_time=1.0,  # Short test
            applied_strain=0.1
        )

        print("\n" + "=" * 60)
        print("Running simulation (10 steps)...")
        print("=" * 60)

        # Run a few steps
        for i in range(10):
            continue_sim = adapter.advance_one_batch()

            t = adapter.get_current_time()
            lysis = adapter.get_lysis_fraction()
            n_edges = len(adapter.get_edges())

            print(f"Step {i+1}: t={t:.4f}s, lysis={lysis:.3f}, n_edges={n_edges}")

            if not continue_sim:
                print(f"Terminated: {adapter.termination_reason}")
                break

        print("\n" + "=" * 60)
        print("Export Test")
        print("=" * 60)

        # Test exports
        edges = adapter.get_edges()
        positions = adapter.get_node_positions()
        forces = adapter.get_forces()

        print(f"Exported {len(edges)} edges")
        print(f"Exported {len(positions)} node positions")
        print(f"Exported {len(forces)} fiber forces")

        # Show sample edge
        if edges:
            sample = edges[0]
            print(f"\nSample edge:")
            print(f"  edge_id: {sample.edge_id}")
            print(f"  n_from: {sample.n_from}, n_to: {sample.n_to}")
            print(f"  S (integrity): {sample.S:.3f}")
            print(f"  M (degradation): {sample.M:.3f}")

        # Show sample force
        if forces:
            fid, force = list(forces.items())[0]
            print(f"\nSample force:")
            print(f"  fiber_id: {fid}")
            print(f"  force: {force:.6e} N")

        print("\n" + "=" * 60)
        print("[PASS] Integration test completed successfully")
        print("=" * 60)

        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"[FAIL] Integration test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_excel_loading()
    exit(0 if success else 1)
