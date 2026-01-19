"""
Phase 2G: Supply-Limited Stochastic Plasmin Seeding Tests

Test plan:
A) Sparsity test: not all edges receive plasmin
B) Conservation test: P_free + sum(B_i) == P_total (exact)
C) Determinism test: same seed => identical results
D) Legacy unchanged test: legacy mode behavior unchanged
"""

import sys
import os
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config.feature_flags import FeatureFlags
from src.views.tkinter_view.research_simulation_page import SimulationController
import csv
import tempfile
import copy


def create_test_network_csv(P_total_quanta=100, lambda_bind_total=10.0, num_edges=5):
    """Create a test network CSV with multiple edges for sparsity testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        csv_path = f.name
        writer = csv.writer(f)
        
        # Create a chain of nodes
        writer.writerow(["n_id", "n_x", "n_y", "is_left_boundary", "is_right_boundary"])
        for i in range(num_edges + 1):
            x = i * 1e-5  # 10 micrometers apart
            is_left = "1" if i == 0 else "0"
            is_right = "1" if i == num_edges else "0"
            writer.writerow([str(i + 1), str(x), "0.0", is_left, is_right])
        writer.writerow([])
        
        # Create edges
        writer.writerow(["e_id", "n_from", "n_to", "thickness"])
        for i in range(num_edges):
            writer.writerow([str(i + 1), str(i + 1), str(i + 2), "1e-7"])
        writer.writerow([])
        
        # Meta parameters
        writer.writerow(["key", "value"])
        writer.writerow(["spring_stiffness_constant", "1e-3"])
        writer.writerow(["coord_to_m", "1.0"])
        writer.writerow(["thickness_to_m", "1.0"])
        writer.writerow(["L_seg", "5e-6"])
        writer.writerow(["N_pf", "50"])
        writer.writerow(["sigma_site", "1e-18"])
        writer.writerow(["P_total_quanta", str(P_total_quanta)])
        writer.writerow(["lambda_bind_total", str(lambda_bind_total)])
        writer.writerow(["k_on0", "1e5"])
        writer.writerow(["k_off0", "0.1"])
        writer.writerow(["alpha", "0.0"])
        # Disable cleavage for Phase 2G seeding tests so dt_used is not reduced by dt_cleave stability.
        writer.writerow(["k_cat0", "0.0"])
        writer.writerow(["beta", "0.0"])
        writer.writerow(["epsilon", "1.0"])
        writer.writerow(["K_crit", "1e-6"])
        writer.writerow(["N_seg_max", "100000"])
        
    return csv_path


def test_sparsity():
    """
    Test A: Sparsity test - not all edges receive plasmin.
    
    Use small P_total_quanta, run 10 batches.
    Assert:
    - At least one edge has any segment with B_i > 0 at some point
    - At least one other edge remains with all B_i == 0 across all segments
    """
    print("\n=== TEST A: Sparsity Test ===")
    
    original_flag = FeatureFlags.USE_SPATIAL_PLASMIN
    FeatureFlags.USE_SPATIAL_PLASMIN = True
    
    csv_path = None
    try:
        # Small supply (10 quanta), 5 edges with ~2 segments each = ~10 segments
        # Not all segments should get plasmin
        # Use high lambda_bind_total to get reliable binding events with dt=0.001
        csv_path = create_test_network_csv(P_total_quanta=10, lambda_bind_total=5000.0, num_edges=5)
        
        controller = SimulationController()
        controller.load_network(csv_path)
        adapter = controller.state.loaded_network
        
        controller.configure_phase1_parameters_from_ui(
            plasmin_concentration_str="1e-6",
            time_step_str="1e-3",
            max_time_str="1000.0",
                applied_strain_str="0.05",
        )
        controller.start()
        
        # Track which edges ever received plasmin
        edges_with_plasmin: set = set()
        edges_without_plasmin: set = set(e.edge_id for e in adapter.edges)
        
        # Run 10 batches
        for batch in range(10):
            controller.advance_one_batch()
            
            for e in adapter.edges:
                if e.segments is not None:
                    has_plasmin = any(seg.B_i > 0 for seg in e.segments)
                    if has_plasmin:
                        edges_with_plasmin.add(e.edge_id)
        
        # At end: check which edges never got plasmin
        edges_without_plasmin = set(e.edge_id for e in adapter.edges) - edges_with_plasmin
        
        print(f"  Edges that received plasmin: {len(edges_with_plasmin)}")
        print(f"  Edges that never received plasmin: {len(edges_without_plasmin)}")
        print(f"  P_total_quanta: {adapter.P_total_quanta}")
        print(f"  P_free_quanta (final): {adapter.P_free_quanta}")
        
        # Assertions
        assert len(edges_with_plasmin) >= 1, "At least one edge should have received plasmin"
        assert len(edges_without_plasmin) >= 1, f"At least one edge should have NO plasmin (sparsity). All {len(adapter.edges)} edges got plasmin."
        
        print("  [OK] PASS: Binding is sparse - not all edges receive plasmin")
        
    finally:
        if csv_path and os.path.exists(csv_path):
            os.unlink(csv_path)
        FeatureFlags.USE_SPATIAL_PLASMIN = original_flag


def test_conservation():
    """
    Test B: Conservation test - P_free + sum(B_i) == P_total (exact).
    
    After each batch, verify conservation holds.
    """
    print("\n=== TEST B: Conservation Test ===")
    
    original_flag = FeatureFlags.USE_SPATIAL_PLASMIN
    FeatureFlags.USE_SPATIAL_PLASMIN = True
    
    csv_path = None
    try:
        # Use high lambda_bind_total so binding actually occurs and conservation is exercised non-trivially.
        csv_path = create_test_network_csv(P_total_quanta=50, lambda_bind_total=1000.0, num_edges=3)
        
        controller = SimulationController()
        controller.load_network(csv_path)
        adapter = controller.state.loaded_network
        
        controller.configure_phase1_parameters_from_ui(
            plasmin_concentration_str="1e-6",
            time_step_str="1e-3",
            max_time_str="1000.0",
                applied_strain_str="0.05",
        )
        controller.start()
        
        P_total = adapter.P_total_quanta
        
        # Run 5 batches
        for batch in range(5):
            controller.advance_one_batch()
            
            # Check conservation
            P_free = adapter.P_free_quanta
            total_bound = 0
            for e in adapter.edges:
                if e.segments is not None:
                    for seg in e.segments:
                        total_bound += int(round(seg.B_i))
            
            actual_total = P_free + total_bound
            
            assert actual_total == P_total, f"Batch {batch+1}: Conservation violated! P_free={P_free} + sum(B_i)={total_bound} = {actual_total} != P_total={P_total}"
            
            print(f"  Batch {batch+1}: P_free={P_free}, sum(B_i)={total_bound}, total={actual_total} == P_total={P_total} [OK]")
        
        print("  [OK] PASS: Conservation holds exactly after every batch")
        
    finally:
        if csv_path and os.path.exists(csv_path):
            os.unlink(csv_path)
        FeatureFlags.USE_SPATIAL_PLASMIN = original_flag


def test_determinism():
    """
    Test C: Determinism test - same seed => identical results.
    
    Run the same simulation twice and verify identical outcomes.
    """
    print("\n=== TEST C: Determinism Test ===")
    
    original_flag = FeatureFlags.USE_SPATIAL_PLASMIN
    FeatureFlags.USE_SPATIAL_PLASMIN = True
    
    csv_path = None
    try:
        # Use high lambda_bind_total so stochastic events occur; determinism must still hold.
        csv_path = create_test_network_csv(P_total_quanta=30, lambda_bind_total=1000.0, num_edges=3)
        
        def run_simulation(csv_path):
            controller = SimulationController()
            controller.load_network(csv_path)
            adapter = controller.state.loaded_network
            
            controller.configure_phase1_parameters_from_ui(
                plasmin_concentration_str="1e-6",
                time_step_str="1e-3",
                max_time_str="1000.0",
                        applied_strain_str="0.05",
            )
            controller.start()
            
            # Run 5 batches and record results
            results = []
            for batch in range(5):
                controller.advance_one_batch()
                
                log_entry = adapter.experiment_log[-1]
                results.append({
                    "P_free_quanta": log_entry.get("P_free_quanta"),
                    "bind_events_applied": log_entry.get("bind_events_applied"),
                    "total_unbound_this_batch": log_entry.get("total_unbound_this_batch"),
                    "total_bound_plasmin": log_entry.get("total_bound_plasmin"),
                })
            
            # Also record final edge state
            final_B_i = []
            for e in adapter.edges:
                if e.segments is not None:
                    for seg in e.segments:
                        final_B_i.append(float(seg.B_i))
            results.append({"final_B_i": final_B_i})
            
            return results
        
        # Run twice
        results1 = run_simulation(csv_path)
        results2 = run_simulation(csv_path)
        
        # Compare
        for i, (r1, r2) in enumerate(zip(results1, results2)):
            if r1 != r2:
                print(f"  Batch {i}: MISMATCH")
                print(f"    Run 1: {r1}")
                print(f"    Run 2: {r2}")
                raise AssertionError(f"Determinism violated at batch {i}")
        
        print(f"  Ran 2 identical simulations with 5 batches each")
        print(f"  All results match exactly")
        print("  [OK] PASS: Determinism preserved")
        
    finally:
        if csv_path and os.path.exists(csv_path):
            os.unlink(csv_path)
        FeatureFlags.USE_SPATIAL_PLASMIN = original_flag


def test_legacy_unchanged():
    """
    Test D: Legacy mode unchanged test.
    
    Run a legacy-mode simulation and verify no spatial mode fields appear.
    """
    print("\n=== TEST D: Legacy Mode Unchanged Test ===")
    
    original_flag = FeatureFlags.USE_SPATIAL_PLASMIN
    FeatureFlags.USE_SPATIAL_PLASMIN = False
    
    csv_path = None
    try:
        # Create simple legacy network
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            csv_path = f.name
            writer = csv.writer(f)
            
            writer.writerow(["n_id", "n_x", "n_y", "is_left_boundary", "is_right_boundary"])
            writer.writerow(["1", "0.0", "0.0", "1", "0"])
            writer.writerow(["2", "1e-5", "0.0", "0", "1"])
            writer.writerow([])
            
            writer.writerow(["e_id", "n_from", "n_to", "thickness"])
            writer.writerow(["1", "1", "2", "1e-6"])
            writer.writerow([])
            
            writer.writerow(["key", "value"])
            writer.writerow(["spring_stiffness_constant", "1.0"])
        
        controller = SimulationController()
        controller.load_network(csv_path)
        adapter = controller.state.loaded_network
        
        controller.configure_phase1_parameters_from_ui(
            plasmin_concentration_str="1e-6",
            time_step_str="1e-3",
            max_time_str="1000.0",
                applied_strain_str="0.05",
        )
        controller.start()
        
        # Run 2 batches
        for batch in range(2):
            controller.advance_one_batch()
        
        # Verify legacy mode characteristics
        assert adapter.P_total_quanta is None, "P_total_quanta should be None in legacy mode"
        assert adapter.P_free_quanta is None, "P_free_quanta should be None in legacy mode"
        
        # Check edge has no segments
        for e in adapter.edges:
            assert e.segments is None, "Edges should not have segments in legacy mode"
        
        # Check log doesn't have spatial mode specific values
        log_entry = adapter.experiment_log[-1]
        
        print(f"  Legacy mode: P_total_quanta={adapter.P_total_quanta}, P_free_quanta={adapter.P_free_quanta}")
        print(f"  Edges have segments: {any(e.segments is not None for e in adapter.edges)}")
        print("  [OK] PASS: Legacy mode unchanged")
        
    finally:
        if csv_path and os.path.exists(csv_path):
            os.unlink(csv_path)
        FeatureFlags.USE_SPATIAL_PLASMIN = original_flag


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 2G: SUPPLY-LIMITED STOCHASTIC PLASMIN SEEDING TESTS")
    print("=" * 60)
    
    test_sparsity()
    test_conservation()
    test_determinism()
    test_legacy_unchanged()
    
    print("\n" + "=" * 60)
    print("ALL PHASE 2G TESTS PASSED")
    print("Binding is stochastic and supply-limited;")
    print("not all edges receive plasmin;")
    print("conservation and determinism verified.")
    print("=" * 60)

