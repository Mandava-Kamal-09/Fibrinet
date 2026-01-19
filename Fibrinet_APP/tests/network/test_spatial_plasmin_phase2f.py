"""
Phase 2F: Spatial Mode Hardening + Terminology Consistency Tests

Test plan:
1. "No ruptured keys" test: experiment_log entries in spatial mode contain ONLY cleaved/cleared keys
2. "sigma_ref slack does not terminate" test: sigma_ref==0 (or forces all zero) does not terminate spatial mode
3. "No division by zero in spatial mode" test: run 2-3 batches with sigma_ref==0, assert no exceptions
4. "Segments preserved" test: after a batch, segments still present and updated (B_i and n_i changed as expected)
"""

import sys
import os
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config.feature_flags import FeatureFlags
from src.views.tkinter_view.research_simulation_page import SimulationController
import csv
import tempfile


def test_no_ruptured_keys_in_spatial_mode():
    """Test 1: Verify experiment_log contains only cleaved/cleared keys, not ruptured keys."""
    print("\n=== TEST 1: No 'ruptured' keys in spatial mode logs ===")
    
    # Enable spatial mode
    original_flag = FeatureFlags.USE_SPATIAL_PLASMIN
    FeatureFlags.USE_SPATIAL_PLASMIN = True
    
    csv_path = None
    try:
        # Create test network
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
            writer.writerow(["spring_stiffness_constant", "1e-3"])
            writer.writerow(["coord_to_m", "1.0"])
            writer.writerow(["thickness_to_m", "1.0"])
            writer.writerow(["L_seg", "5e-7"])
            writer.writerow(["N_pf", "50"])
            writer.writerow(["sigma_site", "1e-18"])
            # Phase 2G stochastic seeding: ensure binding happens deterministically even under sigma_ref slack
            writer.writerow(["P_total_quanta", "10"])
            writer.writerow(["lambda_bind_total", "1000000.0"])
            # Keep legacy spatial params (still parsed)
            writer.writerow(["P_bulk", "1e-6"])
            writer.writerow(["k_on0", "1e5"])
            # Disable unbinding + cleavage so dt_used stays base and binding is guaranteed
            writer.writerow(["k_off0", "0.0"])
            writer.writerow(["alpha", "0.0"])
            writer.writerow(["k_cat0", "0.0"])
            writer.writerow(["beta", "0.0"])
            writer.writerow(["epsilon", "1.0"])
            writer.writerow(["K_crit", "1e-6"])
            writer.writerow(["N_seg_max", "100000"])
        
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
        for batch_idx in range(2):
            controller.advance_one_batch()
        
        # Check experiment_log for ruptured keys
        for i, log_entry in enumerate(adapter.experiment_log):
            # Should NOT contain any "ruptured" keys
            for key in log_entry.keys():
                assert "ruptur" not in key.lower(), f"Batch {i}: Found ruptured key '{key}' in log (should use 'cleaved/cleared')"
            
            # Should contain cleaved keys
            assert "cleaved_edges_total" in log_entry, f"Batch {i}: Missing 'cleaved_edges_total' key"
            assert "newly_cleaved" in log_entry, f"Batch {i}: Missing 'newly_cleaved' key"
        
        print("  [OK] PASS: No 'ruptured' keys found in spatial mode logs")
        
    finally:
        if csv_path and os.path.exists(csv_path):
            os.unlink(csv_path)
        FeatureFlags.USE_SPATIAL_PLASMIN = original_flag


def test_sigma_ref_slack_does_not_terminate():
    """Test 2: Verify sigma_ref==0 does not terminate spatial mode."""
    print("\n=== TEST 2: sigma_ref slack does not terminate spatial mode ===")

    # Enable spatial mode
    original_flag = FeatureFlags.USE_SPATIAL_PLASMIN
    FeatureFlags.USE_SPATIAL_PLASMIN = True
    
    csv_path = None
    try:
        # Create test network with zero applied strain (no tension)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            csv_path = f.name
            writer = csv.writer(f)
            
            writer.writerow(["n_id", "n_x", "n_y", "is_left_boundary", "is_right_boundary"])
            writer.writerow(["1", "0.0", "0.0", "1", "0"])
            writer.writerow(["2", "1e-5", "0.0", "0", "1"])
            writer.writerow([])
            
            writer.writerow(["e_id", "n_from", "n_to", "thickness"])
            writer.writerow(["1", "1", "2", "1e-7"])
            writer.writerow([])
            
            writer.writerow(["key", "value"])
            writer.writerow(["spring_stiffness_constant", "1e-3"])
            writer.writerow(["coord_to_m", "1.0"])
            writer.writerow(["thickness_to_m", "1.0"])
            writer.writerow(["L_seg", "5e-7"])
            writer.writerow(["N_pf", "50"])
            writer.writerow(["sigma_site", "1e-18"])
            # Phase 2G stochastic seeding (ensure binding occurs even with sigma_ref slack)
            writer.writerow(["P_total_quanta", "10"])
            writer.writerow(["lambda_bind_total", "100000.0"])
            writer.writerow(["P_bulk", "1e-6"])
            writer.writerow(["k_on0", "1e5"])
            writer.writerow(["k_off0", "0.0"])  # disable unbinding for deterministic binding presence
            writer.writerow(["alpha", "0.0"])
            writer.writerow(["k_cat0", "0.0"])  # disable cleavage so dt_used remains base dt for this test
            writer.writerow(["beta", "0.0"])
            writer.writerow(["epsilon", "1.0"])
            writer.writerow(["K_crit", "1e-6"])
            writer.writerow(["N_seg_max", "100000"])
        
        controller = SimulationController()
        controller.load_network(csv_path)
        adapter = controller.state.loaded_network
        
        # Apply ZERO strain to create slack network (sigma_ref will be 0 or very small)
        controller.configure_phase1_parameters_from_ui(
            plasmin_concentration_str="1e-6",
            time_step_str="1e-3",
            max_time_str="1000.0",
                applied_strain_str="0.0",  # ZERO strain
        )
        controller.start()
        
        # Run 3 batches - should NOT terminate due to sigma_ref
        for batch_idx in range(3):
            controller.advance_one_batch()
        
        # Check that simulation did NOT terminate
        assert adapter.termination_reason is None, f"Spatial mode should not terminate due to sigma_ref, got: {adapter.termination_reason}"
        
        # Check that binding still ran
        # Get first edge, check B_i values
        edge = adapter.edges[0]
        if edge.segments:
            B_i_vals = [seg.B_i for seg in edge.segments]
            # At least some B_i should be > 0 (binding ran)
            assert any(B > 0.0 for B in B_i_vals), f"Binding should have run even with sigma_ref=0"
        
        print("  [OK] PASS: sigma_ref slack does not terminate spatial mode")
        print(f"    Ran 3 batches successfully")
        print(f"    Termination reason: {adapter.termination_reason} (should be None)")
        
    finally:
        if csv_path and os.path.exists(csv_path):
            os.unlink(csv_path)
        FeatureFlags.USE_SPATIAL_PLASMIN = original_flag


def test_no_division_by_zero_in_spatial_mode():
    """Test 3: Verify no division by zero exceptions with sigma_ref==0."""
    print("\n=== TEST 3: No division by zero in spatial mode ===")

    # Enable spatial mode
    original_flag = FeatureFlags.USE_SPATIAL_PLASMIN
    FeatureFlags.USE_SPATIAL_PLASMIN = True
    
    csv_path = None
    try:
        # Create test network
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            csv_path = f.name
            writer = csv.writer(f)
            
            writer.writerow(["n_id", "n_x", "n_y", "is_left_boundary", "is_right_boundary"])
            writer.writerow(["1", "0.0", "0.0", "1", "0"])
            writer.writerow(["2", "1e-5", "0.0", "0", "1"])
            writer.writerow([])
            
            writer.writerow(["e_id", "n_from", "n_to", "thickness"])
            writer.writerow(["1", "1", "2", "1e-7"])
            writer.writerow([])
            
            writer.writerow(["key", "value"])
            writer.writerow(["spring_stiffness_constant", "1e-3"])
            writer.writerow(["coord_to_m", "1.0"])
            writer.writerow(["thickness_to_m", "1.0"])
            writer.writerow(["L_seg", "5e-7"])
            writer.writerow(["N_pf", "50"])
            writer.writerow(["sigma_site", "1e-18"])
            writer.writerow(["P_bulk", "1e-6"])
            writer.writerow(["k_on0", "1e5"])
            writer.writerow(["k_off0", "0.1"])
            writer.writerow(["alpha", "0.0"])
            writer.writerow(["k_cat0", "1e-3"])
            writer.writerow(["beta", "0.0"])
            writer.writerow(["epsilon", "1.0"])
            writer.writerow(["K_crit", "1e-6"])
            writer.writerow(["N_seg_max", "100000"])
        
        controller = SimulationController()
        controller.load_network(csv_path)
        adapter = controller.state.loaded_network
        
        # Apply ZERO strain
        controller.configure_phase1_parameters_from_ui(
            plasmin_concentration_str="1e-6",
            time_step_str="1e-3",
            max_time_str="1000.0",
                applied_strain_str="0.0",
        )
        controller.start()
        
        # Run 3 batches - should NOT raise any exceptions
        exception_raised = False
        try:
            for batch_idx in range(3):
                controller.advance_one_batch()
        except (ZeroDivisionError, FloatingPointError) as e:
            exception_raised = True
            raise AssertionError(f"Division by zero in spatial mode: {e}")
        
        assert not exception_raised, "Should not raise division by zero"
        
        print("  [OK] PASS: No division by zero in spatial mode with sigma_ref=0")
        print(f"    Ran 3 batches without exceptions")
        
    finally:
        if csv_path and os.path.exists(csv_path):
            os.unlink(csv_path)
        FeatureFlags.USE_SPATIAL_PLASMIN = original_flag


def test_segments_preserved_after_batch():
    """Test 4: Verify segments are preserved and updated correctly after batches."""
    print("\n=== TEST 4: Segments preserved after batch ===")

    # Enable spatial mode
    original_flag = FeatureFlags.USE_SPATIAL_PLASMIN
    FeatureFlags.USE_SPATIAL_PLASMIN = True
    
    csv_path = None
    try:
        # Create test network
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            csv_path = f.name
            writer = csv.writer(f)
            
            writer.writerow(["n_id", "n_x", "n_y", "is_left_boundary", "is_right_boundary"])
            writer.writerow(["1", "0.0", "0.0", "1", "0"])
            writer.writerow(["2", "1e-5", "0.0", "0", "1"])
            writer.writerow([])
            
            writer.writerow(["e_id", "n_from", "n_to", "thickness"])
            writer.writerow(["1", "1", "2", "1e-7"])
            writer.writerow([])
            
            writer.writerow(["key", "value"])
            writer.writerow(["spring_stiffness_constant", "1e-3"])
            writer.writerow(["coord_to_m", "1.0"])
            writer.writerow(["thickness_to_m", "1.0"])
            writer.writerow(["L_seg", "5e-7"])
            writer.writerow(["N_pf", "50"])
            writer.writerow(["sigma_site", "1e-18"])
            # Phase 2G stochastic seeding: ensure at least one binding event occurs quickly
            writer.writerow(["P_total_quanta", "10"])
            writer.writerow(["lambda_bind_total", "100000.0"])
            writer.writerow(["P_bulk", "1e-6"])
            writer.writerow(["k_on0", "1e5"])
            # Disable unbinding; keep small cleavage (optional) to allow n_i to be non-increasing
            writer.writerow(["k_off0", "0.0"])
            writer.writerow(["alpha", "0.0"])
            writer.writerow(["k_cat0", "1e-3"])
            writer.writerow(["beta", "0.0"])
            writer.writerow(["epsilon", "1.0"])
            writer.writerow(["K_crit", "1e-6"])
            writer.writerow(["N_seg_max", "100000"])
        
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
        
        # Get initial segments
        edge_initial = adapter.edges[0]
        assert edge_initial.segments is not None, "Segments should be initialized"
        assert len(edge_initial.segments) > 0, "Should have at least one segment"
        
        B_i_initial = edge_initial.segments[0].B_i
        n_i_initial = edge_initial.segments[0].n_i
        
        # Run 2 batches
        for batch_idx in range(2):
            controller.advance_one_batch()
        
        # Check segments are still present
        edge_final = adapter.edges[0]
        assert edge_final.segments is not None, "Segments should be preserved after batches"
        assert len(edge_final.segments) == len(edge_initial.segments), "Segment count should be unchanged"
        
        # Check B_i was updated on at least one segment (stochastic target not necessarily segment 0)
        B_i_final_vals = [float(seg.B_i) for seg in edge_final.segments]
        assert any(B > 0.0 for B in B_i_final_vals), f"Expected at least one segment to bind plasmin, got B_i={B_i_final_vals}"
        # n_i should be non-increasing (cleavage never increases n_i; may be unchanged if no bound on segment 0)
        n_i_final_vals = [float(seg.n_i) for seg in edge_final.segments]
        assert min(n_i_final_vals) <= float(n_i_initial) + 1e-12, f"Expected n_i to be non-increasing, got n_i={n_i_final_vals}"
        
        print("  [OK] PASS: Segments preserved and updated correctly")
        print(f"    Initial seg0: B_i={B_i_initial:.2e}, n_i={n_i_initial:.2f}")
        print(f"    Final: min(B_i)={min(B_i_final_vals):.2e}, max(B_i)={max(B_i_final_vals):.2e}, min(n_i)={min(n_i_final_vals):.6f}")
        
    finally:
        if csv_path and os.path.exists(csv_path):
            os.unlink(csv_path)
        FeatureFlags.USE_SPATIAL_PLASMIN = original_flag


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 2F: SPATIAL MODE HARDENING TESTS")
    print("=" * 60)
    
    test_no_ruptured_keys_in_spatial_mode()
    test_sigma_ref_slack_does_not_terminate()
    test_no_division_by_zero_in_spatial_mode()
    test_segments_preserved_after_batch()
    
    print("\n" + "=" * 60)
    print("ALL PHASE 2F TESTS PASSED")
    print("=" * 60)

