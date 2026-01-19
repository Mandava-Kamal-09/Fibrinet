"""
Minimal integration test proving Phase 2G stochastic seeding executes.

This test MUST fail if binding update is skipped.

Author: FibriNet Research Simulation Team
Date: 2026-01-01
Phase: 2G
"""

import sys
import os
import pytest

# Set feature flag BEFORE any imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config.feature_flags import FeatureFlags
FeatureFlags.USE_SPATIAL_PLASMIN = True

from src.views.tkinter_view.research_simulation_page import SimulationController
import tempfile
import csv


def test_binding_kinetics_integration():
    """
    Minimal integration test: stochastic seeding must execute and update B_i.
    
    Setup:
    - Tiny network (2 nodes, 1 edge) with spatial mode ON
    - Supply-limited params (P_total_quanta>0, lambda_bind_total>0)
    - Run advance_one_batch() for 1-3 steps
    
    Assert:
    - Some segment has B_i > 0 (binding executed)
    - Conservation holds: P_free + sum(B_i) == P_total
    - dt_used is logged and equals base dt when cleavage is disabled (k_cat0=0)
    
    This test proves the binding block executes inside advance_one_batch().
    """
    print("\n=== Integration Test: Binding Kinetics Execution ===")
    
    # Create minimal test network
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        csv_path = f.name
    
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # Table 0: Nodes
            writer.writerow(["n_id", "n_x", "n_y", "is_left_boundary", "is_right_boundary"])
            writer.writerow(["1", "0.0", "0.0", "1", "0"])
            writer.writerow(["2", "1.0", "0.0", "0", "1"])
            writer.writerow([])
            
            # Table 1: Edges
            writer.writerow(["e_id", "n_from", "n_to", "thickness"])
            writer.writerow(["1", "1", "2", "1e-6"])
            writer.writerow([])
            
            # Table 2: Meta (spatial plasmin params)
            writer.writerow(["key", "value"])
            writer.writerow(["spring_stiffness_constant", "1e-3"])
            writer.writerow(["coord_to_m", "1e-5"])  # coords in 10-micron units
            writer.writerow(["thickness_to_m", "1.0"])  # thickness already in meters
            writer.writerow(["L_seg", "5e-7"])  # 0.5 micron segments
            writer.writerow(["N_pf", "50"])
            writer.writerow(["sigma_site", "1e-18"])
            # Phase 2G stochastic seeding
            writer.writerow(["P_total_quanta", "10"])
            writer.writerow(["lambda_bind_total", "1000000.0"])
            # Keep legacy spatial params (still parsed)
            writer.writerow(["P_bulk", "1e-6"])
            writer.writerow(["k_on0", "1e5"])
            # Disable unbinding + cleavage for deterministic, fast binding-only integration check
            writer.writerow(["k_off0", "0.0"])
            writer.writerow(["alpha", "0.0"])
            writer.writerow(["k_cat0", "0.0"])
            writer.writerow(["beta", "0.0"])
            writer.writerow(["K_crit", "1e-6"])
            writer.writerow(["N_seg_max", "100000"])
        
        # Load network
        controller = SimulationController()
        controller.load_network(csv_path)
        adapter = controller.state.loaded_network
        
        # Verify spatial params loaded
        assert adapter.spatial_plasmin_params is not None, "Spatial params should be loaded"
        assert adapter.spatial_plasmin_params.get("P_total_quanta", 0) > 0, "P_total_quanta should be > 0"
        
        # Verify segments initialized
        edge0 = adapter.edges[0]
        assert edge0.segments is not None, "Edge should have segments"
        assert len(edge0.segments) > 0, "Edge should have at least one segment"
        
        # Record initial state
        B_i_initial = edge0.segments[0].B_i
        S_initial = edge0.S
        
        print(f"  Initial state:")
        print(f"    B_i[0] = {B_i_initial:.6e}")
        print(f"    S = {S_initial}")
        print(f"    Num segments = {len(edge0.segments)}")
        
        assert B_i_initial == 0.0, "B_i should start at 0"
        assert S_initial == 1.0, "S should be 1.0 (fully intact)"
        
        # Configure and start
        controller.configure_phase1_parameters_from_ui(
            plasmin_concentration_str="1e-6",
            time_step_str="1e-4",  # 0.1 ms timestep
            max_time_str="1000.0",
            applied_strain_str="0.05",
        )
        controller.start()
        
        # Advance 3 batches
        for batch_idx in range(3):
            controller.advance_one_batch()
            edge0 = adapter.edges[0]
            B_i = edge0.segments[0].B_i
            S = edge0.S
            print(f"  After batch {batch_idx+1}:")
            print(f"    B_i[0] = {B_i:.6e}")
            print(f"    S = {S}")
        
        # Final assertions
        edge0 = adapter.edges[0]
        # Final assertions
        edge0 = adapter.edges[0]
        B_vals = [float(seg.B_i) for seg in (edge0.segments or [])]
        B_i_final = float(edge0.segments[0].B_i) if edge0.segments else 0.0
        S_final = edge0.S
        
        # CRITICAL: some segment must have B_i > 0 (proves seeding executed)
        assert any(B > 0.0 for B in B_vals), f"Expected at least one bound quantum, got B_vals={B_vals}"
        # Conservation: P_free + sum(B_i) == P_total
        P_total = int(adapter.P_total_quanta)
        P_free = int(adapter.P_free_quanta)
        total_bound = sum(int(round(B)) for B in B_vals)
        assert P_free + total_bound == P_total, f"Conservation violated: P_free={P_free}, sum(B_i)={total_bound}, P_total={P_total}"
        
        # Check no edges fully cleaved yet (no edge removal in Phase 2C)
        cleaved_edges_total = sum(1 for e in adapter.edges if float(e.S) <= 0.0)
        assert cleaved_edges_total == 0, f"No edges should be fully cleaved, got {cleaved_edges_total}"
        
        # Check dt_used in experiment log
        log_entry = adapter.experiment_log[-1]
        dt_used = log_entry.get("dt_used")
        base_dt = 1e-4
        assert dt_used is not None, "dt_used should be logged"
        assert abs(float(dt_used) - float(base_dt)) <= 1e-15, f"Expected dt_used==base_dt when k_cat0=0, got dt_used={dt_used}"
        
        print("\n  PASS: Stochastic seeding executed successfully")
        print(f"  Example segment B_i[0]: {B_i_initial:.6e} -> {B_i_final:.6e}")
        print(f"  P_free={P_free}, total_bound={total_bound}, P_total={P_total}")
        print(f"  dt_used = {dt_used:.6e}")
        
    finally:
        if os.path.exists(csv_path):
            os.unlink(csv_path)
        FeatureFlags.USE_SPATIAL_PLASMIN = False  # Reset flag (script hygiene)


if __name__ == "__main__":
    test_binding_kinetics_integration()
    print("\n" + "="*60)
    print("INTEGRATION TEST PASSED")
    print("="*60 + "\n")

