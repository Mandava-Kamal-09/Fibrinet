"""
Phase 2B (v5.0) unit tests: Cleavage kinetics (n_i updates only).

Tests verify:
- n_i decreases when B_i > 0
- No cleavage when B_i = 0
- dt_cleave stability triggers
- Phase separation: S stays 1.0, no edge removal

Author: FibriNet Research Simulation Team
Date: 2026-01-01
Phase: 2B
"""

import sys
import os

# Set feature flag BEFORE any imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config.feature_flags import FeatureFlags
FeatureFlags.USE_SPATIAL_PLASMIN = True

from src.views.tkinter_view.research_simulation_page import SimulationController
import tempfile
import csv


def test_cleavage_decreases_n_i():
    """
    Test A: Cleavage decreases n_i when B_i > 0.
    
    Setup:
    - Spatial mode ON
    - Moderate params: P_bulk > 0, k_cat0 > 0
    - Run 3 batches
    
    Assert:
    - n_i decreases monotonically for at least one segment
    - B_i > 0 (binding occurred)
    """
    print("\n=== TEST A: Cleavage Decreases n_i ===")
    original_flag = FeatureFlags.USE_SPATIAL_PLASMIN
    FeatureFlags.USE_SPATIAL_PLASMIN = True
    
    # Create minimal test network
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        csv_path = f.name
    
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # Nodes
            writer.writerow(["n_id", "n_x", "n_y", "is_left_boundary", "is_right_boundary"])
            writer.writerow(["1", "0.0", "0.0", "1", "0"])
            writer.writerow(["2", "1e-5", "0.0", "0", "1"])
            writer.writerow([])
            
            # Edges
            writer.writerow(["e_id", "n_from", "n_to", "thickness"])
            writer.writerow(["1", "1", "2", "1e-6"])
            writer.writerow([])
            
            # Meta (spatial plasmin params with moderate cleavage rate)
            writer.writerow(["key", "value"])
            writer.writerow(["spring_stiffness_constant", "1e-3"])
            writer.writerow(["coord_to_m", "1.0"])
            writer.writerow(["thickness_to_m", "1.0"])
            writer.writerow(["L_seg", "5e-7"])  # 0.5 micron segments
            writer.writerow(["N_pf", "50"])
            # Use fewer binding sites so dt_used is not vanishingly small (dt_cleave stability depends on S_i)
            writer.writerow(["sigma_site", "1e-16"])
            # Phase 2G binding: supply-limited stochastic seeding
            writer.writerow(["P_total_quanta", "50"])
            writer.writerow(["lambda_bind_total", "1000000.0"])
            # Keep legacy spatial params (still parsed, but binding no longer uses Langmuir)
            writer.writerow(["P_bulk", "1e-6"])
            writer.writerow(["k_on0", "1e5"])
            # Disable unbinding so binding persists and cleavage is visible/deterministic
            writer.writerow(["k_off0", "0.0"])
            writer.writerow(["alpha", "0.0"])  # no tension dependence for simplicity
            writer.writerow(["k_cat0", "1.0"])  # larger cleavage rate for visibility
            writer.writerow(["beta", "0.0"])  # no tension dependence for simplicity
            writer.writerow(["epsilon", "1.0"])
            writer.writerow(["K_crit", "1e-6"])
            writer.writerow(["N_seg_max", "100000"])
        
        # Load network
        controller = SimulationController()
        controller.load_network(csv_path)
        adapter = controller.state.loaded_network
        
        # Configure and start
        controller.configure_phase1_parameters_from_ui(
            plasmin_concentration_str="1e-6",
            time_step_str="1e-3",  # 1 ms timestep
            max_time_str="1000.0",
                applied_strain_str="0.05",
        )
        controller.start()
        
        # Record initial state
        edge0 = adapter.edges[0]
        n_i_history = [edge0.segments[0].n_i]
        B_i_history = [edge0.segments[0].B_i]
        S_history = [edge0.S]
        
        print(f"  Initial: n_i={n_i_history[0]:.6f}, B_i={B_i_history[0]:.3e}, S={S_history[0]}")
        
        # Advance 3 batches
        for batch_idx in range(3):
            controller.advance_one_batch()
            edge0 = adapter.edges[0]
            n_i_history.append(edge0.segments[0].n_i)
            B_i_history.append(edge0.segments[0].B_i)
            S_history.append(edge0.S)
            print(f"  Batch {batch_idx+1}: n_i={n_i_history[-1]:.6f}, B_i={B_i_history[-1]:.3e}, S={S_history[-1]}")
        
        # Assertions
        # A1: B_i must have increased (binding occurred)
        assert B_i_history[-1] > B_i_history[0], f"B_i should increase, got {B_i_history}"
        
        # A2: n_i must be non-increasing (cleavage never increases n_i)
        for i in range(len(n_i_history) - 1):
            assert n_i_history[i+1] <= n_i_history[i], f"n_i should be non-increasing, got {n_i_history}"
        # Must show at least one decrease once binding occurs
        assert min(n_i_history) < n_i_history[0], f"Expected cleavage to reduce n_i at least once, got {n_i_history}"
        
        # A3: Phase 2C: S should equal f_edge = min(n_i/N_pf) (stiffness coupling now implemented)
        # S should be in a reasonable range and decrease with n_i
        for S in S_history:
            assert 0.0 < S <= 1.0, f"S should be in (0, 1], got {S}"
        
        # A3b: S should decrease as n_i decreases (stiffness coupling)
        for i in range(len(S_history) - 1):
            assert S_history[i+1] <= S_history[i], f"S should decrease with n_i, got {S_history}"
        
        # A4: Edge count unchanged (no edge removal)
        assert len(adapter.edges) == 1, f"Edge count should stay 1, got {len(adapter.edges)}"
        
        print("  [OK] PASS: Cleavage decreases n_i correctly")
        
    finally:
        if os.path.exists(csv_path):
            os.unlink(csv_path)
        FeatureFlags.USE_SPATIAL_PLASMIN = original_flag


def test_no_cleavage_when_no_binding():
    """
    Test B: No cleavage when B_i = 0.
    
    Setup:
    - Spatial mode ON
    - P_bulk = 0 so B_i stays 0
    - Run 2 batches
    
    Assert:
    - All n_i remain exactly N_pf
    """
    print("\n=== TEST B: No Cleavage When B_i=0 ===")
    original_flag = FeatureFlags.USE_SPATIAL_PLASMIN
    FeatureFlags.USE_SPATIAL_PLASMIN = True
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        csv_path = f.name
    
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
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
            # Phase 2G: zero supply => no binding events possible
            writer.writerow(["P_total_quanta", "0"])
            writer.writerow(["lambda_bind_total", "2000.0"])
            # Keep legacy spatial params
            writer.writerow(["P_bulk", "1e-20"])
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
        
        controller.configure_phase1_parameters_from_ui(
            plasmin_concentration_str="1e-9",  # tiny but nonzero (P_bulk=0 in meta overrides this)
            time_step_str="1e-3",
            max_time_str="1000.0",
                applied_strain_str="0.05",
        )
        controller.start()
        
        # Check if segments initialized
        edge0 = adapter.edges[0]
        assert edge0.segments is not None, "Segments should be initialized in spatial mode"
        
        # Record initial
        N_pf = 50.0
        n_i_initial = edge0.segments[0].n_i
        B_i_initial = edge0.segments[0].B_i
        
        print(f"  Initial: n_i={n_i_initial}, B_i={B_i_initial:.3e}")
        
        # Run 2 batches
        for batch_idx in range(2):
            controller.advance_one_batch()
            edge0 = adapter.edges[0]
            n_i = edge0.segments[0].n_i
            B_i = edge0.segments[0].B_i
            print(f"  Batch {batch_idx+1}: n_i={n_i:.6f}, B_i={B_i:.3e}")
            
            # Assert B_i stays exactly zero (no supply)
            assert B_i == 0.0, f"B_i should stay 0 with P_total_quanta=0, got {B_i}"
            
            # Assert n_i stays essentially constant (any change should be tiny)
            assert abs(n_i - N_pf) < 1e-10, f"n_i should stay ~{N_pf} when B_i~0, got {n_i}"
        
        print("  [OK] PASS: No cleavage when B_i=0")
        
    finally:
        if os.path.exists(csv_path):
            os.unlink(csv_path)
        FeatureFlags.USE_SPATIAL_PLASMIN = original_flag


def test_dt_cleave_stability():
    """
    Test C: dt_cleave stability triggers.
    
    Setup:
    - Spatial mode ON
    - Very large k_cat0 so dt_cleave_safe < base_dt
    
    Assert:
    - dt_used < base_dt in spatial mode
    """
    print("\n=== TEST C: dt_cleave Stability Triggers ===")
    original_flag = FeatureFlags.USE_SPATIAL_PLASMIN
    FeatureFlags.USE_SPATIAL_PLASMIN = True
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        csv_path = f.name
    
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
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
            # Phase 2G: disable binding supply/events for this dt-only test
            writer.writerow(["P_total_quanta", "0"])
            writer.writerow(["lambda_bind_total", "0.0"])
            writer.writerow(["P_bulk", "1e-6"])
            writer.writerow(["k_on0", "1e5"])
            writer.writerow(["k_off0", "0.1"])
            writer.writerow(["alpha", "0.0"])
            writer.writerow(["k_cat0", "1e3"])  # VERY LARGE cleavage rate
            writer.writerow(["beta", "0.0"])
            writer.writerow(["epsilon", "1.0"])
            writer.writerow(["K_crit", "1e-6"])
            writer.writerow(["N_seg_max", "100000"])
        
        controller = SimulationController()
        controller.load_network(csv_path)
        adapter = controller.state.loaded_network
        
        base_dt = 1e-3
        controller.configure_phase1_parameters_from_ui(
            plasmin_concentration_str="1e-6",
            time_step_str=str(base_dt),
            max_time_str="1000.0",
                applied_strain_str="0.05",
        )
        controller.start()
        
        # Run 1 batch
        controller.advance_one_batch()
        
        # Check dt_used in log
        log_entry = adapter.experiment_log[-1]
        dt_used = log_entry.get("dt_used")
        
        print(f"  base_dt = {base_dt}")
        print(f"  dt_used = {dt_used}")
        
        assert dt_used is not None, "dt_used must be logged in spatial mode"
        assert float(dt_used) < float(base_dt), f"dt_used should be < base_dt due to cleavage stability, got dt_used={dt_used}, base_dt={base_dt}"
        print("  [OK] PASS: dt_cleave stability triggers correctly")
        
    finally:
        if os.path.exists(csv_path):
            os.unlink(csv_path)
        FeatureFlags.USE_SPATIAL_PLASMIN = original_flag


def test_phase_separation_guards():
    """
    Test D: Phase separation guards.
    
    Setup:
    - Spatial mode ON
    - Run multiple batches
    
    Assert:
    - S remains exactly 1.0
    - No edge removal (edge count unchanged)
    - Cleavage observables logged
    """
    print("\n=== TEST D: Phase Separation Guards ===")
    original_flag = FeatureFlags.USE_SPATIAL_PLASMIN
    FeatureFlags.USE_SPATIAL_PLASMIN = True
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        csv_path = f.name
    
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
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
        
        controller.configure_phase1_parameters_from_ui(
            plasmin_concentration_str="1e-6",
            time_step_str="1e-3",
            max_time_str="1000.0",
                applied_strain_str="0.05",
        )
        controller.start()
        
        initial_edge_count = len(adapter.edges)
        
        # Run 3 batches
        for batch_idx in range(3):
            controller.advance_one_batch()
            
            # D1: Phase 2C: S should equal f_edge = min(n_i/N_pf) and may decrease with cleavage
            for e in adapter.edges:
                assert 0.0 < e.S <= 1.0, f"Batch {batch_idx+1}: S should be in (0, 1], got {e.S}"
            
            # D2: Edge count unchanged
            assert len(adapter.edges) == initial_edge_count, f"Batch {batch_idx+1}: Edge count changed"
            
            # D3: Check observables logged
            log_entry = adapter.experiment_log[-1]
            assert "n_min_frac" in log_entry, "n_min_frac should be in log"
            assert "n_mean_frac" in log_entry, "n_mean_frac should be in log"
            assert "total_bound_plasmin" in log_entry, "total_bound_plasmin should be in log"
            
            n_min_frac = log_entry.get("n_min_frac")
            n_mean_frac = log_entry.get("n_mean_frac")
            total_bound_plasmin = log_entry.get("total_bound_plasmin")
            
            # May be None if termination occurred or no segments
            if n_min_frac is None:
                print(f"  Batch {batch_idx+1}: Observables not computed (possibly due to termination)")
                continue
            
            print(f"  Batch {batch_idx+1}:")
            print(f"    n_min_frac = {n_min_frac:.6f}")
            print(f"    n_mean_frac = {n_mean_frac:.6f}")
            print(f"    total_bound_plasmin = {total_bound_plasmin:.3e}")
            
            # D4: Observables should be sensible
            assert 0.0 <= n_min_frac <= 1.0, f"n_min_frac should be in [0,1], got {n_min_frac}"
            assert 0.0 <= n_mean_frac <= 1.0, f"n_mean_frac should be in [0,1], got {n_mean_frac}"
            assert total_bound_plasmin >= 0.0, f"total_bound_plasmin should be >= 0, got {total_bound_plasmin}"
        
        print("  [OK] PASS: Phase separation guards satisfied")
        
    finally:
        if os.path.exists(csv_path):
            os.unlink(csv_path)
        FeatureFlags.USE_SPATIAL_PLASMIN = original_flag


if __name__ == "__main__":
    print("="*60)
    print("PHASE 2B: CLEAVAGE KINETICS UNIT TESTS")
    print("="*60)
    
    test_cleavage_decreases_n_i()
    test_no_cleavage_when_no_binding()
    test_dt_cleave_stability()
    test_phase_separation_guards()
    
    print("\n" + "="*60)
    print("ALL PHASE 2B TESTS PASSED [OK]")
    print("="*60 + "\n")

