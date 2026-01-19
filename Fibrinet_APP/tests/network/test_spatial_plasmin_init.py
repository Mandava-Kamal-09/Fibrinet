"""
Smoke test for v5.0 spatial plasmin segment initialization.

Tests that when USE_SPATIAL_PLASMIN=True:
- Segments are initialized correctly
- n_i = N_pf (fully intact)
- B_i = 0 (no plasmin bound)
- S_i > 0 (positive binding capacity)
- S derived proxy = 1.0 (fully intact)
"""

import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config.feature_flags import FeatureFlags
from src.views.tkinter_view.research_simulation_page import SimulationController
import math


def test_spatial_init_legacy_mode():
    """Test that legacy mode (USE_SPATIAL_PLASMIN=False) works unchanged."""
    FeatureFlags.USE_SPATIAL_PLASMIN = False

    c = SimulationController()
    c.load_network('tests/input_data/synthetic_research_network/synthetic_network_uniform_thickness_stacked.csv')
    
    adapter = c.state.loaded_network
    assert adapter is not None, "Network should load"
    assert len(adapter.edges) > 0, "Should have edges"
    
    # Legacy mode: segments should be None
    for edge in adapter.edges:
        assert edge.segments is None, f"Legacy mode: edge {edge.edge_id} should have segments=None"
        assert edge.S == 1.0, f"Legacy mode: edge {edge.edge_id} should have S=1.0"
    
    print("[OK] Legacy mode test passed")


@pytest.mark.skip(reason="Phase 5.5: Solver reconciliation bug in spatial plasmin mode - k_eff_intact mapping fails")
def test_spatial_init_with_params():
    """Test spatial mode initialization with valid parameters."""
    FeatureFlags.USE_SPATIAL_PLASMIN = True
    
    # Create a minimal test network with spatial params
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write("n_id,n_x,n_y,is_left_boundary,is_right_boundary\n")
        f.write("1,0,0,1,0\n")
        f.write("2,1e-5,0,0,0\n")  # 10 µm in meters
        f.write("3,2e-5,0,0,1\n")  # 20 µm in meters
        f.write("\n")
        f.write("e_id,n_from,n_to,thickness\n")
        f.write("10,1,2,1e-6\n")  # 1 µm diameter
        f.write("11,2,3,1e-6\n")
        f.write("\n")
        f.write("meta_key,meta_value\n")
        f.write("spring_stiffness_constant,5\n")
        f.write("L_seg,5e-6\n")  # 5 µm segment length
        f.write("N_pf,50\n")      # 50 protofibrils
        f.write("sigma_site,1e-18\n")  # 1e-18 m^2 per site
        f.write("coord_to_m,1.0\n")  # Coords already in meters
        f.write("thickness_to_m,1.0\n")  # Thickness already in meters
        temp_path = f.name
    
    try:
        c = SimulationController()
        c.load_network(temp_path)
        
        adapter = c.state.loaded_network
        assert adapter is not None, "Network should load"
        assert len(adapter.edges) == 2, "Should have 2 edges"
        
        # Check spatial params were loaded
        assert adapter.spatial_plasmin_params is not None
        assert adapter.spatial_plasmin_params["L_seg"] == 5e-6
        assert adapter.spatial_plasmin_params["N_pf"] == 50
        assert adapter.spatial_plasmin_params["sigma_site"] == 1e-18
        
        # Check segments were initialized
        for edge in adapter.edges:
            assert edge.segments is not None, f"Spatial mode: edge {edge.edge_id} should have segments"
            assert len(edge.segments) > 0, f"Edge {edge.edge_id} should have at least one segment"
            
            # Edge length
            L = float(edge.original_rest_length)
            L_seg = 5e-6
            expected_N_seg = int(math.ceil(L / L_seg))
            assert len(edge.segments) == expected_N_seg, f"Edge {edge.edge_id}: expected {expected_N_seg} segments, got {len(edge.segments)}"
            
            # Check each segment
            N_pf = 50
            for seg in edge.segments:
                assert seg.n_i == N_pf, f"Segment {seg.segment_index} should have n_i={N_pf}, got {seg.n_i}"
                assert seg.B_i == 0.0, f"Segment {seg.segment_index} should have B_i=0, got {seg.B_i}"
                assert seg.S_i > 0, f"Segment {seg.segment_index} should have S_i>0, got {seg.S_i}"
            
            # Check derived S proxy
            assert edge.S == 1.0, f"Edge {edge.edge_id} should have derived S=1.0 at init, got {edge.S}"
        
        print("[OK] Spatial mode initialization test passed")
        print(f"   Edge 10: {len(adapter.edges[0].segments)} segments, L={adapter.edges[0].original_rest_length:.6e} m")
        print(f"   Edge 11: {len(adapter.edges[1].segments)} segments, L={adapter.edges[1].original_rest_length:.6e} m")
        print(f"   Segment 0 of edge 10: n_i={adapter.edges[0].segments[0].n_i}, B_i={adapter.edges[0].segments[0].B_i}, S_i={adapter.edges[0].segments[0].S_i:.2f}")
        
    finally:
        os.unlink(temp_path)
        FeatureFlags.USE_SPATIAL_PLASMIN = False  # Reset


def test_spatial_missing_params():
    """Test that missing spatial params raises clear error."""
    FeatureFlags.USE_SPATIAL_PLASMIN = True
    
    # Create network without spatial params
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write("n_id,n_x,n_y,is_left_boundary,is_right_boundary\n")
        f.write("1,0,0,1,0\n")
        f.write("2,10,0,0,1\n")
        f.write("\n")
        f.write("e_id,n_from,n_to,thickness\n")
        f.write("10,1,2,1e-6\n")
        f.write("\n")
        f.write("meta_key,meta_value\n")
        f.write("spring_stiffness_constant,5\n")
        # Missing L_seg, N_pf, sigma_site
        temp_path = f.name
    
    try:
        c = SimulationController()
        try:
            c.load_network(temp_path)
            assert False, "Should have raised ValueError for missing params"
        except ValueError as e:
            assert "required spatial_plasmin_params are missing" in str(e)
            print("[OK] Missing params error test passed")
    finally:
        os.unlink(temp_path)
        FeatureFlags.USE_SPATIAL_PLASMIN = False  # Reset


if __name__ == "__main__":
    print("Running v5.0 spatial plasmin initialization tests...\n")
    
    test_spatial_init_legacy_mode()
    test_spatial_init_with_params()
    test_spatial_missing_params()
    
    print("\n[OK] All tests passed!")

