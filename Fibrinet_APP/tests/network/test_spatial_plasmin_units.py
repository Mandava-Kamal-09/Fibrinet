"""
Phase 1.5 tests: Unit conversion, segment explosion guards, last-segment length.

Tests:
1. Unit conversion (coord_to_m, thickness_to_m) produces correct L and S_i
2. N_seg_max guard catches segment explosion
3. Last segment uses L_i < L_seg when L is not exact multiple
4. Meta key normalization (k_crit vs K_crit)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config.feature_flags import FeatureFlags
from src.views.tkinter_view.research_simulation_page import SimulationController
import math
import tempfile


def test_unit_conversion_micrometers():
    """Test that coord_to_m and thickness_to_m convert units correctly."""
    FeatureFlags.USE_SPATIAL_PLASMIN = True
    
    # Create network with coords in micrometers, thickness in nanometers
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write("n_id,n_x,n_y,is_left_boundary,is_right_boundary\n")
        f.write("1,0,0,1,0\n")        # 0 µm
        f.write("2,10,0,0,0\n")       # 10 µm
        f.write("3,20,0,0,1\n")       # 20 µm
        f.write("\n")
        f.write("e_id,n_from,n_to,thickness\n")
        f.write("10,1,2,500\n")  # 500 nm diameter
        f.write("11,2,3,1000\n") # 1000 nm diameter
        f.write("\n")
        f.write("meta_key,meta_value\n")
        f.write("spring_stiffness_constant,5\n")
        f.write("L_seg,2e-6\n")         # 2 µm segment length (in meters)
        f.write("N_pf,50\n")
        f.write("sigma_site,1e-18\n")
        f.write("coord_to_m,1e-6\n")    # Coords in µm → meters
        f.write("thickness_to_m,1e-9\n") # Thickness in nm → meters
        temp_path = f.name
    
    try:
        c = SimulationController()
        c.load_network(temp_path)
        
        adapter = c.state.loaded_network
        assert len(adapter.edges) == 2
        
        # Check unit conversion was applied
        assert adapter.spatial_plasmin_params["coord_to_m"] == 1e-6
        assert adapter.spatial_plasmin_params["thickness_to_m"] == 1e-9
        
        # Edge 10: L_coord = 10 µm → L = 10e-6 m
        edge10 = adapter.edges[0]
        L_coord_10 = float(edge10.original_rest_length)  # 10 in coord units
        L_m_10 = L_coord_10 * 1e-6  # 10e-6 m
        L_seg = 2e-6
        expected_N_seg_10 = int(math.ceil(L_m_10 / L_seg))  # ceil(10e-6 / 2e-6) = 5
        assert len(edge10.segments) == expected_N_seg_10, f"Edge 10: expected {expected_N_seg_10} segments, got {len(edge10.segments)}"
        
        # Edge 10: D = 500 nm → 500e-9 m
        D_10 = 500e-9
        # First segment: L_i = L_seg = 2e-6
        A_surf_10_first = math.pi * D_10 * L_seg
        expected_S_i_10_first = A_surf_10_first / 1e-18
        actual_S_i_10_first = edge10.segments[0].S_i
        assert abs(actual_S_i_10_first - expected_S_i_10_first) / expected_S_i_10_first < 0.01, \
            f"Edge 10 seg 0: expected S_i={expected_S_i_10_first:.2e}, got {actual_S_i_10_first:.2e}"
        
        # Edge 11: L_coord = 10 µm → L = 10e-6 m, same N_seg
        edge11 = adapter.edges[1]
        assert len(edge11.segments) == expected_N_seg_10
        
        # Edge 11: D = 1000 nm → 1000e-9 m (twice edge 10)
        D_11 = 1000e-9
        A_surf_11_first = math.pi * D_11 * L_seg
        expected_S_i_11_first = A_surf_11_first / 1e-18
        actual_S_i_11_first = edge11.segments[0].S_i
        # S_i should be ~2x edge 10 (diameter doubled)
        ratio = actual_S_i_11_first / actual_S_i_10_first
        assert abs(ratio - 2.0) < 0.01, f"S_i ratio should be ~2.0, got {ratio:.3f}"
        
        print("[OK] Unit conversion test passed")
        print(f"   Edge 10: L={L_m_10:.2e} m, N_seg={len(edge10.segments)}, S_i[0]={actual_S_i_10_first:.2e}")
        print(f"   Edge 11: D={D_11:.2e} m (2x edge 10), S_i[0]={actual_S_i_11_first:.2e} (~2x)")
        
    finally:
        os.unlink(temp_path)
        FeatureFlags.USE_SPATIAL_PLASMIN = False


def test_segment_explosion_guard():
    """Test that N_seg_max catches unrealistic segmentation."""
    FeatureFlags.USE_SPATIAL_PLASMIN = True
    
    # Create network with wrong units (coords in meters but no conversion)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write("n_id,n_x,n_y,is_left_boundary,is_right_boundary\n")
        f.write("1,0,0,1,0\n")
        f.write("2,0.01,0,0,1\n")  # 0.01 m = 10 mm (but treated as meters)
        f.write("\n")
        f.write("e_id,n_from,n_to,thickness\n")
        f.write("10,1,2,1e-6\n")
        f.write("\n")
        f.write("meta_key,meta_value\n")
        f.write("spring_stiffness_constant,5\n")
        f.write("L_seg,1e-6\n")      # 1 µm segment
        f.write("N_pf,50\n")
        f.write("sigma_site,1e-18\n")
        f.write("coord_to_m,1.0\n")  # No conversion (coords already meters)
        f.write("thickness_to_m,1.0\n")
        f.write("N_seg_max,5000\n")  # Low limit to trigger error
        temp_path = f.name
    
    try:
        c = SimulationController()
        try:
            c.load_network(temp_path)
            assert False, "Should have raised ValueError for segment explosion"
        except ValueError as e:
            assert "Segment explosion detected" in str(e)
            assert "N_seg_max" in str(e)
            assert "coord_to_m" in str(e)
            print("[OK] Segment explosion guard test passed")
            print(f"   Error message (excerpt): {str(e)[:200]}...")
    finally:
        os.unlink(temp_path)
        FeatureFlags.USE_SPATIAL_PLASMIN = False


def test_last_segment_length():
    """Test that last segment uses L_i < L_seg when L is not exact multiple."""
    FeatureFlags.USE_SPATIAL_PLASMIN = True
    
    # Create network where L is not exact multiple of L_seg
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write("n_id,n_x,n_y,is_left_boundary,is_right_boundary\n")
        f.write("1,0,0,1,0\n")
        f.write("2,7.5,0,0,1\n")  # 7.5 µm (not multiple of 2 µm)
        f.write("\n")
        f.write("e_id,n_from,n_to,thickness\n")
        f.write("10,1,2,500\n")  # 500 nm diameter
        f.write("\n")
        f.write("meta_key,meta_value\n")
        f.write("spring_stiffness_constant,5\n")
        f.write("L_seg,2e-6\n")      # 2 µm segment
        f.write("N_pf,50\n")
        f.write("sigma_site,1e-18\n")
        f.write("coord_to_m,1e-6\n")
        f.write("thickness_to_m,1e-9\n")
        temp_path = f.name
    
    try:
        c = SimulationController()
        c.load_network(temp_path)
        
        adapter = c.state.loaded_network
        edge = adapter.edges[0]
        
        # L = 7.5e-6 m, L_seg = 2e-6 m
        # N_seg = ceil(7.5 / 2) = 4
        # Segments: [0, 2), [2, 4), [4, 6), [6, 7.5) µm
        # Last segment length: 7.5 - 6 = 1.5 µm
        L = 7.5e-6
        L_seg = 2e-6
        expected_N_seg = int(math.ceil(L / L_seg))  # 4
        assert len(edge.segments) == expected_N_seg, f"Expected {expected_N_seg} segments, got {len(edge.segments)}"
        
        # Check last segment has smaller S_i
        S_i_first = edge.segments[0].S_i
        S_i_last = edge.segments[-1].S_i
        
        # Last segment: L_i = 1.5e-6 m (0.75 * L_seg)
        # S_i should be ~0.75x first segment
        ratio = S_i_last / S_i_first
        expected_ratio = 1.5 / 2.0  # 0.75
        assert abs(ratio - expected_ratio) < 0.01, f"Last segment S_i ratio should be ~{expected_ratio:.2f}, got {ratio:.3f}"
        
        print("[OK] Last segment length test passed")
        print(f"   L={L:.2e} m, L_seg={L_seg:.2e} m, N_seg={len(edge.segments)}")
        print(f"   S_i[0]={S_i_first:.2e}, S_i[last]={S_i_last:.2e} (ratio={ratio:.3f}, expected ~0.75)")
        
    finally:
        os.unlink(temp_path)
        FeatureFlags.USE_SPATIAL_PLASMIN = False


def test_meta_key_normalization():
    """Test that k_crit and K_crit are normalized correctly."""
    FeatureFlags.USE_SPATIAL_PLASMIN = True
    
    # Test 1: k_crit (lowercase) should be accepted
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write("n_id,n_x,n_y,is_left_boundary,is_right_boundary\n")
        f.write("1,0,0,1,0\n")
        f.write("2,10,0,0,1\n")  # 10 µm
        f.write("\n")
        f.write("e_id,n_from,n_to,thickness\n")
        f.write("10,1,2,500\n")  # 500 nm
        f.write("\n")
        f.write("meta_key,meta_value\n")
        f.write("spring_stiffness_constant,5\n")
        f.write("L_seg,2e-6\n")
        f.write("N_pf,50\n")
        f.write("sigma_site,1e-18\n")
        f.write("coord_to_m,1e-6\n")  # µm → m
        f.write("thickness_to_m,1e-9\n")  # nm → m
        f.write("k_crit,1.5e-4\n")  # Lowercase
        temp_path = f.name
    
    try:
        c = SimulationController()
        c.load_network(temp_path)
        adapter = c.state.loaded_network
        assert adapter.spatial_plasmin_params["K_crit"] == 1.5e-4, "k_crit should be normalized to K_crit"
        print("[OK] Meta key normalization test (k_crit) passed")
    finally:
        os.unlink(temp_path)
        FeatureFlags.USE_SPATIAL_PLASMIN = False
    
    # Test 2: Conflicting k_crit and K_crit should raise error
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write("n_id,n_x,n_y,is_left_boundary,is_right_boundary\n")
        f.write("1,0,0,1,0\n")
        f.write("2,10,0,0,1\n")  # 10 µm
        f.write("\n")
        f.write("e_id,n_from,n_to,thickness\n")
        f.write("10,1,2,500\n")  # 500 nm
        f.write("\n")
        f.write("meta_key,meta_value\n")
        f.write("spring_stiffness_constant,5\n")
        f.write("L_seg,2e-6\n")
        f.write("N_pf,50\n")
        f.write("sigma_site,1e-18\n")
        f.write("coord_to_m,1e-6\n")
        f.write("thickness_to_m,1e-9\n")
        f.write("K_crit,1.5e-4\n")  # Uppercase
        f.write("k_crit,2.0e-4\n")  # Lowercase (different value)
        temp_path2 = f.name
    
    try:
        c = SimulationController()
        try:
            c.load_network(temp_path2)
            assert False, "Should have raised ValueError for conflicting K_crit values"
        except ValueError as e:
            assert "Conflicting K_crit values" in str(e)
            print("[OK] Conflicting K_crit detection test passed")
    finally:
        os.unlink(temp_path2)
        FeatureFlags.USE_SPATIAL_PLASMIN = False


def test_default_unit_factors():
    """Test that missing coord_to_m/thickness_to_m default to 1.0."""
    FeatureFlags.USE_SPATIAL_PLASMIN = True
    
    # Create network without explicit unit factors
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write("n_id,n_x,n_y,is_left_boundary,is_right_boundary\n")
        f.write("1,0,0,1,0\n")
        f.write("2,1e-5,0,0,1\n")  # 10 µm in meters
        f.write("\n")
        f.write("e_id,n_from,n_to,thickness\n")
        f.write("10,1,2,1e-6\n")  # 1 µm in meters
        f.write("\n")
        f.write("meta_key,meta_value\n")
        f.write("spring_stiffness_constant,5\n")
        f.write("L_seg,2e-6\n")
        f.write("N_pf,50\n")
        f.write("sigma_site,1e-18\n")
        # No coord_to_m or thickness_to_m
        temp_path = f.name
    
    try:
        c = SimulationController()
        c.load_network(temp_path)
        
        adapter = c.state.loaded_network
        # Should default to 1.0
        assert adapter.spatial_plasmin_params["coord_to_m"] == 1.0
        assert adapter.spatial_plasmin_params["thickness_to_m"] == 1.0
        
        # Should work correctly (coords/thickness already in meters)
        edge = adapter.edges[0]
        L = 1e-5  # meters
        L_seg = 2e-6
        expected_N_seg = int(math.ceil(L / L_seg))  # 5
        assert len(edge.segments) == expected_N_seg
        
        print("[OK] Default unit factors test passed")
        print(f"   coord_to_m={adapter.spatial_plasmin_params['coord_to_m']}, thickness_to_m={adapter.spatial_plasmin_params['thickness_to_m']}")
        
    finally:
        os.unlink(temp_path)
        FeatureFlags.USE_SPATIAL_PLASMIN = False


if __name__ == "__main__":
    print("Running Phase 1.5 unit conversion and guard tests...\n")
    
    test_unit_conversion_micrometers()
    test_segment_explosion_guard()
    test_last_segment_length()
    test_meta_key_normalization()
    test_default_unit_factors()
    
    print("\n[OK] All Phase 1.5 tests passed!")

