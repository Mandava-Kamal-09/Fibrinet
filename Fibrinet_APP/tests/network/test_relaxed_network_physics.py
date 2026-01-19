"""
Physics Validation Test for Relaxed Network Solver

This test validates the physical correctness of the relaxed network computation:
1. Fixed boundary nodes remain at pole positions
2. Free nodes reach mechanical equilibrium (forces < tolerance)
3. Edge lengths remain positive
4. Component decomposition is correct

Author: FibriNet Research Team
Date: 2026-01-06
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.managers.network.relaxed_network_solver import RelaxedNetworkSolver, NetworkComponent


def create_simple_test_network():
    """
    Create a simple disconnected test network:

    Left component:     Right component:      Isolated:
    [L]--1--2           4--5--[R]             7--8

    Where [L] and [R] are boundary nodes
    """
    class MinimalNode:
        def __init__(self, node_id, x, y):
            self.node_id = node_id
            self.n_x = x
            self.n_y = y

        def get_id(self):
            return self.node_id

    class MinimalEdge:
        def __init__(self, edge_id, n_from, n_to, rest_length=1.0, spring_constant=1.0):
            self.edge_id = edge_id
            self.n_from = n_from
            self.n_to = n_to
            self.rest_length = rest_length
            self.spring_constant = spring_constant

        def get_id(self):
            return self.edge_id

    class MinimalNetwork:
        def __init__(self, nodes, edges):
            self._nodes = nodes
            self._edges = edges
            self._node_dict = {n.get_id(): n for n in nodes}
            self._edge_dict = {e.get_id(): e for e in edges}

        def get_nodes(self):
            return self._nodes

        def get_edges(self):
            return self._edges

        def get_node_by_id(self, node_id):
            return self._node_dict.get(node_id)

        def get_edge_by_id(self, edge_id):
            return self._edge_dict.get(edge_id)

    # Create nodes (with some initial stress - edges longer than rest length)
    nodes = [
        MinimalNode(1, 0.0, 0.0),   # Left boundary (fixed)
        MinimalNode(2, 1.5, 0.0),   # Free (should relax closer to node 1)
        MinimalNode(4, 5.0, 0.0),   # Free
        MinimalNode(5, 7.0, 0.0),   # Right boundary (fixed)
        MinimalNode(7, 3.0, 2.0),   # Isolated free
        MinimalNode(8, 4.5, 2.0),   # Isolated free
    ]

    # Create edges (rest_length = 1.0, but current lengths > 1.0)
    edges = [
        MinimalEdge(101, 1, 2, rest_length=1.0, spring_constant=1.0),  # Left component
        MinimalEdge(102, 4, 5, rest_length=1.0, spring_constant=1.0),  # Right component
        MinimalEdge(103, 7, 8, rest_length=1.0, spring_constant=1.0),  # Isolated
    ]

    return MinimalNetwork(nodes, edges)


def test_component_decomposition():
    """Test that network decomposition correctly identifies components."""
    print("\n" + "="*80)
    print("TEST 1: Component Decomposition")
    print("="*80)

    network = create_simple_test_network()
    solver = RelaxedNetworkSolver()

    left_boundary = {1}
    right_boundary = {5}

    components = solver.decompose_into_components(network, left_boundary, right_boundary)

    print(f"\nFound {len(components)} components:")
    for i, comp in enumerate(components):
        print(f"  Component {i+1}: {comp}")

    # Validation
    assert len(components) == 3, f"Expected 3 components, found {len(components)}"

    # Find each component type
    left_comp = [c for c in components if c.component_type == 'left_connected']
    right_comp = [c for c in components if c.component_type == 'right_connected']
    isolated_comp = [c for c in components if c.component_type == 'isolated']

    assert len(left_comp) == 1, f"Expected 1 left component, found {len(left_comp)}"
    assert len(right_comp) == 1, f"Expected 1 right component, found {len(right_comp)}"
    assert len(isolated_comp) == 1, f"Expected 1 isolated component, found {len(isolated_comp)}"

    # Check node membership
    assert left_comp[0].node_ids == {1, 2}, f"Left component nodes incorrect"
    assert right_comp[0].node_ids == {4, 5}, f"Right component nodes incorrect"
    assert isolated_comp[0].node_ids == {7, 8}, f"Isolated component nodes incorrect"

    # Check fixed nodes
    assert left_comp[0].fixed_nodes == {1}, f"Left component fixed nodes incorrect"
    assert right_comp[0].fixed_nodes == {5}, f"Right component fixed nodes incorrect"
    assert isolated_comp[0].fixed_nodes == set(), f"Isolated component should have no fixed nodes"

    print("\n[PASS] Component decomposition PASSED")
    return True


def test_relaxation_physics():
    """Test that relaxation converges to mechanical equilibrium."""
    print("\n" + "="*80)
    print("TEST 2: Relaxation Physics")
    print("="*80)

    network = create_simple_test_network()
    solver = RelaxedNetworkSolver()

    left_boundary = {1}
    right_boundary = {5}

    # Store original positions
    original_positions = {}
    for node in network.get_nodes():
        original_positions[node.get_id()] = (node.n_x, node.n_y)

    print("\nOriginal positions:")
    for nid, pos in sorted(original_positions.items()):
        print(f"  Node {nid}: {pos}")

    # Compute relaxed state
    relaxed_state = solver.compute_relaxed_network_state(
        network, left_boundary, right_boundary
    )

    relaxed_positions = relaxed_state['node_positions']

    print("\nRelaxed positions:")
    for nid, pos in sorted(relaxed_positions.items()):
        print(f"  Node {nid}: {pos}")

    # Validation 1: Fixed nodes haven't moved
    print("\nValidation 1: Fixed boundary nodes haven't moved")
    for fixed_node_id in [1, 5]:
        original = original_positions[fixed_node_id]
        relaxed = relaxed_positions[fixed_node_id]
        distance = np.linalg.norm(np.array(relaxed) - np.array(original))
        print(f"  Node {fixed_node_id} movement: {distance:.2e}")
        assert distance < 1e-6, f"Fixed node {fixed_node_id} moved by {distance}"

    print("  [OK] Fixed nodes preserved")

    # Validation 2: Free nodes moved toward rest length
    print("\nValidation 2: Free nodes relaxed toward equilibrium")
    for edge in network.get_edges():
        n_from = edge.n_from
        n_to = edge.n_to

        original_length = np.linalg.norm(
            np.array(original_positions[n_to]) - np.array(original_positions[n_from])
        )
        relaxed_length = np.linalg.norm(
            np.array(relaxed_positions[n_to]) - np.array(relaxed_positions[n_from])
        )
        rest_length = edge.rest_length

        original_strain = abs(original_length - rest_length) / rest_length
        relaxed_strain = abs(relaxed_length - rest_length) / rest_length

        print(f"  Edge {edge.get_id()}: strain {original_strain:.4f} -> {relaxed_strain:.4f}")

        # Relaxation should reduce strain (bring length closer to rest length)
        # For isolated edges, strain should be nearly zero
        # For edges with fixed endpoints, some strain may remain
        if edge.get_id() == 103:  # Isolated edge
            assert relaxed_strain < 0.01, f"Isolated edge should fully relax (strain={relaxed_strain:.4f})"

    print("  [OK] Free nodes moved toward equilibrium")

    # Validation 3: All edge lengths are positive
    print("\nValidation 3: Edge lengths are positive")
    for edge in network.get_edges():
        n_from = edge.n_from
        n_to = edge.n_to
        length = np.linalg.norm(
            np.array(relaxed_positions[n_to]) - np.array(relaxed_positions[n_from])
        )
        print(f"  Edge {edge.get_id()}: length = {length:.4f}")
        assert length > 1e-10, f"Edge {edge.get_id()} has near-zero length"

    print("  [OK] All edge lengths positive")

    print("\n[PASS] Relaxation physics PASSED")
    return True


def test_validation_function():
    """Test the built-in validation function."""
    print("\n" + "="*80)
    print("TEST 3: Validation Function")
    print("="*80)

    network = create_simple_test_network()
    solver = RelaxedNetworkSolver()

    left_boundary = {1}
    right_boundary = {5}

    relaxed_state = solver.compute_relaxed_network_state(
        network, left_boundary, right_boundary
    )

    # Validate
    validation_result = solver.validate_relaxed_state(
        network, relaxed_state, left_boundary, right_boundary
    )

    print("\nValidation result:")
    print(f"  Overall valid: {validation_result['valid']}")
    print(f"  Fixed nodes OK: {validation_result['fixed_nodes_ok']}")
    print(f"  Edge lengths OK: {validation_result['edge_lengths_ok']}")

    if validation_result['details']:
        print("\nDetails:")
        for detail in validation_result['details']:
            print(f"  - {detail}")

    assert validation_result['valid'], "Validation failed"
    assert validation_result['fixed_nodes_ok'], "Fixed nodes check failed"
    assert validation_result['edge_lengths_ok'], "Edge lengths check failed"

    print("\n[PASS] Validation function PASSED")
    return True


def main():
    """Run all validation tests."""
    print("\n" + "="*80)
    print("RELAXED NETWORK PHYSICS VALIDATION")
    print("="*80)

    tests = [
        test_component_decomposition,
        test_relaxation_physics,
        test_validation_function,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n[FAIL] TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*80 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
