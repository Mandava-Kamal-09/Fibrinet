"""Shared pytest fixtures for FibriNet verification tests."""

import sys
import os
import pytest
import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validation.canonical_networks import line, triangle, square, small_lattice, mini_realistic
from src.core.fibrinet_core_v2 import (
    WLCFiber, NetworkState, HybridMechanochemicalSimulation,
    EnergyMinimizationSolver, StochasticChemistryEngine,
    PhysicalConstants, check_left_right_connectivity,
)

PC = PhysicalConstants()


# Helper: convert canonical network dict → Core V2 NetworkState

def dict_to_network_state(net_dict, spacing_m=1e-6, applied_strain=0.0,
                          force_model='wlc', prestrain=True):
    """Convert canonical network dict to a Core V2 NetworkState.

    Args:
        net_dict: dict from canonical_networks generators
        spacing_m: coordinate-to-meter conversion factor
        applied_strain: strain applied to right boundary
        force_model: 'wlc' or 'ewlc'
        prestrain: whether to apply 23% polymerization prestrain
    """
    nodes = net_dict['nodes']
    edges = net_dict['edges']

    # Build node positions in SI
    node_positions = {}
    left_boundary = set()
    right_boundary = set()

    for n in nodes:
        nid = n['node_id']
        node_positions[nid] = np.array([n['x'] * spacing_m, n['y'] * spacing_m])
        if n['is_left_boundary']:
            left_boundary.add(nid)
        if n['is_right_boundary']:
            right_boundary.add(nid)

    # Apply strain to right boundary
    if applied_strain > 0 and right_boundary:
        x_coords = [pos[0] for pos in node_positions.values()]
        x_span = max(x_coords) - min(x_coords)
        for nid in right_boundary:
            node_positions[nid][0] += applied_strain * x_span

    # Build fibers
    fibers = []
    for e in edges:
        pos_i = node_positions[e['n_from']]
        pos_j = node_positions[e['n_to']]
        geom_length = float(np.linalg.norm(pos_j - pos_i))

        # Prestrain: fiber polymerized under 23% strain
        if prestrain:
            L_c = geom_length / (1.0 + PC.PRESTRAIN)
        else:
            L_c = geom_length

        # Protect against zero-length fibers
        L_c = max(L_c, 1e-12)

        fibers.append(WLCFiber(
            fiber_id=e['edge_id'],
            node_i=e['n_from'],
            node_j=e['n_to'],
            L_c=L_c,
            force_model=force_model,
        ))

    # Boundary conditions
    fixed_nodes = {nid: node_positions[nid].copy() for nid in left_boundary}
    partial_fixed_x = {nid: node_positions[nid][0] for nid in right_boundary}

    state = NetworkState(
        time=0.0,
        fibers=fibers,
        node_positions=node_positions,
        fixed_nodes=fixed_nodes,
        partial_fixed_x=partial_fixed_x,
        left_boundary_nodes=left_boundary,
        right_boundary_nodes=right_boundary,
    )
    state.rebuild_fiber_index()
    return state


# Fixtures

@pytest.fixture
def line_network():
    """Canonical 5-node line network (unstrained, no prestrain)."""
    return dict_to_network_state(line(n=5), prestrain=False)


@pytest.fixture
def line_network_prestrained():
    """5-node line with 23% prestrain applied."""
    return dict_to_network_state(line(n=5), prestrain=True)


@pytest.fixture
def lattice_network():
    """4x6 lattice with diagonal bracing, 10% strain, prestrained."""
    return dict_to_network_state(
        small_lattice(4, 6), applied_strain=0.1, prestrain=True,
    )


@pytest.fixture
def realistic_network():
    """Mini-realistic 30-node network, seed=42."""
    return dict_to_network_state(
        mini_realistic(seed=42, n_nodes=30), applied_strain=0.1, prestrain=True,
    )


@pytest.fixture
def triangle_network():
    """Equilateral triangle, no prestrain."""
    return dict_to_network_state(triangle(), prestrain=False)


@pytest.fixture
def square_network():
    """4-node square with diagonals, 10% strain, prestrained."""
    return dict_to_network_state(
        square(with_diagonals=True), applied_strain=0.1, prestrain=True,
    )


@pytest.fixture
def single_fiber():
    """A single WLC fiber with known L_c for unit testing."""
    L_c = 1e-6  # 1 micron
    return WLCFiber(fiber_id=0, node_i=0, node_j=1, L_c=L_c)


@pytest.fixture
def single_fiber_ewlc():
    """A single eWLC fiber."""
    L_c = 1e-6
    return WLCFiber(fiber_id=0, node_i=0, node_j=1, L_c=L_c, force_model='ewlc')
