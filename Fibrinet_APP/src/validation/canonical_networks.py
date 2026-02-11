"""
Canonical Test Networks for FibriNet Validation.

Provides deterministic network generators for reproducible testing and validation.
Each generator returns a network dictionary in the format expected by the simulation
loader (same node/edge attributes as the input file format).

Network Format:
{
    "nodes": [
        {"node_id": int, "x": float, "y": float, "is_left_boundary": bool, "is_right_boundary": bool},
        ...
    ],
    "edges": [
        {"edge_id": int, "n_from": int, "n_to": int, "thickness": float},
        ...
    ],
    "metadata": {
        "spring_stiffness_constant": float,
        "network_type": str,  # Canonical network identifier
        ...
    }
}

"""

import hashlib
import json
import random
from typing import Any, Dict, List, Optional, Tuple


def network_hash(network: Dict[str, Any]) -> str:
    """
    Compute deterministic hash of a network for provenance tracking.

    Args:
        network: Network dictionary

    Returns:
        SHA-256 hash (first 16 hex chars)
    """
    # Sort for deterministic serialization
    serialized = json.dumps(network, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def line(n: int, spacing: float = 1.0, k0: float = 1.0) -> Dict[str, Any]:
    """
    Generate a linear chain network.

    Creates n nodes in a line along the x-axis, connected sequentially.
    Left boundary is node 0, right boundary is node n-1.

    Args:
        n: Number of nodes (must be >= 2)
        spacing: Distance between adjacent nodes
        k0: Spring stiffness constant

    Returns:
        Network dictionary

    Example:
        line(3) produces:
        0 --- 1 --- 2
        (left)     (right)
    """
    if n < 2:
        raise ValueError("line() requires n >= 2")

    nodes = []
    for i in range(n):
        nodes.append({
            "node_id": i,
            "x": float(i * spacing),
            "y": 0.0,
            "is_left_boundary": (i == 0),
            "is_right_boundary": (i == n - 1),
        })

    edges = []
    for i in range(n - 1):
        edges.append({
            "edge_id": i,
            "n_from": i,
            "n_to": i + 1,
            "thickness": 1.0,
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "spring_stiffness_constant": k0,
            "network_type": "line",
            "n": n,
            "spacing": spacing,
        },
    }


def triangle(side_length: float = 1.0, k0: float = 1.0) -> Dict[str, Any]:
    """
    Generate an equilateral triangle network.

    Creates 3 nodes forming a triangle, fully connected.
    Node 0 (bottom-left) is left boundary, node 1 (bottom-right) is right boundary.

    Args:
        side_length: Length of each side
        k0: Spring stiffness constant

    Returns:
        Network dictionary

    Example:
        triangle() produces:
              2 (top)
             / \\
            /   \\
           0-----1
        (left) (right)
    """
    import math

    # Equilateral triangle vertices
    height = side_length * math.sqrt(3) / 2

    nodes = [
        {"node_id": 0, "x": 0.0, "y": 0.0, "is_left_boundary": True, "is_right_boundary": False},
        {"node_id": 1, "x": side_length, "y": 0.0, "is_left_boundary": False, "is_right_boundary": True},
        {"node_id": 2, "x": side_length / 2, "y": height, "is_left_boundary": False, "is_right_boundary": False},
    ]

    edges = [
        {"edge_id": 0, "n_from": 0, "n_to": 1, "thickness": 1.0},  # Bottom
        {"edge_id": 1, "n_from": 1, "n_to": 2, "thickness": 1.0},  # Right
        {"edge_id": 2, "n_from": 2, "n_to": 0, "thickness": 1.0},  # Left
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "spring_stiffness_constant": k0,
            "network_type": "triangle",
            "side_length": side_length,
        },
    }


def square(side_length: float = 1.0, k0: float = 1.0, with_diagonals: bool = True) -> Dict[str, Any]:
    """
    Generate a square network.

    Creates 4 nodes forming a square, with optional diagonal bracing.
    Left side (nodes 0, 2) is left boundary, right side (nodes 1, 3) is right boundary.

    Args:
        side_length: Length of each side
        k0: Spring stiffness constant
        with_diagonals: If True, add diagonal cross-bracing

    Returns:
        Network dictionary

    Example:
        square() produces:
        2 --- 3
        | \\ / |
        |  X  |
        | / \\ |
        0 --- 1
    """
    nodes = [
        {"node_id": 0, "x": 0.0, "y": 0.0, "is_left_boundary": True, "is_right_boundary": False},
        {"node_id": 1, "x": side_length, "y": 0.0, "is_left_boundary": False, "is_right_boundary": True},
        {"node_id": 2, "x": 0.0, "y": side_length, "is_left_boundary": True, "is_right_boundary": False},
        {"node_id": 3, "x": side_length, "y": side_length, "is_left_boundary": False, "is_right_boundary": True},
    ]

    edges = [
        {"edge_id": 0, "n_from": 0, "n_to": 1, "thickness": 1.0},  # Bottom
        {"edge_id": 1, "n_from": 1, "n_to": 3, "thickness": 1.0},  # Right
        {"edge_id": 2, "n_from": 3, "n_to": 2, "thickness": 1.0},  # Top
        {"edge_id": 3, "n_from": 2, "n_to": 0, "thickness": 1.0},  # Left
    ]

    if with_diagonals:
        edges.extend([
            {"edge_id": 4, "n_from": 0, "n_to": 3, "thickness": 1.0},  # Diagonal 1
            {"edge_id": 5, "n_from": 1, "n_to": 2, "thickness": 1.0},  # Diagonal 2
        ])

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "spring_stiffness_constant": k0,
            "network_type": "square",
            "side_length": side_length,
            "with_diagonals": with_diagonals,
        },
    }


def t_shape(arm_length: float = 1.0, k0: float = 1.0) -> Dict[str, Any]:
    """
    Generate a T-shaped network.

    Creates a T shape with a vertical stem and horizontal crossbar.
    Node 0 (bottom of stem) is left boundary, nodes 3 and 4 (ends of crossbar) are right boundary.

    Args:
        arm_length: Length of each arm
        k0: Spring stiffness constant

    Returns:
        Network dictionary

    Example:
        t_shape() produces:
        3 --- 2 --- 4
              |
              |
              1
              |
              |
              0 (left boundary)

        Nodes 3 and 4 are right boundary.
    """
    nodes = [
        {"node_id": 0, "x": arm_length, "y": 0.0, "is_left_boundary": True, "is_right_boundary": False},
        {"node_id": 1, "x": arm_length, "y": arm_length, "is_left_boundary": False, "is_right_boundary": False},
        {"node_id": 2, "x": arm_length, "y": 2 * arm_length, "is_left_boundary": False, "is_right_boundary": False},
        {"node_id": 3, "x": 0.0, "y": 2 * arm_length, "is_left_boundary": False, "is_right_boundary": True},
        {"node_id": 4, "x": 2 * arm_length, "y": 2 * arm_length, "is_left_boundary": False, "is_right_boundary": True},
    ]

    edges = [
        {"edge_id": 0, "n_from": 0, "n_to": 1, "thickness": 1.0},  # Stem bottom
        {"edge_id": 1, "n_from": 1, "n_to": 2, "thickness": 1.0},  # Stem top
        {"edge_id": 2, "n_from": 3, "n_to": 2, "thickness": 1.0},  # Left arm
        {"edge_id": 3, "n_from": 2, "n_to": 4, "thickness": 1.0},  # Right arm
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "spring_stiffness_constant": k0,
            "network_type": "t_shape",
            "arm_length": arm_length,
        },
    }


def small_lattice(rows: int, cols: int, spacing: float = 1.0, k0: float = 1.0) -> Dict[str, Any]:
    """
    Generate a regular rectangular lattice network.

    Creates a rows x cols grid of nodes with nearest-neighbor and diagonal connections.
    Left column is left boundary, right column is right boundary.

    Args:
        rows: Number of rows (must be >= 2)
        cols: Number of columns (must be >= 2)
        spacing: Distance between adjacent nodes
        k0: Spring stiffness constant

    Returns:
        Network dictionary

    Example:
        small_lattice(2, 3) produces:
        3 --- 4 --- 5
        | \\ / | \\ / |
        |  X  |  X  |
        | / \\ | / \\ |
        0 --- 1 --- 2

        Column 0 (nodes 0, 3) is left boundary
        Column 2 (nodes 2, 5) is right boundary
    """
    if rows < 2 or cols < 2:
        raise ValueError("small_lattice() requires rows >= 2 and cols >= 2")

    # Generate nodes
    nodes = []
    node_id = 0
    for row in range(rows):
        for col in range(cols):
            nodes.append({
                "node_id": node_id,
                "x": float(col * spacing),
                "y": float(row * spacing),
                "is_left_boundary": (col == 0),
                "is_right_boundary": (col == cols - 1),
            })
            node_id += 1

    def node_at(row: int, col: int) -> int:
        """Get node ID at grid position."""
        return row * cols + col

    # Generate edges
    edges = []
    edge_id = 0

    for row in range(rows):
        for col in range(cols):
            current = node_at(row, col)

            # Horizontal edge (right)
            if col < cols - 1:
                edges.append({
                    "edge_id": edge_id,
                    "n_from": current,
                    "n_to": node_at(row, col + 1),
                    "thickness": 1.0,
                })
                edge_id += 1

            # Vertical edge (up)
            if row < rows - 1:
                edges.append({
                    "edge_id": edge_id,
                    "n_from": current,
                    "n_to": node_at(row + 1, col),
                    "thickness": 1.0,
                })
                edge_id += 1

            # Diagonal edge (up-right)
            if row < rows - 1 and col < cols - 1:
                edges.append({
                    "edge_id": edge_id,
                    "n_from": current,
                    "n_to": node_at(row + 1, col + 1),
                    "thickness": 1.0,
                })
                edge_id += 1

            # Diagonal edge (up-left)
            if row < rows - 1 and col > 0:
                edges.append({
                    "edge_id": edge_id,
                    "n_from": current,
                    "n_to": node_at(row + 1, col - 1),
                    "thickness": 1.0,
                })
                edge_id += 1

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "spring_stiffness_constant": k0,
            "network_type": "small_lattice",
            "rows": rows,
            "cols": cols,
            "spacing": spacing,
        },
    }


def mini_realistic(
    seed: int = 42,
    n_nodes: int = 20,
    connection_radius: float = 1.5,
    k0: float = 1.0,
) -> Dict[str, Any]:
    """
    Generate a mini-realistic random network with fixed seed.

    Creates a random point cloud and connects nearby nodes, simulating a
    simple fiber network topology. Deterministic given the same seed.

    Args:
        seed: Random seed for reproducibility
        n_nodes: Number of nodes (must be >= 4)
        connection_radius: Maximum distance for edge creation
        k0: Spring stiffness constant

    Returns:
        Network dictionary

    Notes:
        - Nodes are distributed in a 4x2 bounding box
        - Left boundary: nodes with x < 0.5
        - Right boundary: nodes with x > 3.5
        - Edges connect nodes within connection_radius
    """
    if n_nodes < 4:
        raise ValueError("mini_realistic() requires n_nodes >= 4")

    rng = random.Random(seed)

    # Generate random node positions in a 4x2 box
    positions: List[Tuple[float, float]] = []
    for _ in range(n_nodes):
        x = rng.uniform(0.0, 4.0)
        y = rng.uniform(0.0, 2.0)
        positions.append((x, y))

    # Sort by x-coordinate for deterministic boundary assignment
    positions.sort(key=lambda p: p[0])

    # Create nodes
    nodes = []
    for i, (x, y) in enumerate(positions):
        nodes.append({
            "node_id": i,
            "x": x,
            "y": y,
            "is_left_boundary": (x < 0.5),
            "is_right_boundary": (x > 3.5),
        })

    # Create edges based on distance
    def distance(i: int, j: int) -> float:
        xi, yi = positions[i]
        xj, yj = positions[j]
        return ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5

    edges = []
    edge_id = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if distance(i, j) <= connection_radius:
                edges.append({
                    "edge_id": edge_id,
                    "n_from": i,
                    "n_to": j,
                    "thickness": 1.0,
                })
                edge_id += 1

    # Ensure at least one edge exists
    if len(edges) == 0:
        # Connect first two nodes as fallback
        edges.append({
            "edge_id": 0,
            "n_from": 0,
            "n_to": 1,
            "thickness": 1.0,
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "spring_stiffness_constant": k0,
            "network_type": "mini_realistic",
            "seed": seed,
            "n_nodes": n_nodes,
            "connection_radius": connection_radius,
        },
    }


__all__ = [
    "line",
    "triangle",
    "square",
    "t_shape",
    "small_lattice",
    "mini_realistic",
    "network_hash",
]
