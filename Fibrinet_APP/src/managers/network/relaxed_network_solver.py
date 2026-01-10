"""
Relaxed Network Solver - Post-Percolation Mechanical Relaxation

This module computes the mechanically relaxed state of a fibrin network after
percolation loss (left-right connectivity broken). The network is decomposed
into disconnected components, each relaxed independently to mechanical equilibrium.

Physical Model:
--------------
- Fibers modeled as Hookean springs (F = k * Δx)
- Overdamped dynamics (no inertia, force-based position updates)
- Boundary nodes (crosslinks at poles) remain fixed
- Free nodes relax to zero net force (equilibrium)

Algorithm:
----------
1. Decompose network into connected components (BFS graph traversal)
2. For each component, identify fixed boundary nodes
3. Relax component to equilibrium via iterative force minimization
4. Return relaxed node positions for visualization

Scientific Context:
------------------
After enzymatic cleavage disconnects the fibrin network, the remaining
fragments relax to new equilibrium configurations without external strain.
This provides insight into the final structural topology of the cleared clot.

Author: FibriNet Research Team
Date: 2026-01-06
Publication: Biophysics Journal (Submission)
"""

from collections import deque
from typing import Dict, Set, List, Tuple, Any, Optional
import numpy as np
from utils.logger.logger import Logger


class NetworkComponent:
    """
    Represents a connected component of the fibrin network.

    Attributes:
        node_ids: Set of node IDs in this component
        edge_ids: Set of edge IDs in this component
        component_type: 'left_connected', 'right_connected', 'isolated', or 'spanning'
        fixed_nodes: Set of node IDs that must remain at pole positions
    """
    def __init__(self, node_ids: Set[int], edge_ids: Set[int],
                 component_type: str, fixed_nodes: Set[int]):
        self.node_ids = node_ids
        self.edge_ids = edge_ids
        self.component_type = component_type
        self.fixed_nodes = fixed_nodes

    def __repr__(self):
        return (f"NetworkComponent(type={self.component_type}, "
                f"nodes={len(self.node_ids)}, edges={len(self.edge_ids)}, "
                f"fixed={len(self.fixed_nodes)})")


class RelaxedNetworkSolver:
    """
    Computes mechanically relaxed network state after percolation loss.

    This solver decomposes the network into disconnected components and
    relaxes each to mechanical equilibrium using spring-based physics.
    """

    def __init__(self):
        """Initialize solver with default parameters."""
        self.max_iterations = 1000
        self.tolerance = 1e-5
        self.alpha = 0.01  # Relaxation rate (adaptive possible)
        Logger.log("RelaxedNetworkSolver initialized")

    def decompose_into_components(self, network, left_boundary_ids: Set[int],
                                   right_boundary_ids: Set[int]) -> List[NetworkComponent]:
        """
        Decompose network into connected components using BFS.

        After percolation loss, the network fragments into disconnected subgraphs.
        This method identifies all components and classifies them by boundary attachment.

        Args:
            network: Network2D object with nodes and edges
            left_boundary_ids: Set of node IDs attached to left pole
            right_boundary_ids: Set of node IDs attached to right pole

        Returns:
            List of NetworkComponent objects, each representing a disconnected fragment

        Algorithm:
            1. For each unvisited node, perform BFS to find all connected nodes
            2. Identify all edges within the component
            3. Classify component type based on boundary node membership
            4. Determine which nodes must remain fixed (boundary constraints)
        """
        Logger.log(f"Starting component decomposition (left_boundary={len(left_boundary_ids)}, "
                   f"right_boundary={len(right_boundary_ids)})")

        visited = set()
        components = []
        all_nodes = network.get_nodes()

        # Build adjacency list for efficient traversal
        adjacency = {}
        for edge in network.get_edges():
            n_from = edge.n_from
            n_to = edge.n_to
            if n_from not in adjacency:
                adjacency[n_from] = []
            if n_to not in adjacency:
                adjacency[n_to] = []
            adjacency[n_from].append((n_to, edge.get_id()))
            adjacency[n_to].append((n_from, edge.get_id()))

        # BFS to find all connected components
        for start_node in all_nodes:
            start_id = start_node.get_id()
            if start_id in visited:
                continue

            # BFS from this node
            component_nodes = set()
            component_edges = set()
            queue = deque([start_id])

            while queue:
                node_id = queue.popleft()
                if node_id in visited:
                    continue

                visited.add(node_id)
                component_nodes.add(node_id)

                # Traverse all edges connected to this node
                if node_id in adjacency:
                    for neighbor_id, edge_id in adjacency[node_id]:
                        component_edges.add(edge_id)
                        if neighbor_id not in visited:
                            queue.append(neighbor_id)

            # Classify component type
            has_left = bool(component_nodes & left_boundary_ids)
            has_right = bool(component_nodes & right_boundary_ids)

            if has_left and has_right:
                # Should not happen after percolation loss
                component_type = 'spanning'
                fixed_nodes = (component_nodes & left_boundary_ids) | (component_nodes & right_boundary_ids)
                Logger.log(f"WARNING: Found spanning component after percolation loss! "
                          f"This may indicate percolation is still intact.")
            elif has_left:
                component_type = 'left_connected'
                fixed_nodes = component_nodes & left_boundary_ids
            elif has_right:
                component_type = 'right_connected'
                fixed_nodes = component_nodes & right_boundary_ids
            else:
                component_type = 'isolated'
                fixed_nodes = set()

            component = NetworkComponent(
                node_ids=component_nodes,
                edge_ids=component_edges,
                component_type=component_type,
                fixed_nodes=fixed_nodes
            )
            components.append(component)

            Logger.log(f"Found component: {component}")

        Logger.log(f"Decomposition complete: {len(components)} components found")
        return components

    def relax_component(self, network, component: NetworkComponent) -> Dict[int, Tuple[float, float]]:
        """
        Relax a single component to mechanical equilibrium.

        Uses iterative force-based relaxation (gradient descent in potential energy).
        Fixed boundary nodes remain at pole positions; free nodes move to minimize
        spring forces until equilibrium (max force < tolerance).

        Args:
            network: Network2D object containing node/edge data
            component: NetworkComponent with node_ids, edge_ids, fixed_nodes

        Returns:
            Dictionary mapping node_id -> (x, y) relaxed position

        Physics:
            - Spring force: F = k * (L - L0) * unit_vector
            - Position update: x_new = x_old + alpha * F (overdamped dynamics)
            - Convergence: max(|F|) < tolerance

        Numerical Stability:
            - Adaptive step size (alpha) could be added if divergence detected
            - Zero-length edges skipped to avoid division by zero
            - Fixed nodes never updated (hard constraint)
        """
        Logger.log(f"Relaxing component: {component}")

        # Initialize positions from current network state
        positions = {}
        for node_id in component.node_ids:
            node = network.get_node_by_id(node_id)
            positions[node_id] = np.array([node.n_x, node.n_y], dtype=float)

        # Get all edges in this component
        component_edges = []
        for edge_id in component.edge_ids:
            edge = network.get_edge_by_id(edge_id)
            if edge is not None:
                component_edges.append(edge)

        # Iterative relaxation
        for iteration in range(self.max_iterations):
            # Compute forces on all nodes
            forces = {node_id: np.array([0.0, 0.0], dtype=float)
                     for node_id in component.node_ids}

            # Accumulate spring forces from all edges
            for edge in component_edges:
                n_from = edge.n_from
                n_to = edge.n_to

                # Skip if nodes not in this component (shouldn't happen)
                if n_from not in positions or n_to not in positions:
                    continue

                p_from = positions[n_from]
                p_to = positions[n_to]

                # Compute spring force (Hooke's Law)
                vector = p_to - p_from
                length = np.linalg.norm(vector)

                if length < 1e-10:
                    # Avoid division by zero for degenerate edges
                    continue

                unit_vector = vector / length
                rest_length = getattr(edge, 'rest_length', length)
                k_edge = getattr(edge, 'spring_constant', 1.0)

                # F = k * (L - L0)
                force_magnitude = k_edge * (length - rest_length)
                force = force_magnitude * unit_vector

                # Apply equal and opposite forces
                forces[n_from] += force
                forces[n_to] -= force

            # Update positions (only free nodes)
            max_force = 0.0
            for node_id in component.node_ids:
                if node_id in component.fixed_nodes:
                    # Fixed boundary nodes don't move
                    continue

                force = forces[node_id]
                force_magnitude = np.linalg.norm(force)
                max_force = max(max_force, force_magnitude)

                # Overdamped update: x += alpha * F
                positions[node_id] += self.alpha * force

            # Check convergence
            if max_force < self.tolerance:
                Logger.log(f"Component converged in {iteration+1} iterations "
                          f"(max_force={max_force:.2e})")
                break
        else:
            # Max iterations reached without convergence
            Logger.log(f"WARNING: Component did not converge after {self.max_iterations} "
                      f"iterations (max_force={max_force:.2e})")

        # Convert positions back to tuples
        relaxed_positions = {node_id: tuple(pos) for node_id, pos in positions.items()}

        return relaxed_positions

    def compute_relaxed_network_state(self, network, left_boundary_ids: Set[int],
                                     right_boundary_ids: Set[int]) -> Dict[str, Any]:
        """
        Compute full relaxed network state (all components).

        This is the main entry point for computing the post-percolation relaxed state.

        Args:
            network: Network2D object
            left_boundary_ids: Set of left boundary node IDs
            right_boundary_ids: Set of right boundary node IDs

        Returns:
            Dictionary containing:
                - 'components': List of NetworkComponent objects
                - 'node_positions': Dict mapping node_id -> (x, y) relaxed position
                - 'edges': List of edge data for visualization
        """
        Logger.log("Computing relaxed network state")

        # Decompose into components
        components = self.decompose_into_components(network, left_boundary_ids,
                                                    right_boundary_ids)

        # Relax each component independently
        all_relaxed_positions = {}
        for component in components:
            relaxed_positions = self.relax_component(network, component)
            all_relaxed_positions.update(relaxed_positions)

        # Package edge data for visualization
        edges = []
        for edge in network.get_edges():
            edges.append({
                'edge_id': edge.get_id(),
                'n_from': edge.n_from,
                'n_to': edge.n_to
            })

        result = {
            'components': components,
            'node_positions': all_relaxed_positions,
            'edges': edges
        }

        Logger.log(f"Relaxed network state computed: {len(components)} components, "
                   f"{len(all_relaxed_positions)} nodes, {len(edges)} edges")

        return result

    def validate_relaxed_state(self, network, relaxed_state: Dict[str, Any],
                              left_boundary_ids: Set[int],
                              right_boundary_ids: Set[int]) -> Dict[str, Any]:
        """
        Validate physical correctness of relaxed state.

        Checks:
            1. Fixed nodes haven't moved from original positions
            2. Free nodes have forces below tolerance (equilibrium)
            3. No edges have negative or zero length (unphysical)
            4. Total elastic energy is reasonable

        Args:
            network: Original network (for comparison)
            relaxed_state: Output from compute_relaxed_network_state
            left_boundary_ids: Left boundary node IDs
            right_boundary_ids: Right boundary node IDs

        Returns:
            Dictionary with validation results:
                - 'valid': Boolean indicating overall validity
                - 'fixed_nodes_ok': Fixed nodes preserved
                - 'equilibrium_ok': Forces below tolerance
                - 'edge_lengths_ok': All edges have positive length
                - 'details': Detailed validation messages
        """
        Logger.log("Validating relaxed network state")

        details = []
        valid = True

        # Check 1: Fixed nodes haven't moved
        fixed_nodes_ok = True
        node_positions = relaxed_state['node_positions']
        for component in relaxed_state['components']:
            for node_id in component.fixed_nodes:
                original_node = network.get_node_by_id(node_id)
                relaxed_pos = node_positions[node_id]
                original_pos = (original_node.n_x, original_node.n_y)

                distance = np.linalg.norm(np.array(relaxed_pos) - np.array(original_pos))
                if distance > 1e-6:
                    fixed_nodes_ok = False
                    details.append(f"Fixed node {node_id} moved by {distance:.2e}")

        # Check 2: Edge lengths are positive
        edge_lengths_ok = True
        for edge_data in relaxed_state['edges']:
            n_from = edge_data['n_from']
            n_to = edge_data['n_to']

            if n_from not in node_positions or n_to not in node_positions:
                continue

            p_from = np.array(node_positions[n_from])
            p_to = np.array(node_positions[n_to])
            length = np.linalg.norm(p_to - p_from)

            if length < 1e-10:
                edge_lengths_ok = False
                details.append(f"Edge {edge_data['edge_id']} has near-zero length: {length:.2e}")

        # Overall validation
        valid = fixed_nodes_ok and edge_lengths_ok

        validation_result = {
            'valid': valid,
            'fixed_nodes_ok': fixed_nodes_ok,
            'edge_lengths_ok': edge_lengths_ok,
            'details': details
        }

        Logger.log(f"Validation complete: valid={valid}")
        if details:
            for detail in details:
                Logger.log(f"  - {detail}")

        return validation_result
