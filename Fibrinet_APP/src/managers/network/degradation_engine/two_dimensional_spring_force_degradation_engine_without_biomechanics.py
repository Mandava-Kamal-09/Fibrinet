from utils.logger.logger import Logger
from .degradation_engine_strategy import DegradationEngineStrategy
from ..networks.network_2d import Network2D
import copy
import numpy as np

class TwoDimensionalSpringForceDegradationEngineWithoutBiomechanics(DegradationEngineStrategy):
    """2D spring engine (Hooke's Law), no biomechanics."""

    def degrade_edge(self, network: Network2D, edge_id):
        Logger.log(f"start degrade_edge(self, network, {edge_id})")

        degraded_network = copy.deepcopy(network)
        degraded_network.remove_edge(edge_id)
        self.relax_network(degraded_network)

        Logger.log(f"end degrade_edge(self, network, {edge_id})")
        return degraded_network

    def degrade_node(self, network: Network2D, node_id):
        Logger.log(f"start degrade_node(self, network, {node_id})")

        degraded_network = copy.deepcopy(network)
        degraded_network.remove_node(node_id)
        connected_edges = [
            edge.get_id() for edge in degraded_network.get_edges()
            if edge.n_from == node_id or edge.n_to == node_id
        ]
        for eid in connected_edges:
            degraded_network.remove_edge(eid)
        self.relax_network(degraded_network)

        Logger.log(f"end degrade_node(self, network, {node_id})")
        return degraded_network

    def relax_network(self, network: Network2D):
        """Relax node positions using Hooke's Law until forces are small."""
        Logger.log(f"start 2dwithoutbio relax_network(self, network={network})")
        iterations = 1000
        alpha = 0.01
        tolerance = 1e-5
        for _ in range(iterations):
            forces = self.compute_node_forces(network)
            max_force = 0.0

            for node in network.get_nodes():
                if getattr(node, 'is_fixed', False):
                    continue
                force = forces[node.get_id()]
                max_force = max(max_force, np.linalg.norm(force))
                node.n_x += alpha * force[0]
                node.n_y += alpha * force[1]
            if max_force < tolerance:
                break
        Logger.log(f"end 2dwithoutbio relax_network(self, network={network})")

    def compute_node_forces(self, network: Network2D):
        """Compute net spring force on each node via Hooke's law."""
        forces = {node.get_id(): np.array([0.0, 0.0]) for node in network.get_nodes()}
        k_global = network.meta_data.get("spring_stiffness_constant", 1.0)

        for edge in network.get_edges():
            n_from = network.get_node_by_id(edge.n_from)
            n_to = network.get_node_by_id(edge.n_to)
            p_from = np.array([n_from.n_x, n_from.n_y])
            p_to = np.array([n_to.n_x, n_to.n_y])
            vector = p_to - p_from
            length = np.linalg.norm(vector)
            if length == 0:
                continue
            unit_vector = vector / length
            rest_length = getattr(edge, "rest_length", length)
            k_edge = getattr(edge, "spring_constant", k_global)
            force_magnitude = k_edge * (length - rest_length)
            force = force_magnitude * unit_vector
            forces[n_from.get_id()] += force
            forces[n_to.get_id()] -= force
        return forces


    def get_edge_rest_lengths(self, network: Network2D):
        """Return list of edge rest lengths."""
        return [edge.rest_length for edge in network.get_edges()]
