from utils.logger.logger import Logger
from .degradation_engine_strategy import DegradationEngineStrategy
from ....models.exceptions import NodeNotFoundError, EdgeNotFoundError
import copy

class NoPhysics(DegradationEngineStrategy):
    """Remove edges/nodes without physics."""

    def degrade_edge(self, network, edge_id):
        """Return a copy of network with the edge removed."""
        Logger.log(f"start degrade_edge(self, network, {edge_id})")

        new_network = copy.deepcopy(network)

        if edge_id not in [edge.get_id() for edge in new_network.get_edges()]:
            raise Exception(f"Edge ID '{edge_id}' not found in network.")

        new_network.remove_edge(edge_id)

        connected_node_ids = {edge.get_attribute("n_to") for edge in new_network.get_edges()}.union(
            {edge.get_attribute("n_from") for edge in new_network.get_edges()})
        new_nodes = [node for node in new_network.get_nodes() if node.get_id() in connected_node_ids]
        new_network.nodes = new_nodes

        Logger.log(f"end degrade_edge(self, network, {edge_id})")
        return new_network

    def degrade_node(self, network, node_id):
        """Return a copy of network with the node and its edges removed."""
        Logger.log(f"start degrade_node(self, network, {node_id})")

        new_network = copy.deepcopy(network)

        if node_id not in [node.get_id() for node in new_network.get_nodes()]:
            raise (f"Node ID '{node_id}' not found in network.")


        new_network.remove_node(node_id)

        new_edges = [edge for edge in new_network.get_edges()
                     if edge.get_attribute("n_to") != node_id and edge.get_attribute("n_from") != node_id]
        new_network.edges = new_edges

        connected_node_ids = {edge.get_attribute("n_to") for edge in new_network.get_edges()}.union(
            {edge.get_attribute("n_from") for edge in new_network.get_edges()})
        new_nodes = [node for node in new_network.get_nodes() if node.get_id() in connected_node_ids]
        new_network.nodes = new_nodes

        Logger.log(f"end degrade_node(self, network, {node_id})")
        return new_network
    
    def relax_network(self, network):
        """Return a copy of the input network (no physics)."""
        Logger.log("start relax_network(self, network)")

        new_network = copy.deepcopy(network)

        Logger.log("end relax_network(self, network)")
        return new_network
