from utils.logger.logger import Logger
from .degradation_engine_strategy import DegradationEngineStrategy
from ....models.exceptions import NodeNotFoundError, EdgeNotFoundError
import copy

class NoPhysics(DegradationEngineStrategy):
    """
    Degrades the network by removing edges and nodes without considering physics.
    """

    def degrade_edge(self, network, edge_id):
        """
        Creates a degraded version of the network without the specified edge.

        Params:
            network: network object to be degraded.
            edge_id: id of edge to be removed.

        Returns:
            A new Network object with the specified edge removed.
        """
        Logger.log(f"start degrade_edge(self, network, {edge_id})")

        # CREATE A COPY OF THE NETWORK
        new_network = copy.deepcopy(network)

        if edge_id not in [edge.get_id() for edge in new_network.get_edges()]:
            raise EdgeNotFoundError(f"Edge ID '{edge_id}' not found in network.")

        # REMOVE THE EDGE WITH THE SPECIFIED ID
        new_network.remove_edge(edge_id)

        # REMOVE NODES THAT NO LONGER HAVE ANY EDGES CONNECTED TO THEM
        connected_node_ids = {edge.get_attribute("n_to") for edge in new_network.get_edges()}.union(
            {edge.get_attribute("n_from") for edge in new_network.get_edges()})
        new_nodes = [node for node in new_network.get_nodes() if node.get_id() in connected_node_ids]
        new_network.nodes = new_nodes

        Logger.log(f"end degrade_edge(self, network, {edge_id})")
        return new_network

    def degrade_node(self, network, node_id):
        """
        Creates a degraded version of the network without the specified node and its associated edges.

        Params:
            network: network object to be degraded.
            node_id: id of node to be removed.

        Returns:
            A new Network object with the specified node and its edges removed.
        """
        Logger.log(f"start degrade_node(self, network, {node_id})")

        # CREATE A COPY OF THE NETWORK
        new_network = copy.deepcopy(network)

        if node_id not in [node.get_id() for node in new_network.get_nodes()]:
            raise NodeNotFoundError(f"Node ID '{node_id}' not found in network.")


        # REMOVE THE NODE WITH THE SPECIFIED ID
        new_network.remove_node(node_id)

        # REMOVE EDGES CONNECTED TO THE REMOVED NODE
        new_edges = [edge for edge in new_network.get_edges()
                     if edge.get_attribute("n_to") != node_id and edge.get_attribute("n_from") != node_id]
        new_network.edges = new_edges

        # REMOVE NODES THAT NO LONGER HAVE ANY EDGES CONNECTED TO THEM
        connected_node_ids = {edge.get_attribute("n_to") for edge in new_network.get_edges()}.union(
            {edge.get_attribute("n_from") for edge in new_network.get_edges()})
        new_nodes = [node for node in new_network.get_nodes() if node.get_id() in connected_node_ids]
        new_network.nodes = new_nodes

        Logger.log(f"end degrade_node(self, network, {node_id})")
        return new_network
    
    def relax_network(self, network):
        """
        NoPhysics relaxes the network by creating a copy of the network.

        Params:
            network: network object to be relaxed.

        Returns:
            A new Network object with all edges and nodes removed.
        """
        Logger.log("start relax_network(self, network)")

        # CREATE A COPY OF THE NETWORK
        new_network = copy.deepcopy(network)

        Logger.log("end relax_network(self, network)")
        return new_network
