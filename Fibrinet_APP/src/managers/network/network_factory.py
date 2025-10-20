from collections import defaultdict
from utils.logger.logger import Logger

# TYPES OF NETWORKS
from .networks.base_network import BaseNetwork
from .networks.network_2d import Network2D

# TYPES OF NODES
from .nodes.base_node import BaseNode
from .nodes.node_2d import Node2D
from .nodes.fixable_node_2d import FixableNode2D

# TYPES OF EDGES
from .edges.base_edge import BaseEdge
from .edges.edge_with_rest_length import EdgeWithRestLength

class NetworkFactory:
    """Create networks/nodes/edges from input data matching registered schemas."""
    # DICTIONARY TO STORE REGISTERED NETWORK TYPES
    _network_types = {}
    # REGISTERED NODE TYPES
    _node_types = defaultdict(list)
    # REGISTERED EDGE TYPES
    _edge_types = defaultdict(list)


    @classmethod
    def register_network_type(cls, network_class):
        """Register a network class."""
        # LOGGING THE REGISTRATION OF NETWORK TYPE
        Logger.log(f"Registering network class {network_class}")
        cls._network_types[network_class] = network_class


    @classmethod
    def register_node_type(cls, network_class, node_class):
        """Register a node class for a network class."""
        # LOGGING THE REGISTRATION OF NODE TYPE
        Logger.log(f"Registering node type '{network_class}' with class {node_class}")
        cls._node_types[network_class].append(node_class)

    @classmethod
    def register_edge_type(cls, network_class, edge_class):
        """Register an edge class for a network class."""
        # LOGGING THE REGISTRATION OF EDGE TYPE
        Logger.log(f"Registering edge type '{network_class}' with class {edge_class}")
        cls._edge_types[network_class].append(edge_class)

    @classmethod
    def create_network(cls, data: dict):
        """Build a network object from raw dicts using registered types."""
        Logger.log(f"start create_network(self, {data})")

        # ITERATE THROUGH REGISTERED NETWORK TYPES TO FIND A MATCHING ONE
        for network_type, network_class in cls._network_types.items():
            if cls._matches_schema(data, network_class.schema):
                # LOGGING MATCHING NETWORK TYPE FOUND
                Logger.log(f"Found matching network type '{network_type}'")

                nodes = []
                node_data = data.get("nodes", {})

                if node_data:
                    Logger.log("Processing node data")
                    node_list = []
                    keys = list(node_data.keys())
                    for i in range(len(node_data[keys[0]])):
                        node_dict = {key: node_data[key][i] for key in keys}
                        node_list.append(node_dict)
                    
                    for node in node_list:
                        node_classes = (
                            cls._node_types.get(network_class, []) +
                            cls._node_types.get(BaseNetwork, [])  # fallback
                        )
                        for node_class in node_classes:
                            try:
                                nodes.append(node_class(node))  # try creating it
                                break
                            except Exception as e:
                                Logger.log(f"Node class {node_class} failed: {e}")
                        else:
                            raise ValueError(f"No matching node class for node: {node}")

                data["nodes"] = nodes

                edges = []
                edge_data = data.get("edges", {})
                
                if edge_data:
                    Logger.log("Processing edge data")
                    edge_list = []
                    keys = list(edge_data.keys())
                    for i in range(len(edge_data[keys[0]])):
                        edge_dict = {key: edge_data[key][i] for key in keys}
                        edge_list.append(edge_dict)
                    
                    for edge in edge_list:
                        edge_classes = (
                            cls._edge_types.get(network_class, []) +
                            cls._edge_types.get(BaseNetwork, [])  # fallback
                        )
                        for edge_class in edge_classes:
                            try:
                                edges.append(edge_class(edge))  # Try creating edge
                                break
                            except Exception as e:
                                Logger.log(f"Edge class {edge_class} failed: {e}")
                        else:
                            raise ValueError(f"No matching edge class for edge: {edge}")


                data["edges"] = edges

                network = network_class(data) 
                Logger.log(f"Network created successfully: {network}")
                Logger.log("end create_network(self, data)")

                return network

        raise ValueError("No matching network type found.")

    @classmethod
    def _matches_schema(cls, data:dict, schema):
        """Return True if data contains required meta/node/edge fields."""
        Logger.log(f"start _matches_schema(self, {data})")

        meta_data = data.get("meta_data", {})
        nodes = data.get("nodes", {})
        edges = data.get("edges", {})

        Logger.log("Checking meta_data")
        for attr in schema.get("meta_data", []):
            if attr not in meta_data:
                Logger.log(f"meta_data: Attribute '{attr}' not found in data. Schema mismatch.")
                return False
            else:
                Logger.log(f"meta_data: Attribute '{attr}' found.")
        Logger.log("meta_data check complete")

        # CHECKING NODES
        if nodes:
            Logger.log("Checking nodes")
            keys = list(nodes.keys())
            for i in range(len(nodes[keys[0]])):
                node = {key: nodes[key][i] for key in keys}
                for attr in schema.get("node_attributes", []):
                    if attr not in node:
                        Logger.log(f"nodes: Attribute '{attr}' not found in node {node}. Schema mismatch.")
                        return False
                    else:
                        Logger.log(f"nodes: Attribute '{attr}' found in node {node}.")
            Logger.log("nodes check complete")
        else:
            Logger.log("No nodes found in data.")

        # CHECKING EDGES
        if edges:
            Logger.log("Checking edges")
            keys = list(edges.keys())
            for i in range(len(edges[keys[0]])):
                edge = {key: edges[key][i] for key in keys}
                for attr in schema.get("edge_attributes", []):
                    if attr not in edge:
                        Logger.log(f"edges: Attribute '{attr}' not found in edge {edge}. Schema mismatch.")
                        return False
                    else:
                        Logger.log(f"edges: Attribute '{attr}' found in edge {edge}.")
            Logger.log("edges check complete")
        else:
            Logger.log("No edges found in data.")

        Logger.log("Schema matching successful")
        Logger.log("end _matches_schema(self, data)")
        return True
    
    @classmethod
    def get_all_registered_components(cls):
        """
        Returns all registered network, node, and edge classes.

        :return: A tuple of three lists: (network_classes, node_classes, edge_classes)
        """
        Logger.log("start get_all_registered_components()")
        network_classes = list(set(cls._network_types.values()))
        node_classes = list({cls for sublist in cls._node_types.values() for cls in sublist})
        edge_classes = list({cls for sublist in cls._edge_types.values() for cls in sublist})
        Logger.log("end get_all_registered_components()")
        return network_classes, node_classes, edge_classes


# REGISTER NETWORK, NODE, AND EDGE TYPES
Logger.log("Registering types with factory...")

# NETWORKS
NetworkFactory.register_network_type(Network2D)

# NODES
NetworkFactory.register_node_type(Network2D, Node2D)
NetworkFactory.register_node_type(Network2D, FixableNode2D)

# EDGES
NetworkFactory.register_edge_type(Network2D, BaseEdge)   
NetworkFactory.register_edge_type(Network2D, EdgeWithRestLength)

