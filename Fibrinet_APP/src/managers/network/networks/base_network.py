from utils.logger.logger import Logger
from ..nodes.base_node import BaseNode 
from ..edges.base_edge import BaseEdge 

class BaseNetwork:
    """Network with nodes, edges, and metadata."""

    allowed_node_type = BaseNode
    allowed_edge_type = BaseEdge
    
    def __init__(self, nodes=None, edges=None, meta_data=None, schema=None):
        """Initialize nodes, edges, metadata, and schema."""
        Logger.log(f"start BaseNetwork __init__(self, {nodes}, {edges}, {meta_data}, {schema})")

        # INITIALIZE NETWORK PROPERTIES
        self.nodes = nodes or []
        self.edges = edges or []
        self.meta_data = meta_data or {}
        self.schema = schema or {
            "meta_data": [],
            "meta_data_types": {},
            "node_attributes": self.allowed_node_type.get_schema(),
            "edge_attributes": self.allowed_edge_type.get_schema(),
        }

        Logger.log("===== Network Properties =====")
        Logger.log("Nodes:")
        for node in self.nodes:
            Logger.log(f"{node.__dict__}")
        Logger.log("Edges:")
        for edge in self.edges:
            Logger.log(f"{edge.__dict__}")
        Logger.log("Meta Properties:")
        for key, value in self.meta_data.items():
            Logger.log(f"{key}: {value}")
        Logger.log("==============================")
        Logger.log("end BaseNetwork __init__(self, nodes, edges, meta_data, schema)")

    def safe_cast(self, value, expected_type):
        """Cast value to expected_type; handle common cases."""
        try:
            if expected_type == bool:
                return str(value).strip().lower() in ["true", "1", "yes"]
            return expected_type(value)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid value: '{value}' is not of type {expected_type.__name__}")

    def get_nodes(self):
        """Return nodes list."""
        return self.nodes

    def get_edges(self):
        """Return edges list."""
        return self.edges

    def get_meta_data(self):
        """Return metadata dict."""
        return self.meta_data

    def add_meta_data(self, key, value):
        """Add metadata entry with schema/type check."""
        # VALIDATE KEY AGAINST SCHEMA
        if key not in self.schema.get("meta_data", []):
            raise ValueError(f"Meta data key '{key}' is not allowed by the schema.")
        
        # GET EXPECTED TYPE OR FALLBACK TO VALUE TYPE
        expected_type = self.schema.get("meta_data_types", {}).get(key, type(value))

        # CAST VALUE TO EXPECTED TYPE
        value = self.safe_cast(value, expected_type)

        # ADD META DATA IF KEY NOT PRESENT
        if key not in self.meta_data:
            self.meta_data[key] = value
            Logger.log(f"Meta data added: {key} = {value}")
        else:
            Logger.log(f"Meta data key '{key}' already exists. Use update_meta_data to change it.")

    def update_meta_data(self, key, value):
        """Update metadata entry with schema/type check."""
        # VALIDATE KEY AGAINST SCHEMA
        if key not in self.schema.get("meta_data", []):
            raise ValueError(f"Meta data key '{key}' is not allowed by the schema.")

        # GET EXPECTED TYPE OR FALLBACK TO VALUE TYPE
        expected_type = self.schema.get("meta_data_types", {}).get(key, type(value))

        # CAST VALUE TO EXPECTED TYPE
        value = self.safe_cast(value, expected_type)

        # UPDATE META DATA VALUE
        self.meta_data[key] = value
        Logger.log(f"Meta data updated: {key} = {value}")

    def remove_meta_data(self, key):
        """Remove metadata entry if present."""
        if key in self.meta_data:
            del self.meta_data[key]
            Logger.log(f"Meta data removed: {key}")
        else:
            Logger.log(f"Meta data key '{key}' not found.")

    def get_meta_data_keys(self):
        """Return list of metadata keys."""
        return list(self.meta_data.keys())

    def add_node(self, node):
        """Add a node to the network (ID must be unique)."""
        Logger.log(f"start network add_node(self, {node})")
        if self.get_node_by_id(node.get_id()) is not None:
            raise ValueError(f"Node with ID '{node.get_id()}' already exists in the network.")
        # ADD NODE TO NETWORK
        self.nodes.append(node)
        Logger.log("end network add_node(self, node)")

    def remove_node(self, node_id):
        """Remove a node by ID."""
        # FILTER OUT NODE BY ID
        self.nodes = [node for node in self.nodes if node.get_id() != node_id]

    def add_edge(self, edge):
        """Add a valid edge (existing distinct endpoints with attrs present)."""
        Logger.log("start add_edge()")

        # ENSURE EDGE IS NOT ALREADY IN NETWORK
        if self.get_edge_by_id(edge.get_id()) is not None:
            raise ValueError(f"Edge with ID '{edge.get_id()}' already exists in the network.")

        # ENSURE EDGE HAS REQUIRED ATTRIBUTES
        if not hasattr(edge, "n_from") or not hasattr(edge, "n_to"):
            Logger.log("Edge is missing 'n_from' or 'n_to'")
            raise ValueError("Edge must have 'n_from' and 'n_to' attributes.")

        # CHECK THAT 'n_from' AND 'n_to' ARE DIFFERENT
        if edge.n_from == edge.n_to:
            Logger.log(f"Invalid edge: 'n_from' ({edge.n_from}) and 'n_to' are the same")
            raise ValueError("'n_from' and 'n_to' cannot be the same node.")

        # ENSURE REFERENCED NODES EXIST IN NETWORK
        if self.get_node_by_id(edge.n_from) is None:
            Logger.log(f"Invalid 'n_from' reference: {edge.n_from} not found.")
            raise ValueError(f"'n_from' value {edge.n_from} does not exist in network.")

        if self.get_node_by_id(edge.n_to) is None:
            Logger.log(f"Invalid 'n_to' reference: {edge.n_to} not found.")
            raise ValueError(f"'n_to' value {edge.n_to} does not exist in network.")

        # ADD EDGE TO NETWORK
        self.edges.append(edge)
        Logger.log(f"Edge successfully added: {edge.n_from} -> {edge.n_to}")
        Logger.log("end add_edge()")



    def remove_edge(self, edge_id):
        """Remove an edge by ID and drop isolated nodes."""
        # FILTER OUT EDGE BY ID
        self.edges = [edge for edge in self.edges if edge.get_id() != edge_id]

        # AFTER REMOVING AN EDGE, REMOVE ANY NODES THAT ARE NO LONGER CONNECTED
        if self.edges:
            connected_node_ids = {edge.get_attribute("n_to") for edge in self.edges}.union(
                {edge.get_attribute("n_from") for edge in self.edges}
            )
            self.nodes = [node for node in self.nodes if node.get_id() in connected_node_ids]
        else:
            # NO EDGES LEFT -> NO INTERSECTIONS; CLEAR ALL NODES
            self.nodes = []

    def get_node_by_id(self, node_id):
        """Return node by ID or None."""
        for node in self.nodes:
            if node.get_id() == node_id:
                return node
        return None

    def get_edge_by_id(self, edge_id):
        """Return edge by ID or None."""
        for edge in self.edges:
            if edge.get_id() == edge_id:
                return edge
        return None
    
    def log_network(self):
        """Logs the current state of the network: nodes, edges, and metadata."""
        Logger.log("===== Network State =====")
        
        Logger.log("Nodes:")
        for node in self.nodes:
            Logger.log(f"{node.__dict__}")
        
        Logger.log("Edges:")
        for edge in self.edges:
            Logger.log(f"{edge.__dict__}")
        
        Logger.log("Meta Data:")
        for key, value in self.meta_data.items():
            Logger.log(f"{key}: {value}")
        
        Logger.log("=========================")

