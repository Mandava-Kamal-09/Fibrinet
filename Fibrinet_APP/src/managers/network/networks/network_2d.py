from utils.logger.logger import Logger
from ..nodes.node_2d import Node2D
from ..edges.base_edge import BaseEdge
from .base_network import BaseNetwork

class Network2D(BaseNetwork):
    allowed_node_type = Node2D
    allowed_edge_type = BaseEdge
    schema = {
        "meta_data": ["spring_stiffness_constant"],
        "meta_data_types": {
        "spring_stiffness_constant": float
        },
        "node_attributes": allowed_node_type.get_schema(),
        "edge_attributes": allowed_edge_type.get_schema(),
    }

    def __init__(self, data):
        Logger.log(f"start Network2D __init__(self, {data})")
        meta_data = data.get("meta_data", {})
        nodes = data.get("nodes", []) 
        edges = data.get("edges", [])
        super().__init__(nodes=nodes, edges=edges, meta_data=meta_data, schema=Network2D.schema)
        Logger.log(f"end Network2D __init__(self, {data})")