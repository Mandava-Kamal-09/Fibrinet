from .base_node import BaseNode

class Node2D(BaseNode):
    schema = {
        **BaseNode.get_schema(),
        "n_x": float,
        "n_y": float
    }

