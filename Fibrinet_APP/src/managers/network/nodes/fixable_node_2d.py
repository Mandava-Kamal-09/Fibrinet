from .node_2d import Node2D

class FixableNode2D(Node2D):
    schema = {
    **Node2D.get_schema(),
    "is_fixed": bool
}

