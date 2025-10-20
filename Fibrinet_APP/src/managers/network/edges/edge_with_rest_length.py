from .base_edge import BaseEdge

class EdgeWithRestLength(BaseEdge):
    schema = {
    **BaseEdge.get_schema(),
    "rest_length": float
}