from utils.logger.logger import Logger

class DegradationEngineStrategy():
    """Interface for network degradation engines."""

    # DEGRADATION ENGINE STRATEGY INITILIZATION
    def __init__(self):
        """No-op init with logging."""
        Logger.log("start DegradationEngineStrategy__init__(self)")
        Logger.log("end DegradationEngineStrategy__init__(self)")

    # DEGRADE EDGE
    def degrade_edge(self, network, edge_id):
        """Return a new network with the given edge removed."""
        raise NotImplementedError()
    
    # DEGRADE NODE
    def degrade_node(self, network, node_id):
        """Return a new network with the given node removed."""
        raise NotImplementedError()
    
    # RELAX NETWORK
    def relax_network(self, network):
        """Relax network to equilibrium and return it (strategy-defined)."""
        raise NotImplementedError()
