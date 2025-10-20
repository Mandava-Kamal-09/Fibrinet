from utils.logger.logger import Logger
from .degradation_engine.no_physics import NoPhysics
from .degradation_engine.two_dimensional_spring_force_degradation_engine_without_biomechanics import TwoDimensionalSpringForceDegradationEngineWithoutBiomechanics
from .degradation_engine.degradation_engine_strategy import DegradationEngineStrategy
from .network_state_manager import NetworkStateManager

class NetworkManager:
    """Manage network operations and state transitions."""
    def __init__(self):
        """Initialize manager, state, and default strategy."""
        Logger.log(f"start NetworkManager __init__(self)")
        self.network = None
        self.state_manager = NetworkStateManager()
        self.degradation_engine_strategy = TwoDimensionalSpringForceDegradationEngineWithoutBiomechanics()
        Logger.log(f"end NetworkManager __init__(self)")

    def set_network(self, network):
        """Set the active network."""
        Logger.log(f"set_network: assigning active network")
        self.network = network
        Logger.log(f"set_network: done")

    def get_base_network(self):
        """Return the base network from the state manager."""
        Logger.log(f"get_base_network_state")
        return self.state_manager.get_base_network_state()
    
    def get_network(self):
        """Return the latest network."""
        Logger.log(f"get_network")
        return self.network

    def add_node(self, node):
        """Add a node and persist state."""
        Logger.log(f"start network manager add_node(self, {node})")
        self.network.add_node(node)
        self.state_manager.add_new_network_state(self.network)
        Logger.log(f"end network manager add_node(self, node)")

    def add_edge(self, edge):
        """Add an edge and persist state."""
        Logger.log(f"start add_edge(self, {edge})")
        self.network.add_edge(edge)
        self.state_manager.add_new_network_state(self.network)
        Logger.log(f"end add_edge(self, edge)")

    def update_degradation_engine_strategy(self):
        """Pick strategy based on whether the network is newly created."""
        Logger.log(f"start update_degradation_engine_strategy(self)")
        if self.state_manager.is_new_network:
            Logger.log("Network is new, using NoPhysics strategy.")
            self.set_degradation_engine_strategy("nophysics")
        else:
            Logger.log("Network is not new, using TwoDimensionalSpringForceDegradationEngineWithoutBiomechanics strategy.")
            self.set_degradation_engine_strategy("twodimensionalspringforcedegradationenginewithoutbiomechanics")
        
    def degrade_edge(self, edge_id):
        """Degrade an edge and persist state."""
        Logger.log(f"start degrade_edge(self, {edge_id})")
        self.network = self.degradation_engine_strategy.degrade_edge(self.network, edge_id)
        self.state_manager.add_new_network_state(self.network)
        Logger.log(f"end degrade_edge(self, edge_id)")

    def degrade_node(self, node_id):
        """Degrade a node and persist state."""
        Logger.log(f"start degrade_node(self, {node_id})")
        self.network = self.degradation_engine_strategy.degrade_node(self.network, node_id)
        self.state_manager.add_new_network_state(self.network)
        Logger.log(f"end degrade_node(self, node_id)")

    def undo_degradation(self):
        """Undo last change and update current network."""
        Logger.log(f"start undo_degradation(self)")
        self.state_manager.undo_last_network_state()
        self.network = self.state_manager.current_state
        Logger.log(f"end undo_degradation(self)")

    def redo_degradation(self):
        """Redo last undone change and update current network."""
        Logger.log(f"start redo_degradation(self)")
        self.state_manager.redo_last_network_state()
        self.network = self.state_manager.current_state
        Logger.log(f"end redo_degradation(self)")

    def modify_network_properties(self, network_properties):
        """Placeholder for network property updates."""
        Logger.log(f"start modify_network_properties(self, {network_properties})")
        Logger.log(f"end modify_network_properties(self, network_properties)")

    def set_degradation_engine_strategy(self, degradation_engine_strategy):
        """Set degradation engine strategy by key (case-insensitive)."""
        Logger.log(f"start set_degradation_engine_strategy(self, {degradation_engine_strategy})")

        degradation_engine_strategies = {
            "nophysics": NoPhysics,
            "twodimensionalspringforcedegradationenginewithoutbiomechanics": TwoDimensionalSpringForceDegradationEngineWithoutBiomechanics
        }

        if not degradation_engine_strategy:
            raise Exception("Invalid Degradation Engine Strategy")

        strategy_key = degradation_engine_strategy.lower()

        if strategy_key not in degradation_engine_strategies:
            raise Exception(f"Invalid Degradation Engine Strategy: '{degradation_engine_strategy}'")

        self.degradation_engine_strategy = degradation_engine_strategies[strategy_key]() 

        Logger.log("end set_degradation_engine_strategy(self, degradation_engine_strategy)")

    def reset_network_state_manager(self):
        """Reset the state manager and clear history."""
        Logger.log(f"start reset_network_state(self)")
        self.state_manager.reset_network_state()
        Logger.log(f"end reset_network_state(self)")

    def reset_network_and_state(self):
        """Clear the network and reinitialize state manager."""
        Logger.log(f"start reset_network_and_state(self)")
        self.network = None # Set the network attribute to None
        self.state_manager = NetworkStateManager()
        Logger.log(f"end reset_network_and_state(self)")

    def relax_network(self):
        """Relax the current network using the active strategy."""
        Logger.log(f"start network manager relax_network(self)")
        if self.network is None:
            raise Exception("No network to relax.")
        
        # CHECK THE EXPORT CONDITION IF ITS TRUE THEN CONTINUE OTHERWISE RAISE EXCEPTION
        if self.state_manager._check_export_condition():
            raise Exception("Cannot relax network: export condition not met.")
        
        # SET DEGRADATION ENGINE STRATEGY TO TWO DIMENSIONAL SPRING FORCE DEGRADATION ENGINE WITHOUT BIOMECHANICS
        self.set_degradation_engine_strategy("twodimensionalspringforcedegradationenginewithoutbiomechanics")
        # RELAX THE NETWORK
        self.degradation_engine_strategy.relax_network(self.network)
        # ADD THE RELAXED NETWORK TO THE STATE MANAGER
        self.state_manager.add_new_network_state(self.network)
        # UPDATE THE DEGRADATION ENGINE STRATEGY
        self.update_degradation_engine_strategy()
        Logger.log(f"end network manager  relax_network(self)")