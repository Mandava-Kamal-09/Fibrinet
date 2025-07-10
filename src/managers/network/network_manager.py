from utils.logger.logger import Logger
from .degradation_engine.no_physics import NoPhysics
from .degradation_engine.two_dimensional_spring_force_degradation_engine_without_biomechanics import TwoDimensionalSpringForceDegradationEngineWithoutBiomechanics
from .degradation_engine.degradation_engine_strategy import DegradationEngineStrategy
from .network_state_manager import NetworkStateManager

class NetworkManager:
    """
    Manages network operations and handles logging related to network events.
    """
    # NETWORKMANAGER INITIALIZATION
    def __init__(self):
        """
        Initializes the NetworkManager instance
        """
        Logger.log(f"start NetworkManager __init__(self)")
        self.network = None
        self.state_manager = NetworkStateManager()
        self.degradation_engine_strategy = TwoDimensionalSpringForceDegradationEngineWithoutBiomechanics()
        Logger.log(f"end NetworkManager __init__(self)")

    # SET NETWORK
    def set_network(self, network):
        """
        Sets the network attribtue to network. 

        Args: 
        network: this is a Network object. 
        """
        Logger.log(f"start set_network(self, {network})")
        self.network = network
        Logger.log(f"NetworkManager.network set to: {self.network}")
        Logger.log(f"end set_network_(self, network)")

    # GET BASE NETWORK
    def get_base_network(self):
        """
        Gets the network attribtue.  
        """
        Logger.log(f"start set_network(self)")
        Logger.log(f"end set_network_(self, network)")
        return self.state_manager.get_base_network_state()
    
    # GET LATEST NETWORK
    def get_network(self):
        """
        Gets the latest network attribtue from network state manager.  
        """
        Logger.log(f"start get_network(self)")
        Logger.log(f"end get_network_(self, network)")
        return self.network

    # ADD NODE
    def add_node(self, node):
        """
        Adds a node directly to the network.
        """
        Logger.log(f"start network manager add_node(self, {node})")
        self.network.add_node(node)
        self.state_manager.add_new_network_state(self.network)
        Logger.log(f"end network manager add_node(self, node)")

    # ADD EDGE
    def add_edge(self, edge):
        """
        Adds an edge directly to the network.
        """
        Logger.log(f"start add_edge(self, {edge})")
        self.network.add_edge(edge)
        self.state_manager.add_new_network_state(self.network)
        Logger.log(f"end add_edge(self, edge)")

    def update_degradation_engine_strategy(self):
        """
        Updates the degradation engine strategy based on the network state.
        """
        Logger.log(f"start update_degradation_engine_strategy(self)")
        if self.state_manager.is_new_network:
            Logger.log("Network is new, using NoPhysics strategy.")
            self.set_degradation_engine_strategy("nophysics")
        else:
            Logger.log("Network is not new, using TwoDimensionalSpringForceDegradationEngineWithoutBiomechanics strategy.")
            self.set_degradation_engine_strategy("twodimensionalspringforcedegradationenginewithoutbiomechanics")
        
    # DEGRADE EDGE
    def degrade_edge(self, edge_id):
        """  
        """
        Logger.log(f"start degrade_edge(self, {edge_id})")
        self.network = self.degradation_engine_strategy.degrade_edge(self.network, edge_id)
        self.state_manager.add_new_network_state(self.network)
        Logger.log(f"end degrade_edge(self, edge_id)")

    # DEGRADE NODE
    def degrade_node(self, node_id):
        """  
        """
        Logger.log(f"start degrade_node(self, {node_id})")
        self.network = self.degradation_engine_strategy.degrade_node(self.network, node_id)
        self.state_manager.add_new_network_state(self.network)
        Logger.log(f"end degrade_node(self, node_id)")

    # UNDO DEGRADATION
    def undo_degradation(self):
        """ 
        """
        Logger.log(f"start undo_degradation(self)")
        self.state_manager.undo_last_network_state()
        self.network = self.state_manager.current_state
        Logger.log(f"end undo_degradation(self)")

    # REDO DEGRADATION
    def redo_degradation(self):
        """ 
        """
        Logger.log(f"start redo_degradation(self)")
        self.state_manager.redo_last_network_state()
        self.network = self.state_manager.current_state
        Logger.log(f"end redo_degradation(self)")

    # MODIFY NETWORK PROPERTIES
    def modify_network_properties(self, network_properties):
        """
        """
        Logger.log(f"start modify_network_properties(self, {network_properties})")
        Logger.log(f"end modify_network_properties(self, network_properties)")

    def set_degradation_engine_strategy(self, degradation_engine_strategy):
        """
        Sets the degradation engine strategy based on input string (case-insensitive).
        """
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

    # RESET NETWORK STATE
    def reset_network_state_manager(self):
        """
        Resets the network state manager and clears the history.
        """
        Logger.log(f"start reset_network_state(self)")
        self.state_manager.reset_network_state()
        Logger.log(f"end reset_network_state(self)")

    def reset_network_and_state(self):
        """
        Resets the network manager's network attribute to None and
        clears the network state manager's history.
        """
        Logger.log(f"start reset_network_and_state(self)")
        self.network = None # Set the network attribute to None
        self.state_manager = NetworkStateManager()
        Logger.log(f"end reset_network_and_state(self)")

    def relax_network(self):
        """
        Relaxes the current network using the degradation engine strategy.
        """
        Logger.log(f"start network manager relax_network(self)")
        if self.network is None:
            raise Exception("No network to relax.")
        
        # CHECK THE EXPORT CONDITION IF ITS TRUE THEN CONTINUE OTHERWISE RAISE EXCEPTION
        if self.state_manager._check_export_condition():
            raise StateTransitionError("Cannot relax network: export condition not met.")
        
        # SET DEGRADATION ENGINE STRATEGY TO TWO DIMENSIONAL SPRING FORCE DEGRADATION ENGINE WITHOUT BIOMECHANICS
        self.set_degradation_engine_strategy("twodimensionalspringforcedegradationenginewithoutbiomechanics")
        # RELAX THE NETWORK
        self.degradation_engine_strategy.relax_network(self.network)
        # ADD THE RELAXED NETWORK TO THE STATE MANAGER
        self.state_manager.add_new_network_state(self.network)
        # UPDATE THE DEGRADATION ENGINE STRATEGY
        self.update_degradation_engine_strategy()
        Logger.log(f"end network manager  relax_network(self)")