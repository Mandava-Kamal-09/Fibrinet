from src.managers.input.input_manager import InputManager
from src.managers.view.view_manager import ViewManager
from src.managers.export.export_manager import ExportManager
from src.managers.network.network_manager import NetworkManager
from utils.logger.logger import Logger
from utils.logger.local_file_strategy import LocalFileStrategy
from src.models.system_state import SystemState
from src.models.exceptions import StateTransitionError
from src.managers.network.networks.base_network import BaseNetwork

class SystemController:
    """Coordinates input, view, export, network, and logging layers."""
    def __init__(self):
        """Initialize managers and shared state."""
        Logger.log("start SystemControllerInterface__init__(self)")
        self.input_manager = InputManager()
        self.view_manager = ViewManager(self)
        self.export_manager = ExportManager()
        self.network_manager = NetworkManager()
        self.system_state = SystemState()
        
        Logger.log("SystemControllerInterface initialized.")
        Logger.log("end SystemControllerInterface__init__(self)")

    def input_network(self, input_data):
        """Load a network from a file path or `BaseNetwork`."""
        
        Logger.log(f"input_network: {input_data}")
        try: 
            self.network_manager.reset_network_state_manager()
            if isinstance(input_data, BaseNetwork):
                network = input_data
            else:
                network = self.input_manager.get_network(input_data)
            self.network_manager.set_network(network)
            self.network_manager.state_manager.add_new_network_state(network)
        except Exception as ex: 
            raise ex
        Logger.log(f"input_network: loaded={bool(self.network_manager.network)}")
        if self.network_manager.network:
            self.system_state.network_loaded = True
            original_constant = network.meta_data.get("spring_stiffness_constant")
            self.system_state.original_spring_constant = original_constant
            self.system_state.spring_stiffness_constant = original_constant
            Logger.log(f"Original spring constant stored: {original_constant}")
        
        Logger.log(f"end input_network(self, input_data)")     
    
    def get_spring_constant(self):
        """Return current spring stiffness constant, or None if not loaded."""
        Logger.log(f"start get_spring_constant(self)")
        if self.system_state.network_loaded and self.network_manager.network:
            spring_constant = self.network_manager.network.meta_data.get("spring_stiffness_constant")
            Logger.log(f"Current spring constant: {spring_constant}")
            Logger.log(f"end get_spring_constant(self)")
            return spring_constant
        Logger.log("No network loaded, returning None")
        Logger.log(f"end get_spring_constant(self)")
        return None
    
    def set_spring_constant(self, new_value):
        """Update the spring constant and relax the network."""
        Logger.log(f"start set_spring_constant(self, {new_value})")
        if self.system_state.network_loaded:
            # Update current network's metadata
            self.network_manager.network.update_meta_data("spring_stiffness_constant", new_value)
            # Update system state
            self.system_state.spring_stiffness_constant = new_value
            # Relax the network with the new spring constant to immediately reflect physics changes
            self.network_manager.relax_network()
            Logger.log(f"spring_constant set to {new_value}; network relaxed")
        else:
            Logger.log("StateTransitionError: Cannot modify spring constant, network not loaded.", Logger.LogPriority.ERROR)
            raise StateTransitionError("Cannot modify spring constant, network not loaded.")
        Logger.log(f"end set_spring_constant(self, new_value)")
    
    def get_original_spring_constant(self):
        """Return original spring constant from input metadata, if available."""
        Logger.log(f"start get_original_spring_constant(self)")
        original = self.system_state.original_spring_constant
        Logger.log(f"Original spring constant: {original}")
        Logger.log(f"end get_original_spring_constant(self)")
        return original
    
    def reset_spring_constant(self):
        """Reset spring constant to the original value from input metadata."""
        Logger.log(f"start reset_spring_constant(self)")
        if self.system_state.network_loaded:
            original = self.system_state.original_spring_constant
            if original is not None:
                self.set_spring_constant(original)
                Logger.log(f"Spring constant reset to original value: {original}")
            else:
                Logger.log("No original spring constant available", Logger.LogPriority.ERROR)
                raise StateTransitionError("No original spring constant available")
        else:
            Logger.log("StateTransitionError: Cannot reset spring constant, network not loaded.", Logger.LogPriority.ERROR)
            raise StateTransitionError("Cannot reset spring constant, network not loaded.")
        Logger.log(f"end reset_spring_constant(self)")
    
    # ADD NODE
    def add_node(self, node):
        """Add a node if a network is loaded."""
        Logger.log(f"start controller add_node(self, {node})")
        if self.system_state.network_loaded:
            self.network_manager.add_node(node)
        else:
            Logger.log("StateTransitionError: Cannot add node, network not loaded.", Logger.LogPriority.ERROR)
            raise StateTransitionError()
        Logger.log(f"end controller add_node(self, node)")

    def add_edge(self, edge):
        """Add an edge if a network is loaded."""
        Logger.log(f"start controller add_edge(self, {edge})")
        if self.system_state.network_loaded:
            self.network_manager.add_edge(edge)
        else:
            Logger.log("StateTransitionError: Cannot add edge, network not loaded.", Logger.LogPriority.ERROR)
            raise StateTransitionError()
        Logger.log(f"end controller add_edge(self, edge)")


    def degrade_edge(self, edge_id):
        """Degrade a network edge if a network is loaded."""
        
        Logger.log(f"start degrade_edge(self, {edge_id})")
        if self.system_state.network_loaded:
            self.network_manager.degrade_edge(edge_id)
        else:
            Logger.log("StateTransitionError: Cannot modify network, network not loaded.", Logger.LogPriority.ERROR)
            raise StateTransitionError()
        Logger.log(f"end degrade_edge(self, {edge_id})")

    def degrade_node(self, node_id):
        """Degrade a network node if a network is loaded."""
        
        Logger.log(f"start degrade_node(self, {node_id})")
        if self.system_state.network_loaded:
            self.network_manager.degrade_node(node_id)
        else:
            Logger.log("StateTransitionError: Cannot modify network, network not loaded.", Logger.LogPriority.ERROR)
            raise StateTransitionError()
        Logger.log(f"end degrade_node(self, {node_id})")

    def undo_degradation(self):
        """Undo the last degradation if a network is loaded."""
        
        Logger.log(f"start undo_degradation(self)")
        if self.system_state.network_loaded:
            self.network_manager.undo_degradation()
        else:
            Logger.log("StateTransitionError: Cannot modify network, network not loaded.", Logger.LogPriority.ERROR)
            raise StateTransitionError()
        Logger.log(f"end undo_degradation(self)")

    def redo_degradation(self):
        """Redo the last undone degradation if a network is loaded."""
        
        Logger.log(f"start redo_degradation(self)")

        if self.system_state.network_loaded:
            self.network_manager.redo_degradation()
        else:
            Logger.log("StateTransitionError: Cannot modify network, network not loaded.", Logger.LogPriority.ERROR)
            raise StateTransitionError()
        Logger.log(f"end redo_degradation(self)")
    
    def set_degradation_engine_strategy(self, degradation_engine_strategy):
        """Set the degradation engine strategy on the network manager."""
        
        Logger.log(f"start set_degradation_engine_strategy(self, {degradation_engine_strategy})")
        self.network_manager.set_degradation_engine_strategy(degradation_engine_strategy)
        Logger.log(f"end set_degradation_engine_strategy(self, {degradation_engine_strategy})")

    def export_data(self, export_request):
        """Export data if a network is loaded and export is allowed."""
        Logger.log(f"start export_data(self, {export_request})")
        if self.system_state.network_loaded and not self.network_manager.state_manager.export_disabled:
            self.export_manager.handle_export_request(self.network_manager.state_manager.network_state_history, export_request)
            Logger.log("Export request processed successfully.")
        else:
            Logger.log("StateTransitionError: Cannot export data, invalid state.", Logger.LogPriority.ERROR)
            raise StateTransitionError()
        Logger.log(f"end export_data(self, export_request)")

    def initiate_view(self, view_strategy):
        """Submit a view request to the view manager."""
        
        Logger.log(f"start initiate_view(self, {view_strategy})")
        self.view_manager.initiate_view_strategy(view_strategy, self)
        Logger.log(f"end initiate_view(self, view_strategy)")

    def configure_Logger(self, enabled, **kwargs):
        """Enable/disable logging and optionally set file storage strategy."""
        

        Logger.log(f"start configure_Logger(self, {enabled}, {kwargs})")
        if enabled: Logger.enable_logging()
        else: Logger.disable_logging()
        storage_strategy = kwargs.get("storage_strategy", None)
        if storage_strategy is not None:
            if storage_strategy == "file":
                file_location = kwargs.get("file_location", None)
                if not file_location:
                    raise ValueError("file_location must be provided for 'file' storage strategy.")
                
                # SETS FILE-BASED LOGGING STRATEGY
                Logger.set_log_storage_strategy(LocalFileStrategy(file_location))
                Logger.log(f"Logger set to file storage at {file_location}.")
        else:
            # DEFAULT ONLY ENABLES / DISABLES CURRENT LOGGING STRATEGY
            pass
        Logger.log(f"end configure_Logger(self, **kwargs)")


        
