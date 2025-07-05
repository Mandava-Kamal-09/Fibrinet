from src.managers.input.input_manager import InputManager
from src.managers.view.view_manager import ViewManager
from src.managers.export.export_manager import ExportManager
from src.managers.network.network_manager import NetworkManager
from src.managers.analysis.analysis_manager import AnalysisManager
from utils.logger.logger import Logger
from utils.logger.local_file_strategy import LocalFileStrategy
from src.models.system_state import SystemState
from src.models.exceptions import StateTransitionError
from src.managers.network.networks.base_network import BaseNetwork

class SystemController:
    """
    Main controller interface responsible for managing system components such as input, view, export, 
    network, and logging. Ensures proper state transitions and handles requests.
    """
    # INITIALIZES SYSTEM CONTROLLER AND MANAGERS
    def __init__(self):
        """
        Initializes the system controller and its managers.
        Sets up the default CLI view.
        """

        # INITIALIZES MANAGERS
        Logger.log("start SystemControllerInterface__init__(self)")
        self.input_manager = InputManager()
        self.view_manager = ViewManager(self)
        self.export_manager = ExportManager()
        self.network_manager = NetworkManager()
        self.system_state = SystemState()
        
        Logger.log("SystemControllerInterface initialized.")
        Logger.log("end SystemControllerInterface__init__(self)")

    # HANDLES NETWORK INPUT
    def input_network(self, input_data):
        """
        Handles network input data.
        
        :param input_data: file path or BaseNetwork instance.
        """
        
        Logger.log(f"start input_network(self, {input_data})")
        try: 
            Logger.log("Setting network from input_manager network")
            self.network_manager.reset_network_state_manager()
            if isinstance(input_data, BaseNetwork):
                network = input_data
            else:
                network = self.input_manager.get_network(input_data)
            self.network_manager.set_network(network)
            self.network_manager.state_manager.add_new_network_state(network)
        except Exception as ex: 
            raise ex
        Logger.log(f"Is a network loaded? {bool(self.network_manager.network)}")
        if self.network_manager.network:
            Logger.log("Setting system state to network loaded true")
            self.system_state.network_loaded = True
        
        Logger.log(f"end input_network(self, input_data)")     
    
    # ADD NDOE
    def add_node(self, node):
        """
        Adds a node to the network if the network is loaded.
        """
        Logger.log(f"start controller add_node(self, {node})")
        if self.system_state.network_loaded:
            self.network_manager.add_node(node)
        else:
            Logger.log("StateTransitionError: Cannot add node, network not loaded.", Logger.LogPriority.ERROR)
            raise StateTransitionError()
        Logger.log(f"end controller add_node(self, node)")

    # ADD EDGE
    def add_edge(self, edge):
        """
        Adds an edge to the network if the network is loaded.
        """
        Logger.log(f"start controller add_edge(self, {edge})")
        if self.system_state.network_loaded:
            self.network_manager.add_edge(edge)
        else:
            Logger.log("StateTransitionError: Cannot add edge, network not loaded.", Logger.LogPriority.ERROR)
            raise StateTransitionError()
        Logger.log(f"end controller add_edge(self, edge)")


    # DEGRADES A SPECIFIED NETWORK EDGE IF STATE ALLOWS
    def degrade_edge(self, edge_id):
        """
        Degrades a specified network edge if the network is loaded.
        
        :param edge: The edge to be degraded.
        :raises StateTransitionError: If the network is not loaded.
        """
        
        Logger.log(f"start degrade_edge(self, {edge_id})")
        if self.system_state.network_loaded:
            self.network_manager.degrade_edge(edge_id)
        else:
            Logger.log("StateTransitionError: Cannot modify network, network not loaded.", Logger.LogPriority.ERROR)
            raise StateTransitionError()
        Logger.log(f"end degrade_edge(self, {edge_id})")

    # DEGRADES A SPECIFIED NETWORK NODE IF STATE ALLOWS
    def degrade_node(self, node_id):
        """
        Degrades a specified network node if the network is loaded.
        
        :param node: The node to be degraded.
        :raises StateTransitionError: If the network is not loaded.
        """
        
        Logger.log(f"start degrade_node(self, {node_id})")
        if self.system_state.network_loaded:
            self.network_manager.degrade_node(node_id)
        else:
            Logger.log("StateTransitionError: Cannot modify network, network not loaded.", Logger.LogPriority.ERROR)
            raise StateTransitionError()
        Logger.log(f"end degrade_node(self, {node_id})")

    # UNDOES THE LAST NETWORK DEGRADATION IF STATE ALLOWS
    def undo_degradation(self):
        """
        Undoes the last degradation applied to the network if the network is loaded.
        
        :raises StateTransitionError: If the network is not loaded.
        """
        
        Logger.log(f"start undo_degradation(self)")
        if self.system_state.network_loaded:
            self.network_manager.undo_degradation()
        else:
            Logger.log("StateTransitionError: Cannot modify network, network not loaded.", Logger.LogPriority.ERROR)
            raise StateTransitionError()
        Logger.log(f"end undo_degradation(self)")

    # REDOES THE LAST NETWORK DEGRADATION IF STATE ALLOWS
    def redo_degradation(self):
        """
        Redoes the last degradation applied to the network if the network is loaded.
        
        :raises StateTransitionError: If the network is not loaded.
        """
        
        Logger.log(f"start redo_degradation(self)")

        if self.system_state.network_loaded:
            self.network_manager.redo_degradation()
        else:
            Logger.log("StateTransitionError: Cannot modify network, network not loaded.", Logger.LogPriority.ERROR)
            raise StateTransitionError()
        Logger.log(f"end redo_degradation(self)")
    
    # SETS DEGRADATION ENGINE STRATEGY
    def set_degradation_engine_strategy(self, degradation_engine_strategy):
        """
        Sets degradation engine strategy of network manager. 
        
        :param degradation_engine_strategy: Concrete strategy instance of DegradationEngineStrategy
        :raises StateTransitionError: If the network is not loaded.
        """
        
        Logger.log(f"start set_degradation_engine_strategy(self, {degradation_engine_strategy})")
        self.network_manager.set_degradation_engine_strategy(degradation_engine_strategy)
        Logger.log(f"end set_degradation_engine_strategy(self, {degradation_engine_strategy})")

    # EXPORTS DATA IF NETWORK IS LOADED AND MODIFIED
    def export_data(self, export_request):
        """
        Exports data if the network is loaded and has been modified.
        
        :param export_request: Request specifying export details.
        :raises StateTransitionError: If the state is invalid for export.
        """
        Logger.log(f"start export_data(self, {export_request})")
        if self.system_state.network_loaded and not self.network_manager.state_manager.export_disabled:
            self.export_manager.handle_export_request(self.network_manager.state_manager.network_state_history, export_request)
            Logger.log("Export request processed successfully.")
        else:
            Logger.log("StateTransitionError: Cannot export data, invalid state.", Logger.LogPriority.ERROR)
            raise StateTransitionError()
        Logger.log(f"end export_data(self, export_request)")

    # SUBMITS A VIEW REQUEST TO THE VIEW MANAGER
    def initiate_view(self, view_strategy):
        """
         Submits a view request to the view manager.
        
        :param view: Type of view requested (e.g., CLI, etc.).
        """
        
        Logger.log(f"start initiate_view(self, {view_strategy})")
        self.view_manager.initiate_view_strategy(view_strategy, self)
        Logger.log(f"end initiate_view(self, view_strategy)")

    # CONFIGURES Logger BASED ON PROVIDED SETTINGS
    def configure_Logger(self, enabled, **kwargs):
        """
        Configures the Logger with a specified storage strategy.
        
        :param kwargs: Dictionary containing storage strategy option and details.
        :raises ValueError: If storage strategy is invalid.
        """
        

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

    def run_simulation(self):
        """
        Runs the network simulation.
        """
        Logger.log("start run_simulation(self)")
        if self.system_state.network_loaded:
            self.network_manager.relax_network()
            Logger.log("Simulation run successfully.")
        else:
            Logger.log("StateTransitionError: Cannot run simulation, network not loaded.", Logger.LogPriority.ERROR)
            raise StateTransitionError("Cannot run simulation, network not loaded.")
        Logger.log("end run_simulation(self)")

    def analyze_network(self):
        """
        Analyzes the current network state.
        """
        Logger.log("start analyze_network(self)")
        if self.system_state.network_loaded:
            analysis = self.analysis_manager.analyze_network(self.network_manager.get_network())
            Logger.log("Analysis complete.")
            return analysis
        else:
            Logger.log("StateTransitionError: Cannot analyze network, network not loaded.", Logger.LogPriority.ERROR)
            raise StateTransitionError("Cannot analyze network, network not loaded.")

    def update_parameters(self, params):
        """
        Updates network parameters.
        """
        Logger.log(f"start update_parameters(self, {params})")
        if self.system_state.network_loaded:
            # This is a placeholder for the actual implementation
            # For now, we'll just log the parameters
            Logger.log(f"Updating parameters with: {params}")
            print(f"Parameters updated with: {params}")
        else:
            Logger.log("StateTransitionError: Cannot update parameters, network not loaded.", Logger.LogPriority.ERROR)
            raise StateTransitionError("Cannot update parameters, network not loaded.")
        Logger.log(f"end update_parameters(self, {params})")


        
