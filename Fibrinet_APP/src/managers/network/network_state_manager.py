from utils.logger.logger import Logger
from .networks.base_network import BaseNetwork

class NetworkStateManager:
    """Manage network state history with undo/redo and export guards."""
    # INITIALIZES THE NETWORKSTATEMANAGER
    def __init__(self):
        """Initialize empty history and default flags."""
        Logger.log(f"start NetworkStateManager__init__")
        self.network_state_history = []
        self.current_state = None
        self.current_network_state_index = 0
        self.undo_disabled = True
        self.redo_disabled = True
        self.export_disabled = True
        self.is_new_network = False
        Logger.log(f"end NetworkStateManager__init__")

    def log_network_history(self):
        """Log each network in history; mark current state."""
        Logger.log("start log_all_network_states()")

        if not self.network_state_history:
            Logger.log("No network states in history.")
            return

        for idx, network in enumerate(self.network_state_history):
            is_current = " (CURRENT)" if idx == self.current_network_state_index else ""
            Logger.log(f"--- Network State [{idx}]{is_current} ---")
            network.log_network()

        Logger.log("end log_all_network_states()")


    def log_network_state_manager_attributes(self):
        Logger.log("Logging NetworkStateManager attributes:")
        Logger.log(f"  network_state_history (length): {len(self.network_state_history)}")
        Logger.log(f"  current_state: {self.current_state}")
        Logger.log(f"  current_network_state_index: {self.current_network_state_index}")
        Logger.log(f"  undo_disabled: {self.undo_disabled}")
        Logger.log(f"  redo_disabled: {self.redo_disabled}")
        Logger.log(f"  export_disabled: {self.export_disabled}")


    def _check_export_condition(self):
        """Return True when export should be disabled based on state/content."""
        Logger.log("start _check_export_condition()")

        if not self.current_state:
            Logger.log("No current state — disabling export.")
            return True

        if getattr(self, "is_new_network", False):  # fallback to False if attribute is missing
            if len(self.current_state.get_nodes()) < 2:
                Logger.log("New network: Not enough nodes — disabling export.")
                return True
            if len(self.current_state.get_edges()) < 1:
                Logger.log("New network: Not enough edges — disabling export.")
                return True

        has_all_meta_data = True
        if self.current_state.schema and "meta_data" in self.current_state.schema:
            for key in self.current_state.schema["meta_data"]:
                if key not in self.current_state.meta_data or self.current_state.meta_data[key] is None:
                    Logger.log(f"Missing or None meta_data key: {key} — disabling export.")
                    has_all_meta_data = False
                    break

        Logger.log("end _check_export_condition()")
        return not has_all_meta_data



    def add_new_network_state(self, network_state: BaseNetwork):
        """Append a cloned network state and update flags/indexes."""
        Logger.log(f"start add_new_state({network_state})")
        self.network_state_history = self.network_state_history[:self.current_network_state_index + 1]

        import copy

        cloned_state = copy.deepcopy(network_state)
        self.network_state_history.append(cloned_state)
        self.current_state = cloned_state
        self.current_network_state_index = len(self.network_state_history) - 1
        self.undo_disabled = len(self.network_state_history) <= 1
        self.redo_disabled = True
        self.export_disabled = self._check_export_condition()
        self.log_network_state_manager_attributes()
        Logger.log("end add_new_state")

    def undo_last_network_state(self):
        """Revert to previous state if available."""
        Logger.log("start undo_last_state()")
        if self.undo_disabled or self.current_network_state_index == 0:
            return
        self.current_network_state_index -= 1
        self.current_state = self.network_state_history[self.current_network_state_index]
        self.redo_disabled = False
        if self.current_network_state_index == 0:
            self.undo_disabled = True
        self.export_disabled = self._check_export_condition()
        self.log_network_state_manager_attributes()

        Logger.log("end undo_last_state")

    def redo_last_network_state(self):
        """Advance to next state if available."""
        Logger.log("start redo_last_network_state()")
        if self.redo_disabled or self.current_network_state_index >= len(self.network_state_history) - 1:
            return
        self.current_network_state_index += 1
        self.current_state = self.network_state_history[self.current_network_state_index]
        self.undo_disabled = False
        if self.current_network_state_index == len(self.network_state_history) - 1:
            self.redo_disabled = True
        self.export_disabled = self._check_export_condition()
        
        Logger.log("end redo_last_network_state")

    def get_base_network_state(self):
        """Return the history's base state (index 0) or None."""
        Logger.log("start get_base_network_state()")
        if self.network_state_history:
            base_state = self.network_state_history[0]
            Logger.log("end get_base_network_state")
            return base_state
        
        Logger.log("end get_base_network_state - no states in history")
        return None
    
    def reset_network_state(self):
        """Clear history and reset flags to defaults."""
        Logger.log("start reset_network_state()")
        self.network_state_history = []
        self.current_state = None
        self.current_network_state_index = 0
        self.undo_disabled = True
        self.redo_disabled = True
        self.export_disabled = True
        
        Logger.log("end reset_network_state()")

