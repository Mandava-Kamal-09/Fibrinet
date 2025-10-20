from ..network.networks.base_network import BaseNetwork

class ExportStrategy():
    def generate_export(self, network_state_history: list[BaseNetwork]):
        """Return a list[(filename, bytes)] for this export."""
        pass
