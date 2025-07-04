from .export_strategy import ExportStrategy
from ..network.networks.base_network import BaseNetwork


class DataExportStrategy(ExportStrategy):
    def generate_export(self, network_state_history: list[BaseNetwork]):
        """Concrete strategies must implement this method."""
        pass    
