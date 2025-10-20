from .export_strategy import ExportStrategy
from ..network.networks.base_network import BaseNetwork

class ImageExportStrategy(ExportStrategy):
    def generate_export(self, network_state_history: list[BaseNetwork]):
        """Implemented by concrete image exporters."""
        pass