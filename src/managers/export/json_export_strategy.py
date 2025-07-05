from .data_export_strategy import DataExportStrategy
import json
from ..network.networks.base_network import BaseNetwork
from utils.logger.logger import Logger
import time

class JsonExportStrategy(DataExportStrategy):
    """
    Exports network data to JSON files.
    """
    def generate_export(self, network_state_history: list[BaseNetwork]):
        """
        Generates a JSON file for each network state in the history.

        :param network_state_history: A list of network states to export.
        :return: A list of tuples, where each tuple contains a filename and the file content.
        """
        files = []
        Logger.log("Starting JSON export generation")

        for idx, network in enumerate(network_state_history):
            Logger.log(f"Processing network state {idx} for JSON export")

            nodes_data = [
                {key: value for key, value in node.get_attributes().items() if key != 'attributes'}
                for node in network.get_nodes()
            ]
            edges_data = [
                {key: value for key, value in edge.get_attributes().items() if key != 'attributes'}
                for edge in network.get_edges()
            ]
            meta_data = network.get_meta_data()

            export_data = {
                "meta_data": meta_data,
                "nodes": nodes_data,
                "edges": edges_data
            }

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            unique_filename = f"network_data_{timestamp}_{idx+1}.json"
            
            file_content = json.dumps(export_data, indent=4).encode('utf-8')
            files.append((unique_filename, file_content))

            Logger.log(f"JSON export generated for network state {idx}. Saved as {unique_filename}")

        Logger.log("JSON export generation completed")
        return files