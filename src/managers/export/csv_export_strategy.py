from .data_export_strategy import DataExportStrategy
import pandas as pd
from ..network.networks.base_network import BaseNetwork
from io import StringIO
from utils.logger.logger import Logger
import time

class CsvExportStrategy(DataExportStrategy):
    """
    Exports network data to CSV files.
    """
    def generate_export(self, network_state_history: list[BaseNetwork]):
        """
        Generates CSV files for nodes, edges, and metadata for each network state.

        :param network_state_history: A list of network states to export.
        :return: A list of tuples, where each tuple contains a filename and the file content.
        """
        files = []
        Logger.log("Starting CSV export generation")

        for idx, network in enumerate(network_state_history):
            Logger.log(f"Processing network state {idx} for CSV export")

            nodes_data = [
                {key: value for key, value in node.get_attributes().items() if key != 'attributes'}
                for node in network.get_nodes()
            ]
            edges_data = [
                {key: value for key, value in edge.get_attributes().items() if key != 'attributes'}
                for edge in network.get_edges()
            ]
            meta_data = network.get_meta_data()

            nodes_df = pd.DataFrame(nodes_data)
            edges_df = pd.DataFrame(edges_data)
            meta_df = pd.DataFrame(list(meta_data.items()), columns=["meta_key", "meta_value"])

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Nodes CSV
            if not nodes_df.empty:
                nodes_buffer = StringIO()
                nodes_df.to_csv(nodes_buffer, index=False)
                nodes_filename = f"nodes_data_{timestamp}_{idx+1}.csv"
                files.append((nodes_filename, nodes_buffer.getvalue().encode('utf-8')))

            # Edges CSV
            if not edges_df.empty:
                edges_buffer = StringIO()
                edges_df.to_csv(edges_buffer, index=False)
                edges_filename = f"edges_data_{timestamp}_{idx+1}.csv"
                files.append((edges_filename, edges_buffer.getvalue().encode('utf-8')))

            # Meta CSV
            if not meta_df.empty:
                meta_buffer = StringIO()
                meta_df.to_csv(meta_buffer, index=False)
                meta_filename = f"meta_data_{timestamp}_{idx+1}.csv"
                files.append((meta_filename, meta_buffer.getvalue().encode('utf-8')))

            Logger.log(f"CSV export generated for network state {idx}.")

        Logger.log("CSV export generation completed")
        return files