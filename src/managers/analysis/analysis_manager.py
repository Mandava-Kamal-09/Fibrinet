import pandas as pd
from ..network.networks.base_network import BaseNetwork
from utils.logger.logger import Logger

class AnalysisManager:
    """
    Manages the analysis of network metrics.
    """
    def analyze_network(self, network: BaseNetwork):
        """
        Analyzes the network and returns key metrics as a formatted string.

        :param network: The network object to analyze.
        :return: A string containing the analysis results.
        """
        Logger.log("Starting network analysis")

        if not network:
            return "No network loaded to analyze."

        num_nodes = len(network.get_nodes())
        num_edges = len(network.get_edges())

        if num_nodes == 0:
            return "Network has no nodes."

        # Calculate node degrees
        node_degrees = {node.get_id(): 0 for node in network.get_nodes()}
        for edge in network.get_edges():
            if edge.get_attribute("n_from") in node_degrees:
                node_degrees[edge.get_attribute("n_from")] += 1
            if edge.get_attribute("n_to") in node_degrees:
                node_degrees[edge.get_attribute("n_to")] += 1
        
        avg_degree = sum(node_degrees.values()) / num_nodes if num_nodes > 0 else 0
        
        # Fixed vs non-fixed nodes
        fixed_nodes = sum(1 for node in network.get_nodes() if getattr(node, 'is_fixed', False))
        
        analysis_results = {
            "Number of nodes": num_nodes,
            "Number of edges": num_edges,
            "Average node degree": f"{avg_degree:.2f}",
            "Number of fixed nodes": fixed_nodes,
            "Number of non-fixed nodes": num_nodes - fixed_nodes
        }
        
        # Convert to a formatted string for display
        result_str = "Network Analysis Results:\n"
        for key, value in analysis_results.items():
            result_str += f"- {key}: {value}\n"
            
        Logger.log("Network analysis completed")
        return result_str