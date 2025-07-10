from utils.logger.logger import Logger
from ...models.exceptions import InvalidInputDataError
from ..network.network_factory import NetworkFactory
import pandas as pd
from .data_processing_strategy import DataProcessingStrategy

class ExcelDataStrategy(DataProcessingStrategy):
    """
    Handles data from an excel file.
    """
    # INITIALIZES EXCELDATASTRATEGY
    def __init__(self):
        """
        Initializes the ExcelDataStrategy.
        """
        Logger.log(f"start ExcelDataStrategy __init__(self)")
        Logger.log(f"end ExcelDataStrategy __init__(self)")
    
    # PROCESS INPUT DATA.
    def process(self, input_data):
        """
        Process input data to extract network data. Returns network object.
        """
        Logger.log(f"start Process __init__(self, {input_data})")

        try:
            # Read the Excel file
            xls = pd.ExcelFile(input_data)

            # Read the nodes, edges, and metadata from the sheets
            nodes_df = pd.read_excel(xls, 'Sheet1', skiprows=0, nrows=3, header=0)
            edges_df = pd.read_excel(xls, 'Sheet1', skiprows=4, nrows=2, header=0)
            meta_df = pd.read_excel(xls, 'Sheet1', skiprows=7, nrows=2, header=0)

            # Convert the dataframes to dictionaries
            nodes = nodes_df.to_dict(orient='list')
            edges = edges_df.to_dict(orient='list')
            meta_data = meta_df.set_index('key')['value'].to_dict()

            # Create the tables dictionary
            tables = {
                'nodes': nodes,
                'edges': edges,
                'meta_data': meta_data
            }

            # Create and return the network
            network = NetworkFactory.create_network(tables)
            Logger.log(f"end Process __init__(self, input_data)")
            return network
        except Exception as e:
            Logger.log(f"Error processing Excel file: {e}", Logger.LogPriority.ERROR)
            raise InvalidInputDataError(f"Error processing Excel file: {e}")
    