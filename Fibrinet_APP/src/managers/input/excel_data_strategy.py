from utils.logger.logger import Logger
from ...models.exceptions import InvalidInputDataError
from ..network.network_factory import NetworkFactory
import pandas as pd
from .data_processing_strategy import DataProcessingStrategy

class ExcelDataStrategy(DataProcessingStrategy):
    """Parse Excel input into a `Network`."""
    # INITIALIZES EXCELDATASTRATEGY
    def __init__(self):
        """No-op init with logging."""
        Logger.log(f"start ExcelDataStrategy __init__(self)")
        Logger.log(f"end ExcelDataStrategy __init__(self)")
    
    # PROCESS INPUT DATA.
    def process(self, input_data):
        """Read and split Excel file into nodes, edges, and metadata."""
        Logger.log(f"start Process __init__(self, {input_data})")

        # LOAD THE ENTIRE SHEET INTO A DATAFRAME WITHOUT ASSUMING ANY HEADER
        # THE DATAFRAME IS A TABLE THAT PANDAS WILL USE TO STORE THE EXCEL CONTENT
        df = pd.read_excel(input_data, header=None)

        # VARIABLES TO STORE THE TABLES, TRACK THE CURRENT TABLE, TRACK NUMBER OF TABLES, AND TRACK IS NEW TABLE
        tables = {}
        current_table = {}
        table_number = 0
        is_new_table = True  

        # Helper
        def add_current_table_to_tables():
            if current_table:
                if table_number == 0:
                    tables['nodes'] = current_table
                elif table_number == 1:
                    tables['edges'] = current_table
                elif table_number == 2:
                    meta_data = {}
                    for i in range(len(current_table.get(list(current_table.keys())[0]))):
                        key = current_table.get(list(current_table.keys())[0])[i]
                        value = current_table.get(list(current_table.keys())[1])[i]
                        meta_data[key] = value
                    tables['meta_data'] = meta_data

        # FOR EACH ROW IN DF
        for index, row in df.iterrows():
            # Blank row separates tables
            if pd.isna(row[0]): 
                # ADD CURRENT TABLE TO TABLES 
                add_current_table_to_tables()
                # SET IS NEW TABLE TO TRUE 
                is_new_table = True
                # SET CURRENT TABLE TO EMPTY
                current_table = {}
                # INCREMENT TABLE NUMBER
                table_number += 1
            else:
                # First row in a table is headers
                if is_new_table:  
                    # GET NON NAN VALUES AS HEADERS
                    headers = row.dropna().tolist()  

                    # FOR EACH CELL IN THE CURRENT ROW 
                    for header in headers:
                        # CREATE A NEW KEY IN CURRENT_TABLE AND MAKE THE DEFAULT VALUE A NEW EMPTY LIST
                        current_table[header] = []
                    # SET IS NEW TABLE TO FALSE
                    is_new_table = False

                # Otherwise treat as data rows
                else:  
                    # LOOP THROUGHT THE ROW AND FOR EACH VALUE ADD IT TO ITS CORRESPONDING HEADER KEY
                    for i, value in enumerate(row.dropna()):
                        current_table[headers[i]].append(value)

        # ADD LAST TABLE
        add_current_table_to_tables()

        # CREATE AND RETURN NETWORK
        try:
            network = NetworkFactory.create_network(tables)
            Logger.log(f"end Process __init__(self, input_data)")
            return network
        except ValueError as e:
            Logger.log(f"Error creating network: {e}", Logger.LogPriority.ERROR)
            raise InvalidInputDataError(f"Invalid input data: {e}")
    