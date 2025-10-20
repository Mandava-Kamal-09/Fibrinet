from utils.logger.logger import Logger
from ...models.exceptions import UnsupportedFileTypeError
from .excel_data_strategy import ExcelDataStrategy
import os

class InputDataInterpreter:
    """Pick a data processing strategy based on file type."""
    # INITIALIZES DATAINTERPRETER
    def __init__(self):
        """No-op init with logging."""
        Logger.log(f"start DataInterpreter __init__(self)")
        Logger.log(f"end DataInterpreter __init__(self)")
    
    # GET DATA PROCESSING STRATEGY
    def get_data_processing_strategy(self, input_data):
        """Return a strategy for `input_data` path."""
        Logger.log(f"start get_data_processing_strategy __init__(self, {input_data})")
        # CHECK DOES FILE EXIST
        if not os.path.exists(input_data):
            Logger.log(f"input file not found: {input_data}", Logger.LogPriority.ERROR)
            raise FileNotFoundError(f"Input file not found: {input_data}")

        file_size = os.path.getsize(input_data)
        file_name = os.path.basename(input_data)
        file_extension = os.path.splitext(input_data)[1]
        Logger.log(f"File details - Name: {file_name}, Size: {file_size} bytes, Extension: {file_extension}")

        # Pick strategy by extension
        if file_extension == '.xlsx':
            Logger.log("excel input detected")
            Logger.log(f"end DataInterpreter __init__(self, input_data)")
            return ExcelDataStrategy()

        else:
            Logger.log(f"unsupported file type: {file_extension}")
            raise UnsupportedFileTypeError(f"Unsupported file type: {file_extension}")


        
            
