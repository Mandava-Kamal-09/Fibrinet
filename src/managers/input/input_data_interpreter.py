from utils.logger.logger import Logger
from ...models.exceptions import UnsupportedFileTypeError
from .excel_data_strategy import ExcelDataStrategy
import os

class InputDataInterpreter:
    """
    Interprets input data and provides a DataProcessingStrategy.
    """
    # INITIALIZES DATAINTERPRETER
    def __init__(self):
        """
        Initializes the DataInterpreter.
        """
        Logger.log(f"start DataInterpreter __init__(self)")
        Logger.log(f"end DataInterpreter __init__(self)")
    
    # GET DATA PROCESSING STRATEGY
    def get_data_processing_strategy(self, input_data):
        """
        Returns the correct data processing strategy for the input data.
        """
        Logger.log(f"start get_data_processing_strategy __init__(self, {input_data})")

        if isinstance(input_data, str):
            # CHECK DOES FILE EXIST
            if not os.path.exists(input_data):
                Logger.log(f"File ({input_data}) Not Found", Logger.LogPriority.ERROR)
                raise FileNotFoundError(f"File ({input_data}) Not Found")

            file_extension = os.path.splitext(input_data)[1]
        else:
            # Assume it's a file-like object (BytesIO), default to .xlsx
            file_extension = '.xlsx'

        # DETERMINE THE TYPE OF FILE INPUT DATA IS AND RETURN DATA PROCESSING STRATEGY
        if file_extension == '.xlsx':
            Logger.log("This is an Excel file.")
            Logger.log(f"end DataInterpreter __init__(self, input_data)")
            return ExcelDataStrategy()

        else:
            Logger.log(f"Unsupported file type: {file_extension}")
            raise UnsupportedFileTypeError()


        
            
