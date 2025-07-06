from utils.logger.logger import Logger
from .input_data_interpreter import InputDataInterpreter
from ...models.exceptions import InvalidInputDataError, UnsupportedFileTypeError

class InputManager:
    """
    Handles input management for the system.
    """
    # INITIALIZES INPUTMANAGER
    def __init__(self):
        """
        Initializes the InputManager.
        """
        Logger.log(f"start InputManager __init__(self)")
        self.data_interpreter = InputDataInterpreter()
        Logger.log(f"end InputManager __init__(self)")

    # GET NETWORK FROM INPUT DATA
    def get_network(self, input_data):
        """
        Gets network data from input data.

        Args:
            input_data (str): This is the file location, supplied by the user, that holds the network data.

        Raises:
            InvalidInputDataError: If the data cannot be validated by the data_processing_strategy.
        """
        Logger.log(f"start get_network __init__(self, {input_data})")

        # GET DATA PROCESSING STRATEGY FROM INPUT DATA
        try:
            if isinstance(input_data, str):
                data_processing_strategy = self.data_interpreter.get_data_processing_strategy(input_data)
                return data_processing_strategy.process(input_data)
            else: # Assume it's a file-like object (BytesIO)
                # For in-memory files, we need to determine the strategy based on content or a hint
                # For now, assuming it's always an Excel file if it's not a string path
                from .excel_data_strategy import ExcelDataStrategy
                data_processing_strategy = ExcelDataStrategy()
                return data_processing_strategy.process(input_data)
        except UnsupportedFileTypeError:
            raise
        except FileNotFoundError as ex:
            Logger.log(ex, Logger.LogPriority.ERROR)
            raise
        except Exception as ex:
            Logger.log(f"Error in get_network: {ex}", Logger.LogPriority.ERROR)
            raise
        except InvalidInputDataError:
            raise
        
