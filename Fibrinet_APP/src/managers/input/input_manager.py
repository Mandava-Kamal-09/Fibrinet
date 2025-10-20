from utils.logger.logger import Logger
from .input_data_interpreter import InputDataInterpreter
from ...models.exceptions import InvalidInputDataError, UnsupportedFileTypeError

class InputManager:
    """Coordinate reading input data into a network."""
    
    def __init__(self):
        """Initialize interpreter."""
        Logger.log(f"start InputManager __init__(self)")
        self.data_interpreter = InputDataInterpreter()
        Logger.log(f"end InputManager __init__(self)")

    def get_network(self, input_data):
        """Return a `Network` parsed from `input_data` path."""
        Logger.log(f"start get_network __init__(self, {input_data})")
        try:
            strategy = self.data_interpreter.get_data_processing_strategy(input_data)
        except UnsupportedFileTypeError:
            raise
        except FileNotFoundError as ex:
            Logger.log(ex, Logger.LogPriority.ERROR)
            raise

        try:
            Logger.log(f"end get_network __init__(self, input_data)")
            return strategy.process(input_data)
        except InvalidInputDataError:
            raise
        
