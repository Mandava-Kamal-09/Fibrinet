from utils.logger.logger import Logger

class DataProcessingStrategy():
    """Interface for converting input files into a `Network`."""

    def process(self, input_data):
        """Return a `Network` parsed from `input_data` path."""
        raise NotImplementedError()
