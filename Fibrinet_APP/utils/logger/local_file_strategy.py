from .log_storage_strategy import LogStorageStrategy
import os
from datetime import datetime

class LocalFileStrategy(LogStorageStrategy):
    """
    Handles log storage using a local file.
    """

    # INITIALIZE LOG STORAGE STRATEGY
    def __init__(self, file_location):
        """
        Initializes the local file strategy.

        Args:
            file_location (str): The location of the log file.
        """
        self.file_location = self.resolve_file_path(file_location)
        self.initialize_log_file()

    # RESOLVE FILE PATH TO ABSOLUTE AND CREATE DIRECTORIES IF NEEDED
    def resolve_file_path(self, file_location):
        """
        Converts the given file path to an absolute path and creates directories if needed.

        Args:
            file_location (str): The file path to resolve.

        Returns:
            str: The absolute file path.
        """
        if not os.path.isabs(file_location):
            file_location = os.path.join(os.getcwd(), file_location)
        
        dir_name = os.path.dirname(file_location)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        return file_location

    # INITIALIZE OR RESET THE LOG FILE
    def initialize_log_file(self):
        """
        Creates a new log file or flushes it if it already exists.
        """
        if os.path.exists(self.file_location):
            self.flush_logs()
        else:
            with open(self.file_location, 'w') as log_file:
                log_file.write(f"LOG INITIALIZATION: {datetime.now()}\n")
    
    # STORE A LOG ENTRY IN THE FILE
    def store_log(self, message, priority, timestamp):
        """
        Appends a log entry to the log file.

        Args:
            message (str): The log message.
            priority (str): The priority level of the log.
            timestamp (str): The timestamp of the log entry.
        """
        with open(self.file_location, 'a') as log_file:
            log_file.write(f"[{timestamp}] [{priority}] {message}\n")
    
    # CLEAR ALL LOG ENTRIES FROM THE FILE
    def flush_logs(self):
        """
        Clears all contents from the log file.
        """
        with open(self.file_location, 'w') as log_file:
            log_file.truncate(0)
            log_file.write(f"LOG FLUSHED: {datetime.now()}\n")
