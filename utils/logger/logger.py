import threading
from datetime import datetime
from enum import Enum
import os
from .local_file_strategy import LocalFileStrategy 

class Logger:
    """
    Logger class to handle logging messages with various priorities.
    Implements a static class pattern for global logging.
    """

    class LogPriority(Enum):
        DEBUG = 1
        INFO = 2
        WARNING = 3
        ERROR = 4
        CRITICAL = 5
        DEFAULT = 6


    is_logging_enabled = True
    log_storage_strategy = None
    _log_lock = threading.Lock()
    _strategy_lock = threading.Lock()
    _initialize_lock = threading.Lock()
    _disable_lock = threading.Lock()
    _enable_lock = threading.Lock()
    _flush_lock = threading.Lock()

    # INITIALIZE LOGGER
    @classmethod
    def initialize(cls, file_location=None):
        """
        Initializes the logger. If a file_location is provided, it sets up a file-based logging strategy.
        Otherwise, logging to a file will be disabled.
        """
        cls.logs = []
        cls.log_priority = cls.LogPriority.INFO
        cls.is_logging_enabled = True
        if file_location:
            # Ensure the directory for the log file exists.
            log_dir = os.path.dirname(file_location)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            cls.set_log_storage_strategy(LocalFileStrategy(file_location))
        else:
            cls.log_storage_strategy = None

    # LOG WITH MESSAGE AND PRIORITY
    @classmethod
    def log(cls, message, priority=LogPriority.DEBUG):
        """
        Logs a message with a given priority and stores it using the defined storage strategy.
        
        Parameters:
        message (str): The log message to be stored.
        priority (LogPriority): The priority level of the log (default is DEBUG).
        """
        with cls._log_lock:
            if cls.is_logging_enabled and cls.log_storage_strategy:
                cls.log_storage_strategy.store_log(
                    message, priority.name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )

    # SET LOG STORAGE STRATEGY
    @classmethod
    def set_log_storage_strategy(cls, log_storage_strategy):
        """
        Sets the log storage strategy for the logger.
        
        Parameters:
        log_storage_strategy: The storage strategy to be used for storing logs.
        """
        with cls._strategy_lock:
            cls.log_storage_strategy = log_storage_strategy

    # FLUSH LOGS
    @classmethod
    def flush_logs(cls):
        """
        Flushes all stored logs by calling the flush method on the log storage strategy.
        """
        with cls._flush_lock:
            cls.log("start flush_logs()")
            if cls.is_logging_enabled and cls.log_storage_strategy:
                cls.log_storage_strategy.flush_logs()
            cls.log("end flush_logs()")

    # DISABLE LOGGING
    @classmethod
    def disable_logging(cls):
        """
        Disables logging by setting is_logging_enabled to False.
        """
        with cls._disable_lock:
            cls.log("Logging disabled")
            cls.is_logging_enabled = False

    # ENABLE LOGGING
    @classmethod
    def enable_logging(cls):
        """
        Enables logging by setting is_logging_enabled to True.
        """
        with cls._enable_lock:
            cls.is_logging_enabled = True
            cls.log("Logging enabled")
