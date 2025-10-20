class LogStorageStrategy:
    """
    Abstract base class for log storage strategies.
    This class defines the interface for storing and flushing logs.
    """
    # STORE LOG WITH MESSAGE PRIORITY AND TIMESTAMP
    def store_log(self, message, priority, timestamp):
        """
        Stores a log message with the given priority and timestamp.
        
        Parameters:
        message (str): The log message to be stored.
        priority (int): The priority level of the log.
        timestamp (str): The timestamp of the log message.
        
        Raises:
        NotImplementedError: If this method is not overridden in a subclass.
        """
        # MUST BE IMPLEMENTED IN A SUBCLASS TO DEFINE STORAGE MECHANISM
        raise NotImplementedError()
    
    # FLUSHES ALL STORED LOGS
    def flush_logs(self):
        """
        Flushes all stored logs.
        
        Raises:
        NotImplementedError: If this method is not overridden in a subclass.
        """
        # MUST BE IMPLEMENTED IN A SUBCLASS TO DEFINE FLUSH BEHAVIOR
        raise NotImplementedError()
