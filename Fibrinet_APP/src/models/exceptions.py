class StateTransitionError(Exception):
    """Invalid system state transition."""
    def __init__(self, message="Invalid state transition attempted."):
        super().__init__(message)
    
class InvalidInputDataError(Exception):
    """Input file content failed validation."""
    def __init__(self, message="Unable to validate input data."):
        super().__init__(message)

class UnsupportedFileTypeError(Exception):
    """Unsupported input file type."""
    def __init__(self, message="File type not supported."):
        super().__init__(message)

class InvalidNetworkError(Exception):
    """Invalid network type for the requested operation."""
    def __init__(self, message="Invalid Network Type."):
        super().__init__(message)
        
class NodeNotFoundError(Exception):
    """Node ID not found in the network."""
    def __init__(self, message="Node ID not found in network."):
        super().__init__(message)

class EdgeNotFoundError(Exception):
    """Edge ID not found in the network."""
    def __init__(self, message="Edge ID not found in network."):
        super().__init__(message)