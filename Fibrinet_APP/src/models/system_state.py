class SystemState:
    """Lightweight container for global UI/runtime flags."""

    def __init__(self):
        """Initialize defaults."""
        self.network_loaded = False
        self.spring_stiffness_constant = None  # Current spring constant value
        self.original_spring_constant = None   # Original spring constant from input file
