import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.controllers.system_controller import SystemController
from utils.logger.logger import Logger

def main():
    """
    Entry point to the FibriNet CLI Application.
    """
    try:
        # CONFIGURES LOGGER WITH DEFAULT FILE STORAGE
        Logger.initialize()
        Logger.log("FibriNet CLI starting...")

        # INITIALIZE THE SYSTEM CONTROLLER
        controller = SystemController()
        Logger.log("System controller initialized successfully")

        # START THE CLI VIEW
        controller.initiate_view("cli")
        
    except KeyboardInterrupt:
        print("\n>>> Application interrupted by user")
        sys.exit(0)
    except Exception as ex:
        print(f">>> Fatal error: {ex}")
        Logger.log(f"Fatal error in CLI main: {ex}", Logger.LogPriority.ERROR)
        sys.exit(1)

if __name__ == "__main__":
    main()