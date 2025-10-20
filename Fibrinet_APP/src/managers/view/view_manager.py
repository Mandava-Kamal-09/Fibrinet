from src.managers.view.view_request_interpreter import ViewRequestInterpreter
from utils.logger.logger import Logger

# MANAGES VIEW REQUESTS AND CONTROLS VIEW STRATEGY
class ViewManager:
    """Handles view requests by selecting and managing the appropriate view strategy."""

    def __init__(self, controller):
        """
        Initializes the ViewManager with a ViewRequestInterpreter.

        Args:
            controller: An instance of the SystemControllerInterface.
        """
        Logger.log(f"start ViewManager __init__(self, {controller})")
        
        self.view_request_interpreter = ViewRequestInterpreter()
        self.view_strategy = None

        Logger.log(f"end ViewManager __init__(self, controller)")

    # HANDLES INCOMING VIEW REQUESTS
    def initiate_view_strategy(self, view, controller):
        """
        Processes a view request by selecting and managing the appropriate view strategy.

        Args:
            view (str): The type of view requested (e.g., "CLI" or "Tkinter").
            controller: The controller instance that interacts with the selected view.

        Raises:
            ValueError: If the view request is invalid.
        """
        Logger.log(f"start initiate_view_strategy(self, {view}, controller)")
        
        try: 
            # SELECTS A NEW VIEW STRATEGY
            new_view_strategy = self.view_request_interpreter.get_view_strategy(view, controller)
            Logger.log(f"Selected view strategy: {new_view_strategy}")
        except ValueError:
            raise

        # STOPS CURRENT VIEW STRATEGY IF ACTIVE
        if self.view_strategy:
            Logger.log("Stopping current view strategy before switching.")
            self.view_strategy.stop_view()  

        # SETS NEW VIEW STRATEGY
        self.view_strategy = new_view_strategy

        # STARTS NEW VIEW STRATEGY
        Logger.log("Starting new view strategy.")
        self.view_strategy.start_view()
        
        Logger.log(f"end initiate_view_strategy(self, view, controller)")
