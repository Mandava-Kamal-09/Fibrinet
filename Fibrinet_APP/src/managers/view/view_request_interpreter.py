from src.views.cli_view.cli_view import CommandLineView
from src.views.tkinter_view.optional_gui_loader import get_gui_view_class
TkinterView = get_gui_view_class()
from utils.logger.logger import Logger

# INTERPRETS VIEW REQUESTS AND RETURNS APPROPRIATE VIEW STRATEGY
class ViewRequestInterpreter:
    """
    Handles view requests by selecting and returning the appropriate view strategy.
    """

    def __init__(self):
        """
        Initializes the ViewRequestInterpreter.
        """
        Logger.log(f"start ViewRequestInterpreter __init__(self)")
        Logger.log("ViewRequestInterpreter initialized.")
        Logger.log(f"end ViewRequestInterpreter __init__(self)")

   # RETURNS THE APPROPRIATE VIEW STRATEGY BASED ON REQUEST
    def get_view_strategy(self, view_request, controller):
        """
        Determines and returns the appropriate view strategy based on the provided request.

        Args:
            view_request (str): The type of view requested ("CLI" or "Tkinter").

        Returns:
            CommandLineView or TkinterView: An instance of the selected view class.

        Raises:
            ValueError: If the provided view request is invalid.
        """

        Logger.log(f"start get_view_strategy(self, {view_request}, {controller})")

        if view_request.lower() == "cli":
            Logger.log("CLI view strategy selected.")
            Logger.log(f"end get_view_strategy(self, view_request, controller)")
            return CommandLineView(controller)
        
        elif view_request.lower() == "tkinter":
            Logger.log("Tkinter view strategy selected.")
            Logger.log(f"end get_view_strategy(self, view_request, controller)")
            return TkinterView(controller)
        
        else:
            Logger.log("ValueError: Invalid view request. Choose 'CLI' or 'Tkinter'.", Logger.LogPriority.ERROR)
            raise ValueError("Invalid view request. Choose 'CLI' or 'Tkinter'.")