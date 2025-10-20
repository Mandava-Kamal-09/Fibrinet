from utils.logger.logger import Logger

class ViewStrategy():
    """
    Base class for strategies handling different types of views in the application.
    Subclasses should implement methods to handle view requests, updates, and events.
    """

    # VIEWSTRATEGY INITILIZATION
    def __init__(self, controller):
        """
        Initializes the ViewStrategy instance and sets up the controller and Logger.
        
        Parameters:
        controller (Controller): The controller that manages the view's actions.
        """
        

        Logger.log(f"start InputManager __init__(self, {controller})")
        
        # SETTING CONTROLLER
        self.controller = controller
        
        Logger.log(f"end InputManager __init__(self, controller)")

    # STARTS THE VIEW, INITIALIZING NECESSARY RESOURCES OR DISPLAY MECHANISMS.
    def start_view(self):
        """
        Starts the view, initializing necessary resources or display mechanisms.
        
        Raises:
        NotImplementedError: If not implemented in a subclass.
        """
        # MUST BE IMPLEMENTED IN A SUBCLASS TO START THE VIEW
        raise NotImplementedError()

    # STOPS THE VIEW, RELEASING RESOURCES OR HIDING THE DISPLAY.
    def stop_view(self):
        """
        Stops the view, releasing resources or hiding the display.
        
        Raises:
        NotImplementedError: If not implemented in a subclass.
        """
        # MUST BE IMPLEMENTED IN A SUBCLASS TO STOP THE VIEW
        raise NotImplementedError()
