from src.controllers.system_controller import SystemController
from utils.logger.logger import Logger


def main():
    """FibriNet application entry point."""
    Logger.initialize()
    Logger.disable_logging() 

    controller = SystemController()

    controller.initiate_view("tkinter")

if __name__ == "__main__":
    main()