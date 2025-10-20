import threading
from utils.logger.logger import Logger
import tkinter as tk
from .tkinter_view import TkinterView

class InputConfirmPage(TkinterView):
    """
    A class that represents the confirmation page in the FibriNet GUI application.

    This page appears after the user selects a network data file, displaying the file path
    and asking the user to confirm their selection. The user can either proceed with the file
    or go back to the input page to select a different file.

    Attributes:
        BG_COLOR (str): The background color of the page, inherited from the parent view.
        FG_COLOR (str): The foreground (text) color of the page, inherited from the parent view.
        button_images (dict): A dictionary of button images used on the page, inherited from the parent view.
        PAGE_HEADING_FONT (tuple): The font settings for the page heading.
        PAGE_HEADING_BG (str): The background color for the page heading.
        PAGE_SUBHEADING_FONT (tuple): The font settings for the page subheading.
        PAGE_SUBHEADING_BG (str): The background color for the page subheading.
        view (TkinterView): The parent view that manages the overall GUI and state.

    Methods:
        __init__(view): Initializes the confirmation page with the provided view, setting up page styles and attributes.
        show_page(container): Displays the confirmation page, showing the selected file path and action buttons.
        on_confirm_file(): Logs the confirmed file and transitions to the modify page.
    """

    def __init__(self, view):
        Logger.log(f"start ConfirmPage__init__(self,view)")
        self.view = view

        # CONFIRM PAGE STYLES
        self.BG_COLOR = view.BG_COLOR
        self.FG_COLOR = view.FG_COLOR
        self.button_images = view.button_images
        self.PAGE_HEADING_FONT = view.HEADING_FONT
        self.PAGE_HEADING_BG = view.HEADING_BG
        self.PAGE_SUBHEADING_FONT = view.SUBHEADING_FONT
        self.PAGE_SUBHEADING_BG = view.SUBHEADING_BG
        Logger.log("end ConfirmPage__init__(self,view)")

    def show_page(self, container):
        """
        Displays the Confirm Page with heading, subheading, and button options.

        Args:
            container (tk.Frame): The container frame where the page content will be added.
        """
        Logger.log(f"start show_page(self, container)")
        center_frame = tk.Frame(container, bg=self.BG_COLOR)
        center_frame.pack(expand=True)

        self.confirm_heading = tk.Label(
            center_frame,
            foreground=self.FG_COLOR,
            font=self.PAGE_HEADING_FONT,
            background=self.PAGE_HEADING_BG,
            text="Correct File?"
        )
        self.confirm_heading.pack(pady=(20, 10))

        self.confirm_subheading = tk.Label(
            center_frame,
            foreground=self.FG_COLOR,
            font=self.PAGE_SUBHEADING_FONT,
            background=self.PAGE_SUBHEADING_BG,
            text=self.view.selected_file if self.view.selected_file else "No file selected.",
            wraplength=450,
            justify="center"
        )
        self.confirm_subheading.pack(pady=(0, 20))

        button_frame = tk.Frame(center_frame, bg=self.BG_COLOR)
        button_frame.pack(pady=30)

        self.cancel_button = tk.Button(
            button_frame,
            image=self.button_images["X"],
            bg=self.BG_COLOR,
            cursor="hand2",
            border="0",
            command=lambda: self.view.show_page("input"),
            activebackground=self.view.ACTIVE_BG_COLOR
        )
        self.cancel_button.pack(side=tk.LEFT, padx=60)

        self.confirm_button = tk.Button(
            button_frame,
            image=self.button_images["Checkmark"],
            bg=self.BG_COLOR,
            cursor="hand2",
            border="0",
            command=self.start_file_processing,  # Start file processing with loading page
            activebackground=self.view.ACTIVE_BG_COLOR
        )
        self.confirm_button.pack(side=tk.RIGHT, padx=60)
        Logger.log("end show_page(self, container)")

    def start_file_processing(self):
        """Starts file processing with a loading screen."""
        # Show the loading page
        self.view.show_page("loading")
        
        # Run the file processing in a separate thread to prevent UI freezing
        processing_thread = threading.Thread(target=self.on_confirm_file)
        processing_thread.start()

    def on_confirm_file(self):
        """
        Handles the confirmation of the selected file.
        This method runs in a separate thread to prevent UI freezing during file processing.
        """
        Logger.log(f"start on_confirm_file(self)")
        from ...managers.network.networks.network_2d import Network2D
        from ...models.exceptions import InvalidNetworkError
        from ...models.exceptions import InvalidInputDataError
        
        # LOAD NETWORK DATA
        try:
            self.view.controller.input_network(self.view.selected_file)
            if not isinstance(self.view.controller.network_manager.network, Network2D):
                raise InvalidNetworkError
            Logger.log(f"File confirmed: {self.view.selected_file}")
            self.view.show_page("modify")  # Move this line inside the try block
        except InvalidInputDataError as ex:
            self.view.show_error_page(f"Error in input data. Please check formating and try again.\nError: {ex}")
        except InvalidNetworkError as ex:
            self.view.show_error_page(f"Network not supported (Must be 2D network). Please change network and try again.\nError: {ex}")
        except Exception as ex:
            self.view.show_error_page(f"Something happened while processing the input. Please change data and try again.\nError: {ex}")
        finally:
            Logger.log("end on_confirm_file(self)")
