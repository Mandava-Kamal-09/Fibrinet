from utils.logger.logger import Logger
import tkinter as tk
from .tkinter_view import TkinterView
import threading

class ExportConfirmPage(TkinterView):
    """
    A class that represents the confirmation page for the export file location in the FibriNet GUI application.

    This page appears after the user selects an export file path, asking them to confirm their selection.

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
        show_page(container): Displays the confirmation page, showing the export file path and action buttons.
        on_confirm_export(): Confirms the export and triggers the export process.
    """
    
    # INITIALIZES EXPORTCONFIRMPAGE
    def __init__(self, view):
        Logger.log(f"start ExportConfirmPage__init__(self, view)")
        self.view = view

        # EXPORT CONFIRM PAGE STYLES
        self.BG_COLOR = view.BG_COLOR
        self.FG_COLOR = view.FG_COLOR
        self.button_images = view.button_images
        self.PAGE_HEADING_FONT = view.HEADING_FONT
        self.PAGE_HEADING_BG = view.HEADING_BG
        self.PAGE_SUBHEADING_FONT = view.SUBHEADING_FONT
        self.PAGE_SUBHEADING_BG = view.SUBHEADING_BG
        Logger.log(f"end ExportConfirmPage__init__(self, view)")

    # SHOW PAGE
    def show_page(self, container, from_new_network=False):
        """
        Displays the Export Confirm Page with heading, subheading, and button options.
        
        Args:
            container (tk.Frame): The container frame where the page content will be added.
        """
        Logger.log(f"start show_page(self, container)")
        self.from_new_network = from_new_network
        center_frame = tk.Frame(container, bg=self.BG_COLOR)
        center_frame.pack(expand=True)

        self.confirm_heading = tk.Label(
            center_frame,
            foreground=self.FG_COLOR,
            font=self.PAGE_HEADING_FONT,
            background=self.PAGE_HEADING_BG,
            text="Correct Folder?"
        )
        self.confirm_heading.pack(pady=(20, 10))

        self.confirm_subheading = tk.Label(
            center_frame,
            foreground=self.FG_COLOR,
            font=self.PAGE_SUBHEADING_FONT,
            background=self.PAGE_SUBHEADING_BG,
            text=self.view.export_request_details.get("folder_selected", "No folder selected."),
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
            command=lambda: self.view.show_page("export"),  # Return to the Export Page
            activebackground=self.view.ACTIVE_BG_COLOR
        )
        self.cancel_button.pack(side=tk.LEFT, padx=60)

        self.confirm_button = tk.Button(
            button_frame,
            image=self.button_images["Checkmark"],
            bg=self.BG_COLOR,
            cursor="hand2",
            border="0",
            command=self.start_export,  # Proceed with the export
            activebackground=self.view.ACTIVE_BG_COLOR
        )
        self.confirm_button.pack(side=tk.RIGHT, padx=60)
        Logger.log(f"end show_page(self, container)")

    def start_export(self):
        """Handles the start of the export, showing the loading page and running export in a separate thread."""
        # Show the loading page
        self.view.show_page("loading", from_new_network=self.from_new_network)
        
        # Run the export in a separate thread to prevent UI freezing
        export_thread = threading.Thread(target=self.on_confirm_export)
        export_thread.start()

    # ON CONFIRM EXPORT
    def on_confirm_export(self):
        """
        Handles the confirmation of the export location.
        """
        export_request = f"export_request {self.view.export_request_details.get("data_export_strategy", None)} {self.view.export_request_details.get("image_export_strategy", None)} {self.view.export_request_details.get("folder_selected", None)}"
        # Call export method (this is where the export logic goes)
        try:
            Logger.log(f"Exporting data with request: {export_request}")
            self.view.controller.export_data(export_request)
            
            # Once export is done, show the success page
            self.view.show_page("success", from_new_network=self.from_new_network)
        except Exception as e:
            Logger.log(f"Error during export: {str(e)}")
            self.view.error_message = str(e)
            self.view.show_page("error", error_message=str(e))


