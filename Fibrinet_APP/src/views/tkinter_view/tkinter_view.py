import tkinter as tk
import os
import sys
from src.managers.view.view_strategy import ViewStrategy
from utils.logger.logger import Logger


class TkinterView(ViewStrategy):
    """Create and manage the Tkinter GUI for FibriNet."""

    # NOTE: Do not instantiate the view class at import time. The controller
    # constructs and starts the view via ViewManager.

    # GENERAL STYLE
    FONT_FAMILY = "Consolas"
    HEADING_FONT_SIZE = 45
    SUBHEADING_FONT_SIZE = 15
    SUBHEADING_2_FONT_SIZE = 12
    BG_COLOR = "gray9"
    ACTIVE_BG_COLOR = "gray9"
    FG_COLOR = "white"
    BG_COLOR = BG_COLOR
    HEADING_COLOR = FG_COLOR
    HEADING_FONT = (FONT_FAMILY, HEADING_FONT_SIZE)
    HEADING_COLOR = FG_COLOR
    HEADING_BG = BG_COLOR
    SUBHEADING_2_FONT = (FONT_FAMILY, SUBHEADING_2_FONT_SIZE)
    SUBHEADING_COLOR = FG_COLOR
    SUBHEADING_FONT = (FONT_FAMILY, SUBHEADING_FONT_SIZE)
    SUBHEADING_COLOR = FG_COLOR
    SUBHEADING_BG = BG_COLOR
    ICON_BUTTON_BG = BG_COLOR
    ICON_BUTTON_BORDER = BG_COLOR
    ICON_BUTTON_HOVER_BORDER = BG_COLOR
    ICON_BUTTON_HOVER_BG = BG_COLOR

    def __init__(self, controller):
        """Initialize window, resources, pages, and layout."""
        Logger.log(f"start TkinterView __init__(self, controller)")
        super().__init__(controller)  # Call parent class __init__
        self.running = True
        self.root = tk.Tk()

        # Resource path helper
        def resource_path(relative_path):
            """Return absolute path for dev and PyInstaller bundles."""
            try:
                # PyInstaller sets this attr when running in a bundle
                base_path = sys._MEIPASS
            except AttributeError:
                # Running normally (e.g., from IDE or python script)
                base_path = os.path.dirname(os.path.abspath(__file__))
            
            return os.path.join(base_path, relative_path)

        self.root.iconbitmap(resource_path("images/FibriNet_Icon.ico"))
        self.root.title("FibriNet GUI")
        self.root.geometry("1000x800")
        self.root.attributes('-topmost', True)
        self.root.after(100, lambda: self.root.attributes('-topmost', False))
        self.root.focus_force()
        self.root.minsize(800, 650)

        

        # Use resource_path to load images
        self.image_paths = {
            "Import": resource_path("images/Small_Import.png"),
            "Small_Import": resource_path("images/XSmall_Import.png"),
            "Small_Left_Arrow": resource_path("images/XSmall_Left_Arrow.png"),
            "X": resource_path("images/Small_X.png"),
            "Small_X": resource_path("images/XSmall_X.png"),
            "Small_Right_Arrow": resource_path("images/XSmall_Right_Arrow.png"),
            "Export": resource_path("images/Small_Export.png"),
            "Small_Export": resource_path("images/XSmall_Export.png"),
            "Checkmark": resource_path("images/Small_Checkmark.png"),
            "Plus": resource_path("images/Small_Plus.png")
        }


        self.button_images = {name: tk.PhotoImage(file=path) for name, path in self.image_paths.items()}
        
        from .input_confirm_page import InputConfirmPage
        from .error_page import ErrorPage
        from .export_page import ExportPage
        from .input_page import InputPage
        from .modify_page import ModifyPage
        from .export_confirm_page import ExportConfirmPage
        from .success_page import SuccessPage
        from .loading_page import LoadingPage
        from .new_network_page import NewNetworkPage

        self.page_classes = {
            "input": InputPage(self),
            "input_confirm": InputConfirmPage(self),
            "export_confirm": ExportConfirmPage(self),
            "error": ErrorPage(self),
            "export": ExportPage(self),
            "modify": ModifyPage(self),
            "success": SuccessPage(self),
            "loading": LoadingPage(self), 
            "new_network": NewNetworkPage(self),
        }

        # Initialize with the input page
        self.state = 'input'
        self.page_content = tk.Frame(self.root, bg=self.BG_COLOR)
        self.page_content.pack(fill=tk.BOTH, expand=True)
        Logger.log(f"end TkinterView __init__(self, controller)")

    # SHOW PAGE
    def show_page(self, page_name, **kwargs):
        """
        General method to show any page. 
        
        Args: 
            page_name: this is a string of the page name .
            **kwargs: Arbitrary keyword arguments to pass to the page's show_page method
        """
        Logger.log(f"start show_page(self, {page_name})")
        page_class = self.page_classes.get(page_name)
        if page_class:
            self.clear_body()
            page_class.show_page(self.page_content, **kwargs)  # Calls the show method of the page
        Logger.log(f"end show_page(self, page_name)")

    def show_error_page(self, error_message):
        """Displays the error page with a specific error message."""
        Logger.log(f"start show_error_page(self, error_message={error_message})")
        error_page = self.page_classes.get("error")
        if error_page:
            self.clear_body()
            error_page.show_page(self.page_content, error_message)
        Logger.log(f"end show_error_page(self, error_message)")

    # CLEAR BODY
    def clear_body(self):
        """
        This removes all elements in the current page.
        """
        Logger.log(f"start clear_body(self)")
        for widget in self.page_content.winfo_children():
            try:
                widget.destroy()
            except Exception as e:
                Logger.log(f"Warning destroying widget: {e}")
        Logger.log(f"end clear_body(self)")

    # START VIEW
    def start_view(self):
        """
        Starts the Tkinter view by showing the input page. 
        """
        Logger.log("start start_view(self)")
        self.show_page("input")  # Start with the input page
        self.root.mainloop()
        Logger.log(f"end start_view(self)")

    # STOP VIEW
    def stop_view(self):
        """
        Stop the Tkinter view by showing the input page. 
        """
        Logger.log("start stop_view(self)")
        self.root.quit()
        Logger.log(f"end stop_view(self)")



