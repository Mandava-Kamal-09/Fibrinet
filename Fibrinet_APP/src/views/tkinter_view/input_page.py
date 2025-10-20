from utils.logger.logger import Logger 
import tkinter as tk
from tkinter import filedialog
from .tkinter_view import TkinterView

class InputPage(TkinterView):
    """Input page for loading an existing network or creating a new one."""
    # INITALIZE INPUT PAGE
    def __init__(self, view):
        """Initialize styles and references from parent view."""
        Logger.log(f"start __init__(self, {view})")
        self.view = view

        # INPUT PAGE STYLES
        self.BG_COLOR = view.BG_COLOR
        self.FG_COLOR = view.FG_COLOR
        self.button_images = view.button_images
        self.PAGE_HEADING_FONT = view.HEADING_FONT
        self.PAGE_HEADING_BG = view.HEADING_BG
        self.PAGE_SUBHEADING_FONT = view.SUBHEADING_FONT
        self.PAGE_SUBHEADING_BG = view.SUBHEADING_BG
        Logger.log("end __init__(self, view)")

    # SHOW PAGE
    def show_page(self, container):
        """Render heading and upload/create controls."""
        Logger.log(f"start show_page(self, {container})")
        center_frame = tk.Frame(container, bg=self.BG_COLOR)
        center_frame.pack(expand=True)

        self.input_heading = tk.Label(
            center_frame,
            foreground=self.FG_COLOR,
            font=self.PAGE_HEADING_FONT,
            background=self.PAGE_HEADING_BG,
            text="FibriNet GUI"
        )
        self.input_heading.pack(pady=(20, 10))

        self.input_subheading = tk.Label(
            center_frame,
            foreground=self.FG_COLOR,
            font=self.PAGE_SUBHEADING_FONT,
            background=self.PAGE_SUBHEADING_BG,
            text="Start by uploading an existing network data file (.xlsx), or create a new one from scratch.",
            wraplength=450,
            justify="center"
        )
        self.input_subheading.pack(pady=(0, 20))

        button_frame = tk.Frame(center_frame, bg=self.BG_COLOR)
        button_frame.pack(pady=20)

        self.upload_file_icon_button = tk.Button(
            button_frame,
            image=self.button_images["Import"],
            bg=self.view.ICON_BUTTON_BG,
            border="0",
            cursor="hand2",
            command=self.on_upload_file_icon_button_click,
            padx=10,
            pady=10,
            activebackground=self.view.ACTIVE_BG_COLOR
        )
        self.upload_file_icon_button.grid(row=0, column=0, padx=30) 

        separator = tk.Label(
            button_frame,
            text="|",
            font=("Arial", 14),
            bg=self.BG_COLOR,
            fg=self.FG_COLOR
        )
        separator.grid(row=0, column=1, padx=10)


        self.create_new_network_button = tk.Button(
            button_frame,
            image=self.button_images["Plus"],  
            bg=self.view.ICON_BUTTON_BG,
            border="0",
            cursor="hand2",
            command=self.on_create_new_network_button_click,  
            padx=10,
            pady=10,
            activebackground=self.view.ACTIVE_BG_COLOR
        )
        self.create_new_network_button.grid(row=0, column=2, padx=30)

        Logger.log("end show_page(self, container)")

    # ON CREATE NEW NETWORK BUTTON CLICK
    def on_create_new_network_button_click(self):
        Logger.log("start on_create_new_network_button_click")
        # Placeholder logic — customize this!
        self.view.selected_file = None
        self.view.show_page("new_network")  
        Logger.log("end on_create_new_network_button_click")


    # ON UPLOAD FILE ICON BUTTON CLICK
    def on_upload_file_icon_button_click(self):
        """Open file dialog and route to confirmation page if selected."""
        Logger.log(f"start on_upload_file_icon_button_click") 
        file_path = filedialog.askopenfilename(
            title="Select a Network Data File",
            filetypes=[("Excel Files", "*.xlsx")]
        )
        if file_path: 
            Logger.log(f"File selected: {file_path}")
            self.view.selected_file = file_path
            self.view.show_page("input_confirm")
        else:
            Logger.log("No file selected.")
        Logger.log(f"end on_upload_file_icon_button_click")

