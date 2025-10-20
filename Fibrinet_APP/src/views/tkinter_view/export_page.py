# export_page.py
from utils.logger.logger import Logger
import tkinter as tk
from .tkinter_view import TkinterView
from ...managers.export.excel_export_strategy import ExcelExportStrategy
from ...managers.export.png_export_strategy import PngExportStrategy

class ExportPage(TkinterView):
    def __init__(self, view):
        self.view = view
        self.BG_COLOR = view.BG_COLOR
        self.FG_COLOR = view.FG_COLOR
        self.button_images = view.button_images
        self.PAGE_HEADING_FONT = view.HEADING_FONT
        self.PAGE_HEADING_BG = view.HEADING_BG
        self.PAGE_SUBHEADING_FONT = view.SUBHEADING_FONT
        self.PAGE_SUBHEADING_BG = view.SUBHEADING_BG
        self.from_new_network = False 

    def show_page(self, container, from_new_network=False):
        """Render export options and controls."""
        self.container = container
        center_frame = tk.Frame(container, bg=self.BG_COLOR)
        center_frame.pack(expand=True)
        self.from_new_network = from_new_network

        # Page Heading
        self.export_heading = tk.Label(
            center_frame,
            foreground=self.FG_COLOR,
            font=self.PAGE_HEADING_FONT,
            background=self.PAGE_HEADING_BG,
            text="Export Data"
        )
        self.export_heading.pack(pady=(20, 10))

        # Network Export Heading
        self.network_label = tk.Label(
            center_frame,
            foreground=self.FG_COLOR,
            font=self.PAGE_SUBHEADING_FONT,
            background=self.PAGE_SUBHEADING_BG,
            text="Select Network Export Type"
        )
        self.network_label.pack(pady=(5, 0))

        # Network Export Dropdown
        self.network_type_var = tk.StringVar(value="Do not export")
        self.network_dropdown = tk.OptionMenu(
            center_frame,
            self.network_type_var,
            "Do not export",
            "EXCEL (.xlsx)",
            command=self.validate_submit
        )
        # Set the same font as headings
        self.network_dropdown.config(bg=self.BG_COLOR, fg=self.FG_COLOR, width=25, font=self.PAGE_SUBHEADING_FONT)
        self.network_dropdown.pack(pady=5)

        # Hide photo export for brand-new networks
        if not from_new_network:
            
            # Photo Export Heading
            self.photo_label = tk.Label(
                center_frame,
                foreground=self.FG_COLOR,
                font=self.PAGE_SUBHEADING_FONT,
                background=self.PAGE_SUBHEADING_BG,
                text="Select Photo Export Format"
            )
            self.photo_label.pack(pady=(10, 0))

            # Photo Export Dropdown
            self.photo_format_var = tk.StringVar(value="Do not export")
            self.photo_dropdown = tk.OptionMenu(
                center_frame,
                self.photo_format_var,
                "Do not export",
                "PNG (.png)",
                command=self.validate_submit
            )
            # Set the same font as headings
            self.photo_dropdown.config(bg=self.BG_COLOR, fg=self.FG_COLOR, width=25, font=self.PAGE_SUBHEADING_FONT)
            self.photo_dropdown.pack(pady=5)

        # Buttons
        button_frame = tk.Frame(center_frame, bg=self.BG_COLOR)
        button_frame.pack(pady=20)

        # Back Button
        self.back_button = tk.Button(
            button_frame,
            image=self.button_images["Small_X"],
            bg=self.view.ICON_BUTTON_BG,
            border="0",
            cursor="hand2",
            activebackground=self.view.ACTIVE_BG_COLOR,
            command=lambda: self.view.show_page("new_network" if self.from_new_network else "modify")
        )
        self.back_button.pack(side=tk.LEFT, padx=45)

        # Submit (initially disabled)
        self.submit_button = tk.Button(
            button_frame,
            image=self.button_images["Small_Export"],
            bg=self.view.ICON_BUTTON_BG,
            border="0",
            cursor="hand2",
            activebackground=self.view.ACTIVE_BG_COLOR,
            command=self.show_export_confirm,
            state=tk.DISABLED  # Disabled by default
        )
        self.submit_button.pack(side=tk.RIGHT, padx=45)

    def validate_submit(self, _event=None):
        """Enable submit when any export option is chosen."""
        network_selected = self.network_type_var.get() != "Do not export"
        # If from_new_network is True, do not check photo export
        if self.from_new_network:
            photos_selected = False
        else:
            photos_selected = self.photo_format_var.get() != "Do not export"
        self.submit_button.config(state=tk.NORMAL if network_selected or photos_selected else tk.DISABLED)

    def show_export_confirm(self):
        """Collect options and navigate to confirmation page."""
        network_type = self.network_type_var.get()
        # If from_new_network is True, do not check photo export
        if self.from_new_network:
            photo_format = "Do not export"
        else:
            photo_format = self.photo_format_var.get()

        data_export_strategy = None
        if network_type == "EXCEL (.xlsx)":
            data_export_strategy = "excel_data_export_strategy"

        image_export_strategy = None
        if photo_format == "PNG (.png)":
            image_export_strategy = "png_image_export_strategy"
        
        # Select folder
        from tkinter import filedialog
        folder_selected = filedialog.askdirectory()

        if folder_selected:
            self.view.export_request_details = {
            'data_export_strategy': data_export_strategy,
            'image_export_strategy': image_export_strategy,
            'folder_selected': folder_selected 
            }
            self.view.show_page("export_confirm", from_new_network=self.from_new_network)
