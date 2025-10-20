# error_page.py
from utils.logger.logger import Logger
import tkinter as tk
from .tkinter_view import TkinterView

class ErrorPage(TkinterView):
    def __init__(self, view):
        self.view = view
        self.BG_COLOR = view.BG_COLOR
        self.FG_COLOR = view.FG_COLOR
        self.button_images = view.button_images
        self.PAGE_HEADING_FONT = view.HEADING_FONT
        self.PAGE_HEADING_BG = view.HEADING_BG
        self.PAGE_SUBHEADING_FONT = view.SUBHEADING_FONT
        self.PAGE_SUBHEADING_BG = view.SUBHEADING_BG

    def show_page(self, container, error_message="An error occurred."):
        """Displays the Error Page with a checkmark button to return to input."""
        center_frame = tk.Frame(container, bg=self.BG_COLOR)
        center_frame.pack(expand=True)

        self.error_heading = tk.Label(
            center_frame,
            foreground=self.FG_COLOR,
            font=self.PAGE_HEADING_FONT,
            background=self.PAGE_HEADING_BG,
            text="Error"
        )
        self.error_heading.pack(pady=(20, 10))

        self.error_message = tk.Label(
            center_frame,
            foreground=self.FG_COLOR,
            font=self.PAGE_SUBHEADING_FONT,
            background=self.PAGE_SUBHEADING_BG,
            text=error_message,
            wraplength=450,
            justify="center"
        )
        self.error_message.pack(pady=(0, 20))

        # Checkmark button to return to input page
        self.checkmark_button = tk.Button(
            center_frame,
            image=self.button_images["Checkmark"],
            bg=self.view.ICON_BUTTON_BG,
            border="0",
            cursor="hand2",
            command=lambda: self.view.show_page("input"),
            activebackground=self.view.ACTIVE_BG_COLOR
        )
        self.checkmark_button.pack(pady=20)