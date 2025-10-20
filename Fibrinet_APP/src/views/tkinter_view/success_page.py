from utils.logger.logger import Logger 
import tkinter as tk
from .tkinter_view import TkinterView

class SuccessPage(TkinterView):
    def __init__(self, view):
        self.view = view
        self.BG_COLOR = view.BG_COLOR
        self.FG_COLOR = view.FG_COLOR
        self.button_images = view.button_images
        self.PAGE_HEADING_FONT = view.HEADING_FONT
        self.PAGE_HEADING_BG = view.HEADING_BG
        self.PAGE_SUBHEADING_FONT = view.SUBHEADING_FONT
        self.PAGE_SUBHEADING_BG = view.SUBHEADING_BG

    def show_page(self, container, from_new_network=False):
        """Displays the Success Page"""
        self.from_new_network = from_new_network
        self.container = container
        center_frame = tk.Frame(container, bg=self.BG_COLOR)
        center_frame.pack(expand=True)

        # Page Heading
        self.success_heading = tk.Label(
            center_frame,
            foreground=self.FG_COLOR,
            font=self.PAGE_HEADING_FONT,
            background=self.PAGE_HEADING_BG,
            text="Export Successful!"
        )
        self.success_heading.pack(pady=(20, 10))

        # Success Message
        self.success_message = tk.Label(
            center_frame,
            foreground=self.FG_COLOR,
            font=self.PAGE_SUBHEADING_FONT,
            background=self.PAGE_SUBHEADING_BG,
            text="Your data has been exported successfully."
        )
        self.success_message.pack(pady=(5, 20))

        # Buttons Frame
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
            command=lambda: self.view.show_page("export", from_new_network=self.from_new_network)
        )
        self.back_button.pack(side=tk.LEFT, padx=45)
