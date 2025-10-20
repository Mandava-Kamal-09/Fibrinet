import random
import tkinter as tk
from .tkinter_view import TkinterView
from utils.logger.logger import Logger

class LoadingPage(TkinterView):
    def __init__(self, view):
        self.view = view
        self.BG_COLOR = view.BG_COLOR
        self.FG_COLOR = view.FG_COLOR
        self.PAGE_HEADING_FONT = view.HEADING_FONT
        self.PAGE_HEADING_BG = view.HEADING_BG
        self.PAGE_SUBHEADING_FONT = view.SUBHEADING_FONT
        self.PAGE_SUBHEADING_BG = view.SUBHEADING_BG
        self.button_images = view.button_images
        self.is_animating = False  # Control spinner loop

    def show_page(self, container, from_new_network=False):
        """Displays a loading page with a loading message and cancel button"""
        self.from_new_network = from_new_network
        self.is_animating = True

        center_frame = tk.Frame(container, bg=self.BG_COLOR)
        center_frame.pack(expand=True)

        # Heading
        self.loading_heading = tk.Label(
            center_frame,
            foreground=self.FG_COLOR,
            font=self.PAGE_HEADING_FONT,
            background=self.PAGE_HEADING_BG,
            text="Processing Data..."
        )
        self.loading_heading.pack(pady=(20, 10))

        # Subheading
        self.loading_message = tk.Label(
            center_frame,
            foreground=self.FG_COLOR,
            font=self.PAGE_SUBHEADING_FONT,
            background=self.PAGE_SUBHEADING_BG,
            text="Please wait while processing.",
            wraplength=450,
            justify="center"
        )
        self.loading_message.pack(pady=(0, 20))

        # Spinner Frame
        self.spinner_frame = tk.Frame(center_frame, bg=self.BG_COLOR)
        self.spinner_frame.pack(pady=(20, 0))

        self.spinner_index = 0
        self.spinner_frames = [".", "..", "...", "....", " "]
        self.spinner_labels = []

        for i in range(3):
            spinner_label = tk.Label(
                self.spinner_frame,
                foreground=self.FG_COLOR,
                font=("Courier", 24),
                background=self.BG_COLOR,
                text=" "
            )
            spinner_label.grid(row=0, column=i, padx=10)
            self.spinner_labels.append(spinner_label)

        # Cancel Button
        self.cancel_button = tk.Button(
            center_frame,
            image=self.button_images["X"],
            bg=self.BG_COLOR,
            border="0",
            cursor="hand2",
            command=self.cancel_loading,
            activebackground=self.view.ACTIVE_BG_COLOR
        )
        self.cancel_button.pack(pady=30)

        # Start animation
        self.animate_spinner()

    def animate_spinner(self):
        """Randomly change spinner frames"""
        if not self.is_animating:
            return
        for label in self.spinner_labels:
            label.config(text=random.choice(self.spinner_frames))
        self.spinner_frame.after(300, self.animate_spinner)

    def cancel_loading(self):
        """Handle cancel action"""
        Logger.log("Loading cancelled by user.")
        self.is_animating = False  # Stop the spinner animation
        self.view.show_page("export")
