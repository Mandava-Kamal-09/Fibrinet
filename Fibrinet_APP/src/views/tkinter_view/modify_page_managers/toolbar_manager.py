import tkinter as tk
from utils.logger.logger import Logger

class ToolbarManager:
    def __init__(self, modify_page):
        """
        Initializes the ToolbarManager with action and info bars.
        """
        Logger.log(f"start ToolbarManager__init__(self, modify_page)")
        self.modify_page = modify_page
        self.toolbar_frame = None
        self.action_bar = None
        self.info_bar = None
        self.buttons = {}
        Logger.log(f"end ToolbarManager__init__(self, modify_page)")

    def setup_toolbar(self, controller, container):
        """Sets up the main toolbar frame, including the action and info bars."""
        Logger.log(f"start setup_toolbar(self, container)")
        self.toolbar_height = 100  # Set toolbar height
        self.toolbar_frame = tk.Frame(container, bg=self.modify_page.ACTION_BAR_BG_COLOR, height=self.toolbar_height)
        self.toolbar_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.toolbar_frame.pack_propagate(False)  # Prevent shrinking

        # Setup action and info bars
        self.setup_action_bar(controller)
        self.setup_info_bar()

        # Bind resizing event
        self.toolbar_frame.bind("<Configure>", self.update_toolbar_sizes)
        Logger.log(f"end setup_toolbar(self, container)")

    def update_toolbar_sizes(self, event):
        """Adjusts the sizes of action and info bars when the window is resized."""
        Logger.log(f"start update_toolbar_sizes(self, event)")
        new_width = event.width
        self.action_bar.place(x=0, y=0, relwidth=0.65, height=self.toolbar_height)  # 65% width
        self.info_bar.place(relx=0.65, y=0, relwidth=0.35, height=self.toolbar_height)  # 35% width
        Logger.log(f"end update_toolbar_sizes(self, event)")

    # ----- ACTION BAR -----
    def setup_action_bar(self, controller):
        """Creates the action bar and populates it with buttons."""
        Logger.log(f"start setup_action_bar(self)")
        self.action_bar = tk.Frame(self.toolbar_frame, bg=self.modify_page.ACTION_BAR_BG_COLOR)
        self.action_bar.place(x=0, y=0, relwidth=0.65, height=self.toolbar_height)
        self.add_action_bar_buttons(controller)
        Logger.log(f"end setup_action_bar(self)")

    def add_action_bar_buttons(self, controller):
        """Adds buttons to the action bar."""
        Logger.log(f"start add_action_bar_buttons(self)")
        # Create a frame inside action_bar to hold buttons
        self.button_frame = tk.Frame(self.action_bar, bg=self.modify_page.BG_COLOR)
        self.button_frame.place(relx=0.5, rely=0.5, anchor="center")  # Centered placement

        button_config = {
            "import": (self.modify_page.view.button_images["Small_Import"], self.on_import_click),
            "undo": (self.modify_page.view.button_images["Small_Left_Arrow"], self.on_undo_click),
            "remove": (self.modify_page.view.button_images["Small_X"], self.on_remove_click),
            "redo": (self.modify_page.view.button_images["Small_Right_Arrow"], self.on_redo_click),
            "export": (self.modify_page.view.button_images["Small_Export"], self.on_export_click),
        }
        
        for name, (icon, command) in button_config.items():
            button = tk.Button(
                self.button_frame, 
                image=icon, 
                command=command,
                bg=self.modify_page.ICON_BUTTON_BG,
                cursor="arrow",
                border=0,
                state=tk.DISABLED,
                activebackground=self.modify_page.ACTIVE_BG_COLOR
                )
            # Enable the "import" button, it's always active
            if name == "import":
                button.config(state=tk.ACTIVE, cursor="hand2")

            # Enable or disable other buttons based on the network manager's flags
            if name == "undo":
                if controller.network_manager.state_manager.undo_disabled:
                    button.config(state=tk.DISABLED)
                else:
                    button.config(state=tk.ACTIVE)
            
            if name == "redo":
                if controller.network_manager.state_manager.redo_disabled:
                    button.config(state=tk.DISABLED)
                else:
                    button.config(state=tk.ACTIVE)
            
            if name == "export":
                if controller.network_manager.state_manager.export_disabled:
                    button.config(state=tk.DISABLED)
                else:
                    button.config(state=tk.ACTIVE)
            button.pack(side=tk.LEFT, padx=15)
            self.buttons[name] = button
        Logger.log(f"end add_action_bar_buttons(self)")

    def enable_action_bar_button(self, button_name):
        """Enables a button in the action bar."""
        Logger.log(f"start enable_action_bar_button(self, button_name)")
        if button_name in self.buttons:
            self.buttons[button_name].config(state="normal")
        Logger.log(f"end enable_action_bar_button(self, button_name)")

    def disable_action_bar_button(self, button_name):
        """Disables a button in the action bar."""
        Logger.log(f"start disable_action_bar_button(self, button_name)")
        if button_name in self.buttons:
            Logger.log("button found")
            self.buttons[button_name].config(state=tk.DISABLED)
        Logger.log(f"end disable_action_bar_button(self, button_name)")

    # ----- BUTTON ACTIONS -----
    def on_import_click(self):
        Logger.log(f"start on_import_click(self)")
        self.modify_page.on_import()
        Logger.log(f"end on_import_click(self)")

    def on_undo_click(self):
        Logger.log(f"start on_undo_click(self)")
        self.modify_page.on_undo()
        self.update_info_bar("Undo performed")
        Logger.log(f"end on_undo_click(self)")

    def on_remove_click(self):
        Logger.log(f"start on_remove_click(self)")
        self.modify_page.remove_selected_element()
        self.update_info_bar("Element removed")
        Logger.log(f"end on_remove_click(self)")

    def on_redo_click(self):
        Logger.log(f"start on_redo_click(self)")
        self.modify_page.on_redo()
        self.update_info_bar("Redo performed")
        Logger.log(f"end on_redo_click(self)")

    def on_export_click(self):
        Logger.log(f"start on_export_click(self)")
        self.modify_page.on_export()
        self.update_info_bar("Network exported successfully")
        Logger.log(f"end on_export_click(self)")

    # ----- INFO BAR -----
    def setup_info_bar(self):
        """Creates the info bar with larger, centered, and wrapped text."""
        Logger.log(f"start setup_info_bar(self)")
        self.info_bar = tk.Label(
            self.toolbar_frame,
            text="Network imported successfully",
            fg="white",
            bg=self.modify_page.INFO_BAR_BG_COLOR,
            font=("Arial", 16, "bold"),  # Larger font
            anchor="center",  # Center text vertically
            justify="center",  # Center text horizontally
            wraplength=250  # Adjust wrapping width
        )
        self.info_bar.place(relx=0.65, y=0, relwidth=0.35, height=self.toolbar_height)
        Logger.log(f"end setup_info_bar(self)")

    def update_info_bar(self, info):
        """Updates the info bar with a new message."""
        Logger.log(f"start update_info_bar(self, {info})")
        if hasattr(self, "info_bar") and self.info_bar.winfo_exists():
            self.info_bar.config(text=info)
        else:
            Logger.log("Error: info_bar does not exist or has been destroyed")
        Logger.log(f"end update_info_bar(self, info)")

    def clear_info_bar(self):
        """Clears the info bar."""
        Logger.log(f"start clear_info_bar(self)")
        self.info_bar.config(text="")
        Logger.log(f"end on_clear_info_bar(self)")
