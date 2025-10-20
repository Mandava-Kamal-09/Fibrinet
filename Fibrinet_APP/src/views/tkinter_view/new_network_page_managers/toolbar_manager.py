import tkinter as tk
from utils.logger.logger import Logger

class ToolbarManager:
    # ----- TOOLBAR MANAGEMENT -----
    def __init__(self, new_network_page):
        """
        Initializes the ToolbarManager with action and info bars.
        """
        Logger.log(f"start ToolbarManager__init__(self, new_network_page)")
        self.new_network_page = new_network_page
        self.BG_COLOR = new_network_page.BG_COLOR
        self.FG_COLOR = new_network_page.FG_COLOR
        self.button_images = new_network_page.button_images
        self.PAGE_HEADING_FONT = new_network_page.HEADING_FONT
        self.PAGE_HEADING_BG = new_network_page.HEADING_BG
        self.PAGE_SUBHEADING_FONT = new_network_page.SUBHEADING_FONT
        self.PAGE_SUBHEADING_2_FONT = new_network_page.SUBHEADING_2_FONT
        self.PAGE_SUBHEADING_BG = new_network_page.SUBHEADING_BG
        self.FONT_FAMILY = new_network_page.FONT_FAMILY 
        self.ACTIVE_BG_COLOR = new_network_page.ACTIVE_BG_COLOR
        self.toolbar_frame = None
        self.action_bar = None
        self.info_bar = None
        self.side_bar = None
        self.MAX_SIDE_WIDTH = 400
        self.network_meta_data = None
        self.node_schema = None
        self.edge_schema = None
        self.selected_network_name = None
        self.selected_node_name = None
        self.selected_edge_name = None
        self.current_network_instance = None
        self.buttons = {}
        Logger.log(f"end ToolbarManager__init__(self, new_network_page)")

    def setup_toolbar(self, controller, container):
        """Sets up the main toolbar frame, including the action and info bars."""
        Logger.log(f"start setup_toolbar(self, container)")
        self.toolbar_height = 100  # Set toolbar height
        self.toolbar_frame = tk.Frame(container, bg=self.new_network_page.ACTION_BAR_BG_COLOR, height=self.toolbar_height)
        self.toolbar_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.toolbar_frame.pack_propagate(False)  # Prevent shrinking

        # Setup action and info bars
        self.setup_action_bar(controller)
        self.setup_info_bar()
        self.setup_side_bar(container)

        # Bind resizing event
        self.toolbar_frame.bind("<Configure>", self.update_toolbar_sizes)
        Logger.log(f"end setup_toolbar(self, container)")

    def update_toolbar_sizes(self, event):
        Logger.log("start update_toolbar_sizes(self, event)")
        container_width = event.width
        
        # Calculate info_bar width capped by max width
        info_width_px = int(container_width * 0.35)
        
        # Convert to relative width
        info_relwidth = info_width_px / container_width
        
        # Action bar fills remaining space
        action_relwidth = 1 - info_relwidth
        
        # Place bars using relwidth and relx
        self.action_bar.place(x=0, y=0, relwidth=action_relwidth, height=self.toolbar_height)
        self.info_bar.place(relx=action_relwidth, y=0, relwidth=info_relwidth, height=self.toolbar_height)
        
        container = self.toolbar_frame.master
        self.place_side_bar(container)
        Logger.log("end update_toolbar_sizes(self, event)")

    # ----- SIDE BAR -----
    # ----- LAYOUT & SETUP -----
    def place_side_bar(self, container):
        Logger.log("start place_side_bar(self, container)")
        Logger.log("PLACE SIDE BAR")
        toolbar_height = self.toolbar_frame.winfo_height()
        container_width = container.winfo_width()
        container_height = container.winfo_height()
        
        # Use same max width logic for sidebar
        side_width_px = min(int(container_width * 0.35), self.MAX_SIDE_WIDTH)
        side_relx = 1 - (side_width_px / container_width)
        sidebar_height = container_height - toolbar_height
        
        self.side_bar.place(
            relx=side_relx,
            rely=0,
            relwidth=side_width_px / container_width,
            height=sidebar_height
        )
        Logger.log("end place_side_bar(self, container)")

    def setup_side_bar(self, container):
        Logger.log("start setup_side_bar(self)")
        
        self.side_bar = tk.Frame(container, bg=self.new_network_page.SIDE_BAR_BG_COLOR)
        
        # Bind place_side_bar to container resize event to keep it updated
        container.bind("<Configure>", lambda e: self.place_side_bar(container))
        
        # Call once after a short delay to initialize
        container.after(150, lambda: self.place_side_bar(container))

        self.display_network_selector(container)
        
        Logger.log("end setup_side_bar(self)")

    # ----- NETWORK SELECTION -----
    def display_network_selector(self, container):
            Logger.log("start add_side_bar_buttons")

            button_frame = tk.Frame(self.side_bar, bg=self.new_network_page.SIDE_BAR_BG_COLOR)
            button_frame.pack(fill="both", expand=True, pady=10)

            self.dropdown_container = tk.Frame(button_frame, bg=self.new_network_page.SIDE_BAR_BG_COLOR)
            self.dropdown_container.pack(fill="both", expand=True)

            self.side_bar_buttons = {}

            from ....managers.network.network_factory import NetworkFactory
            self.network_classes, self.node_classes, self.edge_classes = NetworkFactory.get_all_registered_components()
            self.network_class_lookup = {cls.__name__: cls for cls in self.network_classes}
            self.node_class_lookup = {cls.__name__: cls for cls in self.node_classes}
            self.edge_class_lookup = {cls.__name__: cls for cls in self.edge_classes}

            # NETWORK DROPDOWN
            tk.Label(self.dropdown_container, 
                    text="Select Network:", 
                    bg=self.new_network_page.SIDE_BAR_BG_COLOR, 
                    fg=self.FG_COLOR, 
                    font=self.PAGE_SUBHEADING_FONT
            ).pack(anchor="w", padx=10)

            self.network_var = tk.StringVar(value="-- Select Network --")
            self.network_dropdown = tk.OptionMenu(
                self.dropdown_container, self.network_var, "-- Select Network --", *[cls.__name__ for cls in self.network_classes],
                command=self.on_network_selected
            )
            self.network_dropdown.config(
                bg=self.new_network_page.SIDE_BAR_BG_COLOR,
                fg=self.FG_COLOR,
                highlightthickness=0,
                font=self.PAGE_SUBHEADING_2_FONT,
                activebackground=self.ACTIVE_BG_COLOR,
                activeforeground=self.FG_COLOR,
                borderwidth=0,
                relief="flat",
                width=20,
                anchor="w",
                padx=5
            )
            self.network_dropdown["menu"].config(
                bg=self.new_network_page.SIDE_BAR_BG_COLOR,
                fg=self.FG_COLOR,
                font=self.PAGE_SUBHEADING_FONT
            )
            self.network_dropdown.pack(fill="x", padx=10, pady=5)

            # NODE DROPDOWN (start hidden)
            self.node_label = tk.Label(self.dropdown_container, 
                                    text="Select Node:", 
                                    bg=self.new_network_page.SIDE_BAR_BG_COLOR, 
                                    fg=self.FG_COLOR, 
                                    font=self.PAGE_SUBHEADING_FONT)
            self.node_var = tk.StringVar(value="-- Select Node --")
            self.node_dropdown = tk.OptionMenu(self.dropdown_container, self.node_var, "")
            self.node_label.pack_forget()
            self.node_dropdown.pack_forget()

            self.node_dropdown.config(
                bg=self.new_network_page.SIDE_BAR_BG_COLOR,
                fg=self.FG_COLOR,
                highlightthickness=0,
                font=self.PAGE_SUBHEADING_2_FONT,
                activebackground=self.ACTIVE_BG_COLOR,
                activeforeground=self.FG_COLOR,
                borderwidth=0,
                relief="flat",
                width=20,
                anchor="w",
                padx=5
            )
            self.node_dropdown["menu"].config(
                bg=self.new_network_page.SIDE_BAR_BG_COLOR,
                fg=self.FG_COLOR,
                font=self.PAGE_SUBHEADING_FONT
            )

            # EDGE DROPDOWN (start hidden)
            self.edge_label = tk.Label(self.dropdown_container, 
                                    text="Select Edge:", 
                                    bg=self.new_network_page.SIDE_BAR_BG_COLOR, 
                                    fg=self.FG_COLOR, 
                                    font=self.PAGE_SUBHEADING_FONT)
            self.edge_var = tk.StringVar(value="-- Select Edge --")
            self.edge_dropdown = tk.OptionMenu(self.dropdown_container, self.edge_var, "")
            self.edge_label.pack_forget()
            self.edge_dropdown.pack_forget()

            self.edge_dropdown.config(
                bg=self.new_network_page.SIDE_BAR_BG_COLOR,
                fg=self.FG_COLOR,
                highlightthickness=0,
                font=self.PAGE_SUBHEADING_2_FONT,
                activebackground=self.ACTIVE_BG_COLOR,
                activeforeground=self.FG_COLOR,
                borderwidth=0,
                relief="flat",
                width=20,
                anchor="w",
                padx=5
            )
            self.edge_dropdown["menu"].config(
                bg=self.new_network_page.SIDE_BAR_BG_COLOR,
                fg=self.FG_COLOR,
                font=self.PAGE_SUBHEADING_FONT
            )

            # Confirm button
            self.confirm_button = tk.Button(
                button_frame,
                text="Confirm Selection",
                state=tk.DISABLED,
                bg=self.new_network_page.ICON_BUTTON_BG,
                fg=self.FG_COLOR,
                font=self.PAGE_SUBHEADING_2_FONT,
                activebackground=self.ACTIVE_BG_COLOR,
                command=self.on_confirm_selection
            )
            self.confirm_button.pack(fill="x", padx=10, pady=(0, 8))

            Logger.log("end add_side_bar_buttons")

    def on_network_selected(self, selected_name):
        Logger.log(f"Network selected: {selected_name}")

        if selected_name == "-- Select Network --":
            self.node_label.pack_forget()
            self.node_dropdown.pack_forget()
            self.edge_label.pack_forget()
            self.edge_dropdown.pack_forget()

            self.node_var.set("-- Select Node --")
            self.edge_var.set("-- Select Edge --")
            self.check_confirm_ready()
            return

        # Get the selected network class
        selected_network_class = self.network_class_lookup[selected_name]

        # Create a new instance with default empty data to start fresh
        self.current_network_instance = selected_network_class(data={})

        allowed_node_type = getattr(selected_network_class, 'allowed_node_type', object)
        allowed_edge_type = getattr(selected_network_class, 'allowed_edge_type', object)

        # Filter node classes
        filtered_node_classes = [cls for cls in self.node_classes if issubclass(cls, allowed_node_type)]

        # Filter edge classes
        filtered_edge_classes = [cls for cls in self.edge_classes if issubclass(cls, allowed_edge_type)]

        # Show and populate node dropdown
        self.node_label.pack(anchor="w", padx=10, pady=(10, 0))
        self.node_dropdown.pack(fill="x", padx=10, pady=5)
        self.node_dropdown['menu'].delete(0, 'end')
        self.node_dropdown['menu'].add_command(
            label="-- Select Node --",
            command=lambda: self.node_var.set("-- Select Node --")
        )
        for cls in filtered_node_classes:
            self.node_dropdown['menu'].add_command(
                label=cls.__name__,
                command=lambda val=cls.__name__: self.node_var.set(val)
            )
        self.node_var.set("-- Select Node --")

        # Show and populate edge dropdown
        self.edge_label.pack(anchor="w", padx=10, pady=(10, 0))
        self.edge_dropdown.pack(fill="x", padx=10, pady=5)
        self.edge_dropdown['menu'].delete(0, 'end')
        self.edge_dropdown['menu'].add_command(
            label="-- Select Edge --",
            command=lambda: self.edge_var.set("-- Select Edge --")
        )
        for cls in filtered_edge_classes:
            self.edge_dropdown['menu'].add_command(
                label=cls.__name__,
                command=lambda val=cls.__name__: self.edge_var.set(val)
            )
        self.edge_var.set("-- Select Edge --")

        # Triggers
        self.network_var.trace_add("write", lambda *args: self.check_confirm_ready())
        self.node_var.trace_add("write", lambda *args: self.check_confirm_ready())
        self.edge_var.trace_add("write", lambda *args: self.check_confirm_ready())

        self.check_confirm_ready()

    def check_confirm_ready(self):
        Logger.log("start check_confirm_ready()")
        Logger.log(self.network_var.get() != "-- Select Network --")
        Logger.log(self.node_var.get() != "-- Select Node --")
        Logger.log(self.edge_var.get() != "-- Select Edge --")
        if (self.network_var.get() != "-- Select Network --" and
            self.node_var.get() != "-- Select Node --" and
            self.edge_var.get() != "-- Select Edge --"):
            self.confirm_button.config(state=tk.NORMAL, cursor="hand2")
        else:
            self.confirm_button.config(state=tk.DISABLED, cursor="arrow")

    def on_confirm_selection(self):
        Logger.log(f"Confirmed: Network={self.network_var.get()}, Node={self.node_var.get()}, Edge={self.edge_var.get()}")

        # Get selected class names
        self.selected_network_name = self.network_var.get()
        self.selected_node_name = self.node_var.get()
        self.selected_edge_name = self.edge_var.get()

        # Resolve classes
        network_cls = next((cls for cls in self.network_classes if cls.__name__ == self.selected_network_name), None)
        node_cls = next((cls for cls in self.node_classes if cls.__name__ == self.selected_node_name), None)
        edge_cls = next((cls for cls in self.edge_classes if cls.__name__ == self.selected_edge_name), None)

        # Access schemas (if available)
        self.network_meta_data = self.to_list(getattr(network_cls, "schema", {}).get("meta_data", []))
        self.node_schema = getattr(node_cls, "schema", None)
        self.edge_schema = getattr(edge_cls, "schema", None)

        # SET NETWORK IN NETWORK MANAGER
        self.new_network_page.view.controller.input_network(self.current_network_instance)

        # Log schemas
        Logger.log(f"Network Meta Data: {self.network_meta_data}")
        Logger.log(f"Node schema: {self.node_schema}")
        Logger.log(f"Edge schema: {self.edge_schema}")

        self.display_network_builder_menu()

    # ----- NETWORK BUILDER MENU -----
    def display_network_builder_menu(self):
        Logger.log("start display_network_builder()")
        # Clear the sidebar
        for widget in self.side_bar.winfo_children():
            widget.destroy()

        button_frame = tk.Frame(self.side_bar, bg=self.new_network_page.SIDE_BAR_BG_COLOR)
        button_frame.pack(fill="both", expand=True, pady=10)

        # Heading Label
        tk.Label(
            button_frame,
            text=self.selected_network_name,
            bg=self.new_network_page.SIDE_BAR_BG_COLOR,
            fg=self.FG_COLOR,
            font=self.PAGE_SUBHEADING_FONT
        ).pack(anchor="w", fill="x", padx=10, pady=(0, 10))

        # Button style config
        button_kwargs = {
            "bg": self.new_network_page.ICON_BUTTON_BG,
            "fg": self.FG_COLOR,
            "activebackground": self.ACTIVE_BG_COLOR,
            "activeforeground": self.FG_COLOR,
            "font": self.PAGE_SUBHEADING_2_FONT,
        }

        # Edit Meta Data button
        tk.Button(button_frame, text="Edit Meta Data", command=self.display_edit_meta_data, **button_kwargs).pack(fill="x", padx=10, pady=5)

        # Add Node button
        tk.Button(button_frame, text="Add Node", command=self.display_add_node, **button_kwargs).pack(fill="x", padx=10, pady=5)

        # Add Edge button
        tk.Button(button_frame, text="Add Edge", command=self.display_add_edge, **button_kwargs).pack(fill="x", padx=10, pady=5)

        # Relax Network button
        tk.Button(
            button_frame,
            text="Relax Network",
            state="disabled" if self.new_network_page.view.controller.network_manager.state_manager.export_disabled else "normal",
            command=self.relax_network,
            **button_kwargs
        ).pack(fill="x", padx=10, pady=5)


        # Print Network History button
        tk.Button(
            button_frame,
            text="Print Network History",
            command=self.new_network_page.view.controller.network_manager.state_manager.log_network_history,
            **button_kwargs
        ).pack(fill="x", padx=10, pady=5)


        Logger.log("end display_network_builder()")

    # ----- META DATA MANAGEMENT -----
    def display_edit_meta_data(self):
        Logger.log("start display_edit_meta_data()")
        for widget in self.side_bar.winfo_children():
            widget.destroy()

        self.meta_entries = {}

        form_frame = tk.Frame(self.side_bar, bg=self.new_network_page.SIDE_BAR_BG_COLOR)
        form_frame.pack(fill="both", expand=True, pady=10)

        button_frame = tk.Frame(self.side_bar, bg=self.new_network_page.SIDE_BAR_BG_COLOR)
        button_frame.pack(fill="x", pady=(0, 10))

        tk.Label(
            form_frame,
            text="Edit Meta Data",
            bg=self.new_network_page.SIDE_BAR_BG_COLOR,
            fg=self.FG_COLOR,
            font=self.PAGE_SUBHEADING_FONT
        ).pack(anchor="w", padx=10, pady=(0, 10))

        # Get keys and meta_values

        # USE CONTROLLER'S CURRENT NETWORK INSTANCE IF AVAILABLE
        if self.new_network_page.view.controller.network_manager.network:
            keys = self.new_network_page.view.controller.network_manager.network.get_meta_data_keys()
            meta_values = self.new_network_page.view.controller.network_manager.network.get_meta_data()
        else:
            keys = self.network_meta_data or []
            meta_values = {}

        def validate_entries(*args):
            all_filled = True
            all_valid = True

            for key, (entry, error_label) in self.meta_entries.items():
                value = entry.get().strip()

                if not value:
                    all_filled = False
                    # entry.config(bg="#ffe6e6")  # light red bg for empty
                    error_label.config(text="Required field")
                    all_valid = False
                    continue

                expected_type = str
                if self.new_network_page.view.controller.network_manager.network:
                    expected_type = self.new_network_page.view.controller.network_manager.network.schema.get("meta_data_types", {}).get(key, str)

                try:
                    self.new_network_page.view.controller.network_manager.network.safe_cast(value, expected_type)
                    entry.config(bg=self.BG_COLOR)  # reset bg color
                    error_label.config(text="")
                except ValueError:
                    # entry.config(bg="#ffe6e6")  # light red bg for invalid
                    error_label.config(text=f"Must be {expected_type.__name__}")
                    all_valid = False

            save_btn.config(state="normal" if all_filled and all_valid else "disabled")

        for key in self.network_meta_data:
            tk.Label(
                form_frame, text=key, bg=self.new_network_page.SIDE_BAR_BG_COLOR,
                fg=self.FG_COLOR, font=self.PAGE_SUBHEADING_2_FONT
            ).pack(anchor="w", padx=10)

            entry = tk.Entry(
                form_frame, font=(self.FONT_FAMILY, 12), bg=self.BG_COLOR,
                fg=self.FG_COLOR, relief="groove", insertbackground=self.FG_COLOR
            )
            entry.pack(fill="x", padx=10, pady=(0, 0))

            error_label = tk.Label(
                form_frame, text="", fg="red",
                bg=self.new_network_page.SIDE_BAR_BG_COLOR,
                font=(self.FONT_FAMILY, 9)
            )
            error_label.pack(anchor="w", padx=10, pady=(0, 8))

            entry.insert(0, str(meta_values.get(key, "")))
            entry.bind("<KeyRelease>", validate_entries)

            self.meta_entries[key] = (entry, error_label)

        save_btn = tk.Button(
            button_frame, text="Save Meta Data", command=self.save_meta_data,
            bg=self.new_network_page.ICON_BUTTON_BG,
            fg=self.FG_COLOR,
            activebackground=self.ACTIVE_BG_COLOR,
            activeforeground=self.FG_COLOR,
            font=self.PAGE_SUBHEADING_2_FONT,
            state="disabled"
        )
        save_btn.pack(fill="x", padx=10, pady=(0, 8))

        cancel_btn = tk.Button(
            button_frame, text="Cancel", command=self.display_network_builder_menu,
            bg=self.new_network_page.ICON_BUTTON_BG,
            fg=self.FG_COLOR,
            activebackground=self.ACTIVE_BG_COLOR,
            activeforeground=self.FG_COLOR,
            font=self.PAGE_SUBHEADING_2_FONT
        )
        cancel_btn.pack(fill="x", padx=10)

        Logger.log("end display_edit_meta_data()")

    def save_meta_data(self):
        # Final validation before saving
        all_valid = True
        for key, (entry, error_label) in self.meta_entries.items():
            value = entry.get().strip()
            expected_type = str
            if self.new_network_page.view.controller.network_manager.network:
                expected_type = self.new_network_page.view.controller.network_manager.network.schema.get("meta_data_types", {}).get(key, str)

            try:
                casted_value = self.new_network_page.view.controller.network_manager.network.safe_cast(value, expected_type)
                entry.config(bg=self.BG_COLOR)
                error_label.config(text="")
            except ValueError:
                entry.config(bg="#ffe6e6")
                error_label.config(text=f"Must be {expected_type.__name__}")
                all_valid = False

        if not all_valid:
            return  # Don't save if invalid

        # Save values to network meta_data
        for key, (entry, _) in self.meta_entries.items():
            value = entry.get().strip()
            expected_type = self.new_network_page.view.controller.network_manager.network.schema.get("meta_data_types", {}).get(key, str)
            casted_value = self.new_network_page.view.controller.network_manager.network.safe_cast(value, expected_type)
            self.new_network_page.view.controller.network_manager.network.update_meta_data(key, casted_value)
            
        # ADD NEW NETWORK STATE    
        self.new_network_page.view.controller.network_manager.state_manager.add_new_network_state(self.new_network_page.view.controller.network_manager.network)    

        # UPDATE PAGE
        self.new_network_page.update_page("Meta data saved successfully!")

        # Go back to menu
        self.display_network_builder_menu()

    # ----- NODE MANAGEMENT -----
    def display_add_node(self):
        Logger.log("start display_add_node()")


        for widget in self.side_bar.winfo_children():
            widget.destroy()

        self.node_entries = {}

        form_frame = tk.Frame(self.side_bar, bg=self.new_network_page.SIDE_BAR_BG_COLOR)
        form_frame.pack(fill="both", expand=True, pady=10)

        button_frame = tk.Frame(self.side_bar, bg=self.new_network_page.SIDE_BAR_BG_COLOR)
        button_frame.pack(fill="x", pady=(0, 10))

        tk.Label(
            form_frame,
            text="Add Node",
            bg=self.new_network_page.SIDE_BAR_BG_COLOR,
            fg=self.FG_COLOR,
            font=self.PAGE_SUBHEADING_FONT
        ).pack(anchor="w", padx=10, pady=(0, 10))

        def validate_entries(*args):
            all_filled = True
            all_valid = True

            for key, data in self.node_entries.items():
                entry = data["entry"]
                field_type = data["type"]
                error_label = data["error"]

                value = entry.get().strip()

                if not value:
                    all_filled = False
                    error_label.config(text="Required field")
                    all_valid = False
                    continue

                try:
                    self.new_network_page.view.controller.network_manager.network.safe_cast(value, field_type)
                    entry.config(bg=self.BG_COLOR)
                    error_label.config(text="")
                except ValueError:
                    error_label.config(text=f"Must be {field_type.__name__}")
                    all_valid = False

            save_btn.config(state="normal" if all_filled and all_valid else "disabled")

        for field_name, field_type in self.node_schema.items():
            tk.Label(
                form_frame, text=field_name, bg=self.new_network_page.SIDE_BAR_BG_COLOR,
                fg=self.FG_COLOR, font=self.PAGE_SUBHEADING_2_FONT
            ).pack(anchor="w", padx=10)

            entry = tk.Entry(
                form_frame, font=(self.FONT_FAMILY, 12), bg=self.BG_COLOR,
                fg=self.FG_COLOR, relief="groove", insertbackground=self.FG_COLOR
            )
            entry.pack(fill="x", padx=10, pady=(0, 0))
            entry.bind("<KeyRelease>", validate_entries)

            error_label = tk.Label(
                form_frame, text="", fg="red",
                bg=self.new_network_page.SIDE_BAR_BG_COLOR,
                font=(self.FONT_FAMILY, 9)
            )
            error_label.pack(anchor="w", padx=10, pady=(0, 8))

            self.node_entries[field_name] = {
                "entry": entry,
                "type": field_type,
                "error": error_label
            }

        self.node_error_label = tk.Label(
            button_frame, text="", fg="red",
            bg=self.new_network_page.SIDE_BAR_BG_COLOR,
            font=(self.FONT_FAMILY, 10, "bold")
        )
        self.node_error_label.pack(fill="x", padx=10, pady=(0, 8))

        save_btn = tk.Button(
            button_frame, text="Save Node", command=self.save_node_data,
            bg=self.new_network_page.ICON_BUTTON_BG,
            fg=self.FG_COLOR,
            activebackground=self.ACTIVE_BG_COLOR,
            activeforeground=self.FG_COLOR,
            font=self.PAGE_SUBHEADING_2_FONT,
            state="disabled"
        )
        save_btn.pack(fill="x", padx=10, pady=(0, 8))

        cancel_btn = tk.Button(
            button_frame, text="Cancel", command=self.display_network_builder_menu,
            bg=self.new_network_page.ICON_BUTTON_BG,
            fg=self.FG_COLOR,
            activebackground=self.ACTIVE_BG_COLOR,
            activeforeground=self.FG_COLOR,
            font=self.PAGE_SUBHEADING_2_FONT
        )
        cancel_btn.pack(fill="x", padx=10)

        Logger.log("end display_add_node()")

    def save_node_data(self):
        self.new_network_page.save_element_data(
            entries=self.node_entries,
            class_lookup=self.node_class_lookup,
            selected_name=self.selected_node_name,
            error_label=self.node_error_label,
            add_method=self.new_network_page.view.controller.add_node,
            element_name="node"
        )


    # ----- EDGE MANAGEMENT -----
    def display_add_edge(self):
        Logger.log("start display_add_edge()")
        for widget in self.side_bar.winfo_children():
            widget.destroy()

        self.edge_entries = {}

        form_frame = tk.Frame(self.side_bar, bg=self.new_network_page.SIDE_BAR_BG_COLOR)
        form_frame.pack(fill="both", expand=True, pady=10)

        button_frame = tk.Frame(self.side_bar, bg=self.new_network_page.SIDE_BAR_BG_COLOR)
        button_frame.pack(fill="x", pady=(0, 10))

        tk.Label(
            form_frame,
            text="Add Edge",
            bg=self.new_network_page.SIDE_BAR_BG_COLOR,
            fg=self.FG_COLOR,
            font=self.PAGE_SUBHEADING_FONT
        ).pack(anchor="w", padx=10, pady=(0, 10))

        def validate_entries(*args):
            all_filled = True
            all_valid = True

            for key, data in self.edge_entries.items():
                entry = data["entry"]
                field_type = data["type"]
                error_label = data["error"]

                value = entry.get().strip()

                if not value:
                    all_filled = False
                    error_label.config(text="Required field")
                    all_valid = False
                    continue

                try:
                    self.new_network_page.view.controller.network_manager.network.safe_cast(value, field_type)
                    entry.config(bg=self.BG_COLOR)
                    error_label.config(text="")
                except ValueError:
                    error_label.config(text=f"Must be {field_type.__name__}")
                    all_valid = False

            save_btn.config(state="normal" if all_filled and all_valid else "disabled")

        for field_name, field_type in self.edge_schema.items():
            tk.Label(
                form_frame, text=field_name, bg=self.new_network_page.SIDE_BAR_BG_COLOR,
                fg=self.FG_COLOR, font=self.PAGE_SUBHEADING_2_FONT
            ).pack(anchor="w", padx=10)

            entry = tk.Entry(
                form_frame, font=(self.FONT_FAMILY, 12), bg=self.BG_COLOR,
                fg=self.FG_COLOR, relief="groove", insertbackground=self.FG_COLOR
            )
            entry.pack(fill="x", padx=10, pady=(0, 0))
            entry.bind("<KeyRelease>", validate_entries)

            error_label = tk.Label(
                form_frame, text="", fg="red",
                bg=self.new_network_page.SIDE_BAR_BG_COLOR,
                font=(self.FONT_FAMILY, 9)
            )
            error_label.pack(anchor="w", padx=10, pady=(0, 8))

            self.edge_entries[field_name] = {
                "entry": entry,
                "type": field_type,
                "error": error_label
            }

        self.edge_error_label = tk.Label(
        button_frame, text="", fg="red",
        bg=self.new_network_page.SIDE_BAR_BG_COLOR,
        font=(self.FONT_FAMILY, 10, "bold")
        )
        self.edge_error_label.pack(fill="x", padx=10, pady=(0, 8))

        save_btn = tk.Button(
            button_frame, text="Save Edge", command=self.save_edge_data,
            bg=self.new_network_page.ICON_BUTTON_BG,
            fg=self.FG_COLOR,
            activebackground=self.ACTIVE_BG_COLOR,
            activeforeground=self.FG_COLOR,
            font=self.PAGE_SUBHEADING_2_FONT,
            state="disabled"
        )
        save_btn.pack(fill="x", padx=10, pady=(0, 8))

        cancel_btn = tk.Button(
            button_frame, text="Cancel", command=self.display_network_builder_menu,
            bg=self.new_network_page.ICON_BUTTON_BG,
            fg=self.FG_COLOR,
            activebackground=self.ACTIVE_BG_COLOR,
            activeforeground=self.FG_COLOR,
            font=self.PAGE_SUBHEADING_2_FONT
        )
        cancel_btn.pack(fill="x", padx=10)

        Logger.log("end display_add_edge()")
        
    def save_edge_data(self):
        self.new_network_page.save_element_data(
            entries=self.edge_entries,
            class_lookup=self.edge_class_lookup,
            selected_name=self.selected_edge_name,
            error_label=self.edge_error_label,
            add_method=self.new_network_page.view.controller.add_edge,
            element_name="edge"
        )

    # ----- RELAX NETWORK -----
    def relax_network(self):
        Logger.log("start toolbar relax_network()")
        if self.new_network_page.view.controller.network_manager.network:
            self.new_network_page.view.controller.network_manager.relax_network()
            self.new_network_page.update_page("Network relaxed successfully")
        else:
            self.update_info_bar("No network to relax")
        Logger.log("end toolbar relax_network()")

    # ----- HELPERS -----
    def to_list(self, obj):
        if obj is None:
            return []
        if isinstance(obj, dict):
            return list(obj.values())
        if isinstance(obj, (list, set, tuple)):
            return list(obj)
        return [obj]  # fallback for single item

    # ----- ACTION BAR -----
    def setup_action_bar(self, controller):
        """Creates the action bar and populates it with buttons."""
        Logger.log(f"start setup_action_bar(self)")
        self.action_bar = tk.Frame(self.toolbar_frame, bg=self.new_network_page.ACTION_BAR_BG_COLOR)
        self.action_bar.place(x=0, y=0, relwidth=0.65, height=self.toolbar_height)
        self.add_action_bar_buttons(controller)
        Logger.log(f"end setup_action_bar(self)")

    def add_action_bar_buttons(self, controller):
        """Adds buttons to the action bar."""
        Logger.log(f"start add_action_bar_buttons(self)")
        # Create a frame inside action_bar to hold buttons
        self.button_frame = tk.Frame(self.action_bar, bg=self.new_network_page.BG_COLOR)
        self.button_frame.place(relx=0.5, rely=0.5, anchor="center")  # Centered placement

        button_config = {
            "import": (self.new_network_page.view.button_images["Small_Import"], self.on_import_click),
            "undo": (self.new_network_page.view.button_images["Small_Left_Arrow"], self.on_undo_click),
            "remove": (self.new_network_page.view.button_images["Small_X"], self.on_remove_click),
            "redo": (self.new_network_page.view.button_images["Small_Right_Arrow"], self.on_redo_click),
            "export": (self.new_network_page.view.button_images["Small_Export"], self.on_export_click),
        }
        
        for name, (icon, command) in button_config.items():
            button = tk.Button(
                self.button_frame, 
                image=icon, 
                command=command,
                bg=self.new_network_page.ICON_BUTTON_BG,
                cursor="arrow",
                border=0,
                state=tk.DISABLED,
                activebackground=self.new_network_page.ACTIVE_BG_COLOR
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
        self.new_network_page.on_import()
        Logger.log(f"end on_import_click(self)")

    def on_undo_click(self):
        Logger.log(f"start on_undo_click(self)")
        self.new_network_page.on_undo()
        self.update_info_bar("Undo performed")
        Logger.log(f"end on_undo_click(self)")

    def on_remove_click(self):
        Logger.log(f"start on_remove_click(self)")
        self.new_network_page.remove_selected_element()
        self.update_info_bar("Element removed")
        Logger.log(f"end on_remove_click(self)")

    def on_redo_click(self):
        Logger.log(f"start on_redo_click(self)")
        self.new_network_page.on_redo()
        self.update_info_bar("Redo performed")
        Logger.log(f"end on_redo_click(self)")

    def on_export_click(self):
        Logger.log(f"start on_export_click(self)")
        self.new_network_page.on_export()
        self.update_info_bar("Network exported successfully")
        Logger.log(f"end on_export_click(self)")

    # ----- INFO BAR -----
    def setup_info_bar(self):
        """Creates the info bar with larger, centered, and wrapped text."""
        Logger.log(f"start setup_info_bar(self)")
        self.info_bar = tk.Label(
            self.toolbar_frame,
            text="Network Builder Started",
            fg="white",
            bg=self.new_network_page.INFO_BAR_BG_COLOR,
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
