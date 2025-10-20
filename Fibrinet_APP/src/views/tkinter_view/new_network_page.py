from utils.logger.logger import Logger 
import tkinter as tk
from .tkinter_view import TkinterView
from .canvas_manager import CanvasManager
from .new_network_page_managers.toolbar_manager import ToolbarManager

class NewNetworkPage(TkinterView):
    """
    The NewNetworkPage allows users to build a new network.
    """

    def __init__(self, view):
        """Initialize NewNetworkPage with canvas and toolbar managers."""
        Logger.log("start NewNetworkPage.__init__(self, view)")
        self.view = view

        # NEW NETWORK PAGE STYLE
        self.ACTION_BAR_BG_COLOR = self.view.BG_COLOR
        self.INFO_BAR_BG_COLOR = "gray11"
        self.SIDE_BAR_BG_COLOR = "gray13"
        self.BG_COLOR = view.BG_COLOR
        self.FG_COLOR = view.FG_COLOR
        self.button_images = view.button_images
        self.PAGE_HEADING_FONT = view.HEADING_FONT
        self.PAGE_HEADING_BG = view.HEADING_BG
        self.PAGE_SUBHEADING_FONT = view.SUBHEADING_FONT
        self.SUBHEADING_2_FONT = view.SUBHEADING_2_FONT
        self.PAGE_SUBHEADING_BG = view.SUBHEADING_BG
        self.canvas_manager = CanvasManager(self)
        self.toolbar_manager = ToolbarManager(self)

        Logger.log("end NewNetworkPage.__init__(self, view)")

    def show_page(self, container):
        """
        Displays the NewNetworkPage, including the canvas for the network and the toolbar.
        
        Args:
            container (tk.Frame): The container frame where the page content will be added.
        """
        Logger.log(f"start show_page(self, container)")
        # SETUP THE CANVAS
        self.canvas_manager.setup_canvas(container)
        # SETUP THE TOOLBAR
        self.toolbar_manager.setup_toolbar(self.view.controller, container)
        # CHECK TO SEE IF A NETWORK IS ALREADY LOADED
        if self.view.controller.network_manager.get_network() is None:
            # IF NO NETWORK IS LOADED, CLEAR THE CANVAS AND SETUP THE PAGE
            Logger.log("No network loaded, setting up new network page.")
            self.view.controller.network_manager.reset_network_and_state() 
            self.canvas_manager.clear_canvas()
            self.update_canvas()  # Draw the empty canvas
        else:
            # IF A NETWORK IS ALREADY LOADED
            Logger.log("Network already loaded, drawing existing network.")
            # IF A NETWORK IS LOADED DISPLAY IT AND THE MENU
            self.update_canvas()
            self.toolbar_manager.display_network_builder_menu()
        
        # SET NETWORK STATE MANAGER TO IS NEW NETWORK
        self.view.controller.network_manager.state_manager.is_new_network = True
        
        # UPDATE THE DEGRADATION ENGINE STRATEGY
        self.view.controller.network_manager.update_degradation_engine_strategy()
        Logger.log(f"end show_page(self, container)")

    def remove_selected_element(self):
        """
        Removes the currently selected element (node/edge) from the network.
        This updates the canvas after the element is removed.
        """
        Logger.log(f"start remove_selected_element(self)")

        if self.canvas_manager.selected_element:
            # GET ELEMENT ID
            element_id = self.canvas_manager.selected_element
            
            # GET ELEMENT TYPE
            element_type = "Node" if self.canvas_manager.get_element_type(element_id) == "node" else "Edge"
            
            # GET INPUT ID 
            input_id = self.canvas_manager.convert_element_id_to_input_id(element_id)
            
            # REMOVE ELEMENT
            Logger.log(f"Removing {element_type} {input_id}")
            if element_type == "Node": self.view.controller.degrade_node(input_id)
            elif element_type == "Edge": self.view.controller.degrade_edge(input_id)
            else: raise Exception()
            
            # CLEAR SELECTED ELEMENT
            self.canvas_manager.clear_selected_element()

            # UPDATE PAGE
            self.update_page(info_bar_message=f"{element_type} {input_id} removed")
        Logger.log(f"end remove_selected_element(self)")

    def save_element_data(self, entries, class_lookup, selected_name, error_label, add_method, element_name):
        Logger.log(f"start save_{element_name}_data()")
        all_valid = True

        # Validate inputs
        for key, data in entries.items():
            entry = data["entry"]
            field_type = data["type"]
            error_lbl = data["error"]
            value = entry.get().strip()
            try:
                casted_value = self.view.controller.network_manager.get_network().safe_cast(value, field_type)
                entry.config(bg=self.BG_COLOR)
                error_lbl.config(text="")
            except ValueError:
                entry.config(bg="#ffe6e6")
                error_lbl.config(text=f"Must be {field_type.__name__}")
                all_valid = False

        if not all_valid:
            Logger.log(f"{element_name.capitalize()} data validation failed. Not saving.")
            return

        # Build the data dict
        element_data = {}
        for key, data in entries.items():
            value = data["entry"].get().strip()
            field_type = data["type"]
            casted_value = self.view.controller.network_manager.get_network().safe_cast(value, field_type)
            element_data[key] = casted_value

        element_cls = class_lookup.get(selected_name)
        if not element_cls:
            Logger.log(f"{element_name.capitalize()} class {selected_name} not found.")
            error_label.config(text=f"{element_name.capitalize()} class {selected_name} not found.")
            return

        try:
            new_element = element_cls(attributes=element_data)
            add_method(new_element)
        except ValueError as e:
            Logger.log(f"Error adding {element_name}: {e}")
            error_label.config(text=str(e))
        else:
            # Extract appropriate ID field
            id_field = "n_id" if element_name == "node" else "e_id"
            element_id = getattr(new_element, id_field, "unknown")

            Logger.log(f"{element_name.capitalize()} {element_id} added successfully!")
            error_label.config(text="")
            self.update_page(info_bar_message=f"{element_name.capitalize()} {element_id} added successfully!")
            self.toolbar_manager.display_network_builder_menu()

        Logger.log(f"end save_{element_name}_data()")

    def update_page(self, info_bar_message):
        Logger.log(f"start update_page(self, info_bar_message={info_bar_message})")
        self.update_canvas()
        self.update_toolbar()
        self.update_info_bar(info=info_bar_message)
        Logger.log(f"end update_page(self, info_bar_message)")
    
    def update_canvas(self):
        """
        Updates the canvas by redrawing the entire network to reflect the latest state.
        """
        Logger.log(f"start update_canvas(self)")
        if not self.view.controller.network_manager.get_network():
            Logger.log("No network available to update canvas.")
            return
        self.view.controller.network_manager.get_network().log_network()
        self.canvas_manager.draw_2d_network(self.view.controller.network_manager.get_network())
        Logger.log(f"end update_canvas(self)")

    def update_toolbar(self):
        Logger.log(f"start update_toolbar(self)")
        self.toolbar_manager.disable_action_bar_button("remove") if self.canvas_manager.select_element == None else self.toolbar_manager.enable_action_bar_button("remove")
        self.toolbar_manager.disable_action_bar_button("undo") if self.view.controller.network_manager.state_manager.undo_disabled else self.toolbar_manager.enable_action_bar_button("undo")
        self.toolbar_manager.disable_action_bar_button("redo") if self.view.controller.network_manager.state_manager.redo_disabled else self.toolbar_manager.enable_action_bar_button("redo")
        self.toolbar_manager.disable_action_bar_button("export") if self.view.controller.network_manager.state_manager.export_disabled else self.toolbar_manager.enable_action_bar_button("export")
        Logger.log(f"end update_toolbar(self)")
    
    def on_undo(self):
        """
        Handles the undo action, reverting the network to the previous state.
        The canvas is updated after undoing the action.
        """
        Logger.log(f"start on_undo(self)")
        self.view.controller.undo_degradation()
        self.update_page(info_bar_message="Undo Performed")
        Logger.log(f"end on_undo(self)")
    
    def on_redo(self):
        """
        Handles the redo action, reapplying the most recent undone action.
        The canvas is updated after redoing the action.
        """
        Logger.log(f"start on_redo(self)")
        self.view.controller.redo_degradation()
        self.update_page(info_bar_message="Undo Performed")
        Logger.log(f"end on_redo(self)")
    
    def on_export(self):
        """
        Exports the current state of the network to a file.
        """
        Logger.log(f"start on_export(self)")
        self.view.show_page("export", from_new_network=True)
        Logger.log(f"end on_export(self)")
    
    def on_import(self):
        """
        Prepares the page for a new network import by clearing the canvas,
        resetting managers, and then showing the input page.
        """
        Logger.log(f"start on_import(self)")
        self.view.controller.network_manager.reset_network_and_state() 
        self.canvas_manager.clear_canvas()
        Logger.log("Canvas cleared and network state reset.")
        self.view.show_page("input")
        Logger.log(f"end on_import(self)")

    def element_selected(self, element_id, element_type):
        """
        Handles the action when an edge is selected, updating the info bar in the toolbar.
        """
        Logger.log(f"start element_selected(self, {element_id}, {element_type})")
        self.update_info_bar(f"{"Node" if element_type == "node" else "Edge"} {element_id} selected.")
        self.toolbar_manager.enable_action_bar_button("remove")
        Logger.log(f"end element_selected(self, element_id, element_type)")

    def update_info_bar(self, info):
        """
        Updates the info bar with information.
        """
        Logger.log(f"start update_info_bar(self, info)")
        self.toolbar_manager.update_info_bar(info)
        Logger.log(f"end update_info_bar(self, info)")



