from utils.logger.logger import Logger 
import tkinter as tk
from .tkinter_view import TkinterView
from .canvas_manager import CanvasManager
from .modify_page_managers.toolbar_manager import ToolbarManager

class ModifyPage(TkinterView):
    """
    The ModifyPage allows users to interact with the network.
    Users can modify, remove, and navigate through different states.
    """

    def __init__(self, view):
        """Initialize ModifyPage with canvas and toolbar managers."""
        Logger.log("start ModifyPage.__init__(self, view)")
        self.view = view

        # MODIFY PAGE STYLE
        self.ACTION_BAR_BG_COLOR = self.view.BG_COLOR
        self.INFO_BAR_BG_COLOR = "gray11"
        self.BG_COLOR = view.BG_COLOR
        self.FG_COLOR = view.FG_COLOR
        self.button_images = view.button_images
        self.PAGE_HEADING_FONT = view.HEADING_FONT
        self.PAGE_HEADING_BG = view.HEADING_BG
        self.PAGE_SUBHEADING_FONT = view.SUBHEADING_FONT
        self.PAGE_SUBHEADING_BG = view.SUBHEADING_BG

        self.canvas_manager = CanvasManager(self)
        self.toolbar_manager = ToolbarManager(self)

        Logger.log("end ModifyPage.__init__(self, view)")

    def show_page(self, container):
        """
        Displays the Modify Page, including the canvas for the network and the toolbar.
        
        Args:
            container (tk.Frame): The container frame where the page content will be added.
        """
        Logger.log(f"start show_page(self, container)")
        self.canvas_manager.setup_canvas(container)
        self.canvas_manager.draw_2d_network(self.view.controller.network_manager.get_network())
        self.toolbar_manager.setup_toolbar(self.view.controller, container)
        # SET NETWORK STATE MANAGER
        self.view.controller.network_manager.state_manager.is_new_network = False
        
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
            else: raise ValueError(f"Unknown element type: {element_type}")
            
            # CLEAR SELECTED ELEMENT
            self.canvas_manager.clear_selected_element()

            # UPDATE PAGE
            self.update_page(info_bar_message=f"{element_type} {input_id} removed")
        Logger.log(f"end remove_selected_element(self)")

    def update_page(self, info_bar_message):
        self.update_canvas()
        self.update_toolbar()
        self.update_info_bar(info=info_bar_message)
    
    def update_canvas(self):
        """
        Updates the canvas by redrawing the entire network to reflect the latest state.
        """
        Logger.log(f"start update_canvas(self)")
        self.canvas_manager.draw_2d_network(self.view.controller.network_manager.get_network())
        Logger.log(f"end update_canvas(self)")

    def update_toolbar(self):
        self.toolbar_manager.disable_action_bar_button("remove") if self.canvas_manager.select_element == None else self.toolbar_manager.enable_action_bar_button("remove")
        self.toolbar_manager.disable_action_bar_button("undo") if self.view.controller.network_manager.state_manager.undo_disabled else self.toolbar_manager.enable_action_bar_button("undo")
        self.toolbar_manager.disable_action_bar_button("redo") if self.view.controller.network_manager.state_manager.redo_disabled else self.toolbar_manager.enable_action_bar_button("redo")
        self.toolbar_manager.disable_action_bar_button("export") if self.view.controller.network_manager.state_manager.export_disabled else self.toolbar_manager.enable_action_bar_button("export")

    
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
        self.view.show_page("export")
        Logger.log(f"end on_export(self)")
    
    def on_import(self):
        """
        Imports new network data, reloading the network and updating the canvas.
        """
        Logger.log(f"start on_import(self)")
        self.view.controller.network_manager.reset_network_and_state() 
        self.canvas_manager.clear_canvas()
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
