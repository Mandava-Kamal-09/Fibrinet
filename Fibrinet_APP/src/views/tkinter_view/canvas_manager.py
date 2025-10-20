import tkinter as tk
from utils.logger.logger import Logger  # Importing the logger
from ...managers.network.networks.network_2d import Network2D

class CanvasManager:
    def __init__(self, containing_page):
        """Initialize CanvasManager with a reference to the page containing the canvas and create the canvas."""
        self.containing_page = containing_page
        self.canvas = None
        self.node_drawings = {}  # Maps node ID to canvas object
        self.node_labels = {}
        self.edge_drawings = {}  # Maps edge ID to canvas object
        self.current_network = None  # Store the last drawn network
        self.selected_element = None  # Store selected node or edge
        self.CANVAS_BG_COLOR = "grey90"
        self.NODE_COLOR = "black"
        self.SELECTED_NODE_COLOR = "red"
        self.EDGE_COLOR = "black"
        self.SELECTED_EDGE_COLOR = "red"
        self.NODE_RADIUS = 4  # Fixed node radius in pixels
        self.EDGE_WIDTH = 3
        Logger.log("CanvasManager initialized")

    def setup_canvas(self, container):
        """Creates and configures the canvas inside the given container."""
        Logger.log("Setting up the canvas...")
        canvas_frame = tk.Frame(container, bg=self.CANVAS_BG_COLOR)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(
            canvas_frame,
            bg=self.CANVAS_BG_COLOR,
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind resizing event and click event
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        Logger.log("Canvas setup completed")

    def draw_2d_network(self, network: Network2D):
        """Renders the given network onto the canvas, scaled to fit in the first quadrant."""
        Logger.log(f"start draw_2d_network(self, network={network})")
        Logger.log("Drawing network...")
        self.clear_canvas()
        self.current_network = network  # Store for redrawing
        Logger.log(f"--------------")
        network.log_network()

        if not network.nodes:
            return

        nodes = {node.n_id: node for node in network.nodes}
        edges = {edge.e_id: edge for edge in network.edges}

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width == 1 or height == 1:
            return  # Avoid drawing before fully rendered

        # Compute bounds, ensuring only first quadrant
        max_x = max(node.n_x for node in nodes.values())
        max_y = max(node.n_y for node in nodes.values())

        # Add space for labels
        label_padding = 30

        # Compute scaling
        scale_x = (width - label_padding) / (max_x + 20)
        scale_y = (height - label_padding) / (max_y + 20)
        scale = min(scale_x, scale_y)  # Maintain aspect ratio

        # Compute translation to keep (0,0) at bottom-left
        offset_x = 10 + label_padding  # Adjusted padding for labels
        offset_y = 10 + label_padding  # Adjusted padding for labels

        # Draw edges
        for edge in edges.values():
            node_from = nodes[edge.n_from]
            node_to = nodes[edge.n_to]
            x1, y1 = node_from.n_x * scale + offset_x, height - (node_from.n_y * scale + offset_y)
            x2, y2 = node_to.n_x * scale + offset_x, height - (node_to.n_y * scale + offset_y)
            edge_id = self.canvas.create_line(x1, y1, x2, y2, fill=self.EDGE_COLOR, width=self.EDGE_WIDTH)
            self.edge_drawings[edge_id] = edge

       # Draw nodes with a constant radius
        for node in nodes.values():
            x = node.n_x * scale + offset_x
            y = height - (node.n_y * scale + offset_y)

            # Determine radius based on whether node is fixed
            if getattr(node, "is_fixed", False):  # Use getattr to safely check attribute
                radius = self.NODE_RADIUS + 2  # Make fixed nodes slightly bigger
            else:
                radius = self.NODE_RADIUS

            node_id = self.canvas.create_oval(
                x - radius, y - radius,
                x + radius, y + radius,
                fill=self.NODE_COLOR, outline=self.NODE_COLOR
            )

            self.node_drawings[node_id] = node


        #     # Add label at the center of the node
        #     label_id = self.canvas.create_text(x, y, text=str(node.n_id), fill="white", font=("Arial", 10, "bold"))
        #     self.node_labels[node_id] = label_id  # Store label ID

                
        # # Draw axes
        # self.draw_axes(width, height, scale, offset_x, offset_y, max_x, max_y)
        Logger.log("end draw_2d_network(self, network)")

    def draw_axes(self, width, height, scale, offset_x, offset_y, max_x, max_y):
        """Draws X and Y axes starting from (0,0) in the bottom-left corner with number scale."""
        Logger.log("Drawing axes...")
        self.canvas.create_line(offset_x, 0, offset_x, height, fill="red")  # Y-axis
        self.canvas.create_line(0, height - offset_y, width, height - offset_y, fill="red")  # X-axis
        
        # Draw axis labels
        step_x = max_x / 10
        for i in range(11):
            real_x = i * step_x
            x_pos = real_x * scale + offset_x
            self.canvas.create_text(x_pos, height - offset_y + 20, text=f"{real_x:.1f}", fill="black", font=("Arial", 8))
        
        step_y = max_y / 10
        for i in range(11):
            real_y = i * step_y
            y_pos = height - (real_y * scale + offset_y)
            self.canvas.create_text(offset_x - 25, y_pos, text=f"{real_y:.1f}", fill="black", font=("Arial", 8))
        
    def clear_canvas(self):
        """Clears all drawings from the canvas and resets tracking dictionaries."""
        Logger.log("Clearing canvas...")
        if self.canvas:
            self.canvas.delete("all")  # Deletes all items on the canvas
        self.node_drawings = {}
        self.node_labels = {}
        self.edge_drawings = {}
        self.selected_element = None
        self.current_network = None  # Reset current network
        Logger.log("Canvas cleared.")

    def on_resize(self, event):
        """Redraws the network when the canvas is resized."""
        if self.current_network:
            self.draw_2d_network(self.current_network)

    def get_selected_element(self):
        """Returns the currently selected element (node/edge) from the canvas."""
        return self.selected_element
    
    def highlight_element(self, element_id):
        if element_id in self.node_drawings:
            self.canvas.itemconfig(element_id, fill=self.SELECTED_NODE_COLOR)  # Highlight node
        elif element_id in self.edge_drawings:
            self.canvas.itemconfig(element_id, fill=self.SELECTED_EDGE_COLOR)  # Highlight edge
    
    def remove_element_highlight(self, element_id):
        if element_id in self.node_drawings:
                self.canvas.itemconfig(self.selected_element, fill=self.NODE_COLOR)  # Reset node color
        elif element_id in self.edge_drawings:
            self.canvas.itemconfig(self.selected_element, fill=self.EDGE_COLOR)  # Reset edge color


    def select_element(self, element_id):
        """Highlights the given element (node/edge) on the canvas and sends the corresponding property ID."""
        # Remove highlight from previously selected element
        if self.selected_element:
            self.remove_element_highlight(self.selected_element)

        # Highlight the new element
        self.highlight_element(element_id)

        self.selected_element = element_id  # Update selected element

        self.containing_page.element_selected(self.convert_element_id_to_input_id(element_id), self.get_element_type(element_id))

    def convert_element_id_to_input_id(self, element_id):
        if element_id in self.node_drawings:
            node = self.node_drawings[element_id]
            return node.n_id  # Send N_ID instead
        elif element_id in self.edge_drawings:
            edge = self.edge_drawings[element_id]
            return edge.e_id  # Send E_ID instead
    
    def get_element_type(self, element_id):
        if element_id in self.node_drawings: return "node"
        elif element_id in self.edge_drawings: return "edge"

    def on_canvas_click(self, event):
        """Handles user clicks on the canvas and determines if an element was selected."""
        clicked_element = self.canvas.find_closest(event.x, event.y)
        if clicked_element:
            element_id = clicked_element[0]
            self.select_element(element_id)
            Logger.log(f"Element {element_id} selected")

    def clear_selected_element(self):
        """Unhighlights the currently selected element and clears the selection."""
        if self.selected_element:
            self.remove_element_highlight(self.selected_element)  # Unhighlight the element
            Logger.log(f"Element {self.selected_element} unhighlighted")
    
        self.selected_element = None  # Clear selection
        Logger.log("Selected element cleared")
