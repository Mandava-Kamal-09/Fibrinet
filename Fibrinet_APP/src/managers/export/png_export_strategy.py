from PIL import Image, ImageDraw
from ..network.networks.base_network import BaseNetwork
from .image_export_strategy import ImageExportStrategy
import io

class PngExportStrategy(ImageExportStrategy):
    def generate_export(self, network_state_history: list[BaseNetwork]):
        """Create a combined PNG preview of all states."""
        files = []
        
        # Create an image for each network state
        images = [self._create_network_image(network) for network in network_state_history]
        
        # Combine all the images into one
        if images:
            combined_img = self._combine_images_vertically(images)
            buffer = io.BytesIO()
            combined_img.save(buffer, format='PNG')
            files.append(("combined_network.png", buffer.getvalue()))
        
        return files

    def _create_network_image(self, network: BaseNetwork, padding=50, scale=10):
        """Draw a single network state image."""
        if not network.nodes:
            return Image.new("RGB", (100, 100), "white")  # Return a blank image if no nodes exist

        nodes = {node.n_id: node for node in network.nodes}
        edges = {edge.e_id: edge for edge in network.edges}

        # Bounds
        min_x = min(node.n_x for node in nodes.values())
        max_x = max(node.n_x for node in nodes.values())
        min_y = min(node.n_y for node in nodes.values())
        max_y = max(node.n_y for node in nodes.values())

        # Dimensions
        width = int((max_x - min_x) * scale + 2 * padding)
        height = int((max_y - min_y) * scale + 2 * padding)

        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        # Edges
        for edge in edges.values():
            node_from = nodes[edge.n_from]
            node_to = nodes[edge.n_to]

            x1 = int((node_from.n_x - min_x) * scale + padding)
            y1 = int(height - ((node_from.n_y - min_y) * scale + padding))  # Invert Y-coordinate
            x2 = int((node_to.n_x - min_x) * scale + padding)
            y2 = int(height - ((node_to.n_y - min_y) * scale + padding))  # Invert Y-coordinate

            draw.line((x1, y1, x2, y2), fill="black", width=3)

        # Nodes
        for node in nodes.values():
            x = int((node.n_x - min_x) * scale + padding)
            y = int(height - ((node.n_y - min_y) * scale + padding))  # Invert Y-coordinate

            draw.ellipse([x-5, y-5, x+5, y+5], fill="black", outline="black")

        return img
    
    def _combine_images_vertically(self, image_list):
        """Stack images vertically into one PNG."""
        max_width = max(img.width for img in image_list)
        total_height = sum(img.height for img in image_list)
        
        combined_img = Image.new("RGB", (max_width, total_height), "white")  # White background
        y_offset = 0
        for img in image_list:
            combined_img.paste(img, (0, y_offset))
            y_offset += img.height
        
        return combined_img
