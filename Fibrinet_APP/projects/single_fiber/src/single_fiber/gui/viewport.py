"""
Chain viewport for 2D visualization.

Renders N-segment chain with:
    - Nodes as circles
    - Segments as lines (color-coded by tension/strain)
    - Axis markers
    - Scale bar
"""

import dearpygui.dearpygui as dpg
import numpy as np
from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass


@dataclass
class ViewportConfig:
    """Configuration for chain viewport."""
    width: int = 800
    height: int = 600
    padding: float = 50.0  # Pixels of padding around content
    node_radius: float = 8.0
    segment_thickness: float = 3.0
    fixed_node_color: Tuple[int, int, int, int] = (100, 100, 100, 255)
    free_node_color: Tuple[int, int, int, int] = (50, 150, 255, 255)
    dragged_node_color: Tuple[int, int, int, int] = (255, 200, 50, 255)
    ruptured_segment_color: Tuple[int, int, int, int] = (200, 50, 50, 128)
    intact_segment_color: Tuple[int, int, int, int] = (50, 200, 50, 255)
    background_color: Tuple[int, int, int, int] = (30, 30, 30, 255)


class ChainViewport:
    """
    2D viewport for visualizing N-segment chain.

    Handles coordinate transformation between nm (physics) and pixels (screen).
    """

    def __init__(self, config: Optional[ViewportConfig] = None):
        """
        Initialize viewport.

        Args:
            config: Viewport configuration (optional).
        """
        self.config = config or ViewportConfig()

        # Drawing state
        self._drawlist_tag: Optional[int] = None
        self._window_tag: Optional[int] = None

        # Transform parameters (nm -> pixel)
        self._scale = 1.0  # pixels per nm
        self._offset_x = 0.0  # pixel offset
        self._offset_y = 0.0

        # Bounds in nm
        self._bounds_min = np.array([0.0, 0.0, 0.0])
        self._bounds_max = np.array([100.0, 100.0, 100.0])

        # Interaction callbacks
        self._on_node_click: Optional[Callable[[int], None]] = None
        self._on_node_drag: Optional[Callable[[int, float, float], None]] = None
        self._on_node_release: Optional[Callable[[int], None]] = None

        # Current drag state
        self._dragging_node: Optional[int] = None
        self._node_positions_px: List[Tuple[float, float]] = []

        # View stability: lock bounds after initial setup to prevent jitter
        self._bounds_locked = False
        self._initial_bounds_set = False

    def create(
        self,
        parent_tag: Optional[int] = None,
        pos: Optional[Tuple[int, int]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> int:
        """
        Create viewport window and drawlist.

        Args:
            parent_tag: Parent window tag (optional).
            pos: Window position (x, y).
            width: Override width from config.
            height: Override height from config.

        Returns:
            Window tag.
        """
        # Allow overriding config dimensions
        actual_width = width if width is not None else self.config.width
        actual_height = height if height is not None else self.config.height

        window_kwargs = {
            "label": "Chain Viewport",
            "width": actual_width,
            "height": actual_height,
            "no_scrollbar": True,
            "no_scroll_with_mouse": True,
            "tag": "viewport_window"
        }
        if pos is not None:
            window_kwargs["pos"] = pos

        # Update config with actual dimensions for transforms
        self.config.width = actual_width
        self.config.height = actual_height

        with dpg.window(**window_kwargs) as self._window_tag:

            # Create drawlist for rendering
            self._drawlist_tag = dpg.add_drawlist(
                width=actual_width - 20,
                height=actual_height - 40,
                tag="chain_drawlist"
            )

            # Register mouse handlers
            with dpg.handler_registry():
                dpg.add_mouse_click_handler(callback=self._on_mouse_click)
                dpg.add_mouse_drag_handler(callback=self._on_mouse_drag)
                dpg.add_mouse_release_handler(callback=self._on_mouse_release)

        return self._window_tag

    def set_bounds(
        self,
        min_nm: np.ndarray,
        max_nm: np.ndarray,
        auto_fit: bool = True
    ) -> None:
        """
        Set coordinate bounds for visualization.

        Args:
            min_nm: Minimum corner in nm (3,).
            max_nm: Maximum corner in nm (3,).
            auto_fit: Whether to auto-fit view to bounds.
        """
        self._bounds_min = np.asarray(min_nm)
        self._bounds_max = np.asarray(max_nm)

        if auto_fit:
            self._compute_transform()

    def reset_view(self) -> None:
        """
        Reset view bounds to allow recalculation on next draw.

        Call this when the simulation is reset to re-fit the view
        to the initial chain configuration.
        """
        self._initial_bounds_set = False

    def _compute_transform(self) -> None:
        """Compute nm -> pixel transform to fit content with padding."""
        range_nm = self._bounds_max - self._bounds_min
        range_nm = np.maximum(range_nm, 1e-6)  # Avoid division by zero

        # Use X and Y dimensions for 2D view
        range_x = range_nm[0]
        range_y = range_nm[1]

        # Available space (with padding)
        avail_w = self.config.width - 2 * self.config.padding
        avail_h = self.config.height - 2 * self.config.padding

        # Scale to fit, maintaining aspect ratio
        scale_x = avail_w / range_x
        scale_y = avail_h / range_y
        self._scale = min(scale_x, scale_y)

        # Center the content
        content_w = range_x * self._scale
        content_h = range_y * self._scale
        self._offset_x = self.config.padding + (avail_w - content_w) / 2
        self._offset_y = self.config.padding + (avail_h - content_h) / 2

    def nm_to_pixel(self, pos_nm: np.ndarray) -> Tuple[float, float]:
        """
        Convert nm position to pixel coordinates.

        Args:
            pos_nm: Position in nm (3,) or (2,).

        Returns:
            (x_px, y_px) tuple.
        """
        x_nm = pos_nm[0] - self._bounds_min[0]
        y_nm = pos_nm[1] - self._bounds_min[1] if len(pos_nm) > 1 else 0.0

        x_px = self._offset_x + x_nm * self._scale
        # Flip Y for screen coordinates (origin at top-left)
        y_px = self.config.height - self.config.padding - y_nm * self._scale

        return (x_px, y_px)

    def pixel_to_nm(self, x_px: float, y_px: float) -> np.ndarray:
        """
        Convert pixel coordinates to nm position.

        Args:
            x_px: X pixel coordinate.
            y_px: Y pixel coordinate.

        Returns:
            Position in nm (3,) with z=0.
        """
        x_nm = (x_px - self._offset_x) / self._scale + self._bounds_min[0]
        y_nm = (self.config.height - self.config.padding - y_px) / self._scale + self._bounds_min[1]

        return np.array([x_nm, y_nm, 0.0])

    def draw_chain(
        self,
        nodes_nm: np.ndarray,
        segment_intact: List[bool],
        segment_tensions: Optional[List[float]] = None,
        dragged_node: Optional[int] = None,
        fixed_nodes: Optional[List[int]] = None
    ) -> None:
        """
        Draw the chain on the viewport.

        Args:
            nodes_nm: Node positions, shape (N+1, 3).
            segment_intact: List of booleans for each segment.
            segment_tensions: Optional tension values for color coding.
            dragged_node: Index of node being dragged (for highlighting).
            fixed_nodes: List of fixed node indices.
        """
        if self._drawlist_tag is None:
            return

        fixed_set = set(fixed_nodes or [0])

        # Clear previous drawing
        dpg.delete_item(self._drawlist_tag, children_only=True)

        # Store node positions for hit testing
        self._node_positions_px = []

        # Update bounds based on nodes (only on first draw to prevent jitter)
        # VISUAL STABILITY: Once initial bounds are set, we lock them to prevent
        # the view from jumping around as the chain stretches. This gives smooth,
        # predictable visualization during loading simulations.
        if len(nodes_nm) > 0 and not self._initial_bounds_set:
            # Use generous initial margin to accommodate expected stretching
            margin = 50.0  # nm margin (larger for expected deformation)
            min_pos = np.min(nodes_nm, axis=0) - margin
            max_pos = np.max(nodes_nm, axis=0) + margin
            # Extend X bounds to accommodate loading (chain stretches in +X)
            max_pos[0] = max_pos[0] + 100.0  # Extra room for stretching
            self.set_bounds(min_pos, max_pos)
            self._initial_bounds_set = True

        n_nodes = len(nodes_nm)
        n_segments = n_nodes - 1

        # Draw segments
        for i in range(n_segments):
            p1 = self.nm_to_pixel(nodes_nm[i])
            p2 = self.nm_to_pixel(nodes_nm[i + 1])

            if segment_intact[i]:
                color = self.config.intact_segment_color
            else:
                color = self.config.ruptured_segment_color

            dpg.draw_line(
                p1, p2,
                color=color,
                thickness=self.config.segment_thickness,
                parent=self._drawlist_tag
            )

        # Draw nodes
        for i in range(n_nodes):
            pos_px = self.nm_to_pixel(nodes_nm[i])
            self._node_positions_px.append(pos_px)

            if i == dragged_node:
                color = self.config.dragged_node_color
                radius = self.config.node_radius * 1.5
            elif i in fixed_set:
                color = self.config.fixed_node_color
                radius = self.config.node_radius
            else:
                color = self.config.free_node_color
                radius = self.config.node_radius

            dpg.draw_circle(
                pos_px,
                radius,
                color=color,
                fill=color,
                parent=self._drawlist_tag
            )

            # Draw node index label
            dpg.draw_text(
                (pos_px[0] + radius + 2, pos_px[1] - radius),
                str(i),
                color=(200, 200, 200, 255),
                size=12,
                parent=self._drawlist_tag
            )

        # Draw scale bar
        self._draw_scale_bar()

    def _draw_scale_bar(self) -> None:
        """Draw a scale bar in the bottom-left corner with adaptive length."""
        if self._drawlist_tag is None:
            return

        # Target scale bar length in pixels (aim for ~80-150 px)
        target_px = 100.0

        # Compute the nm length that would give target_px
        raw_nm = target_px / self._scale

        # Round to a nice value (1, 2, 5, 10, 20, 50, 100, 200, 500, ...)
        nice_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        scale_nm = nice_values[0]
        for val in nice_values:
            if val <= raw_nm * 1.5:  # Allow up to 1.5x target
                scale_nm = val

        scale_px = scale_nm * self._scale

        x1 = self.config.padding
        x2 = x1 + scale_px
        y = self.config.height - 30

        dpg.draw_line(
            (x1, y), (x2, y),
            color=(200, 200, 200, 255),
            thickness=2,
            parent=self._drawlist_tag
        )
        dpg.draw_text(
            (x1, y + 5),
            f"{scale_nm:.0f} nm",
            color=(200, 200, 200, 255),
            size=11,
            parent=self._drawlist_tag
        )

    def set_callbacks(
        self,
        on_node_click: Optional[Callable[[int], None]] = None,
        on_node_drag: Optional[Callable[[int, float, float], None]] = None,
        on_node_release: Optional[Callable[[int], None]] = None
    ) -> None:
        """
        Set interaction callbacks.

        Args:
            on_node_click: Called when node is clicked (node_index).
            on_node_drag: Called during drag (node_index, x_nm, y_nm).
            on_node_release: Called when drag ends (node_index).
        """
        self._on_node_click = on_node_click
        self._on_node_drag = on_node_drag
        self._on_node_release = on_node_release

    def _find_node_at(self, x_px: float, y_px: float) -> Optional[int]:
        """Find node index at pixel position, or None."""
        hit_radius = self.config.node_radius * 2

        for i, (nx, ny) in enumerate(self._node_positions_px):
            dist = np.sqrt((x_px - nx)**2 + (y_px - ny)**2)
            if dist <= hit_radius:
                return i

        return None

    def _on_mouse_click(self, sender, app_data) -> None:
        """Handle mouse click."""
        if app_data != 0:  # Left click only
            return

        mouse_pos = dpg.get_mouse_pos(local=False)
        node_idx = self._find_node_at(mouse_pos[0], mouse_pos[1])

        if node_idx is not None:
            self._dragging_node = node_idx
            if self._on_node_click:
                self._on_node_click(node_idx)

    def _on_mouse_drag(self, sender, app_data) -> None:
        """Handle mouse drag."""
        if self._dragging_node is None:
            return

        mouse_pos = dpg.get_mouse_pos(local=False)
        pos_nm = self.pixel_to_nm(mouse_pos[0], mouse_pos[1])

        if self._on_node_drag:
            self._on_node_drag(self._dragging_node, pos_nm[0], pos_nm[1])

    def _on_mouse_release(self, sender, app_data) -> None:
        """Handle mouse release."""
        if self._dragging_node is not None:
            if self._on_node_release:
                self._on_node_release(self._dragging_node)
            self._dragging_node = None
