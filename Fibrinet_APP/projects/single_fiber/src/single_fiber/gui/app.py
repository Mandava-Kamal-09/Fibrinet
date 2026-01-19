"""
Main application for single fiber GUI.

Provides a complete DearPyGui-based application for
interactive chain visualization and simulation.
"""

import dearpygui.dearpygui as dpg
import numpy as np
from typing import Optional
from pathlib import Path

from ..config import SimulationConfig, load_config
from .controller import ChainController, SimulationMode
from .viewport import ChainViewport, ViewportConfig
from .panels import StatusPanel, ControlPanel, ParameterPanel


class SingleFiberApp:
    """
    Main application for single fiber simulation GUI.

    Usage:
        app = SingleFiberApp(config, n_segments=5)
        app.run()
    """

    def __init__(
        self,
        config: SimulationConfig,
        n_segments: int = 1,
        title: str = "Single Fiber Simulation"
    ):
        """
        Initialize application.

        Args:
            config: Simulation configuration.
            n_segments: Number of segments in chain.
            title: Window title.
        """
        self.config = config
        self.n_segments = n_segments
        self.title = title

        # Create controller (handles physics)
        self.controller = ChainController(
            config,
            n_segments,
            on_state_changed=self._on_state_changed
        )

        # Create viewport and panels
        self.viewport = ChainViewport(ViewportConfig())
        self.status_panel = StatusPanel()
        self.control_panel = ControlPanel(
            on_play=self._on_play,
            on_pause=self._on_pause,
            on_reset=self._on_reset,
            on_step=self._on_step,
            on_time_scale=self._on_time_scale
        )
        self.param_panel = ParameterPanel()

        # Animation state
        self._is_running = False
        self._frame_count = 0

    def run(self) -> None:
        """Run the application (blocking)."""
        # Initialize DearPyGui
        dpg.create_context()
        dpg.create_viewport(
            title=self.title,
            width=1050,
            height=650
        )

        # Create UI
        self._create_ui()

        # Setup and show viewport
        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Initial render
        self._update_display()

        # Main loop
        self._is_running = True
        while dpg.is_dearpygui_running() and self._is_running:
            self._frame_update()
            dpg.render_dearpygui_frame()

        # Cleanup
        dpg.destroy_context()

    def stop(self) -> None:
        """Stop the application."""
        self._is_running = False

    def _create_ui(self) -> None:
        """Create all UI elements."""
        # Main viewport
        self.viewport.create()
        self.viewport.set_callbacks(
            on_node_click=self._on_node_click,
            on_node_drag=self._on_node_drag,
            on_node_release=self._on_node_release
        )

        # Panels
        self.status_panel.create()
        self.control_panel.create()

        # Parameter panel
        param_dict = self._get_param_dict()
        self.param_panel.create(param_dict)

    def _get_param_dict(self) -> dict:
        """Get parameter dictionary for display."""
        params = {
            "Law": self.config.model.law,
            "Segments": self.n_segments,
        }

        if self.config.model.law == "hooke" and self.config.model.hooke:
            params["k (pN/nm)"] = self.config.model.hooke.k_pN_per_nm
            params["L0 (nm)"] = self.config.model.hooke.L0_nm
        elif self.config.model.law == "wlc" and self.config.model.wlc:
            params["Lp (nm)"] = self.config.model.wlc.Lp_nm
            params["Lc (nm)"] = self.config.model.wlc.Lc_nm
            params["kBT (pN*nm)"] = self.config.model.wlc.kBT_pN_nm

        params["dt (us)"] = self.config.dynamics.dt_us
        params["gamma"] = self.config.dynamics.gamma_pN_us_per_nm
        params["v (nm/us)"] = self.config.loading.v_nm_per_us

        return params

    def _frame_update(self) -> None:
        """Per-frame update (called each render frame)."""
        self._frame_count += 1

        # If playing, advance simulation
        if self.controller.mode == SimulationMode.PLAYING:
            # Step multiple times per frame for smoother animation
            # NOTE: steps_per_frame is controlled by time_scale, NOT physics dt
            steps_per_frame = self.controller.get_steps_per_frame()
            for _ in range(steps_per_frame):
                if not self.controller.is_complete():
                    self.controller.step()
                else:
                    self.controller.pause()
                    break

    def _update_display(self) -> None:
        """Update all display elements."""
        # Get current state
        chain_state = self.controller.get_chain_state()
        ctrl_state = self.controller.get_state()

        # Update viewport
        segment_intact = [s.is_intact for s in chain_state.segments]
        self.viewport.draw_chain(
            chain_state.nodes_nm,
            segment_intact,
            dragged_node=ctrl_state.dragged_node,
            fixed_nodes=self.controller.fixed_nodes
        )

        # Update panels
        self.status_panel.update(ctrl_state)
        self.control_panel.update_play_button(
            ctrl_state.mode == SimulationMode.PLAYING
        )

    def _on_state_changed(self) -> None:
        """Called by controller when state changes."""
        self._update_display()

    def _on_play(self) -> None:
        """Handle play button."""
        self.controller.toggle_play_pause()

    def _on_pause(self) -> None:
        """Handle pause."""
        self.controller.pause()

    def _on_reset(self) -> None:
        """Handle reset button."""
        self.controller.reset()
        self.viewport.reset_view()  # Reset view bounds for new chain config

    def _on_step(self) -> None:
        """Handle step button."""
        self.controller.step()

    def _on_time_scale(self, scale: float) -> None:
        """Handle time scale change."""
        self.controller.set_time_scale(scale)

    def _on_node_click(self, node_idx: int) -> None:
        """Handle node click."""
        self.controller.start_drag(node_idx)

    def _on_node_drag(self, node_idx: int, x_nm: float, y_nm: float) -> None:
        """Handle node drag."""
        self.controller.update_drag(node_idx, x_nm, y_nm)

    def _on_node_release(self, node_idx: int) -> None:
        """Handle node release."""
        self.controller.end_drag(node_idx)


def run_gui(config_path: Optional[str] = None, n_segments: int = 5) -> None:
    """
    Launch GUI from config file or defaults.

    Args:
        config_path: Path to YAML config file (optional).
        n_segments: Number of segments in chain.
    """
    if config_path:
        config = load_config(Path(config_path))
    else:
        # Create default config
        from ..config import (
            SimulationConfig, ModelConfig, HookeConfig,
            GeometryConfig, DynamicsConfig, LoadingConfig
        )
        config = SimulationConfig(
            model=ModelConfig(
                law="hooke",
                hooke=HookeConfig(k_pN_per_nm=0.1, L0_nm=100.0 / n_segments)
            ),
            geometry=GeometryConfig(
                x1_nm=[0.0, 0.0, 0.0],
                x2_nm=[100.0, 0.0, 0.0]
            ),
            dynamics=DynamicsConfig(
                dt_us=0.1,
                gamma_pN_us_per_nm=1.0
            ),
            loading=LoadingConfig(
                v_nm_per_us=0.5,
                t_end_us=100.0
            )
        )

    app = SingleFiberApp(config, n_segments=n_segments)
    app.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Single Fiber GUI")
    parser.add_argument(
        "-c", "--config",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "-n", "--segments",
        type=int,
        default=5,
        help="Number of segments (default: 5)"
    )
    args = parser.parse_args()

    run_gui(args.config, args.segments)
