"""
UI panels for chain simulation GUI.

Provides:
    - Status panel (time, strain, tension, etc.)
    - Control panel (play/pause, reset, settings)
"""

import dearpygui.dearpygui as dpg
from typing import Optional, Callable
from dataclasses import dataclass

from .controller import ControllerState, SimulationMode


class StatusPanel:
    """
    Status panel showing simulation state.

    Displays:
        - Current time
        - Global strain
        - Maximum tension
        - Segment status
        - Relaxation status
    """

    def __init__(self):
        self._window_tag: Optional[int] = None
        self._time_text: Optional[int] = None
        self._strain_text: Optional[int] = None
        self._tension_text: Optional[int] = None
        self._segments_text: Optional[int] = None
        self._relax_text: Optional[int] = None
        self._mode_text: Optional[int] = None
        self._warning_text: Optional[int] = None

    def create(self, pos: tuple = (820, 10), width: int = 200) -> int:
        """
        Create status panel window.

        Args:
            pos: Window position (x, y).
            width: Panel width.

        Returns:
            Window tag.
        """
        with dpg.window(
            label="Status",
            pos=pos,
            width=width,
            height=250,
            no_resize=True,
            no_move=False,
            tag="status_panel"
        ) as self._window_tag:

            dpg.add_text("Simulation Status", color=(200, 200, 255))
            dpg.add_separator()

            self._mode_text = dpg.add_text("Mode: PAUSED")
            self._time_text = dpg.add_text("Time: 0.00 us")
            dpg.add_spacer(height=5)

            dpg.add_text("Mechanics", color=(150, 200, 150))
            self._strain_text = dpg.add_text("Strain: 0.00%")
            self._tension_text = dpg.add_text("Tension: 0.00 pN")
            dpg.add_spacer(height=5)

            dpg.add_text("Chain", color=(200, 150, 150))
            self._segments_text = dpg.add_text("Segments: 0/0 intact")
            self._relax_text = dpg.add_text("Relaxation: OK")
            dpg.add_spacer(height=5)

            # Warning display (hidden by default)
            self._warning_text = dpg.add_text("", color=(255, 200, 50), wrap=190)

        return self._window_tag

    def update(self, state: ControllerState) -> None:
        """
        Update status display.

        Args:
            state: Current controller state.
        """
        if self._window_tag is None:
            return

        # Mode
        mode_str = state.mode.name
        if state.dragged_node is not None:
            mode_str = f"DRAGGING node {state.dragged_node}"
        dpg.set_value(self._mode_text, f"Mode: {mode_str}")

        # Time
        dpg.set_value(self._time_text, f"Time: {state.t_us:.2f} us")

        # Strain
        strain_pct = state.global_strain * 100
        dpg.set_value(self._strain_text, f"Strain: {strain_pct:.1f}%")

        # Tension
        dpg.set_value(self._tension_text, f"Tension: {state.max_tension_pN:.2f} pN")

        # Segments
        dpg.set_value(
            self._segments_text,
            f"Segments: {state.n_intact}/{state.n_segments} intact"
        )

        # Relaxation
        if state.relax_converged:
            dpg.set_value(self._relax_text, "Relaxation: OK")
            dpg.configure_item(self._relax_text, color=(100, 255, 100))
        else:
            dpg.set_value(self._relax_text, "Relaxation: NOT CONVERGED")
            dpg.configure_item(self._relax_text, color=(255, 100, 100))

        # Warnings
        if state.warnings:
            warning_text = "Warnings:\n" + "\n".join(f"- {w}" for w in state.warnings[:3])
            dpg.set_value(self._warning_text, warning_text)
        else:
            dpg.set_value(self._warning_text, "")


class ControlPanel:
    """
    Control panel for simulation interaction.

    Provides:
        - Play/Pause/Reset buttons
        - Time scale slider
        - Segment count control
    """

    def __init__(
        self,
        on_play: Optional[Callable[[], None]] = None,
        on_pause: Optional[Callable[[], None]] = None,
        on_reset: Optional[Callable[[], None]] = None,
        on_step: Optional[Callable[[], None]] = None,
        on_time_scale: Optional[Callable[[float], None]] = None
    ):
        self._on_play = on_play
        self._on_pause = on_pause
        self._on_reset = on_reset
        self._on_step = on_step
        self._on_time_scale = on_time_scale

        self._window_tag: Optional[int] = None
        self._play_button: Optional[int] = None

    def create(self, pos: tuple = (820, 270), width: int = 200) -> int:
        """
        Create control panel window.

        Args:
            pos: Window position (x, y).
            width: Panel width.

        Returns:
            Window tag.
        """
        with dpg.window(
            label="Controls",
            pos=pos,
            width=width,
            height=180,
            no_resize=True,
            no_move=False,
            tag="control_panel"
        ) as self._window_tag:

            dpg.add_text("Playback", color=(200, 200, 255))
            dpg.add_separator()

            with dpg.group(horizontal=True):
                self._play_button = dpg.add_button(
                    label="Play",
                    width=60,
                    callback=self._play_clicked
                )
                dpg.add_button(
                    label="Step",
                    width=60,
                    callback=self._step_clicked
                )

            dpg.add_button(
                label="Reset",
                width=125,
                callback=self._reset_clicked
            )

            dpg.add_spacer(height=10)
            dpg.add_text("Time Scale", color=(150, 200, 150))

            dpg.add_slider_float(
                label="",
                default_value=1.0,
                min_value=0.1,
                max_value=5.0,
                width=120,
                callback=self._time_scale_changed,
                format="%.1fx"
            )

            dpg.add_spacer(height=10)
            dpg.add_text("Tips:", color=(150, 150, 200))
            dpg.add_text("- Click & drag nodes", bullet=True)
            dpg.add_text("- Node 0 is fixed", bullet=True)

        return self._window_tag

    def update_play_button(self, is_playing: bool) -> None:
        """Update play button label based on state."""
        if self._play_button is not None:
            label = "Pause" if is_playing else "Play"
            dpg.configure_item(self._play_button, label=label)

    def _play_clicked(self) -> None:
        if self._on_play:
            self._on_play()

    def _step_clicked(self) -> None:
        if self._on_step:
            self._on_step()

    def _reset_clicked(self) -> None:
        if self._on_reset:
            self._on_reset()

    def _time_scale_changed(self, sender, app_data) -> None:
        if self._on_time_scale:
            self._on_time_scale(app_data)


class ParameterPanel:
    """
    Parameter display panel showing simulation parameters.
    """

    def __init__(self):
        self._window_tag: Optional[int] = None

    def create(
        self,
        config_dict: dict,
        pos: tuple = (820, 460),
        width: int = 200
    ) -> int:
        """
        Create parameter panel.

        Args:
            config_dict: Dictionary of parameters to display.
            pos: Window position.
            width: Panel width.

        Returns:
            Window tag.
        """
        with dpg.window(
            label="Parameters",
            pos=pos,
            width=width,
            height=200,
            no_resize=True,
            no_move=False,
            tag="param_panel"
        ) as self._window_tag:

            dpg.add_text("Model Parameters", color=(200, 200, 255))
            dpg.add_separator()

            for key, value in config_dict.items():
                if isinstance(value, float):
                    dpg.add_text(f"{key}: {value:.4g}")
                else:
                    dpg.add_text(f"{key}: {value}")

        return self._window_tag
