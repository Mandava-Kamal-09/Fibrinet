"""
UI panels for chain simulation GUI.

Provides:
    - Status panel (time, strain, tension, etc.)
    - Control panel (play/pause, reset, settings)
    - Enzyme panel (hazard model selection, parameters)
    - Mode panel (Novice/Advanced toggle + presets)
    - Model Scope panel (model limitations)
"""

import dearpygui.dearpygui as dpg
from typing import Optional, Callable, Dict, Any, List

from .controller import ControllerState, SimulationMode


# =============================================================================
# Mode Panel (Novice/Advanced + Presets)
# =============================================================================

class ModePanel:
    """
    Panel for Novice/Advanced mode toggle and presets selection.

    Novice mode: Shows only essential controls, hides parameter sliders,
                 provides curated presets with safe ranges.
    Advanced mode: Full parameter control, all hazard models, sweep hooks.
    """

    def __init__(
        self,
        on_mode_change: Optional[Callable[[bool], None]] = None,
        on_preset_change: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize mode panel.

        Args:
            on_mode_change: Callback when mode changes (advanced: bool)
            on_preset_change: Callback when preset changes (preset_name)
        """
        self._on_mode_change = on_mode_change
        self._on_preset_change = on_preset_change

        self._window_tag: Optional[int] = None
        self._mode_checkbox: Optional[int] = None
        self._preset_combo: Optional[int] = None
        self._preset_desc: Optional[int] = None

        self._is_advanced = False
        self._preset_names: List[str] = []
        self._preset_display_names: Dict[str, str] = {}

    def create(self, pos: tuple = (10, 10), width: int = 400) -> int:
        """
        Create mode panel window.

        Args:
            pos: Window position
            width: Panel width

        Returns:
            Window tag
        """
        # Load presets
        try:
            from .presets import list_presets, get_preset_display_names
            self._preset_names = list_presets(novice_only=not self._is_advanced)
            self._preset_display_names = get_preset_display_names()
        except ImportError:
            self._preset_names = []
            self._preset_display_names = {}

        preset_display = [
            self._preset_display_names.get(n, n) for n in self._preset_names
        ]

        with dpg.window(
            label="Mode & Presets",
            pos=pos,
            width=width,
            height=140,
            no_resize=True,
            no_move=False,
            tag="mode_panel"
        ) as self._window_tag:

            with dpg.group(horizontal=True):
                dpg.add_text("Mode:", color=(200, 200, 255))
                self._mode_checkbox = dpg.add_checkbox(
                    label="Advanced Mode",
                    default_value=False,
                    callback=self._mode_changed
                )
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text(
                        "Novice: Curated presets, safe ranges\n"
                        "Advanced: Full parameter control"
                    )

            dpg.add_separator()

            dpg.add_text("Preset:", color=(150, 200, 150))
            with dpg.group(horizontal=True):
                self._preset_combo = dpg.add_combo(
                    items=preset_display,
                    default_value=preset_display[0] if preset_display else "",
                    width=300,
                    callback=self._preset_changed
                )
                dpg.add_button(
                    label="Info",
                    width=50,
                    callback=self._show_preset_info
                )

            self._preset_desc = dpg.add_text(
                "",
                wrap=390,
                color=(180, 180, 180)
            )

            # Show initial preset description
            if self._preset_names:
                self._update_preset_description(self._preset_names[0])

        return self._window_tag

    def _mode_changed(self, sender, app_data) -> None:
        """Handle mode toggle."""
        self._is_advanced = app_data

        # Update preset list based on mode
        try:
            from .presets import list_presets, get_preset_display_names
            self._preset_names = list_presets(novice_only=not self._is_advanced)
            self._preset_display_names = get_preset_display_names()
            preset_display = [
                self._preset_display_names.get(n, n) for n in self._preset_names
            ]
            dpg.configure_item(self._preset_combo, items=preset_display)
            if preset_display:
                dpg.set_value(self._preset_combo, preset_display[0])
                self._update_preset_description(self._preset_names[0])
        except ImportError:
            pass

        if self._on_mode_change:
            self._on_mode_change(app_data)

    def _preset_changed(self, sender, app_data) -> None:
        """Handle preset selection."""
        # Find preset name from display name
        preset_name = None
        for name, display in self._preset_display_names.items():
            if display == app_data:
                preset_name = name
                break

        if preset_name:
            self._update_preset_description(preset_name)
            if self._on_preset_change:
                self._on_preset_change(preset_name)

    def _update_preset_description(self, preset_name: str) -> None:
        """Update preset description text."""
        try:
            from .presets import get_preset
            preset = get_preset(preset_name)
            desc = preset.description[:100]
            if len(preset.description) > 100:
                desc += "..."
            dpg.set_value(self._preset_desc, desc)
        except (ImportError, KeyError):
            dpg.set_value(self._preset_desc, "")

    def _show_preset_info(self) -> None:
        """Show detailed preset info."""
        try:
            from .presets import get_preset
            current_display = dpg.get_value(self._preset_combo)
            for name, display in self._preset_display_names.items():
                if display == current_display:
                    preset = get_preset(name)
                    print(f"\n{'='*60}")
                    print(f"PRESET: {preset.display_name}")
                    print(f"{'='*60}")
                    print(f"\nDescription:\n{preset.description}")
                    print(f"\nExpected Behavior:\n{preset.expected_behavior}")
                    print(f"\nSafety Notes:\n{preset.safety_notes}")
                    print(f"{'='*60}\n")
                    break
        except (ImportError, KeyError):
            print("Preset info not available")

    @property
    def is_advanced(self) -> bool:
        """Return whether advanced mode is enabled."""
        return self._is_advanced

    def get_current_preset_name(self) -> Optional[str]:
        """Get currently selected preset name."""
        current_display = dpg.get_value(self._preset_combo)
        for name, display in self._preset_display_names.items():
            if display == current_display:
                return name
        return None


# =============================================================================
# Model Scope Panel
# =============================================================================

class ModelScopePanel:
    """
    Panel displaying model limitations and scope.

    Always visible to prevent misinterpretation of results.
    """

    def __init__(self):
        self._window_tag: Optional[int] = None

    def create(self, pos: tuple = (420, 10), width: int = 390) -> int:
        """
        Create model scope panel.

        Args:
            pos: Window position
            width: Panel width

        Returns:
            Window tag
        """
        with dpg.window(
            label="Model Scope & Limitations",
            pos=pos,
            width=width,
            height=140,
            no_resize=True,
            no_move=False,
            no_close=True,
            tag="scope_panel"
        ) as self._window_tag:

            dpg.add_text("This simulation is:", color=(255, 200, 100))

            limitations = [
                "Overdamped quasi-static (no inertia)",
                "No thermal fluctuations / Brownian motion",
                "No bending or torsional mechanics",
                "No fiber-fiber interactions",
                "Hazard models are candidate forms (not fitted to data)",
            ]

            for item in limitations:
                dpg.add_text(f"- {item}", color=(200, 200, 200), indent=10)

        return self._window_tag


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
            dpg.add_text("Animation Speed", color=(150, 200, 150))

            with dpg.tooltip(dpg.last_item()):
                dpg.add_text("Controls visual playback only.\nPhysics timestep dt is fixed.")

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


class EnzymePanel:
    """
    Enzyme control panel for strain-enzyme coupling experiments.

    Provides:
        - Hazard model dropdown
        - Auto-generated parameter sliders
        - Live hazard rate display
        - Rupture probability per step
    """

    # Physical descriptions for tooltips
    MODEL_TOOLTIPS = {
        "constant": "Constant hazard rate, independent of strain/tension.\nNull model for baseline enzymatic activity.",
        "linear_strain": "Linear strain-dependent: λ = λ₀(1 + αε).\nMild mechanosensitivity.",
        "exponential_strain": "Exponential strain-dependent: λ = λ₀·exp(αε).\nStrong mechanosensitivity with cooperative unfolding.",
        "bell_slip": "Bell model slip bond: λ = λ₀·exp(βT).\nClassic force-accelerated bond rupture.",
        "catch_slip": "Catch-slip bond: biphasic force response.\nLow force stabilizes, high force destabilizes.",
    }

    def __init__(
        self,
        on_model_change: Optional[Callable[[str], None]] = None,
        on_param_change: Optional[Callable[[str, float], None]] = None,
        on_enable_change: Optional[Callable[[bool], None]] = None
    ):
        """
        Initialize enzyme panel.

        Args:
            on_model_change: Callback when hazard model changes (model_name)
            on_param_change: Callback when parameter changes (param_name, value)
            on_enable_change: Callback when enzyme enabled/disabled (enabled)
        """
        self._on_model_change = on_model_change
        self._on_param_change = on_param_change
        self._on_enable_change = on_enable_change

        self._window_tag: Optional[int] = None
        self._model_combo: Optional[int] = None
        self._enable_checkbox: Optional[int] = None
        self._hazard_rate_text: Optional[int] = None
        self._mean_time_text: Optional[int] = None
        self._rupture_prob_text: Optional[int] = None
        self._param_sliders: Dict[str, int] = {}
        self._param_container: Optional[int] = None

        # Current state
        self._current_model: str = "constant"
        self._current_params: Dict[str, float] = {}
        self._enabled: bool = False

    def create(self, pos: tuple = (10, 620), width: int = 400) -> int:
        """
        Create enzyme panel window.

        Args:
            pos: Window position (x, y)
            width: Panel width

        Returns:
            Window tag
        """
        with dpg.window(
            label="Enzyme Coupling",
            pos=pos,
            width=width,
            height=200,
            no_resize=True,
            no_move=False,
            tag="enzyme_panel"
        ) as self._window_tag:

            dpg.add_text("Strain-Enzyme Coupling", color=(200, 200, 255))
            dpg.add_separator()

            # Enable checkbox
            with dpg.group(horizontal=True):
                self._enable_checkbox = dpg.add_checkbox(
                    label="Enable Enzyme",
                    default_value=False,
                    callback=self._enable_changed
                )
                dpg.add_text("(not fitted to data)", color=(150, 150, 150))

            dpg.add_spacer(height=5)

            # Model selection
            dpg.add_text("Hazard Model:", color=(150, 200, 150))
            with dpg.group(horizontal=True):
                self._model_combo = dpg.add_combo(
                    items=["constant", "linear_strain", "exponential_strain",
                           "bell_slip", "catch_slip"],
                    default_value="constant",
                    width=150,
                    callback=self._model_changed
                )
                dpg.add_button(
                    label="?",
                    width=25,
                    callback=self._show_model_help
                )

            dpg.add_spacer(height=5)

            # Parameter sliders container
            dpg.add_text("Parameters:", color=(150, 200, 150))
            self._param_container = dpg.add_group()

            dpg.add_spacer(height=5)

            # Live display
            dpg.add_text("Current State:", color=(200, 150, 150))
            self._hazard_rate_text = dpg.add_text("λ = 0.000 /µs")
            self._mean_time_text = dpg.add_text("Mean time: not defined")
            self._rupture_prob_text = dpg.add_text("P(rupture in Δt) = 0.0%")

        # Initialize with default model
        self._update_param_sliders("constant")

        return self._window_tag

    def _update_param_sliders(self, model_name: str) -> None:
        """Update parameter sliders for selected model."""
        # Import registry here to avoid circular import
        try:
            from ..enzyme_models import get_hazard_spec, get_default_params
        except ImportError:
            return

        # Clear existing sliders
        if self._param_container is not None:
            dpg.delete_item(self._param_container, children_only=True)
        self._param_sliders.clear()

        try:
            spec = get_hazard_spec(model_name)
            defaults = get_default_params(model_name)
        except KeyError:
            return

        # Create sliders for each parameter
        for param_name in spec.required_params:
            desc, units, (min_val, max_val) = spec.param_descriptions[param_name]
            default = defaults.get(param_name, (min_val + max_val) / 2)

            with dpg.group(horizontal=True, parent=self._param_container):
                dpg.add_text(f"{param_name}:", indent=10)

                slider = dpg.add_slider_float(
                    default_value=default,
                    min_value=min_val,
                    max_value=max_val,
                    width=150,
                    format="%.4g",
                    callback=lambda s, a, p=param_name: self._param_changed(p, a)
                )
                self._param_sliders[param_name] = slider

                dpg.add_text(f"({units})", color=(150, 150, 150))

            self._current_params[param_name] = default

    def _model_changed(self, sender, app_data) -> None:
        """Handle hazard model selection change."""
        self._current_model = app_data
        self._update_param_sliders(app_data)

        if self._on_model_change:
            self._on_model_change(app_data)

    def _param_changed(self, param_name: str, value: float) -> None:
        """Handle parameter slider change."""
        self._current_params[param_name] = value

        if self._on_param_change:
            self._on_param_change(param_name, value)

    def _enable_changed(self, sender, app_data) -> None:
        """Handle enable checkbox change."""
        self._enabled = app_data

        if self._on_enable_change:
            self._on_enable_change(app_data)

    def _show_model_help(self) -> None:
        """Show help popup for current model."""
        tooltip = self.MODEL_TOOLTIPS.get(
            self._current_model,
            "No description available."
        )
        # Use DearPyGui's popup or just print for now
        print(f"\n{self._current_model}:\n{tooltip}\n")

    def update_hazard_display(
        self,
        hazard_rate: float,
        dt_us: float = 0.1
    ) -> None:
        """
        Update live hazard rate display.

        Args:
            hazard_rate: Current hazard rate in 1/µs
            dt_us: Time step for probability calculation
        """
        if self._hazard_rate_text is None:
            return

        # Format hazard rate
        if hazard_rate < 0.001:
            rate_str = f"λ = {hazard_rate:.2e} /µs"
        else:
            rate_str = f"λ = {hazard_rate:.4f} /µs"
        dpg.set_value(self._hazard_rate_text, rate_str)

        # Format mean time (1/λ)
        if self._mean_time_text is not None:
            if hazard_rate > 0:
                mean_time = 1.0 / hazard_rate
                if mean_time > 1e6:
                    mean_str = f"Mean time: {mean_time:.2e} µs"
                elif mean_time > 100:
                    mean_str = f"Mean time: {mean_time:.0f} µs"
                else:
                    mean_str = f"Mean time: {mean_time:.2f} µs"
            else:
                mean_str = "Mean time: ∞ (no cleavage)"
            dpg.set_value(self._mean_time_text, mean_str)

        # Compute and display rupture probability with explicit dt
        import math
        if hazard_rate > 0 and dt_us > 0:
            prob = 1 - math.exp(-hazard_rate * dt_us)
            prob_pct = prob * 100
            if prob_pct < 0.01:
                prob_str = f"P(rupture in Δt={dt_us:.3g} µs) = {prob_pct:.2e}%"
            else:
                prob_str = f"P(rupture in Δt={dt_us:.3g} µs) = {prob_pct:.3f}%"
        else:
            prob_str = f"P(rupture in Δt={dt_us:.3g} µs) = 0.0%"

        dpg.set_value(self._rupture_prob_text, prob_str)

    @property
    def enabled(self) -> bool:
        """Return whether enzyme is enabled."""
        return self._enabled

    @property
    def current_model(self) -> str:
        """Return current hazard model name."""
        return self._current_model

    @property
    def current_params(self) -> Dict[str, float]:
        """Return copy of current parameters."""
        return self._current_params.copy()

    def get_hazard_rate(self, strain: float, tension_pN: float) -> float:
        """
        Compute current hazard rate for given state.

        Args:
            strain: Current strain
            tension_pN: Current tension in pN

        Returns:
            Hazard rate in 1/µs, or 0 if disabled
        """
        if not self._enabled:
            return 0.0

        try:
            from ..enzyme_models import get_hazard
            hazard_fn = get_hazard(self._current_model)
            return hazard_fn(strain, tension_pN, self._current_params)
        except Exception:
            return 0.0
