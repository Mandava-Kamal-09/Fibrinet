"""
GUI Integration Template for Core V2

Template demonstrating how to integrate Core V2 engine into research_simulation_page.py.
Covers unit conversion, parameter mapping, and simulation loop patterns.

Note: This is a reference template - do not run directly.
"""

import sys
import os

# Add project root to path for imports (if using this as a standalone script)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# =============================================================================
# CATCH A: Unit Conversion (Pixels vs Meters)
# =============================================================================

# At the top of research_simulation_page.py, add:
from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter

# In your class initialization or load_network method:
def load_network_with_core_v2(self, path):
    """
    Load network using Core V2 engine.

    CRITICAL: Set the correct scale factor based on your coordinate system.
    """
    # CATCH A FIX: Define scale factor
    # Option 1: If coordinates are in pixels from a 1000px-wide image representing 100 microns:
    PIXELS_TO_METERS = 100e-6 / 1000.0  # = 1e-7 m/px

    # Option 2: If coordinates are already in microns:
    MICRONS_TO_METERS = 1e-6

    # Option 3: Let the adapter auto-detect (uses default 1e-6)
    # (Good for test networks with coordinates ~5-30)

    # Create adapter
    self.adapter = CoreV2GUIAdapter()

    # Override scale factor if needed
    # self.adapter.coord_to_m = PIXELS_TO_METERS  # Uncomment if using pixels

    # Load network
    self.adapter.load_from_excel(path)

    print(f"[Catch A Check] coord_to_m = {self.adapter.coord_to_m:.6e}")
    print(f"[Catch A Check] Use 'python src/core/fibrinet_core_v2_adapter.py {path}' to verify units")


# =============================================================================
# CATCH B: Zombie Canvas (GUI Throttling)
# =============================================================================

class SimulationGUI:
    """Example GUI class showing Catch B mitigation."""

    def __init__(self):
        self.adapter = None
        self.is_running = False

        # CATCH B: Throttling parameters
        self.physics_steps_per_frame = 10  # Run 10 physics steps per 1 GUI update
        self.frame_delay_ms = 0  # 0 = as fast as possible while keeping GUI responsive

    def configure_parameters(self, plasmin_conc, dt, max_time, strain):
        """Configure simulation parameters from GUI inputs."""
        self.adapter.configure_parameters(
            plasmin_concentration=plasmin_conc,
            time_step=dt,
            max_time=max_time,
            applied_strain=strain
        )

    def start_simulation(self):
        """Start button handler."""
        if self.adapter is None:
            print("ERROR: No network loaded")
            return

        # Initialize simulation
        self.adapter.start_simulation()
        self.is_running = True

        # Start non-blocking loop (Catch C)
        self.run_simulation_step()

    # =========================================================================
    # CATCH C: Non-Blocking Loop
    # =========================================================================

    def run_simulation_step(self):
        """
        Non-blocking simulation step (Catch C mitigation).

        This is called via .after() to keep GUI responsive.
        DO NOT use a while loop here!
        """
        if not self.is_running:
            return

        # CATCH B: Batch multiple physics steps before updating GUI
        for _ in range(self.physics_steps_per_frame):
            has_more_steps = self.adapter.advance_one_batch()

            if not has_more_steps:
                # Simulation terminated
                self.is_running = False
                self.on_simulation_complete()
                return

        # Get render data (only once per frame, not per physics step!)
        render_data = self.adapter.get_render_data()

        # Update GUI (this is the slow part - only do it once per frame)
        self.update_canvas(render_data)
        self.update_metrics_display()

        # CATCH C: Schedule next frame using .after() instead of while loop
        # This returns control to Tkinter, keeping window responsive
        self.root.after(self.frame_delay_ms, self.run_simulation_step)

    def pause_simulation(self):
        """Pause button handler."""
        self.is_running = False

    def resume_simulation(self):
        """Resume button handler."""
        if self.adapter is None or self.adapter.simulation is None:
            print("ERROR: No simulation to resume")
            return

        self.is_running = True
        self.run_simulation_step()  # Restart loop

    def stop_simulation(self):
        """Stop button handler."""
        self.is_running = False
        # Optionally reset adapter state

    def on_simulation_complete(self):
        """Called when simulation terminates naturally."""
        print(f"Simulation complete: {self.adapter.termination_reason}")
        print(f"Final time: {self.adapter.get_current_time():.2f}s")
        print(f"Lysis fraction: {self.adapter.get_lysis_fraction():.3f}")

        # Final render
        render_data = self.adapter.get_render_data()
        self.update_canvas(render_data)
        self.update_metrics_display()

    # =========================================================================
    # Rendering Methods (Use render_data from adapter)
    # =========================================================================

    def update_canvas(self, render_data):
        """
        Update Tkinter canvas with network state.

        Args:
            render_data: Dict from adapter.get_render_data()
        """
        # Clear canvas
        self.canvas.delete("all")

        nodes = render_data['nodes']  # {node_id: (x, y)} in pixels/abstract units
        edges = render_data['edges']  # [(edge_id, n_from, n_to, is_ruptured), ...]
        forces = render_data['forces']  # {edge_id: force [N]}

        # Draw edges
        for edge_id, n_from, n_to, is_ruptured in edges:
            x1, y1 = nodes[n_from]
            x2, y2 = nodes[n_to]

            # Color based on state
            if is_ruptured:
                color = "red"
                width = 1
            else:
                # Color by force (optional)
                force = forces.get(edge_id, 0.0)
                color = self.get_force_color(force)
                width = 2

            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

        # Draw nodes (optional)
        for node_id, (x, y) in nodes.items():
            self.canvas.create_oval(x-2, y-2, x+2, y+2, fill="black")

    def get_force_color(self, force):
        """Map force to color for visualization."""
        # Normalize force to [0, 1] for color mapping
        # Adjust max_force based on your system
        max_force = 1e-11  # Example: 10 pN
        normalized = min(force / max_force, 1.0)

        # Blue (low) -> Red (high)
        if normalized < 0.5:
            # Blue to green
            r = 0
            g = int(255 * normalized * 2)
            b = 255
        else:
            # Green to red
            r = int(255 * (normalized - 0.5) * 2)
            g = 255 - r
            b = 0

        return f"#{r:02x}{g:02x}{b:02x}"

    def update_metrics_display(self):
        """Update text display of simulation metrics."""
        if self.adapter is None:
            return

        t = self.adapter.get_current_time()
        lysis = self.adapter.get_lysis_fraction()
        n_ruptured = self.adapter.simulation.state.n_ruptured if self.adapter.simulation else 0
        n_total = len(self.adapter.simulation.state.fibers) if self.adapter.simulation else 0

        # Update labels (example)
        self.time_label.config(text=f"Time: {t:.2f} s")
        self.lysis_label.config(text=f"Lysis: {lysis:.1%}")
        self.rupture_label.config(text=f"Ruptured: {n_ruptured}/{n_total}")


# =============================================================================
# Integration Checklist
# =============================================================================

"""
Phase 2 Integration Steps:
--------------------------

1. [DONE] Add Core V2 adapter import
2. [DONE] Replace Phase1NetworkAdapter with CoreV2GUIAdapter
3. [DONE] Set correct coord_to_m scale factor (Catch A)
4. [DONE] Implement batched physics steps (Catch B)
5. [DONE] Use .after() loop instead of while (Catch C)
6. [TEST] Verify with test network
7. [COMPARE] Check results against legacy system

Safety Checks Before Running:
------------------------------

1. Run unit verification:
   python src/core/fibrinet_core_v2_adapter.py test/input_data/TestNetwork.xlsx

2. Check that coord_to_m gives reasonable lengths (1e-7 to 1e-3 m)

3. Start with small test network (< 100 edges)

4. Use high physics_steps_per_frame (10-100) for smooth GUI

5. Monitor console for physics warnings

Expected Behavior:
------------------

- GUI should remain responsive (clickable, not "Not Responding")
- Canvas should update smoothly every few physics steps
- Simulation should run faster than legacy (vectorized solver)
- Forces should be displayed in Newtons (may be small: ~1e-12 N)
- Lysis fraction should increase over time
- Ruptured edges should turn red

Known Differences from Legacy:
-------------------------------

1. Forces in Newtons (not abstract units)
2. Time in seconds (not arbitrary steps)
3. Avalanche dynamics may emerge (cooperative rupture)
4. Energy minimization may give slightly different relaxed positions
5. Stochastic chemistry may show fluctuations
"""
