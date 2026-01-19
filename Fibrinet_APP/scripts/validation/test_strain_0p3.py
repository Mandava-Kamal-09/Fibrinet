"""
Quick validation test: Strain=0.3 with timestep capping
=========================================================
This test verifies that the timestep cap (dt_chem=min(dt, 0.005))
helps stabilize the simulation at strain=0.3.
"""

import sys
import os
import time

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter

# Load network
input_file = os.path.join(_project_root, "test", "input_data", "fibrin_network_big.xlsx")
adapter = CoreV2GUIAdapter()
adapter.load_from_excel(input_file)

# Configure with strain=0.3 (previously problematic)
# User requests dt=0.01 but adapter will cap it to 0.005
adapter.configure_parameters(
    plasmin_concentration=1.0,
    time_step=0.01,  # Will be capped to 0.005 internally
    max_time=100.0,
    applied_strain=0.3
)

print("=" * 70)
print("STRAIN 0.3 STABILITY TEST")
print("=" * 70)
print(f"Applied strain: 30%")
print(f"Requested dt: 0.01s (will be capped to 0.005s)")
print(f"Max time: 100s")
print("-" * 70)

adapter.start_simulation()
start_wall_time = time.time()

# Run simulation
while True:
    running = adapter.advance_one_batch()
    sim_time = adapter.get_current_time()
    lysis = adapter.get_lysis_fraction()

    # Print progress
    if int(sim_time * 10) % 20 == 0:
        mean_tension = adapter.prev_mean_tension if adapter.prev_mean_tension else 0.0
        max_tension = adapter.get_max_tension()
        print(f"t={sim_time:5.1f}s | Lysis={lysis*100:5.1f}% | Mean F={mean_tension:.2e} N | Max F={max_tension:.2e} N", end='\r')

    if not running:
        print(f"\nt={sim_time:.1f}s | Terminated: {adapter.termination_reason}")
        break

wall_time = time.time() - start_wall_time
final_lysis = adapter.get_lysis_fraction()

print("-" * 70)
print(f"RESULTS:")
print(f"  Termination: {adapter.termination_reason}")
print(f"  Simulation time: {sim_time:.1f}s")
print(f"  Wall clock time: {wall_time:.1f}s")
print(f"  Final lysis: {final_lysis*100:.1f}%")
print(f"  Fibers cleaved: {adapter.simulation.state.n_ruptured}/{len(adapter.simulation.state.fibers)}")

if adapter.termination_reason == "network_cleared":
    print(f"  [CLEARED] Network cleared at t={sim_time:.1f}s")
else:
    print(f"  [TIMEOUT] Did not clear within {adapter.max_time}s")

print("=" * 70)
print("\nExpected: Simulation should run stably without force singularities")
print("Check: Mean and max forces should remain <1e-7 N throughout")
print("=" * 70)
