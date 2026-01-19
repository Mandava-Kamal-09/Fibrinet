"""
Quick test: Manual batch advancement (no GUI)
This confirms the Core V2 adapter works step-by-step
"""
import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter

# Load network
adapter = CoreV2GUIAdapter()
network_file = os.path.join(_project_root, "test/input_data/fibrin_network_big.xlsx")
print(f"Loading: {network_file}")
adapter.load_from_excel(network_file)
print(f"[OK] Loaded: {len(adapter._edges_raw)} edges")

# Configure
adapter.configure_parameters(
    plasmin_concentration=1.0,
    time_step=0.01,
    max_time=1000.0,
    applied_strain=0.1  # Low strain
)
print("[OK] Parameters configured")

# Start
adapter.start_simulation()
print("[OK] Simulation initialized")
print(f"  Initial time: {adapter.get_current_time():.3f}s")
print(f"  Initial lysis: {adapter.get_lysis_fraction()*100:.1f}%")

# Manual batch advancement (5 batches)
print("\n--- Advancing 5 batches manually ---")
for i in range(5):
    running = adapter.advance_one_batch()
    t = adapter.get_current_time()
    lysis = adapter.get_lysis_fraction()
    n_ruptured = adapter.simulation.state.n_ruptured if adapter.simulation else 0

    print(f"Batch {i+1}: t={t:.3f}s, lysis={lysis*100:.1f}%, ruptured={n_ruptured}, running={running}")

    if not running:
        print(f"  → Terminated: {adapter.termination_reason}")
        break

print("\n[SUCCESS] Manual advancement works!")
print("If this runs successfully, the GUI should work too.")
print("Make sure you're NOT clicking 'Start' button in GUI!")
