import sys
import os
sys.path.insert(0, "Fibrinet_APP")

from src.config.feature_flags import FeatureFlags
from src.views.tkinter_view.research_simulation_page import SimulationController
import tempfile
import csv

print("Setting spatial flag to True...")
FeatureFlags.USE_SPATIAL_PLASMIN = True
print(f"Flag value: {FeatureFlags.USE_SPATIAL_PLASMIN}")

# Create test network
with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
    csv_path = f.name

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["n_id", "n_x", "n_y", "is_left_boundary", "is_right_boundary"])
    writer.writerow(["1", "0.0", "0.0", "1", "0"])
    writer.writerow(["2", "1.0", "0.0", "0", "1"])
    writer.writerow([])
    writer.writerow(["e_id", "n_from", "n_to", "thickness"])
    writer.writerow(["1", "1", "2", "1e-6"])
    writer.writerow([])
    writer.writerow(["key", "value"])
    writer.writerow(["spring_stiffness_constant", "1e-3"])
    writer.writerow(["coord_to_m", "1e-5"])
    writer.writerow(["thickness_to_m", "1.0"])
    writer.writerow(["L_seg", "5e-7"])
    writer.writerow(["N_pf", "50"])
    writer.writerow(["sigma_site", "1e-18"])
    writer.writerow(["P_bulk", "1e-6"])
    writer.writerow(["k_on0", "1e5"])
    writer.writerow(["k_off0", "0.1"])
    writer.writerow(["alpha", "0.0"])
    writer.writerow(["k_cat0", "1.0"])
    writer.writerow(["beta", "0.0"])
    writer.writerow(["K_crit", "1e-6"])
    writer.writerow(["N_seg_max", "100000"])

print("Loading network...")
controller = SimulationController()
controller.load_network(csv_path)

adapter = controller.state.loaded_network
edge0 = adapter.edges[0]
print(f"Initial: B_i={edge0.segments[0].B_i if edge0.segments else 'no segments'}")

controller.configure_phase1_parameters_from_ui(
    plasmin_concentration_str="1e-6",
    time_step_str="1e-4",
    max_time_str="1000.0",
    applied_strain_str="0.05",
)

print("Starting simulation...")
controller.start()

edge0 = adapter.edges[0]
print(f"After start: B_i={edge0.segments[0].B_i if edge0.segments else 'no segments'}")

print("Advancing one batch...")
sys.stdout.flush()
controller.advance_one_batch()
sys.stdout.flush()

edge0 = adapter.edges[0]
print(f"After batch: B_i={edge0.segments[0].B_i if edge0.segments else 'no segments'}")

os.unlink(csv_path)
print("Done")

