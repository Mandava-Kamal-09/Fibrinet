"""
FibriNet Core V2 - End-to-End Strain Sweep Test
===============================================
This script demonstrates how to programmatically control the simulation
to perform a parameter study (Strain vs. Lysis Time).

It follows "Beginner Friendly" principles:
1. Clear setup
2. Automatic data loading
3. Visual progress updates
4. Automatic plotting and CSV export
"""

import sys
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter

def run_strain_sweep():
    # 1. Configuration
    # ----------------
    # Input file is in test/input_data relative to project root
    input_file = os.path.join(_project_root, "test", "input_data", "fibrin_network_big.xlsx")
    output_dir = os.path.join(_project_root, "test_results")
    os.makedirs(output_dir, exist_ok=True)

    # OPTIMAL: 11 strain points for smooth curve (0.0 to 0.5 in 0.05 increments)
    strains_to_test = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    results = []
    
    print(f"=== FibriNet Core V2: Strain Sweep Test ===")
    print(f"Input Network: {input_file}")
    print(f"Strains to test: {strains_to_test}")
    print("-" * 50)

    # 2. Initialize Adapter
    # ---------------------
    adapter = CoreV2GUIAdapter()
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found at {input_file}")
        return

    # Load once (or reload if needed - adapter handles re-creation in start_simulation)
    try:
        adapter.load_from_excel(input_file)
    except Exception as e:
        print(f"ERROR loading Excel: {e}")
        return

    # 3. Simulation Loop
    # ------------------
    for strain in strains_to_test:
        print(f"\n>> Running Strain = {strain*100:.0f}% ...")
        
        # Configure Simulation
        # OPTIMAL: λ₀=1.0, dt=0.01s (Safe), MaxTime=2000s (allow high-strain lysis)
        adapter.configure_parameters(
            plasmin_concentration=1.0,
            time_step=0.01,
            max_time=2000.0,
            applied_strain=strain
        )
        
        adapter.start_simulation()
        
        # Run until Lysis > 50% or finished
        start_time = time.time()
        t_50 = None
        
        while True:
            # Advance one step
            try:
                running = adapter.advance_one_batch()
            except Exception as e:
                print(f"  Simulation Crashed: {e}")
                break
                
            sim_time = adapter.get_current_time()
            lysis_frac = adapter.get_lysis_fraction()
            
            # Simple progress bar with Tension check
            if int(sim_time * 10) % 50 == 0:  # Every 5 sec sim time
                forces = adapter.get_forces()
                mean_tension = np.mean(list(forces.values())) if forces else 0.0
                print(f"   t={sim_time:.1f}s | Lysis={lysis_frac*100:.1f}% | Mean Tension={mean_tension:.2e} N", end='\r')
            
            # Record t50
            if t_50 is None and lysis_frac >= 0.5:
                t_50 = sim_time
            
            if not running:
                print(f"   t={sim_time:.1f}s | Finished ({adapter.termination_reason})")
                break
                
        duration = time.time() - start_time
        final_lysis = adapter.get_lysis_fraction()
        
        # Store result (use max type if t50 never reached)
        result_t50 = t_50 if t_50 is not None else 300.0
        results.append({
            "Strain": strain, 
            "T50_Lysis_Time_s": result_t50,
            "Final_Lysis": final_lysis,
            "Wall_Clock_Time_s": duration
        })
        
        print(f"   -> Result: T50 = {result_t50:.2f} s")

    # 4. Export & Plot
    # ----------------
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "strain_sweep_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to: {csv_path}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df["Strain"], df["T50_Lysis_Time_s"], 'o-', linewidth=2, markersize=8, color="#d32f2f")
    plt.title("Effect of Strain on Fibrinolysis Rate", fontsize=14)
    plt.xlabel("Applied Strain (ε)", fontsize=12)
    plt.ylabel("Time to 50% Lysis (s)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add theoretical curve hint
    # Theory: t50 ~ exp(beta * strain)
    # Fit simple expo to see alignment? No, just raw data for now.
    
    plot_path = os.path.join(output_dir, "strain_sweep_plot.png")
    plt.savefig(plot_path)
    print(f"Saved plot to: {plot_path}")
    print("\nTest Complete!")

if __name__ == "__main__":
    run_strain_sweep()
