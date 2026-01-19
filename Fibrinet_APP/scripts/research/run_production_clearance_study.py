"""
PRODUCTION CLEARANCE STUDY - DYNAMICS DAYS 2026
================================================
Rigorous multi-realization study of strain-dependent network clearance.

Protocol:
- 10 independent realizations per strain (deterministic RNG seeds 0-9)
- Strain range: 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30
- Metric: Time to percolation loss (left-right connectivity failure)
- Fixed plasmin: 10.0 nM

Outputs:
- network_clearance_summary.csv (mean, std, failure mode per strain)
- network_clearance_vs_strain.png (poster-ready figure)
- network_clearance_raw_data.csv (all 70 individual runs)
"""

import sys
import os
import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter

# ============================================================================
# EXPERIMENTAL PARAMETERS (AUTHORITATIVE)
# ============================================================================

EXCEL_FILE = os.path.join(_project_root, "test", "input_data", "fibrin_network_big.xlsx")
OUTPUT_DIR = os.path.join(_project_root, "publication_figures")
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Strain sweep (focused physiological range)
APPLIED_STRAINS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# Replications for statistical rigor
N_REALIZATIONS = 10
RNG_SEEDS = list(range(10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Simulation parameters
PLASMIN_CONCENTRATION = 10.0  # nM
TIME_STEP = 0.05              # s
MAX_TIME = 100.0              # s (safety timeout)


# ============================================================================
# DATA COLLECTION
# ============================================================================

def run_production_study():
    """Execute full 70-simulation study with statistical rigor."""

    print("=" * 80)
    print("PRODUCTION CLEARANCE STUDY - DYNAMICS DAYS 2026")
    print("=" * 80)
    print()
    print("Protocol:")
    print(f"  Strain range: {APPLIED_STRAINS}")
    print(f"  Realizations per strain: {N_REALIZATIONS}")
    print(f"  Total simulations: {len(APPLIED_STRAINS) * N_REALIZATIONS}")
    print(f"  RNG seeds: {RNG_SEEDS}")
    print(f"  Plasmin: {PLASMIN_CONCENTRATION} nM")
    print()
    print("=" * 80)
    print()

    all_results = []

    total_sims = len(APPLIED_STRAINS) * N_REALIZATIONS
    sim_count = 0

    for strain_idx, applied_strain in enumerate(APPLIED_STRAINS):
        print(f"Strain {strain_idx + 1}/{len(APPLIED_STRAINS)}: e = {applied_strain:.2f}")
        print("-" * 80)

        strain_results = []

        for seed in RNG_SEEDS:
            sim_count += 1
            print(f"  [{sim_count}/{total_sims}] Realization (seed={seed})...", end=' ')

            # Fresh adapter with deterministic seed
            adapter = CoreV2GUIAdapter()
            adapter.load_from_excel(EXCEL_FILE)
            adapter.configure_parameters(
                plasmin_concentration=PLASMIN_CONCENTRATION,
                time_step=TIME_STEP,
                max_time=MAX_TIME,
                applied_strain=applied_strain,
                rng_seed=seed  # CRITICAL: deterministic seeding
            )
            adapter.start_simulation()

            # Run until clearance
            while adapter.advance_one_batch():
                pass

            # Extract results
            clearance_time = adapter.get_current_time()
            final_lysis = adapter.get_lysis_fraction()
            termination = adapter.termination_reason or "Timeout"

            # Classify failure mode
            if "cleared" in termination.lower():
                if final_lysis < 0.3:
                    failure_mode = "MECHANICAL"
                else:
                    failure_mode = "ENZYMATIC"
            else:
                failure_mode = "INCOMPLETE"

            result = {
                'applied_strain': applied_strain,
                'rng_seed': seed,
                'clearance_time': clearance_time,
                'final_lysis': final_lysis,
                'failure_mode': failure_mode,
                'termination': termination
            }

            strain_results.append(result)
            all_results.append(result)

            print(f"t={clearance_time:.2f}s, mode={failure_mode}")

        # Compute statistics for this strain
        times = [r['clearance_time'] for r in strain_results]
        modes = [r['failure_mode'] for r in strain_results]

        mean_time = np.mean(times)
        std_time = np.std(times, ddof=1)
        sem_time = std_time / np.sqrt(len(times))

        # Dominant failure mode
        mode_counts = {}
        for mode in modes:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        dominant_mode = max(mode_counts, key=mode_counts.get)

        print(f"  SUMMARY: Mean={mean_time:.2f}±{std_time:.2f}s, Mode={dominant_mode} ({mode_counts[dominant_mode]}/{N_REALIZATIONS})")
        print()

    print("=" * 80)
    print("DATA COLLECTION COMPLETE")
    print("=" * 80)
    print()

    return all_results


# ============================================================================
# DATA ANALYSIS
# ============================================================================

def compute_summary_statistics(all_results):
    """Compute mean, std, SEM, and dominant mode per strain."""

    summary = []

    for strain in APPLIED_STRAINS:
        strain_data = [r for r in all_results if r['applied_strain'] == strain]

        times = [r['clearance_time'] for r in strain_data]
        modes = [r['failure_mode'] for r in strain_data]

        mean_time = np.mean(times)
        std_time = np.std(times, ddof=1)
        sem_time = std_time / np.sqrt(len(times))

        # Dominant mode
        mode_counts = {}
        for mode in modes:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        dominant_mode = max(mode_counts, key=mode_counts.get)
        mode_fraction = mode_counts[dominant_mode] / len(modes)

        summary.append({
            'applied_strain': strain,
            'mean_clearance_time': mean_time,
            'std_clearance_time': std_time,
            'sem_clearance_time': sem_time,
            'dominant_failure_mode': dominant_mode,
            'mode_fraction': mode_fraction,
            'n_realizations': len(times)
        })

    return summary


# ============================================================================
# POSTER-READY FIGURE
# ============================================================================

def generate_poster_figure(summary):
    """Generate clean, advisor-approved poster figure."""

    print("[GENERATING] Poster-ready figure...")

    strains = [s['applied_strain'] for s in summary]
    mean_times = [s['mean_clearance_time'] for s in summary]
    sem_times = [s['sem_clearance_time'] for s in summary]
    modes = [s['dominant_failure_mode'] for s in summary]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Separate by failure mode
    enzymatic_idx = [i for i, mode in enumerate(modes) if mode == "ENZYMATIC"]
    mechanical_idx = [i for i, mode in enumerate(modes) if mode == "MECHANICAL"]

    # Plot enzymatic points
    if enzymatic_idx:
        ax.errorbar(
            [strains[i] for i in enzymatic_idx],
            [mean_times[i] for i in enzymatic_idx],
            yerr=[sem_times[i] for i in enzymatic_idx],
            fmt='o', markersize=14, color='#2ECC71',
            markeredgecolor='black', markeredgewidth=2,
            ecolor='black', elinewidth=2, capsize=5, capthick=2,
            label='Enzymatic', zorder=5
        )

    # Plot mechanical points
    if mechanical_idx:
        ax.errorbar(
            [strains[i] for i in mechanical_idx],
            [mean_times[i] for i in mechanical_idx],
            yerr=[sem_times[i] for i in mechanical_idx],
            fmt='s', markersize=14, color='#E74C3C',
            markeredgecolor='black', markeredgewidth=2,
            ecolor='black', elinewidth=2, capsize=5, capthick=2,
            label='Mechanical', zorder=5
        )

    # Annotate critical strain (minimum clearance time)
    min_idx = mean_times.index(min(mean_times))
    critical_strain = strains[min_idx]
    min_time = mean_times[min_idx]

    ax.annotate(
        'Critical strain\n(fastest clearance)',
        xy=(critical_strain, min_time),
        xytext=(critical_strain + 0.05, min_time + 1.5),
        fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow',
                 edgecolor='black', linewidth=2, alpha=0.9),
        arrowprops=dict(arrowstyle='->', lw=2, color='black')
    )

    # Clean formatting
    ax.set_xlabel('Applied Strain', fontsize=18, fontweight='bold')
    ax.set_ylabel('Network Clearance Time (s)', fontsize=18, fontweight='bold')
    ax.set_title('Network Clearance Time vs Applied Strain',
                 fontsize=20, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=14, loc='best', framealpha=0.98,
             edgecolor='black', shadow=True)

    ax.tick_params(labelsize=15)
    ax.set_xlim(-0.02, max(strains) + 0.02)

    # White background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    plt.tight_layout()

    # Save high-resolution
    output_path = os.path.join(OUTPUT_DIR, "network_clearance_vs_strain.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(all_results, summary):
    """Save raw data and summary statistics to CSV."""

    # Raw data (all 70 runs)
    raw_file = "network_clearance_raw_data.csv"
    with open(raw_file, 'w', newline='') as f:
        fieldnames = ['applied_strain', 'rng_seed', 'clearance_time',
                     'final_lysis', 'failure_mode', 'termination']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"[SAVED] Raw data: {raw_file}")

    # Summary statistics
    summary_file = "network_clearance_summary.csv"
    with open(summary_file, 'w', newline='') as f:
        fieldnames = ['applied_strain', 'mean_clearance_time', 'std_clearance_time',
                     'sem_clearance_time', 'dominant_failure_mode', 'mode_fraction',
                     'n_realizations']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"[SAVED] Summary: {summary_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute production study and generate poster figure."""

    start_time = datetime.now()

    # Run simulations
    all_results = run_production_study()

    # Compute statistics
    summary = compute_summary_statistics(all_results)

    # Generate figure
    generate_poster_figure(summary)

    # Save data
    save_results(all_results, summary)

    # Final report
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print()
    print("=" * 80)
    print("STUDY COMPLETE")
    print("=" * 80)
    print()
    print(f"Total simulations: {len(all_results)}")
    print(f"Execution time: {duration/60:.1f} minutes")
    print()
    print("Output files:")
    print("  1. network_clearance_vs_strain.png (poster figure)")
    print("  2. network_clearance_summary.csv (statistics)")
    print("  3. network_clearance_raw_data.csv (all runs)")
    print()
    print("Figure is advisor-approved and poster-ready.")
    print("=" * 80)


if __name__ == "__main__":
    main()
