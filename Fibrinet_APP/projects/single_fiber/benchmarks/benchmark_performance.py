"""
Performance benchmarks for single fiber simulation.

Measures:
- Physics steps per second for various configurations
- Scalability with segment count

Usage:
    python -m projects.single_fiber.benchmarks.benchmark_performance
"""

import sys
import time
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from projects.single_fiber.src.single_fiber.config import (
    SimulationConfig, ModelConfig, HookeConfig, WLCConfig,
    GeometryConfig, DynamicsConfig, LoadingConfig
)
from projects.single_fiber.src.single_fiber.chain_state import ChainState
from projects.single_fiber.src.single_fiber.chain_model import ChainModel
from projects.single_fiber.src.single_fiber.chain_integrator import (
    ChainIntegrator, ChainLoadingController
)

import numpy as np


def create_hooke_config(n_segments: int, L0_total: float = 100.0) -> SimulationConfig:
    """Create Hookean spring config for benchmarking."""
    return SimulationConfig(
        model=ModelConfig(
            law="hooke",
            hooke=HookeConfig(k_pN_per_nm=0.1, L0_nm=L0_total / n_segments)
        ),
        geometry=GeometryConfig(
            x1_nm=[0.0, 0.0, 0.0],
            x2_nm=[L0_total, 0.0, 0.0]
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


def create_wlc_config(n_segments: int, L0_total: float = 100.0) -> SimulationConfig:
    """Create WLC config for benchmarking."""
    return SimulationConfig(
        model=ModelConfig(
            law="wlc",
            wlc=WLCConfig(Lp_nm=50.0, Lc_nm=L0_total * 1.5 / n_segments, kBT_pN_nm=4.1)
        ),
        geometry=GeometryConfig(
            x1_nm=[0.0, 0.0, 0.0],
            x2_nm=[L0_total, 0.0, 0.0]
        ),
        dynamics=DynamicsConfig(
            dt_us=0.1,
            gamma_pN_us_per_nm=1.0
        ),
        loading=LoadingConfig(
            v_nm_per_us=0.3,
            t_end_us=100.0
        )
    )


def benchmark_physics_steps(
    config: SimulationConfig,
    n_segments: int,
    n_steps: int = 1000,
    warmup_steps: int = 100
) -> dict:
    """
    Benchmark physics step rate.

    Args:
        config: Simulation configuration
        n_segments: Number of chain segments
        n_steps: Number of steps to benchmark
        warmup_steps: Warmup steps (not counted)

    Returns:
        Dict with timing results
    """
    # Initialize state
    x1 = np.array(config.geometry.x1_nm)
    x2 = np.array(config.geometry.x2_nm)
    state = ChainState.from_endpoints(x1, x2, n_segments)

    model = ChainModel(config.model)
    integrator = ChainIntegrator(config.dynamics)

    end_node_pos = state.nodes_nm[-1].copy()
    loading = ChainLoadingController(config.loading, end_node_pos)

    dt = config.dynamics.dt_us
    t = 0.0

    # Warmup
    for _ in range(warmup_steps):
        t += dt
        target = loading.target_position(t)
        state, _, _ = integrator.step_with_relaxation(
            state, model, target, t, fixed_boundary_node=0
        )

    # Benchmark
    start_time = time.perf_counter()

    for _ in range(n_steps):
        t += dt
        target = loading.target_position(t)
        state, _, _ = integrator.step_with_relaxation(
            state, model, target, t, fixed_boundary_node=0
        )

    end_time = time.perf_counter()

    elapsed = end_time - start_time
    steps_per_sec = n_steps / elapsed
    us_per_step = (elapsed / n_steps) * 1e6

    return {
        "n_segments": n_segments,
        "n_steps": n_steps,
        "elapsed_sec": elapsed,
        "steps_per_sec": steps_per_sec,
        "microseconds_per_step": us_per_step,
    }


def run_benchmark_suite() -> None:
    """Run full benchmark suite and print results."""
    print("=" * 70)
    print("SINGLE FIBER SIMULATION PERFORMANCE BENCHMARK")
    print("=" * 70)
    print()

    segment_counts = [1, 5, 10, 20, 50]
    n_steps = 1000

    # Hooke benchmarks
    print("HOOKEAN SPRING MODEL")
    print("-" * 70)
    print(f"{'Segments':>10} {'Steps/sec':>15} {'us/step':>15} {'Status':>15}")
    print("-" * 70)

    hooke_results = []
    for n_seg in segment_counts:
        config = create_hooke_config(n_seg)
        result = benchmark_physics_steps(config, n_seg, n_steps)
        hooke_results.append(result)

        status = "Fast" if result["steps_per_sec"] > 1000 else "Acceptable"
        print(f"{n_seg:>10} {result['steps_per_sec']:>15.1f} {result['microseconds_per_step']:>15.1f} {status:>15}")

    print()

    # WLC benchmarks
    print("WLC MODEL")
    print("-" * 70)
    print(f"{'Segments':>10} {'Steps/sec':>15} {'us/step':>15} {'Status':>15}")
    print("-" * 70)

    wlc_results = []
    for n_seg in segment_counts:
        config = create_wlc_config(n_seg)
        result = benchmark_physics_steps(config, n_seg, n_steps)
        wlc_results.append(result)

        status = "Fast" if result["steps_per_sec"] > 1000 else "Acceptable"
        print(f"{n_seg:>10} {result['steps_per_sec']:>15.1f} {result['microseconds_per_step']:>15.1f} {status:>15}")

    print()
    print("=" * 70)

    # Summary
    avg_hooke = sum(r["steps_per_sec"] for r in hooke_results) / len(hooke_results)
    avg_wlc = sum(r["steps_per_sec"] for r in wlc_results) / len(wlc_results)

    print("SUMMARY")
    print("-" * 70)
    print(f"Average Hooke steps/sec: {avg_hooke:.1f}")
    print(f"Average WLC steps/sec:   {avg_wlc:.1f}")
    print()

    # Interactive target
    target_fps = 60
    steps_per_frame = 5  # Default animation setting
    min_steps_per_sec = target_fps * steps_per_frame

    print(f"Interactive requirement: {min_steps_per_sec} steps/sec (60 FPS, 5 steps/frame)")
    if avg_hooke >= min_steps_per_sec and avg_wlc >= min_steps_per_sec:
        print("Status: PASS - Sufficient for interactive use")
    else:
        print("Status: WARNING - May be slow for interactive use")

    print("=" * 70)

    return hooke_results, wlc_results


if __name__ == "__main__":
    run_benchmark_suite()
