"""
Benchmarks for single fiber simulation.

Run directly:
    python projects/single_fiber/benchmarks/benchmark_performance.py
"""

# Lazy imports to avoid circular dependency issues
__all__ = [
    "run_benchmark_suite",
    "benchmark_physics_steps",
    "create_hooke_config",
    "create_wlc_config",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name in __all__:
        from . import benchmark_performance
        return getattr(benchmark_performance, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
