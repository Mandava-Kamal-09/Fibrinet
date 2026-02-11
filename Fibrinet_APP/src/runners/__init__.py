"""
FibriNet Runners Package.

Provides headless simulation runners for batch execution and testing.

NO TKINTER IMPORTS ALLOWED IN THIS PACKAGE.
"""

from src.runners.research_sim_runner import ResearchSimRunner, SimulationResult, BatchResult

__all__ = ["ResearchSimRunner", "SimulationResult", "BatchResult"]
