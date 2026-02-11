"""
FibriNet Event System.

Provides pub/sub event infrastructure for decoupling UI from simulation logic.
No Tkinter imports in this package.
"""

from src.events.events import (
    Event,
    RunStarted,
    RunCompleted,
    RunFailed,
    BatchStarted,
    BatchCompleted,
    StepCompleted,
    EdgeRuptured,
    ExportWritten,
)
from src.events.event_bus import EventBus

__all__ = [
    "Event",
    "EventBus",
    "RunStarted",
    "RunCompleted",
    "RunFailed",
    "BatchStarted",
    "BatchCompleted",
    "StepCompleted",
    "EdgeRuptured",
    "ExportWritten",
]
