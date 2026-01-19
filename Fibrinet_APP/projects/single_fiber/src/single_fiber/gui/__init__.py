"""
Single Fiber GUI - DearPyGui-based visualization.

Phase 3 GUI foundation providing:
    - Real-time chain visualization
    - Interactive node dragging
    - Status and control panels
    - Event-safe architecture (GUI-independent physics)
"""

from .app import SingleFiberApp
from .controller import ChainController
from .viewport import ChainViewport

__all__ = ["SingleFiberApp", "ChainController", "ChainViewport"]
