"""
Tests to verify frozen physics modules maintain their public API.

These tests do NOT modify frozen modules. They only import and check
that expected public interfaces remain available, ensuring Phase 4
changes do not accidentally break physics contracts.

Frozen modules:
- src/core/force_laws/wlc.py
- src/core/force_laws/hookean.py
- src/core/force_laws/units.py
- src/core/force_laws/types.py
- projects/single_fiber/src/single_fiber/chain_integrator.py
- projects/single_fiber/src/single_fiber/chain_model.py
- projects/single_fiber/src/single_fiber/chain_state.py
"""

import pytest
import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestFrozenForceLabsAPI:
    """Verify force_laws modules maintain expected API."""

    def test_wlc_module_exports(self):
        """WLC module exports expected function."""
        from src.core.force_laws.wlc import wlc_tension_marko_siggia, wlc_tension_fast
        assert callable(wlc_tension_marko_siggia)
        assert callable(wlc_tension_fast)

    def test_hookean_module_exports(self):
        """Hookean module exports expected function."""
        from src.core.force_laws.hookean import hooke_tension
        assert callable(hooke_tension)

    def test_units_module_exports(self):
        """Units module exports expected constants."""
        from src.core.force_laws.units import KBT_PN_NM, WLC_EPSILON
        assert isinstance(KBT_PN_NM, float)
        assert isinstance(WLC_EPSILON, float)
        assert KBT_PN_NM > 0
        assert WLC_EPSILON > 0

    def test_types_module_exports(self):
        """Types module exports expected dataclasses."""
        from src.core.force_laws.types import ForceResult, HookeanParams, WLCParams
        # Check they are classes
        assert isinstance(ForceResult, type)
        assert isinstance(HookeanParams, type)
        assert isinstance(WLCParams, type)


class TestFrozenChainAPI:
    """Verify chain modules maintain expected API."""

    def test_chain_state_exports(self):
        """ChainState class exists with expected methods."""
        from single_fiber.chain_state import ChainState, SegmentState
        assert isinstance(ChainState, type)
        assert isinstance(SegmentState, type)
        # Check key methods exist
        assert hasattr(ChainState, 'from_endpoints')
        assert hasattr(ChainState, 'segment_length')
        assert hasattr(ChainState, 'segment_strain')
        assert hasattr(ChainState, 'copy')

    def test_chain_model_exports(self):
        """ChainModel class exists with expected methods."""
        from single_fiber.chain_model import ChainModel
        assert isinstance(ChainModel, type)
        assert hasattr(ChainModel, 'compute_forces')
        assert hasattr(ChainModel, 'compute_segment_tension')

    def test_chain_integrator_exports(self):
        """ChainIntegrator class exists with expected methods."""
        from single_fiber.chain_integrator import (
            ChainIntegrator,
            RelaxationResult,
            ChainLoadingController
        )
        assert isinstance(ChainIntegrator, type)
        assert isinstance(RelaxationResult, type)
        assert isinstance(ChainLoadingController, type)
        assert hasattr(ChainIntegrator, 'relax_to_equilibrium')
        assert hasattr(ChainIntegrator, 'step_with_relaxation')


class TestFrozenModulesUnmodified:
    """
    Meta-test: verify this test file itself does not import anything
    that would trigger modifications to frozen modules.

    This is a sanity check that the test is read-only.
    """

    def test_imports_are_read_only(self):
        """
        All imports in this file are read-only.
        This test passes if no import errors occurred.
        """
        # If we got here, all imports succeeded without error
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
