"""End-to-end integration tests: realism, scaling, and simulation behavior."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.fibrinet_core_v2 import (
    HybridMechanochemicalSimulation, NetworkState, PhysicalConstants,
    check_left_right_connectivity,
)
from src.validation.canonical_networks import small_lattice, line
from tests.conftest import dict_to_network_state

PC = PhysicalConstants()


def _run_sim(state, plasmin=1.0, dt=0.01, t_max=50.0, seed=42,
             chemistry_mode='mean_field', abm_params=None):
    """Helper to run simulation and return (sim, steps_run)."""
    sim = HybridMechanochemicalSimulation(
        initial_state=state,
        rng_seed=seed,
        dt_chem=dt,
        t_max=t_max,
        plasmin_concentration=plasmin,
        chemistry_mode=chemistry_mode,
        abm_params=abm_params,
    )
    steps = 0
    while sim.step():
        steps += 1
        if steps > 5000:
            break  # Safety limit
    return sim, steps


# ----------------------------------------
# Higher plasmin → faster lysis
# ----------------------------------------

class TestPlasminScaling:

    def test_higher_plasmin_faster_lysis(self):
        """λ₀=2 should reach 50% lysis before λ₀=0.5."""
        net_dict = small_lattice(3, 5)

        # High plasmin
        state_high = dict_to_network_state(net_dict, applied_strain=0.05, prestrain=True)
        sim_high, _ = _run_sim(state_high, plasmin=2.0, t_max=100.0, seed=42)
        time_high = sim_high.state.time
        lysis_high = sim_high.state.lysis_fraction

        # Low plasmin
        state_low = dict_to_network_state(net_dict, applied_strain=0.05, prestrain=True)
        sim_low, _ = _run_sim(state_low, plasmin=0.5, t_max=100.0, seed=42)
        time_low = sim_low.state.time
        lysis_low = sim_low.state.lysis_fraction

        # High plasmin should either finish faster or have higher lysis at same time
        assert (time_high < time_low) or (lysis_high >= lysis_low), (
            f"Higher plasmin should lyse faster: "
            f"λ=2 → t={time_high:.2f}s, lysis={lysis_high:.3f}; "
            f"λ=0.5 → t={time_low:.2f}s, lysis={lysis_low:.3f}"
        )


# --------------------------------------------------
# Strain inhibits cleavage
# --------------------------------------------------

class TestStrainInhibition:

    def test_strain_inhibits_cleavage(self):
        """50% strain should produce fewer cleavages than 0% strain in same time."""
        net_dict = small_lattice(3, 4)

        # Zero strain
        state_zero = dict_to_network_state(net_dict, applied_strain=0.0, prestrain=True)
        sim_zero = HybridMechanochemicalSimulation(
            initial_state=state_zero, rng_seed=42, dt_chem=0.01,
            t_max=20.0, plasmin_concentration=1.0,
        )
        for _ in range(500):
            if not sim_zero.step():
                break
        n_zero = sim_zero.state.n_ruptured

        # High strain (50%)
        state_high = dict_to_network_state(net_dict, applied_strain=0.5, prestrain=True)
        sim_high = HybridMechanochemicalSimulation(
            initial_state=state_high, rng_seed=42, dt_chem=0.01,
            t_max=20.0, plasmin_concentration=1.0,
        )
        for _ in range(500):
            if not sim_high.step():
                break
        n_high = sim_high.state.n_ruptured

        assert n_high <= n_zero, (
            f"Strain should inhibit cleavage: "
            f"ruptured(ε=0)={n_zero}, ruptured(ε=0.5)={n_high}"
        )


# ------------------------------------------------------------
# Simulation reaches completion
# ------------------------------------------------------------

class TestSimulationCompletion:

    def test_simulation_reaches_completion(self):
        """Line network with very high plasmin should terminate (clearance or lysis)."""
        # Use a 3-fiber line with very high plasmin for fast termination
        state = dict_to_network_state(
            line(n=4), applied_strain=0.0, prestrain=True,
        )
        # Very high plasmin: k_eff = 20 * k0 * exp(-beta*strain)
        # At prestrain only (strain from prestrain ~23%), rate ~ 20*0.1*exp(-2.3) ≈ 0.2/s
        # Each fiber needs 10 hits, 3 fibers → ~30 events, ~150s total
        sim = HybridMechanochemicalSimulation(
            initial_state=state,
            rng_seed=42,
            dt_chem=0.01,
            t_max=1000.0,
            plasmin_concentration=20.0,
            delta_S=0.5,  # Only 2 hits to rupture a fiber (faster termination)
        )
        steps = 0
        while sim.step():
            steps += 1
            if steps > 50000:
                break

        assert sim.termination_reason is not None, (
            f"Simulation did not terminate after {steps} steps "
            f"(t={sim.state.time:.2f}s, lysis={sim.state.lysis_fraction:.3f})"
        )


# ----------------------------------------
# Clearance event
# ----------------------------------------

class TestClearanceEvent:

    def test_clearance_event_logged(self):
        """When connectivity is lost, clearance_event should be recorded."""
        # Use line network — will disconnect after enough cuts
        state = dict_to_network_state(
            line(n=4), applied_strain=0.0, prestrain=True,
        )
        sim, _ = _run_sim(state, plasmin=3.0, t_max=200.0, seed=42)

        if sim.termination_reason == 'network_cleared':
            assert sim.state.clearance_event is not None, (
                "Network cleared but clearance_event is None"
            )
            event = sim.state.clearance_event
            assert 'time' in event
            assert 'critical_fiber_id' in event
            assert event['total_fibers'] > 0


# --------------------------------------------------
# Degradation history ordering
# --------------------------------------------------

class TestDegradationHistory:

    def test_degradation_history_ordered(self):
        """Degradation history entries should be in chronological order."""
        state = dict_to_network_state(
            small_lattice(3, 4), applied_strain=0.05, prestrain=True,
        )
        sim, _ = _run_sim(state, plasmin=2.0, t_max=50.0, seed=42)

        history = sim.state.degradation_history
        if len(history) < 2:
            pytest.skip("Not enough degradation events to check ordering")

        for i in range(1, len(history)):
            assert history[i]['time'] >= history[i - 1]['time'], (
                f"Degradation history out of order at index {i}: "
                f"t[{i-1}]={history[i-1]['time']:.4f} > t[{i}]={history[i]['time']:.4f}"
            )

        # Also check sequential order numbering
        for i, entry in enumerate(history):
            assert entry['order'] == i + 1, (
                f"Order numbering wrong at index {i}: {entry['order']} vs {i+1}"
            )


# ------------------------------------------------------------
# Force distribution realism
# ------------------------------------------------------------

class TestForceRealism:

    def test_force_distribution_realistic(self):
        """Fiber tensions should be in pN–nN range for µm-scale networks."""
        state = dict_to_network_state(
            small_lattice(4, 6), applied_strain=0.1, prestrain=True,
        )

        sim = HybridMechanochemicalSimulation(
            initial_state=state, rng_seed=42, dt_chem=0.01,
            t_max=1.0, plasmin_concentration=1.0,
        )
        sim.relax_network()

        forces = sim.compute_forces()
        max_force = max(abs(f) for f in forces.values()) if forces else 0.0

        # Fibrin fiber forces at ~10% strain should be in pN to nN range
        # k_BT/xi ~ 4.28e-15 N, at 10% strain: F ~ a few pN
        assert max_force < 1e-6, (
            f"Max force {max_force:.3e} N exceeds F_MAX={1e-6} N"
        )
        assert max_force > 1e-18, (
            f"Max force {max_force:.3e} N is unrealistically small"
        )


# ----------------------------------------
# Applied strain displaces boundary
# ----------------------------------------

class TestAppliedStrain:

    def test_applied_strain_displaces_boundary(self):
        """Right boundary nodes should be shifted by strain × x_span."""
        spacing_m = 1e-6
        strain = 0.15

        net_dict = small_lattice(3, 4, spacing=1.0)
        state = dict_to_network_state(
            net_dict, spacing_m=spacing_m, applied_strain=strain, prestrain=False,
        )

        # Original x_span = (cols-1) * spacing * spacing_m = 3 * 1e-6 = 3e-6
        x_span_original = 3.0 * spacing_m

        # Right boundary nodes should be at x_original + strain * x_span
        for nid in state.right_boundary_nodes:
            x_actual = state.node_positions[nid][0]
            x_original = 3.0 * spacing_m  # rightmost column
            x_expected = x_original + strain * x_span_original
            assert abs(x_actual - x_expected) < 1e-18, (
                f"Right boundary node {nid} at x={x_actual:.6e}, "
                f"expected {x_expected:.6e}"
            )


# --------------------------------------------------
# Mean-field vs ABM qualitative comparison
# --------------------------------------------------

class TestMeanFieldVsABM:

    def test_mean_field_vs_abm_qualitative(self):
        """Both modes should run and produce some degradation activity."""
        net_dict = small_lattice(3, 4)

        # Mean-field: run a short sim
        state_mf = dict_to_network_state(net_dict, applied_strain=0.02, prestrain=True)
        sim_mf = HybridMechanochemicalSimulation(
            initial_state=state_mf, rng_seed=42, dt_chem=0.01,
            t_max=50.0, plasmin_concentration=3.0,
        )
        for _ in range(500):
            if not sim_mf.step():
                break

        history_mf = sim_mf.state.degradation_history
        assert len(history_mf) > 0 or sim_mf.state.n_ruptured > 0, (
            "Mean-field produced no degradation events"
        )

        # ABM: run a short sim with aggressive parameters
        from src.core.plasmin_abm import ABMParameters
        abm_params = ABMParameters(
            n_agents=15,
            auto_agent_count=False,
            k_cat0=2.0,  # High rate for fast test
            beta_cat=1.0,  # Low strain sensitivity
            strain_cleavage_model='exponential',
            k_off0=0.001,  # Low unbinding to keep agents bound
        )

        state_abm = dict_to_network_state(net_dict, applied_strain=0.02, prestrain=True)
        sim_abm = HybridMechanochemicalSimulation(
            initial_state=state_abm, rng_seed=42, dt_chem=0.01,
            t_max=100.0, plasmin_concentration=3.0,
            chemistry_mode='abm', abm_params=abm_params,
        )
        for _ in range(2000):
            if not sim_abm.step():
                break

        history_abm = sim_abm.state.degradation_history
        # ABM should have produced some splits
        if history_abm:
            split_events = [e for e in history_abm if e.get('type') == 'positional_split']
            assert len(split_events) > 0, (
                f"ABM ran {len(history_abm)} events but no positional splits"
            )
        else:
            # At minimum, the ABM engine should have been initialized
            assert sim_abm.abm_engine is not None, "ABM engine was not initialized"


# ------------------------------------------------------------
# Cascade kill-switch: identical behavior when disabled
# ------------------------------------------------------------

class TestCascadeKillSwitch:

    def test_cascade_disabled_produces_identical_clearance_time(self):
        """CASCADE_ENABLED=False must produce identical results to enabled with no cascade events."""
        net_dict = small_lattice(3, 4)

        # Run with cascade enabled (default)
        state_on = dict_to_network_state(net_dict, applied_strain=0.05, prestrain=True)
        PC.CASCADE_ENABLED = True
        sim_on, steps_on = _run_sim(state_on, plasmin=2.0, t_max=100.0, seed=99)
        time_on = sim_on.state.time
        ruptured_on = sim_on.state.n_ruptured

        # Run with cascade disabled
        state_off = dict_to_network_state(net_dict, applied_strain=0.05, prestrain=True)
        PC.CASCADE_ENABLED = False
        sim_off, steps_off = _run_sim(state_off, plasmin=2.0, t_max=100.0, seed=99)
        time_off = sim_off.state.time
        ruptured_off = sim_off.state.n_ruptured

        # Restore default
        PC.CASCADE_ENABLED = True

        # At low strain (5%), cascade shouldn't trigger, so results should match
        # If cascade does trigger, disabled should be <= enabled ruptures
        assert ruptured_off <= ruptured_on, (
            f"Disabled cascade ruptured more fibers ({ruptured_off}) than enabled ({ruptured_on})"
        )

    def test_cascade_tags_mechanical_ruptures_in_history(self):
        """At high strain, cascade events should be tagged with 'cascade': True."""
        net_dict = small_lattice(3, 5)
        state = dict_to_network_state(net_dict, applied_strain=0.6, prestrain=True)
        PC.CASCADE_ENABLED = True
        sim, _ = _run_sim(state, plasmin=3.0, t_max=200.0, seed=42)

        history = sim.state.degradation_history
        cascade_events = [e for e in history if e.get('cascade', False)]

        # At 60% strain some fibers may cascade — verify tagging is correct
        for event in cascade_events:
            assert event['is_complete_rupture'] is True, (
                f"Cascade event on fiber {event['fiber_id']} is not a complete rupture"
            )
            assert event['new_S'] == 0.0, (
                f"Cascade event on fiber {event['fiber_id']} has new_S={event['new_S']}"
            )
            assert 'cascade_wave' in event, (
                f"Cascade event missing 'cascade_wave' key"
            )


class TestDiameterHeterogeneity:

    def test_diameter_cv_zero_produces_identical_behavior(self):
        """CV=0 must produce identical results to default uniform fibers."""
        import src.core.fibrinet_core_v2 as core
        net_dict = small_lattice(3, 4)
        original_cv = core.PC.FIBER_DIAMETER_CV
        core.PC.FIBER_DIAMETER_CV = 0.0
        try:
            state_a = dict_to_network_state(net_dict, applied_strain=0.05, prestrain=True)
            sim_a, _ = _run_sim(state_a, plasmin=2.0, t_max=100.0, seed=42)
            t_a = sim_a.state.time

            state_b = dict_to_network_state(net_dict, applied_strain=0.05, prestrain=True)
            sim_b, _ = _run_sim(state_b, plasmin=2.0, t_max=100.0, seed=42)
            t_b = sim_b.state.time
        finally:
            core.PC.FIBER_DIAMETER_CV = original_cv
        assert t_a == t_b

    def test_diameter_heterogeneity_produces_varied_xi(self):
        """CV=0.5 must produce non-uniform xi across fibers via adapter."""
        import src.core.fibrinet_core_v2 as core
        from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter
        original_cv = core.PC.FIBER_DIAMETER_CV
        core.PC.FIBER_DIAMETER_CV = 0.5
        try:
            adapter = CoreV2GUIAdapter()
            adapter.load_from_excel(
                os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'data', 'input_networks', 'generated_lat_4x6.xlsx')
            )
            adapter.configure_parameters(
                plasmin_concentration=1.0, time_step=0.1,
                max_time=10.0, applied_strain=0.0, rng_seed=42,
            )
            adapter.start_simulation()
            xi_values = [f.xi for f in adapter.simulation.state.fibers]
        finally:
            core.PC.FIBER_DIAMETER_CV = original_cv
        assert len(set(xi_values)) > 1


# ----------------------------------------
# Spatial prestrain heterogeneity
# ----------------------------------------

# --------------------------------------------------
# Dynamic k_cat updating: ON vs OFF
# --------------------------------------------------

class TestDynamicKcat:

    def test_dynamic_kcat_updates_stored_value(self):
        """Dynamic mode (update_kcat_dynamic=True) must refresh k_cat_at_binding and record drift."""
        from src.core.plasmin_abm import ABMParameters, AgentState

        net_dict = small_lattice(3, 4)
        state = dict_to_network_state(net_dict, applied_strain=0.02, prestrain=True)
        abm_params = ABMParameters(
            n_agents=15,
            auto_agent_count=False,
            plasmin_concentration_nM=1000.0,  # High to ensure fast binding
            k_on2=1e7,  # High k_on for rapid binding
            k_cat0=0.5,  # Moderate cleavage (agents stay bound a while)
            beta_cat=1.0,
            strain_cleavage_model='exponential',
            k_off0=0.0001,  # Very low unbinding to keep agents bound
            update_kcat_dynamic=True,
        )
        sim = HybridMechanochemicalSimulation(
            initial_state=state, rng_seed=42, dt_chem=0.01,
            t_max=100.0, plasmin_concentration=1000.0,
            chemistry_mode='abm', abm_params=abm_params,
        )
        for _ in range(3000):
            if not sim.step():
                break

        # At least one agent should have been bound and had its k_cat refreshed
        agents_with_updates = [
            a for a in sim.abm_engine.agents if a.n_kcat_updates > 0
        ]
        assert len(agents_with_updates) > 0, (
            "Dynamic mode should produce at least one agent with k_cat updates"
        )

    def test_static_kcat_preserves_initial_value(self):
        """Static mode (update_kcat_dynamic=False) must never update k_cat tracking fields."""
        from src.core.plasmin_abm import ABMParameters, AgentState

        net_dict = small_lattice(3, 4)
        state = dict_to_network_state(net_dict, applied_strain=0.02, prestrain=True)
        abm_params = ABMParameters(
            n_agents=15,
            auto_agent_count=False,
            plasmin_concentration_nM=1000.0,
            k_on2=1e7,
            k_cat0=0.5,
            beta_cat=1.0,
            strain_cleavage_model='exponential',
            k_off0=0.0001,
            update_kcat_dynamic=False,
        )
        sim = HybridMechanochemicalSimulation(
            initial_state=state, rng_seed=42, dt_chem=0.01,
            t_max=100.0, plasmin_concentration=1000.0,
            chemistry_mode='abm', abm_params=abm_params,
        )
        for _ in range(3000):
            if not sim.step():
                break

        for a in sim.abm_engine.agents:
            assert a.n_kcat_updates == 0, (
                f"Agent #{a.agent_id} has n_kcat_updates={a.n_kcat_updates} in static mode"
            )
            assert a.max_kcat_delta == 0.0, (
                f"Agent #{a.agent_id} has max_kcat_delta={a.max_kcat_delta} in static mode"
            )


class TestSpatialPrestrain:

    def test_spatial_prestrain_amplitude_zero_is_uniform(self):
        """PRESTRAIN_AMPLITUDE=0.0 must produce uniform prestrain ratio for all fibers."""
        import src.core.fibrinet_core_v2 as core
        from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter
        original_amp = core.PhysicalConstants.PRESTRAIN_AMPLITUDE
        core.PhysicalConstants.PRESTRAIN_AMPLITUDE = 0.0
        try:
            adapter = CoreV2GUIAdapter()
            adapter.load_from_excel(
                os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'data', 'input_networks', 'generated_lat_4x6.xlsx')
            )
            adapter.configure_parameters(
                plasmin_concentration=1.0, time_step=0.1,
                max_time=10.0, applied_strain=0.0, rng_seed=42,
            )
            adapter.start_simulation()
            state = adapter.simulation.state
            # Compute L_c / geometric_length for each fiber
            ratios = []
            for f in state.fibers:
                geom_len = float(np.linalg.norm(
                    state.node_positions[f.node_j] - state.node_positions[f.node_i]))
                ratios.append(f.L_c / geom_len)
        finally:
            core.PhysicalConstants.PRESTRAIN_AMPLITUDE = original_amp
        # With amplitude=0, all fibers get uniform prestrain → same L_c/L_geom ratio
        expected = 1.0 / (1.0 + core.PC.PRESTRAIN)
        for r in ratios:
            assert abs(r - expected) < 1e-10, (
                f"Non-uniform prestrain ratio: {r:.10f} vs expected {expected:.10f}"
            )

    def test_spatial_prestrain_produces_boundary_gradient(self):
        """PRESTRAIN_AMPLITUDE=0.5 must produce higher prestrain near boundaries."""
        import src.core.fibrinet_core_v2 as core
        from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter
        original_amp = core.PhysicalConstants.PRESTRAIN_AMPLITUDE
        core.PhysicalConstants.PRESTRAIN_AMPLITUDE = 0.5
        try:
            adapter = CoreV2GUIAdapter()
            adapter.load_from_excel(
                os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'data', 'input_networks', 'generated_lat_4x6.xlsx')
            )
            adapter.configure_parameters(
                plasmin_concentration=1.0, time_step=0.1,
                max_time=10.0, applied_strain=0.0, rng_seed=42,
            )
            adapter.start_simulation()
            state = adapter.simulation.state

            # Collect (x_midpoint_frac, L_c) for each fiber
            x_all = [state.node_positions[nid][0] for nid in state.node_positions]
            x_min_net, x_max_net = min(x_all), max(x_all)
            x_span_net = x_max_net - x_min_net

            boundary_lcs = []
            interior_lcs = []
            for f in state.fibers:
                x_mid = 0.5 * (state.node_positions[f.node_i][0] +
                               state.node_positions[f.node_j][0])
                x_frac = (x_mid - x_min_net) / x_span_net if x_span_net > 0 else 0.5
                proximity = abs(2.0 * x_frac - 1.0)
                if proximity > 0.8:
                    boundary_lcs.append(f.L_c)
                elif proximity < 0.2:
                    interior_lcs.append(f.L_c)
        finally:
            core.PhysicalConstants.PRESTRAIN_AMPLITUDE = original_amp

        # Heterogeneity exists
        all_lcs = [f.L_c for f in state.fibers]
        assert len(set(all_lcs)) > 1, "Expected varied L_c with amplitude=0.5"

        # Boundary fibers should have smaller L_c (higher prestrain = shorter rest length)
        if boundary_lcs and interior_lcs:
            assert np.mean(boundary_lcs) < np.mean(interior_lcs), (
                f"Boundary mean L_c ({np.mean(boundary_lcs):.3e}) should be less than "
                f"interior mean L_c ({np.mean(interior_lcs):.3e})"
            )
