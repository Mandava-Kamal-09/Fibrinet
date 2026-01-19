# FibriNet Research Simulation Tool

## Project Overview
FibriNet simulates 2D fibrin network mechanical degradation under tensile strain with spatial plasmin binding and force-dependent rupture. The codebase supports **two physics engines**: the legacy Hookean spring model and the new **Core V2** Worm-Like Chain (WLC) engine with stress-based Bell model rupture.

**Critical distinction**: The system operates in **spatial mode** (`FeatureFlags.USE_SPATIAL_PLASMIN=True`) or **legacy mode** (scalar degradation). All code changes must preserve backward compatibility and deterministic replay.

## Architecture

### Component Hierarchy (MVC Pattern)
```
FibriNet.py / cli_main.py (entry points)
  └─ SystemController (src/controllers/system_controller.py)
      ├─ NetworkManager (manages network state, relaxation)
      ├─ ViewManager (routes to GUI/CLI views)
      ├─ InputManager (loads Excel/CSV networks)
      ├─ ExportManager (exports results)
      └─ SystemState (global runtime flags)

Views:
  - TkinterView → ResearchSimulationPage (GUI, 8000+ lines)
  - CLIView (command-line interface)
```

### Physics Engines
1. **Core V2** (`src/core/fibrinet_core_v2.py` + `fibrinet_core_v2_adapter.py`)
   - WLC mechanics: `F(ε) = (k_B T / ξ) × [1/(4(1-ε)²) - 1/4 + ε]`
   - Stress-based rupture: `k(F, S) = k₀ × exp((F/max(S, 0.05)) × x_b/k_B T)`
   - L-BFGS-B energy minimization with analytical Jacobian
   - **Key benefit**: 100× faster than finite differences, enables avalanche dynamics

2. **Legacy Engine** (embedded in ResearchSimulationPage)
   - Hookean springs, scalar stiffness `S`, force-balance relaxation
   - Spatial plasmin mode: discrete protofibril segments, binding sites, local damage
   - Legacy mode: global scalar `S` degradation

### Spatial Plasmin System (Phases 0-2G)
**Only active when `FeatureFlags.USE_SPATIAL_PLASMIN=True`**

Core concepts:
- **Segments**: Each fiber discretized into `N_seg` segments with length `L_seg_target`
- **Binding sites**: `S_i = surface_area × (L_i / L_seg_target)` per segment
- **Supply-limited binding**: Global plasmin pool `P_total_quanta`, stochastic Poisson binding events weighted by available sites
- **Cleavage**: `dn/dt = -k_cat(T_edge) × B_i` (deterministic Euler)
- **Unbinding**: Binomial sampling `U_i ~ Binomial(B_i, p_unbind)`
- **Conservation**: `P_free_quanta + sum(B_i) == P_total_quanta` (exact integer equality enforced every batch)

**Critical batch order** (spatial mode):
1. Use cached post-relaxation forces from previous batch
2. Unbind plasmin (Binomial)
3. Bind plasmin (Poisson, supply-limited)
4. Update cleavage (Euler)
5. Update stiffness `S = min(n_i/N_pf)` across segments
6. Detect fractures (`min(n_i/N_pf) ≤ n_crit_fraction`)
7. Relax network (mechanical equilibrium)
8. Log observables

### Data Models
- **Phase1EdgeSnapshot** (frozen dataclass): Immutable edge state snapshot
  - Legacy: `S` (scalar stiffness), `M` (memory)
  - Spatial: `segments` (list of protofibril counts `n_i`), `plasmin_sites` (binding locations)
- **PlasminBindingSite** (frozen dataclass): `(edge_id, segment_index, x, y)`
- **NetworkSnapshot**: Complete network state for checkpointing/replay

## Development Workflows

### Running Simulations
**GUI**: `python FibriNet.py` (Tkinter interface)  
**CLI**: `python cli_main.py` (text-based)  
**Collapse Analysis**: `python analyze_collapse_cli.py <network.xlsx> --out-dir exports/ --max-steps 1000`

### Testing
Tests are organized by phase in `test/`:
```bash
# Spatial plasmin tests (run standalone, not pytest)
python test/test_spatial_plasmin_units.py
python test/test_spatial_plasmin_binding.py
python test/test_spatial_plasmin_cleavage.py
python test/test_spatial_plasmin_seeding.py

# Core V2 integration
python test_core_v2_integration.py
```

**Testing pattern**: Tests use `ResearchSimulationPage` directly, not pytest fixtures. They validate:
- Deterministic replay (RNG seeding)
- Conservation laws (plasmin pool, energy-force consistency)
- Physical invariants (stiffness monotonicity, segment length guards)

### Network File Format
**Excel/CSV with stacked tables**:
1. **nodes** table: `n_x, n_y, is_left_boundary, is_right_boundary`
2. **edges** table: `n_from, n_to, thickness, rest_length`
3. **meta_data** table (optional): Key-value pairs (e.g., `K_crit=0.7`, `coord_to_m=1e-6`)

**Unit conversion**: Use `coord_to_m` (default 1e-6) and `thickness_to_m` (default 1e-9) in metadata for SI conversion.

**Parser**: `_parse_delimited_tables_from_xlsx()` and `_parse_delimited_tables_from_csv()` in [research_simulation_page.py](src/views/tkinter_view/research_simulation_page.py#L28-L145)

## Critical Conventions

### Feature Flag System
**Never mutate feature flags mid-simulation**. Flags are set at startup in [feature_flags.py](src/config/feature_flags.py):
```python
from src.config.feature_flags import FeatureFlags
FeatureFlags.USE_SPATIAL_PLASMIN = True  # Set before creating adapters
```

**Determinism**: Changing flags invalidates existing experiment logs and checkpoints.

### Immutability & Determinism
- All edge snapshots are **frozen dataclasses** (no mutations)
- Use `.replace()` to create modified copies
- RNG seeding: Every stochastic operation uses `np.random.default_rng(seed)` with deterministic seeds
- **No direct NumPy random calls**: Always inject RNG via function parameters

### Logging
Experiment logs (`iteration_log.csv`) must include:
- All parameters for reproducibility (λ₀, Δt, strain, N_pf, P_total_quanta)
- Batch observables (time, lysis%, active fibers, forces, energy)
- Spatial mode: `P_free_quanta, bind_events_requested, bind_events_applied, total_unbound_this_batch`

**Access logs**: `exports/<experiment_name>/iteration_log.csv`

### Code Organization Rules
1. **Pure functions**: PlasminManager, EdgeEvolutionEngine are stateless
2. **No side effects**: Controllers orchestrate; managers compute
3. **Type hints**: All public APIs must have type annotations
4. **Docstrings**: Include mathematical formulas, invariants, and integration points

## Integration Points

### Core V2 ↔ GUI Adapter
[fibrinet_core_v2_adapter.py](src/core/fibrinet_core_v2_adapter.py) bridges Core V2 with legacy GUI:
- `create_adapter_from_excel()`: Load network and configure
- `advance_one_batch()`: Single timestep execution
- `get_edges()`, `get_node_positions()`, `get_forces()`: Export for visualization
- **Unit conversion**: Adapter handles abstract units ↔ SI internally

### NetworkManager ↔ State Management
[NetworkManager](src/managers/network/network_manager.py) maintains:
- `network`: Current network instance (BaseNetwork subclass)
- `state_manager`: NetworkStateManager for undo/replay
- `relax_network()`: Trigger mechanical equilibrium

**State transitions**: Use `SystemState` flags (`network_loaded`, `spring_stiffness_constant`) to validate operations.

## Scientific Guardrails

### Numerical Stability Guards
- `S_MIN_BELL = 0.05`: Prevent stress blow-up in Bell model
- `MAX_STRAIN = 0.99`: Prevent WLC singularity at ε=1
- `MAX_BELL_EXPONENT = 100.0`: Prevent exp overflow
- `N_SEG_MAX = 1000`: Prevent segment explosion (unrealistic discretization)

### Physical Invariants to Validate
1. **Energy-force consistency**: `|F - dU/dx|/F < 1e-6`
2. **Stiffness monotonicity**: `S` never increases (damage irreversible)
3. **Conservation**: Plasmin pool + bound = constant
4. **Percolation termination**: Network lysis when left-right path lost (spatial mode)

## Common Pitfalls

1. **Don't mix feature flag modes**: Spatial and legacy have different data models (segments vs. scalar S)
2. **Don't mutate edge snapshots**: Use `.replace()` or create new instances
3. **Don't use raw `random.random()`**: Inject `np.random.Generator` instances for determinism
4. **Don't modify `dt_used` mid-batch**: Compute once, use consistently for binding/unbinding/cleavage
5. **Don't skip checkpoint validation**: Always test `load_checkpoint()` → `export_network_snapshot()` roundtrip

## Key Files Reference
- **Physics**: [fibrinet_core_v2.py](src/core/fibrinet_core_v2.py), [research_simulation_page.py](src/views/tkinter_view/research_simulation_page.py) (legacy/spatial)
- **Managers**: [plasmin_manager.py](src/managers/plasmin_manager.py), [edge_evolution_engine.py](src/managers/edge_evolution_engine.py)
- **Config**: [feature_flags.py](src/config/feature_flags.py)
- **Documentation**: [FIBRINET_FUNCTIONAL_DOCUMENTATION.md](FIBRINET_FUNCTIONAL_DOCUMENTATION.md), [CORE_V2_INTEGRATION_STATUS.md](CORE_V2_INTEGRATION_STATUS.md), [USER_GUIDE_CORE_V2.md](USER_GUIDE_CORE_V2.md)
- **Implementation history**: `readme/IMPLEMENTATION_SUMMARY_v5.0_PHASE*.md` (detailed phase logs)

## When Debugging
1. Check feature flags: Is spatial mode intended?
2. Validate RNG seeding: Are results deterministic across runs?
3. Inspect experiment log: Does `P_free_quanta + sum(B_i) == P_total_quanta` hold?
4. Check segment counts: Is `N_seg` reasonable (<1000)?
5. Review phase documentation: Each phase has test files and implementation summaries in `readme/`
