## Phase 2G (v5.0) — Supply-limited stochastic plasmin seeding

### Goal
Replace the **continuous “binding everywhere” Langmuir Euler update** with **sparse, stochastic, supply-limited** plasmin binding in **spatial mode only**, while preserving determinism/replay safety and leaving legacy mode unchanged.

### Key changes
- **Global plasmin pool (spatial mode only)**
  - Added `P_total_quanta` and `P_free_quanta` to the adapter.
  - Initialized at `Start`: `P_free_quanta = P_total_quanta`.
  - Logged every batch for deterministic replay.

- **Replace binding update**
  - Disabled the spatial-mode Langmuir Euler “bind everywhere” block.
  - New spatial-mode binding:
    - `N_bind_events ~ Poisson(lambda_bind_total * dt_used)` capped by `P_free_quanta`
    - Weighted sampling over **all segments** by `available = max(0, S_i - B_i)`
    - Each applied event: `B_i += 1`, `P_free_quanta -= 1`
  - New spatial-mode unbinding:
    - `p_unbind = 1 - exp(-k_off(T_edge) * dt_used)`
    - `U_i ~ Binomial(B_i, p_unbind)` (deterministic RNG)
    - `B_i -= U_i`, `P_free_quanta += U_i`
  - **Conservation enforced** each batch:
    - `P_free_quanta + sum_i B_i == P_total_quanta` (exact integer equality)

- **dt_used consistency**
  - Spatial-mode `dt_used` is computed **once per batch** and used consistently for:
    - Poisson binding events
    - Binomial unbinding
    - Cleavage update (Phase 2B) and stability guard
  - Binding no longer contributes an Euler stability constraint; `dt_used` is constrained by **cleavage stability** only (as already defined in Phase 2B).

- **Checkpoint/replay persistence**
  - `export_network_snapshot()` now includes:
    - `P_total_quanta`, `P_free_quanta`
    - `spatial_plasmin_params`
    - per-edge `segments` (when present)
  - `load_checkpoint()` restores the above (when present).

### Logging additions (spatial mode only)
Added to per-batch experiment log:
- `P_total_quanta`
- `P_free_quanta`
- `bind_events_requested`
- `bind_events_applied`
- `total_unbound_this_batch`
- `total_bound_this_batch` (integer quanta; equals `sum(B_i)`)

### Files changed
- `Fibrinet_APP/src/views/tkinter_view/research_simulation_page.py`
  - Added plasmin pool fields on adapter
  - Replaced spatial-mode binding logic with stochastic seeding + conservation
  - Ensured `dt_used` is computed once and used consistently
  - Snapshot/checkpoint persistence extended for pool + segments + spatial params
  - Added Phase 2G log fields
- `Fibrinet_APP/test/test_spatial_plasmin_seeding.py` (new tests; updated to account for dt_used behavior)
- Updated existing tests to match Phase 2G behavior:
  - `Fibrinet_APP/test/test_spatial_plasmin_binding.py`
  - `Fibrinet_APP/test/test_spatial_plasmin_cleavage.py`
  - `Fibrinet_APP/test/test_binding_integration.py`
  - `Fibrinet_APP/test/test_spatial_plasmin_phase2f.py`

### Tests run (local)
- `python test/test_spatial_plasmin_seeding.py`
- `python test/test_spatial_plasmin_binding.py`
- `python test/test_spatial_plasmin_cleavage.py`
- `python test/test_binding_integration.py`
- `python test/test_spatial_plasmin_phase2f.py`
- `python test/test_spatial_plasmin_stiffness.py`

### Explicit scope confirmation
- **No edge removal**
- **No fracture criterion**
- **No percolation termination changes**
- **Legacy mode unchanged**


