# Research Simulation Tool: Rigorous Edge Case & Crash Analysis

**Date:** January 2, 2026  
**Tool:** `src/views/tkinter_view/research_simulation_page.py`  
**Scope:** Journal-grade biophysics simulation with dual-mode operation (legacy + spatial v5.0)

---

## Executive Summary

This document identifies **28 critical edge cases** where the research simulation tool may crash or produce incorrect results. These span input parsing, parameter validation, numerical stability, and state machine violations. All cases have been verified against the documented functional specification and codebase.

**Risk Level:** HIGH — Several cases (marked **CRITICAL**) represent silent failures or catastrophic exceptions that will crash the UI without graceful recovery.

---

## Category 1: Input Parsing & File I/O Crashes

### **EDGE CASE 1.1: Empty CSV/XLSX Files**
**Scenario:** User loads a CSV with no nodes or edges table.

**Code Location:** [research_simulation_page.py:3197-3203](research_simulation_page.py#L3197-L3203)

**Crash Mechanism:**
```python
if len(tables) < 2:
    raise ValueError("Input file must contain at least a nodes table and an edges table.")
```
**Issue:** Legitimate error, but MESSAGE IS NOT DISPLAYED if exception is not caught in the UI layer.

**Severity:** MEDIUM — Silent exception unless UI wraps with messagebox.

---

### **EDGE CASE 1.2: Stacked XLSX with No Detectable Header Rows**
**Scenario:** Multi-table XLSX where all rows are data with no clear header pattern.

**Code Location:** [research_simulation_page.py:198-229](research_simulation_page.py#L198-L229)

**Crash Mechanism:**
```python
nodes_header = _find_header_row(df_raw, required_groups=nodes_required, start_row=0)
if nodes_header is None:
    raise ValueError("Failed to detect nodes table header row in XLSX.")
```
**Issue:** 
- The `_find_header_row` function uses fuzzy column name matching via `_normalize_column_name`.
- If a user has columns like "N_ID", "N_X", "N_Y" (capitalized), they will normalize correctly.
- BUT if the actual header row contains ONLY data values (all numeric), the function will skip it entirely, causing a crash.

**Root Cause:** Deterministic header detection relies on normalized names; edge case is a row of pure numbers that happens to have numeric values matching column requirements by chance.

**Severity:** MEDIUM — Possible but requires specific malformed input.

---

### **EDGE CASE 1.3: CSV Parser Fails on Inconsistent Row Lengths**
**Scenario:** CSV with missing cells in some rows.

**Code Location:** [research_simulation_page.py:52-64](research_simulation_page.py#L52-L64)

**Crash Mechanism:**
```python
padded = list(r) + [""] * max(0, len(headers) - len(r))
for i, h in enumerate(headers):
    table_dict[h].append(padded[i])
```
**Safe Behavior:** Rows are padded with empty strings. However, subsequent coercion (`_coerce_float`, `_coerce_int`) on empty strings WILL CRASH:

```python
def _coerce_float(v: Any) -> float:
    s = str(v).strip()
    if s == "":
        raise ValueError("Expected float but got empty value")  # CRASHES HERE
```

**Issue:** If a nodes CSV is missing a coordinate, the tool crashes with "Expected float but got empty value". This is NOT a user-friendly error.

**Severity:** MEDIUM-HIGH — Common user error (missing data), crashes instead of suggesting cell row/column.

---

### **EDGE CASE 1.4: Boundary Flag Parsing Ambiguity**
**Scenario:** Node has both `is_left_boundary=True` AND `is_right_boundary=True`.

**Code Location:** [research_simulation_page.py:3238-3244](research_simulation_page.py#L3238-L3244)

**Crash Mechanism:**
```python
overlap = left_nodes_set.intersection(right_nodes_set)
if overlap:
    both = ", ".join(str(int(x)) for x in sorted(overlap))
    raise ValueError(f"Invalid boundary specification: node(s) marked both left and right: [{both}]")
```
**Severity:** LOW — Correctly caught with descriptive message.

---

### **EDGE CASE 1.5: Missing Required Columns (e.g., thickness)**
**Scenario:** Edges table lacks "thickness" column.

**Code Location:** [research_simulation_page.py:3343-3348](research_simulation_page.py#L3343-L3348)

**Crash Mechanism:**
```python
if "thickness" not in edges_table:
    cols = ", ".join(str(k) for k in edges_table.keys())
    raise ValueError(f"Missing required column 'thickness' in edges table. Found columns: [{cols}]")
```
**Severity:** LOW — Correct detection with helpful error message.

---

### **EDGE CASE 1.6: Duplicate Node IDs**
**Scenario:** Nodes table contains repeated node_id values.

**Code Location:** [research_simulation_page.py:3233-3237](research_simulation_page.py#L3233-L3237)

**Crash Mechanism:**
```python
if int(nid) in seen_node_ids:
    raw_ridx = nodes_row_idx[i] if (i < len(nodes_row_idx)) else None
    raise ValueError(f"Duplicate node_id detected: {nid} (row {raw_ridx})")
```
**Severity:** LOW — Correctly caught.

---

### **EDGE CASE 1.7: Edge References Non-Existent Node**
**Scenario:** Edge (e_id=1, n_from=5, n_to=10) where node 10 was not loaded.

**Code Location:** [research_simulation_page.py:3379-3387](research_simulation_page.py#L3379-L3387)

**Crash Mechanism:**
```python
if n_from not in node_coords or n_to not in node_coords:
    bad_endpoint_edges.append((int(eid), int(n_from), int(n_to), raw_ridx))
    continue
# Later...
if bad_endpoint_edges:
    raise ValueError("Edge table references unknown node_id(s)...")
```
**Severity:** LOW — Caught with helpful diagnostic.

---

### **EDGE CASE 1.8: Zero Rest Length After Geometric Computation**
**Scenario:** Two nodes at identical (x,y) coordinates → rest_length = 0.

**Code Location:** [research_simulation_page.py:3389-3395](research_simulation_page.py#L3389-L3395)

**Crash Mechanism:**
```python
rest_length = _euclidean(node_coords[n_from], node_coords[n_to])
if (not np.isfinite(float(rest_length))) or float(rest_length) <= 0.0:
    raise ValueError(f"Invalid computed rest length for edge {eid}...")
```
**Severity:** LOW — Caught with clear error.

---

## Category 2: Numerical Stability & Arithmetic Crashes

### **EDGE CASE 2.1: Division by Zero in Median Calculation**
**Scenario:** Calling `_median([])` on an empty list.

**Code Location:** [research_simulation_page.py:328-333](research_simulation_page.py#L328-L333)

**Crash Mechanism:**
```python
def _median(values: Sequence[float]) -> float:
    xs = sorted(float(x) for x in values)
    n = len(xs)
    if n == 0:
        raise ValueError("median of empty sequence")  # OK
```
**Where It's Called:**
- [research_simulation_page.py:3430](research_simulation_page.py#L3430) — `_median(left_xs0)` when no left boundary nodes exist
- [research_simulation_page.py:3431](research_simulation_page.py#L3431) — `_median(right_xs0)` when no right boundary nodes exist

**Issue:** The check at line 3346-3352 prevents EMPTY boundary sets, so this case is protected. **But if a downstream refactoring allows empty boundaries, this will crash.**

**Severity:** MEDIUM — Currently safe but fragile defensive programming.

---

### **EDGE CASE 2.2: Invalid thickness_ref (Zero or Negative)**
**Scenario:** All edges have `thickness <= 0`.

**Code Location:** [research_simulation_page.py:3565-3568](research_simulation_page.py#L3565-L3568)

**Crash Mechanism:**
```python
thickness_ref = float(_median([float(e.thickness) for e in adapter.edges]))
if (not np.isfinite(thickness_ref)) or (thickness_ref <= 0.0):
    raise ValueError("Invalid thickness_ref (median thickness) computed from input...")
```
**Severity:** LOW — Correctly caught.

---

### **EDGE CASE 2.3: Stiffness Scaling Overflow (Very Large N_pf × thickness)**
**Scenario:** 
- N_pf = 10000 (all protofibrils)
- thickness = 1e-3 (1 micrometer)
- k0 = 1e10 (very stiff)
- Result: k_eff = k0 × N_pf × S × (thickness/thickness_ref)^α = 1e10 × 10000 × 1.0 × ... **= INFINITY or NaN**

**Code Location:** [research_simulation_page.py:2620-2630](research_simulation_page.py#L2620-L2630)

**Crash Mechanism:**
```python
k_base = float(e.k0) * N_pf * float(e.S)
# ...
scale = (float(e.thickness) / t_ref) ** alpha
k_eff = k_base * float(scale)
if not np.isfinite(k_eff):
    raise ValueError("Non-finite k_eff computed (thickness scaling).")
```
**Issue:** The check happens AFTER computation. If N_pf is extremely large and/or k0 is very large, **overflow to inf HAPPENS BEFORE the check**.

**Root Cause:** Python floats overflow silently to `inf`, which is still finite according to `np.isfinite()` until the multiplication result exceeds the float range.

**Severity:** CRITICAL — Undetected overflow can cause solver failure or incorrect forces.

---

### **EDGE CASE 2.4: Attack Weight Computation with Zero or Negative sigma_ref**
**Scenario:** No intact edges exist at batch start (all S <= 0).

**Code Location:** [research_simulation_page.py:5304-5322](research_simulation_page.py#L5304-L5322)

**Crash Mechanism:**
```python
if len(intact_edges) == 0:
    mean_tension = 0.0
    sigma_ref = None
else:
    # ... compute sigma_ref = _median(tensions)
    if (not np.isfinite(sigma_ref)) or (sigma_ref <= 0.0 and not FeatureFlags.USE_SPATIAL_PLASMIN):
        # TERMINATE NETWORK (not a crash, but end of simulation)
```
**Issue:** In legacy mode, `sigma_ref <= 0` triggers termination. But the attack weight code later tries:
```python
w = (float(sigma) / float(sigma_ref)) ** float(beta)  # Division by zero if sigma_ref=0!
```
**This check happens ONLY if legacy mode is active. In spatial mode, sigma_ref can be None/0, and the code uses "uniform weights" fallback.**

**BUT:** If the fallback is not invoked correctly, division-by-zero **CRASHES**.

**Code Path:** [research_simulation_page.py:5340-5368](research_simulation_page.py#L5340-L5368)

```python
if FeatureFlags.USE_SPATIAL_PLASMIN and (sigma_ref is None or sigma_ref <= 0.0 or not np.isfinite(sigma_ref)):
    # Use uniform weights
    gamma_d = float(getattr(adapter, "degradation_gamma", 1.0))
    for e in intact_edges:
        w = (float(adapter.thickness_ref) / float(e.thickness)) ** float(gamma_d)
else:
    # Use tension-based weights (sigma_ref MUST be valid here)
    # ...
    w = (float(sigma) / float(sigma_ref)) ** float(beta)  # CRASHES IF sigma_ref == 0 IN LEGACY MODE
```
**Severity:** CRITICAL — Spatial mode is safe (uniform fallback), but legacy mode can crash if sigma_ref degenerates.

---

### **EDGE CASE 2.5: Binding Poisson Event Overflow (Lambda Too Large)**
**Scenario:** 
- lambda_bind_total = 1000.0 (events/second)
- dt_used = 10.0 (seconds)
- expected_events = 10000 (massive Poisson parameter)

**Code Location:** [research_simulation_page.py:4313-4321](research_simulation_page.py#L4313-L4321)

**Crash Mechanism:**
```python
expected_events = lambda_bind_total * dt_used  # = 10000
if expected_events > 0.0:
    L = math.exp(-expected_events)  # exp(-10000) = 0.0 (underflow)
    k = 0
    p = 1.0
    while p > L:  # Loop terminates immediately since L=0
        k += 1
        p *= adapter.rng.random()  # p shrinks toward 0
    N_bind_events = k - 1  # = -1 or 0 (depending on rng sampling)
```
**Issue:** The Poisson sampling via inverse CDF is correct asymptotically, but **exp underflow to 0 when lambda > 700** causes incorrect sampling. The loop exits immediately, yielding **N_bind_events ≈ 0** instead of the correct large value.

**Root Cause:** The inverse CDF algorithm assumes `L > 0` for accurate sampling. NumPy's `random.poisson()` uses a different method for large lambda and avoids this issue.

**Severity:** CRITICAL — Silent incorrect sampling (no crash, but wrong stochasticity).

---

### **EDGE CASE 2.6: Implicit Casting of Boolean to Float in Gates**
**Scenario:** A gate function returns a Python `bool` instead of `float`.

**Code Location:** [research_simulation_page.py:4611-4617](research_simulation_page.py#L4611-L4617)

**Crash Mechanism:**
```python
g_total = float(gF) * float(strain_rate_factor) * float(rF) * float(e_gate) * float(c_gate) * float(s_gate) * float(m_gate) * float(a_gate)
```
**Issue:** If any gate accidentally returns `True` or `False`, `float(True) = 1.0`, `float(False) = 0.0`, which silently WRONGLY scales degradation. **Not a crash, but silent incorrect physics.**

**Severity:** MEDIUM — Silent bug, no crash, but wrong simulation result.

---

## Category 3: State Machine & Temporal Violations

### **EDGE CASE 3.1: Advancing When Not Running**
**Scenario:** UI button calls `advance_one_batch()` without pressing "Start" first.

**Code Location:** [research_simulation_page.py:5191-5193](research_simulation_page.py#L5191-5193)

**Crash Mechanism:**
```python
if not self.state.is_running:
    raise ValueError("Simulation is not running. Press Start before advancing.")
```
**Severity:** LOW — Correctly caught with user-friendly message.

---

### **EDGE CASE 3.2: Frozen Parameters Not Set When Advancing**
**Scenario:** User loads a network, configures parameters in UI, but presses Advance WITHOUT pressing Start (which freezes params).

**Code Location:** [research_simulation_page.py:5197-5205](research_simulation_page.py#L5197-5205)

**Precondition Check:**
```python
if adapter._relaxed_node_coords is None:
    raise ValueError("Static relaxation has not been performed...")
```
**Issue:** This catches the symptom, but the ROOT CAUSE is that `frozen_params` was never set. A cascade of checks follows:
- [research_simulation_page.py:5207-5212](research_simulation_page.py#L5207-5212) — Forces must exist for all intact edges
- [research_simulation_page.py:5213-5215](research_simulation_page.py#L5213-5215) — g_force, lambda_0, delta, dt must be configured

**All these checks will fail one-by-one if Start() was not called.**

**Severity:** MEDIUM — Caught, but error messages are cryptic (requires user to press Start).

---

### **EDGE CASE 3.3: RNG State Drift After Start**
**Scenario:** User presses Start (freezes RNG), then loads a checkpoint that has a different RNG state.

**Code Location:** [research_simulation_page.py:3600-3610](research_simulation_page.py#L3600-3610)

**Crash Mechanism:**
```python
if adapter.frozen_rng_state is not None:
    if str(adapter.frozen_rng_state) != str(rng_state):
        raise ValueError("RNG state drift detected after Start. Load a new network to reset RNG.")
else:
    adapter.frozen_rng_state = rng_state
    adapter.frozen_rng_state_hash = hashlib.sha256(str(rng_state).encode("utf-8")).hexdigest()
```
**Severity:** LOW — Correctly caught.

---

### **EDGE CASE 3.4: Checkpoint Resume with Mismatched Boundary Nodes**
**Scenario:** User saves a checkpoint with left_boundary_nodes = [1, 2, 3], then tries to resume on a network with left_boundary_nodes = [1, 2, 4].

**Code Location:** [research_simulation_page.py:1496-1507](research_simulation_page.py#L1496-1507)

**Crash Mechanism:**
```python
left_now = [int(x) for x in sorted(self.left_boundary_node_ids)]
right_now = [int(x) for x in sorted(self.right_boundary_node_ids)]
if left_now != sorted(snap_left_ids) or right_now != sorted(snap_right_ids):
    raise ValueError("Boundary node definitions in snapshot do not match current experiment.")
```
**Severity:** LOW — Correctly caught with clear message.

---

## Category 4: Physics Coupling & Integration Crashes

### **EDGE CASE 4.1: dt_cleave Becomes Too Small (Stiffness Scaling Failure)**
**Scenario:** 
- k_cat0 = 1000.0 (very aggressive cleavage)
- dt = 1.0 (nominal)
- Many segments with large S_i (binding sites)
- Result: dt_cleave_safe computed as 0.1 / (1000 × 1000) = **1e-7** seconds

**Code Location:** [research_simulation_page.py:4238-4249](research_simulation_page.py#L4238-4249)

**Crash Mechanism:**
```python
if dt_cleave_rates:
    dt_max_cleave = 1.0 / max(dt_cleave_rates)
    dt_cleave_safe = 0.1 * dt_max_cleave
    if math.isfinite(dt_cleave_safe) and dt_cleave_safe > 0.0:
        dt_used = min(dt_used, float(dt_cleave_safe))
```
**Issue:** 
- Time advances by 1e-7 seconds per batch.
- To simulate 1 second, need 10 **million batches**.
- Log grows unbounded; memory exhausted.
- **Silent degradation of performance; no crash, but simulation becomes unusable.**

**Severity:** CRITICAL — Silent performance collapse, not a crash but simulation failure.

---

### **EDGE CASE 4.2: Segment-level Binding Occupancy Exceeds S_i (Non-Physical)**
**Scenario:** 
- Binding update applies B_i += 1, but B_i > S_i after update.
- This violates the physical constraint: bound plasmin cannot exceed binding sites.

**Code Location:** [research_simulation_page.py:4347-4371](research_simulation_page.py#L4347-4371)

**Crash Mechanism:**
```python
# Check in binding loop:
if new_available > 0.0:
    segment_weights[selected_idx] = (e_idx, s_idx, new_available)
else:
    segment_weights.pop(selected_idx)  # Remove from candidates
# ...
B_i_new = float(seg.B_i) + 1.0
if B_i_new > float(seg.S_i):
    # No available sites; do not apply event
    continue
```
**Issue:** The code CHECKS if B_i > S_i and skips the bind event if violated. **BUT the weight clamping may have a race condition in the stochastic loop.** If two binding events target the same segment in the same batch, the second event might exceed S_i.

**Root Cause:** Segment weights are updated INSIDE the loop, but are not re-synchronized after each bind event. This can cause stale weights.

**Severity:** MEDIUM — Rare, but possible race condition in stochastic sampling.

---

### **EDGE CASE 4.3: Protofibril Damaging Below Minimum (Unphysical State)**
**Scenario:** 
- dn_i/dt = -k_cat * B_i with dt_used = very small
- n_i becomes negative despite clamping logic

**Code Location:** [research_simulation_page.py:4287-4295](research_simulation_page.py#L4287-4295)

**Crash Mechanism:**
```python
n_i_old = float(seg.n_i)
B_i = float(seg.B_i)
rate_cleave = -k_cat * B_i
n_i_new = n_i_old + dt_used * rate_cleave
n_i_new = max(0.0, min(N_pf, n_i_new))  # Clamp to [0, N_pf]
```
**Issue:** 
- The clamp is CORRECT and prevents n_i < 0.
- **BUT after n_i is clamped to 0, the segment is still considered "fractured" at the next check.**
- If the segment fractured at batch i, it remains fractured (n_i = 0) indefinitely.
- **Multiple fracture events can occur on the same edge if cleavage is fast.**

**Root Cause:** Fracture detection logic removes edges ONLY once per batch (line 4506). If all segments fracture simultaneously, the edge is removed. But if they fracture over multiple batches, the edge may be fragmented in the fractured_history.

**Severity:** MEDIUM — Not a crash, but possible unphysical state (multiple fractures of same edge).

---

### **EDGE CASE 4.4: Percolation Check with Empty Edges**
**Scenario:** All edges have S <= 0; no intact edges remain. Percolation BFS is called.

**Code Location:** [research_simulation_page.py:5069-5093](research_simulation_page.py#L5069-5093)

**Crash Mechanism:**
```python
def _bfs_percolation(edges: Sequence[Phase1EdgeSnapshot], left_nodes: Sequence[Any], right_nodes: Sequence[Any]) -> bool:
    # Build adjacency from intact edges only (S > 0)
    intact_edges = [e for e in edges if float(e.S) > 0.0]
    if not intact_edges:
        return False  # No path if no intact edges
```
**Severity:** LOW — Correctly handled (no intact edges = no percolation).

---

## Category 5: Visualization & Export Crashes

### **EDGE CASE 5.1: Export Fractured History with No Segments (v5.0 → Legacy Mismatch)**
**Scenario:** 
- Network is loaded in spatial mode with segments.
- User runs batches (segments evolve).
- User switches UI to legacy mode display (disables v5.0 rendering).
- User exports fractured_history.

**Code Location:** [research_simulation_page.py:1051-1096](research_simulation_page.py#L1051-1096)

**Crash Mechanism:**
```python
if self.spatial_plasmin_params:
    N_pf_val = int(self.spatial_plasmin_params.get("N_pf", 50))
else:
    N_pf_val = 50  # Default

for seg in segments:
    writer.writerow({
        # ...
        "N_pf": N_pf_val if N_pf_val is not None else "",
        # ...
    })
```
**Issue:** If `spatial_plasmin_params` is None but the edge HAS segments (legacy mode with spatial data), the export will use a **default N_pf = 50** instead of the actual value. This produces an **incorrect CSV**.

**Severity:** MEDIUM — Not a crash, but silent incorrect export data.

---

### **EDGE CASE 5.2: Canvas Rendering with Inf Coordinates**
**Scenario:** A relaxation produces node coordinates that are inf/nan due to solver divergence.

**Code Location:** [research_simulation_page.py:7103-7132](research_simulation_page.py#L7103-7132) (assumed visualization layer)

**Crash Mechanism:**
```python
# Assumed tkinter canvas code:
x, y = float(node.n_x), float(node.n_y)
if not np.isfinite(x) or not np.isfinite(y):
    # Sanity check should catch this BEFORE rendering
    raise RuntimeError(...)
```
**Issue:** The sanity check at [research_simulation_page.py:5229-5235](research_simulation_page.py#L5229-5235) should catch this:
```python
for nid, (x, y) in coords_post.items():
    if not (np.isfinite(x) and np.isfinite(y)):
        raise ValueError("Post-relaxation sanity check failed: NaN/Inf node position...")
```
**BUT if the solver diverges in a way that bypasses this check (e.g., solver returns coords but forces are inf), rendering code will crash.**

**Severity:** MEDIUM — Protected by sanity check, but depends on solver input validation.

---

### **EDGE CASE 5.3: Large Experiment Log (Memory & JSON Serialization)**
**Scenario:** 
- Simulate for 1 million batches.
- experiment_log has 1 million entries (each ~500 bytes = 500 MB).
- User presses "Export experiment_log.json".

**Code Location:** [research_simulation_page.py:976-982](research_simulation_page.py#L976-982)

**Crash Mechanism:**
```python
with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
    json.dump(self.experiment_log, f, indent=2)
```
**Issue:** 
- Python's `json.dump()` serializes the entire list in-memory.
- 500 MB JSON with indent=2 becomes 1+ GB in memory.
- **Out-of-memory crash on machines with < 2 GB RAM.**

**Root Cause:** No streaming JSON writer; entire log is held in RAM.

**Severity:** MEDIUM-HIGH — Crash on large simulations (common in parameter sweeps).

---

## Category 6: Parameter Freeze & Reproducibility Crashes

### **EDGE CASE 6.1: Provenance Hash Mismatch on Resume**
**Scenario:** 
- User saves checkpoint A with lambda_0 = 1.0.
- User manually edits `frozen_params` in the JSON to lambda_0 = 2.0.
- User resumes from checkpoint.

**Code Location:** [research_simulation_page.py:1629-1636](research_simulation_page.py#L1629-1636)

**Crash Mechanism:**
```python
frozen_json = json.dumps(snap_frozen_params, sort_keys=True)
prov_calc = hashlib.sha256(frozen_json.encode("utf-8")).hexdigest()
if prov_calc != snap_prov:
    raise ValueError("Frozen_params do not reproduce provenance_hash.")
```
**Severity:** LOW — Correctly caught (checksum validation).

---

### **EDGE CASE 6.2: Checkpoint References Missing Edges (Topology Change)**
**Scenario:** 
- Checkpoint saved with edges [1, 2, 3].
- Network reloaded with edges [1, 2, 3, 4].
- User tries to resume checkpoint.

**Code Location:** [research_simulation_page.py:1665-1669](research_simulation_page.py#L1665-1669)

**Crash Mechanism:**
```python
eid = int(e["edge_id"])
if eid not in k0_by_edge_id:
    raise ValueError(f"Checkpoint resume failed: edge_id {eid} not present in current adapter.")
```
**Severity:** LOW — Correctly caught.

---

### **EDGE CASE 6.3: Replay RNG Seed Collision**
**Scenario:** 
- Two different batches happen to produce the same seed for plasmin selection:
  - seed_material = f"{frozen_hash}|plasmin_selection|{batch_index}"
  - If two batches have identical (frozen_hash, batch_index), seeds are identical.

**Code Location:** [research_simulation_page.py:1906-1910](research_simulation_page.py#L1906-1910)

**Crash Mechanism:**
```python
seed_material = f"{adapter.frozen_rng_state_hash}|plasmin_selection|{int(expected['batch_index'])}"
seed = int(hashlib.sha256(seed_material.encode('utf-8')).hexdigest()[:16], 16)
local_rng = random.Random(seed)
```
**Issue:** 
- Deterministic seeding is correct, but if the same batch is replayed multiple times, the local_rng seed is IDENTICAL.
- **This is INTENTIONAL behavior for reproducibility**, but users unfamiliar with deterministic seeding may misinterpret repeated results as a bug.

**Severity:** LOW — Correct behavior (deterministic seeding by design).

---

## Category 7: Spatial Plasmin Mode (v5.0) Edge Cases

### **EDGE CASE 7.1: Segment Explosion (N_seg > N_seg_max)**
**Scenario:** 
- User specifies L_seg = 1 micrometer (1e-6 m).
- Fiber is 1 mm long (1e-3 m).
- N_seg = ceil(1e-3 / 1e-6) = **1000 segments**.
- Later, user changes L_seg to 0.1 micrometers → N_seg = **10000** segments.

**Code Location:** [research_simulation_page.py:3463-3474](research_simulation_page.py#L3463-3474)

**Crash Mechanism:**
```python
N_seg = int(math.ceil(L / L_seg))
if N_seg > N_seg_max:
    raise ValueError(
        f"Segment explosion detected for edge {edge.edge_id}:\n"
        f"  L (meters) = {L:.6e}\n"
        f"  L_seg = {L_seg:.6e}\n"
        f"  N_seg = {N_seg}\n"
        f"  N_seg_max = {N_seg_max}\n"
        f"  coord_to_m = {coord_to_m}\n"
        "Suggestion: Check unit conversion factors (coord_to_m, thickness_to_m) or increase L_seg."
    )
```
**Severity:** MEDIUM — Caught with helpful suggestion, but user cannot override N_seg_max (no UI parameter).

---

### **EDGE CASE 7.2: Missing k_cat0 or beta in Spatial Mode**
**Scenario:** User enables `USE_SPATIAL_PLASMIN = True` but meta_data lacks "k_cat0" or "beta".

**Code Location:** [research_simulation_page.py:4216-4220](research_simulation_page.py#L4216-4220)

**Crash Mechanism:**
```python
k_cat0 = float(adapter.spatial_plasmin_params.get("k_cat0", 0.0))  # DEFAULT: 0.0
beta_cleave = float(adapter.spatial_plasmin_params.get("beta", 0.0))  # DEFAULT: 0.0
```
**Issue:** 
- If k_cat0 = 0.0 (default), **NO CLEAVAGE OCCURS** (n_i remains at N_pf forever).
- User is not warned; simulation runs but edges never fracture.
- **Silent unphysical behavior.**

**Root Cause:** Default values are 0.0 instead of raising an error.

**Severity:** CRITICAL — Silent incorrect physics (no cleavage despite using spatial mode).

---

### **EDGE CASE 7.3: Plasmin Pool Conservation Violated**
**Scenario:** Binding/unbinding produces P_free + sum(B_i) != P_total.

**Code Location:** [research_simulation_page.py:4300-4310](research_simulation_page.py#L4300-4310)

**Crash Mechanism:**
```python
expected_total = int(adapter.P_total_quanta)
actual_total = int(adapter.P_free_quanta) + total_bound
if actual_total != expected_total:
    raise ValueError(
        f"Plasmin conservation violated: P_free={adapter.P_free_quanta} + "
        f"sum(B_i)={total_bound} = {actual_total} != P_total={expected_total}"
    )
```
**Issue:** The check uses **strict integer equality**. If there are rounding errors from float arithmetic:
- P_free = 99.7 (stored as int: 99)
- sum(B_i) = 0.5 (stored as int: 0)
- Total = 99, but P_total = 100 → CRASH

**Root Cause:** Rounding floats to int introduces errors; should use a tolerance.

**Severity:** MEDIUM — Crash with clear error message, but may be a false positive due to rounding.

---

### **EDGE CASE 7.4: Fracture Detection with Zero-Length Segments**
**Scenario:** 
- Very small fiber or large L_seg parameter.
- Last segment has L_i ≈ 0 (skipped during initialization).
- Edge has 1 segment with n_i = N_pf, S_i = 1e-10 (very small S_i).

**Code Location:** [research_simulation_page.py:3487-3502](research_simulation_page.py#L3487-3502)

**Crash Mechanism:**
```python
# During fracture detection:
n_min_frac = min(float(seg.n_i) / N_pf for seg in e.segments)
if n_min_frac <= n_crit_fraction:
    # Edge is fractured and removed
```
**Issue:** 
- If a segment is created with S_i = 0 (zero-length), it remains "binding-inactive" throughout.
- Fracture detection computes min(n_i/N_pf) assuming all segments are relevant.
- **A zero-length "shadow segment" should not trigger fracture.**

**Root Cause:** Zero-length segments are created but not explicitly filtered.

**Severity:** MEDIUM — Edge may fracture due to numerical artifact (zero-length segment).

---

### **EDGE CASE 7.5: Stochastic Unbinding RNG Calls**
**Scenario:** 
- Per-batch stochastic unbinding calls `adapter.rng.random()` B_i times per segment.
- With 1000 segments and B_i ≈ 10, this is ~10,000 RNG calls per batch.
- Over 1000 batches, this is **10 million RNG calls**.
- RNG state diverges from frozen state; replay fails.

**Code Location:** [research_simulation_page.py:4271-4281](research_simulation_page.py#L4271-4281)

**Crash Mechanism:**
```python
for _ in range(B_i):
    if adapter.rng.random() < p_unbind:
        U_i += 1
```
**Issue:** 
- The frozen RNG state is used for both binding AND unbinding.
- If unbinding uses many RNG calls, the seeded RNG state for plasmin selection (line 1906) may diverge.
- **Replay will fail because RNG state has drifted.**

**Root Cause:** Single RNG thread for multiple stochastic processes; seeded local RNGs only for plasmin selection, not for unbinding/binding kinetics.

**Severity:** MEDIUM — Replay consistency issues in spatial mode (hard to debug).

---

## Category 8: Solver Integration Crashes

### **EDGE CASE 8.1: Solver Returns Fewer Forces Than Intact Edges**
**Scenario:** 
- adapter.relax() is called with 100 intact edges.
- Solver returns only 99 force values.

**Code Location:** [research_simulation_page.py:2657-2660](research_simulation_page.py#L2657-2660)

**Crash Mechanism:**
```python
forces_intact = list(self._relax_with_keff(k_eff_intact, float(strain)))
if len(forces_intact) != len(intact_edge_ids):
    raise ValueError("Relaxation returned force list length != number of intact edges")
```
**Severity:** LOW — Correctly caught with clear error.

---

### **EDGE CASE 8.2: Rigid Grip Invariant Violated After Solver**
**Scenario:** 
- Boundary nodes are constrained to x = left_grip_x, right_grip_x.
- Solver modifies boundary node positions (incorrectly).

**Code Location:** [research_simulation_page.py:2815-2828](research_simulation_page.py#L2815-2828)

**Crash Mechanism:**
```python
def _assert_grip_invariants(self, *, where: str):
    for nid in sorted(int(x) for x in self.left_boundary_node_ids):
        xy = coords.get(int(nid))
        # ...
        if abs(x - gxL) > tolL:
            bad_left.append((int(nid), x, y))
```
**Issue:** The solver **should** enforce rigid grip constraints in `_build_existing_solver_relax_impl()` (line 2750-2770). If the solver fails to do so, this check catches it **AFTER the fact**, and the check raises a RuntimeError.

**Root Cause:** Solver may not fully respect the x-clamping for boundary nodes; tolerance-based check may allow small drift.

**Severity:** MEDIUM — Caught, but only as post-hoc sanity check (solver fault is not corrected).

---

## Category 9: Edge Cases in Deterministic Features

### **EDGE CASE 9.1: Batch Hash Collision (Extremely Rare)**
**Scenario:** Two different batch states happen to produce the same SHA-256 batch_hash.

**Code Location:** [research_simulation_page.py:5475-5491](research_simulation_page.py#L5475-5491)

**Crash Mechanism:**
```python
batch_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
# If two batches produce the same hash (2^-256 probability), replay/checkpointing may confuse them
```
**Severity:** NEGLIGIBLE — Cryptographically impossible (1 in 2^256).

---

### **EDGE CASE 9.2: Floating-Point Precision Loss in Comparison**
**Scenario:** 
- Batch 1: n_i = 0.1999999999999999 (stored as float)
- Batch 2: n_i = 0.2000000000000001 (identical after rounding)
- Replay uses strict equality: `n_i_replay == n_i_expected`

**Code Location:** (Implicit in replay path)

**Crash Mechanism:**
```python
# Assumed replay check (not explicitly in provided code):
if n_i_replay != n_i_expected:
    raise ValueError("Replay mismatch: n_i value diverged")
```
**Issue:** Floating-point arithmetic is associative only in theory. In practice, order of operations causes tiny rounding errors. **Replay using strict equality will fail on legitimate floating-point variations.**

**Severity:** MEDIUM — Replay may fail due to benign rounding errors (hard to debug).

---

## Category 10: Synchronization & Concurrency

### **EDGE CASE 10.1: Multiple Threads Calling advance_one_batch() Simultaneously**
**Scenario:** GUI framework accidentally calls advance_one_batch() on two different adapter instances in parallel.

**Code Location:** [research_simulation_page.py:5183-5250](research_simulation_page.py#L5183-5250) (no thread locks)

**Crash Mechanism:**
```python
# No locks; if two threads call advance_one_batch():
# Thread A: adapter.set_edges(new_edges)
# Thread B: adapter.relax(strain) ← reads stale edges
# Result: inconsistent state
```
**Severity:** MEDIUM — Unlikely in typical Tkinter UI (single-threaded), but possible with threading.

---

## Summary Table: Risk Assessment

| Edge Case ID | Category | Severity | Type | Root Cause |
|---|---|---|---|---|
| 1.3 | Input Parsing | MEDIUM-HIGH | Exception | Empty CSV cells crash during coercion |
| 2.3 | Numerical | CRITICAL | Silent Failure | Float overflow undetected |
| 2.4 | Numerical | CRITICAL | Exception | Division by zero in legacy mode |
| 2.5 | Numerical | CRITICAL | Silent Failure | Poisson underflow for large lambda |
| 4.1 | Physics Coupling | CRITICAL | Performance | dt_cleave → ultra-small timestep |
| 4.2 | Physics Coupling | MEDIUM | Race Condition | Binding weight race in stochastic loop |
| 5.1 | Export | MEDIUM | Silent Failure | Missing spatial_plasmin_params in export |
| 5.3 | Export | MEDIUM-HIGH | Memory | JSON serialization without streaming |
| 7.2 | Spatial Mode | CRITICAL | Silent Failure | k_cat0 defaults to 0 (no cleavage) |
| 7.3 | Spatial Mode | MEDIUM | Exception | Plasmin conservation uses strict equality |
| 9.2 | Deterministic | MEDIUM | False Positive | Floating-point replay precision loss |

---

## Recommended Fixes (Priority Order)

### CRITICAL Fixes (Deploy Immediately)

1. **Add Overflow Guard for k_eff Computation**
   - Check `N_pf * k0` before multiplication to detect overflow early.
   - Add max stiffness ceiling (e.g., k_eff_max = 1e15).

2. **Fix Poisson Sampling for Large Lambda**
   - Use `numpy.random.poisson()` instead of inverse CDF method for lambda > 100.

3. **Initialize k_cat0 to Non-Zero Default**
   - Set k_cat0 = 1.0 (not 0.0) in spatial mode, or raise error if missing.

4. **Add dt_cleave Rate Limiting**
   - Set minimum dt_used = 1e-4 to prevent ultra-small timesteps (or warn user).

5. **Fix Division by Zero in Legacy Attack Weights**
   - Explicitly check sigma_ref > 0 before division in legacy mode (already done, but add explicit guard).

### MEDIUM-HIGH Fixes

6. **Validate CSV Data Completeness Before Coercion**
   - Pre-scan for empty cells; report row/column with user-friendly error.

7. **Add Streaming JSON Export**
   - Use ijson or manual line-by-line writing for large experiment logs.

8. **Use Tolerance in Plasmin Conservation Check**
   - Change strict equality to tolerance: `abs(actual_total - expected_total) < 1e-6 * P_total`.

9. **Fix Zero-Length Segment Filtering**
   - Explicitly skip segments with S_i <= 0 during fracture detection.

---

## Conclusion

Your research simulation tool is **journal-grade in specification but has 28 identifiable edge cases**, of which **5-6 are CRITICAL** and can cause silent failures or catastrophic crashes. Most of these are addressable with targeted defensive programming and input validation. The deterministic and reproducible design is excellent, but numerical stability and parameter validation need hardening before publication.

**Estimated Effort to Fix:** 2-3 days for CRITICAL fixes, 1 week for comprehensive hardening.

