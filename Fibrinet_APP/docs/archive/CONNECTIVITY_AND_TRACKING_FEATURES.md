# Graph Connectivity Detection and Degradation Tracking

**Version**: Core V2 with Network Clearance Detection
**Date**: 2026-01-03
**Status**: Implemented and Ready for Testing

---

## Overview

This document describes the new features implemented to match your experimental design:

1. **Graph Connectivity Detection**: Checks if left and right poles are connected
2. **Network Clearance Termination**: Stops when poles are disconnected
3. **Degradation History Tracking**: Records order, time, and details of fiber cleavages
4. **Research Data Export**: Exports degradation sequence for analysis

These features directly support your research goal: **studying how strain (tension) affects fibrinolysis**.

---

## New Features

### 1. Graph Connectivity Detection

**What it does**:
- Uses BFS (Breadth-First Search) to check if any path exists from left boundary nodes to right boundary nodes
- Only considers active (non-ruptured) fibers in the path search
- Checks after EVERY fiber cleavage (per your requirement)

**Implementation**:
- File: `src/core/fibrinet_core_v2.py`
- Function: `check_left_right_connectivity(state: NetworkState) -> bool`
- Returns: `True` if connected, `False` if cleared (disconnected)

**How it works**:
```python
# After each fiber cleavage:
if not check_left_right_connectivity(self.state):
    termination_reason = "network_cleared"
    # Simulation stops
```

**Biological Realism**:
- Mimics experimental setup where clearance = loss of mechanical continuity between poles
- More realistic than percentage-based clearance (which doesn't account for network topology)

---

### 2. Network Clearance Termination

**What changed**:
- **OLD**: Simulation stops when X% of fibers are broken (arbitrary threshold)
- **NEW**: Simulation stops when NO path connects left and right poles (biological clearance)

**Termination Conditions** (in priority order):
1. **network_cleared**: Left-right poles disconnected (NEW!)
2. **time_limit**: Reached max simulation time
3. **lysis_threshold**: Percentage threshold (legacy, kept for compatibility)
4. **complete_rupture**: All fibers broken

**Console Output**:
```
[Core V2] Network cleared at t=8.45s (left-right poles disconnected)
```

**Why this matters for your research**:
- Captures the exact moment when the network loses mechanical integrity
- Provides clearance time as a function of applied strain
- Enables you to study: "How does strain affect time-to-clearance?"

---

### 3. Degradation History Tracking

**What it records** (for EVERY fiber that completely ruptures):
- `order`: Sequential degradation order (1, 2, 3, ...)
- `time_s`: Exact time of rupture [seconds]
- `fiber_id`: Unique fiber identifier
- `length_m`: Fiber length at rupture [meters]
- `strain`: Fiber strain at rupture (dimensionless)
- `node_i`, `node_j`: Fiber endpoint nodes

**Data Structure**:
```python
state.degradation_history = [
    {
        'order': 1,
        'time': 2.34,
        'fiber_id': 17,
        'length': 1.5e-5,
        'strain': 0.18,
        'node_i': 3,
        'node_j': 8
    },
    # ... more entries
]
```

**Research Applications**:
- Identify which fibers cleave first (most vulnerable)
- Correlate strain with degradation rate
- Analyze spatial patterns (which regions clear first)
- Study avalanche dynamics (rapid cascades)

---

### 4. Degradation Order Export

**New GUI Button**: "Export Degradation Order"

**Where to find it**:
- Research Simulation page
- Located below "Export Fractured History" button

**How to use**:
1. Run a simulation to completion (network cleared)
2. Click **"Export Degradation Order"**
3. Choose save location (e.g., `degradation_history.csv`)
4. Click Save

**Output CSV Format**:
```csv
order,time_s,fiber_id,length_m,strain,node_i,node_j
1,2.34,17,1.5e-05,0.18,3,8
2,3.12,23,1.8e-05,0.25,5,12
3,3.89,9,1.2e-05,0.12,1,4
...
```

**Analysis Tips**:
- Open in Excel/Google Sheets for visualization
- Plot `strain` vs `order` to see if high-strain fibers cleave later
- Plot `time_s` vs `order` to detect avalanches (sudden jumps in order)
- Use `node_i`, `node_j` to reconstruct spatial degradation patterns

---

## Experimental Design Support

### Your Requirements (from conversation):

✅ **"Network to be stretched first, then perform simulation"**
   - Applied strain is set BEFORE simulation starts
   - 23% prestrain is applied to all fibers (Cone et al. 2020)
   - Right boundary poles are stretched by `applied_strain` parameter

✅ **"Check after every fiber cleavage"**
   - Connectivity check runs immediately after each `apply_cleavage()` call
   - No batching or periodic checks - happens after EVERY cleavage

✅ **"When there is no fibrin edge connecting left and right poles"**
   - BFS algorithm detects this exactly
   - Terminates with reason: `"network_cleared"`

✅ **"Order of degradation of fibrin edges"**
   - Degradation history tracks sequential order
   - Export to CSV for analysis

✅ **"Degradation time for each edge"**
   - Each entry includes exact rupture time in seconds

✅ **"Make biologically realistic decisions"**
   - Graph connectivity = mechanical integrity (realistic)
   - Strain-based cleavage model (matches literature)
   - Prestrain implementation (Cone et al. 2020)

---

## Testing the New Features

### Test 1: Verify Connectivity Detection

**Steps**:
1. Load network: `test/input_data/fibrin_network_big.xlsx`
2. Set parameters:
   - Plasmin: 1.0
   - Time step: 0.01
   - Max time: 100.0 (long, to ensure clearance happens)
   - Applied strain: 0.1

3. Click **Start**
4. Watch console for:
   ```
   [Core V2] Boundary nodes set: 4 left, 7 right
   [Core V2] Network cleared at t=XX.XXs (left-right poles disconnected)
   ```

**Expected**:
- Simulation should terminate with "network_cleared" reason
- Time-to-clearance should be logged

---

### Test 2: Export Degradation History

**Steps**:
1. After simulation completes (Test 1)
2. Click **"Export Degradation Order"**
3. Save as `test_degradation.csv`
4. Open file in Excel

**Expected CSV**:
- Header row with column names
- Rows sorted by `order` (1, 2, 3, ...)
- `time_s` values increasing (fibers cleave sequentially)
- `strain` values showing variation (some fibers under more strain)

**Verify**:
- Total rows = number of cleaved fibers before clearance
- All `fiber_id` values are unique
- No missing data (all columns filled)

---

### Test 3: Strain Effect on Clearance Time

**Purpose**: Verify that your strain-inhibited model works

**Steps**:

**Run A - Low Strain**:
1. Load network
2. Set Applied Strain: `0.05` (5%)
3. Run simulation
4. **Note clearance time** (from console)

**Run B - High Strain**:
1. Reload network
2. Set Applied Strain: `0.3` (30%)
3. Run simulation
4. **Note clearance time** (from console)

**Expected Physics**:
- **High strain → LONGER clearance time** (fibers resist cleavage when stretched)
- Example: Low strain clears in 10s, High strain clears in 30s

**If opposite occurs**: Bug in strain-inhibition model (report immediately)

---

## Files Modified

| File | Changes | Lines |
|------|---------|-------|
| `src/core/fibrinet_core_v2.py` | • Added `check_left_right_connectivity()` function<br>• Added degradation tracking fields to `NetworkState`<br>• Updated `apply_cleavage()` to record history<br>• Added connectivity check in `step()` method<br>• New termination reason: `"network_cleared"` | 615-664, 268-274, 760-791, 815-820 |
| `src/core/fibrinet_core_v2_adapter.py` | • Added boundary node sets to `NetworkState` creation<br>• Added `export_degradation_history()` method | 359-363, 706-742 |
| `src/views/tkinter_view/research_simulation_page.py` | • Added "Export Degradation Order" button<br>• Added `_on_export_degradation_order()` handler | 6100-6116, 7296-7331 |

---

## Physics Model Consistency

### Strain-Based Enzymatic Inhibition

The degradation tracking integrates seamlessly with your strain-inhibited fibrinolysis model:

**Cleavage Rate**:
```
k(ε) = k₀ × exp(-β × ε)

where:
  k₀ = 0.1 [1/s]     (baseline plasmin cleavage rate)
  β = 10.0           (mechanosensitivity)
  ε = (L - L_c) / L_c (fiber strain)
```

**Expected Behavior**:
- High-strain fibers (ε > 0.2) cleave 10× slower
- Low-strain fibers (ε < 0.05) cleave at baseline rate
- Degradation history should show low-strain fibers cleave first

**Validation**:
- Export degradation history
- Plot `strain` vs `order` in Excel
- Should see negative correlation (high strain → late degradation)

---

## Research Workflow

### Typical Experiment

1. **Setup**:
   - Load fibrin network
   - Set plasmin concentration
   - Set applied strain (independent variable)

2. **Run Simulation**:
   - Click Start
   - Wait for "network_cleared" message
   - Note clearance time

3. **Export Data**:
   - Export Degradation Order (CSV)
   - Export Experiment Log (CSV)
   - Export Network Snapshot (PNG)

4. **Analysis** (in Excel/Python):
   - Load degradation history CSV
   - Plot strain vs order
   - Plot time vs order (detect avalanches)
   - Calculate mean strain of early-cleaving fibers
   - Compare across different applied strains

5. **Publication**:
   - Figure 1: Clearance time vs applied strain
   - Figure 2: Degradation order colored by strain
   - Figure 3: Spatial degradation pattern
   - Table 1: Degradation statistics per strain level

---

## Known Limitations

1. **Plasmin Visualization** (green dots on edges):
   - Not yet implemented
   - Planned for future update
   - Degradation tracking works without visualization

2. **Intermediate States**:
   - Currently only tracks final rupture (S = 0)
   - Partial damage (S > 0) not logged
   - Consider if needed for research

3. **Disconnected Subgraphs**:
   - Only tracks left-right disconnection
   - Doesn't log intermediate fragmented clusters
   - Could be added if needed for analysis

---

## Next Steps

### Immediate Testing:
- Run Test 1-3 above to verify functionality
- Report any crashes or unexpected behavior
- Confirm clearance detection works correctly

### Future Enhancements (if needed):
- **Plasmin visualization**: Green dots showing enzyme locations
- **Edge highlighting**: Show which edges have plasmin attached
- **Subgraph logging**: Track all disconnected components
- **Strain field visualization**: Color edges by current strain

### Research Questions You Can Now Answer:
1. How does strain affect clearance time? (export data → plot)
2. Do high-strain fibers cleave last? (degradation history)
3. What is the critical strain threshold? (run multiple strains)
4. Do avalanches occur? (time jumps in degradation order)
5. Is clearance spatially localized? (node_i, node_j patterns)

---

## Console Output Examples

### Successful Run:
```
[Core V2] Network loaded: 50 fibers
[Core V2] Applied 23.0% initial prestrain to all fibers
[Core V2] Boundary nodes set: 4 left, 7 right
[Core V2] Simulation started
[Core V2 Heartbeat] t=1.00s, clearance=0.0%, cleaved=0
[Core V2 Heartbeat] t=2.00s, clearance=4.0%, cleaved=2
...
[Core V2 Heartbeat] t=8.00s, clearance=32.0%, cleaved=16
[Core V2] Network cleared at t=8.45s (left-right poles disconnected)
```

### Export Output:
```
Degradation history exported to test_degradation.csv
  Total fibers cleaved: 18
  Clearance time: 8.45s
```

---

## Technical Details

### BFS Algorithm Complexity:
- Time: O(N + E) where N = nodes, E = edges
- Space: O(N) for visited set
- Runs after each cleavage (fast enough for real-time)

### Memory Usage:
- Degradation history: ~100 bytes per cleaved fiber
- Total overhead: < 1 MB for networks with 1000+ fibers

### Determinism:
- BFS order depends on adjacency list construction
- Same network → same connectivity result
- Degradation order may vary (stochastic chemistry)

---

## References

**Implemented Based On**:
- Your experimental design (conversation 2026-01-03)
- Cone et al. (2020): 23% prestrain
- Li et al. (2017): Strain inhibits lysis
- Adhikari et al. (2012): 10-fold reduction at 23% strain

**Graph Theory**:
- BFS: Breadth-First Search (Cormen et al., CLRS)
- Connectivity: s-t path detection in undirected graphs

---

**Ready for Research!** Your tool now matches the biological experiment design and can export data for publication-quality analysis.
