# FibriNet Core V2 - Beginner's User Guide

**Welcome!** This guide will walk you through running your first fibrin network lysis simulation with the new Core V2 physics engine.

---

## Quick Start (3 Minutes)

### Step 1: Launch the Application

Open a terminal/command prompt and run:

```bash
cd C:\Users\manda\Documents\UCO\Fibrinet-main\Fibrinet_APP
python fibrinet.py
```

**What you'll see:**
- A window titled "FibriNet" opens
- Main menu with three options

### Step 2: Open Research Simulation

- Click **"Research Simulation"** button
- You'll see the simulation interface with:
  - Left panel: Controls
  - Right panel: Network visualization

### Step 3: Load a Network

1. Click **"Browse..."** button
2. Navigate to `test/input_data/`
3. Select `TestNetwork.xlsx` (or any `.xlsx` network file)
4. Click **"Load"** button

**What happens:**
- Console prints: `[Core V2] Network loaded: X fibers`
- Network appears in visualization panel
- Blue lines = fibers, gray dots = nodes

### Step 4: Set Parameters

Fill in these fields (left panel):

| Parameter | Recommended Value | What It Does |
|-----------|------------------|--------------|
| **Plasmin Concentration** | `1.0` | How much enzyme (higher = faster lysis) |
| **Time Step** | `0.01` | Simulation timestep in seconds |
| **Max Time** | `100.0` | When to auto-stop (seconds) |
| **Applied Strain** | `0.1` | How much to stretch (0.1 = 10% strain) |

### Step 5: Start Simulation

1. Click **"Start"** button
2. Watch the simulation run!

**What you'll see:**
- Network animates in real-time
- Blue fibers turn **red** when they rupture
- Metrics update every frame:
  - **Time**: Current simulation time
  - **Lysis %**: Percentage of ruptured fibers
  - **Active Fibers**: How many intact
  - **Cleaved Fibers**: How many ruptured

### Step 6: Control the Simulation

- **Pause**: Click "Pause" to freeze (click "Start" again to resume)
- **Watch it complete**: Simulation stops automatically when done

**Done!** You just ran your first Core V2 simulation.

---

## Understanding the Interface

### Left Panel (Controls)

#### 1. Network Loading
- **Path field**: Shows selected file
- **Browse...**: Pick a network file (`.xlsx` or `.csv`)
- **Load**: Load the selected network into memory

#### 2. Parameters
- **Plasmin Concentration (λ₀)**: Enzyme concentration
  - Range: `0.01` to `10.0`
  - Default: `1.0`
  - Higher = faster lysis

- **Time Step (Δt)**: How often physics updates
  - Range: `0.001` to `0.1` seconds
  - Default: `0.01`
  - Smaller = more accurate but slower

- **Max Time**: Auto-stop time
  - Range: `1.0` to `1000.0` seconds
  - Default: `100.0`

- **Applied Strain (ε)**: How much to stretch network
  - Range: `0.0` to `0.5`
  - Default: `0.1` (10% stretch)
  - Higher = more tension = faster rupture

#### 3. Controls
- **Start**: Begin simulation (or resume if paused)
- **Pause**: Freeze simulation
- **Stop**: Reset simulation to beginning

#### 4. Metrics (Read-Only)
- **Time**: Elapsed time in minutes
- **Lysis %**: Fraction of ruptured fibers
- **Active Fibers**: Number of intact fibers
- **Cleaved Fibers**: Number of ruptured fibers
- **Running / Paused**: Current state

### Right Panel (Visualization)

- **Blue lines**: Intact fibers (healthy)
- **Red lines**: Ruptured fibers (dead)
- **Gray dots**: Network nodes
- **Network updates in real-time** (10 FPS)

---

## Input File Format

Your Excel file must have this structure:

### Sheet 1: `nodes` (or first table in stacked format)

| n_id | n_x | n_y | is_left_boundary | is_right_boundary |
|------|-----|-----|------------------|-------------------|
| 1    | 10  | 5   | 1                | 0                 |
| 2    | 15  | 5   | 0                | 0                 |
| 3    | 20  | 10  | 0                | 1                 |

**Required columns:**
- `n_id` or `node_id`: Unique node number
- `n_x` or `x`: X coordinate
- `n_y` or `y`: Y coordinate
- `is_left_boundary`: `1` if node is on left edge, `0` otherwise
- `is_right_boundary`: `1` if node is on right edge, `0` otherwise

### Sheet 2: `edges` (or second table in stacked format)

| e_id | n_from | n_to | thickness |
|------|--------|------|-----------|
| 1    | 1      | 2    | 1.0       |
| 2    | 2      | 3    | 1.0       |

**Required columns:**
- `e_id` or `edge_id`: Unique edge number
- `n_from` or `from`: Start node ID
- `n_to` or `to`: End node ID
- `thickness`: Fiber thickness (can be `1.0` for all)

### Sheet 3: `meta_data` (optional)

| meta_key | meta_value |
|----------|------------|
| coord_to_m | 1e-6 |
| thickness_to_m | 1e-6 |

**Optional parameters:**
- `coord_to_m`: Unit conversion (default: `1e-6` = coordinates in microns)
- `thickness_to_m`: Thickness unit conversion (default: `1e-6`)

---

## Parameter Guide

### Choosing Parameters for Your Experiment

#### Plasmin Concentration (λ₀)

**What it does:** Controls how fast fibers degrade

| Value | Effect | Use Case |
|-------|--------|----------|
| `0.1` | Slow lysis | Study early-stage degradation |
| `1.0` | Normal lysis | Standard experiments |
| `5.0` | Fast lysis | Test avalanche dynamics |

**Tip:** Start with `1.0` and adjust up/down based on how fast you want things to happen.

#### Time Step (Δt)

**What it does:** How often the simulation updates

| Value | Effect | When to Use |
|-------|--------|-------------|
| `0.001` | Very accurate, slow | High-precision studies |
| `0.01` | Good balance | Most experiments ✓ |
| `0.1` | Fast, less accurate | Quick tests |

**Tip:** Use `0.01` unless you need extreme precision.

#### Applied Strain (ε)

**What it does:** How much you stretch the network

| Value | Effect | Biological Meaning |
|-------|--------|-------------------|
| `0.0` | No stretch | Resting state |
| `0.1` | 10% stretch | Mild tension ✓ |
| `0.3` | 30% stretch | High tension |
| `0.5` | 50% stretch | Extreme tension |

**Tip:** Use `0.1` for normal conditions. Higher strain → faster rupture.

---

## Running Your First Experiment

### Example 1: Standard Lysis Curve

**Goal:** Measure how lysis fraction increases over time

**Parameters:**
```
Plasmin Concentration: 1.0
Time Step: 0.01
Max Time: 100.0
Applied Strain: 0.1
```

**What to watch:**
- Lysis % should gradually increase
- First ruptures happen ~5-20 seconds
- Network fully lysed by ~50-100 seconds

**Result:** You'll get a sigmoid-shaped lysis curve (slow → fast → slow)

### Example 2: High Strain Experiment

**Goal:** See avalanche dynamics (cooperative rupture)

**Parameters:**
```
Plasmin Concentration: 1.0
Time Step: 0.01
Max Time: 50.0
Applied Strain: 0.3
```

**What to watch:**
- Faster initial ruptures
- Possible avalanches (many fibers rupture at once)
- Red fibers cluster together

**Result:** Network may collapse catastrophically once critical fibers rupture

### Example 3: Slow Degradation Study

**Goal:** Watch individual fiber ruptures

**Parameters:**
```
Plasmin Concentration: 0.1
Time Step: 0.01
Max Time: 200.0
Applied Strain: 0.05
```

**What to watch:**
- Slow, isolated ruptures
- Clear visualization of each event
- Long observation time

**Result:** Detailed view of degradation mechanism

---

## Understanding the Output

### Console Output

While simulation runs, you'll see:

```
[Core V2] Network loaded: 15 fibers
[Core V2] Unit scale: coord_to_m = 1.000000e-06
[Core V2] Simulation started
```

**What it means:**
- `15 fibers`: Network has 15 edges
- `coord_to_m = 1e-06`: Coordinates are in microns
- `Simulation started`: Physics engine running

When it finishes:

```
[Core V2] Terminated: lysis_threshold
[Core V2] Final time: 45.23s
[Core V2] Lysis fraction: 0.901
```

**What it means:**
- `lysis_threshold`: Stopped because >90% lysed
- `45.23s`: Took 45 seconds of simulation time
- `0.901`: 90.1% of fibers ruptured

### Termination Reasons

| Reason | Meaning |
|--------|---------|
| `lysis_threshold` | > 90% of fibers ruptured (normal) |
| `time_limit` | Reached max time before completing |
| `complete_rupture` | 100% of fibers ruptured |

---

## Troubleshooting

### Problem: "No network loaded"

**Solution:**
1. Click "Browse..." and select a file
2. Click "Load" button
3. Wait for "Network loaded" message
4. Then click "Start"

### Problem: "Invalid parameter"

**Solution:**
- Check all fields have numbers
- No letters or symbols
- Use `.` for decimals (not `,`)
- Example: `1.0` not `1,0`

### Problem: GUI freezes / "Not Responding"

**This shouldn't happen** - Core V2 uses non-blocking loop.

**If it does:**
1. Wait 10 seconds
2. If still frozen, close and restart
3. Try smaller network (< 100 fibers)
4. Report as bug

### Problem: Network doesn't appear after loading

**Solution:**
1. Check console for error messages
2. Verify Excel file has `is_left_boundary` and `is_right_boundary` columns
3. Ensure all node IDs in edges table exist in nodes table

### Problem: Simulation completes instantly

**Check:**
- Applied strain might be too high (try `0.1`)
- Plasmin concentration might be too high (try `1.0`)
- Network might be very small

### Problem: Simulation never finishes

**Check:**
- Plasmin concentration might be too low (try `1.0` instead of `0.01`)
- Max time might be too short (try `200.0`)
- Network might have no boundary nodes (can't apply strain)

---

## Advanced Features

### Exporting Metadata (For Publication)

After a simulation, you can export all parameters:

```python
# In Python console or script:
from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter

adapter = CoreV2GUIAdapter()
adapter.load_from_excel("test/input_data/TestNetwork.xlsx")
adapter.configure_parameters(plasmin_concentration=1.0, time_step=0.01, max_time=100.0, applied_strain=0.1)
adapter.start_simulation()

# After simulation runs...
adapter.export_metadata_to_file("my_experiment_metadata.json")
```

**This creates a JSON file with:**
- All physics equations used
- Numerical guards and safety limits
- Model assumptions
- Physical constants
- Your parameter values

**Use this for:**
- Publication supplementary materials
- Peer review defense
- Reproducibility

---

## Tips for Success

### 1. Start Small
- Use test networks (< 100 fibers) first
- Get familiar with controls
- Understand parameter effects

### 2. Watch the Metrics
- Lysis % tells you overall progress
- Active fibers shows remaining load-bearing capacity
- Time tells you how long simulation has run

### 3. Experiment with Parameters
- Try different strains: `0.05`, `0.1`, `0.2`, `0.3`
- Try different concentrations: `0.5`, `1.0`, `2.0`, `5.0`
- See how they interact!

### 4. Pause to Observe
- Click Pause to freeze and look closely
- Zoom in on interesting rupture patterns
- Click Start to resume

### 5. Console is Your Friend
- Keep console visible
- Watch for `[Core V2]` messages
- Errors print here (helpful for debugging)

---

## Example Workflow

**Scenario:** You want to compare lysis at different strains

**Steps:**

1. **Run 1: Low Strain**
   - Load `test/input_data/TestNetwork.xlsx`
   - Set: `λ₀=1.0`, `Δt=0.01`, `ε=0.05`, `t_max=100.0`
   - Click Start, watch until complete
   - Note final time and lysis curve shape

2. **Run 2: Medium Strain**
   - Keep same network loaded
   - Set: `ε=0.1` (only change)
   - Click Start
   - Compare: Did it lyse faster?

3. **Run 3: High Strain**
   - Keep same network loaded
   - Set: `ε=0.2`
   - Click Start
   - Compare: Avalanche behavior?

**Result:** You've now characterized strain-dependent lysis!

---

## Next Steps

### Once You're Comfortable:

1. **Create Your Own Networks**
   - Edit Excel files with custom geometries
   - Test different topologies
   - Vary fiber thickness

2. **Run Parameter Sweeps**
   - Systematically vary one parameter
   - Record results
   - Plot lysis curves

3. **Compare to Experiments**
   - Match simulation parameters to your lab data
   - Validate model predictions
   - Publish results!

---

## Quick Reference

### Keyboard Shortcuts
- **None yet** - Use mouse for all controls

### File Locations
- Test networks: `test/input_data/`
- Core V2 engine: `src/core/fibrinet_core_v2.py`
- GUI code: `src/views/tkinter_view/research_simulation_page.py`
- This guide: `USER_GUIDE_CORE_V2.md`

### Default Parameter Values

| Parameter | Default | Range |
|-----------|---------|-------|
| Plasmin Concentration | `1.0` | `0.01` - `10.0` |
| Time Step | `0.01` s | `0.001` - `0.1` s |
| Max Time | `100.0` s | `1.0` - `1000.0` s |
| Applied Strain | `0.1` | `0.0` - `0.5` |

---

## Getting Help

### If You're Stuck:

1. **Check this guide** - Most common issues are here
2. **Check console** - Error messages explain what's wrong
3. **Try test network** - Verify Core V2 works with known-good file
4. **Check file format** - Excel files must match required structure

### For Advanced Questions:

- See: `CORE_V2_INTEGRATION_STATUS.md` (technical details)
- See: `LAUNCH_CHECKLIST.md` (peer review defense)
- See: Docstrings in `src/core/fibrinet_core_v2.py` (physics equations)

---

## You're Ready!

Core V2 is now fully integrated and ready to use. Just run `python fibrinet.py` and start exploring!

**Happy simulating! 🔬**
