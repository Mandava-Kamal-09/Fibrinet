# FibriNet Research Simulation Tool - Complete Testing Guide

**Version**: Core V2 with Strain-Inhibited Fibrinolysis
**Date**: 2026-01-03
**Target Users**: Beginners and researchers with no programming experience

---

## Table of Contents
1. [Before You Start](#before-you-start)
2. [Test 1: Launch and Interface Check](#test-1-launch-and-interface-check)
3. [Test 2: Load and Visualize Network](#test-2-load-and-visualize-network)
4. [Test 3: Run Basic Simulation](#test-3-run-basic-simulation)
5. [Test 4: Control Simulation (Pause/Resume)](#test-4-control-simulation-pauseresume)
6. [Test 5: Export Results](#test-5-export-results)
7. [Test 6: Physics Validation](#test-6-physics-validation)
8. [Troubleshooting](#troubleshooting)
9. [Success Checklist](#success-checklist)

---

## Before You Start

### What You Need:
- ✅ FibriNet installed and working
- ✅ Python virtual environment activated (`.venv`)
- ✅ Test network file: `test/input_data/fibrin_network_big.xlsx`
- ✅ Terminal/console open to see diagnostic messages

### Activate Environment:
```bash
# Windows PowerShell:
.venv\Scripts\Activate.ps1

# Windows CMD:
.venv\Scripts\activate.bat

# Mac/Linux:
source .venv/bin/activate
```

### Launch FibriNet:
```bash
python FibriNet.py
```

**Expected**: A window opens with menu options.

---

## Test 1: Launch and Interface Check

### Step 1.1: Open Research Simulation
1. In the main menu, click **"Research Simulation"**
2. A new page should appear with:
   - File selection field (top)
   - Parameter input boxes (left side)
   - Large empty canvas (center/right)
   - Control buttons at bottom
   - Metrics display (bottom left)

### Step 1.2: Check Interface Elements

**VERIFY YOU SEE**:
- ✅ "Select Network File" field with "Browse" button
- ✅ Parameter boxes labeled:
  - Plasmin Concentration
  - Time Step (s)
  - Max Time (s)
  - Applied Strain
- ✅ Buttons: **Start**, **Pause**, **Resume**, **Stop**
- ✅ Metrics display showing:
  - Time (min)
  - Running (True/False)
  - Paused (True/False)
  - Active Fibers
  - Cleaved Fibers
  - Lysis %

**✓ PASS**: All elements visible
**✗ FAIL**: Missing buttons or fields → See [Troubleshooting](#troubleshooting)

---

## Test 2: Load and Visualize Network

### Step 2.1: Select Network File
1. Click **"Browse"** button next to "Select Network File"
2. Navigate to: `test/input_data/`
3. Select file: **`fibrin_network_big.xlsx`**
4. Click **"Open"**

### Step 2.2: Load Network
1. After selecting file, click **"Load Network"** (or the file loads automatically)
2. **A popup appears**: "Network loaded - 50 fibers"
3. Click **"OK"** to close popup

### Step 2.3: Check Terminal Output

**EXPECTED CONSOLE MESSAGES**:
```
Loaded network from C:/.../fibrin_network_big.xlsx:
  Nodes: 41
  Edges: 50
  Left boundary: 4 nodes
  Right boundary: 7 nodes
  Unit conversion: coord_to_m=1e-05, thickness_to_m=1e-09
[Core V2] Network loaded: 50 fibers
[Core V2] Unit scale: coord_to_m = 1.000000e-05
[Core V2] Applied 23.0% initial prestrain to all fibers
[Core V2 Render] Render data: 41 nodes, 50 edges
[Core V2 Render] Canvas dimensions: 800x600 (or similar)
[Core V2 Render] SUCCESS: Rendered 50 edges and 41 nodes
```

### Step 2.4: Check Visualization

**VERIFY ON CANVAS**:
- ✅ **Blue lines** (fibers/edges) forming a network
- ✅ **Small gray circles** (nodes) at connection points
- ✅ Network is **centered** and **scaled to fit** canvas
- ✅ Network appears **connected** (not scattered random lines)
- ✅ Network has **4 nodes on left edge**, **7 nodes on right edge**

**WHAT IT SHOULD LOOK LIKE**:
- A mesh/web of blue lines connecting nodes
- Denser in some areas, sparser in others (natural variation)
- Left and right boundaries clearly visible
- No overlapping text or garbled display

**✓ PASS**: Network visible with blue fibers and gray nodes
**✗ FAIL**: Blank canvas or error → See [Troubleshooting](#troubleshooting)

---

## Test 3: Run Basic Simulation

### Step 3.1: Set Parameters
Fill in the parameter boxes with these values:

| Parameter | Value | What it means |
|-----------|-------|---------------|
| **Plasmin Concentration** | `1.0` | Amount of enzyme degrading fibers |
| **Time Step (s)** | `0.01` | How fine the simulation steps are |
| **Max Time (s)** | `10.0` | Stop simulation after 10 seconds |
| **Applied Strain** | `0.1` | Stretch network by 10% |

**HOW TO ENTER**:
1. Click in each text box
2. Delete old value (if any)
3. Type new value exactly as shown
4. Press `Tab` or click next box

### Step 3.2: Start Simulation
1. Click **"Start"** button
2. **IMMEDIATELY WATCH**:
   - Canvas should start updating
   - Some blue fibers may turn **red** (cleaved/broken)
   - Metrics at bottom should show changing numbers

### Step 3.3: Monitor Console Output

**EXPECTED MESSAGES** (appearing every ~1 second):
```
[Core V2] Simulation started
[Core V2 Heartbeat] t=1.00s, clearance=0.0%, cleaved=0
[Core V2 Heartbeat] t=2.00s, clearance=2.0%, cleaved=1
[Core V2 Heartbeat] t=3.00s, clearance=4.0%, cleaved=2
...
```

**WHAT TO WATCH FOR**:
- ✅ `clearance` percentage **increases** over time
- ✅ `cleaved` count **increases** over time
- ✅ Time `t` **increments** smoothly
- ✅ No error messages or crashes

### Step 3.4: Observe Canvas During Simulation

**VISUAL CHANGES**:
- ✅ Some **blue fibers turn red** (these are cleaved)
- ✅ Network may **deform slightly** as fibers break
- ✅ Red fibers may **disappear** if completely broken
- ✅ Canvas **updates smoothly** (no freezing)

### Step 3.5: Wait for Completion
The simulation will **automatically stop** when:
- Time reaches 10 seconds, OR
- Clearance reaches threshold (~90%)

**FINAL POPUP**:
```
Simulation complete
Reason: clearance_threshold (or max_time)
Time: 10.00s
Clearance: 85.0%
```

**✓ PASS**: Simulation runs, fibers cleave, completes successfully
**✗ FAIL**: Crashes, freezes, or no changes → See [Troubleshooting](#troubleshooting)

---

## Test 4: Control Simulation (Pause/Resume)

### Step 4.1: Start a New Simulation
1. Click **"Stop"** (if previous simulation is still running)
2. Reload network if needed (Browse → fibrin_network_big.xlsx)
3. Set same parameters as Test 3
4. Click **"Start"**

### Step 4.2: Pause Simulation
1. **While simulation is running**, click **"Pause"** button
2. **OBSERVE**:
   - ✅ Canvas **stops updating** (fibers freeze in place)
   - ✅ Console shows: `[Core V2] Paused`
   - ✅ Metrics stop changing
   - ✅ "Paused" metric shows **True**

### Step 4.3: Resume Simulation
1. Click **"Resume"** button
2. **OBSERVE**:
   - ✅ Canvas **starts updating again**
   - ✅ Console shows: `[Core V2] Resumed`
   - ✅ Heartbeat messages resume
   - ✅ "Paused" metric shows **False**

### Step 4.4: Test GUI Responsiveness
**While simulation is running**:
1. Try to **drag the window** → Should work smoothly
2. Try to **resize the window** → Should work smoothly
3. Click **Pause** → Should respond immediately
4. Click **Resume** → Should respond immediately

**✓ PASS**: Pause/Resume work, GUI stays responsive
**✗ FAIL**: GUI freezes or buttons don't respond → See [Troubleshooting](#troubleshooting)

---

## Test 5: Export Results

### Step 5.1: Run a Complete Simulation
1. Load network (`fibrin_network_big.xlsx`)
2. Set parameters (use Test 3 values)
3. Click **Start**
4. **Wait for simulation to complete** (popup message)
5. Click **OK** to close completion popup

### Step 5.2: Export Network Snapshot (Image)
1. Look for **"Export Network Snapshot"** button
2. Click the button
3. A **file save dialog** appears
4. Choose location (e.g., Desktop)
5. Enter filename: `test_network_snapshot.png`
6. Click **"Save"**

**VERIFY**:
- ✅ File created at chosen location
- ✅ Open the PNG file → Should show network visualization
- ✅ Red fibers (cleaved) visible if any broke during simulation

### Step 5.3: Export Experiment Log (CSV)
1. Click **"Export Experiment Log"** button
2. Save as: `test_experiment_log.csv`
3. Click **"Save"**

**VERIFY**:
- ✅ CSV file created
- ✅ Open with Excel/Google Sheets
- ✅ Contains columns like:
  - batch_index
  - time
  - cleaved_edges_total
  - lysis_fraction
  - mean_tension

**CHECK DATA QUALITY**:
- ✅ Time column **increases** each row
- ✅ cleaved_edges_total **increases or stays same** (never decreases)
- ✅ lysis_fraction values between **0.0 and 1.0**

### Step 5.4: Export Fractured History (Optional)
1. Click **"Export Fractured History"** button
2. Save as: `test_fractured_history.csv`
3. Open file and verify it lists cleaved edges

**✓ PASS**: All exports work, files contain valid data
**✗ FAIL**: Export fails or files are empty → See [Troubleshooting](#troubleshooting)

---

## Test 6: Physics Validation

### Test 6A: Strain Inhibits Cleavage (High vs Low Strain)

**Purpose**: Verify that stretched fibers cleave SLOWER (per literature)

#### Part 1: High Strain Simulation
1. Load network
2. Set parameters:
   - Plasmin Concentration: `1.0`
   - Time Step: `0.01`
   - Max Time: `10.0`
   - **Applied Strain: `0.3`** ← HIGH (30% stretch)
3. Click Start
4. **Note the final clearance %** when simulation ends
5. **Write down**: "High strain clearance = ____%"

#### Part 2: Low Strain Simulation
1. **Reload** the same network (Browse → fibrin_network_big.xlsx)
2. Set parameters:
   - Plasmin Concentration: `1.0`
   - Time Step: `0.01`
   - Max Time: `10.0`
   - **Applied Strain: `0.05`** ← LOW (5% stretch)
3. Click Start
4. **Note the final clearance %** when simulation ends
5. **Write down**: "Low strain clearance = ____%"

#### Compare Results
**EXPECTED PHYSICS**:
- ✅ **Low strain** clearance should be **HIGHER** than high strain
- ✅ Example: Low = 80%, High = 50% (stretched fibers resist cleavage)
- ✅ This matches literature (Adhikari et al. 2012, Li et al. 2017)

**✓ PASS**: Low strain → more clearance (stretched fibers cleave slower)
**✗ FAIL**: High strain → more clearance (physics is inverted!) → Report bug

---

### Test 6B: Prestrain Effect

**Purpose**: Verify that 23% initial prestrain is applied

1. Load network
2. Check console for: `[Core V2] Applied 23.0% initial prestrain to all fibers`
3. Set Applied Strain to `0.0` (no additional stretch)
4. Click Start
5. **OBSERVE**:
   - ✅ Network should **already be under tension** (even with 0 applied strain)
   - ✅ Some fibers may cleave due to prestrain alone
   - ✅ Forces are non-zero from the start

**✓ PASS**: Console confirms 23% prestrain applied
**✗ FAIL**: Message missing → Report bug

---

### Test 6C: Force Distribution Visualization

**Purpose**: Verify force calculations are working

1. Run any simulation
2. During/after simulation, observe canvas colors:
   - ✅ **Blue fibers**: Intact, under normal stress
   - ✅ **Red fibers**: Cleaved/broken
3. Check console logs for force-related messages
4. Export experiment log CSV
5. Open CSV and check `mean_tension` column:
   - ✅ Values should be **positive** (tension forces)
   - ✅ Values should **vary** over time as network degrades

**✓ PASS**: Forces calculated, visualized, and logged correctly
**✗ FAIL**: All tensions are zero or negative → Report bug

---

## Troubleshooting

### Problem: Blank canvas after loading
**Solutions**:
1. Check console for error messages
2. Verify file path is correct
3. Try a different network file (e.g., `TestNetwork.xlsx`)
4. Close and restart FibriNet
5. Re-activate virtual environment

### Problem: Start button does nothing
**Solutions**:
1. Check console for errors
2. Verify network is loaded (should see success popup)
3. Ensure all parameters are filled in (no empty boxes)
4. Check parameters are valid numbers (no letters)
5. Try clicking Stop, then Start again

### Problem: Simulation freezes
**Solutions**:
1. Click Pause, wait 5 seconds, click Resume
2. Check console for error stack traces
3. Reduce Max Time to 5.0 seconds
4. Try smaller network (TestNetwork.xlsx)
5. Close and restart FibriNet

### Problem: No heartbeat messages in console
**Solutions**:
1. Verify simulation is running ("Running" metric = True)
2. Wait longer (heartbeats appear every ~1 second)
3. Check that you're looking at the correct console window
4. Try clicking Resume if paused

### Problem: Export fails
**Solutions**:
1. Check that simulation has run (experiment log won't exist if never started)
2. Verify you have write permissions to save location
3. Try saving to Desktop instead of Documents
4. Close any open CSV files before exporting again
5. Try a different filename

### Problem: Physics validation fails (wrong clearance trend)
**Report to developer with**:
1. Console output (copy entire log)
2. Parameter values used
3. Final clearance percentages observed
4. Expected vs actual behavior

---

## Success Checklist

### ✅ Condition 1: Complete GUI Functionality
- [ ] Interface loads with all elements visible
- [ ] Network file browser works
- [ ] Network visualizes on canvas (blue fibers, gray nodes)
- [ ] Start button starts simulation
- [ ] Pause button pauses simulation
- [ ] Resume button resumes simulation
- [ ] Stop button stops simulation
- [ ] Metrics update in real-time
- [ ] GUI remains responsive during simulation

### ✅ Condition 2: Complete Physics Implementation
- [ ] 23% prestrain applied (console confirms)
- [ ] Strain inhibits cleavage (low strain → more clearance)
- [ ] Forces calculated (mean_tension in CSV)
- [ ] Fibers cleave over time (clearance increases)
- [ ] Red fibers appear (visual feedback)
- [ ] Network deforms realistically

### ✅ Condition 3: Complete Output of Results
- [ ] Network snapshot exports to PNG
- [ ] Experiment log exports to CSV
- [ ] CSV contains valid data (increasing time, valid fractions)
- [ ] Fractured history exports successfully
- [ ] Files open in external programs (image viewer, Excel)

### ✅ Condition 4: Bug-Free Operation
- [ ] No crashes during normal use
- [ ] No freezes or hangs
- [ ] No error popups (except user mistakes like missing file)
- [ ] Console shows diagnostic messages (no stack traces)
- [ ] Pause/Resume work without issues
- [ ] Multiple simulations can be run sequentially

---

## Advanced Testing (Optional)

### Large Network Test
1. Load: `test/input_data/mega_complex.xlsx` (if available)
2. Verify: Smooth visualization and simulation
3. Purpose: Test performance with hundreds of fibers

### Edge Cases
1. **Zero strain**: Set Applied Strain = 0.0, verify simulation still works
2. **High plasmin**: Set Plasmin Concentration = 10.0, verify fast cleavage
3. **Long simulation**: Set Max Time = 100.0, verify no memory leaks

### Reproducibility Test
1. Run same simulation twice with identical parameters
2. Export experiment logs from both runs
3. Compare CSV files → Should have nearly identical results (small random variation OK)

---

## Getting Help

### If tests fail:
1. Check [Troubleshooting](#troubleshooting) section first
2. Copy **entire console output** to a text file
3. Note which specific test failed
4. Report issue with:
   - Test number and step (e.g., "Test 3, Step 3.2")
   - Console output
   - Screenshots if possible
   - Operating system (Windows/Mac/Linux)

### Contact Developer:
- Include: `TESTING_GUIDE.md` test results
- Attach: Console logs and exported CSV files
- Specify: Which success checklist items passed/failed

---

## Conclusion

**If all tests pass**, FibriNet is fully functional and ready for research use!

**Congratulations!** You've verified:
- ✅ GUI works completely
- ✅ Physics implemented correctly (strain-inhibited fibrinolysis)
- ✅ Results export successfully
- ✅ No bugs or errors

**Next Steps**:
- Use FibriNet for your research simulations
- Import your own network files
- Adjust parameters for your experiments
- Export data for analysis

---

**Last Updated**: 2026-01-03
**FibriNet Version**: Core V2 (Strain-Inhibited Enzymatic Cleavage)
**Physics Model**: Cone et al. (2020) + Li et al. (2017) + Adhikari et al. (2012)
