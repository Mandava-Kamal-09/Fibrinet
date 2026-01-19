# Rookie Guide: Single Fiber Simulation

Welcome! This guide walks you through your first simulation step-by-step.

## What You'll Learn

1. How to launch the GUI
2. What the display shows
3. How to run a simulation
4. How NOT to misinterpret results

## Step 1: Launch the GUI

From the Fibrinet_APP folder, run:

```bash
python -m projects.single_fiber.src.single_fiber.gui.app
```

A window will open with several panels.

## Step 2: Understand the Interface

### Top-Left: Mode & Presets

- **Mode**: Start in "Novice" (checkbox unchecked). Advanced mode shows more controls.
- **Preset**: Choose a starting configuration. Start with "Hookean Spring (No Enzyme)".

### Top-Right: Model Scope & Limitations

**Read this panel carefully!** It tells you what this simulation can and cannot do:

- This is overdamped (no bouncing/oscillations)
- No thermal noise (no random jiggling)
- No bending (fiber is perfectly straight)
- Hazard models are hypotheses (not fitted to real data)

### Center: Chain Viewport

The green line is your fiber. Circles are nodes:
- Gray circle (node 0): Fixed - cannot move
- Blue circles: Free nodes - can be dragged

### Right Side: Status, Controls, Parameters

- **Status**: Shows current time, strain, tension
- **Controls**: Play/Pause/Reset buttons
- **Parameters**: Current simulation settings

### Bottom: Enzyme Coupling (if enabled)

Shows hazard rate and cleavage probability when enzyme is enabled.

## Step 3: Run Your First Simulation

1. Select preset: "Hookean Spring (No Enzyme)"
2. Click **Play**
3. Watch the chain stretch as the right end is pulled

What you'll see:
- Strain increases (shown in %)
- Tension increases (in piconewtons, pN)
- Chain gets longer

4. Click **Reset** to start over

## Step 4: Try Different Presets

### WLC Baseline
- Select "WLC Polymer (No Enzyme)"
- Click Play
- Watch tension increase faster as chain nears its limit
- **Rupture!** Chain breaks when extension reaches contour length

### With Enzyme
- Select "Hooke + Strain-Dependent Enzyme"
- The enzyme panel shows hazard rate (λ)
- Higher strain = higher hazard = more likely to cleave
- Cleavage is stochastic (random) - run multiple times for different outcomes

## Step 5: Interactive Exploration

**Drag a node:**
1. Click Pause
2. Click and drag a blue node
3. Watch tension update in real-time
4. Release to let chain relax

## Common Mistakes to Avoid

### 1. "The hazard rate is the real enzyme rate"
**Wrong.** These are *candidate models*, not fitted parameters. Use them to explore "what if" scenarios.

### 2. "More segments = more accurate"
**Partially wrong.** More segments give finer spatial resolution but don't change the underlying physics. 5 segments is fine for most purposes.

### 3. "The simulation predicts rupture time"
**Wrong.** Enzyme rupture is stochastic. The simulation shows *one possible outcome*. Run many replicates (use sweeps) for statistics.

### 4. "Animation speed affects physics"
**Wrong.** Animation speed only changes how fast you see updates. Physics timestep is fixed.

## Understanding the Numbers

### Strain (%)
- 0% = at rest length
- 50% = stretched to 1.5x rest length
- 100% = doubled in length

### Tension (pN)
- Typical range: 0-100 pN for these parameters
- Higher tension = more likely to cleave (for most hazard models)

### Hazard Rate (λ)
- Units: 1/μs (events per microsecond)
- λ = 0.01/μs means ~100 μs average time to cleavage
- **Mean time to event = 1/λ**

### P(rupture in Δt)
- Probability of cleavage in one timestep
- Small numbers (0.001% = 1 in 100,000 steps)
- Shown with explicit timestep so you know what "per step" means

## Next Steps

1. **Try all presets** to see different behaviors
2. **Enable Advanced Mode** for full parameter control
3. **Run parameter sweeps** for systematic exploration:
   ```bash
   python -m projects.single_fiber.src.single_fiber.enzyme_models.sweep_runner \
       projects/single_fiber/protocols/hazard_comparison.yaml
   ```

## Getting Help

- Full documentation: [README.md](README.md)
- Report issues: https://github.com/Mandava-Kamal-09/Fibrinet/issues

Happy simulating!
