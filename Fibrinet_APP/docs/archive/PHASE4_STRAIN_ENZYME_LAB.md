# Phase 4: Strain–Enzyme Coupling Lab

## Goal

Build a flexible experimentation framework to explore strain/tension-dependent enzyme cleavage hazard models. The framework enables:

1. Defining multiple candidate hazard functions (constant, linear, exponential, catch-bond, etc.)
2. Running parameter sweeps across coupling strengths
3. Computing metrics (survival curves, mean rupture times, variance)
4. Visualizing results via GUI controls
5. Later fitting to whatever experimental dataset is provided

## Explicit Non-Goals

- **No single-paper validation target**: Parameters remain flexible and data-agnostic
- **No hard-coded ground truth**: The framework explores candidate relations; fitting happens when real data arrives
- **No modifications to frozen physics modules**

## Frozen Modules (DO NOT MODIFY)

The following modules are frozen and must not be edited during Phase 4:

```
src/core/force_laws/wlc.py
src/core/force_laws/hookean.py
src/core/force_laws/units.py
src/core/force_laws/types.py
projects/single_fiber/src/single_fiber/model.py
projects/single_fiber/src/single_fiber/state.py
projects/single_fiber/src/single_fiber/chain_integrator.py
projects/single_fiber/src/single_fiber/chain_model.py
projects/single_fiber/src/single_fiber/chain_state.py
```

Any strain–enzyme coupling must be implemented as a *separate layer* that reads state but does not modify physics internals.

## Planned Deliverables

### 1. Coupling Library (`enzyme_models/`)
- `hazard_functions.py`: Registry of hazard rate functions
  - `constant(strain, params)` — baseline, no coupling
  - `linear(strain, params)` — λ = λ₀ + α·ε
  - `exponential(strain, params)` — λ = λ₀·exp(β·ε)
  - `catch_slip(strain, params)` — biphasic catch-then-slip
- `coupling_registry.py`: Register/retrieve hazard functions by name
- `sampler.py`: Poisson sampling with strain-dependent rates

### 2. Parameter Sweeps (`sweeps/`)
- `sweep_runner.py`: Batch execution over parameter grids
- `metrics.py`: Survival probability, hazard curves, mean/variance
- `export.py`: Structured output (CSV, JSON) for analysis

### 3. GUI Controls
- Hazard function selector dropdown
- Coupling parameter sliders (α, β, λ₀)
- Real-time hazard rate display
- Survival curve plot panel

### 4. Reproducibility
- Sweep configs stored as YAML
- Random seeds explicit and logged
- Full parameter provenance in output metadata

## Architecture

```
enzyme_models/
├── __init__.py          # Registry access
├── hazard_functions.py  # Hazard rate implementations
├── coupling_registry.py # Name → function mapping
├── sampler.py           # Poisson process with variable rate
└── README.md            # Contract and usage

sweeps/
├── sweep_runner.py      # Batch execution
├── metrics.py           # Statistical analysis
└── export.py            # Output formatting
```

## Success Criteria

Phase 4 is complete when:

1. At least 3 hazard functions are implemented and tested
2. Parameter sweep produces reproducible survival curves
3. GUI allows interactive exploration of coupling effects
4. No frozen modules are modified (enforced by test)
5. Framework is ready to accept any experimental dataset for fitting
