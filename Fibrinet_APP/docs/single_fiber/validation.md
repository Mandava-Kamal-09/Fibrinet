# Validation Approach Note

## Phase 4 Definition

Phase 4 is defined as **data-agnostic experimentation**, not validation against any specific published dataset.

The goal is to build a flexible strain–enzyme coupling framework that:
- Implements multiple candidate hazard models
- Allows parameter sweeps and sensitivity analysis
- Produces reproducible metrics (survival curves, statistics)
- Can later be fit to whatever experimental data becomes available

## No Pre-Committed Validation Target

- No specific paper (Liu et al., Weisel et al., or any other) is designated as the ground truth
- Parameters are intentionally kept flexible for future calibration
- The framework supports exploration before any fitting is attempted

## Validation Strategy (When Data Arrives)

When experimental data is provided:
1. Define objective function (e.g., log-likelihood, MSE on survival curves)
2. Use sweep infrastructure to explore parameter space
3. Report best-fit parameters with confidence intervals
4. Document model limitations and discrepancies

## Current Status

- Phase 3: COMPLETE (physics engine implemented)
- Phase 4: SCAFFOLD ONLY (experimentation framework structure defined)

To verify tests pass: `pytest projects/single_fiber/tests -v`

The tool implements single-fiber overdamped mechanics with WLC/Hookean force laws. Any claims about biological fidelity require calibration against appropriate experimental data.
