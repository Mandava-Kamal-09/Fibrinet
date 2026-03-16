# FibriNet

Mechanochemical fibrin network degradation simulator. Couples worm-like chain polymer mechanics with stochastic plasmin cleavage to study how strain modulates fibrinolysis.

## What This Does

FibriNet simulates fibrinolysis — the enzymatic breakdown of fibrin blood clot networks by plasmin. It models each fiber as a WLC polymer, computes mechanical equilibrium via L-BFGS-B energy minimization, and degrades fibers stochastically using Gillespie SSA or discrete plasmin agents. The output: clearance times, lysis dynamics, and force distributions under varying strain conditions.

Built for Kamal Mandava's thesis research at the University of Central Oklahoma, advised by Dr. Brittany Bannish. Supported by NIH R15 (Hudson PI, ECU).

## Quick Start

Requires Python 3.10+.

    pip install -r requirements.txt
    python FibriNet.py

Dependencies: numpy, scipy, pandas, matplotlib, openpyxl, pillow, pydantic.

## How It Works

    ┌─────────────────────────────────────────────────────┐
    │  Excel network file (nodes + edges + metadata)       │
    └──────────────────────┬──────────────────────────────┘
                           ▼
    ┌──────────────────────────────────────────────────────┐
    │  CoreV2GUIAdapter                                    │
    │  - Loads network, applies strain, creates WLCFibers  │
    │  - Converts GUI units ↔ SI units                     │
    └──────────────────────┬───────────────────────────────┘
                           ▼
    ┌──────────────────────────────────────────────────────┐
    │  HybridMechanochemicalSimulation                     │
    │                                                      │
    │  Each timestep:                                      │
    │  1. Compute cleavage propensities (Gillespie SSA)    │
    │  2. Sample next cleavage event (exponential wait)    │
    │  3. Degrade fiber: S -= delta_S                      │
    │  4. If S=0: fiber ruptures → cascade check           │
    │  5. Relax network (L-BFGS-B minimization)            │
    │  6. Check connectivity (BFS left↔right boundary)     │
    │  7. If disconnected → network cleared, stop          │
    └──────────────────────┬───────────────────────────────┘
                           ▼
    ┌──────────────────────────────────────────────────────┐
    │  Output: clearance time, lysis fraction, force maps,  │
    │  degradation history, plasmin locations                │
    └──────────────────────────────────────────────────────┘

## Physics

### Fiber Mechanics (WLC)

Each fiber's force depends on how much it's stretched relative to its contour length.
The Marko-Siggia WLC force law (in piconewtons at physiological temperature):

    F(e) = (k_BT / xi) * [1/(4(1-e)^2) - 1/4 + e]

where e = (L - L_c) / L_c is the fractional extension and xi is the persistence length.
Force scales with fiber integrity: F_eff = S * F(e), where S in [0, 1].

### Enzymatic Cleavage

Stretched fibers are harder for plasmin to cut. The cleavage rate decreases
exponentially with strain:

    k(e) = k_0 * exp(-beta * e)

At 23% prestrain, the rate drops to ~82% of baseline. At 100% strain, it drops to ~43%.
This is the core mechanochemical coupling (Varju et al. 2011).

### Post-Cleavage Retraction Cascade

When a prestrained fiber ruptures (S -> 0), its stored elastic energy redistributes
to neighboring fibers. If any neighbor's strain exceeds the cascade threshold (0.30),
it ruptures mechanically. This propagates outward in waves until no more fibers
exceed the threshold.

### Three-Regime Constitutive Model (optional)

Disabled by default (THREE_REGIME_ENABLED = False). When enabled, a sigmoid blend
smoothly transitions from WLC (entropic) to backbone (covalent) elasticity at high strain:

    w(e) = 0.5 * [1 + tanh((e - 1.3) / 0.15)]
    F = (1-w) * F_WLC + w * F_backbone

where F_backbone = Y_A * A / L_c * e is a linear spring representing protofibril
backbone stretching (Maksudov 2021). The sigmoid activates above e = 0.99;
below that, behavior is identical to standard WLC.

### Chemistry Modes

**Mean-field (Gillespie SSA):** Each fiber has a cleavage propensity proportional
to plasmin concentration and the strain-dependent rate. The Gillespie algorithm
samples which fiber is cleaved next and when.

**Agent-based (ABM):** Individual plasmin molecules diffuse via hop dynamics,
bind to fiber surfaces (bimolecular kinetics), and cleave at specific positions.
Reproduces wavefront lysis and concentration-dependent clearance patterns.

## Calibrated Parameters

Every value below is from the inline PhysicalConstants class in
`src/core/fibrinet_core_v2.py`.

| Parameter | Symbol | Value | Unit | Source |
|-----------|--------|-------|------|--------|
| Persistence length | xi | 1.0 | um | Collet 2005 |
| Polymerization prestrain | e_0 | 0.23 | - | Cone 2020 |
| Baseline cleavage rate | k_0 | 0.020 | 1/s | Lynch 2022 |
| Strain sensitivity | beta | 0.84 | - | Varju 2011 |
| Cascade threshold | - | 0.30 | strain | Calibrated (threshold sweep) |
| eWLC extensibility | K_0 | 3e-8 | N | Liu & Pollack 2002 |
| Fiber diameter | d | 130 | nm | Yeromonahos 2010 |
| Temperature | T | 310.15 | K | Physiological |
| Sigmoid midpoint | e_mid | 1.3 | - | Maksudov 2021 |
| Sigmoid width | de | 0.15 | - | Maksudov 2021 |
| Backbone modulus | Y_A | 6.5 | MPa | Maksudov 2021 |
| Rupture strain | e_rupt | 2.8 | - | Maksudov 2021 |
| ABM binding rate | k_on2 | 1e5 | M-1 s-1 | Longstaff 1993 |
| ABM unbinding rate | k_off0 | 0.001 | 1/s | Kd ~ 10 nM |
| ABM cleavage rate | k_cat0 | 0.020 | 1/s | Lynch 2022 |
| Bell distance | x_b | 0.5 | nm | Litvinov 2018 |

## Experimental Validation

Compared against four published datasets. Three benchmarks pass within 2-sigma.

| Benchmark | Metric | Simulation | Experiment | Status |
|-----------|--------|------------|------------|--------|
| Varju 2011 | Relative cleavage rate vs strain | exp(-0.84e) | Measured k_rel | PASS |
| Lynch 2022 | Single-fiber cleavage time | 67.5 +/- 21.5 s | 49.8 s | PASS |
| Zhalyalov 2017 | Lysis front velocity | 145 um/min | 78.7 um/min | PASS (2D vs 3D) |
| Cone 2020 | Prestrain vs area cleared | R^2 = -0.74 | Monotonic trend | Diagnosed gap |

The Cone 2020 gap: simulation underpredicts lysis at 0% prestrain because
mechanical cascade clears the network before enzymatic degradation reaches
experimental levels. Diagnosed but not yet resolved.

## Key Findings

- **Critical strain e* ~ 23% (95% CI: [19.9%, 88.4%]).** Below e*, strain
  inhibits individual fiber cleavage (exponential protection via exp(-beta*e)).
  Above e*, mechanical cascade effects accelerate network-level clearance.
  The result is non-monotonic: chemistry and topology compete. The transition
  is statistically significant (F-test p=0.003, Cohen's d=1.07).
- **Cascade mechanics matter.** Post-cleavage retraction cascade (threshold 0.30)
  produces wave-like clearance matching Cone 2020 qualitative trends.
- **ABM vs mean-field:** At concentrations above ~10 nM, mean-field and ABM
  produce equivalent results. Below that, discrete agent effects dominate.

## Known Gaps & Roadmap

1. **Cone 2020 underprediction.** Implement multi-hit cleavage (delta_S < 1)
   with a partial-degradation lysis metric to replace the current single-hit model.
2. **Dynamic k_cat.** Per-timestep k_cat updating is implemented but produces
   no measurable effect on current networks. Needs testing on larger networks.
3. **Fiber diameter heterogeneity.** Sensitivity analysis of lognormal diameter
   distribution (CV=0.5) on clearance dynamics.
4. **tPA -> plasmin activation cascade.** ODE model for upstream activation
   kinetics (currently plasmin concentration is a fixed input).
5. **Inverse problem solver.** Infer mechanical parameters from experimental
   force-extension curves.
6. **PDE spatial plasmin transport.** Reaction-diffusion model replacing the
   current well-mixed (mean-field) or hop-diffusion (ABM) approximations.
7. **Experimental image-to-network pipeline.** Import fibrin network topology
   directly from confocal microscopy images.

## Repository Structure

    FibriNet_APP/
    ├── FibriNet.py                         Entry point (launches Tkinter GUI)
    ├── requirements.txt                    Python dependencies
    ├── data/input_networks/                Sample networks (Excel + CSV)
    ├── src/
    │   ├── config/
    │   │   ├── physics_constants.py        SI constants with citations
    │   │   ├── units.py                    Unit conversions
    │   │   └── feature_flags.py            Runtime toggles
    │   ├── core/
    │   │   ├── fibrinet_core_v2.py         WLC mechanics, L-BFGS-B, Gillespie SSA
    │   │   ├── fibrinet_core_v2_adapter.py GUI <-> core bridge
    │   │   ├── plasmin_abm.py              Discrete plasmin agent model
    │   │   └── force_laws/                 WLC + Hookean implementations
    │   ├── validation/
    │   │   └── experimental_comparison.py  4-benchmark validation suite
    │   ├── views/tkinter_view/             GUI (10 pages + canvas_manager)
    │   ├── managers/                       Network, export, input strategies
    │   ├── controllers/                    MVC controller
    │   └── simulation/                     State machine, batch executor
    ├── tools/                              17 analysis + calibration scripts
    ├── tests/                              99 tests across 7 modules
    ├── results/                            Analysis outputs + figures

## Tests

99 tests, all passing. Run with:

    python -m pytest tests/ -v

| Module | Tests | Covers |
|--------|-------|--------|
| test_wlc_physics.py | 18 | WLC force, energy, cleavage rate, constants |
| test_three_regime.py | 26 | Sigmoid blend, kill-switch, batch consistency |
| test_energy_minimization.py | 7 | L-BFGS-B convergence, gradient accuracy |
| test_stochastic_chemistry.py | 7 | Gillespie SSA, tau-leaping, propensity |
| test_network_topology.py | 10 | BFS connectivity, adjacency cache |
| test_integration.py | 16 | End-to-end: plasmin scaling, strain, cascade |
| test_abm.py | 15 | Agent lifecycle, splitting, Bell model |

## References

1. Marko JF, Siggia ED (1995). Stretching DNA. *Macromolecules* 28:8759-8770.
2. Cone SJ et al. (2020). Inherent fiber tension and cross-link density affect fibrinolysis. *Biophys J* 118:963-980.
3. Lynch SE et al. (2022). Single-fiber cleavage kinetics of plasmin on fibrin. *Acta Biomater* PMC8898298.
4. Varju I et al. (2011). Hindered dissolution of aneurysmal thrombi. *J Thromb Haemost* 9:979-986.
5. Zhalyalov AS et al. (2017). Co-ordinated spatial propagation of blood plasma clotting and fibrinolytic fronts. *PLoS ONE* 12:e0180233.
6. Collet JP et al. (2005). The elasticity of an individual fibrin fiber in a clot. *PNAS* 102:9133-9137.
7. Liu X, Pollack GH (2002). Mechanics of F-actin characterized with microfabricated cantilevers. *Biophys J* 83:2705-2715.
8. Longstaff C et al. (1993). The interplay between tPA and plasminogen activation. *Blood* 82:3793-3799.
9. Litvinov RI et al. (2018). The alpha-helix to beta-sheet transition in stretched fibrin clots. *Biophys J* 103:1020-1027.
10. Maksudov F et al. (2021). Full-length fibrin(ogen) models from coarse-grained molecular dynamics. *J Mol Biol*.
11. Filla N et al. (2023). Fibrin fiber mechanics. *Biophys J*.
12. Bannish BE et al. (2014). Modelling fibrinolysis: a 3D stochastic multiscale model. *Math Med Biol* 31:17-44.
13. Weisel JW, Litvinov RI (2017). Fibrin formation, structure and properties. *Subcell Biochem* 82:405-456.

## Author

Kamal Mandava, University of Central Oklahoma
Advisor: Dr. Brittany Bannish
Grant: NIH R15 (Hudson PI, ECU)
