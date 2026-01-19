# FibriNet Roadmap 2026
## Post-Dynamics Days Structural Improvements

**Created**: January 18, 2026
**Last Updated**: January 18, 2026
**Status**: Active Planning

---

## Mission Statement

This document tracks the major structural improvements planned for FibriNet following the successful Dynamics Days 2026 conference presentation. These goals are **fixed targets** - we add, edit, or modify approaches, but **never pivot away** from these core objectives.

---

## Task Overview

| ID | Task | Priority | Status | Target |
|----|------|----------|--------|--------|
| T1 | UI Framework Migration | High | Prototype Required | TBD |
| T2 | Single Fibrin Fiber Simulator (Hooke vs WLC) | High | **Specification Locked** | TBD |
| T3 | Distributed Strain via Affine Field + Gel-Coupling | High | Planning | TBD |
| T4 | Repository Cleanup | Medium | Manifest Required | TBD |

---

## T1: UI Framework Migration

### Problem Statement
Tkinter is not efficient for real-time scientific visualization. Network rendering becomes sluggish with larger networks and frequent updates.

### Requirements
- Efficient rendering for real-time network visualization
- Cross-platform compatibility (Windows, Linux, macOS)
- Support for scientific plotting/visualization
- Preserve existing view interface abstraction (nodes/edges primitives, pan/zoom, redraw loop)

### Options Under Consideration

| Framework | Rendering Performance | Refactor Scope | Long-term Value | Notes |
|-----------|----------------------|----------------|-----------------|-------|
| **CustomTkinter** | No improvement (still Tk canvas) | Minimal UI styling only | Low | Cosmetic upgrade only; does not solve rendering bottleneck |
| **DearPyGui** | Strong (GPU-accelerated) | Moderate (new draw calls) | Medium | Fastest path to smooth rendering; designed for scientific viz |
| **PySide6/PyQt6** | Good (QPainter/OpenGL) | Moderate-to-High | High | Best long-term app structure; dock panels, inspectors, export tooling |

### Decision Rules

| Goal | Recommended Framework |
|------|----------------------|
| **Fastest smooth rendering with least time** | DearPyGui |
| **Long-term extensible desktop app** (dock panels, inspectors, export tooling) | PySide6 |

### Working Hypothesis
**DearPyGui** - GPU-accelerated rendering specifically designed for scientific/engineering applications. This hypothesis requires validation via prototype benchmark.

### Required Before Decision
- [ ] Prototype benchmark: measure **fps vs N edges** for DearPyGui and PySide6
- [ ] Test with network sizes: 50, 200, 500, 1000 edges
- [ ] Evaluate callback migration complexity from Tkinter

### Implementation Notes
_To be added after prototype benchmark completed_

---

## T2: Single Fibrin Fiber Simulator (Hooke vs WLC)

### Problem Statement
Need a 3D single-fiber simulation tool to compare Hookean (linear) vs Worm-Like Chain (entropic) force laws for fibrin mechanics. This supports parameter extraction, model validation against Liu et al. experimental data, and future enzyme-strain coupling research.

### Specification Status: **LOCKED**

All design decisions below are finalized and should not be changed without explicit review.

---

### Physics Model

#### Unit System (Locked)
| Quantity | Unit | Notes |
|----------|------|-------|
| Length | nm | Nanometers |
| Force | pN | Piconewtons |
| Time | μs | Microseconds |
| Mass | ~10⁻¹⁹ kg | Per 100nm segment; can use effective mass |
| k_BT | 4.1 pN·nm | Room temperature (300 K) |

**Consistency requirement**: F = m·a must yield acceleration in nm/μs²

#### Dynamics Model (Locked)
- **Primary**: Deterministic quasi-static (no Brownian motion)
  - WLC formula uses k_BT as constant factor (mean-force curve)
  - No random thermal forces
  - Suitable for force-extension curve generation

- **Optional extension**: Langevin/overdamped dynamics
  - Add thermal noise + viscous drag
  - Overdamped limit: m → 0, solve γv = F_spring + F_ext + F_random
  - γ ~ 10⁻⁶ to 10⁻⁵ kg/s per node (physical viscosity)

#### Damping (Locked)
- **Overdamped model** (skip inertia or use tiny m with large γ)
- Physical justification: fibrin in aqueous solution has Reynolds number ≪ 1
- Drag coefficient: γ ~ 10⁻⁶–10⁻⁵ kg/s per node

#### Force Laws (Locked)

**Hookean Spring:**
```
F = k(L - L₀)
```
Where:
- k = EA/L_fib (spring constant from Young's modulus)
- L = current length
- L₀ = rest length

**Worm-Like Chain (Marko-Siggia):**
```
F = (k_BT / P) * [1/(4(1 - x/L₀)²) - 1/4 + x/L₀]
```
Where:
- k_BT = 4.1 pN·nm (thermal energy at 300K)
- P = persistence length (tunable, ~tens to hundreds of nm)
- x = current extension
- L₀ = contour length (maximum extension)

**Low-strain limit**: WLC reduces to Hooke with k_eff = 3k_BT/(P·L₀)

#### Rest Length vs Contour Length (Locked)
- **For deterministic mode**: L_rest = L₀ (contour length)
- **For thermal mode**: Initialize shorter than L₀ or simulate to equilibrium
- Start with deterministic: treat chain as unstrained at L = L₀ when F = 0

#### WLC Singularity Handling (Locked)
- **Strategy**: Fiber rupture at x ≥ L₀
- When extension reaches contour length, spring is **removed** (broken)
- Alternative (optional): Extensible WLC with linear backbone term for smoother transition
- **Document behavior clearly in code and UI**

---

### Architecture (Locked)

#### Code Organization
```
Fibrinet_APP/
├── src/
│   ├── core/
│   │   ├── force_laws/           # SHARED - used by both projects
│   │   │   ├── __init__.py
│   │   │   ├── hookean.py        # F = k(L - L₀)
│   │   │   ├── wlc.py            # Marko-Siggia WLC
│   │   │   └── extensible_wlc.py # Optional: WLC + linear term
│   │   └── ...
│   └── ...
├── projects/
│   └── single_fiber/
│       ├── __init__.py
│       ├── main.py               # CLI entrypoint
│       ├── fiber.py              # Fiber class (nodes, springs)
│       ├── integrator.py         # Verlet/overdamped integrator
│       ├── simulation.py         # Main simulation loop
│       ├── boundary_conditions.py
│       ├── enzyme_module.py      # Cleavage interface (stubs)
│       └── gui/                   # GUI (added later)
│           └── ...
└── ...
```

#### Dependency Direction (Enforced)
```
projects/single_fiber/* MAY import src/core/force_laws/*  ✓
src/* MUST NEVER import projects/*                        ✗
```

#### Shared Core Rationale
- WLC and Hookean force laws implemented **once** in `src/core/force_laws/`
- Both single-fiber simulator and network model use same functions
- Ensures consistency, avoids diverging implementations
- Facilitates parameter extraction: fit single-fiber → use in network

---

### Boundary Conditions (Locked)

| Mode | Description | Implementation |
|------|-------------|----------------|
| **Displacement-controlled** (Primary) | Move end node at constant velocity | Prescribed x(t) = x₀ + v·t |
| Force-controlled (Optional) | Apply constant force to end node | Add F_ext to free end |
| Interactive (Optional) | User drags end in GUI | Update target position from mouse |

**Primary mode**: Displacement-controlled (constant-velocity ramp)
- Replicates AFM/single-molecule experiments
- Gives well-defined strain history
- Force evolves as needed

---

### Enzyme Module Interface (Locked)

```python
def enzyme_check(
    spring_strains: np.ndarray,  # ε_i for each spring
    time: float,
    dt: float,
    **kwargs
) -> np.ndarray:
    """
    Compute per-spring cleavage rates.

    Returns:
        k_i: array of cleavage rates (1/μs) for each spring

    The calling code applies Gillespie or Poisson process:
        p_i = 1 - exp(-k_i * dt)  # probability of cleavage in dt
    """
    # Bell model (same as network simulation)
    k0 = kwargs.get('k0', 0.01)   # base rate
    beta = kwargs.get('beta', 1.0) # strain sensitivity

    return k0 * np.exp(-beta * spring_strains)
```

**Interface contract:**
- Input: current strains, time, timestep
- Output: per-spring cleavage rates k_i
- Mechanism: Bell model k(ε) = k₀ exp(-βε) (same as network)
- Simulation applies stochastic process using rates

---

### Validation Criteria (Locked)

Reference: Liu et al. (2010) - "The mechanical properties of single fibrin fibers"

| Property | Target Value | Tolerance | Source |
|----------|--------------|-----------|--------|
| Initial elastic modulus | ~4 MPa | ±20% | Liu et al. Table 1 |
| Breaking strain | ≥150% (2.5×) | - | Liu et al. (uncrosslinked: 330%) |
| Strain-stiffening onset | ~100% strain | ±10% | Liu et al. Fig 3 |
| Modulus increase at high strain | 2-3× initial | - | Liu et al. |

**Acceptance test**: Simulated stress-strain curve must show:
1. Initial linear region with slope ~4 MPa
2. Extension to ≥150% before failure
3. Strain-stiffening (increased slope) beyond ~100% strain

---

### GUI Strategy (Locked)

1. **Phase 1: CLI-first**
   - Implement core physics without GUI
   - Command-line interface for running simulations
   - Output data to CSV/JSON for external plotting
   - Validate against Liu et al. benchmarks

2. **Phase 2: Add GUI**
   - Use **DearPyGui** (match FibriNet main project)
   - 3D fiber visualization with node/spring rendering
   - Parameter controls (k, P, damping, velocity)
   - Real-time stress-strain plot
   - Model toggle (Hooke vs WLC)

---

### Dimensionality (Locked)

**3D implementation** (future-proofing)
- Store node positions as (N, 3) arrays
- Even for uniaxial tension, use full 3D coordinates
- Allows future extension to bending, buckling, torsion
- Better visualization in GUI

---

### Model Comparison (Locked)

| Aspect | Hookean | WLC |
|--------|---------|-----|
| Force law | F = k(L - L₀) | Marko-Siggia formula |
| Low-strain behavior | Linear | Linear (k_eff = 3k_BT/PL₀) |
| High-strain behavior | Linear (same slope) | Strain-stiffening (diverges at L₀) |
| Expected fit to fibrin | Poor at high strain | Good (matches Liu et al.) |
| Computational cost | Low | Slightly higher (nonlinear) |
| Use case | Baseline comparison | Production/publication |

**Both retained**: User can select model based on accuracy vs simplicity needs.
**Expectation**: WLC will match experimental data; Hooke serves as linear baseline.

---

### Implementation Roadmap

| Phase | Deliverable | Dependencies |
|-------|-------------|--------------|
| 1 | Shared force laws in `src/core/force_laws/` | None |
| 2 | Fiber data structure (nodes, springs) | Phase 1 |
| 3 | Integrator (Verlet/overdamped) | Phase 2 |
| 4 | CLI simulation runner | Phases 1-3 |
| 5 | Validation against Liu et al. | Phase 4 |
| 6 | Enzyme module stubs | Phase 4 |
| 7 | DearPyGui visualization | Phase 5 |
| 8 | Interactive controls | Phase 7 |

---

### Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-18 | Units: nm, pN, μs | Standard for single-molecule biophysics |
| 2026-01-18 | Deterministic primary | Simpler; Langevin optional later |
| 2026-01-18 | Overdamped dynamics | Low Reynolds number in solution |
| 2026-01-18 | Shared core architecture | Avoid code duplication with network model |
| 2026-01-18 | CLI-first, DearPyGui later | Focus on physics validation first |
| 2026-01-18 | Fiber rupture at x ≥ L₀ | Clean handling of WLC singularity |
| 2026-01-18 | Bell model enzyme interface | Consistency with network simulation |
| 2026-01-18 | 3D coordinates | Future-proofing for bending/buckling |
| 2026-01-18 | Both Hooke and WLC retained | Exploratory comparison; WLC expected to win |

---

## T3: Distributed Strain via Affine Field + Gel-Coupling (Reduces Boundary Artifact)

### Problem Statement
Current implementation applies strain only to right boundary nodes (boundary-pulling model). This creates **boundary artifacts** where strain concentrates at edges. Dr. Hudson's lab experiments use gel-embedded networks where a **global deformation field** is applied to the material.

### Scientific Background

**Key Clarification**: Gel embedding does NOT mean "equal strain on all fibers." It means a **uniform macroscopic deformation gradient** is applied to all node positions. Edge strain will still vary with:
- Fiber orientation relative to stretch direction
- Fiber rest length
- Network topology (connectivity)
- Non-affine relaxation behavior

**Current Model (Boundary Pulling)**:
```
- Left boundary nodes: FIXED
- Right boundary nodes: DISPLACED
- Interior nodes: Force equilibrium
- Result: Strain concentrates at boundaries (boundary artifact)
```

**Target Model (Global Deformation Field)**:
```
- ALL nodes: Transformed by deformation gradient F
- x' = F · x where F = [[1+ε, 0], [0, 1]] for uniaxial stretch
- Edge strain emerges from node positions (NOT enforced directly)
- Result: Uniform macroscopic strain field; heterogeneous microscopic strain
```

### Experimental Justification
In Dr. Hudson's lab experiments:
1. Fibrin network is polymerized within a gel matrix
2. The gel is stretched using mechanical apparatus
3. Gel transmits deformation field throughout the material
4. Node positions follow the macroscopic deformation
5. Individual fiber strains emerge from geometry and topology (non-uniform)

### Implementation Options

#### Option A: Affine Initialize + Relax (Fast, Minimal Code)
- Apply deformation gradient F to all node positions at initialization
- Then equilibrate with network forces (allow non-affine relaxation)
- Simple implementation; captures first-order effect
- **Limitation**: No continuous gel coupling during simulation

#### Option B: Gel-Coupling Term (Recommended, Most Faithful)
- Maintain gel-rest positions that move affinely with applied strain
- Add gel spring forces each timestep:
  ```
  f_gel,i = -k_gel * (x_i - x'_i)
  ```
  where x'_i is the affinely-transformed rest position
- Allows non-affine behavior while maintaining gel coupling
- Most faithful to "embedded in gel" physics
- Tunable: k_gel controls gel stiffness (soft gel → more non-affine)

#### Option C: Boundary-Only (Legacy Baseline)
- Keep current implementation unchanged
- Use as **comparison baseline only**
- Do NOT claim this represents gel-embedding experiments
- Label clearly as "boundary-pulling model" in all outputs

### Decision Matrix

| Criterion | Option A | Option B | Option C |
|-----------|----------|----------|----------|
| Physical accuracy | Medium | High | Low (boundary artifact) |
| Implementation complexity | Low | Medium | None (existing) |
| Non-affine behavior | Post-initialization only | Continuous | Forced at boundaries |
| Recommended use | Quick studies | Production/publication | Legacy comparison |

### Decision
- [ ] Pending review of implementation options
- [ ] Recommend: **Option B** for production, **Option C** retained for comparison

### Files to Modify
- `src/managers/network/degradation_engine/*.py` - Add gel coupling force term
- `src/core/fibrinet_core_v2_adapter.py` - Add k_gel parameter, strain field application
- Simulation initialization - Apply deformation gradient to initial positions

### Implementation Notes
_To be added after decision_

---

## T4: Repository Cleanup

### Problem Statement
Repository contains untracked files including one-off diagnostic scripts, temporary exports, and duplicate documentation that obscure the functional codebase.

### Cleanup Principles

1. **KEEP list** names explicit **entrypoints** (GUI runner, CLI runner, production scripts)
2. **DELETE** restricted to generated artifacts already covered by `.gitignore`
3. **ARCHIVE** policy: move one-off scripts to `archive/` with date + rationale
4. **`.gitignore`** is a first-class deliverable, not a checklist bullet

### File Manifest

| Path | Category | Reason | Owner | Deadline |
|------|----------|--------|-------|----------|
| `src/` | KEEP | Core source code | - | - |
| `test/` | KEEP | Test suite | - | - |
| `main.py` | KEEP | GUI entrypoint | - | - |
| `analyze_collapse_cli.py` | KEEP | CLI entrypoint | - | - |
| `run_production_clearance_study.py` | KEEP | Production job script | - | - |
| `requirements.txt` | KEEP | Dependencies | - | - |
| `README.md` | KEEP | Main documentation | - | - |
| `ROADMAP_2026.md` | KEEP | Planning document | - | - |
| `Guide for CLI/` | KEEP | User documentation | - | - |
| `.gitignore` | KEEP/UPDATE | Must cover generated artifacts | - | - |
| `exports/` | DELETE | Generated output (add to .gitignore) | - | - |
| `*.csv` (root) | DELETE | Generated results (add to .gitignore) | - | - |
| `*.png` (root) | DELETE | Generated figures (add to .gitignore) | - | - |
| `nul` | DELETE | Windows artifact | - | - |
| `publication_figures/` | REVIEW | May archive or .gitignore | - | - |
| `validation_results/` | REVIEW | May archive or .gitignore | - | - |
| `diagnostic_*.py` | ARCHIVE | One-off analysis scripts | - | - |
| `generate_*.py` | ARCHIVE | One-off generation scripts | - | - |
| `run_*.py` (except production) | ARCHIVE | One-off job scripts | - | - |
| `test_*.py` (root, not test/) | ARCHIVE | One-off test scripts | - | - |
| `prove_*.py` | ARCHIVE | One-off validation scripts | - | - |
| `*.md` (excess root docs) | REVIEW | Consolidate or archive | - | - |

### .gitignore Deliverable

Required additions to `.gitignore`:
```gitignore
# Generated outputs
exports/
publication_figures/
validation_results/

# Result files
*.csv
!requirements.txt

# Generated images (root only)
/*.png

# Windows artifacts
nul

# Archive (tracked separately)
archive/

# Python
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# IDE
.vscode/
.idea/
*.swp
```

### Archive Policy

Files moved to `archive/` must include:
```
archive/
├── YYYY-MM-DD_<descriptive_name>/
│   ├── README.md  (rationale for archiving)
│   └── <archived files>
```

### Cleanup Checklist
- [ ] Run `git status --porcelain | wc -l` to get exact untracked count
- [ ] Complete manifest table with all files
- [ ] Get user approval on categorization
- [ ] Update `.gitignore` FIRST
- [ ] Create `archive/` structure
- [ ] Move ARCHIVE files with README rationale
- [ ] Delete DELETE files
- [ ] Commit clean state
- [ ] Verify `git status` shows clean working tree

### Decision
- [ ] Manifest completion pending
- [ ] User approval required before execution

---

## Progress Log

### January 18, 2026
- Created ROADMAP_2026.md
- Documented all four major tasks
- **REVISION**: Corrected T3 physics language (uniform deformation field, not equal fiber strain)
- **REVISION**: Fixed T1 ranking to reflect actual refactor scope vs rendering performance
- **REVISION**: Added structural constraints to T2 to prevent scope creep
- **REVISION**: Made T4 executable with manifest table and .gitignore deliverable
- **T2 LOCKED**: Full specification finalized with all 12 design decisions documented
  - Units: nm, pN, μs, k_BT = 4.1 pN·nm
  - Dynamics: Deterministic quasi-static (Langevin optional)
  - Damping: Overdamped (low Reynolds number)
  - Architecture: Shared core in src/core/force_laws/
  - GUI: CLI-first, then DearPyGui
  - WLC singularity: Fiber rupture at x ≥ L₀
  - Enzyme interface: Bell model rates k(ε) = k₀ exp(-βε)
  - Validation: Liu et al. benchmarks (~4 MPa, ≥150% extensibility)

---

## Guidelines

### Do's
- Add new tasks as they arise
- Update status regularly
- Document decisions and rationale
- Keep implementation notes current
- Validate hypotheses with prototypes before committing

### Don'ts
- **NEVER pivot away from core objectives**
- Don't delete tasks - mark as "Deferred" or "Cancelled" with reason
- Don't start implementation without documented decision
- Don't confuse macroscopic deformation field with microscopic fiber strain

---

## Glossary

| Term | Definition |
|------|------------|
| **Affine deformation** | Transformation where x' = F·x for all points; straight lines remain straight |
| **Non-affine behavior** | Local deviations from affine due to network topology and force equilibrium |
| **Deformation gradient (F)** | Tensor mapping reference positions to deformed positions |
| **Macroscopic strain** | Average strain over the entire material (uniform field) |
| **Microscopic strain** | Local strain on individual fibers (heterogeneous, emerges from geometry) |
| **Gel-coupling** | Spring forces connecting network nodes to affinely-moving gel positions |
| **Boundary artifact** | Non-physical strain concentration at fixed/displaced boundaries |
| **WLC** | Worm-Like Chain - entropic polymer model with persistence length |
| **Marko-Siggia** | Interpolation formula for WLC force-extension relation |
| **Contour length (L₀)** | Maximum extension of WLC polymer (inextensible backbone) |
| **Persistence length (P)** | Characteristic bending stiffness length scale of polymer |
| **Bell model** | Strain-dependent rate: k(ε) = k₀ exp(-βε) |
| **Overdamped** | Dynamics regime where inertia is negligible (low Reynolds number) |

---

## Next Steps

1. **T2**: Begin Phase 1 - Implement shared force laws in `src/core/force_laws/`
2. **T1**: Build prototype benchmark (fps vs N edges) for DearPyGui and PySide6
3. **T3**: Review Option A vs Option B implementation; recommend Option B
4. **T4**: Run `git status --porcelain`, complete manifest, get approval

---

_This document is the single source of truth for FibriNet 2026 development goals._
