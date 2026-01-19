# FibriNet Research Simulation Tool: Functional Documentation

**Document Type:** Functional Truth Record
**Source:** `src/views/tkinter_view/research_simulation_page.py`
**Purpose:** Complete specification of implemented behavior for scientific reproducibility

---

## 1. Lay-Level Overview (What the Tool Does)

FibriNet Research Simulation is a 2D graph-based mechanical degradation simulator for fibrin networks under uniaxial tensile strain. The tool models network evolution under fixed strain via discrete batch updates that couple mechanics and chemistry.

### Physical System Modeled

A planar network of elastic fibers (edges) connecting nodes, with:
- Rigid boundary conditions (left and right grips at fixed x-coordinates)
- Linear spring mechanics (Hookean fibers)
- Discrete protofibril structure within each fiber (spatial mode only)
- Force-dependent and topology-dependent degradation kinetics

### User Inputs

1. **Network geometry** (CSV or XLSX):
   - Node positions (n_x, n_y)
   - Edge connectivity (n_from, n_to)
   - Fiber properties (thickness, rest_length)
   - Boundary flags (is_left_boundary, is_right_boundary)

2. **Simulation parameters** (UI or frozen in checkpoint):
   - Applied strain (epsilon): fixed after Start
   - Timestep (dt): batch duration
   - Degradation rate (lambda_0): baseline cleavage rate
   - Force response parameters (alpha, F0, Hill coefficient n)
   - Thickness scaling exponent (alpha_thickness)
   - Plasmin pool size (spatial mode: P_total_quanta, N_pf)
   - Gate parameters (rupture, cooperativity, shielding, memory, anisotropy)

### Per-Batch Processes (Spatial Mode with `USE_SPATIAL_PLASMIN=True`)

Each batch execution performs the following steps in strict order:

1. **Force state usage**: Use cached post-relaxation forces from previous batch
2. **Unbinding kinetics**: Stochastic unbinding of bound plasmin (Binomial sampling)
3. **Binding kinetics**: Stochastic supply-limited binding of free plasmin (Poisson sampling, weighted by available sites)
4. **Cleavage kinetics**: Deterministic explicit Euler update of intact protofibrils (dn/dt = -k_cat(T) × B_i)
5. **Stiffness update**: Compute per-edge weakest-link stiffness (S = min(n_i/N_pf))
6. **Fracture detection**: Identify and remove edges where min(n_i/N_pf) ≤ n_crit_fraction
7. **Mechanical relaxation**: Single conjugate gradient relaxation to new equilibrium geometry
8. **Logging**: Append batch observables to experiment log

### Per-Batch Processes (Legacy Mode with `USE_SPATIAL_PLASMIN=False`)

1. **Force state usage**: Use cached post-relaxation forces from previous batch
2. **Memory update**: M_i ← (1-μ)M_i + μ max(F_i, 0)
3. **Gate computation**: Compute force response (Hill), strain-rate, rupture, energy, cooperativity, shielding, memory, anisotropy gates
4. **Plasmin selection**: Weighted stochastic selection of N_plasmin edges for degradation
5. **Scalar degradation**: S_i ← S_i - λ_eff × g_total × dt
6. **Plastic rest-length update**: L_rest ← L_rest + rate × (F - F_threshold) × dt (if F > threshold)
7. **Mechanical relaxation**: Single relaxation to new equilibrium geometry
8. **Logging**: Append batch observables to experiment log

### Network Clearance Criterion

**Spatial mode**: Loss of left-to-right percolation connectivity (graph disconnection via BFS after edge removal)

**Legacy mode**: Median tension (sigma_ref) becomes non-positive or non-finite (network slack/collapse)

### Questions the Tool Is Designed to Answer

1. How does network topology evolve under fixed tensile strain?
2. What is the time-to-failure (global lysis time) for a given strain and parameter set?
3. How do mechanical forces redistribute after localized fiber cleavage?
4. What is the sensitivity of network lifetime to plasmin pool size, binding rates, or fiber thickness distribution?
5. How does segment-level protofibril damage correlate with edge-level stiffness loss?

---

## 2. Key Assumptions and Limitations

### Spatial Dimensionality
- **2D planar network**: All nodes constrained to (x,y) plane (research_simulation_page.py:2286-2300)
- **No z-coordinate**: Out-of-plane mechanics not represented

### Fiber Representation
- **Graph edges**: Fibers modeled as 1D line segments connecting node pairs (research_simulation_page.py:462-516)
- **Linear springs**: Hookean force law F = k_eff × (L - L_rest) (research_simulation_page.py:2486-2496)
- **No bending stiffness**: Edges have only axial resistance
- **No tortuosity**: Fibers are straight lines in relaxed state

### Protofibril Discretization (Spatial Mode Only)
- **Uniform N_pf**: All fibers initialized with identical protofibril count (research_simulation_page.py:4418)
- **Discrete segments**: Each fiber subdivided into uniform-length segments (research_simulation_page.py:437-460)
- **Continuous n_i**: Intact protofibril count evolved via explicit Euler (research_simulation_page.py:4433-4435), not integer stochastic cleavage
- **Binding sites S_i**: Proportional to segment surface area (implicit effective volume calculation)

### Damage Representation
- **Weakest-link stiffness**: Per-edge S = min(n_i/N_pf) over all segments (research_simulation_page.py:4468-4470)
- **No damage healing**: n_i never increases; L_rest never decreases (research_simulation_page.py:4584-4586)
- **No rebinding after fracture**: Removed edges (min(n_i/N_pf) ≤ n_crit) are permanently deleted from topology (research_simulation_page.py:4533-4535)

### Binding/Cleavage Stochasticity
- **Stochastic unbinding**: Binomial sampling per segment (p_unbind = 1 - exp(-k_off(T) × dt)) (research_simulation_page.py:4263-4275)
- **Stochastic binding**: Poisson arrival (λ_bind_total × dt), weighted sampling without replacement (research_simulation_page.py:4303-4392)
- **Deterministic cleavage**: Explicit Euler integration (no stochastic single-protofibril severing) (research_simulation_page.py:4433-4435)
- **Frozen RNG state**: Reproducibility via SHA-256-derived per-batch seeds (research_simulation_page.py:4130-4134)

### Mechanical Relaxation Timing
- **Post-degradation relaxation only**: One conjugate gradient solve per batch, after all chemistry updates (research_simulation_page.py:4806, 4538)
- **No pre-degradation relax**: Force state from previous batch is used for gate computations (research_simulation_page.py:3906-3913)
- **No sub-batch relaxation**: Geometry frozen during multi-step chemistry (unbind → bind → cleave)

### Termination Criterion
- **Spatial mode**: Percolation failure (BFS disconnection check) (research_simulation_page.py:5069-5093)
- **Legacy mode**: sigma_ref ≤ 0 or non-finite (research_simulation_page.py:3937-4069)
- **No strain-driven termination**: Simulation does not stop at fixed lysis fraction threshold unless percolation/sigma_ref fails

### Hard Limitations
- **No 3D volume**: Surface area / volume calculations are implicit effective values (no explicit geometry)
- **No rebinding**: Once plasmin unbinds, it returns to free pool but lost binding site history is not tracked
- **No fiber healing**: n_i, L_rest, S, M are monotonic or non-decreasing (never reversed)
- **No nonlinear constitutive law**: Stiffness scaling is linear in S (damage softening only) and power-law in thickness (k ∝ (t/t_ref)^α)
- **No explicit solvent**: Plasmin diffusion not modeled; binding is spatially uniform weighted sampling
- **No plasmin degradation**: P_total_quanta conserved; no consumption or replenishment
- **No fiber branching**: Topology is fixed graph (node/edge set changes only via edge removal)
- **No cross-linking dynamics**: Edges do not form or merge during simulation

---

## 3. Technical Overview

### 3.1 Required Background Knowledge

**Graph-based fiber networks**: Nodes (2D points) connected by edges (fibers). Topology defined by adjacency relationships; connectivity determines load-bearing capacity.

**Linear spring mechanics**: Each edge exerts force F = k × (L - L_rest), where k is stiffness, L is current length, L_rest is rest length. Network equilibrium found by minimizing total elastic energy via conjugate gradient.

**Stochastic binding kinetics**: Plasmin binding events sampled from Poisson distribution (arrival rate λ_bind_total × dt). Target segments chosen via weighted random sampling (weight = available sites S_i - B_i).

**Force-dependent reaction rates**: Cleavage rate k_cat(T) = k_cat0 × exp(β × T) increases exponentially with fiber tension T. Unbinding rate k_off(T) = k_off0 × exp(-α × T) decreases exponentially with tension (catch-bond-like).

**Percolation-based failure**: Network fails when no continuous path of intact edges exists from left boundary to right boundary (BFS reachability test).

---

### 3.2 Architecture and Execution Flow

#### Input Parsing
1. **File type detection**: `.csv` → `_parse_delimited_tables_from_csv`; `.xlsx` → `_parse_delimited_tables_from_xlsx` (research_simulation_page.py:30-229)
2. **Table extraction**: Expect 3 tables (nodes, edges, meta_data) separated by blank lines or in dedicated sheets
3. **Column normalization**: Strip whitespace, lowercase, replace spaces with underscores (research_simulation_page.py:232-233)
4. **Required columns**:
   - Nodes: one of {n_id, node_id, id}, n_x, n_y, is_left_boundary, is_right_boundary
   - Edges: one of {e_id, edge_id, id}, one of {n_from, from, source}, one of {n_to, to, target}, thickness
5. **Boundary flag coercion**: Accept True/False, 1/0, "1"/"0" (research_simulation_page.py:271-299)
6. **Edge initialization**: Compute rest_length = euclidean(n_from, n_to) from imported node positions
7. **Segment initialization (spatial mode)**: Subdivide each edge into N_segments uniform-length segments; compute S_i from segment length and fiber thickness; initialize n_i = N_pf, B_i = 0

#### Per-Batch Update Order (Spatial Mode)

```
batch_index = len(experiment_log)
time_start = current_time

# --- PRE-BATCH STATE ---
intact_edges = [e for e in edges if S(e) > 0]
forces_cached = {e.edge_id: F_from_previous_relax}

# STEP 1: Compute batch-level observables
mean_tension = mean([max(0, F) for F in forces_cached])
sigma_ref = median([max(0, F) for F in forces_cached])

# STEP 2: Plasmin selection (weighted stochastic; uses sigma_ref)
attack_weights = {e: (F/sigma_ref)^beta * (t_ref/t_e)^gamma}
if N_plasmin < len(intact_edges):
    selected_edges = weighted_sample_without_replacement(N_plasmin)
else:
    selected_edges = all_intact_edges

# --- CHEMISTRY UPDATE (uses dt_used, not dt) ---
dt_used = dt
dt_cleave_rates = [k_cat(T_e) * S_i for all segments]
if dt_cleave_rates:
    dt_cleave_safe = 0.1 / max(dt_cleave_rates)
    dt_used = min(dt, dt_cleave_safe)

# STEP 3: Unbinding (stochastic, per-segment)
for edge in edges:
    T_edge = forces_cached[edge.edge_id]
    k_off = k_off0 * exp(-alpha * T_edge)
    p_unbind = 1 - exp(-k_off * dt_used)
    for seg in edge.segments:
        U_i ~ Binomial(B_i, p_unbind)
        B_i -= U_i
        P_free_quanta += U_i

# STEP 4: Binding (stochastic, supply-limited, weighted sampling)
N_bind_events ~ Poisson(lambda_bind_total * dt_used)
N_bind_events = min(N_bind_events, P_free_quanta)
segment_weights = [(e_idx, s_idx, S_i - B_i) for all segments if available > 0]
for _ in range(N_bind_events):
    (e_idx, s_idx) = weighted_random_choice(segment_weights)
    B_i += 1
    P_free_quanta -= 1
    update segment_weights

# STEP 5: Conservation check
assert P_free_quanta + sum(B_i) == P_total_quanta

# STEP 6: Cleavage (deterministic explicit Euler)
for edge in edges:
    T_edge = forces_cached[edge.edge_id]
    k_cat = k_cat0 * exp(beta_cleave * T_edge)
    for seg in edge.segments:
        dn_i = -k_cat * B_i * dt_used
        n_i_new = clamp(n_i + dn_i, 0, N_pf)

# STEP 7: Stiffness update (weakest-link)
for edge in edges:
    S_new = min([n_i / N_pf for seg in edge.segments])

# STEP 8: Fracture detection and removal
fractured_edges = [e for e in edges if min(n_i/N_pf) <= n_crit_fraction]
for e in fractured_edges:
    archive edge to fractured_history (with segments, tension_at_failure, strain_at_failure)
    remove e from edges

# STEP 9: Force redistribution (relax once after edge removal)
relax(applied_strain) → updates forces_cached, relaxed_node_coords

# STEP 10: Percolation check (termination criterion)
is_connected = BFS_reachability(left_boundary, right_boundary, edges)
if not is_connected:
    termination_reason = "network_percolation_failure"
    stop simulation

# --- POST-BATCH STATE ---
time_end = time_start + dt_used
log_entry = {
    batch_index, time_end, strain, sigma_ref,
    plasmin_selected_edge_ids, newly_lysed_edge_ids,
    intact_edges, mean_tension, lysis_fraction,
    dt_used, n_min_frac, n_mean_frac, total_bound_plasmin,
    P_free_quanta, bind_events_applied, total_unbound_this_batch,
    batch_hash, provenance_hash, rng_state_hash, params
}
experiment_log.append(log_entry)
```

#### Per-Batch Update Order (Legacy Mode)

```
batch_index = len(experiment_log)
time_start = current_time

# --- PRE-BATCH STATE ---
intact_edges = [e for e in edges if S(e) > 0]
forces_cached = {e.edge_id: F_from_previous_relax}

# STEP 1: Compute batch-level observables
mean_tension = mean([max(0, F) for F in forces_cached])
sigma_ref = median([max(0, F) for F in forces_cached])
if sigma_ref <= 0 or not finite:
    termination_reason = "network_lost_load_bearing_capacity"
    stop simulation

# STEP 2: Plasmin selection (weighted stochastic)
attack_weights = {e: (F/sigma_ref)^beta * (t_ref/t_e)^gamma}
if N_plasmin < len(intact_edges):
    selected_edges = weighted_sample_without_replacement(N_plasmin)
else:
    selected_edges = all_intact_edges

# STEP 3: Memory update (pre-degradation)
for edge in intact_edges:
    M_i = (1 - mu) * M_i + mu * max(F_i, 0)

# STEP 4: Strain-rate factor (uses mean_tension from this batch and prev_mean_tension from last batch)
if prev_mean_tension is not None:
    dF = mean_tension - prev_mean_tension
    strain_rate = dF / dt
    strain_rate_factor = 1 + beta_rate * tanh(strain_rate / eps0)
else:
    strain_rate_factor = 1.0

# STEP 5: Topology map (for cooperativity gate)
build node_to_edge_ids map for neighborhood lookup

# STEP 6: Per-edge degradation update
for edge in intact_edges:
    F = forces_cached[edge.edge_id]

    # Plastic rest-length update
    if F > plastic_F_threshold:
        L_rest_effective += plastic_rate * (F - plastic_F_threshold) * dt

    # Gate computation
    gF = 1 + force_alpha * (F^n / (F^n + F0^n))                      # Hill force response
    rF = 1 + rupture_gamma * max(0, F - rupture_F_threshold)         # Rupture amplification
    eG = 1 + fracture_eta * max(0, E_i - fracture_Gc)                 # Energy gate
    cG = 1 + coop_chi * mean_neighbor_damage                          # Cooperativity
    sG = clamp(F_tension / (mean_tension + shield_eps), 0, 1)         # Shielding
    mG = 1 + memory_rho * M_i                                          # Memory
    aG = 1 + aniso_kappa * |cos(theta)|                                # Anisotropy

    g_total = gF * strain_rate_factor * rF * eG * cG * sG * mG * aG
    g_total = min(g_total, g_max)

    # Effective degradation rate (only for selected edges)
    if edge in selected_edges:
        lambda_eff = lambda_0 * (F/sigma_ref)^beta * (t_ref/t_edge)^gamma
    else:
        lambda_eff = 0

    # Scalar S update
    S_new = S - lambda_eff * g_total * dt
    S_new = clamp(S_new, 0, 1)

    # Lysis tracking (set once when S crosses 0)
    if S_old > 0 and S_new <= 0:
        lysis_batch_index = batch_index
        lysis_time = time_start + dt

# STEP 7: Cleavage density fail-safe
newly_cleaved = count(S_old > 0 and S_new <= 0)
if newly_cleaved / len(intact_edges) > cleavage_batch_cap:
    abort batch (raise ValueError)

# STEP 8: Mechanical relaxation (once after all chemistry)
relax(applied_strain) → updates forces_cached, relaxed_node_coords

# STEP 9: Post-relax observables
post_mean_tension = mean([max(0, F) for F in forces_cached if S > 0])
lysis_fraction = 1 - sum(k0 * S) / sum(k0)

# STEP 10: Global lysis time (set once)
if lysis_fraction >= global_lysis_threshold and global_lysis_batch_index is None:
    global_lysis_batch_index = batch_index
    global_lysis_time = time_start + dt

# --- POST-BATCH STATE ---
prev_mean_tension = post_mean_tension
time_end = time_start + dt
log_entry = {batch_index, time_end, strain, ...}
experiment_log.append(log_entry)
```

---

### 3.3 Governing Equations (AS IMPLEMENTED)

#### Effective Stiffness (Spatial Mode)

```
k_eff = k0 × N_pf × S × (thickness / thickness_ref)^α
```

Where:
- `k0`: Baseline stiffness per protofibril (input parameter)
- `N_pf`: Total protofibril count per fiber (frozen at Start)
- `S`: Weakest-link integrity = min(n_i / N_pf) over all segments
- `thickness`: Per-edge imported thickness (immutable experimental data)
- `thickness_ref`: Reference thickness (median over all edges, frozen at Start)
- `α`: Thickness scaling exponent (frozen at Start)

**Code location**: research_simulation_page.py:2581-2601

#### Effective Stiffness (Legacy Mode)

```
k_eff = k0 × S × (thickness / thickness_ref)^α
```

Where S is scalar integrity (not derived from protofibrils).

**Code location**: research_simulation_page.py:2581-2601

#### Damage Fraction (Spatial Mode Only)

```
S = min_i (n_i / N_pf)
```

Where:
- `n_i`: Intact protofibril count in segment i (continuous, updated via cleavage kinetics)
- `N_pf`: Total protofibril count (constant)

**Code location**: research_simulation_page.py:4468-4470

#### Cleavage Kinetics (Spatial Mode Only)

```
dn_i/dt = -k_cat(T) × B_i
k_cat(T) = k_cat0 × exp(β × T)
```

Where:
- `n_i`: Intact protofibril count in segment i
- `k_cat(T)`: Force-dependent cleavage rate
- `k_cat0`: Baseline cleavage rate (frozen at Start, from spatial_plasmin_params)
- `β`: Force sensitivity exponent (frozen at Start, from spatial_plasmin_params)
- `T`: Fiber tension (cached from previous relax)
- `B_i`: Bound plasmin count on segment i

**Numerical integration**: Explicit Euler with adaptive timestep:
```
dt_cleave_safe = 0.1 / max(k_cat(T) × S_i)
dt_used = min(dt, dt_cleave_safe)
n_i_new = n_i + (-k_cat(T) × B_i) × dt_used
n_i_new = clamp(n_i_new, 0, N_pf)
```

**Code location**: research_simulation_page.py:4234-4249, 4418-4448

#### Unbinding Kinetics (Spatial Mode Only)

```
k_off(T) = k_off0 × exp(-α × T)
p_unbind = 1 - exp(-k_off(T) × dt_used)
U_i ~ Binomial(B_i, p_unbind)
B_i_new = B_i - U_i
P_free_quanta += U_i
```

Where:
- `k_off(T)`: Force-dependent unbinding rate (catch-bond-like: higher tension → lower unbinding)
- `k_off0`: Baseline unbinding rate (frozen at Start)
- `α`: Force sensitivity exponent (frozen at Start)
- `T`: Fiber tension (cached)
- `U_i`: Stochastic unbinding event count (Binomial sample)
- `P_free_quanta`: Free plasmin pool (conserved quantity)

**Code location**: research_simulation_page.py:4251-4296

#### Binding Kinetics (Spatial Mode Only)

```
N_bind_events ~ Poisson(λ_bind_total × dt_used)
N_bind_events = min(N_bind_events, P_free_quanta)

Segment selection: weighted sampling without replacement
weight_i = S_i - B_i (available sites)
```

Where:
- `λ_bind_total`: Total binding rate (frozen at Start)
- `P_free_quanta`: Free plasmin pool (supply-limited)
- `S_i`: Maximum binding site capacity on segment i
- `B_i`: Currently bound plasmin on segment i

**Code location**: research_simulation_page.py:4298-4396

#### Fracture Criterion (Spatial Mode Only)

```
min(n_i / N_pf) ≤ n_crit_fraction → edge fractured and removed
```

Where:
- `n_crit_fraction`: Critical protofibril fraction (frozen at Start, default 0.1)

**Code location**: research_simulation_page.py:4488-4538

#### Scalar Degradation (Legacy Mode Only)

```
λ_eff = λ_0 × (σ / σ_ref)^β × (t_ref / t)^γ × g_total
dS/dt = -λ_eff
S_new = S - λ_eff × dt
S_new = clamp(S_new, 0, 1)
```

Where:
- `λ_0`: Baseline degradation rate (frozen at Start)
- `σ`: Fiber tension (cached, tension-only: max(F, 0))
- `σ_ref`: Reference tension = median(tensions) over intact edges
- `β`: Stress sensitivity exponent (frozen at Start, default 1.0)
- `t_ref`: Reference thickness (frozen at Start)
- `t`: Fiber thickness (imported)
- `γ`: Thickness sensitivity exponent (frozen at Start, default 1.0)
- `g_total`: Composite gate factor (product of all gates)

**Code location**: research_simulation_page.py:4696-4730

#### Force-Dependent Gates (Legacy Mode Only)

**Hill Force Response**:
```
g_F(F) = 1 + α × (F^n / (F^n + F0^n))
```
Where α > 0, F0 > 0, n ≥ 1 (frozen at Start).

**Strain-Rate Factor**:
```
strain_rate = (mean_tension - prev_mean_tension) / dt
g_rate = 1 + β_rate × tanh(strain_rate / eps0)
```
Where β_rate ≥ 0, eps0 > 0 (frozen at Start).

**Rupture Amplification**:
```
g_rupture = 1 + γ × max(0, F - F_crit)
```
Where γ ≥ 0, F_crit > 0 (frozen at Start).

**Energy Gate** (fracture toughness proxy):
```
E_i = 0.5 × k0 × S × (L - L_rest)^2
g_energy = 1 + η × max(0, E_i - Gc)
```
Where η ≥ 0, Gc > 0 (frozen at Start).

**Cooperativity Gate** (neighbor damage):
```
D_local = mean([1 - S_j for j in neighbors if S_j > 0])
g_coop = 1 + χ × D_local
```
Where χ ≥ 0 (frozen at Start).

**Shielding Gate** (load redistribution saturation):
```
g_shield = clamp(F_tension / (mean_tension + ε), 0, 1)
```
Where ε > 0 (frozen at Start).

**Memory Gate** (damage accumulation):
```
M_i(t+dt) = (1-μ) × M_i(t) + μ × max(F_i, 0)
g_memory = 1 + ρ × M_i
```
Where μ ∈ [0,1], ρ ≥ 0 (frozen at Start).

**Anisotropy Gate** (load-alignment sensitivity):
```
a = |cos(θ)| = |Δx| / L
g_aniso = 1 + κ × a
```
Where κ ≥ 0 (frozen at Start), θ = angle between fiber and uniaxial x-direction.

**Composite**:
```
g_total = g_F × g_rate × g_rupture × g_energy × g_coop × g_shield × g_memory × g_aniso
g_total = min(g_total, g_max)
```

**Code location**: research_simulation_page.py:4574-4695

#### Plastic Rest-Length Evolution (Legacy Mode Only)

```
if F > F_threshold:
    dL_rest/dt = rate × (F - F_threshold)
    L_rest_new = L_rest + rate × (F - F_threshold) × dt
```

Constraint: `L_rest_effective ≥ original_rest_length` (irreversible).

**Code location**: research_simulation_page.py:4575-4586

#### Termination Criterion

**Spatial mode**:
```
is_connected = BFS_reachability(left_boundary, right_boundary, intact_edges)
if not is_connected:
    termination_reason = "network_percolation_failure"
```

**Code location**: research_simulation_page.py:5061-5093

**Legacy mode**:
```
σ_ref = median([max(0, F) for F in intact_edge_forces])
if σ_ref ≤ 0 or not finite(σ_ref):
    termination_reason = "network_lost_load_bearing_capacity"
```

**Code location**: research_simulation_page.py:3927-4069

---

### 3.4 Outputs and Their Interpretation

#### Visualization Layers

**Edge Rendering** (research_simulation_page.py:7103-7132):
- **Color**: Mapped to S (weakest-link stiffness):
  - `S ≥ 0.9`: deepskyblue2 (intact)
  - `0.5 ≤ S < 0.9`: steelblue (moderate damage)
  - `0.2 ≤ S < 0.5`: gray (low stiffness)
  - `S < 0.2`: red (critical)
- **Width**: Proportional to k_eff (visual proxy):
  - Spatial mode: `width = max(1, int(0.04 × N_pf × S))`
  - Legacy mode: `width = max(1, int(2 × S))`

**Segment Rendering** (Spatial Mode Only, research_simulation_page.py:7196-7270):
- **Fill Color**: Mapped to damage (integrity = n_i / N_pf):
  - `n/N ≥ 0.9`: green (intact)
  - `0.5 ≤ n/N < 0.9`: yellow (moderate damage)
  - `0.2 ≤ n/N < 0.5`: orange (severe damage)
  - `n/N < 0.2`: red (near failure)
- **Outline Color**: Mapped to binding occupancy (B_i / S_i):
  - `B/S < 0.01`: blue (no binding)
  - `0.01 ≤ B/S < 0.3`: cyan (low binding)
  - `0.3 ≤ B/S < 0.7`: purple (medium binding)
  - `B/S ≥ 0.7`: magenta (high binding)
- **Position**: Uniformly spaced along edge (parametric t = segment_index / (num_segments - 1))

**Fractured Edge Rendering** (Spatial Mode Only, research_simulation_page.py:7272-7305):
- **Style**: Dashed gray lines (dash pattern 4,4)
- **Source**: `fractured_history` archive (edges removed via min(n_i/N_pf) ≤ n_crit_fraction)
- **Node positions**: Use current relaxed coordinates (may drift post-fracture)

**Boundary Poles** (research_simulation_page.py:7096-7101):
- **Rendering**: Vertical red lines at x = left_grip_x and x = right_grip_x
- **Semantics**: Constraint manifolds (not physical connectors)
- **Boundary nodes**: Rendered at grip x-coordinates with current y (y is unconstrained)

#### Exported Files

**experiment_log.csv / .json** (research_simulation_page.py:858-977):
- **Format**: One row per batch
- **Column ordering**: Deterministic (batch_index, time, strain, intact_edges, ...)
- **When recorded**: Immediately after successful batch (post-relax, post-sanity-checks)
- **CSV fields**:
  - `batch_index`: 0-indexed batch counter
  - `time`: Cumulative simulation time (seconds)
  - `strain`: Applied uniaxial strain (frozen after Start)
  - `intact_edges`: Count of edges with S > 0
  - `cleaved_edges_total`: Count of edges with S ≤ 0
  - `newly_cleaved`: Edges that transitioned S > 0 → S ≤ 0 this batch
  - `mean_tension`: Mean of max(0, F) over intact edges (post-relax)
  - `lysis_fraction`: 1 - sum(k0 × S) / sum(k0)
  - `dt_used`: Actual timestep used (may differ from base dt in spatial mode due to cleavage stability)
  - **Spatial mode only**:
    - `n_min_frac`: min(n_i / N_pf) over all segments
    - `n_mean_frac`: mean(n_i / N_pf) over all segments
    - `total_bound_plasmin`: sum(B_i) over all segments
    - `P_free_quanta`: Free plasmin pool size
    - `bind_events_applied`: Number of binding events successfully applied
    - `total_unbound_this_batch`: Number of unbinding events this batch
  - **Flattened parameters** (prefixed `param_`): lambda_0, dt, delta, force_alpha, ...

**fractured_history.csv** (Spatial Mode Only, research_simulation_page.py:1021-1096):
- **Format**: One row per segment per fractured edge
- **When recorded**: Immediately after edge removal (min(n_i/N_pf) ≤ n_crit_fraction)
- **Fields**:
  - `edge_id`: Removed edge identifier
  - `batch_index`: Batch when fracture occurred
  - `segment_index`: Segment position along edge
  - `n_i`: Final intact protofibril count
  - `N_pf`: Total protofibril count (constant)
  - `B_i`: Final bound plasmin count
  - `S_i`: Maximum binding site capacity
  - `final_edge_stiffness`: Weakest-link S at fracture
  - `tension_at_failure`: Cached force from previous relax (pre-fracture)
  - `strain_at_failure`: (L_current - L_rest) / L_rest at fracture

**network_snapshot.json** (research_simulation_page.py:1098-1192):
- **Format**: JSON with deterministic ordering (nodes by node_id, edges by edge_id)
- **When exported**: User-triggered (not automatic per batch)
- **Schema**:
  - `provenance_hash`: SHA-256 hash of frozen_params
  - `frozen_params`: All frozen parameters (lambda_0, dt, ..., boundary nodes, grips)
  - `rng_state_hash`: SHA-256 hash of frozen_rng_state
  - `frozen_rng_state`: Serialized RNG state (JSON-safe via _jsonify)
  - `batch_hash`: SHA-256 hash of latest batch payload
  - `nodes`: [{node_id, x, y}] from relaxed coordinates
  - `edges`: [{edge_id, n_from, n_to, S, M, original_rest_length, L_rest_effective, thickness, lysis_batch_index, lysis_time, segments}]
  - **Spatial mode only**:
    - `P_total_quanta`, `P_free_quanta`: Plasmin pool state
    - `spatial_plasmin_params`: Frozen parameters (N_pf, k_cat0, beta, k_off0, alpha, lambda_bind_total, n_crit_fraction)

#### Output Interpretation Caveats

**tension_at_failure**:
- **Source**: Cached force from **previous batch** relax (pre-fracture geometry)
- **Timing lag**: Force reflects geometry before chemistry update that caused fracture
- **Interpretation**: Tension immediately before the batch that fractured the edge, not the instantaneous tension at fracture

**strain_at_failure**:
- **Calculation**: (L_current - L_rest_effective) / L_rest_effective
- **L_current**: Euclidean distance from relaxed_node_coords (pre-fracture)
- **Timing lag**: Same as tension_at_failure (pre-fracture geometry)

**One-batch force update lag** (research_simulation_page.py:3896-3900):
- **Cached forces**: Used for gate computations are from **previous batch** post-relax
- **Current batch**: Chemistry updates (unbind, bind, cleave, degrade) use previous force state
- **Relax timing**: Single relax occurs **after** all chemistry updates
- **Implication**: Force-dependent gates (Hill, rupture, cleavage) use slightly outdated force field

**mean_tension**:
- **Computation**: Mean of max(0, F) over intact edges **after** relax
- **Inclusion criterion**: Only edges with S > 0 (spatial mode: min(n_i/N_pf) > 0)
- **Post-fracture**: If edges removed, mean_tension reflects redistributed forces

**lysis_fraction**:
- **Definition**: 1 - sum(k_eff) / sum(k0 × N_pf)
- **Spatial mode**: Includes protofibril scaling (k_eff = k0 × N_pf × S)
- **Legacy mode**: k_eff = k0 × S
- **Not normalized**: Does not account for removed edges (denominator is initial sum(k0))

---

## Final Verification Checklist

### Stiffness Computation
- [x] Spatial mode: k_eff = k0 × N_pf × S × (t/t_ref)^α (research_simulation_page.py:2583-2585)
- [x] Legacy mode: k_eff = k0 × S × (t/t_ref)^α (research_simulation_page.py:2587)
- [x] S computed from segments: S = min(n_i/N_pf) (research_simulation_page.py:4468-4470)
- [x] Thickness scaling applied in relax() before solver (research_simulation_page.py:2589-2600)

### Fracture Logic
- [x] Spatial mode criterion: min(n_i/N_pf) ≤ n_crit_fraction (research_simulation_page.py:4495)
- [x] Edge removal: Delete from topology, archive to fractured_history (research_simulation_page.py:4500-4535)
- [x] Force redistribution: relax() called once after edge removal (research_simulation_page.py:4538)

### relax() Invocation Timing
- [x] Called once per batch, **after** all chemistry updates (research_simulation_page.py:4806 legacy, 4538 spatial)
- [x] Spatial mode: relax after edge removal (post-fracture) (research_simulation_page.py:4538)
- [x] Pre-batch forces cached from previous relax (research_simulation_page.py:3906-3913)

### Export Functions
- [x] experiment_log: CSV/JSON with deterministic column order (research_simulation_page.py:858-977)
- [x] fractured_history: CSV with segment-level data (research_simulation_page.py:1021-1096)
- [x] network_snapshot: JSON with full state (research_simulation_page.py:1098-1192)
- [x] Atomic writes via tempfile + os.replace (research_simulation_page.py:870-881)

### Visualization Scaling
- [x] Edge color: S-based (deepskyblue2 → red) (research_simulation_page.py:7115-7122)
- [x] Edge width: k_eff visual proxy (spatial: 0.04 × N_pf × S, legacy: 2 × S) (research_simulation_page.py:7124-7132)
- [x] Segment fill: n_i/N_pf damage (green → red) (research_simulation_page.py:7233-7242)
- [x] Segment outline: B_i/S_i occupancy (blue → magenta) (research_simulation_page.py:7244-7253)
- [x] Fractured edges: dashed gray (research_simulation_page.py:7272-7305)

---

**End of Functional Documentation**

**Certification**: Every statement in this document maps to a specific code path in `src/views/tkinter_view/research_simulation_page.py`. No speculative claims, biological interpretations, or future work are included. This is a functional truth record suitable for Methods sections and reproducibility audits.
