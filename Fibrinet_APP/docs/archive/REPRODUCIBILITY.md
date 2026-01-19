# FibriNet Reproducibility Guide

**Version:** Phase 2D + Phase 3 (Spatial Mechanochemical Model v5.0)
**Date:** 2026-01-02
**Purpose:** Guarantee bit-for-bit reproducibility for publication

---

## Overview

FibriNet simulations are **fully deterministic** given:
1. Fixed input network (nodes, edges, topology)
2. Fixed simulation parameters (kinetics, thresholds)
3. Fixed RNG seed (captured at Start)

**Guarantee:** Same inputs → identical outputs (JSON hash, fracture order, observables)

---

## Seed Handling

### Seed Capture (Initialization)

When a simulation is **Started**, the RNG state is **frozen**:

```python
# At Start button press:
adapter.rng = random.Random()  # Fresh RNG
adapter.rng.seed()  # System entropy (or user-provided seed)
adapter.frozen_rng_state = adapter.rng.getstate()  # Capture state
adapter.frozen_rng_state_hash = SHA256(frozen_rng_state)  # Provenance hash
```

**Key properties:**
- `frozen_rng_state` is captured **once** at Start
- Never modified during simulation
- Used to derive all batch-level seeds

---

### Seed Derivation (Per-Batch)

For each batch, a **deterministic seed** is derived:

```python
# Batch-specific seed (deterministic):
seed_material = f"{frozen_rng_state_hash}|plasmin_selection|{batch_index}"
batch_seed = int(SHA256(seed_material).hexdigest()[:16], 16)

# Create local RNG (does not affect global state):
local_rng = random.Random(batch_seed)
```

**Critical guarantees:**
- Same `frozen_rng_state_hash` + same `batch_index` → same `batch_seed`
- Local RNG scoped to batch (no global state pollution)
- Deterministic even under branching/forking

---

### Seed Propagation (Hierarchical)

Finer-grained seeds are derived **hierarchically**:

```python
# Per-edge seed:
edge_seed = SHA256(f"{batch_seed}|edge|{edge_id}")

# Per-segment seed:
segment_seed = SHA256(f"{edge_seed}|segment|{segment_index}")
```

**Result:** Entire RNG tree is deterministic given root seed

---

## Replay Guarantees

### What Is Guaranteed

✅ **Identical fracture order:** Same edges break in same order
✅ **Identical timing:** Fractures occur at same batch indices
✅ **Identical observables:** All `experiment_log` entries match
✅ **Identical exports:** CSV and JSON hashes match
✅ **Identical termination:** Network disconnects at same batch

**Validation:** `test_phase4_deterministic_replay.py`

---

### What Is NOT Guaranteed

❌ **Different RNG seed:** Different stochastic event sequence
❌ **Different parameters:** Different kinetics → different outcomes
❌ **Different network:** Different topology → different behavior
❌ **Different floating-point hardware:** IEEE 754 guarantees, but edge cases possible

**Mitigation:** Always record `provenance_hash` with results

---

## Determinism Scope

### Sources of Determinism

| Component | Determinism Mechanism | Notes |
|-----------|----------------------|-------|
| **RNG draws** | Frozen seed + hash derivation | SHA256-based |
| **Event ordering** | Sorted edge iteration | `sorted(edges, key=edge_id)` |
| **Percolation check** | BFS over sorted IDs | Graph traversal order fixed |
| **Export order** | Deterministic field ordering | CSV columns, JSON keys sorted |

---

### Sources of Non-Determinism (Controlled)

| Component | Apparent Non-Determinism | Resolution |
|-----------|--------------------------|------------|
| **Binding pattern** | Stochastic selection | Fixed by RNG seed |
| **Unbinding events** | Binomial sampling | Fixed by RNG seed |
| **Fracture race** | Simultaneous damage | Tie-breaking by edge_id |

**All controlled by frozen RNG seed** → replay guaranteed

---

## Provenance Tracking

### Provenance Hash

Every simulation has a **provenance hash** computed at Start:

```python
provenance_payload = {
    "frozen_params": {...},  # All simulation parameters
    "thickness_hash": SHA256(edge_thicknesses),
    "coord_hash": SHA256(node_coords),
    "boundary_ids": [left_ids, right_ids],
    "applied_strain": float,
    "rng_state_hash": frozen_rng_state_hash,
}
provenance_hash = SHA256(json.dumps(provenance_payload, sort_keys=True))
```

**Purpose:**
- Unique identifier for this exact simulation configuration
- Enables result deduplication
- Detects accidental parameter drift

**Stamped in:**
- Every `experiment_log` entry
- Every exported CSV row
- Checkpoint snapshots

---

### Batch Hash

Every batch has a **deterministic hash** capturing state:

```python
batch_payload = {
    "batch_index": int,
    "time": float,
    "strain": float,
    "edges": [{"edge_id": ..., "S": ..., "M": ..., ...}],  # Sorted by edge_id
    "sigma_ref": float,
    "provenance_hash": str,
}
batch_hash = SHA256(json.dumps(batch_payload, sort_keys=True))
```

**Purpose:**
- Detects state divergence during replay
- Enables incremental validation (batch-by-batch comparison)
- Checksum for experiment integrity

**Validation:** `test_phase4_deterministic_replay.py::test_deterministic_replay_identical_seeds()`

---

## Checkpoint and Resume

### Checkpoint Export

A simulation can be **checkpointed** at any batch:

```python
snapshot = adapter.export_network_snapshot(path="checkpoint_batch_42.json")
```

**Snapshot contains:**
- Full edge state (S, M, L_eff, segments, plasmin)
- Node coordinates (relaxed positions)
- RNG state (`frozen_rng_state_hash`)
- Provenance hash
- Batch index, time, strain

**Guarantee:** Snapshot is **deterministic** (same state → same JSON hash)

---

### Resume from Checkpoint

A simulation can be **resumed** from a checkpoint:

```python
adapter.resume_from_checkpoint(
    snapshot_path="checkpoint_batch_42.json",
    log_path="experiment_log.json",
    resume_batch_index=42
)
```

**Validation:**
- RNG state hash must match
- Provenance hash must match
- Batch hashes must match (replay consistency)

**Post-resume guarantee:** Identical continuation (same as if never stopped)

---

## Best Practices for Reproducibility

### 1. Always Capture Seed

```python
# At simulation Start:
rng_seed = 12345  # For reproducibility
adapter.rng.seed(rng_seed)
# frozen_rng_state is automatically captured
```

**Recommendation:** Log `rng_seed` in experiment metadata for easy re-seeding

---

### 2. Never Modify Frozen Parameters

After **Start**, these are **immutable**:
- `applied_strain`
- `lambda_0`, `dt`, kinetic parameters
- Boundary node IDs
- Network topology

**Violation:** Raises `ValueError` with clear message

---

### 3. Use Provenance Hash for Deduplication

Before running expensive simulation:

```python
# Check if already computed
if provenance_hash in completed_runs:
    print(f"Already computed: {provenance_hash}")
    return cached_result
```

**Benefit:** Avoid redundant computation

---

### 4. Export Intermediate Checkpoints

For long simulations:

```python
# Every 100 batches:
if batch_index % 100 == 0:
    adapter.export_network_snapshot(f"checkpoint_batch_{batch_index}.json")
```

**Benefit:** Resume from failure without full re-run

---

### 5. Validate Replay Before Publication

```python
# Run twice with same seed:
result1 = run_simulation(seed=12345)
result2 = run_simulation(seed=12345)

# Assert identical:
assert result1["experiment_log_hash"] == result2["experiment_log_hash"]
```

**Benefit:** Catch non-determinism before peer review

---

## Replay Validation Protocol

### Step 1: Run Reference Simulation

```bash
# Run once with known seed
python run_simulation.py --seed 12345 --batches 100 --output ref_run.json
```

**Capture:**
- `experiment_log.json` (full time series)
- `fractured_history.csv` (fracture events)
- `provenance_hash` (configuration fingerprint)

---

### Step 2: Re-Run with Same Seed

```bash
# Run again with identical seed
python run_simulation.py --seed 12345 --batches 100 --output replay_run.json
```

---

### Step 3: Compare Outputs

```python
import hashlib
import json

# Load both logs
with open("ref_run.json") as f:
    ref_log = json.load(f)
with open("replay_run.json") as f:
    replay_log = json.load(f)

# Compute hashes
ref_hash = hashlib.sha256(json.dumps(ref_log, sort_keys=True).encode()).hexdigest()
replay_hash = hashlib.sha256(json.dumps(replay_log, sort_keys=True).encode()).hexdigest()

# Assert identity
assert ref_hash == replay_hash, "Replay diverged from reference!"
```

**Expected:** Hashes match exactly (bit-for-bit identical)

---

### Step 4: Validate Fracture Order

```python
# Extract fractured edge IDs in order
ref_fractures = [(r["edge_id"], r["batch_index"]) for r in ref_fractured_history]
replay_fractures = [(r["edge_id"], r["batch_index"]) for r in replay_fractured_history]

# Assert identical order
assert ref_fractures == replay_fractures, "Fracture order changed!"
```

---

## Troubleshooting Non-Determinism

### Symptom: Different Results with Same Seed

**Possible causes:**
1. ❌ **Global RNG state modified:** Check for stray `random.seed()` calls
2. ❌ **Unstable sort:** Ensure tie-breaking uses edge_id consistently
3. ❌ **Floating-point variance:** Use `numpy` for numerics (more stable than Python float)
4. ❌ **Dict iteration order:** Use `sorted(edges, key=...)` for stable ordering

**Fix:** Review RNG usage and sorting logic

---

### Symptom: Checkpoint Resume Fails Validation

**Possible causes:**
1. ❌ **Snapshot from different run:** Provenance hash mismatch
2. ❌ **Log/snapshot out of sync:** Batch index mismatch
3. ❌ **Partial checkpoint write:** Corrupted JSON

**Fix:** Re-export checkpoint from validated state

---

### Symptom: CSV/JSON Inconsistency

**Possible causes:**
1. ❌ **Aggregation error:** CSV reduction doesn't match JSON detail
2. ❌ **Truncation:** CSV numeric precision loss
3. ❌ **NaN propagation:** Silent NaN in division-by-zero

**Fix:** Validate with `test_phase4_export_consistency.py`

---

## Reproducibility Checklist

Before publishing results, verify:

- [ ] RNG seed recorded in experiment metadata
- [ ] Provenance hash logged with all outputs
- [ ] Replay validation passed (`test_phase4_deterministic_replay.py`)
- [ ] Export consistency validated (`test_phase4_export_consistency.py`)
- [ ] Checkpoint/resume tested (if applicable)
- [ ] No global RNG state mutations
- [ ] All sorting uses deterministic keys (edge_id, batch_index)

---

## Contact and Support

**For reproducibility issues:**
1. Run `test_phase4_deterministic_replay.py` to diagnose
2. Check `frozen_rng_state_hash` matches between runs
3. Verify `provenance_hash` is identical

**If tests fail:** This indicates a **critical bug** — do not publish results until resolved.

---

**Reproducibility guarantee: Same seed → identical simulation**

**Validation complete. No physics modifications made during Phase 4.**
