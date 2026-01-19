# Enzyme Models Package

## Contract

This package provides strain/tension-dependent hazard rate functions for enzymatic cleavage modeling.

### Interface

Every hazard function must satisfy:

```python
def hazard_fn(strain: float, params: Dict[str, Any]) -> float:
    """
    Compute instantaneous hazard rate.

    Args:
        strain: Current engineering strain (dimensionless, ε = ΔL/L₀)
        params: Model-specific parameters (e.g., lambda0, alpha, beta)

    Returns:
        Hazard rate in 1/µs (instantaneous cleavage probability per unit time)

    Constraints:
        - Return value must be non-negative
        - Return value should be bounded (< 1e6) to prevent numerical issues
    """
```

### Registration

Functions are registered by name:

```python
from single_fiber.enzyme_models import register_hazard, get_hazard_function

def my_hazard(strain, params):
    return params["lambda0"] * (1 + params["alpha"] * strain)

register_hazard("my_model", my_hazard)

# Later retrieval
fn = get_hazard_function("my_model")
rate = fn(0.3, {"lambda0": 0.01, "alpha": 5.0})
```

### Planned Models

| Name | Formula | Parameters |
|------|---------|------------|
| `constant` | λ = λ₀ | `lambda0` |
| `linear` | λ = λ₀ + α·ε | `lambda0`, `alpha` |
| `exponential` | λ = λ₀·exp(β·ε) | `lambda0`, `beta` |
| `catch_slip` | λ = λ_c·exp(-k_c·ε) + λ_s·exp(k_s·ε) | `lambda_c`, `k_c`, `lambda_s`, `k_s` |

### Constraints

1. **Read-only physics**: Hazard functions receive strain as input; they do NOT modify physics state
2. **Determinism**: Given identical strain and params, output is deterministic
3. **No side effects**: Functions are pure; stochastic sampling is separate
4. **Units**: All rates in 1/µs to match simulation time units

### Testing

Each hazard function must have tests verifying:
- Non-negative output for valid inputs
- Correct limiting behavior (ε → 0, ε → ∞)
- Parameter validation (e.g., lambda0 > 0)

## Status

**Phase 4 Scaffold**: Registry structure defined, implementations pending.
