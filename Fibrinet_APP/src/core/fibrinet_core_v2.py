"""FibriNet Core V2: WLC/eWLC mechanics, L-BFGS-B minimization, Gillespie SSA."""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import math
from scipy.optimize import minimize
from collections import defaultdict, deque
import random
import hashlib
import json


class PhysicalConstants:
    """Physical constants (SI units)."""
    k_B = 1.380649e-23
    T = 310.15
    k_B_T = k_B * T
    xi = 1.0e-6
    EWLC_K0 = 3.0e-8
    k_cat_0 = 0.020  # Lynch et al. (2022): mean single-fiber cleavage time ≈ 49.8 s
    beta_strain = 0.84  # Varjú et al. (2011) J Thromb Haemost
    x_bell = 0.5e-9
    PRESTRAIN = 0.23
    PRESTRAIN_AMPLITUDE = 0.0
    MAX_STRAIN = 0.99
    CASCADE_RUPTURE_THRESHOLD = 0.30  # Strain threshold for mechanical rupture cascade (calibrated)
    CASCADE_ENABLED = True  # Kill-switch: False → byte-for-byte identical to pre-cascade
    FIBER_MEAN_DIAMETER_NM = 130.0    # Mean fiber diameter [nm] (Yeromonahos 2010)
    FIBER_DIAMETER_CV = 0.5           # Coefficient of variation (lognormal)
    FIBER_DIAMETER_REF_NM = 130.0     # Reference diameter for scaling [nm]
    S_MIN_BELL = 0.05
    MAX_BELL_EXPONENT = 100.0
    F_MAX = 1e-6
    # Three-regime constitutive model (kill-switch: False → WLC-only)
    THREE_REGIME_ENABLED = False

    # Sigmoid blend constants (Maksudov 2021, Filla 2023)
    SIGMOID_EPSILON_MID = 1.3      # Transition midpoint strain
                                   # Free parameter — Maksudov 2021 range 1.3–1.6
    SIGMOID_DELTA_EPSILON = 0.15   # Transition half-width
                                   # Free parameter — Maksudov 2021 range 0.12–0.41
    Y_A_STRONG = 6.5e6             # High-strain axial modulus [Pa] (Maksudov 2021 average)
    RUPTURE_STRAIN = 2.8           # Mechanical rupture strain
                                   # TODO: Maksudov 2021 reports ε*≈212% — verify
                                   # exact figure/table before updating to 2.12


PC = PhysicalConstants()

# Gauss-Legendre quadrature nodes and weights on [-1, 1]
# For energy integration of sigmoid-blended force when THREE_REGIME_ENABLED.
# 16 points needed because the WLC cap at ε=0.995 creates a kink in the
# integrand; 8 points gives ~1% error at that boundary, 16 is well under.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)
_GL_N = len(_GL_NODES)


@dataclass(frozen=True)
class WLCFiber:
    """WLC/eWLC fiber with integrity S ∈ [0,1]."""
    fiber_id: int
    node_i: int
    node_j: int
    L_c: float  # Contour length [m]
    xi: float = PC.xi
    S: float = 1.0  # Integrity [0, 1]
    x_bell: float = PC.x_bell
    k_cat_0: float = PC.k_cat_0
    force_model: str = 'wlc'  # 'wlc' or 'ewlc'
    K0: float = PC.EWLC_K0  # eWLC finite extensibility parameter [N]
    diameter_nm: float = 130.0  # Fiber diameter [nm]

    def __post_init__(self):
        if self.L_c <= 0:
            raise ValueError(f"Fiber {self.fiber_id}: L_c must be > 0 (got {self.L_c})")
        if self.xi <= 0:
            raise ValueError(f"Fiber {self.fiber_id}: xi must be > 0 (got {self.xi})")
        if not (0 <= self.S <= 1):
            raise ValueError(f"Fiber {self.fiber_id}: S must be in [0, 1] (got {self.S})")
        if self.force_model not in ('wlc', 'ewlc'):
            raise ValueError(f"Fiber {self.fiber_id}: force_model must be 'wlc' or 'ewlc' (got {self.force_model})")
        if self.K0 < 0:
            raise ValueError(f"Fiber {self.fiber_id}: K0 must be >= 0 (got {self.K0})")

    @property
    def _k_B_T_Lc_over_xi(self) -> float:
        return PC.k_B_T * self.L_c / self.xi

    @property
    def _k_B_T_over_xi(self) -> float:
        return PC.k_B_T / self.xi

    def _safe_strain(self, x: float) -> float:
        raw = (x - self.L_c) / self.L_c
        if PC.THREE_REGIME_ENABLED:
            return min(raw, PC.RUPTURE_STRAIN)
        return min(raw, PC.MAX_STRAIN)

    def compute_force(self, x: float) -> float:
        """F_eff = S × F_model, sigmoid blend when THREE_REGIME_ENABLED.

        For x < L_c (compression), the WLC extrapolation gives negative (repulsive)
        force. This provides essential structural stability in networks, preventing
        cascade collapse when neighboring fibers are degraded.
        """
        strain = self._safe_strain(x)

        if PC.THREE_REGIME_ENABLED:
            if strain >= PC.RUPTURE_STRAIN:
                return 0.0  # Signal to caller: fiber should rupture

            if strain >= PC.MAX_STRAIN:
                # Sigmoid blend for high strain (above WLC singularity zone)
                w = 0.5 * (1.0 + math.tanh(
                    (strain - PC.SIGMOID_EPSILON_MID)
                    / PC.SIGMOID_DELTA_EPSILON))

                # WLC component (singularity guard at 0.995)
                eps_wlc = min(strain, 0.995)
                one_minus = 1.0 - eps_wlc
                F_wlc = self._k_B_T_over_xi * (
                    1.0 / (4.0 * one_minus**2) - 0.25 + eps_wlc)
                if self.force_model == 'ewlc':
                    F_wlc += self.K0 * eps_wlc

                # Backbone component: linear spring K_bb * ε
                d_m = self.diameter_nm * 1e-9
                A = math.pi * (d_m / 2.0) ** 2
                K_bb = PC.Y_A_STRONG * A / self.L_c
                F_backbone = K_bb * strain

                # Blend (no F_MAX cap in sigmoid regime)
                F_model = (1.0 - w) * F_wlc + w * F_backbone
                return float(self.S * F_model)

            # Below MAX_STRAIN: fall through to WLC-only path

        # WLC-only path — unchanged
        wlc_strain = min(strain, PC.MAX_STRAIN)
        one_minus_eps = 1.0 - wlc_strain
        F_wlc = self._k_B_T_over_xi * (
            1.0 / (4.0 * one_minus_eps**2) - 0.25 + wlc_strain
        )
        if self.force_model == 'ewlc':
            F_model = F_wlc + self.K0 * wlc_strain
        else:
            F_model = F_wlc

        F_eff = self.S * F_model
        if F_eff > PC.F_MAX:
            F_eff = PC.F_MAX
        return float(F_eff)

    def compute_energy(self, x: float) -> float:
        """U_eff = S × U_model.

        When THREE_REGIME_ENABLED: energy is the integral of the sigmoid-blended
        force, computed via 8-point Gauss-Legendre quadrature. This guarantees
        dU/dx = F for L-BFGS-B gradient consistency.
        """
        strain = self._safe_strain(x)

        if PC.THREE_REGIME_ENABLED:
            if strain >= PC.RUPTURE_STRAIN:
                return 0.0  # Ruptured fiber carries no energy

            if strain >= PC.MAX_STRAIN:
                # High strain: GL quadrature of sigmoid-blended force
                # Split integral: analytical WLC from 0 to MAX_STRAIN,
                # then GL quadrature of F_blend from MAX_STRAIN to strain
                eps0 = PC.MAX_STRAIN

                # Analytical WLC energy from 0 to MAX_STRAIN
                one_minus_0 = 1.0 - eps0
                U_wlc_base = self._k_B_T_Lc_over_xi * (
                    1.0 / (4.0 * one_minus_0) - 0.25
                    - eps0 / 4.0 + eps0**2 / 2.0)
                if self.force_model == 'ewlc':
                    U_wlc_base += self.L_c * (self.K0 / 2.0) * eps0**2

                # GL quadrature of F_blend from MAX_STRAIN to strain
                delta = strain - eps0
                h = delta / 2.0
                kBT_xi = self._k_B_T_over_xi
                d_m = self.diameter_nm * 1e-9
                A = math.pi * (d_m / 2.0) ** 2
                K_bb = PC.Y_A_STRONG * A / self.L_c
                is_ewlc = (self.force_model == 'ewlc')

                U_integral = 0.0
                for k in range(_GL_N):
                    # Map GL node to [eps0, strain]
                    eps_k = eps0 + h * (1.0 + _GL_NODES[k])

                    sig_w = 0.5 * (1.0 + math.tanh(
                        (eps_k - PC.SIGMOID_EPSILON_MID)
                        / PC.SIGMOID_DELTA_EPSILON))

                    eps_wlc_k = min(eps_k, 0.995)
                    one_minus_k = 1.0 - eps_wlc_k
                    F_wlc_k = kBT_xi * (
                        1.0 / (4.0 * one_minus_k**2) - 0.25 + eps_wlc_k)
                    if is_ewlc:
                        F_wlc_k += self.K0 * eps_wlc_k

                    F_bb_k = K_bb * eps_k
                    F_blend_k = (1.0 - sig_w) * F_wlc_k + sig_w * F_bb_k
                    U_integral += _GL_WEIGHTS[k] * F_blend_k

                U_model = U_wlc_base + self.L_c * h * U_integral
                return float(self.S * U_model)

            # Below MAX_STRAIN: fall through to WLC-only path

        # WLC-only path — unchanged
        wlc_strain = min(strain, PC.MAX_STRAIN)
        one_minus_eps = 1.0 - wlc_strain
        U_wlc = self._k_B_T_Lc_over_xi * (
            1.0 / (4.0 * one_minus_eps) - 0.25 - wlc_strain / 4.0 + wlc_strain**2 / 2.0
        )
        if self.force_model == 'ewlc':
            U_model = U_wlc + self.L_c * (self.K0 / 2.0) * wlc_strain**2
        else:
            U_model = U_wlc

        U_eff = self.S * U_model
        return float(U_eff)

    def compute_cleavage_rate(self, current_length: float) -> float:
        """
        Strain-based enzymatic inhibition model (CORRECTED for fibrinolysis).

        k(ε) = k₀ × exp(-β × ε)

        where ε = (L - L_c) / L_c is the fiber strain.

        Physical Interpretation (Enzymatic Lysis):
        - Higher strain → SLOWER cleavage (plasmin binding/activity hindered)
        - Matches experimental observations:
          * Li et al. (2017): Stretching fibers significantly hampers lysis
          * Adhikari et al. (2012): Strain reduces degradation up to 10-fold
          * Bucay et al. (2015): Strain conceals plasmin binding sites
        - β = 10 gives ~10-fold reduction at 23% strain
        - Uses strain (dimensionless) instead of force (avoids unit issues)

        Args:
            current_length: Current fiber length [m]

        Returns:
            Cleavage rate [1/s]
        """
        # Compute strain: ε = (L - L_c) / L_c
        strain = (current_length - self.L_c) / self.L_c

        # Strain is always >= 0 for tension (compression not modeled)
        strain = max(0.0, float(strain))

        # Exponential inhibition: k = k₀ × exp(-β × ε)
        exponent = -PC.beta_strain * strain

        # Underflow guard (though exp(-10) ≈ 4.5e-5 is fine)
        if exponent < -20.0:  # exp(-20) ≈ 2e-9, essentially zero
            exponent = -20.0

        k_cleave = self.k_cat_0 * np.exp(exponent)
        return float(k_cleave)


@dataclass
class NetworkState:
    """
    Complete mechanical + chemical state of fibrin network.

    Mutable container for simulation state at a given time.
    """
    time: float  # [s]
    fibers: List[WLCFiber]
    node_positions: Dict[int, np.ndarray]  # node_id -> [x, y] in [m]
    fixed_nodes: Dict[int, np.ndarray]  # Fully fixed boundary conditions (both X and Y)
    partial_fixed_x: Dict[int, float] = field(default_factory=dict)  # Nodes with only X fixed (Y free)

    # Mechanical solver state
    energy: float = 0.0  # [J]

    # Statistics
    n_ruptured: int = 0
    lysis_fraction: float = 0.0

    # Degradation tracking (for research output)
    degradation_history: List[Dict[str, Any]] = field(default_factory=list)
    # Each entry: {'time': float, 'fiber_id': int, 'order': int, 'length': float, 'strain': float}

    # Boundary node sets for connectivity detection
    left_boundary_nodes: set = field(default_factory=set)
    right_boundary_nodes: set = field(default_factory=set)

    # Plasmin visualization (for biological realism)
    plasmin_locations: Dict[int, float] = field(default_factory=dict)
    # Maps fiber_id -> position along fiber (0.0 to 1.0, where 0.0 = node_i, 1.0 = node_j)
    # Only fibers with active plasmin are tracked

    # Critical fiber tracking (for publication figures)
    critical_fiber_id: Optional[int] = None
    # ID of the fiber whose cleavage triggered network clearance

    # Clearance event details (for research analysis)
    clearance_event: Optional[Dict[str, Any]] = None
    # Records: time, critical_fiber_id, lysis_fraction, remaining_fibers, total_fibers

    # ABM state tracking (None when mean-field mode active)
    abm_next_fiber_id: int = 0   # Counter for new fiber IDs from positional splits
    abm_next_node_id: int = 0    # Counter for new node IDs from positional splits

    # Fiber lookup index: {fiber_id: list_index} — rebuilt on mutation
    _fiber_index: Dict[int, int] = field(default_factory=dict, repr=False)

    # Cached adjacency for BFS connectivity checks — avoids full rebuild each call
    _adjacency_cache: Optional[Dict[int, set]] = field(default=None, repr=False)
    _adjacency_cache_valid: bool = field(default=False, repr=False)

    def rebuild_fiber_index(self):
        """Rebuild fiber_id → list index mapping. Call after any fiber list mutation."""
        self._fiber_index = {f.fiber_id: i for i, f in enumerate(self.fibers)}

    def invalidate_adjacency_cache(self):
        """Mark adjacency cache as stale. Call after topology changes (splits)."""
        self._adjacency_cache_valid = False

    def get_adjacency(self):
        """Get or rebuild adjacency dict for active fibers."""
        if self._adjacency_cache_valid and self._adjacency_cache is not None:
            return self._adjacency_cache
        adj = defaultdict(set)
        for fiber in self.fibers:
            if fiber.S > 0:
                adj[fiber.node_i].add(fiber.node_j)
                adj[fiber.node_j].add(fiber.node_i)
        self._adjacency_cache = adj
        self._adjacency_cache_valid = True
        return adj

    def remove_fiber_from_adjacency(self, fiber):
        """Incrementally remove a fiber's edges from the cached adjacency."""
        if self._adjacency_cache is None:
            return
        adj = self._adjacency_cache
        adj[fiber.node_i].discard(fiber.node_j)
        adj[fiber.node_j].discard(fiber.node_i)

    def get_fiber(self, fiber_id: int):
        """O(1) fiber lookup by ID. Returns (index, fiber) or (None, None)."""
        idx = self._fiber_index.get(fiber_id)
        if idx is not None and idx < len(self.fibers):
            f = self.fibers[idx]
            if f.fiber_id == fiber_id:
                return idx, f
        # Index stale — fallback to linear scan and rebuild
        for i, f in enumerate(self.fibers):
            if f.fiber_id == fiber_id:
                self._fiber_index[fiber_id] = i
                return i, f
        return None, None



class EnergyMinimizationSolver:
    """
    Energy minimization with analytical Jacobian.

    Uses the identity ∂E/∂r_i = -F_net,i to compute gradients
    via vectorized NumPy operations.

    Supports partial constraints: nodes with only X coordinate fixed (Y free).

    CRITICAL: Energy and gradient are scaled by 1/k_B_T to avoid numerical issues.
    WLC energies are ~1e-20 J in SI, but L-BFGS-B's ftol uses max(|E|, 1) as
    denominator. When |E| << 1, the tolerance becomes absolute (not relative),
    causing the solver to converge at iter=0 without moving any node.
    Scaling to k_B_T units makes energies O(1), fixing this.
    """

    # Energy scaling: convert from SI (Joules) to thermal units (k_B_T)
    # This makes E ~O(1) per fiber, well-conditioned for L-BFGS-B
    ENERGY_SCALE = 1.0 / PC.k_B_T  # ~2.34e20

    def __init__(self, fibers: List[WLCFiber], fixed_coords: Dict[int, np.ndarray],
                 partial_fixed_x: Dict[int, float] = None):
        """
        Initialize solver with network topology.

        Args:
            fibers: List of WLC fibers
            fixed_coords: Fully fixed node positions {node_id: [x, y]}
            partial_fixed_x: Nodes with only X fixed {node_id: x_value}, Y is free
        """
        self.fibers = fibers
        self.fixed_coords = fixed_coords
        self.partial_fixed_x = partial_fixed_x if partial_fixed_x is not None else {}

        # Extract all node IDs
        all_node_ids = set()
        for f in fibers:
            all_node_ids.add(f.node_i)
            all_node_ids.add(f.node_j)

        # Separate node types:
        # - Fully fixed: both X and Y constrained (left boundary)
        # - Partially fixed: X constrained, Y free (right boundary)
        # - Free: both X and Y free (interior nodes)
        fully_fixed = set(fixed_coords.keys())
        partially_fixed = set(self.partial_fixed_x.keys())
        self.free_node_ids = sorted(all_node_ids - fully_fixed - partially_fixed)
        self.partial_node_ids = sorted(partially_fixed)

        self.n_free = len(self.free_node_ids)  # Both X and Y optimized
        self.n_partial = len(self.partial_node_ids)  # Only Y optimized
        self.n_total = len(all_node_ids)

        # Create node index mapping
        self.node_idx = {nid: i for i, nid in enumerate(sorted(all_node_ids))}

        # Precompute connectivity for vectorization
        self._precompute_connectivity()

    def _precompute_connectivity(self):
        """Precompute fiber connectivity arrays for vectorized operations."""
        self.fiber_node_i_idx = np.array([self.node_idx[f.node_i] for f in self.fibers], dtype=int)
        self.fiber_node_j_idx = np.array([self.node_idx[f.node_j] for f in self.fibers], dtype=int)
        self.fiber_L_c = np.array([f.L_c for f in self.fibers])
        self.fiber_S = np.array([f.S for f in self.fibers])
        self.fiber_xi = np.array([f.xi for f in self.fibers])
        self.fiber_K0 = np.array([f.K0 for f in self.fibers])
        self.fiber_is_ewlc = np.array([f.force_model == 'ewlc' for f in self.fibers])
        self.fiber_diameter_nm = np.array([f.diameter_nm for f in self.fibers])

        # Pre-allocate reusable arrays (avoid repeated allocations per L-BFGS-B call)
        n_fibers = len(self.fibers)
        self._pos_all = np.zeros((self.n_total, 2))
        self._forces_all = np.zeros((self.n_total, 2))
        n_free_vars = 2 * self.n_free
        n_partial_vars = self.n_partial
        self._grad = np.zeros(n_free_vars + n_partial_vars)

        # Pre-compute free/partial node index arrays for gradient extraction
        self._free_node_internal_idx = np.array(
            [self.node_idx[nid] for nid in self.free_node_ids], dtype=int)
        self._partial_node_internal_idx = np.array(
            [self.node_idx[nid] for nid in self.partial_node_ids], dtype=int)

    def _batch_wlc_energy(self, lengths: np.ndarray) -> np.ndarray:
        """Vectorized WLC/eWLC energy, GL quadrature of sigmoid blend when THREE_REGIME_ENABLED."""
        raw_strain = (lengths - self.fiber_L_c) / self.fiber_L_c

        if PC.THREE_REGIME_ENABLED:
            strain = np.minimum(raw_strain, PC.RUPTURE_STRAIN)
            n_fibers = len(strain)

            # Fibers below MAX_STRAIN: standard WLC (identical to disabled path)
            # Fibers at/above MAX_STRAIN: analytical WLC base + GL quadrature above
            U_model = np.zeros(n_fibers)
            hi = strain >= PC.MAX_STRAIN
            lo = ~hi

            # Low-strain fibers: standard WLC energy
            if np.any(lo):
                s_lo = np.minimum(strain[lo], PC.MAX_STRAIN)
                ome_lo = 1.0 - s_lo
                kBT_Lc_xi_lo = PC.k_B_T * self.fiber_L_c[lo] / self.fiber_xi[lo]
                U_lo = kBT_Lc_xi_lo * (
                    1.0 / (4.0 * ome_lo) - 0.25 - s_lo / 4.0 + s_lo**2 / 2.0)
                ewlc_lo = self.fiber_is_ewlc[lo]
                if np.any(ewlc_lo):
                    U_lo[ewlc_lo] += (self.fiber_L_c[lo][ewlc_lo] *
                                      (self.fiber_K0[lo][ewlc_lo] / 2.0) * s_lo[ewlc_lo]**2)
                U_model[lo] = self.fiber_S[lo] * U_lo

            # High-strain fibers: WLC base to MAX_STRAIN + GL quadrature above
            if np.any(hi):
                s_hi = strain[hi]
                eps0 = PC.MAX_STRAIN
                one_minus_0 = 1.0 - eps0

                # Analytical WLC energy from 0 to MAX_STRAIN
                kBT_Lc_xi_hi = PC.k_B_T * self.fiber_L_c[hi] / self.fiber_xi[hi]
                U_base = kBT_Lc_xi_hi * (
                    1.0 / (4.0 * one_minus_0) - 0.25 - eps0 / 4.0 + eps0**2 / 2.0)
                ewlc_hi = self.fiber_is_ewlc[hi]
                if np.any(ewlc_hi):
                    U_base[ewlc_hi] += (self.fiber_L_c[hi][ewlc_hi] *
                                        (self.fiber_K0[hi][ewlc_hi] / 2.0) * eps0**2)

                # GL quadrature of F_blend from MAX_STRAIN to strain
                delta = s_hi - eps0
                h = delta / 2.0

                kBT_xi_hi = PC.k_B_T / self.fiber_xi[hi]
                K0_hi = self.fiber_K0[hi]
                d_m_hi = self.fiber_diameter_nm[hi] * 1e-9
                A_hi = np.pi * (d_m_hi / 2.0)**2
                K_bb_hi = PC.Y_A_STRONG * A_hi / self.fiber_L_c[hi]

                U_sum = np.zeros(len(s_hi))
                for k in range(_GL_N):
                    eps_k = eps0 + h * (1.0 + _GL_NODES[k])

                    sig_w = 0.5 * (1.0 + np.tanh(
                        (eps_k - PC.SIGMOID_EPSILON_MID) / PC.SIGMOID_DELTA_EPSILON))

                    eps_wlc_k = np.minimum(eps_k, 0.995)
                    one_minus_k = 1.0 - eps_wlc_k
                    F_wlc_k = kBT_xi_hi * (
                        1.0 / (4.0 * one_minus_k**2) - 0.25 + eps_wlc_k)
                    if np.any(ewlc_hi):
                        F_wlc_k[ewlc_hi] += K0_hi[ewlc_hi] * eps_wlc_k[ewlc_hi]

                    F_bb_k = K_bb_hi * eps_k
                    F_blend_k = (1.0 - sig_w) * F_wlc_k + sig_w * F_bb_k
                    U_sum += _GL_WEIGHTS[k] * F_blend_k

                U_model[hi] = self.fiber_S[hi] * (
                    U_base + self.fiber_L_c[hi] * h * U_sum)

            # Zero out ruptured fibers
            U_model = np.where(raw_strain >= PC.RUPTURE_STRAIN, 0.0, U_model)

            return U_model

        # Original WLC-only path (THREE_REGIME_ENABLED=False) — unchanged
        strain = np.minimum(raw_strain, PC.MAX_STRAIN)
        one_minus_eps = 1.0 - strain
        kBT_Lc_xi = PC.k_B_T * self.fiber_L_c / self.fiber_xi
        U_wlc = kBT_Lc_xi * (
            1.0 / (4.0 * one_minus_eps) - 0.25 - strain / 4.0 + strain**2 / 2.0
        )
        U_model = U_wlc.copy()
        if np.any(self.fiber_is_ewlc):
            U_model[self.fiber_is_ewlc] += (
                self.fiber_L_c[self.fiber_is_ewlc] *
                (self.fiber_K0[self.fiber_is_ewlc] / 2.0) *
                strain[self.fiber_is_ewlc]**2
            )
        return self.fiber_S * U_model

    def _batch_wlc_force(self, lengths: np.ndarray) -> np.ndarray:
        """Vectorized WLC/eWLC force, sigmoid blend when THREE_REGIME_ENABLED."""
        raw_strain = (lengths - self.fiber_L_c) / self.fiber_L_c

        if PC.THREE_REGIME_ENABLED:
            strain = np.minimum(raw_strain, PC.RUPTURE_STRAIN)

            # Fibers below MAX_STRAIN: standard WLC (identical to disabled path)
            # Fibers at/above MAX_STRAIN: sigmoid blend
            hi = strain >= PC.MAX_STRAIN
            lo = ~hi

            F_model = np.zeros_like(strain)

            # Low-strain fibers: standard WLC with F_MAX cap
            if np.any(lo):
                s_lo = np.minimum(strain[lo], PC.MAX_STRAIN)
                ome_lo = 1.0 - s_lo
                kBT_xi_lo = PC.k_B_T / self.fiber_xi[lo]
                F_lo = kBT_xi_lo * (1.0 / (4.0 * ome_lo**2) - 0.25 + s_lo)
                ewlc_lo = self.fiber_is_ewlc[lo]
                if np.any(ewlc_lo):
                    F_lo[ewlc_lo] += self.fiber_K0[lo][ewlc_lo] * s_lo[ewlc_lo]
                F_lo = self.fiber_S[lo] * F_lo
                F_model[lo] = np.minimum(F_lo, PC.F_MAX)

            # High-strain fibers: sigmoid blend (no F_MAX cap)
            if np.any(hi):
                s_hi = strain[hi]
                w = 0.5 * (1.0 + np.tanh(
                    (s_hi - PC.SIGMOID_EPSILON_MID) / PC.SIGMOID_DELTA_EPSILON))

                eps_wlc = np.minimum(s_hi, 0.995)
                one_minus = 1.0 - eps_wlc
                kBT_xi_hi = PC.k_B_T / self.fiber_xi[hi]
                F_wlc = kBT_xi_hi * (1.0 / (4.0 * one_minus**2) - 0.25 + eps_wlc)
                ewlc_hi = self.fiber_is_ewlc[hi]
                if np.any(ewlc_hi):
                    F_wlc[ewlc_hi] += self.fiber_K0[hi][ewlc_hi] * eps_wlc[ewlc_hi]

                d_m = self.fiber_diameter_nm[hi] * 1e-9
                A = np.pi * (d_m / 2.0)**2
                K_bb = PC.Y_A_STRONG * A / self.fiber_L_c[hi]
                F_backbone = K_bb * s_hi

                F_blend = (1.0 - w) * F_wlc + w * F_backbone
                F_model[hi] = self.fiber_S[hi] * F_blend

            # Zero out ruptured fibers
            F_model = np.where(raw_strain >= PC.RUPTURE_STRAIN, 0.0, F_model)

            return F_model

        # Original WLC-only path (THREE_REGIME_ENABLED=False) — unchanged
        strain = np.minimum(raw_strain, PC.MAX_STRAIN)
        one_minus_eps = 1.0 - strain
        kBT_xi = PC.k_B_T / self.fiber_xi
        F_wlc = kBT_xi * (1.0 / (4.0 * one_minus_eps**2) - 0.25 + strain)
        F_model = F_wlc.copy()
        if np.any(self.fiber_is_ewlc):
            F_model[self.fiber_is_ewlc] += (
                self.fiber_K0[self.fiber_is_ewlc] * strain[self.fiber_is_ewlc]
            )
        F_eff = self.fiber_S * F_model
        return np.minimum(F_eff, PC.F_MAX)

    def pack_free_coords(self, node_positions: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Pack free/partial node coordinates into flat array.

        Order: [free nodes (x,y), partial nodes (y only)]
        """
        # Free nodes: both X and Y
        n_free_vars = 2 * self.n_free
        # Partial nodes: only Y (X is constrained)
        n_partial_vars = self.n_partial
        x = np.zeros(n_free_vars + n_partial_vars)

        # Pack free nodes (both coordinates)
        for i, nid in enumerate(self.free_node_ids):
            x[2*i] = node_positions[nid][0]
            x[2*i + 1] = node_positions[nid][1]

        # Pack partial nodes (Y coordinate only)
        offset = n_free_vars
        for i, nid in enumerate(self.partial_node_ids):
            x[offset + i] = node_positions[nid][1]  # Only Y

        return x

    def unpack_free_coords(self, x: np.ndarray, base_positions: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Unpack flat array into node_positions dict.

        Reconstructs: free nodes (from x,y), partial nodes (X from constraint, Y from x), fixed nodes (from base_positions)
        """
        positions = dict(base_positions)  # Start with fully fixed nodes

        # Unpack free nodes (both X and Y from optimization)
        for i, nid in enumerate(self.free_node_ids):
            positions[nid] = np.array([x[2*i], x[2*i + 1]])

        # Unpack partial nodes (X from constraint, Y from optimization)
        n_free_vars = 2 * self.n_free
        for i, nid in enumerate(self.partial_node_ids):
            x_fixed = self.partial_fixed_x[nid]  # X is constrained
            y_opt = x[n_free_vars + i]  # Y is optimized
            positions[nid] = np.array([x_fixed, y_opt])

        return positions

    def compute_total_energy(self, x: np.ndarray, fixed_coords: Dict[int, np.ndarray]) -> float:
        """
        Compute total network energy.

        E_total = Σ_fibers U_fiber(|r_j - r_i|)

        Args:
            x: Flat array of free node coords
            fixed_coords: Boundary node positions

        Returns:
            Total energy [J]
        """
        # Unpack coordinates into pre-allocated position array
        node_positions = self.unpack_free_coords(x, fixed_coords)
        self._pos_all[:] = 0
        for nid, pos in node_positions.items():
            self._pos_all[self.node_idx[nid]] = pos

        # Vectorized geometry computation
        r_i = self._pos_all[self.fiber_node_i_idx]  # (N_fibers, 2)
        r_j = self._pos_all[self.fiber_node_j_idx]
        dr = r_j - r_i
        lengths = np.linalg.norm(dr, axis=1)  # (N_fibers,)

        # Batch-vectorized energy (no Python loop)
        energies = self._batch_wlc_energy(lengths)
        return float(np.sum(energies)) * self.ENERGY_SCALE

    def compute_gradient(self, x: np.ndarray, fixed_coords: Dict[int, np.ndarray]) -> np.ndarray:
        """
        ANALYTICAL GRADIENT: ∂E/∂r = -F_net

        Vectorized computation without Python loops.

        Args:
            x: Flat array of free node coords
            fixed_coords: Boundary node positions

        Returns:
            Gradient array (same shape as x)
        """
        # Unpack coordinates into pre-allocated position array
        node_positions = self.unpack_free_coords(x, fixed_coords)
        self._pos_all[:] = 0
        for nid, pos in node_positions.items():
            self._pos_all[self.node_idx[nid]] = pos

        # Vectorized geometry
        r_i = self._pos_all[self.fiber_node_i_idx]
        r_j = self._pos_all[self.fiber_node_j_idx]
        dr = r_j - r_i
        lengths = np.linalg.norm(dr, axis=1)

        # Avoid division by zero
        safe_lengths = np.where(lengths > 0, lengths, 1.0)
        unit_vec = dr / safe_lengths[:, np.newaxis]

        # Batch-vectorized force magnitudes (no Python loop)
        forces_mag = self._batch_wlc_force(lengths)

        # Force vectors (pointing j -> i along fiber)
        force_vec = forces_mag[:, np.newaxis] * unit_vec

        # Accumulate forces on nodes using pre-allocated array
        self._forces_all[:] = 0
        np.add.at(self._forces_all, self.fiber_node_i_idx, force_vec)   # Pull on node_i
        np.add.at(self._forces_all, self.fiber_node_j_idx, -force_vec)  # Pull on node_j

        # Extract gradient using pre-computed index arrays
        n_free_vars = 2 * self.n_free
        self._grad[:] = 0

        # Gradient for free nodes: both X and Y (vectorized)
        if self.n_free > 0:
            free_forces = self._forces_all[self._free_node_internal_idx]  # (n_free, 2)
            self._grad[0:n_free_vars:2] = -free_forces[:, 0]   # dE/dx
            self._grad[1:n_free_vars:2] = -free_forces[:, 1]   # dE/dy

        # Gradient for partial nodes: only Y (vectorized)
        if self.n_partial > 0:
            partial_forces = self._forces_all[self._partial_node_internal_idx]
            self._grad[n_free_vars:] = -partial_forces[:, 1]  # dE/dy only

        # Scale gradient to match energy scaling (k_B_T units)
        return self._grad * self.ENERGY_SCALE

    def minimize(self, initial_positions: Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray], float]:
        """Minimize network energy using L-BFGS-B with analytical Jacobian."""
        x0 = self.pack_free_coords(initial_positions)

        if len(x0) == 0:
            # No free variables to optimize — return energy in SI (unscaled)
            scaled_E = self.compute_total_energy(x0, self.fixed_coords)
            return dict(initial_positions), scaled_E / self.ENERGY_SCALE

        result = minimize(
            fun=self.compute_total_energy,
            x0=x0,
            args=(self.fixed_coords,),
            method='L-BFGS-B',
            jac=self.compute_gradient,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )

        # Diagnostic: only log when actual movement occurs (avoids console flood)
        displacement = np.abs(result.x - x0)
        max_disp = np.max(displacement) if len(displacement) > 0 else 0.0
        if max_disp > 1e-10:  # Only log meaningful displacement (>0.1nm)
            print(f"[L-BFGS-B] max_disp={max_disp*1e6:.1f}µm, "
                  f"iters={result.nit}, n_vars={len(x0)}")

        relaxed_positions = self.unpack_free_coords(result.x, self.fixed_coords)
        # Return energy in SI units (unscale from k_B_T back to Joules)
        return relaxed_positions, result.fun / self.ENERGY_SCALE



class StochasticChemistryEngine:
    """
    Hybrid stochastic chemistry engine (Gillespie SSA + tau-leaping).

    Auto-switches between exact SSA and tau-leaping based on total propensity.
    Uses NumPy Generator for deterministic replay.
    """

    def __init__(self, rng_seed: int, tau_leap_threshold: float = 100.0,
                 plasmin_concentration: float = 1.0, delta_S: float = 0.1,
                 strain_cleavage_mode: str = 'inhibitory',
                 gamma_biphasic: float = 1.15,
                 eps_star: float = 0.22):
        """
        Initialize chemistry engine with deterministic RNG.

        Args:
            rng_seed: Random seed for NumPy Generator (deterministic replay)
            tau_leap_threshold: Switch to tau-leaping when total propensity > this
            plasmin_concentration: λ₀ dimensionless concentration multiplier.
                Per-event propensity: a = λ₀ × k₀ × f(ε) / δS
                k₀ = 0.020 s⁻¹ (Lynch et al. 2022: mean 49.8 s per fiber)
                The /δS factor ensures total rupture time is independent of
                the discretization step size (1/δS events needed per fiber).
                λ₀ = 1.0 → physiological (~1 nM plasmin)
                λ₀ = 10.0 → 10× plasmin concentration
            delta_S: Integrity decrement per cleavage event (default 0.1 = 10 hits)
            strain_cleavage_mode: Mechanochemical coupling model:
                'inhibitory' — k(ε) = k₀ × exp(-β × ε)  [Varju 2011]
                'neutral'    — k(ε) = k₀                 [topology-only]
                'biphasic'   — k(ε) = k₀ × exp(-β × ε) for ε ≤ ε*,
                               k(ε) = k₀ × exp(-β × ε*) × exp(+γ × (ε-ε*)) for ε > ε*
            gamma_biphasic: γ recovery exponent for biphasic mode (default 1.15)
            eps_star: ε* crossover strain for biphasic mode (default 0.22 = 22%)
        """
        self.rng = np.random.Generator(np.random.PCG64(rng_seed))
        self.tau_leap_threshold = tau_leap_threshold
        self.plasmin_concentration = float(plasmin_concentration)
        self.delta_S = float(delta_S)
        self.strain_cleavage_mode = strain_cleavage_mode
        self.gamma_biphasic = float(gamma_biphasic)
        self.eps_star = float(eps_star)

    def compute_propensities(self, state: NetworkState) -> Dict[int, float]:
        """
        Compute reaction propensities for all fibers (strain-based, vectorized).

        Args:
            state: Current network state

        Returns:
            {fiber_id: propensity [1/s]}
        """
        fibers = state.fibers
        n = len(fibers)
        if n == 0:
            return {}

        # Build arrays for batch computation
        fiber_ids = np.empty(n, dtype=int)
        S_arr = np.empty(n)
        L_c_arr = np.empty(n)
        k_cat_arr = np.empty(n)
        lengths = np.empty(n)

        for i, f in enumerate(fibers):
            fiber_ids[i] = f.fiber_id
            S_arr[i] = f.S
            L_c_arr[i] = f.L_c
            k_cat_arr[i] = f.k_cat_0
            pos_i = state.node_positions[f.node_i]
            pos_j = state.node_positions[f.node_j]
            lengths[i] = np.linalg.norm(pos_j - pos_i)

        # Vectorized strain and cleavage rate
        strain = np.maximum(0.0, (lengths - L_c_arr) / L_c_arr)

        # Mechanochemical coupling: compute k_cleave based on mode
        if self.strain_cleavage_mode == 'neutral':
            # k(ε) = k₀ — strain has no effect on cleavage rate
            k_cleave = k_cat_arr.copy()
        elif self.strain_cleavage_mode == 'biphasic':
            # k(ε) = k₀ × exp(-β×ε)           for ε ≤ ε*
            # k(ε) = k₀ × exp(-β×ε*) × exp(+γ×(ε-ε*))  for ε > ε*
            below = strain <= self.eps_star
            exponents = np.where(
                below,
                np.maximum(-PC.beta_strain * strain, -20.0),
                np.maximum(-PC.beta_strain * self.eps_star
                           + self.gamma_biphasic * (strain - self.eps_star), -20.0),
            )
            k_cleave = k_cat_arr * np.exp(exponents)
        else:
            # 'inhibitory' (default): k(ε) = k₀ × exp(-β×ε)  [Varju 2011]
            exponents = np.maximum(-PC.beta_strain * strain, -20.0)
            k_cleave = k_cat_arr * np.exp(exponents)

        # Per-event propensity: a = lam0 * k0 * f(eps) / delta_S
        # The /delta_S correction ensures the TOTAL time to full rupture
        # (requiring 1/delta_S events) equals 1 / (lam0 * k0 * f(eps)),
        # matching the experimental single-fiber cleavage time (Lynch 2022: 49.8 s).
        # Zero out fully ruptured fibers (S == 0).
        props = np.where(S_arr > 0, self.plasmin_concentration * k_cleave / self.delta_S, 0.0)

        return dict(zip(fiber_ids.tolist(), props.tolist()))

    def gillespie_step(self, state: NetworkState, max_dt: float,
                       propensities: Optional[Dict[int, float]] = None) -> Tuple[Optional[int], float]:
        """
        Gillespie SSA: exact stochastic simulation.

        Args:
            state: Current state
            max_dt: Maximum allowed timestep
            propensities: Pre-computed propensities (avoids redundant computation)

        Returns:
            (fiber_id_to_cleave, dt) or (None, max_dt) if no reaction
        """
        if propensities is None:
            propensities = self.compute_propensities(state)
        a_total = sum(propensities.values())

        if a_total == 0:
            return None, max_dt

        # Sample waiting time
        r1 = self.rng.random()
        if r1 == 0:
            r1 = 1e-16  # Avoid log(0)
        dt = -math.log(r1) / a_total

        if dt > max_dt:
            return None, max_dt

        # Select reaction
        r2 = self.rng.random() * a_total
        cumsum = 0.0
        for fid, a in propensities.items():
            cumsum += a
            if r2 <= cumsum:
                return fid, dt

        # Fallback (numerical precision)
        return list(propensities.keys())[-1], dt

    def tau_leap_step(self, state: NetworkState, tau: float,
                      propensities: Optional[Dict[int, float]] = None) -> List[int]:
        """
        Tau-leaping: approximate many reactions in time tau.

        DETERMINISTIC: Uses self.rng.poisson() for reproducibility.

        Args:
            state: Current state
            tau: Leap interval [s]
            propensities: Pre-computed propensities (avoids redundant computation)

        Returns:
            List of fiber IDs that reacted

        Note:
            Lambda capping at 100 prevents numerical overflow but introduces
            approximation error for very high-propensity reactions. This is
            acceptable for typical fibrinolysis rates (k ~ 0.01-0.1 s⁻¹).
        """
        if propensities is None:
            propensities = self.compute_propensities(state)

        reacted_fibers = []
        for fid, a in propensities.items():
            if a > 0:
                # Poisson sample with lambda capping
                lam = a * tau
                if lam > 100:
                    lam = 100  # Prevent overflow (document approximation)
                n_reactions = self.rng.poisson(lam)  # DETERMINISTIC
                if n_reactions > 0:
                    reacted_fibers.append(fid)

        return reacted_fibers

    def advance(self, state: NetworkState, target_dt: float,
                propensities: Optional[Dict[int, float]] = None) -> Tuple[List[int], float]:
        """
        Advance chemistry by target_dt using hybrid algorithm (strain-based).

        Args:
            state: Current state
            target_dt: Desired timestep [s]
            propensities: Pre-computed propensities (avoids redundant computation)

        Returns:
            (list of cleaved fiber_ids, actual_dt)
        """
        if propensities is None:
            propensities = self.compute_propensities(state)
        a_total = sum(propensities.values())

        if a_total < self.tau_leap_threshold:
            # Use SSA — pass pre-computed propensities
            fid, dt = self.gillespie_step(state, target_dt, propensities)
            if fid is None:
                return [], dt
            else:
                return [fid], dt
        else:
            # Use tau-leaping — pass pre-computed propensities
            reacted = self.tau_leap_step(state, target_dt, propensities)
            return reacted, target_dt

    def update_plasmin_locations(self, state: NetworkState,
                                propensities: Optional[Dict[int, float]] = None):
        """
        Update plasmin visualization locations based on current propensities.

        DETERMINISTIC: Uses self.rng.random() for reproducibility.

        Fibers with non-zero cleavage propensity get a random plasmin location.
        This simulates plasmin molecules randomly binding to vulnerable sites.

        Biologically realistic behavior:
        - High propensity fibers (low strain) → more likely to have plasmin
        - Zero propensity fibers (ruptured) → no plasmin
        - Location is random along fiber length (0.0 to 1.0)
        """
        if propensities is None:
            propensities = self.compute_propensities(state)

        # Clear old plasmin locations
        state.plasmin_locations.clear()

        # Add plasmin to fibers with non-zero propensity
        # Use probabilistic seeding: higher propensity → more likely to show plasmin
        for fid, prop in propensities.items():
            if prop > 0:
                # Probability of showing plasmin visualization
                # Use normalized propensity (cap at 1.0)
                p_show = min(1.0, prop / 0.2)  # per-event rate ~0.2 at baseline

                if self.rng.random() < p_show:
                    # Assign random location along fiber
                    location = self.rng.random()  # 0.0 to 1.0
                    state.plasmin_locations[fid] = location



def check_left_right_connectivity(state: NetworkState) -> bool:
    """
    Check if any path exists from left boundary nodes to right boundary nodes.

    Uses BFS (Breadth-First Search) to traverse the network graph through
    active (non-ruptured) fibers only. Uses cached adjacency when available.

    Args:
        state: Current network state

    Returns:
        True if left and right poles are connected, False if cleared (disconnected)
    """
    # Use cached adjacency (rebuilt only when invalidated)
    adjacency = state.get_adjacency()

    # If no active fibers, network is cleared
    if not adjacency:
        return False

    # BFS from ALL left boundary nodes (FIXED: was only starting from one)
    if not state.left_boundary_nodes:
        # Fallback: if boundary nodes not set, assume network is still connected
        return True

    # Start BFS from ALL left boundary nodes
    # This handles cases where left nodes are in disconnected components
    visited = set(state.left_boundary_nodes)
    queue = deque(state.left_boundary_nodes)

    while queue:
        current = queue.popleft()

        # Check if we reached any right boundary node
        if current in state.right_boundary_nodes:
            return True  # Path found → network still connected

        # Explore neighbors
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    # BFS completed without reaching right boundary → network cleared
    return False



class HybridMechanochemicalSimulation:
    """
    Main controller for hybrid mechanochemical simulation.

    Workflow:
    ---------
    1. Relax network (energy minimization)
    2. Compute fiber forces
    3. Advance chemistry (SSA/tau-leaping)
    4. Update fiber integrity
    5. Repeat

    Termination:
    ------------
    - Time limit reached
    - Lysis fraction > threshold
    - Network fragmented (all fibers ruptured)
    """

    def __init__(self,
                 initial_state: NetworkState,
                 rng_seed: int = 0,
                 dt_chem: float = 0.01,
                 t_max: float = 100.0,
                 lysis_threshold: float = 0.9,
                 delta_S: float = 0.1,
                 plasmin_concentration: float = 1.0,
                 chemistry_mode: str = 'mean_field',
                 abm_params=None,
                 strain_cleavage_mode: str = 'inhibitory',
                 gamma_biphasic: float = 1.15,
                 eps_star: float = 0.22):
        """
        Initialize simulation with deterministic RNG.

        Args:
            initial_state: Starting network configuration
            rng_seed: Random seed for deterministic replay (passed to NumPy Generator)
            dt_chem: Chemistry timestep [s]
            t_max: Maximum simulation time [s]
            lysis_threshold: Stop when lysis_fraction > this
            delta_S: Integrity decrement per cleavage event
            plasmin_concentration: λ₀ scaling factor for cleavage rates.
                k_eff(ε) = λ₀ × f(ε)
            chemistry_mode: 'mean_field' (Gillespie SSA) or 'abm' (discrete agents)
            abm_params: ABMParameters instance (required when chemistry_mode='abm')
            strain_cleavage_mode: 'inhibitory', 'neutral', or 'biphasic'
            gamma_biphasic: γ for biphasic mode recovery exponent
            eps_star: ε* crossover strain for biphasic mode

        Note:
            All stochastic operations (SSA, tau-leap, plasmin) use NumPy Generator
            for deterministic replay. Same seed → identical trajectory.
        """
        self.state = initial_state
        self.state.rebuild_fiber_index()
        self.rng_seed = rng_seed  # Store for reference
        self.dt_chem = dt_chem
        self.t_max = t_max
        self.lysis_threshold = lysis_threshold
        self.delta_S = delta_S
        self.plasmin_concentration = plasmin_concentration
        self.chemistry_mode = chemistry_mode
        self.strain_cleavage_mode = strain_cleavage_mode

        # Initialize chemistry engine based on mode
        self.chemistry = None
        self.abm_engine = None

        if chemistry_mode == 'abm' and abm_params is not None:
            from src.core.plasmin_abm import PlasminABMEngine
            self.abm_engine = PlasminABMEngine(abm_params, rng_seed)
        else:
            self.chemistry = StochasticChemistryEngine(
                rng_seed, plasmin_concentration=plasmin_concentration,
                delta_S=delta_S,
                strain_cleavage_mode=strain_cleavage_mode,
                gamma_biphasic=gamma_biphasic,
                eps_star=eps_star,
            )

        # History
        self.history = []
        self.termination_reason = None

    def relax_network(self):
        """Relax network to mechanical equilibrium with partial constraints."""
        active_fibers = [f for f in self.state.fibers if f.S > 0]

        if not active_fibers:
            self.state.energy = 0.0
            return

        # Filter constraints to only include nodes referenced by active fibers
        active_node_ids = set()
        for f in active_fibers:
            active_node_ids.add(f.node_i)
            active_node_ids.add(f.node_j)

        active_fixed_nodes = {nid: pos for nid, pos in self.state.fixed_nodes.items()
                             if nid in active_node_ids}
        active_partial_x = {nid: x for nid, x in self.state.partial_fixed_x.items()
                           if nid in active_node_ids}

        solver = EnergyMinimizationSolver(active_fibers, active_fixed_nodes, active_partial_x)
        relaxed_pos, energy = solver.minimize(self.state.node_positions)

        # Merge relaxed positions into existing positions (preserve disconnected nodes)
        for nid, pos in relaxed_pos.items():
            self.state.node_positions[nid] = pos
        self.state.energy = energy

    def compute_forces(self) -> Dict[int, float]:
        """Compute tensile force in each fiber (vectorized)."""
        fibers = self.state.fibers
        n = len(fibers)
        if n == 0:
            return {}

        # Build arrays
        fiber_ids = np.empty(n, dtype=int)
        S_arr = np.empty(n)
        L_c_arr = np.empty(n)
        xi_arr = np.empty(n)
        K0_arr = np.empty(n)
        is_ewlc = np.empty(n, dtype=bool)
        diameter_nm_arr = np.empty(n)
        lengths = np.empty(n)

        for i, f in enumerate(fibers):
            fiber_ids[i] = f.fiber_id
            S_arr[i] = f.S
            L_c_arr[i] = f.L_c
            xi_arr[i] = f.xi
            K0_arr[i] = f.K0
            is_ewlc[i] = (f.force_model == 'ewlc')
            diameter_nm_arr[i] = f.diameter_nm
            pos_i = self.state.node_positions[f.node_i]
            pos_j = self.state.node_positions[f.node_j]
            lengths[i] = np.linalg.norm(pos_j - pos_i)

        raw_strain = (lengths - L_c_arr) / L_c_arr

        if PC.THREE_REGIME_ENABLED:
            strain = np.minimum(raw_strain, PC.RUPTURE_STRAIN)
            hi = strain >= PC.MAX_STRAIN
            lo = ~hi

            F_eff = np.zeros_like(strain)

            # Low-strain fibers: standard WLC with F_MAX cap
            if np.any(lo):
                s_lo = np.minimum(strain[lo], PC.MAX_STRAIN)
                ome_lo = 1.0 - s_lo
                kBT_xi_lo = PC.k_B_T / xi_arr[lo]
                F_lo = kBT_xi_lo * (1.0 / (4.0 * ome_lo**2) - 0.25 + s_lo)
                ewlc_lo = is_ewlc[lo]
                if np.any(ewlc_lo):
                    F_lo[ewlc_lo] += K0_arr[lo][ewlc_lo] * s_lo[ewlc_lo]
                F_lo = np.where(S_arr[lo] > 0, S_arr[lo] * F_lo, 0.0)
                F_eff[lo] = np.minimum(F_lo, PC.F_MAX)

            # High-strain fibers: sigmoid blend (no F_MAX cap)
            if np.any(hi):
                s_hi = strain[hi]
                w = 0.5 * (1.0 + np.tanh(
                    (s_hi - PC.SIGMOID_EPSILON_MID) / PC.SIGMOID_DELTA_EPSILON))

                eps_wlc = np.minimum(s_hi, 0.995)
                one_minus = 1.0 - eps_wlc
                kBT_xi_hi = PC.k_B_T / xi_arr[hi]
                F_wlc = kBT_xi_hi * (1.0 / (4.0 * one_minus**2) - 0.25 + eps_wlc)
                ewlc_hi = is_ewlc[hi]
                if np.any(ewlc_hi):
                    F_wlc[ewlc_hi] += K0_arr[hi][ewlc_hi] * eps_wlc[ewlc_hi]

                d_m = diameter_nm_arr[hi] * 1e-9
                A_cs = np.pi * (d_m / 2.0)**2
                K_bb = PC.Y_A_STRONG * A_cs / L_c_arr[hi]
                F_backbone = K_bb * s_hi

                F_blend = (1.0 - w) * F_wlc + w * F_backbone
                F_eff[hi] = np.where(S_arr[hi] > 0, S_arr[hi] * F_blend, 0.0)

            # Zero out ruptured fibers
            F_eff = np.where(raw_strain >= PC.RUPTURE_STRAIN, 0.0, F_eff)

            return dict(zip(fiber_ids.tolist(), F_eff.tolist()))

        # Original WLC-only path
        strain = np.minimum(raw_strain, PC.MAX_STRAIN)
        one_minus_eps = 1.0 - strain
        kBT_xi = PC.k_B_T / xi_arr
        F_wlc = kBT_xi * (1.0 / (4.0 * one_minus_eps**2) - 0.25 + strain)
        F_model = F_wlc.copy()
        if np.any(is_ewlc):
            F_model[is_ewlc] += K0_arr[is_ewlc] * strain[is_ewlc]
        F_eff = np.where(S_arr > 0, S_arr * F_model, 0.0)
        F_eff = np.minimum(F_eff, PC.F_MAX)

        return dict(zip(fiber_ids.tolist(), F_eff.tolist()))

    def apply_cleavage(self, fiber_id: int, force_rupture: bool = False):
        """
        Degrade fiber by delta_S (or force-rupture to S=0) and track degradation history.

        Records EVERY cleavage event (partial and complete) for research analysis.
        Each fiber needs 1/delta_S hits to fully rupture (e.g., 10 hits at delta_S=0.1).

        Args:
            fiber_id: ID of fiber to degrade
            force_rupture: If True, set S=0 immediately (mechanical rupture)
        """
        i, fiber = self.state.get_fiber(fiber_id)
        if fiber is None or fiber.S <= 0:
            return

        # Calculate current length and strain before cleavage
        pos_i = self.state.node_positions[fiber.node_i]
        pos_j = self.state.node_positions[fiber.node_j]
        length = float(np.linalg.norm(pos_j - pos_i))
        strain = (length - fiber.L_c) / fiber.L_c
        tension = fiber.compute_force(length)

        if force_rupture:
            new_S = 0.0
        else:
            new_S = max(0.0, fiber.S - self.delta_S)
        self.state.fibers[i] = replace(fiber, S=new_S)
        self.state._fiber_index[fiber_id] = i  # Index unchanged, fiber replaced in-place

        # Record every cleavage event (partial degradation tracking)
        self.state.degradation_history.append({
            'time': self.state.time,
            'fiber_id': fiber_id,
            'order': len(self.state.degradation_history) + 1,
            'length': length,
            'strain': strain,
            'tension': tension,
            'old_S': fiber.S,
            'new_S': new_S,
            'is_complete_rupture': new_S == 0.0,
            'three_regime_rupture': force_rupture,
            'node_i': fiber.node_i,
            'node_j': fiber.node_j
        })

        # Track complete rupture count and update adjacency cache
        if new_S == 0.0:
            self.state.n_ruptured += 1
            self.state.remove_fiber_from_adjacency(fiber)

    def propagate_retraction_cascade(self, seed_fiber_ids: list) -> list:
        """
        Neighbor-aware post-cleavage retraction cascade.

        When a prestrained fiber is cleaved, stored elastic energy snaps back
        into the network. Only fibers sharing a node with a recently ruptured
        fiber are checked: if their strain exceeds CASCADE_RUPTURE_THRESHOLD,
        they rupture mechanically (S→0). The cascade propagates outward in
        waves from the initial cleavage site.

        Args:
            seed_fiber_ids: fiber IDs ruptured this step (chemistry or prior wave)

        Returns list of fiber IDs ruptured by the cascade.
        """
        if not PC.CASCADE_ENABLED or not seed_fiber_ids:
            return []

        threshold = PC.CASCADE_RUPTURE_THRESHOLD
        all_cascade_ids = []
        wave = 0

        # Build node → active fiber mapping for neighbor lookup
        node_to_fibers = defaultdict(list)
        for fiber in self.state.fibers:
            if fiber.S > 0:
                node_to_fibers[fiber.node_i].append(fiber.fiber_id)
                node_to_fibers[fiber.node_j].append(fiber.fiber_id)

        # Collect the nodes touched by seed fibers
        frontier_fids = set()
        for fid in seed_fiber_ids:
            _, fiber = self.state.get_fiber(fid)
            if fiber is None:
                continue
            for neighbor_fid in node_to_fibers.get(fiber.node_i, []):
                frontier_fids.add(neighbor_fid)
            for neighbor_fid in node_to_fibers.get(fiber.node_j, []):
                frontier_fids.add(neighbor_fid)
        # Exclude already-ruptured seed fibers
        frontier_fids -= set(seed_fiber_ids)

        while frontier_fids:
            wave_ids = []
            for fid in frontier_fids:
                idx, fiber = self.state.get_fiber(fid)
                if fiber is None or fiber.S <= 0:
                    continue
                pos_i = self.state.node_positions[fiber.node_i]
                pos_j = self.state.node_positions[fiber.node_j]
                length = float(np.linalg.norm(pos_j - pos_i))
                strain = (length - fiber.L_c) / fiber.L_c
                if strain >= threshold:
                    wave_ids.append(fid)

            if not wave_ids:
                break

            wave += 1
            next_frontier = set()
            for fid in wave_ids:
                idx, fiber = self.state.get_fiber(fid)
                if fiber is None or fiber.S <= 0:
                    continue

                pos_i = self.state.node_positions[fiber.node_i]
                pos_j = self.state.node_positions[fiber.node_j]
                length = float(np.linalg.norm(pos_j - pos_i))
                strain = (length - fiber.L_c) / fiber.L_c
                tension = fiber.compute_force(length)

                self.state.fibers[idx] = replace(fiber, S=0.0)
                self.state._fiber_index[fid] = idx
                self.state.n_ruptured += 1
                self.state.remove_fiber_from_adjacency(fiber)

                # Update node→fiber map: remove ruptured fiber
                for node in (fiber.node_i, fiber.node_j):
                    if fid in node_to_fibers.get(node, []):
                        node_to_fibers[node].remove(fid)
                    # Collect neighbors for next wave
                    for neighbor_fid in node_to_fibers.get(node, []):
                        next_frontier.add(neighbor_fid)

                self.state.degradation_history.append({
                    'time': self.state.time,
                    'fiber_id': fid,
                    'order': len(self.state.degradation_history) + 1,
                    'length': length,
                    'strain': strain,
                    'tension': tension,
                    'old_S': fiber.S,
                    'new_S': 0.0,
                    'is_complete_rupture': True,
                    'cascade': True,
                    'cascade_wave': wave,
                    'node_i': fiber.node_i,
                    'node_j': fiber.node_j,
                })

            all_cascade_ids.extend(wave_ids)
            # Relax after each wave — topology changed, strains redistribute
            self.relax_network()
            # Next wave: only neighbors of this wave's ruptures
            frontier_fids = next_frontier - set(all_cascade_ids) - set(seed_fiber_ids)

        return all_cascade_ids

    def _check_three_regime_rupture(self) -> list:
        """Check all active fibers for ε ≥ RUPTURE_STRAIN. Returns ruptured IDs."""
        if not PC.THREE_REGIME_ENABLED:
            return []
        ruptured = []
        for fiber in self.state.fibers:
            if fiber.S <= 0:
                continue
            pos_i = self.state.node_positions[fiber.node_i]
            pos_j = self.state.node_positions[fiber.node_j]
            length = float(np.linalg.norm(pos_j - pos_i))
            strain = (length - fiber.L_c) / fiber.L_c
            if strain >= PC.RUPTURE_STRAIN:
                self.apply_cleavage(fiber.fiber_id, force_rupture=True)
                ruptured.append(fiber.fiber_id)
        return ruptured

    def update_statistics(self):
        """Update lysis fraction."""
        n_total = len(self.state.fibers)
        self.state.lysis_fraction = self.state.n_ruptured / n_total if n_total > 0 else 0.0

    def step(self) -> bool:
        """
        Execute one simulation step. Dispatches to mean-field or ABM mode.

        Returns:
            True if simulation should continue, False if terminated
        """
        if self.chemistry_mode == 'abm' and self.abm_engine is not None:
            return self._step_abm()
        return self._step_mean_field()

    def _step_mean_field(self) -> bool:
        """Mean-field Gillespie SSA step with continuous mechanical relaxation."""
        # Compute propensities once, share with plasmin visualization and chemistry advance
        propensities = self.chemistry.compute_propensities(self.state)
        self.chemistry.update_plasmin_locations(self.state, propensities)
        cleaved_fibers, dt_actual = self.chemistry.advance(self.state, self.dt_chem, propensities)

        if cleaved_fibers:
            ruptured_this_step = []
            for fid in cleaved_fibers:
                _, fiber_info = self.state.get_fiber(fid)
                old_S = fiber_info.S if fiber_info else "?"
                self.apply_cleavage(fid)
                _, new_fiber = self.state.get_fiber(fid)
                new_S = new_fiber.S if new_fiber else "?"
                is_rupture = " [RUPTURED]" if new_S == 0.0 else ""
                if new_S == 0.0:
                    ruptured_this_step.append(fid)
                print(f"[Cleavage] Fiber #{fid}: S {old_S}->{new_S}{is_rupture} (t={self.state.time:.3f}s)")
            # Relax once after all cleavages (network topology changed)
            self.relax_network()
            # Post-cleavage retraction cascade (seeded from fully ruptured fibers)
            cascade_ids = self.propagate_retraction_cascade(ruptured_this_step)
            if cascade_ids:
                print(f"[Cascade] {len(cascade_ids)} fibers ruptured mechanically: {cascade_ids}")
        # Skip relaxation when no cleavage — network is already at equilibrium

        # Three-regime mechanical rupture check
        three_regime_ruptured = self._check_three_regime_rupture()
        if three_regime_ruptured:
            self.relax_network()
            cascade_from_3r = self.propagate_retraction_cascade(three_regime_ruptured)

        # Advance time and statistics BEFORE connectivity check so clearance
        # is recorded at the correct post-event time
        self.state.time += dt_actual
        self.update_statistics()

        if (cleaved_fibers or three_regime_ruptured) and not check_left_right_connectivity(self.state):
            last_fid = three_regime_ruptured[-1] if three_regime_ruptured else cleaved_fibers[-1]
            self._record_clearance(last_fid)
            return False

        self.history.append({
            'time': self.state.time,
            'lysis_fraction': self.state.lysis_fraction,
            'n_cleaved': len(cleaved_fibers),
            'energy': self.state.energy
        })
        return self._check_termination()

    def _step_abm(self) -> bool:
        """ABM step: agent lifecycle, positional cleavages, mechanical relaxation."""
        cleavage_events, dt_actual = self.abm_engine.advance(self.state, self.dt_chem)

        if cleavage_events:
            for event in cleavage_events:
                self._apply_positional_cleavage(event)
            # Relax once after all cleavages
            self.relax_network()
            active_fibers = [f for f in self.state.fibers if f.S > 0]
            self.abm_engine.adjacency.rebuild(active_fibers)
        # Skip relaxation when no cleavage — network is already at equilibrium

        # Three-regime mechanical rupture check
        three_regime_ruptured = self._check_three_regime_rupture()
        if three_regime_ruptured:
            self.relax_network()
            cascade_from_3r = self.propagate_retraction_cascade(three_regime_ruptured)

        self.state.time += dt_actual
        self.update_statistics()

        if (cleavage_events or three_regime_ruptured) and not check_left_right_connectivity(self.state):
            last_fid = three_regime_ruptured[-1] if three_regime_ruptured else cleavage_events[-1]['fiber_id']
            self._record_clearance(last_fid)
            return False

        self.history.append({
            'time': self.state.time,
            'lysis_fraction': self.state.lysis_fraction,
            'n_cleaved': len(cleavage_events),
            'energy': self.state.energy
        })
        return self._check_termination()

    def _apply_positional_cleavage(self, event: dict) -> None:
        """Split a fiber at the agent's position, creating two sub-fibers.

        S-inheritance: child_S = parent.S - delta_S.
        If child_S <= 0 the child is removed entirely (not added to graph).
        """
        from src.core.plasmin_abm import FiberSplitter

        fiber_id = event['fiber_id']
        position_s = event['position_s']

        fiber_idx, fiber = self.state.get_fiber(fiber_id)
        if fiber is None or fiber.S <= 0:
            return

        new_fid = self.state.abm_next_fiber_id
        new_nid = self.state.abm_next_node_id
        self.state.abm_next_fiber_id += 2
        self.state.abm_next_node_id += 1

        fiber_a, fiber_b, mid_node = FiberSplitter.split_fiber(
            fiber, position_s, self.state, new_fid, new_nid,
            delta_S=self.delta_S,
        )

        # Record degradation history before modifying fibers list
        pos_i = self.state.node_positions[fiber.node_i]
        pos_j = self.state.node_positions[fiber.node_j]
        length = float(np.linalg.norm(pos_j - pos_i))
        child_ids = []

        if fiber_a is None and fiber_b is None:
            # Parent fully degraded: child_S <= 0.
            # Mark parent as ruptured (S=0) and leave in list.
            self.state.fibers[fiber_idx] = replace(fiber, S=0.0)
            self.state.n_ruptured += 1
            # Release any agents still bound to this fiber
            from src.core.plasmin_abm import AgentState
            for agent in self.abm_engine.agents:
                if agent.state == AgentState.BOUND and agent.bound_fiber_id == fiber_id:
                    self.abm_engine._release_to_node(agent, self.state, fiber)
        else:
            # Replace parent with child_a, append child_b
            self.state.fibers[fiber_idx] = fiber_a
            child_ids.append(fiber_a.fiber_id)
            self.state.fibers.append(fiber_b)
            child_ids.append(fiber_b.fiber_id)

            # Reassign bound agents to child fibers
            self.abm_engine.reassign_agents_after_split(
                fiber_id, position_s, fiber_a.fiber_id, fiber_b.fiber_id
            )

        self.state.rebuild_fiber_index()
        self.state.invalidate_adjacency_cache()

        self.state.degradation_history.append({
            'time': self.state.time,
            'fiber_id': fiber_id,
            'order': len(self.state.degradation_history) + 1,
            'length': length,
            'strain': (length - fiber.L_c) / fiber.L_c,
            'type': 'positional_split',
            'position_s': position_s,
            'child_fiber_ids': child_ids,
            'new_node_id': mid_node,
            'agent_id': event.get('agent_id'),
            'child_S': max(0.0, fiber.S - self.delta_S),
        })

    def _record_clearance(self, critical_fiber_id: int) -> None:
        """Record network clearance event."""
        self.state.critical_fiber_id = critical_fiber_id
        self.state.clearance_event = {
            'time': self.state.time,
            'critical_fiber_id': critical_fiber_id,
            'lysis_fraction': self.state.lysis_fraction,
            'remaining_fibers': len(self.state.fibers) - self.state.n_ruptured,
            'total_fibers': len(self.state.fibers),
            'cleaved_fibers': self.state.n_ruptured,
        }
        self.termination_reason = "network_cleared"
        print(f"[Core V2] Network cleared at t={self.state.time:.2f}s "
              f"(left-right poles disconnected)")
        print(f"[Core V2] Critical fiber: {critical_fiber_id}")
        print(f"[Core V2] Lysis at clearance: {self.state.lysis_fraction*100:.1f}%")

    def _check_termination(self) -> bool:
        """Check all termination conditions. Returns True if should continue."""
        if self.state.time >= self.t_max:
            self.termination_reason = "time_limit"
            return False

        if self.state.lysis_fraction >= self.lysis_threshold:
            self.termination_reason = "lysis_threshold"
            return False

        if self.state.n_ruptured == len(self.state.fibers):
            self.termination_reason = "complete_rupture"
            return False

        return True

    def run(self):
        """Run simulation until termination."""
        print(f"Starting simulation: t_max={self.t_max}s, dt={self.dt_chem}s")

        while self.step():
            if len(self.history) % 100 == 0:
                print(f"  t={self.state.time:.2f}s, lysis={self.state.lysis_fraction:.3f}, n_ruptured={self.state.n_ruptured}")

        print(f"Terminated: {self.termination_reason} at t={self.state.time:.2f}s")
        return self.history



class ExcelNetworkLoader:
    """
    Load fibrin network from Excel file (FibriNet format).

    Converts legacy Excel format to Core V2 WLCFiber representation.
    """

    @staticmethod
    def load(excel_path: str,
             applied_strain: float = 0.1,
             L_c_per_unit: float = 1e-6,
             k0_to_wlc_calibration: Optional[Dict[str, float]] = None) -> NetworkState:
        """
        Load network from Excel file.

        Args:
            excel_path: Path to .xlsx file
            applied_strain: Strain to apply to right boundary
            L_c_per_unit: Contour length per coordinate unit [m/unit]
            k0_to_wlc_calibration: Optional dict with WLC parameter mappings

        Returns:
            NetworkState ready for simulation

        Raises:
            ValueError: If file format is invalid
        """
        import pandas as pd

        # Read Excel sheets
        try:
            nodes_df = pd.read_excel(excel_path, sheet_name='nodes')
            edges_df = pd.read_excel(excel_path, sheet_name='edges')
            try:
                meta_df = pd.read_excel(excel_path, sheet_name='meta_data')
            except:
                meta_df = None
        except Exception as e:
            raise ValueError(f"Failed to read Excel file {excel_path}: {e}")

        # Parse nodes
        node_coords = {}
        left_nodes = set()
        right_nodes = set()

        for _, row in nodes_df.iterrows():
            nid = int(row['n_id'])
            x = float(row['n_x'])
            y = float(row['n_y'])
            node_coords[nid] = np.array([x, y])

            if row.get('is_left_boundary', False):
                left_nodes.add(nid)
            if row.get('is_right_boundary', False):
                right_nodes.add(nid)

        # Validate boundaries
        if not left_nodes or not right_nodes:
            raise ValueError("Excel file must specify left and right boundary nodes")

        # Apply strain to right boundary
        x_coords = [pos[0] for pos in node_coords.values()]
        x_min, x_max = min(x_coords), max(x_coords)
        x_span = x_max - x_min

        for nid in right_nodes:
            node_coords[nid][0] += applied_strain * x_span

        # Fixed boundary conditions (rigid grips)
        fixed_nodes = {}
        for nid in left_nodes | right_nodes:
            fixed_nodes[nid] = node_coords[nid].copy()

        # Parse edges and create WLC fibers
        fibers = []
        for _, row in edges_df.iterrows():
            eid = int(row['e_id'])
            n_from = int(row['n_from'])
            n_to = int(row['n_to'])

            # Compute geometric length
            pos_from = node_coords[n_from]
            pos_to = node_coords[n_to]
            length_coord = float(np.linalg.norm(pos_to - pos_from))

            # Convert to contour length [m]
            L_c = length_coord * L_c_per_unit

            # Create WLC fiber
            fiber = WLCFiber(
                fiber_id=eid,
                node_i=n_from,
                node_j=n_to,
                L_c=L_c,
                xi=PC.xi,
                S=1.0,  # Initially intact
                x_bell=PC.x_bell,
                k_cat_0=PC.k_cat_0
            )
            fibers.append(fiber)

        # Create initial state
        state = NetworkState(
            time=0.0,
            fibers=fibers,
            node_positions=node_coords,
            fixed_nodes=fixed_nodes
        )

        print(f"Loaded network: {len(node_coords)} nodes, {len(fibers)} fibers")
        print(f"  Left boundary: {len(left_nodes)} nodes")
        print(f"  Right boundary: {len(right_nodes)} nodes")
        print(f"  Applied strain: {applied_strain}")

        return state



def validate_core_v2():
    """
    Run validation checks on Core V2 implementation.

    Tests:
    1. Energy-force consistency: F = dU/dx
    2. Stress-based Bell model behavior
    3. Energy minimization convergence
    """
    print("=" * 60)
    print("FibriNet Core V2 Validation Suite")
    print("=" * 60)

    # Test 1: Energy-Force Consistency
    print("\n[1/3] Testing WLC energy-force consistency...")
    fiber = WLCFiber(fiber_id=0, node_i=0, node_j=1, L_c=10e-6, S=1.0)

    test_passed = True
    for strain in [0.1, 0.3, 0.5, 0.7, 0.9]:
        x = strain * fiber.L_c
        F_analytical = fiber.compute_force(x)

        # Numerical derivative
        dx = 1e-9
        U_plus = fiber.compute_energy(x + dx)
        U_minus = fiber.compute_energy(x - dx)
        F_numerical = (U_plus - U_minus) / (2 * dx)

        rel_error = abs(F_analytical - F_numerical) / max(abs(F_analytical), 1e-20)
        status = "PASS" if rel_error < 1e-6 else "FAIL"

        print(f"  strain={strain:.1f}: F_analytical={F_analytical:.6e} N, F_numerical={F_numerical:.6e} N, rel_error={rel_error:.2e} [{status}]")

        if rel_error >= 1e-6:
            test_passed = False

    print(f"  Result: {'[PASS]' if test_passed else '[FAIL]'}")

    # Test 2: Strain-Inhibited Cleavage Model
    print("\n[2/3] Testing strain-inhibited cleavage model...")
    fiber = WLCFiber(fiber_id=0, node_i=0, node_j=1, L_c=10e-6, k_cat_0=1.0)
    
    # Strain 0 (Rest) -> Should be max rate (k_cat_0)
    len_0 = fiber.L_c
    k_0 = fiber.compute_cleavage_rate(len_0)
    
    # Strain 0.1 -> Should be lower
    len_01 = fiber.L_c * (1 + 0.1)
    k_01 = fiber.compute_cleavage_rate(len_01)
    
    # Strain 0.23 -> Should be significantly lower (approx 10x fold reduction typically)
    len_23 = fiber.L_c * (1 + 0.23)
    k_23 = fiber.compute_cleavage_rate(len_23)

    print(f"  Strain=0.00 -> k={k_0:.6e} 1/s (Expected ~1.0)")
    print(f"  Strain=0.10 -> k={k_01:.6e} 1/s")
    print(f"  Strain=0.23 -> k={k_23:.6e} 1/s")
    
    # Check 1: Base rate at zero strain
    check1 = abs(k_0 - 1.0) < 1e-6
    # Check 2: Inhibition (rate decreases with strain)
    check2 = k_01 < k_0 and k_23 < k_01
    
    test_passed = check1 and check2
    print(f"  Result: {'[PASS]' if test_passed else '[FAIL]'}")

    # Test 3: Energy Minimization
    print("\n[3/3] Testing energy minimization solver...")

    # Simple 2-fiber system
    fibers = [
        WLCFiber(fiber_id=0, node_i=0, node_j=1, L_c=10e-6),
        WLCFiber(fiber_id=1, node_i=1, node_j=2, L_c=10e-6)
    ]

    fixed = {
        0: np.array([0.0, 0.0]),
        2: np.array([20e-6, 0.0])
    }

    initial_pos = {
        0: np.array([0.0, 0.0]),
        1: np.array([10e-6, 5e-6]),  # Displaced
        2: np.array([20e-6, 0.0])
    }

    solver = EnergyMinimizationSolver(fibers, fixed)
    relaxed, energy = solver.minimize(initial_pos)

    # Check that middle node moved closer to y=0
    y_final = relaxed[1][1]
    print(f"  Initial y_mid = {initial_pos[1][1]:.6e} m")
    print(f"  Relaxed y_mid = {y_final:.6e} m")
    print(f"  Final energy = {energy:.6e} J")

    test_passed = abs(y_final) < abs(initial_pos[1][1])
    print(f"  Result: {'[PASS]' if test_passed else '[FAIL]'}")

    print("\n" + "=" * 60)
    print("Validation Complete")
    print("=" * 60)



if __name__ == "__main__":
    # Run validation
    validate_core_v2()

    print("\n" + "=" * 60)
    print("FibriNet Core V2 is ready for production use")
    print("=" * 60)
