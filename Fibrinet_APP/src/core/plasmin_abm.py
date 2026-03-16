"""
Plasmin Agent-Based Model (ABM) for FibriNet.

Discrete plasmin agents enter from network boundaries (wavefront lysis),
bind/unbind fibers with strain-dependent kinetics, cleave fibers at specific
positions (splitting fibers in two), and diffuse via graph-based hops.

Binding uses a bimolecular rate: k_on_eff [s^-1] = k_on2 [M^-1 s^-1] x C [M],
where C = plasmin_concentration_nM * 1e-9 (nM -> M).

References:
    Longstaff et al. (1993)     — k_on2 ~ 1e5 M^-1 s^-1
    Litvinov et al. (2018)      — Bell model: k_off(F) = k_off0 * exp(F*d/kBT)
    Adhikari et al. (2012)      — Strain inhibition: exp(-beta*strain)
    Bannish et al. (2014)       — Positional cleavage
    Diamond & Anand (1993)      — Wavefront entry
    Bell (1978)                 — Force-dependent unbinding, delta=0.5 nm
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from enum import Enum
import numpy as np
import math


class AgentState(Enum):
    IN_SOLUTION = "in_solution"
    BOUND = "bound"
    COOLDOWN = "cooldown"


# Strain-cleavage relationship models

def exponential_inhibition(strain: float, k0: float, beta: float) -> float:
    """k(e) = k0 * exp(-beta * e), Adhikari et al. (2012)."""
    strain = max(0.0, strain)
    exponent = max(-beta * strain, -20.0)
    return k0 * math.exp(exponent)


def linear_inhibition(strain: float, k0: float, beta: float) -> float:
    """k(e) = k0 * max(0, 1 - beta * e)."""
    return k0 * max(0.0, 1.0 - beta * max(0.0, strain))


def constant_rate(strain: float, k0: float, beta: float) -> float:
    """k(e) = k0, strain-independent baseline."""
    return k0


STRAIN_CLEAVAGE_MODELS: Dict[str, Callable[[float, float, float], float]] = {
    'exponential': exponential_inhibition,
    'linear': linear_inhibition,
    'constant': constant_rate,
}


@dataclass
class ABMParameters:
    """Tunable ABM parameters, populated from GUI or defaults."""

    n_agents: int = 10
    auto_agent_count: bool = True
    plasmin_concentration_nM: float = 1.0  # [nM], maps to lambda_0

    # Binding: bimolecular, Longstaff et al. (1993)
    # k_on_eff [s^-1] = k_on2 [M^-1 s^-1] x C_plasmin [M]
    k_on2: float = 1e5            # M^-1 s^-1
    alpha_on: float = 5.0

    # Unbinding: Bell model, Kd = k_off0/k_on_eff ≈ 10 nM
    k_off0: float = 0.001         # s^-1
    delta_off: float = 0.5e-9     # m (Bell 1978)

    # Cleavage: Lynch et al. (2022), mean single-fiber time ≈ 49.8 s
    k_cat0: float = 0.020         # s^-1
    beta_cat: float = 0.84  # Varjú et al. (2011)

    strain_cleavage_model: str = 'exponential'
    p_stay: float = 0.5

    # Runtime toggles
    strain_dependent_k_on: bool = True
    strain_dependent_k_off: bool = True
    update_kcat_dynamic: bool = True  # recompute k_cat each timestep from current strain

    def get_strain_cleavage_fn(self) -> Callable[[float, float, float], float]:
        return STRAIN_CLEAVAGE_MODELS.get(
            self.strain_cleavage_model, exponential_inhibition
        )

    @staticmethod
    def compute_agent_count(concentration_nM: float, volume_um3: float) -> int:
        """N = C_M * V_m3 * N_A = C_nM * V_um3 * 6.022e-4."""
        return max(1, round(concentration_nM * volume_um3 * 6.022e-4))


@dataclass(slots=True)
class PlasminAgent:
    """Mutable state for a single plasmin agent."""

    agent_id: int
    state: AgentState
    current_node: Optional[int] = None
    bound_fiber_id: Optional[int] = None
    bound_position_s: Optional[float] = None
    k_cat_at_binding: Optional[float] = None
    entry_time: float = 0.0
    entry_node: Optional[int] = None
    n_cleavages: int = 0
    n_bindings: int = 0
    n_kcat_updates: int = 0
    max_kcat_delta: float = 0.0


class NetworkAdjacency:
    """Graph structure for agent hop diffusion. O(1) neighbor lookup."""

    def __init__(self):
        self.node_to_fibers: Dict[int, Set[int]] = {}
        self.node_to_neighbors: Dict[int, Set[int]] = {}
        self.fiber_endpoints: Dict[int, Tuple[int, int]] = {}

    def rebuild(self, fibers) -> None:
        self.node_to_fibers.clear()
        self.node_to_neighbors.clear()
        self.fiber_endpoints.clear()

        for f in fibers:
            fid = f.fiber_id
            ni, nj = f.node_i, f.node_j
            self.fiber_endpoints[fid] = (ni, nj)
            self.node_to_fibers.setdefault(ni, set()).add(fid)
            self.node_to_fibers.setdefault(nj, set()).add(fid)
            self.node_to_neighbors.setdefault(ni, set()).add(nj)
            self.node_to_neighbors.setdefault(nj, set()).add(ni)

    def get_neighbors(self, node_id: int) -> Set[int]:
        return self.node_to_neighbors.get(node_id, set())

    def get_fibers_at_node(self, node_id: int) -> Set[int]:
        return self.node_to_fibers.get(node_id, set())

    def get_random_neighbor(self, node_id: int, rng: np.random.Generator) -> Optional[int]:
        neighbors = self.node_to_neighbors.get(node_id, set())
        if not neighbors:
            return None
        return rng.choice(list(neighbors))

    def get_all_nodes(self) -> Set[int]:
        return set(self.node_to_neighbors.keys())


class FiberSplitter:
    """Positional cleavage: split a fiber at position s into two sub-fibers."""

    @staticmethod
    def split_fiber(fiber, position_s: float, state,
                    next_fiber_id: int, next_node_id: int,
                    delta_S: float = 0.1):
        """Split fiber into (fiber_a, fiber_b, new_node_id).

        Interpolates a new node at position_s along the fiber, splits L_c
        proportionally. Children inherit damaged integrity:
            child_S = parent.S - delta_S
        If child_S <= 0 the child is not created (returns None for that slot).

        Returns:
            (fiber_a_or_None, fiber_b_or_None, new_node_id)
        """
        from src.core.fibrinet_core_v2 import WLCFiber

        position_s = max(0.01, min(0.99, position_s))

        child_S = fiber.S - delta_S
        if child_S <= 0:
            # Both children would be dead — parent is fully degraded
            return None, None, next_node_id

        pos_i = state.node_positions[fiber.node_i]
        pos_j = state.node_positions[fiber.node_j]
        state.node_positions[next_node_id] = (pos_i + position_s * (pos_j - pos_i)).copy()

        L_c_a = max(1e-9, position_s * fiber.L_c)
        L_c_b = max(1e-9, (1.0 - position_s) * fiber.L_c)

        fiber_a = WLCFiber(
            fiber_id=next_fiber_id,
            node_i=fiber.node_i, node_j=next_node_id,
            L_c=L_c_a, xi=fiber.xi, S=child_S,
            x_bell=fiber.x_bell, k_cat_0=fiber.k_cat_0,
            force_model=fiber.force_model, K0=fiber.K0,
            diameter_nm=fiber.diameter_nm,
        )
        fiber_b = WLCFiber(
            fiber_id=next_fiber_id + 1,
            node_i=next_node_id, node_j=fiber.node_j,
            L_c=L_c_b, xi=fiber.xi, S=child_S,
            x_bell=fiber.x_bell, k_cat_0=fiber.k_cat_0,
            force_model=fiber.force_model, K0=fiber.K0,
            diameter_nm=fiber.diameter_nm,
        )
        return fiber_a, fiber_b, next_node_id


class PlasminABMEngine:
    """Discrete agent-based plasmin simulation engine.

    Replaces StochasticChemistryEngine when ABM mode is active.
    """

    def __init__(self, params: ABMParameters, rng_seed: int = 0):
        self.params = params
        self.rng = np.random.Generator(np.random.PCG64(rng_seed))
        self.agents: List[PlasminAgent] = []
        self.adjacency = NetworkAdjacency()
        self._next_agent_id = 0
        self._agents_spawned = 0
        self._spawn_queue: List[int] = []
        self._initialized = False
        self._total_splits = 0
        self._kcat_max_delta_this_step = 0.0
        self._kBT = 1.380649e-23 * 310.15  # Thermal energy at 37 C

    def initialize(self, state) -> None:
        """Build adjacency graph, compute agent count, prepare spawn queue."""
        active_fibers = [f for f in state.fibers if f.S > 0]
        self.adjacency.rebuild(active_fibers)

        if self.params.auto_agent_count:
            xs = [pos[0] for pos in state.node_positions.values()]
            ys = [pos[1] for pos in state.node_positions.values()]
            x_span = max(xs) - min(xs) if xs else 1e-6
            y_span = max(ys) - min(ys) if ys else 1e-6
            depth = 1e-6  # 2D network assumed 1 um thick
            volume_um3 = x_span * y_span * depth * 1e18
            self.params.n_agents = ABMParameters.compute_agent_count(
                self.params.plasmin_concentration_nM, volume_um3
            )
            print(f"[ABM] Auto agent count: {self.params.n_agents} "
                  f"(C={self.params.plasmin_concentration_nM:.1f} nM, "
                  f"V={volume_um3:.0f} um3)")

        boundary_nodes = sorted(state.left_boundary_nodes | state.right_boundary_nodes)
        if not boundary_nodes:
            boundary_nodes = sorted(state.node_positions.keys())[:1]

        self._spawn_queue = [
            boundary_nodes[i % len(boundary_nodes)]
            for i in range(self.params.n_agents)
        ]
        self._initialized = True
        print(f"[ABM] Initialized: {self.params.n_agents} agents, "
              f"{len(active_fibers)} fibers, "
              f"{len(self.adjacency.get_all_nodes())} nodes")

    def advance(self, state, dt: float) -> Tuple[List[dict], float]:
        """Execute one ABM lifecycle step. Returns (cleavage_events, dt)."""
        if not self._initialized:
            self.initialize(state)

        self._spawn_agents(state, dt)
        self._step_unbound_agents(state, dt)
        cleavage_events = self._step_bound_agents(state, dt)
        return cleavage_events, dt

    def _spawn_agents(self, state, dt: float) -> None:
        if self._agents_spawned >= self.params.n_agents:
            return
        node_id = self._spawn_queue[self._agents_spawned]
        agent = PlasminAgent(
            agent_id=self._next_agent_id,
            state=AgentState.IN_SOLUTION,
            current_node=node_id,
            entry_time=state.time,
            entry_node=node_id,
        )
        self.agents.append(agent)
        self._next_agent_id += 1
        self._agents_spawned += 1

    def _step_unbound_agents(self, state, dt: float) -> None:
        for agent in self.agents:
            if agent.state == AgentState.COOLDOWN:
                next_node = self.adjacency.get_random_neighbor(agent.current_node, self.rng)
                if next_node is not None:
                    agent.current_node = next_node
                agent.state = AgentState.IN_SOLUTION
                # No binding attempt on the same step as a cooldown hop

            elif agent.state == AgentState.IN_SOLUTION:
                next_node = self.adjacency.get_random_neighbor(agent.current_node, self.rng)
                if next_node is not None:
                    agent.current_node = next_node
                self._attempt_binding(agent, state, dt)

    def _attempt_binding(self, agent: PlasminAgent, state, dt: float) -> bool:
        """Try to bind agent to a fiber at its current node."""
        fibers_at_node = self.adjacency.get_fibers_at_node(agent.current_node)
        if not fibers_at_node:
            return False

        fiber_ids = list(fibers_at_node)
        self.rng.shuffle(fiber_ids)

        for fid in fiber_ids:
            fiber = self._find_fiber(state, fid)
            if fiber is None or fiber.S <= 0:
                continue

            strain = self._compute_fiber_strain(fiber, state)

            if self.params.strain_dependent_k_on:
                k_on2 = self.params.k_on2 * math.exp(-self.params.alpha_on * max(0.0, strain))
            else:
                k_on2 = self.params.k_on2

            # Bimolecular binding: k_on_eff [s^-1] = k_on2 [M^-1 s^-1] x C [M]
            C_plasmin_M = self.params.plasmin_concentration_nM * 1e-9
            k_on_eff = k_on2 * C_plasmin_M
            p_bind = 1.0 - math.exp(-k_on_eff * dt)

            if self.rng.random() < p_bind:
                agent.state = AgentState.BOUND
                agent.bound_fiber_id = fid
                agent.bound_position_s = float(self.rng.random())
                agent.current_node = None
                agent.n_bindings += 1

                # C1 rule: capture k_cat at binding
                cleavage_fn = self.params.get_strain_cleavage_fn()
                agent.k_cat_at_binding = cleavage_fn(
                    strain, self.params.k_cat0, self.params.beta_cat
                )
                return True

        return False

    def _step_bound_agents(self, state, dt: float) -> List[dict]:
        cleavage_events = []

        # Gather bound agents for batch processing
        bound_agents = [a for a in self.agents if a.state == AgentState.BOUND]
        if not bound_agents:
            return cleavage_events

        # Batch-resolve fibers and release agents on dead fibers
        active_bound = []
        for agent in bound_agents:
            fiber = self._find_fiber(state, agent.bound_fiber_id)
            if fiber is None or fiber.S <= 0:
                self._release_to_node(agent, state, fiber)
            else:
                active_bound.append((agent, fiber))

        if not active_bound:
            return cleavage_events

        n = len(active_bound)

        # Batch-compute forces and rates for unbinding (Bell model)
        forces = np.empty(n)
        for i, (agent, fiber) in enumerate(active_bound):
            forces[i] = abs(fiber.compute_force(self._compute_fiber_length(fiber, state)))

        if self.params.strain_dependent_k_off:
            bell_exp = np.minimum(forces * self.params.delta_off / self._kBT, 20.0)
            k_off_arr = self.params.k_off0 * np.exp(bell_exp)
        else:
            k_off_arr = np.full(n, self.params.k_off0)

        p_unbind = 1.0 - np.exp(-k_off_arr * dt)
        r_unbind = self.rng.random(n)

        # Batch-compute cleavage rates
        k_cat_arr = np.empty(n)
        if self.params.update_kcat_dynamic:
            cleavage_fn = self.params.get_strain_cleavage_fn()
            for i, (agent, fiber) in enumerate(active_bound):
                strain = self._compute_fiber_strain(fiber, state)
                new_kcat = cleavage_fn(strain, self.params.k_cat0, self.params.beta_cat)
                k_cat_arr[i] = new_kcat
                if agent.k_cat_at_binding is not None:
                    delta = abs(new_kcat - agent.k_cat_at_binding)
                    agent.max_kcat_delta = max(agent.max_kcat_delta, delta)
                    agent.n_kcat_updates += 1
                agent.k_cat_at_binding = new_kcat
        else:
            for i, (agent, fiber) in enumerate(active_bound):
                k_cat_arr[i] = agent.k_cat_at_binding or 0.0

        p_cleave = 1.0 - np.exp(-k_cat_arr * dt)
        r_cleave = self.rng.random(n)

        # Apply decisions
        for i, (agent, fiber) in enumerate(active_bound):
            if r_unbind[i] < p_unbind[i]:
                self._release_to_node(agent, state, fiber)
                continue

            if r_cleave[i] < p_cleave[i]:
                agent.n_cleavages += 1
                self._total_splits += 1
                print(f"[ABM] Fiber #{agent.bound_fiber_id} split at "
                      f"s={agent.bound_position_s:.2f} by agent #{agent.agent_id} "
                      f"(t={state.time:.3f}s)")
                event = {
                    'fiber_id': agent.bound_fiber_id,
                    'position_s': agent.bound_position_s,
                    'agent_id': agent.agent_id,
                    'type': 'split',
                }
                cleavage_events.append(event)
                self._handle_post_cleavage(agent, fiber, state)

        self._kcat_max_delta_this_step = max(
            (a.max_kcat_delta for a in self.agents if a.state == AgentState.BOUND),
            default=0.0,
        )

        return cleavage_events

    def _attempt_unbinding(self, agent: PlasminAgent, fiber,
                           state, dt: float) -> bool:
        """Bell model unbinding: k_off(F) = k_off0 * exp(F*d/kBT)."""
        if self.params.strain_dependent_k_off:
            force = abs(fiber.compute_force(self._compute_fiber_length(fiber, state)))
            bell_exp = min(force * self.params.delta_off / self._kBT, 20.0)
            k_off = self.params.k_off0 * math.exp(bell_exp)
        else:
            k_off = self.params.k_off0

        if self.rng.random() < (1.0 - math.exp(-k_off * dt)):
            self._release_to_node(agent, state, fiber)
            return True
        return False

    def _attempt_cleavage(self, agent: PlasminAgent, fiber,
                          state, dt: float) -> Optional[dict]:
        """Test for cleavage event using configured strain-cleavage model."""
        if self.params.update_kcat_dynamic:
            strain = self._compute_fiber_strain(fiber, state)
            cleavage_fn = self.params.get_strain_cleavage_fn()
            k_cat = cleavage_fn(strain, self.params.k_cat0, self.params.beta_cat)
            if agent.k_cat_at_binding is not None:
                delta = abs(k_cat - agent.k_cat_at_binding)
                agent.max_kcat_delta = max(agent.max_kcat_delta, delta)
                agent.n_kcat_updates += 1
            agent.k_cat_at_binding = k_cat
        else:
            k_cat = agent.k_cat_at_binding or 0.0

        if self.rng.random() < (1.0 - math.exp(-k_cat * dt)):
            agent.n_cleavages += 1
            self._total_splits += 1
            print(f"[ABM] Fiber #{agent.bound_fiber_id} split at "
                  f"s={agent.bound_position_s:.2f} by agent #{agent.agent_id} "
                  f"(t={state.time:.3f}s)")
            return {
                'fiber_id': agent.bound_fiber_id,
                'position_s': agent.bound_position_s,
                'agent_id': agent.agent_id,
                'type': 'split',
            }
        return None

    def _handle_post_cleavage(self, agent: PlasminAgent, fiber, state) -> None:
        if self.rng.random() < self.params.p_stay:
            pass  # Agent stays BOUND; caller handles reassignment to child fiber
        else:
            self._release_to_node(agent, state, fiber)

    def reassign_agents_after_split(self, old_fiber_id: int, split_s: float,
                                    fiber_a_id: int, fiber_b_id: int) -> None:
        """Reassign agents bound to a split fiber to the correct child.

        Agents at position_s < split_s go to fiber_a; rescaled to [0,1].
        Agents at position_s >= split_s go to fiber_b; rescaled to [0,1].
        """
        for agent in self.agents:
            if agent.state == AgentState.BOUND and agent.bound_fiber_id == old_fiber_id:
                if agent.bound_position_s < split_s:
                    agent.bound_fiber_id = fiber_a_id
                    agent.bound_position_s = agent.bound_position_s / split_s
                else:
                    agent.bound_fiber_id = fiber_b_id
                    agent.bound_position_s = (
                        (agent.bound_position_s - split_s) / (1.0 - split_s)
                    )

    def _find_fiber(self, state, fiber_id: int):
        """O(1) fiber lookup via state index."""
        _, fiber = state.get_fiber(fiber_id)
        return fiber

    def _compute_fiber_length(self, fiber, state) -> float:
        pos_i = state.node_positions[fiber.node_i]
        pos_j = state.node_positions[fiber.node_j]
        return float(np.linalg.norm(pos_j - pos_i))

    def _compute_fiber_strain(self, fiber, state) -> float:
        length = self._compute_fiber_length(fiber, state)
        return max(0.0, (length - fiber.L_c) / fiber.L_c)

    def _release_to_node(self, agent: PlasminAgent, state, fiber) -> None:
        """Release agent to the closer endpoint of its bound fiber."""
        if fiber is not None:
            if agent.bound_position_s is not None and agent.bound_position_s < 0.5:
                agent.current_node = fiber.node_i
            else:
                agent.current_node = fiber.node_j
        elif state.node_positions:
            agent.current_node = next(iter(state.node_positions))

        agent.state = AgentState.COOLDOWN
        agent.bound_fiber_id = None
        agent.bound_position_s = None
        agent.k_cat_at_binding = None

    def get_agent_locations(self) -> Dict[str, List]:
        bound, unbound, cooldown = [], [], []
        for agent in self.agents:
            if agent.state == AgentState.BOUND:
                bound.append((agent.bound_fiber_id, agent.bound_position_s, agent.agent_id))
            elif agent.state == AgentState.IN_SOLUTION:
                unbound.append((agent.current_node, agent.agent_id))
            elif agent.state == AgentState.COOLDOWN:
                cooldown.append((agent.current_node, agent.agent_id))
        return {'bound': bound, 'unbound': unbound, 'cooldown': cooldown}

    def get_statistics(self) -> Dict[str, Any]:
        n_bound = sum(1 for a in self.agents if a.state == AgentState.BOUND)
        n_free = sum(1 for a in self.agents
                     if a.state in (AgentState.IN_SOLUTION, AgentState.COOLDOWN))
        return {
            'total': len(self.agents),
            'spawned': self._agents_spawned,
            'target': self.params.n_agents,
            'bound': n_bound,
            'free': n_free,
            'total_splits': self._total_splits,
            'total_bindings': sum(a.n_bindings for a in self.agents),
            'total_cleavages': sum(a.n_cleavages for a in self.agents),
            'kcat_max_delta': self._kcat_max_delta_this_step,
            'kcat_dynamic': self.params.update_kcat_dynamic,
        }
