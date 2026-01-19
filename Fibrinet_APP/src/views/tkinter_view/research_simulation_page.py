from utils.logger.logger import Logger
import tkinter as tk
from tkinter import filedialog, messagebox
from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Any, Mapping, Callable, Sequence
import math
import random
import os
import sys
import csv
import pandas as pd
import numpy as np
import json
import tempfile
import hashlib
import copy
import time
from tkinter import simpledialog
from .tkinter_view import TkinterView
from src.config.feature_flags import FeatureFlags
from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter

#
# Phase 2.0 (force-dependent degradation) constants:
# - Deterministic, fixed (no UI yet)
# - Setting PHASE2_FORCE_ALPHA = 0.0 yields force-independent (uniform) degradation.
#
PHASE2_FORCE_ALPHA = 0.01


def _parse_delimited_tables_from_csv(path: str) -> list[dict[str, list[Any]]]:
    """
    Deterministic, read-only CSV multi-table parser.

    Expected format (as used by existing test data):
    - Table 0: nodes (header row, then data rows)
    - blank line
    - Table 1: edges (header row, then data rows)
    - blank line
    - Table 2 (optional): meta_data (header row, then key/value rows)
    """
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all(str(cell).strip() == "" for cell in row):
                if current:
                    tables.append(current)
                    current = []
                continue
            current.append([str(cell).strip() for cell in row])
    if current:
        tables.append(current)

    out: list[dict[str, list[Any]]] = []
    for table_idx, t in enumerate(tables):
        if not t:
            continue
        headers = t[0]
        data_rows = t[1:]
        table_dict: dict[str, list[Any]] = {h: [] for h in headers}
        for row_idx, r in enumerate(data_rows):
            # Pad missing cells deterministically
            padded = list(r) + [""] * max(0, len(headers) - len(r))
            for i, h in enumerate(headers):
                table_dict[h].append(padded[i])

        # FIX C1: CSV empty cell pre-validation (user experience improvement)
        # Scan for empty cells BEFORE coercion to provide helpful error messages.
        for col_name, values in table_dict.items():
            for row_idx, val in enumerate(values):
                if str(val).strip() == "":
                    # Table names for helpful error messages
                    table_names = ["nodes", "edges", "meta_data"]
                    table_name = table_names[table_idx] if table_idx < len(table_names) else f"table_{table_idx}"
                    raise ValueError(
                        f"Empty cell detected in CSV file at {path}:\n"
                        f"  Table: {table_name}\n"
                        f"  Row: {row_idx + 2} (data row {row_idx + 1}, accounting for header)\n"
                        f"  Column: '{col_name}'\n"
                        f"Please fill in all required cells or remove incomplete rows."
                    )
        out.append(table_dict)
    return out


def _parse_delimited_tables_from_xlsx(path: str) -> list[dict[str, list[Any]]]:
    """
    Deterministic, read-only XLSX multi-table parser.

    Supports:
    - Fast path: sheets named "nodes"/"edges" (optional "meta_data")
    - Fallback: multiple stacked tables in a single sheet, detected via header-row scanning
      and sliced deterministically without relying on blank rows.

    Critical safety goals:
    - Do not mix meta/header rows into the edges table.
    - Preserve raw cell values (dtype=object) so callers can validate strictly (no NaN->0 coercions).
    """

    def _norm_cell(v: Any) -> str | None:
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        s = str(v).strip()
        if s == "":
            return None
        return _normalize_column_name(s)

    def _find_header_row(
        df_raw: "pd.DataFrame",
        *,
        required_groups: Sequence[Sequence[str]],
        start_row: int = 0,
    ) -> int | None:
        """
        Find the first row index >= start_row that contains all required header groups.
        Each required group is a list of alternative normalized header names; at least one must be present.
        """
        n_rows = int(df_raw.shape[0])
        for ridx in range(int(start_row), n_rows):
            row = df_raw.iloc[ridx, :].tolist()
            cells = [_norm_cell(c) for c in row]
            present = set(c for c in cells if c is not None)
            ok = True
            for group in required_groups:
                if not any(_normalize_column_name(g) in present for g in group):
                    ok = False
                    break
            if ok:
                return int(ridx)
        return None

    def _slice_table(
        df_raw: "pd.DataFrame",
        *,
        header_row_idx: int,
        end_row_exclusive: int,
    ) -> dict[str, list[Any]]:
        """
        Slice a table from df_raw using a header row and a stop row.
        Returns a dict of {header_cell_value: column_values}, plus "__row_index__" for diagnostics.
        """
        header_row = df_raw.iloc[int(header_row_idx), :].tolist()
        header_cells: list[tuple[int, Any]] = []
        for cidx, hv in enumerate(header_row):
            n = _norm_cell(hv)
            if n is None:
                continue
            header_cells.append((int(cidx), hv))
        if not header_cells:
            return {}
        header_cells.sort(key=lambda t: t[0])
        col_indices = [cidx for cidx, _ in header_cells]
        headers = [hv for _, hv in header_cells]

        out: dict[str, list[Any]] = {h: [] for h in headers}
        out["__row_index__"] = []
        for ridx in range(int(header_row_idx) + 1, int(end_row_exclusive)):
            row = df_raw.iloc[int(ridx), :]
            # Treat fully-empty rows as separators and skip deterministically.
            nonempty = False
            for cidx in col_indices:
                v = row.iloc[int(cidx)]
                if _norm_cell(v) is not None:
                    nonempty = True
                    break
            if not nonempty:
                continue
            out["__row_index__"].append(int(ridx))
            for (cidx, h) in zip(col_indices, headers):
                out[h].append(row.iloc[int(cidx)])
        return out

    # Fast path: dedicated sheets if present.
    try:
        xl = pd.ExcelFile(path)
        sheet_names_norm = {_normalize_column_name(s): s for s in xl.sheet_names}
    except Exception:
        xl = None
        sheet_names_norm = {}

    if xl is not None and ("nodes" in sheet_names_norm) and ("edges" in sheet_names_norm):
        nodes_df = pd.read_excel(path, sheet_name=sheet_names_norm["nodes"], dtype=object)
        edges_df = pd.read_excel(path, sheet_name=sheet_names_norm["edges"], dtype=object)
        meta_df = None
        if "meta_data" in sheet_names_norm:
            meta_df = pd.read_excel(path, sheet_name=sheet_names_norm["meta_data"], dtype=object)
        tables: list[dict[str, list[Any]]] = []
        tables.append({c: nodes_df[c].tolist() for c in nodes_df.columns})
        tables.append({c: edges_df[c].tolist() for c in edges_df.columns})
        if meta_df is not None:
            tables.append({c: meta_df[c].tolist() for c in meta_df.columns})
        return tables

    # Fallback: stacked tables in the first sheet, with header=None.
    df_raw = pd.read_excel(path, sheet_name=0, header=None, dtype=object)

    # Header detection using candidate groups (normalized).
    nodes_required = [
        ["n_id", "node_id", "id"],
        ["n_x", "x"],
        ["n_y", "y"],
        ["is_left_boundary"],
        ["is_right_boundary"],
    ]
    edges_required = [
        ["e_id", "edge_id", "id"],
        ["n_from", "from", "source"],
        ["n_to", "to", "target"],
        ["thickness"],
    ]
    meta_required = [
        ["meta_key", "key"],
        ["meta_value", "value"],
    ]

    nodes_header = _find_header_row(df_raw, required_groups=nodes_required, start_row=0)
    if nodes_header is None:
        raise ValueError("Failed to detect nodes table header row in XLSX.")
    edges_header = _find_header_row(df_raw, required_groups=edges_required, start_row=int(nodes_header) + 1)
    if edges_header is None:
        raise ValueError("Failed to detect edges table header row in XLSX.")
    meta_header = _find_header_row(df_raw, required_groups=meta_required, start_row=int(edges_header) + 1)

    # Determine stop rows deterministically (end at next header row, else EOF).
    n_rows = int(df_raw.shape[0])
    nodes_end = int(edges_header)
    edges_end = int(meta_header) if meta_header is not None else n_rows
    meta_end = n_rows

    nodes_table = _slice_table(df_raw, header_row_idx=int(nodes_header), end_row_exclusive=int(nodes_end))
    edges_table = _slice_table(df_raw, header_row_idx=int(edges_header), end_row_exclusive=int(edges_end))
    meta_table = _slice_table(df_raw, header_row_idx=int(meta_header), end_row_exclusive=int(meta_end)) if meta_header is not None else {}

    # Deterministic output ordering: nodes, edges, meta (optional).
    out_tables: list[dict[str, list[Any]]] = [nodes_table, edges_table]
    if meta_table:
        out_tables.append(meta_table)
    return out_tables


def _normalize_column_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _require_column(table: Mapping[str, list[Any]], candidates: Sequence[str], *, table_name: str) -> str:
    norm = {_normalize_column_name(k): k for k in table.keys()}
    for c in candidates:
        key = norm.get(_normalize_column_name(c))
        if key is not None:
            return key
    raise ValueError(f"Missing required column in {table_name}: one of {list(candidates)}")


def _coerce_int(v: Any, *, sheet: str = None, row: int = None, column: str = None) -> int:
    """Coerce value to int with optional context for error messages."""
    if isinstance(v, int):
        return v
    # Do not silently coerce NaN/None; fail loudly.
    try:
        if v is None or pd.isna(v):
            ctx = ""
            if sheet and row is not None and column:
                ctx = f" [Sheet: {sheet}, Row: {row}, Column: {column}]"
            elif sheet and column:
                ctx = f" [Sheet: {sheet}, Column: {column}]"
            raise ValueError(f"Expected int but got NaN/None{ctx}")
    except ValueError:
        raise
    except Exception:
        pass
    s = str(v).strip()
    if s == "":
        ctx = ""
        if sheet and row is not None and column:
            ctx = f" [Sheet: {sheet}, Row: {row}, Column: {column}]"
        elif sheet and column:
            ctx = f" [Sheet: {sheet}, Column: {column}]"
        raise ValueError(f"Expected int but got empty value{ctx}")
    try:
        return int(float(s))
    except (ValueError, TypeError) as e:
        ctx = ""
        if sheet and row is not None and column:
            ctx = f" [Sheet: {sheet}, Row: {row}, Column: {column}]"
        elif sheet and column:
            ctx = f" [Sheet: {sheet}, Column: {column}]"
        raise ValueError(f"Cannot convert '{s}' to int{ctx}") from e


def _coerce_float(v: Any, *, sheet: str = None, row: int = None, column: str = None) -> float:
    """Coerce value to float with optional context for error messages."""
    if isinstance(v, float):
        return v
    if isinstance(v, int):
        return float(v)
    s = str(v).strip()
    if s == "":
        ctx = ""
        if sheet and row is not None and column:
            ctx = f" [Sheet: {sheet}, Row: {row}, Column: {column}]"
        elif sheet and column:
            ctx = f" [Sheet: {sheet}, Column: {column}]"
        raise ValueError(f"Expected float but got empty value{ctx}")
    try:
        return float(s)
    except (ValueError, TypeError) as e:
        ctx = ""
        if sheet and row is not None and column:
            ctx = f" [Sheet: {sheet}, Row: {row}, Column: {column}]"
        elif sheet and column:
            ctx = f" [Sheet: {sheet}, Column: {column}]"
        raise ValueError(f"Cannot convert '{s}' to float{ctx}") from e


def _coerce_bool_boundary_flag(v: Any, *, node_id: Any, column_name: str) -> bool:
    """
    Coerce explicit boundary flags from input tables.

    Accepted forms (per spec):
    - True/False
    - 1/0 (int/float) or "1"/"0"
    """
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, (int, float)):
        try:
            fv = float(v)
        except Exception:
            fv = None
        if fv is not None and np.isfinite(fv):
            if fv == 1.0:
                return True
            if fv == 0.0:
                return False
    s = str(v).strip().lower()
    if s in ("true", "t", "1"):
        return True
    if s in ("false", "f", "0"):
        return False
    raise ValueError(
        f"Invalid boundary flag value for node {node_id} in column '{column_name}': {v} "
        "(expected True/False or 1/0)"
    )


def _coerce_bool_input_flag(v: Any, *, node_id: Any, column_name: str) -> bool:
    """
    Coerce boolean-like input flags from input tables (strict, deterministic).

    Accepted forms:
    - True/False
    - 1/0 (int/float) or "1"/"0"
    """
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, (int, float)):
        try:
            fv = float(v)
        except Exception:
            fv = None
        if fv is not None and np.isfinite(fv):
            if fv == 1.0:
                return True
            if fv == 0.0:
                return False
    s = str(v).strip().lower()
    if s in ("true", "t", "1"):
        return True
    if s in ("false", "f", "0"):
        return False
    raise ValueError(
        f"Invalid boolean flag value for node {node_id} in column '{column_name}': {v} "
        "(expected True/False or 1/0)"
    )


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return float(math.sqrt(dx * dx + dy * dy))


def _jsonify(obj: Any) -> Any:
    """Convert tuples recursively into lists so objects are JSON-serializable."""
    if isinstance(obj, tuple):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, list):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    return obj


def _tuplify(obj: Any) -> Any:
    """Convert lists recursively into tuples (inverse of _jsonify for RNG state restore)."""
    if isinstance(obj, list):
        return tuple(_tuplify(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _tuplify(v) for k, v in obj.items()}
    return obj


def _median(values: Sequence[float]) -> float:
    """Deterministic median for a non-empty sequence of floats."""
    xs = sorted(float(x) for x in values)
    n = len(xs)
    if n == 0:
        raise ValueError("median of empty sequence")
    mid = n // 2
    if (n % 2) == 1:
        return float(xs[mid])
    return 0.5 * (float(xs[mid - 1]) + float(xs[mid]))


def _is_spatial_mode_active() -> bool:
    """
    Helper: Check if spatial plasmin mode (v5.0) is active.
    
    Returns:
    --------
    bool: True if USE_SPATIAL_PLASMIN feature flag is enabled.
    
    Phase 2F: Single source of truth for spatial vs. legacy mode check.
    """
    from src.config.feature_flags import FeatureFlags
    return bool(FeatureFlags.USE_SPATIAL_PLASMIN)


def _copy_edge_with_updates(edge: "Phase1EdgeSnapshot", **updates) -> "Phase1EdgeSnapshot":
    """
    Helper: Create a new Phase1EdgeSnapshot with updated fields, preserving segments.
    
    Parameters:
    -----------
    edge : Phase1EdgeSnapshot
        The original edge snapshot.
    **updates : dict
        Fields to update (passed to dataclasses.replace).
    
    Returns:
    --------
    Phase1EdgeSnapshot
        New snapshot with updated fields and segments preserved (if present).
    
    Phase 2F: Single source of truth for immutable edge updates that preserve segments.
    
    Examples:
    ---------
    >>> new_edge = _copy_edge_with_updates(old_edge, S=0.8, M=0.1)
    >>> # segments automatically preserved if present in old_edge
    """
    # Ensure segments are explicitly preserved unless user is replacing them
    if "segments" not in updates and hasattr(edge, "segments"):
        updates["segments"] = edge.segments
    return replace(edge, **updates)


@dataclass
class SimulationState:
    """
    GUI-only state container for the Research Simulation page.

    Notes:
    - No numerical evolution logic is implemented here.
    - Values change only via explicit UI actions.
    """
    loaded_network: object | None
    strain_value: float
    time: float  # seconds
    is_running: bool
    is_paused: bool


@dataclass
class LoadedNetworkPlaceholder:
    """Placeholder object representing a loaded network selection (no parsing)."""
    path: str


@dataclass(frozen=True)
class FiberSegment:
    """
    Immutable per-segment state for v5.0 spatial mechanochemical fibrinolysis.
    
    A fiber is subdivided into uniform-length segments to track localized
    binding and damage. This is the fundamental unit of plasmin kinetics.
    
    Fields:
    -------
    segment_index : int
        Position along fiber (0-indexed)
    n_i : float
        Intact protofibrils in this segment ∈ [0, N_pf]
        (Deterministic expected value; no stochastic sampling in v5.0)
    B_i : float
        Bound plasmin count on this segment ∈ [0, S_i]
    S_i : float
        Maximum binding site capacity (from surface area)
    """
    segment_index: int
    n_i: float  # Intact protofibrils [0, N_pf]
    B_i: float  # Bound plasmin [0, S_i]
    S_i: float  # Max binding sites


@dataclass(frozen=True)
class Phase1EdgeSnapshot:
    """
    Immutable edge snapshot for Phase 1 (v1 + optional v2/v5 spatial plasmin).

    Core Fields (Legacy):
    - edge_id: stable identifier
    - n_from / n_to: node endpoint identifiers
    - k0: baseline stiffness per protofibril; fiber-level stiffness emerges via multiplication by N_pf
    - original_rest_length: imported rest length (constant; never modified)
    - L_rest_effective: effective rest length (persistent; may increase irreversibly in Phase 2.3)
    - M: short-term damage memory (persistent; Phase 2.8)
    - S: current strength (scalar integrity in legacy; derived proxy in spatial mode)
    - thickness: imported per-edge thickness (immutable experimental data)
    - lysis_batch_index: first batch index where S <= 0 (set once)
    - lysis_time: simulation time at which lysis_batch_index occurred (set once)
    
    Spatial Plasmin Extension (v2, deprecated; kept for compatibility):
    - plasmin_sites: tuple of PlasminBindingSite (immutable collection)
      * Empty tuple (default) → legacy mode (no spatial info)
      * DEPRECATED in v5.0 in favor of segments
    
    Spatial Plasmin v5.0 Extension:
    - segments: tuple of FiberSegment | None
      * None → legacy mode (scalar S integrity)
      * Non-empty tuple → v5.0 spatial mechanochemical model active
      * Each segment tracks binding occupancy (B_i) and intact protofibrils (n_i)
    
    Backward Compatibility:
    - plasmin_sites defaults to empty tuple (v2 legacy)
    - segments defaults to None (legacy mode)
    - S field: derived proxy when segments exist (S = min(n_i/N_pf))
    
    Physics Model Selection (controlled by FeatureFlags.USE_SPATIAL_PLASMIN):
    - False (default): S is scalar integrity (legacy)
    - True: S is derived from segments (min(n_i/N_pf))
    
    Immutability:
    - All fields frozen (no mutation after creation)
    - To update: create new instance via dataclasses.replace()
    """
    edge_id: Any
    n_from: Any
    n_to: Any
    k0: float
    original_rest_length: float
    L_rest_effective: float
    M: float
    S: float
    thickness: float
    lysis_batch_index: int | None
    lysis_time: float | None
    plasmin_sites: tuple = tuple()  # v2 spatial (deprecated in v5.0)
    segments: tuple[FiberSegment, ...] | None = None  # v5.0 spatial mechanochemical
    
    @property
    def S_effective(self) -> float:
        """
        Effective integrity S for this edge.
        
        Physics Model Selection (via FeatureFlags.USE_SPATIAL_PLASMIN):
        =================================================================
        
        Legacy Mode (USE_SPATIAL_PLASMIN=False):
        - Returns stored S field directly
        - Used in current advance_one_batch() logic unchanged
        
        Spatial Mode v5.0 (USE_SPATIAL_PLASMIN=True):
        - Computes S from segment-level protofibril damage
        - S_effective = min(n_i / N_pf) across all segments (weakest-link)
        - If no segments (None), returns stored S (legacy compatibility)
        - Maintains backward compatibility with force computation
        
        Returns:
        ========
        float: Integrity value ∈ [0.0, 1.0]
        
        Examples:
        =========
        >>> # Legacy mode (default)
        >>> edge = Phase1EdgeSnapshot(..., S=0.8, segments=None)
        >>> edge.S_effective  # → 0.8
        
        >>> # Spatial v5.0 mode with segments
        >>> seg1 = FiberSegment(0, n_i=45, B_i=0, S_i=100)  # 45/50 = 0.9 intact
        >>> seg2 = FiberSegment(1, n_i=30, B_i=5, S_i=100)  # 30/50 = 0.6 intact
        >>> edge = Phase1EdgeSnapshot(..., S=0.9, segments=(seg1, seg2))
        >>> edge.S_effective  # → 0.6 (min across segments)
        """
        if not FeatureFlags.USE_SPATIAL_PLASMIN:
            # Legacy mode: return stored S
            return float(self.S)

        # Spatial v2 mode: compute from plasmin_sites damage
        if self.plasmin_sites and len(self.plasmin_sites) > 0:
            max_damage = max(site.damage_depth for site in self.plasmin_sites)
            return 1.0 - max_damage

        # Spatial v5.0 mode: compute from segments
        if self.segments is None or len(self.segments) == 0:
            # No segments initialized yet (legacy fallback)
            return float(self.S)

        # Weakest-link: minimum intact fraction across all segments
        # Note: N_pf is a parameter; we need it to compute f_i = n_i / N_pf
        # For now, we assume S was pre-computed at initialization as min(n_i/N_pf)
        # This property returns stored S when segments exist (derived proxy)
        return float(self.S)
    
    @property
    def is_cleaved(self) -> bool:
        """
        Check if this edge is cleaved (fiber has failed).
        
        Legacy Mode (USE_SPATIAL_PLASMIN=False):
        - Returns True if S <= 0.0
        
        Spatial Mode (USE_SPATIAL_PLASMIN=True):
        - Returns True if ANY plasmin site has critical damage
        - Critical damage = damage_depth >= FeatureFlags.SPATIAL_PLASMIN_CRITICAL_DAMAGE
        
        Returns:
        ========
        bool: True if edge is cleaved, False if intact.
        
        Examples:
        =========
        >>> # Legacy: cleaved if S <= 0
        >>> edge = Phase1EdgeSnapshot(..., S=0.0, ...)
        >>> edge.is_cleaved  # → True
        
        >>> # Spatial: cleaved if any site damage >= critical
        >>> site = PlasminBindingSite(..., damage_depth=0.75, ...)
        >>> edge = Phase1EdgeSnapshot(..., plasmin_sites=(site,), ...)
        >>> edge.is_cleaved  # → True (0.75 >= 0.7)
        """
        if not FeatureFlags.USE_SPATIAL_PLASMIN:
            # Legacy mode: check scalar S
            return float(self.S) <= 0.0
        
        # Spatial mode: check if any site is severed
        if not self.plasmin_sites:
            # No plasmin, edge is intact
            return False
        
        critical = FeatureFlags.SPATIAL_PLASMIN_CRITICAL_DAMAGE
        return any(site.damage_depth >= critical for site in self.plasmin_sites)

    @property
    def is_ruptured(self) -> bool:
        """
        Check if this edge is ruptured (fiber has failed).

        Alias for is_cleaved for backward compatibility with test contracts.

        Legacy Mode (USE_SPATIAL_PLASMIN=False):
        - Returns True if S <= 0.0

        Spatial Mode (USE_SPATIAL_PLASMIN=True):
        - Returns True if ANY plasmin site has critical damage
        - Critical damage = damage_depth >= FeatureFlags.SPATIAL_PLASMIN_CRITICAL_DAMAGE

        Returns:
        ========
        bool: True if edge is ruptured, False if intact.
        """
        return self.is_cleaved


class Phase1NetworkAdapter:
    """
    Minimal adapter boundary for Phase 1 Research Simulation.

    Debug logging can be enabled via environment variable:
        export FIBRINET_DEBUG=1

    When enabled, logs additional information per batch:
    - sigma_ref (median tension)
    - max force across all edges
    - mean degradation rate

    Purpose:
    - Decouple Research Simulation GUI/controller/step logic from internal network and solver
      implementations.
    - Prevent solver leakage: no internal solver objects or network internals cross this boundary.

    Holds:
    - edges: immutable snapshots (Phase1EdgeSnapshot) including current S
    - forces: per-edge forces from the last relaxation (observable cache; NOT persisted on edges)

    Exposes:
    - relax(strain) -> per-edge forces mapping (edge_id -> force)

    Notes:
    - This adapter introduces NO new behavior or physics. It delegates relaxation to an injected
      implementation hook that wraps the existing linear spring-force solver.
    - In Phase 1, cleaved status is derived strictly as S <= 0 when needed; it is never stored.
    """

    def __init__(
        self,
        *,
        path: str,
        node_coords: Mapping[Any, tuple[float, float]] | None = None,
        left_boundary_node_ids: Sequence[Any] | None = None,
        right_boundary_node_ids: Sequence[Any] | None = None,
        relax_impl: Callable[[Sequence[Phase1EdgeSnapshot], Sequence[float], float], Sequence[float]] | None = None,
    ):
        self.path = path
        self._edges: tuple[Phase1EdgeSnapshot, ...] = tuple()
        self._forces_by_edge_id: dict[Any, float] = {}
        # Initial (imported) coordinates are immutable by convention; we keep a private baseline copy.
        self.node_coords: dict[Any, tuple[float, float]] = dict(node_coords or {})
        self._initial_node_coords: dict[Any, tuple[float, float]] = dict(self.node_coords)
        # Cached relaxed coordinates (observable-only; not persisted on nodes/edges elsewhere).
        self._relaxed_node_coords: dict[Any, tuple[float, float]] | None = None
        # Boundary membership is determined ONCE at load time using ORIGINAL coordinates only.
        # Membership NEVER changes after load.
        # Store as immutable sets for safety; deterministic ordering is applied only for display/use.
        self.left_boundary_node_ids = frozenset(left_boundary_node_ids or ())
        self.right_boundary_node_ids = frozenset(right_boundary_node_ids or ())
        # Rigid clamp boundary condition (x+y fixed for boundary nodes):
        # Capture each boundary node's imported y-coordinate ONCE at load time.
        # This is experimental input and must not drift.
        self.initial_boundary_y: dict[int, float] = {}

        # Phase 1 step parameters + hooks (must be provided by future loader; not inferred here).
        self.lambda_0 = None
        self.delta = None
        self.dt = None
        self.g_force = None
        self.modifier = None

        # Phase 2.2: previous-batch mechanical state (strain-rate proxy input).
        # Initialized to None at network load (required).
        self.prev_mean_tension: float | None = None

        # Phase 2.3: plastic rest-length remodeling constants (wired at Start; no UI).
        self.plastic_F_threshold = None
        self.plastic_rate = None

        # Phase 3.1: deterministic, append-only experiment log (cleared on new load only).
        self.experiment_log: list[dict[str, Any]] = []
        # Phase 3.5: deterministic parameter freeze + provenance stamp.
        # Cleared on new network load only.
        self.frozen_params: dict[str, Any] | None = None
        self.provenance_hash: str | None = None
        # Applied strain is a fixed experimental parameter (read at Start, frozen, then immutable).
        self.applied_strain: float | None = None
        # Rigid grip kinematic constraints (uniaxial x only):
        # - Computed ONCE at Start from original boundary-node x positions + applied_strain.
        # - After Start, these are frozen experimental inputs and must not drift.
        self.left_grip_x: float | None = None
        self.right_grip_x: float | None = None
        # Stage 2 thickness-aware mechanics: thickness_ref and alpha are frozen at Start.
        self.thickness_ref: float | None = None
        self.thickness_alpha: float | None = None
        # Stage 3 degradation-rate modifiers (frozen at Start; no UI).
        self.degradation_beta: float | None = None
        self.degradation_gamma: float | None = None

        # Stage 4 observational lysis tracking (frozen threshold at Start).
        self.global_lysis_threshold: float | None = None
        self.global_lysis_batch_index: int | None = None
        self.global_lysis_time: float | None = None

        # Stage 5: limited plasmin exposure (competition model), frozen at Start.
        # Defaults preserve legacy behavior ("saturating").
        self.plasmin_mode: str | None = None  # "saturating" | "limited"
        self.N_plasmin: int | None = None

        # Terminal-state handling (model-side, deterministic):
        # If the network loses load-bearing capacity (sigma_ref invalid), the experiment terminates cleanly.
        self.termination_reason: str | None = None
        self.termination_batch_index: int | None = None
        self.termination_time: float | None = None
        self.termination_cleavage_fraction: float | None = None

        # Phase 3.6: deterministic RNG freeze + replay consistency.
        # - frozen_rng_state is captured once at Start.
        # - Cleared on new network load only.
        self.rng = None
        self.frozen_rng_state = None
        self.frozen_rng_state_hash: str | None = None

        # Phase 4.2: branching provenance (set only on fork; stamped into future log entries).
        self.branch_parent_batch_index: int | None = None
        self.branch_parent_batch_hash: str | None = None
        # Phase 4.3: sweep stamping (set only for sweep branches; stamped into future log entries).
        self.sweep_param: str | None = None
        self.sweep_value: float | None = None
        # Phase 4.4: grid sweep stamping (set only for grid branches; stamped into future log entries).
        self.grid_params: dict[str, float] | None = None

        # Injected wrapper around the existing solver. If not provided, relax() fails loudly.
        self._relax_impl = relax_impl
        
        # v5.0 spatial plasmin parameters (set at load; frozen at Start)
        self.spatial_plasmin_params: dict[str, Any] | None = None
        
        # Phase 2G: Global plasmin pool (supply-limited stochastic seeding)
        # P_total_quanta: total plasmin quanta in the system (fixed at Start)
        # P_free_quanta: currently unbound plasmin quanta (changes each batch)
        # Conservation: P_free_quanta + sum(B_i across all segments) == P_total_quanta
        self.P_total_quanta: int | None = None
        self.P_free_quanta: int | None = None

        # Phase 2D: Fractured edge history (edge removal tracking)
        # Each entry: {"edge_id", "batch_index", "segments", "final_stiffness", "tension_at_failure", "strain_at_failure"}
        self.fractured_history: list[dict[str, Any]] = []

    @property
    def edges(self) -> tuple[Phase1EdgeSnapshot, ...]:
        return self._edges

    @property
    def forces(self) -> dict[Any, float]:
        # Observable cache only; callers must treat as read-only.
        return dict(self._forces_by_edge_id)

    @property
    def render_node_coords(self) -> dict[Any, tuple[float, float]]:
        """
        Coordinates used for visualization:
        - relaxed coords if available (after relax())
        - otherwise imported coords
        """
        return dict(self._relaxed_node_coords or self.node_coords)

    @property
    def left_attachment_node_ids(self) -> tuple[Any, ...]:
        """Back-compat alias used by existing visualization code paths."""
        return tuple(sorted(self.left_boundary_node_ids))

    @property
    def right_attachment_node_ids(self) -> tuple[Any, ...]:
        """Back-compat alias used by existing visualization code paths."""
        return tuple(sorted(self.right_boundary_node_ids))

    def set_edges(self, edges: Sequence[Phase1EdgeSnapshot]):
        # Replace snapshots atomically; snapshots are immutable.
        self._edges = tuple(edges)

    def phase2_g_force(self, F: float) -> float:
        """
        Phase 2.0 force-response function (deterministic, monotonic in tensile force).
        g_force(F) = 1 + α * F_tension, with α fixed by design (no UI yet).
        """
        try:
            f = float(F)
        except Exception:
            f = 0.0
        f_tension = f if f > 0.0 else 0.0
        return 1.0 + PHASE2_FORCE_ALPHA * f_tension

    def phase2_1_g_force(self, F: float) -> float:
        """
        Phase 2.1 nonlinear, bounded, monotone force-response (Hill / Michaelis–Menten style):

            g(F) = 1 + alpha * (F^n / (F^n + F0^n))

        Where:
        - alpha > 0: max amplification (bounded above by 1 + alpha)
        - F0 > 0: half-activation force scale
        - n >= 1: Hill coefficient (steepness)

        Safety:
        - Clamp F to tension only: F = max(F, 0)
        - Assert g(F) is finite (no silent fallback)
        """
        # Defensive clamp (required)
        F = float(F)
        F = max(F, 0.0)

        alpha = float(getattr(self, "force_alpha"))
        F0 = float(getattr(self, "force_F0"))
        n = float(getattr(self, "force_hill_n"))

        # Compute saturation term deterministically (bounded in [0, 1])
        Fn = F ** n
        F0n = F0 ** n
        gF = 1.0 + alpha * (Fn / (Fn + F0n))

        # Defensive finite check (required)
        assert np.isfinite(gF)
        return float(gF)

    def phase2_2_strain_rate_factor(self, *, mean_tension: float, dt: float) -> float:
        """
        Phase 2.2 strain-rate proxy multiplier (deterministic):

          if prev_mean_tension is None: factor = 1.0
          else:
            dF = mean_tension - prev_mean_tension
            strain_rate = dF / dt
            factor = 1 + beta * tanh(strain_rate / eps0)

        Safety:
        - If no intact edges exist upstream, caller should pass mean_tension=0 and use factor=1.0.
        - Asserts factor is finite.
        - Setting beta = 0.0 recovers Phase 2.1 behavior.
        """
        prev = self.prev_mean_tension
        if prev is None:
            return 1.0

        beta = float(getattr(self, "rate_beta"))
        eps0 = float(getattr(self, "rate_eps0"))
        if dt <= 0.0:
            raise ValueError("dt must be > 0 for Phase 2.2 strain-rate factor.")
        if eps0 <= 0.0:
            raise ValueError("rate_eps0 must be > 0 for Phase 2.2 strain-rate factor.")

        dF = float(mean_tension) - float(prev)
        strain_rate = dF / float(dt)
        factor = 1.0 + beta * float(np.tanh(strain_rate / float(eps0)))
        assert np.isfinite(factor)
        return float(factor)

    def configure_existing_solver_relaxation(self):
        """
        Phase 1C wiring: bind this adapter's relaxation hook to the existing deterministic
        2D linear spring-force relaxation solver (no solver logic modifications).
        """
        self._relax_impl = self._build_existing_solver_relax_impl()

    # ---------------------------
    # Phase 3.2 deterministic export (CSV + JSON)
    # ---------------------------
    def export_experiment_log_json(self, path: str):
        """
        Export exactly `experiment_log` as JSON (structured), deterministic:
        - preserves batch order
        - indent=2
        - no timestamps/metadata added
        - no modification of in-memory log
        - no partial writes (atomic replace)
        """
        # FIX D2: Export guard for empty logs (user experience)
        if not self.experiment_log:
            raise ValueError(
                "No completed or aborted batches to export.\n"
                "The experiment_log is empty. Run at least one batch before exporting."
            )
        out_path = str(path)
        tmp_dir = os.path.dirname(os.path.abspath(out_path)) or "."
        os.makedirs(tmp_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix="experiment_log_", suffix=".tmp", dir=tmp_dir, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                json.dump(self.experiment_log, f, indent=2)
                f.write("\n")
            os.replace(tmp_path, out_path)
        except Exception:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise

    def export_experiment_log_csv(self, path: str):
        """
        Export exactly `experiment_log` as flat CSV, deterministic:
        - one row per batch
        - preserves batch order
        - deterministic column ordering
        - flattens params with `param_` prefix
        - numeric values only
        - no modification of in-memory log
        - no partial writes (atomic replace)
        """
        # FIX D2: Export guard for empty logs (user experience)
        if not self.experiment_log:
            raise ValueError(
                "No completed or aborted batches to export.\n"
                "The experiment_log is empty. Run at least one batch before exporting."
            )
        out_path = str(path)
        tmp_dir = os.path.dirname(os.path.abspath(out_path)) or "."
        os.makedirs(tmp_dir, exist_ok=True)

        base_cols = [
            "batch_index",
            "time",
            "strain",
            "intact_edges",
            "cleaved_edges_total",
            "newly_cleaved",
            "mean_tension",
            "lysis_fraction",
            # Phase 3E: Per-edge aggregates (spatial mode only; nullable)
            "edge_n_min_global",
            "edge_n_mean_global",
            "edge_B_total_sum",
        ]
        param_cols = [
            "param_lambda_0",
            "param_dt",
            "param_delta",
            "param_force_alpha",
            "param_force_F0",
            "param_force_hill_n",
            "param_rate_beta",
            "param_plastic_rate",
            "param_rupture_gamma",
            "param_fracture_Gc",
            "param_fracture_eta",
            "param_coop_chi",
            "param_aniso_kappa",
            "param_memory_mu",
            "param_memory_rho",
        ]
        fieldnames = base_cols + param_cols

        def _row(entry: dict[str, Any]) -> dict[str, float]:
            params = entry.get("params") or {}
            # All values numeric (cast to float; batch_index/int counts cast later by reader if desired)
            return {
                "batch_index": float(entry["batch_index"]),
                "time": float(entry["time"]),
                "strain": float(entry["strain"]),
                "intact_edges": float(entry["intact_edges"]),
                "cleaved_edges_total": float(entry["cleaved_edges_total"]),
                "newly_cleaved": float(entry["newly_cleaved"]),
                "mean_tension": float(entry["mean_tension"]),
                "lysis_fraction": float(entry["lysis_fraction"]),
                # Phase 3E: Per-edge aggregates (nullable)
                "edge_n_min_global": float(entry["edge_n_min_global"]) if entry.get("edge_n_min_global") is not None else "",
                "edge_n_mean_global": float(entry["edge_n_mean_global"]) if entry.get("edge_n_mean_global") is not None else "",
                "edge_B_total_sum": float(entry["edge_B_total_sum"]) if entry.get("edge_B_total_sum") is not None else "",
                "param_lambda_0": float(params["lambda_0"]),
                "param_dt": float(params["dt"]),
                "param_delta": float(params["delta"]),
                "param_force_alpha": float(params["force_alpha"]),
                "param_force_F0": float(params["force_F0"]),
                "param_force_hill_n": float(params["force_hill_n"]),
                "param_rate_beta": float(params["rate_beta"]),
                "param_plastic_rate": float(params["plastic_rate"]),
                "param_rupture_gamma": float(params["rupture_gamma"]),
                "param_fracture_Gc": float(params["fracture_Gc"]),
                "param_fracture_eta": float(params["fracture_eta"]),
                "param_coop_chi": float(params["coop_chi"]),
                "param_aniso_kappa": float(params["aniso_kappa"]),
                "param_memory_mu": float(params["memory_mu"]),
                "param_memory_rho": float(params["memory_rho"]),
            }

        fd, tmp_path = tempfile.mkstemp(prefix="experiment_log_", suffix=".tmp", dir=tmp_dir, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for entry in self.experiment_log:
                    writer.writerow(_row(entry))
            os.replace(tmp_path, out_path)
        except Exception:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise

    def export_edge_lysis_csv(self, path: str):
        """
        Stage 4 export: one row per edge with lysis metadata (deterministic ordering).
        No partial writes (atomic replace).
        """
        out_path = str(path)
        tmp_dir = os.path.dirname(os.path.abspath(out_path)) or "."
        os.makedirs(tmp_dir, exist_ok=True)

        fieldnames = [
            "edge_id",
            "n_from",
            "n_to",
            "thickness",
            "lysis_batch_index",
            "lysis_time",
        ]

        fd, tmp_path = tempfile.mkstemp(prefix="edge_lysis_", suffix=".tmp", dir=tmp_dir, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for e in sorted(self._edges, key=lambda ee: int(ee.edge_id)):
                    writer.writerow(
                        {
                            "edge_id": int(e.edge_id),
                            "n_from": int(e.n_from),
                            "n_to": int(e.n_to),
                            "thickness": float(e.thickness),
                            "lysis_batch_index": (int(e.lysis_batch_index) if e.lysis_batch_index is not None else ""),
                            "lysis_time": (float(e.lysis_time) if e.lysis_time is not None else ""),
                        }
                    )
            os.replace(tmp_path, out_path)
        except Exception:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise

    def export_fractured_history_csv(self, path: str):
        """
        Phase 3F: Export fractured edge history as CSV.
        One row per segment per fractured edge (deterministic ordering).
        No partial writes (atomic replace).
        """
        # FIX D2: Export guard for empty logs (user experience)
        if not self.fractured_history:
            raise ValueError(
                "No fractured edges to export.\n"
                "The fractured_history is empty. Run the simulation until edges fracture before exporting."
            )

        out_path = str(path)
        tmp_dir = os.path.dirname(os.path.abspath(out_path)) or "."
        os.makedirs(tmp_dir, exist_ok=True)

        fieldnames = [
            "edge_id",
            "batch_index",
            "segment_index",
            "n_i",
            "N_pf",
            "B_i",
            "S_i",
            "final_edge_stiffness",
            "tension_at_failure",
            "strain_at_failure",
        ]

        fd, tmp_path = tempfile.mkstemp(prefix="fractured_history_", suffix=".tmp", dir=tmp_dir, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                # Sort by edge_id, then batch_index for deterministic output
                sorted_history = sorted(self.fractured_history, key=lambda r: (int(r["edge_id"]), int(r["batch_index"])))

                for record in sorted_history:
                    edge_id = int(record["edge_id"])
                    batch_index = int(record["batch_index"])
                    final_S = float(record["final_stiffness"])
                    segments = record.get("segments")
                    tension_at_failure = record.get("tension_at_failure", "")
                    strain_at_failure = record.get("strain_at_failure", "")

                    if segments:
                        # Get N_pf from first segment (same for all segments in edge)
                        N_pf_val = None
                        if len(segments) > 0 and hasattr(segments[0], "n_i"):
                            # Infer N_pf from spatial_plasmin_params (more reliable than per-segment)
                            # But for export, include explicit value
                            if self.spatial_plasmin_params:
                                N_pf_val = int(self.spatial_plasmin_params.get("N_pf", 50))
                            else:
                                # Fallback: estimate from first segment's scale
                                N_pf_val = 50  # Default

                        # One row per segment
                        for seg in segments:
                            writer.writerow({
                                "edge_id": edge_id,
                                "batch_index": batch_index,
                                "segment_index": int(seg.segment_index),
                                "n_i": float(seg.n_i),
                                "N_pf": N_pf_val if N_pf_val is not None else "",
                                "B_i": float(seg.B_i),
                                "S_i": float(seg.S_i),
                                "final_edge_stiffness": final_S,
                                "tension_at_failure": tension_at_failure if tension_at_failure != "" else "",
                                "strain_at_failure": strain_at_failure if strain_at_failure not in (None, "") else "",
                            })
            os.replace(tmp_path, out_path)
        except Exception:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise

    def export_network_snapshot(self, path: str):
        """
        Export a deterministic, read-only snapshot of the CURRENT network state (JSON only).

        Snapshot schema:
        - nodes: [{node_id, x, y}] from current relaxed coordinates
        - edges: [{edge_id, n_from, n_to, S, M, original_rest_length, L_rest_effective, ...}]
          - In spatial mode (v5.0), each edge may also include:
            - segments: [{segment_index, n_i, B_i, S_i}]
        - Phase 2G (v5.0) spatial mode only:
          - P_total_quanta, P_free_quanta (plasmin pool state)
          - spatial_plasmin_params (frozen input params needed to resume)

        Rules:
        - deterministic ordering: nodes by node_id, edges by edge_id
        - no forces, no solver objects, no UI state
        - atomic write via temp file + os.replace
        """
        out_path = str(path)
        tmp_dir = os.path.dirname(os.path.abspath(out_path)) or "."
        os.makedirs(tmp_dir, exist_ok=True)

        if self._relaxed_node_coords is None:
            raise ValueError("No relaxed geometry available. Run relaxation before exporting a snapshot.")

        nodes_sorted = []
        for nid in sorted(self._relaxed_node_coords.keys()):
            x, y = self._relaxed_node_coords[nid]
            nodes_sorted.append({"node_id": int(nid), "x": float(x), "y": float(y)})

        edges_sorted = []
        for e in sorted(self._edges, key=lambda ee: int(ee.edge_id)):
            edge_row = {
                "edge_id": int(e.edge_id),
                "n_from": int(e.n_from),
                "n_to": int(e.n_to),
                "S": float(e.S),
                "M": float(e.M),
                "original_rest_length": float(e.original_rest_length),
                "L_rest_effective": float(e.L_rest_effective),
                "thickness": float(e.thickness),
                "lysis_batch_index": (int(e.lysis_batch_index) if e.lysis_batch_index is not None else None),
                "lysis_time": (float(e.lysis_time) if e.lysis_time is not None else None),
            }
            if e.segments is not None:
                edge_row["segments"] = [
                    {
                        "segment_index": int(s.segment_index),
                        "n_i": float(s.n_i),
                        "B_i": float(s.B_i),
                        "S_i": float(s.S_i),
                    }
                    for s in e.segments
                ]
            edges_sorted.append(edge_row)

        # Include latest batch_hash if available (current state is expected to correspond to latest batch).
        latest_batch_hash = None
        latest_batch_duration_sec = None
        if self.experiment_log:
            latest_batch_hash = self.experiment_log[-1].get("batch_hash")
            latest_batch_duration_sec = self.experiment_log[-1].get("batch_duration_sec")

        snapshot = {
            "provenance_hash": self.provenance_hash,
            "frozen_params": copy.deepcopy(self.frozen_params),
            "rng_state_hash": self.frozen_rng_state_hash,
            # Phase 4.1 checkpointing requires restoring RNG state exactly.
            # Store raw frozen RNG state in JSON-safe form (no randomness; deterministic serialization).
            "frozen_rng_state": _jsonify(self.frozen_rng_state),
            "batch_hash": latest_batch_hash,
            "batch_duration_sec": latest_batch_duration_sec,
            "global_lysis_batch_index": self.global_lysis_batch_index,
            "global_lysis_time": self.global_lysis_time,
            # Phase 2G (v5.0): plasmin pool state for deterministic checkpoint/replay (spatial mode only)
            "P_total_quanta": (int(self.P_total_quanta) if self.P_total_quanta is not None else None),
            "P_free_quanta": (int(self.P_free_quanta) if self.P_free_quanta is not None else None),
            # v5.0 spatial plasmin params (needed to resume spatial kinetics deterministically)
            "spatial_plasmin_params": (copy.deepcopy(self.spatial_plasmin_params) if isinstance(self.spatial_plasmin_params, dict) else None),
            "nodes": nodes_sorted,
            "edges": edges_sorted,
        }

        fd, tmp_path = tempfile.mkstemp(prefix="network_snapshot_", suffix=".tmp", dir=tmp_dir, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                json.dump(snapshot, f, indent=2)
                f.write("\n")
            os.replace(tmp_path, out_path)
        except Exception:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise

    def replay_single_batch(self, snapshot_path: str) -> dict:
        """
        Phase 3.4 deterministic replay check (read-only).

        Loads a previously exported Phase 3.3 network snapshot (JSON), reconstructs a temporary
        adapter clone, runs exactly one "Advance One Batch" worth of logic (no UI calls),
        and returns key observables:
          - newly_cleaved
          - mean_tension
          - lysis_fraction

        Notes:
        - Uses CURRENT adapter parameters/constants (lambda_0, dt, gates, guardrails).
        - Does NOT mutate the live adapter, even on success.
        - Requires at least one experiment log entry to supply the comparison strain.
        """
        if not self.experiment_log:
            raise ValueError("No experiment log entries available for replay comparison.")

        # "Next" log entry for comparison is the latest executed batch.
        expected = self.experiment_log[-1]
        strain = float(expected["strain"])

        with open(snapshot_path, "r", encoding="utf-8") as f:
            snap = json.load(f)

        if not isinstance(snap, dict) or "nodes" not in snap or "edges" not in snap:
            raise ValueError("Invalid snapshot format: expected JSON object with 'nodes' and 'edges'.")

        # Phase 3.5 replay enforcement: provenance must match.
        snap_hash = snap.get("provenance_hash", None)
        if self.provenance_hash is None:
            raise ValueError("Replay failed: current adapter has no provenance_hash (parameters not frozen).")
        if snap_hash != self.provenance_hash:
            raise ValueError("Replay failed: snapshot provenance_hash does not match current experiment.")

        # Stage 2 requirement: snapshot must include thickness_ref/thickness_alpha in frozen_params.
        snap_frozen_params = snap.get("frozen_params", None)
        if not isinstance(snap_frozen_params, dict):
            raise ValueError("Replay failed: snapshot frozen_params missing or invalid.")
        if "thickness_ref" not in snap_frozen_params or "thickness_alpha" not in snap_frozen_params:
            raise ValueError("Replay failed: missing thickness_ref/thickness_alpha in snapshot (Stage 2 required).")
        if "beta" not in snap_frozen_params or "gamma" not in snap_frozen_params:
            raise ValueError("Replay failed: missing beta/gamma in snapshot (Stage 3 required).")
        if "plasmin_mode" not in snap_frozen_params or "N_plasmin" not in snap_frozen_params:
            raise ValueError("Replay failed: missing plasmin_mode/N_plasmin in snapshot (Stage 5 required).")
        # Explicit boundary membership is required (no heuristics; deterministic clamps).
        if "left_boundary_node_ids" not in snap_frozen_params or "right_boundary_node_ids" not in snap_frozen_params:
            raise ValueError(
                "Replay failed: missing left_boundary_node_ids/right_boundary_node_ids in snapshot frozen_params. "
                "Boundary nodes must be explicitly specified via is_left_boundary / is_right_boundary."
            )
        if "left_grip_x" not in snap_frozen_params or "right_grip_x" not in snap_frozen_params:
            raise ValueError("Replay failed: missing left_grip_x/right_grip_x in snapshot frozen_params (rigid grips required).")
        if "initial_boundary_y" not in snap_frozen_params:
            raise ValueError("Replay failed: missing initial_boundary_y in snapshot frozen_params (rigid clamps require y-fix).")

        # Boundary consistency enforcement (deterministic).
        snap_left = snap_frozen_params.get("left_boundary_node_ids")
        snap_right = snap_frozen_params.get("right_boundary_node_ids")
        if not isinstance(snap_left, list) or not isinstance(snap_right, list):
            raise ValueError("Replay failed: invalid boundary node lists in snapshot frozen_params.")
        snap_left_ids = [int(x) for x in snap_left]
        snap_right_ids = [int(x) for x in snap_right]
        if not snap_left_ids or not snap_right_ids:
            raise ValueError("Replay failed: boundary node lists are empty.")
        if set(snap_left_ids).intersection(set(snap_right_ids)):
            raise ValueError("Replay failed: invalid boundary specification (node marked both left and right).")
        left_now = [int(x) for x in sorted(self.left_boundary_node_ids)]
        right_now = [int(x) for x in sorted(self.right_boundary_node_ids)]
        if left_now != sorted(snap_left_ids) or right_now != sorted(snap_right_ids):
            raise ValueError("Replay failed: boundary node definitions in snapshot do not match current experiment.")
        if isinstance(self.frozen_params, dict):
            if list(self.frozen_params.get("left_boundary_node_ids", [])) != left_now or list(self.frozen_params.get("right_boundary_node_ids", [])) != right_now:
                raise ValueError("Replay failed: current frozen_params boundary node lists do not match current adapter boundary sets.")
            # Grip definitions must match exactly (frozen experimental inputs).
            if float(self.frozen_params.get("left_grip_x")) != float(snap_frozen_params.get("left_grip_x")) or float(self.frozen_params.get("right_grip_x")) != float(snap_frozen_params.get("right_grip_x")):
                raise ValueError("Replay failed: rigid grip x positions in snapshot do not match current experiment.")
            # Boundary y constraints must match exactly (frozen experimental input).
            if list(self.frozen_params.get("initial_boundary_y", [])) != list(snap_frozen_params.get("initial_boundary_y", [])):
                raise ValueError("Replay failed: initial_boundary_y in snapshot does not match current experiment.")

        # Phase 3.6 replay enforcement: RNG hash must match (stochastic reproducibility spine).
        snap_rng_hash = snap.get("rng_state_hash", None)
        if self.frozen_rng_state_hash is None or self.frozen_rng_state is None:
            raise ValueError("Replay failed: current adapter has no frozen_rng_state (RNG not frozen).")
        if snap_rng_hash != self.frozen_rng_state_hash:
            raise ValueError("Replay failed: snapshot rng_state_hash does not match current experiment.")

        nodes_in = snap["nodes"]
        edges_in = snap["edges"]
        if not isinstance(nodes_in, list) or not isinstance(edges_in, list):
            raise ValueError("Invalid snapshot format: 'nodes' and 'edges' must be lists.")

        # Reconstruct relaxed node coordinates from snapshot.
        relaxed_coords: dict[int, tuple[float, float]] = {}
        for n in nodes_in:
            nid = int(n["node_id"])
            relaxed_coords[nid] = (float(n["x"]), float(n["y"]))
        # Final authority: clamp snapshot geometry x-coordinates to grips for deterministic replay.
        if (getattr(self, "left_grip_x", None) is not None) and (getattr(self, "right_grip_x", None) is not None):
            gxL = float(getattr(self, "left_grip_x"))
            gxR = float(getattr(self, "right_grip_x"))
            for nid in self.left_boundary_node_ids:
                if int(nid) in relaxed_coords:
                    _x, y = relaxed_coords[int(nid)]
                    y0 = float(self.initial_boundary_y.get(int(nid), float(y)))
                    relaxed_coords[int(nid)] = (gxL, float(y0))
            for nid in self.right_boundary_node_ids:
                if int(nid) in relaxed_coords:
                    _x, y = relaxed_coords[int(nid)]
                    y0 = float(self.initial_boundary_y.get(int(nid), float(y)))
                    relaxed_coords[int(nid)] = (gxR, float(y0))

        # Build edge_id -> k0 from current adapter (snapshot does not include k0).
        k0_by_edge_id = {int(e.edge_id): float(e.k0) for e in self._edges}

        edges_replayed: list[Phase1EdgeSnapshot] = []
        for e in edges_in:
            eid = int(e["edge_id"])
            if eid not in k0_by_edge_id:
                raise ValueError(f"Snapshot edge_id {eid} not present in current adapter.")
            if "thickness" not in e:
                raise ValueError(f"Snapshot missing required per-edge thickness for edge_id {eid}.")
            if "lysis_batch_index" not in e or "lysis_time" not in e:
                raise ValueError(f"Snapshot missing required lysis metadata for edge_id {eid}.")
            thickness = float(e["thickness"])
            if (not np.isfinite(thickness)) or (thickness <= 0.0):
                raise ValueError(f"Invalid thickness in snapshot for edge_id {eid}: {e.get('thickness')}")
            lysis_batch_index = e.get("lysis_batch_index", None)
            lysis_time = e.get("lysis_time", None)
            if lysis_batch_index is not None:
                lysis_batch_index = int(lysis_batch_index)
            if lysis_time is not None:
                lysis_time = float(lysis_time)
            # v5.0 spatial mode: restore per-segment state if present
            segments = None
            seg_in = e.get("segments", None)
            if isinstance(seg_in, list):
                rebuilt_segments: list[FiberSegment] = []
                for s in seg_in:
                    rebuilt_segments.append(
                        FiberSegment(
                            segment_index=int(s["segment_index"]),
                            n_i=float(s["n_i"]),
                            B_i=float(s["B_i"]),
                            S_i=float(s["S_i"]),
                        )
                    )
                segments = tuple(rebuilt_segments)
            edges_replayed.append(
                Phase1EdgeSnapshot(
                    edge_id=eid,
                    n_from=int(e["n_from"]),
                    n_to=int(e["n_to"]),
                    k0=float(k0_by_edge_id[eid]),
                    original_rest_length=float(e["original_rest_length"]),
                    L_rest_effective=float(e["L_rest_effective"]),
                    M=float(e["M"]),
                    S=float(e["S"]),
                    thickness=float(thickness),
                    lysis_batch_index=lysis_batch_index,
                    lysis_time=lysis_time,
                    segments=segments,
                )
            )
            # Rest-length validity (critical for mechanics): must be finite and > 0.
            if (not np.isfinite(float(edges_replayed[-1].original_rest_length))) or float(edges_replayed[-1].original_rest_length) <= 0.0:
                raise ValueError(f"Replay failed: invalid original_rest_length for edge_id {eid} (must be finite and > 0).")
            if (not np.isfinite(float(edges_replayed[-1].L_rest_effective))) or float(edges_replayed[-1].L_rest_effective) <= 0.0:
                raise ValueError(f"Replay failed: invalid L_rest_effective for edge_id {eid} (must be finite and > 0).")

        # Thickness consistency (explicit; reproducibility spine).
        snap_thickness_pairs = [(int(ed["edge_id"]), float(ed["thickness"])) for ed in edges_in]
        snap_thickness_pairs.sort(key=lambda t: t[0])
        snap_thickness_hash = hashlib.sha256(json.dumps(snap_thickness_pairs, sort_keys=True).encode("utf-8")).hexdigest()
        if isinstance(snap_frozen_params, dict) and ("thickness_hash" in snap_frozen_params):
            if str(snap_frozen_params.get("thickness_hash")) != str(snap_thickness_hash):
                raise ValueError("Replay failed: snapshot thickness_hash does not match snapshot edge thickness data.")
        if isinstance(self.frozen_params, dict) and ("thickness_hash" in self.frozen_params):
            if str(self.frozen_params.get("thickness_hash")) != str(snap_thickness_hash):
                raise ValueError("Replay failed: thickness_hash mismatch between snapshot and current experiment.")

        # Temporary adapter clone (no live state mutation).
        clone = Phase1NetworkAdapter(
            path=self.path,
            node_coords=dict(self.node_coords),
            left_boundary_node_ids=tuple(sorted(self.left_boundary_node_ids)),
            right_boundary_node_ids=tuple(sorted(self.right_boundary_node_ids)),
            relax_impl=self._relax_impl,
        )
        clone.set_edges(edges_replayed)
        clone._relaxed_node_coords = dict(relaxed_coords)
        clone.global_lysis_batch_index = snap.get("global_lysis_batch_index", None)
        clone.global_lysis_time = snap.get("global_lysis_time", None)
        # v5.0 spatial mode: restore spatial params + plasmin pool state for deterministic replay
        snap_spatial_params = snap.get("spatial_plasmin_params", None)
        if isinstance(snap_spatial_params, dict):
            clone.spatial_plasmin_params = copy.deepcopy(snap_spatial_params)
        elif isinstance(self.spatial_plasmin_params, dict):
            # Backward compatibility for snapshots without explicit spatial_plasmin_params field
            clone.spatial_plasmin_params = copy.deepcopy(self.spatial_plasmin_params)
        else:
            clone.spatial_plasmin_params = None
        snap_P_total = snap.get("P_total_quanta", None)
        snap_P_free = snap.get("P_free_quanta", None)
        if (snap_P_total is None) != (snap_P_free is None):
            raise ValueError("Replay failed: snapshot plasmin pool is incomplete (need both P_total_quanta and P_free_quanta).")
        if snap_P_total is not None:
            clone.P_total_quanta = int(snap_P_total)
            clone.P_free_quanta = int(snap_P_free)
        else:
            clone.P_total_quanta = self.P_total_quanta
            clone.P_free_quanta = self.P_free_quanta

        # Copy parameters/constants (copied, not referenced).
        clone.lambda_0 = float(self.lambda_0)
        clone.dt = float(self.dt)
        clone.delta = float(self.delta)
        clone.g_force = self.g_force
        clone.modifier = self.modifier
        # Rigid grips: copy frozen grip positions for deterministic x-clamping.
        clone.left_grip_x = float(self.left_grip_x) if (self.left_grip_x is not None) else None
        clone.right_grip_x = float(self.right_grip_x) if (self.right_grip_x is not None) else None
        # Rigid clamps: copy frozen boundary y constraints.
        clone.initial_boundary_y = dict(getattr(self, "initial_boundary_y", {}))
        clone.prev_mean_tension = float(self.prev_mean_tension) if (self.prev_mean_tension is not None) else None
        clone.plastic_F_threshold = float(self.plastic_F_threshold)
        clone.plastic_rate = float(self.plastic_rate)
        clone.rupture_force_threshold = float(self.rupture_force_threshold)
        clone.rupture_gamma = float(self.rupture_gamma)
        clone.fracture_Gc = float(self.fracture_Gc)
        clone.fracture_eta = float(self.fracture_eta)
        clone.coop_chi = float(self.coop_chi)
        clone.shield_eps = float(self.shield_eps)
        clone.memory_mu = float(self.memory_mu)
        clone.memory_rho = float(self.memory_rho)
        clone.aniso_kappa = float(self.aniso_kappa)
        clone.g_max = float(self.g_max)
        clone.cleavage_batch_cap = float(self.cleavage_batch_cap)
        clone.rate_beta = float(self.rate_beta)
        clone.rate_eps0 = float(self.rate_eps0)
        clone.thickness_ref = float(self.thickness_ref) if (self.thickness_ref is not None) else None
        clone.thickness_alpha = float(self.thickness_alpha) if (self.thickness_alpha is not None) else None
        clone.degradation_beta = float(getattr(self, "degradation_beta", 1.0)) if (getattr(self, "degradation_beta", None) is not None) else None
        clone.degradation_gamma = float(getattr(self, "degradation_gamma", 1.0)) if (getattr(self, "degradation_gamma", None) is not None) else None
        clone.plasmin_mode = str(self.frozen_params.get("plasmin_mode")) if isinstance(self.frozen_params, dict) and ("plasmin_mode" in self.frozen_params) else None
        clone.N_plasmin = int(self.frozen_params.get("N_plasmin")) if isinstance(self.frozen_params, dict) and ("N_plasmin" in self.frozen_params) else None

        # Phase 3.6: restore frozen RNG state on clone (no reseeding).
        clone.rng = random.Random()
        clone.rng.setstate(self.frozen_rng_state)

        # Compute cached pre-batch forces from snapshot geometry deterministically:
        # F = k_eff * (L - L_rest_effective)
        # where k_eff matches the solver input stiffness (including Stage 2 thickness scaling).
        forces_by_edge_id: dict[int, float] = {}
        for e in clone.edges:
            if float(e.S) <= 0.0:
                forces_by_edge_id[int(e.edge_id)] = 0.0
                continue
            p_from = clone._relaxed_node_coords.get(int(e.n_from))
            p_to = clone._relaxed_node_coords.get(int(e.n_to))
            if p_from is None or p_to is None:
                raise ValueError("Snapshot missing node coordinates referenced by an edge.")
            L = _euclidean(p_from, p_to)
            # Protofibril-based stiffness scaling (spatial mode only)
            if FeatureFlags.USE_SPATIAL_PLASMIN and clone.spatial_plasmin_params:
                N_pf = float(clone.spatial_plasmin_params.get("N_pf", 50))
                k_base = float(e.k0) * N_pf * float(e.S)
            else:
                k_base = float(e.k0) * float(e.S)
            if clone.thickness_ref is None or clone.thickness_alpha is None:
                k_eff = k_base
            else:
                t_ref = float(clone.thickness_ref)
                if (not np.isfinite(t_ref)) or (t_ref <= 0.0):
                    raise ValueError("Replay failed: invalid thickness_ref (must be finite and > 0).")
                alpha = float(clone.thickness_alpha)
                k_eff = k_base * ((float(e.thickness) / t_ref) ** alpha)
            if not np.isfinite(k_eff):
                raise ValueError("Replay failed: non-finite k_eff from thickness scaling.")
            # NUMERICAL STABILITY: Hard ceiling on effective stiffness (k_eff_max = 1e12 N/m)
            k_eff_max = 1e12  # N/m
            if k_eff > k_eff_max:
                raise ValueError(
                    f"Effective stiffness overflow during replay for edge {e.edge_id}: "
                    f"k_eff = {k_eff:.3e} exceeds k_eff_max = {k_eff_max:.3e} N/m."
                )
            F = float(k_eff) * (float(L) - float(e.L_rest_effective))
            forces_by_edge_id[int(e.edge_id)] = float(F)
        clone._forces_by_edge_id = dict(forces_by_edge_id)

        # Run exactly one batch worth of logic using the SAME ordering as advance_one_batch.
        # (Guardrails apply; no logging, no UI calls.)
        # --- Pre-batch edge consistency ---
        for e in clone.edges:
            if not (0.0 <= float(e.S) <= 1.0):
                raise ValueError("Replay failed: S out of bounds.")
            if float(e.L_rest_effective) < float(e.original_rest_length):
                raise ValueError("Replay failed: L_rest_effective < original_rest_length.")
            if float(e.M) < 0.0:
                raise ValueError("Replay failed: M < 0.")

        intact_edges = [e for e in clone.edges if float(e.S) > 0.0]
        force_list = []
        for e in intact_edges:
            if int(e.edge_id) not in clone._forces_by_edge_id:
                raise ValueError("Replay failed: missing cached forces.")
            force_list.append(float(clone._forces_by_edge_id[int(e.edge_id)]))

        # mean tension and sigma_ref (tension-only)
        if intact_edges:
            tension_forces = [max(0.0, float(f)) for f in force_list]
            mean_tension = float(sum(tension_forces) / len(tension_forces))
            sigma_ref = float(_median(tension_forces))
            if (not np.isfinite(sigma_ref)) or (sigma_ref <= 0.0):
                # Terminal-state handling (deterministic): loss of load-bearing capacity.
                # If the expected log entry indicates termination, treat replay as a terminal match.
                if isinstance(expected, dict) and expected.get("termination_reason") == "network_lost_load_bearing_capacity":
                    return {
                        "terminated": True,
                        "termination_reason": "network_lost_load_bearing_capacity",
                        "newly_cleaved": int(expected.get("newly_cleaved", 0)),
                        "mean_tension": float(expected.get("mean_tension", 0.0)),
                        "lysis_fraction": float(expected.get("lysis_fraction", 0.0)),
                    }
                # Otherwise, this is unexpected and indicates mismatch.
                raise ValueError("Replay failed: invalid sigma_ref (median tension) without termination log entry.")
        else:
            mean_tension = 0.0
            sigma_ref = None
        assert np.isfinite(mean_tension)

        # Stage 5: deterministic plasmin target selection for this replayed batch.
        if intact_edges:
            if clone.thickness_ref is None:
                raise ValueError("Replay failed: missing thickness_ref.")
            if sigma_ref is None:
                raise ValueError("Replay failed: sigma_ref missing for intact edges.")
            beta = float(getattr(clone, "degradation_beta", 1.0))
            gamma_d = float(getattr(clone, "degradation_gamma", 1.0))
            weights = []
            for ee in intact_edges:
                sigma = max(0.0, float(clone._forces_by_edge_id[int(ee.edge_id)]))
                w = (float(sigma) / float(sigma_ref)) ** float(beta)
                w *= (float(clone.thickness_ref) / float(ee.thickness)) ** float(gamma_d)
                if not np.isfinite(w) or w < 0.0:
                    raise ValueError("Replay failed: invalid attack weight.")
                weights.append((int(ee.edge_id), float(w)))
            weights.sort(key=lambda t: t[0])

            mode = str(clone.plasmin_mode or "saturating").strip().lower()
            Np = int(clone.N_plasmin) if (clone.N_plasmin is not None) else 1
            if mode not in ("saturating", "limited"):
                raise ValueError("Replay failed: invalid plasmin_mode.")
            if Np <= 0:
                raise ValueError("Replay failed: invalid N_plasmin.")
            if mode == "saturating" or Np >= len(weights):
                selected_edge_ids = [eid for eid, _ in weights]
            else:
                if self.frozen_rng_state_hash is None:
                    raise ValueError("Replay failed: missing frozen_rng_state_hash.")
                seed_material = f"{self.frozen_rng_state_hash}|plasmin_selection|{int(expected['batch_index'])}"
                seed = int(hashlib.sha256(seed_material.encode('utf-8')).hexdigest()[:16], 16)
                local_rng = random.Random(seed)
                candidates = list(weights)
                selected_edge_ids = []
                for _ in range(int(Np)):
                    total_w = float(sum(w for _, w in candidates))
                    if total_w <= 0.0 or (not np.isfinite(total_w)):
                        raise ValueError("Replay failed: invalid total attack weight.")
                    r = local_rng.random() * total_w
                    cum = 0.0
                    pick_idx = None
                    for j, (eid, w) in enumerate(candidates):
                        cum += float(w)
                        if r <= cum:
                            pick_idx = j
                            break
                    if pick_idx is None:
                        pick_idx = len(candidates) - 1
                    selected_edge_ids.append(int(candidates[pick_idx][0]))
                    del candidates[pick_idx]
                selected_edge_ids = sorted(selected_edge_ids)
            selected_edge_id_set = set(int(x) for x in selected_edge_ids)
        else:
            selected_edge_ids = []
            selected_edge_id_set = set()

        # memory update
        mu = float(clone.memory_mu)
        rho = float(clone.memory_rho)
        M_next_by_id: dict[int, float] = {}
        for e in intact_edges:
            F = float(clone._forces_by_edge_id[int(e.edge_id)])
            M_new = (1.0 - mu) * float(e.M) + mu * max(float(F), 0.0)
            assert M_new >= 0.0
            assert np.isfinite(M_new)
            M_next_by_id[int(e.edge_id)] = float(M_new)

        # strain-rate factor
        dt = float(clone.dt)
        if intact_edges:
            strain_rate_factor = float(clone.phase2_2_strain_rate_factor(mean_tension=mean_tension, dt=dt))
        else:
            strain_rate_factor = 1.0
        assert np.isfinite(strain_rate_factor)

        # neighborhood map (topology)
        s_by_edge_id = {int(e.edge_id): float(e.S) for e in clone.edges}
        node_to_edge_ids: dict[int, list[int]] = {}
        for e in clone.edges:
            node_to_edge_ids.setdefault(int(e.n_from), []).append(int(e.edge_id))
            node_to_edge_ids.setdefault(int(e.n_to), []).append(int(e.edge_id))

        coords_pre = clone._relaxed_node_coords
        if coords_pre is None:
            raise ValueError("Replay failed: missing cached node positions.")

        lambda_0 = float(clone.lambda_0)
        new_edges: list[Phase1EdgeSnapshot] = []
        newly_cleaved = 0
        intact_pre = len(intact_edges)
        newly_lysed_edge_ids: list[int] = []
        for e in clone.edges:
            S_old = float(e.S)
            L_eff = float(e.L_rest_effective)
            M_i = float(e.M)
            if S_old > 0.0:
                F = float(clone._forces_by_edge_id[int(e.edge_id)])

                # plastic update
                if F > float(clone.plastic_F_threshold):
                    dL = float(clone.plastic_rate) * (F - float(clone.plastic_F_threshold)) * dt
                    assert np.isfinite(dL)
                    L_eff = L_eff + dL
                assert L_eff >= float(e.original_rest_length)
                assert np.isfinite(L_eff)

                # rupture amplification
                F_crit = float(clone.rupture_force_threshold)
                gamma = float(clone.rupture_gamma)
                rF = 1.0 if F <= F_crit else (1.0 + gamma * (F - F_crit))
                assert rF >= 1.0
                assert np.isfinite(rF)

                # energy gate
                p_from = coords_pre.get(int(e.n_from))
                p_to = coords_pre.get(int(e.n_to))
                L = _euclidean(p_from, p_to)
                dL_geom = float(L) - float(L_eff)
                # Protofibril-based stiffness scaling (spatial mode only)
                if FeatureFlags.USE_SPATIAL_PLASMIN and clone.spatial_plasmin_params:
                    N_pf = float(clone.spatial_plasmin_params.get("N_pf", 50))
                    E_i = 0.5 * float(e.k0) * N_pf * float(S_old) * (dL_geom * dL_geom)
                else:
                    E_i = 0.5 * float(e.k0) * float(S_old) * (dL_geom * dL_geom)
                assert E_i >= 0.0
                assert np.isfinite(E_i)
                e_gate = 1.0 if E_i <= float(clone.fracture_Gc) else (1.0 + float(clone.fracture_eta) * (E_i - float(clone.fracture_Gc)))
                assert e_gate >= 1.0
                assert np.isfinite(e_gate)

                # cooperativity
                neighbor_ids = set(node_to_edge_ids.get(int(e.n_from), [])) | set(node_to_edge_ids.get(int(e.n_to), []))
                neighbor_ids.discard(int(e.edge_id))
                damage_terms = [1.0 - float(s_by_edge_id[nid]) for nid in neighbor_ids if float(s_by_edge_id[nid]) > 0.0]
                D_local = float(sum(damage_terms) / len(damage_terms)) if damage_terms else 0.0
                assert D_local >= 0.0
                assert np.isfinite(D_local)
                c_gate = 1.0 + float(clone.coop_chi) * D_local
                assert c_gate >= 1.0
                assert np.isfinite(c_gate)

                # shielding
                eps = float(clone.shield_eps)
                F_tension = max(0.0, float(F))
                f_load = float(F_tension) / float(mean_tension + eps)
                assert f_load >= 0.0
                assert np.isfinite(f_load)
                s_gate = 1.0 if f_load >= 1.0 else max(0.0, f_load)
                assert 0.0 <= s_gate <= 1.0
                assert np.isfinite(s_gate)

                # memory gate
                M_i = float(M_next_by_id.get(int(e.edge_id), float(e.M)))
                m_gate = 1.0 + rho * M_i
                assert m_gate >= 1.0
                assert np.isfinite(m_gate)

                # anisotropy
                dx = float(p_to[0]) - float(p_from[0])
                dy = float(p_to[1]) - float(p_from[1])
                L_dir = float(math.sqrt(dx * dx + dy * dy))
                a = abs(dx / L_dir) if L_dir > 0.0 else 0.0
                assert 0.0 <= a <= 1.0
                a_gate = 1.0 + float(clone.aniso_kappa) * a
                assert a_gate >= 1.0
                assert np.isfinite(a_gate)

                g_total = float(clone.g_force(F)) * float(strain_rate_factor) * float(rF) * float(e_gate) * float(c_gate) * float(s_gate) * float(m_gate) * float(a_gate)
                assert np.isfinite(g_total)
                if g_total > float(clone.g_max):
                    g_total = float(clone.g_max)

                # Stage 5: limited plasmin exposure (only selected edges degrade this batch).
                if (int(e.edge_id) not in selected_edge_id_set):
                    lambda_eff = 0.0
                elif intact_edges:
                    if sigma_ref is None:
                        raise ValueError("Replay failed: sigma_ref missing for intact edges.")
                    if clone.thickness_ref is None:
                        raise ValueError("Replay failed: missing thickness_ref.")
                    beta = float(getattr(clone, "degradation_beta", 1.0))
                    gamma_d = float(getattr(clone, "degradation_gamma", 1.0))
                    sigma = max(0.0, float(F))
                    stress_factor = (float(sigma) / float(sigma_ref)) ** float(beta)
                    thickness_factor = (float(clone.thickness_ref) / float(e.thickness)) ** float(gamma_d)
                    lambda_eff = float(lambda_0) * float(stress_factor) * float(thickness_factor)
                else:
                    lambda_eff = 0.0
                if not np.isfinite(lambda_eff):
                    raise ValueError("Replay failed: invalid lambda_eff (NaN/Inf).")
                if lambda_eff < 0.0:
                    raise ValueError("Replay failed: invalid lambda_eff (negative).")

                S_new = float(S_old) - float(lambda_eff) * float(g_total) * float(dt)
                if S_new < 0.0:
                    S_new = 0.0
                elif S_new > 1.0:
                    S_new = 1.0
            else:
                S_new = 0.0

            if S_old > 0.0 and S_new <= 0.0:
                newly_cleaved += 1

            # Stage 4: lysis tracking (set once on first S<=0).
            lysis_batch_index = e.lysis_batch_index
            lysis_time = e.lysis_time
            if lysis_batch_index is None and float(S_old) > 0.0 and float(S_new) <= 0.0:
                lysis_batch_index = int(expected["batch_index"])
                lysis_time = float(expected["time"])
                newly_lysed_edge_ids.append(int(e.edge_id))

            new_edges.append(
                Phase1EdgeSnapshot(
                    edge_id=int(e.edge_id),
                    n_from=int(e.n_from),
                    n_to=int(e.n_to),
                    k0=float(e.k0),
                    original_rest_length=float(e.original_rest_length),
                    L_rest_effective=float(L_eff),
                    M=float(M_i),
                    S=float(S_new),
                    thickness=float(e.thickness),
                    lysis_batch_index=(int(lysis_batch_index) if lysis_batch_index is not None else None),
                    lysis_time=(float(lysis_time) if lysis_time is not None else None),
                )
            )

        # cleavage batch cap check (abort)
        if intact_pre > 0:
            frac = float(newly_cleaved) / float(intact_pre)
            if frac > float(clone.cleavage_batch_cap):
                raise ValueError("Replay aborted: cleavage batch cap exceeded.")

        clone.set_edges(new_edges)
        # exactly one relaxation after degradation (as in main path)
        clone.relax(float(strain))

        # post-relax mean tension and lysis
        post_intact_forces = [max(0.0, float(clone._forces_by_edge_id[int(e.edge_id)])) for e in clone.edges if float(e.S) > 0.0]
        post_mean_tension = float(sum(post_intact_forces) / len(post_intact_forces)) if post_intact_forces else 0.0
        assert np.isfinite(post_mean_tension)
        # Protofibril-based stiffness scaling for lysis fraction calculation
        if FeatureFlags.USE_SPATIAL_PLASMIN and clone.spatial_plasmin_params:
            N_pf = float(clone.spatial_plasmin_params.get("N_pf", 50))
            total_k0 = sum(float(e.k0) * N_pf for e in clone.edges)
            total_keff = sum(float(e.k0) * N_pf * float(e.S) for e in clone.edges)
        else:
            total_k0 = sum(float(e.k0) for e in clone.edges)
            total_keff = sum(float(e.k0) * float(e.S) for e in clone.edges)
        if total_k0 == 0.0:
            raise ValueError("Replay failed: sum(k0) == 0")
        lysis_fraction = 1.0 - (total_keff / total_k0)
        assert np.isfinite(lysis_fraction)

        # Stage 4: global lysis time (set once; must match main logic).
        threshold = float(clone.frozen_params.get("global_lysis_threshold")) if isinstance(clone.frozen_params, dict) and ("global_lysis_threshold" in clone.frozen_params) else 0.9
        if clone.global_lysis_batch_index is None and float(lysis_fraction) >= float(threshold):
            clone.global_lysis_batch_index = int(expected["batch_index"])
            clone.global_lysis_time = float(expected["time"])

        # Phase 3.6: replay RNG consistency check against logged hash.
        replay_rng_hash = hashlib.sha256(str(clone.rng.getstate()).encode("utf-8")).hexdigest()
        expected_rng_hash = str(expected.get("rng_state_hash"))
        if replay_rng_hash != expected_rng_hash:
            raise ValueError("Replay failed: RNG state hash mismatch after replay.")

        # Phase 3.7: deterministic batch_hash enforcement (end-to-end integrity).
        def _batch_hash_payload(batch_index: int, time_val: float, strain_val: float, sigma_ref_val: float | None, selected_ids: list[int], edges_sorted: list[Phase1EdgeSnapshot]):
            return {
                "batch_index": int(batch_index),
                "time": float(time_val),
                "strain": float(strain_val),
                "sigma_ref": (float(sigma_ref_val) if sigma_ref_val is not None else None),
                "plasmin_selected_edge_ids": list(selected_ids),
                "global_lysis_batch_index": clone.global_lysis_batch_index,
                "global_lysis_time": clone.global_lysis_time,
                "edges": [
                    {
                        "edge_id": int(e.edge_id),
                        "S": float(e.S),
                        "M": float(e.M),
                        "original_rest_length": float(e.original_rest_length),
                        "L_rest_effective": float(e.L_rest_effective),
                        "thickness": float(e.thickness),
                        "lysis_batch_index": (int(e.lysis_batch_index) if e.lysis_batch_index is not None else None),
                        "lysis_time": (float(e.lysis_time) if e.lysis_time is not None else None),
                    }
                    for e in edges_sorted
                ],
                "frozen_params": copy.deepcopy(self.frozen_params),
                "provenance_hash": self.provenance_hash,
                "rng_state_hash": self.frozen_rng_state_hash,
            }

        edges_sorted_after = sorted(clone.edges, key=lambda ee: int(ee.edge_id))
        payload = _batch_hash_payload(
            batch_index=int(expected["batch_index"]),
            time_val=float(expected["time"]),
            strain_val=float(expected["strain"]),
            sigma_ref_val=sigma_ref,
            selected_ids=selected_edge_ids,
            edges_sorted=edges_sorted_after,
        )
        replay_batch_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        expected_batch_hash = expected.get("batch_hash")
        if replay_batch_hash != expected_batch_hash:
            raise ValueError("Replay failed: batch_hash mismatch (end-to-end integrity check).")

        exp_sel = expected.get("plasmin_selected_edge_ids", None)
        if exp_sel is not None:
            if list(exp_sel) != list(selected_edge_ids):
                raise ValueError("Replay failed: plasmin_selected_edge_ids mismatch.")

        # Stage 4: replay lysis ordering/time checks (fail loudly on mismatch when present).
        exp_newly = expected.get("newly_lysed_edge_ids", None)
        exp_cum = expected.get("cumulative_lysed_edge_ids", None)
        if exp_newly is not None:
            if list(exp_newly) != sorted([int(eid) for eid in newly_lysed_edge_ids]):
                raise ValueError("Replay failed: newly_lysed_edge_ids mismatch.")
        if exp_cum is not None:
            cum_now = sorted([int(e.edge_id) for e in clone.edges if e.lysis_batch_index is not None])
            if list(exp_cum) != cum_now:
                raise ValueError("Replay failed: cumulative_lysed_edge_ids mismatch.")

        return {
            "newly_cleaved": int(newly_cleaved),
            "mean_tension": float(post_mean_tension),
            "lysis_fraction": float(lysis_fraction),
        }

    def load_checkpoint(self, snapshot_path: str, log_path: str, resume_batch_index: int):
        """
        Phase 4.1: deterministic checkpoint/resume (batch-index addressable).

        Loads:
        - Phase 3.3 snapshot (JSON)
        - Experiment log (CSV or JSON)

        Validates:
        - provenance_hash matches
        - rng_state_hash matches
        - batch_hash at resume_batch_index matches snapshot batch_hash

        Restores (atomically on success):
        - edge state (S, M, original_rest_length, L_rest_effective)
          - v5.0 spatial mode: segments per edge (n_i, B_i, S_i)
        - relaxed node positions from snapshot
        - frozen_params + provenance_hash
        - frozen_rng_state (sets adapter.rng state)
        - v5.0 spatial mode: spatial_plasmin_params + plasmin pool (P_total_quanta, P_free_quanta)
        - experiment_log truncated to resume_batch_index + 1
        """
        if not isinstance(resume_batch_index, int) or resume_batch_index < 0:
            raise ValueError("resume_batch_index must be a non-negative integer.")

        # Load snapshot
        with open(snapshot_path, "r", encoding="utf-8") as f:
            snap = json.load(f)
        if not isinstance(snap, dict):
            raise ValueError("Invalid snapshot format.")

        snap_prov = snap.get("provenance_hash")
        snap_frozen_params = snap.get("frozen_params")
        snap_rng_hash = snap.get("rng_state_hash")
        snap_rng_state = snap.get("frozen_rng_state")
        snap_batch_hash = snap.get("batch_hash")
        if snap_prov is None or snap_frozen_params is None or snap_rng_hash is None or snap_batch_hash is None:
            raise ValueError("Snapshot missing required provenance fields (provenance_hash/frozen_params/rng_state_hash/batch_hash).")
        if snap_rng_state is None:
            raise ValueError("Snapshot missing frozen_rng_state required for resume.")

        # Load log
        log_ext = os.path.splitext(log_path)[1].lower()
        if log_ext == ".json":
            with open(log_path, "r", encoding="utf-8") as f:
                log_entries = json.load(f)
            if not isinstance(log_entries, list):
                raise ValueError("Invalid log JSON format: expected a list.")
        elif log_ext == ".csv":
            log_entries = []
            with open(log_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Rehydrate structure matching Phase 3.1/3.7 schema.
                    params = {}
                    for k, v in row.items():
                        if k.startswith("param_"):
                            params[k.replace("param_", "")] = float(v)
                    log_entries.append(
                        {
                            "batch_index": int(float(row["batch_index"])),
                            "provenance_hash": row.get("provenance_hash"),
                            "rng_state_hash": row.get("rng_state_hash"),
                            "batch_hash": row.get("batch_hash"),
                            "batch_duration_sec": float(row.get("batch_duration_sec", 0.0)),
                            "time": float(row["time"]),
                            "strain": float(row["strain"]),
                            "intact_edges": int(float(row["intact_edges"])),
                            "cleaved_edges_total": int(float(row["cleaved_edges_total"])),
                            "newly_cleaved": int(float(row["newly_cleaved"])),
                            "mean_tension": float(row["mean_tension"]),
                            "lysis_fraction": float(row["lysis_fraction"]),
                            "params": params,
                        }
                    )
        else:
            raise ValueError("Unsupported log type. Provide .csv or .json.")

        if resume_batch_index >= len(log_entries):
            raise ValueError("resume_batch_index out of range for provided log.")
        entry = log_entries[resume_batch_index]

        # Validate provenance/RNG/batch hash against log entry at resume index
        if entry.get("provenance_hash") != snap_prov:
            raise ValueError("Checkpoint resume failed: provenance_hash mismatch between snapshot and log.")
        if entry.get("rng_state_hash") != snap_rng_hash:
            raise ValueError("Checkpoint resume failed: rng_state_hash mismatch between snapshot and log.")
        if entry.get("batch_hash") != snap_batch_hash:
            raise ValueError("Checkpoint resume failed: batch_hash mismatch between snapshot and log entry at resume index.")

        # Validate frozen_params hash matches provenance_hash (deterministic)
        if not isinstance(snap_frozen_params, dict):
            raise ValueError("Checkpoint resume failed: frozen_params missing or invalid.")
        # Stage 2 requirement: older checkpoints without thickness_ref must fail loudly.
        if "thickness_ref" not in snap_frozen_params or "thickness_alpha" not in snap_frozen_params:
            raise ValueError("Checkpoint resume failed: missing thickness_ref/thickness_alpha (Stage 2 required).")
        # Stage 3 requirement: beta/gamma must be present.
        if "beta" not in snap_frozen_params or "gamma" not in snap_frozen_params:
            raise ValueError("Checkpoint resume failed: missing beta/gamma (Stage 3 required).")
        # Explicit boundary membership is required (no heuristics; deterministic clamps).
        if "left_boundary_node_ids" not in snap_frozen_params or "right_boundary_node_ids" not in snap_frozen_params:
            raise ValueError(
                "Checkpoint resume failed: missing left_boundary_node_ids/right_boundary_node_ids in frozen_params. "
                "Boundary nodes must be explicitly specified via is_left_boundary / is_right_boundary."
            )
        if "left_grip_x" not in snap_frozen_params or "right_grip_x" not in snap_frozen_params:
            raise ValueError("Checkpoint resume failed: missing left_grip_x/right_grip_x in frozen_params (rigid grips required).")
        if "initial_boundary_y" not in snap_frozen_params:
            raise ValueError("Checkpoint resume failed: missing initial_boundary_y in frozen_params (rigid clamps require y-fix).")
        # Stage 4 requirement: global lysis threshold must be present.
        if "global_lysis_threshold" not in snap_frozen_params:
            raise ValueError("Checkpoint resume failed: missing global_lysis_threshold (Stage 4 required).")
        # Stage 5 requirement: plasmin mode selection must be present.
        if "plasmin_mode" not in snap_frozen_params or "N_plasmin" not in snap_frozen_params:
            raise ValueError("Checkpoint resume failed: missing plasmin_mode/N_plasmin (Stage 5 required).")
        frozen_json = json.dumps(snap_frozen_params, sort_keys=True)
        prov_calc = hashlib.sha256(frozen_json.encode("utf-8")).hexdigest()
        if prov_calc != snap_prov:
            raise ValueError("Checkpoint resume failed: frozen_params do not reproduce provenance_hash.")

        # Reconstruct edge snapshots from snapshot edges, preserving current k0 mapping if present.
        edges_in = snap.get("edges")
        nodes_in = snap.get("nodes")
        if not isinstance(edges_in, list) or not isinstance(nodes_in, list):
            raise ValueError("Invalid snapshot format: 'nodes' and 'edges' must be lists.")

        # If snapshot provides node_coords/boundaries, prefer them; otherwise keep existing.
        initial_node_coords = snap.get("initial_node_coords", None)
        left_boundary_ids = snap.get("left_boundary_node_ids", None)
        right_boundary_ids = snap.get("right_boundary_node_ids", None)
        if isinstance(initial_node_coords, dict):
            self.node_coords = {int(k): (float(v[0]), float(v[1])) for k, v in initial_node_coords.items()}
            self._initial_node_coords = dict(self.node_coords)

        # Boundary membership is an explicit, frozen experimental input.
        # Source of truth: snapshot.frozen_params (hashed into provenance).
        fp_left = snap_frozen_params.get("left_boundary_node_ids")
        fp_right = snap_frozen_params.get("right_boundary_node_ids")
        if not isinstance(fp_left, list) or not isinstance(fp_right, list):
            raise ValueError("Checkpoint resume failed: invalid boundary node lists in frozen_params.")
        fp_left_ids = [int(x) for x in fp_left]
        fp_right_ids = [int(x) for x in fp_right]
        if not fp_left_ids or not fp_right_ids:
            raise ValueError("Checkpoint resume failed: boundary node lists are empty.")
        if set(fp_left_ids).intersection(set(fp_right_ids)):
            raise ValueError("Checkpoint resume failed: invalid boundary specification (node marked both left and right).")
        # Optional cross-check if snapshot includes explicit boundary lists at top-level.
        if isinstance(left_boundary_ids, list) and sorted([int(x) for x in left_boundary_ids]) != sorted(fp_left_ids):
            raise ValueError("Checkpoint resume failed: left_boundary_node_ids mismatch between snapshot and frozen_params.")
        if isinstance(right_boundary_ids, list) and sorted([int(x) for x in right_boundary_ids]) != sorted(fp_right_ids):
            raise ValueError("Checkpoint resume failed: right_boundary_node_ids mismatch between snapshot and frozen_params.")
        # Ensure boundary node IDs exist in imported node set.
        for nid in fp_left_ids + fp_right_ids:
            if int(nid) not in self._initial_node_coords:
                raise ValueError(f"Checkpoint resume failed: boundary node_id {nid} not present in snapshot/imported nodes.")
        self.left_boundary_node_ids = frozenset(fp_left_ids)
        self.right_boundary_node_ids = frozenset(fp_right_ids)

        relaxed_coords: dict[int, tuple[float, float]] = {int(n["node_id"]): (float(n["x"]), float(n["y"])) for n in nodes_in}
        self._relaxed_node_coords = dict(relaxed_coords)

        # Keep k0 from existing edges if possible (snapshot edges don't include it).
        k0_by_edge_id = {int(e.edge_id): float(e.k0) for e in self._edges}
        rebuilt_edges: list[Phase1EdgeSnapshot] = []
        for e in edges_in:
            eid = int(e["edge_id"])
            if eid not in k0_by_edge_id:
                raise ValueError(f"Checkpoint resume failed: edge_id {eid} not present in current adapter.")
            if "thickness" not in e:
                raise ValueError(f"Checkpoint resume failed: snapshot missing required per-edge thickness for edge_id {eid}.")
            if "lysis_batch_index" not in e or "lysis_time" not in e:
                raise ValueError(f"Checkpoint resume failed: snapshot missing required lysis metadata for edge_id {eid}.")
            thickness = float(e["thickness"])
            if (not np.isfinite(thickness)) or (thickness <= 0.0):
                raise ValueError(f"Checkpoint resume failed: invalid thickness in snapshot for edge_id {eid}: {e.get('thickness')}")
            lysis_batch_index = e.get("lysis_batch_index", None)
            lysis_time = e.get("lysis_time", None)
            if lysis_batch_index is not None:
                lysis_batch_index = int(lysis_batch_index)
            if lysis_time is not None:
                lysis_time = float(lysis_time)
            # v5.0 spatial mode: restore per-segment state if present
            segments = None
            seg_in = e.get("segments", None)
            if isinstance(seg_in, list):
                rebuilt_segments: list[FiberSegment] = []
                for s in seg_in:
                    rebuilt_segments.append(
                        FiberSegment(
                            segment_index=int(s["segment_index"]),
                            n_i=float(s["n_i"]),
                            B_i=float(s["B_i"]),
                            S_i=float(s["S_i"]),
                        )
                    )
                segments = tuple(rebuilt_segments)
            rebuilt_edges.append(
                Phase1EdgeSnapshot(
                    edge_id=eid,
                    n_from=int(e["n_from"]),
                    n_to=int(e["n_to"]),
                    k0=float(k0_by_edge_id[eid]),
                    original_rest_length=float(e["original_rest_length"]),
                    L_rest_effective=float(e["L_rest_effective"]),
                    M=float(e["M"]),
                    S=float(e["S"]),
                    thickness=float(thickness),
                    lysis_batch_index=lysis_batch_index,
                    lysis_time=lysis_time,
                    segments=segments,
                )
            )
            # Rest-length validity (critical for mechanics): must be finite and > 0.
            if (not np.isfinite(float(rebuilt_edges[-1].original_rest_length))) or float(rebuilt_edges[-1].original_rest_length) <= 0.0:
                raise ValueError(f"Checkpoint resume failed: invalid original_rest_length for edge_id {eid} (must be finite and > 0).")
            if (not np.isfinite(float(rebuilt_edges[-1].L_rest_effective))) or float(rebuilt_edges[-1].L_rest_effective) <= 0.0:
                raise ValueError(f"Checkpoint resume failed: invalid L_rest_effective for edge_id {eid} (must be finite and > 0).")
        self.set_edges(rebuilt_edges)

        # Restore frozen params/provenance
        self.frozen_params = copy.deepcopy(snap_frozen_params)
        self.provenance_hash = str(snap_prov)
        # Restore rigid grip positions (frozen experimental inputs).
        self.left_grip_x = float(self.frozen_params.get("left_grip_x"))
        self.right_grip_x = float(self.frozen_params.get("right_grip_x"))
        if (not np.isfinite(float(self.left_grip_x))) or (not np.isfinite(float(self.right_grip_x))):
            raise ValueError("Checkpoint resume failed: non-finite rigid grip x positions.")
        # Restore boundary y constraints (frozen experimental inputs).
        iby = self.frozen_params.get("initial_boundary_y")
        if not isinstance(iby, list):
            raise ValueError("Checkpoint resume failed: invalid initial_boundary_y in frozen_params (expected list of (node_id,y)).")
        restored: dict[int, float] = {}
        for pair in iby:
            try:
                nid = int(pair[0])
                y0 = float(pair[1])
            except Exception as e:
                raise ValueError("Checkpoint resume failed: invalid initial_boundary_y entry.") from e
            if not np.isfinite(y0):
                raise ValueError(f"Checkpoint resume failed: non-finite boundary y for node_id {nid}.")
            restored[nid] = float(y0)
        self.initial_boundary_y = dict(restored)

        # Final authority: clamp restored relaxed geometry x-coordinates to grips (y remains free).
        if self._relaxed_node_coords is not None:
            coords = dict(self._relaxed_node_coords)
            gxL = float(self.left_grip_x)
            gxR = float(self.right_grip_x)
            for nid in self.left_boundary_node_ids:
                if int(nid) in coords:
                    _x, y = coords[int(nid)]
                    y0 = float(self.initial_boundary_y.get(int(nid), float(y)))
                    coords[int(nid)] = (gxL, float(y0))
            for nid in self.right_boundary_node_ids:
                if int(nid) in coords:
                    _x, y = coords[int(nid)]
                    y0 = float(self.initial_boundary_y.get(int(nid), float(y)))
                    coords[int(nid)] = (gxR, float(y0))
            self._relaxed_node_coords = dict(coords)
        self.thickness_ref = float(self.frozen_params.get("thickness_ref"))
        self.thickness_alpha = float(self.frozen_params.get("thickness_alpha"))
        self.degradation_beta = float(self.frozen_params.get("beta"))
        self.degradation_gamma = float(self.frozen_params.get("gamma"))
        self.plasmin_mode = str(self.frozen_params.get("plasmin_mode"))
        self.N_plasmin = int(self.frozen_params.get("N_plasmin"))
        # v5.0 spatial mode: restore spatial params + plasmin pool state (if present)
        snap_spatial_params = snap.get("spatial_plasmin_params", None)
        self.spatial_plasmin_params = copy.deepcopy(snap_spatial_params) if isinstance(snap_spatial_params, dict) else None
        snap_P_total = snap.get("P_total_quanta", None)
        snap_P_free = snap.get("P_free_quanta", None)
        if (snap_P_total is None) != (snap_P_free is None):
            raise ValueError("Checkpoint resume failed: snapshot plasmin pool is incomplete (need both P_total_quanta and P_free_quanta).")
        self.P_total_quanta = (int(snap_P_total) if snap_P_total is not None else None)
        self.P_free_quanta = (int(snap_P_free) if snap_P_free is not None else None)

        # Restore RNG state exactly
        rng_state_tuple = _tuplify(snap_rng_state)
        self.frozen_rng_state = rng_state_tuple
        self.frozen_rng_state_hash = hashlib.sha256(str(rng_state_tuple).encode("utf-8")).hexdigest()
        if self.frozen_rng_state_hash != snap_rng_hash:
            raise ValueError("Checkpoint resume failed: frozen_rng_state does not match rng_state_hash.")
        if self.rng is None:
            self.rng = random.Random()
        self.rng.setstate(self.frozen_rng_state)

        # Restore experiment log up to resume index (next batch index is resume_index + 1).
        self.experiment_log = list(log_entries[: resume_batch_index + 1])

        # Restore global lysis tracking if present in snapshot/log (Stage 4).
        self.global_lysis_threshold = float(self.frozen_params.get("global_lysis_threshold")) if isinstance(self.frozen_params, dict) and ("global_lysis_threshold" in self.frozen_params) else None
        self.global_lysis_batch_index = snap.get("global_lysis_batch_index", None)
        self.global_lysis_time = snap.get("global_lysis_time", None)

        # Restore prev_mean_tension for Phase 2.2 continuity (from log entry at resume index).
        self.prev_mean_tension = float(entry.get("mean_tension")) if entry.get("mean_tension") is not None else None

        # Thickness consistency (explicit; reproducibility spine).
        thickness_pairs = [(int(es.edge_id), float(es.thickness)) for es in self._edges]
        thickness_pairs.sort(key=lambda t: t[0])
        thickness_hash = hashlib.sha256(json.dumps(thickness_pairs, sort_keys=True).encode("utf-8")).hexdigest()
        if isinstance(self.frozen_params, dict) and ("thickness_hash" in self.frozen_params):
            if str(self.frozen_params.get("thickness_hash")) != str(thickness_hash):
                raise ValueError("Checkpoint resume failed: thickness_hash mismatch between frozen_params and snapshot edge thickness data.")

        # Stage 4: lysis history consistency check if log includes IDs (JSON log).
        entry_newly = entry.get("newly_lysed_edge_ids", None) if isinstance(entry, dict) else None
        entry_cum = entry.get("cumulative_lysed_edge_ids", None) if isinstance(entry, dict) else None
        if entry_cum is not None:
            cum_now = sorted([int(e.edge_id) for e in self._edges if e.lysis_batch_index is not None])
            if list(entry_cum) != cum_now:
                raise ValueError("Checkpoint resume failed: cumulative_lysed_edge_ids mismatch between log and snapshot.")

        # Applied strain fixed-parameter compatibility:
        # - If present in frozen_params, enforce that it matches the logged strain at resume index.
        # - If absent (older experiments), do not enforce.
        if isinstance(snap_frozen_params, dict) and "applied_strain" in snap_frozen_params:
            if float(snap_frozen_params.get("applied_strain")) != float(entry.get("strain", 0.0)):
                raise ValueError("Checkpoint resume failed: applied_strain in frozen_params does not match logged strain.")
            self.applied_strain = float(snap_frozen_params.get("applied_strain"))

        # Recompute cached forces from restored geometry deterministically (must match solver input stiffness).
        forces_by_edge_id: dict[int, float] = {}
        for e in self._edges:
            if float(e.S) <= 0.0:
                forces_by_edge_id[int(e.edge_id)] = 0.0
                continue
            p_from = self._relaxed_node_coords.get(int(e.n_from))
            p_to = self._relaxed_node_coords.get(int(e.n_to))
            if p_from is None or p_to is None:
                raise ValueError("Checkpoint resume failed: snapshot missing node coordinates referenced by an edge.")
            L = _euclidean(p_from, p_to)
            # Protofibril-based stiffness scaling (spatial mode only)
            if FeatureFlags.USE_SPATIAL_PLASMIN and self.spatial_plasmin_params:
                N_pf = float(self.spatial_plasmin_params.get("N_pf", 50))
                k_base = float(e.k0) * N_pf * float(e.S)
            else:
                k_base = float(e.k0) * float(e.S)
            if self.thickness_ref is None or self.thickness_alpha is None:
                k_eff = k_base
            else:
                t_ref = float(self.thickness_ref)
                alpha = float(self.thickness_alpha)
                scale = (float(e.thickness) / t_ref) ** alpha
                k_eff = k_base * float(scale)
            # NUMERICAL STABILITY: Hard ceiling on effective stiffness (k_eff_max = 1e12 N/m)
            k_eff_max = 1e12  # N/m
            if not np.isfinite(k_eff):
                raise ValueError(f"Non-finite k_eff during checkpoint resume for edge {e.edge_id}.")
            if k_eff > k_eff_max:
                raise ValueError(
                    f"Effective stiffness overflow during checkpoint resume for edge {e.edge_id}: "
                    f"k_eff = {k_eff:.3e} exceeds k_eff_max = {k_eff_max:.3e} N/m."
                )
            forces_by_edge_id[int(e.edge_id)] = float(float(k_eff) * (float(L) - float(e.L_rest_effective)))
        self._forces_by_edge_id = dict(forces_by_edge_id)

        return {"resume_batch_index": int(resume_batch_index), "resume_time": float(entry["time"]), "resume_strain": float(entry.get("strain", 0.0))}

    def fork_from_checkpoint(self, snapshot_path: str, log_path: str, resume_batch_index: int) -> "Phase1NetworkAdapter":
        """
        Phase 4.2: deterministic branching / fork-from-checkpoint (copy-on-write).

        Produces a NEW adapter initialized from a valid checkpoint without mutating the current adapter.
        The fork preserves provenance_hash/frozen_params/rng lineage and records its parent batch metadata.
        """
        # Validate and restore checkpoint state on a temporary adapter (so the live adapter is unchanged).
        temp = Phase1NetworkAdapter(
            path=self.path,
            node_coords=dict(self.node_coords),
            left_boundary_node_ids=tuple(sorted(self.left_boundary_node_ids)),
            right_boundary_node_ids=tuple(sorted(self.right_boundary_node_ids)),
            relax_impl=self._relax_impl,
        )
        # Provide current k0 mapping (snapshot edges do not include k0).
        temp.set_edges(list(self._edges))
        info = temp.load_checkpoint(snapshot_path, log_path, int(resume_batch_index))

        parent_index = int(info["resume_batch_index"])
        parent_hash = temp.experiment_log[parent_index].get("batch_hash")

        # Deep-copy restored state into the forked adapter (copy-on-write).
        forked = Phase1NetworkAdapter(
            path=temp.path,
            node_coords=dict(temp.node_coords),
            left_boundary_node_ids=tuple(sorted(temp.left_boundary_node_ids)),
            right_boundary_node_ids=tuple(sorted(temp.right_boundary_node_ids)),
            relax_impl=temp._relax_impl,
        )
        forked.set_edges(list(temp.edges))
        forked._initial_node_coords = dict(temp._initial_node_coords)
        forked._relaxed_node_coords = dict(temp._relaxed_node_coords) if temp._relaxed_node_coords is not None else None
        forked._forces_by_edge_id = dict(temp._forces_by_edge_id)

        # Preserve provenance / frozen params
        forked.frozen_params = copy.deepcopy(temp.frozen_params)
        forked.provenance_hash = temp.provenance_hash

        # Preserve RNG lineage and restore RNG state
        forked.frozen_rng_state = temp.frozen_rng_state
        forked.frozen_rng_state_hash = temp.frozen_rng_state_hash
        forked.rng = random.Random()
        if forked.frozen_rng_state is None:
            raise ValueError("Fork failed: missing frozen RNG state.")
        forked.rng.setstate(forked.frozen_rng_state)

        # Preserve model parameters/constants (copied)
        forked.lambda_0 = temp.lambda_0
        forked.dt = temp.dt
        forked.delta = temp.delta
        forked.g_force = temp.g_force
        forked.modifier = temp.modifier
        forked.prev_mean_tension = temp.prev_mean_tension
        forked.rate_beta = temp.rate_beta
        forked.rate_eps0 = temp.rate_eps0
        forked.plastic_F_threshold = temp.plastic_F_threshold
        forked.plastic_rate = temp.plastic_rate
        forked.rupture_force_threshold = temp.rupture_force_threshold
        forked.rupture_gamma = temp.rupture_gamma
        forked.fracture_Gc = temp.fracture_Gc
        forked.fracture_eta = temp.fracture_eta
        forked.coop_chi = temp.coop_chi
        forked.shield_eps = temp.shield_eps
        forked.memory_mu = temp.memory_mu
        forked.memory_rho = temp.memory_rho
        forked.aniso_kappa = temp.aniso_kappa
        forked.g_max = temp.g_max
        forked.cleavage_batch_cap = temp.cleavage_batch_cap
        forked.force_alpha = getattr(temp, "force_alpha", None)
        forked.force_F0 = getattr(temp, "force_F0", None)
        forked.force_hill_n = getattr(temp, "force_hill_n", None)

        # Preserve log up to parent batch; next batch index begins at parent_index + 1.
        forked.experiment_log = list(temp.experiment_log)

        # Branch parent provenance metadata (stamped into future log entries).
        forked.branch_parent_batch_index = parent_index
        forked.branch_parent_batch_hash = str(parent_hash) if parent_hash is not None else None

        return forked

    def _build_existing_solver_relax_impl(self):
        """
        Build an internal relax_impl closure that:
        - constructs a fresh network instance (no shared objects)
        - fixes boundary nodes (left/right attachments)
        - applies strain by increasing pole separation deterministically
        - calls the existing relaxation solver once
        - caches relaxed node coordinates on the adapter
        - returns per-edge forces for intact edges only in the required order
        """
        # Local imports: keep solver/network internals contained inside adapter boundary.
        from src.managers.network.degradation_engine.two_dimensional_spring_force_degradation_engine_without_biomechanics import (
            TwoDimensionalSpringForceDegradationEngineWithoutBiomechanics,
        )
        from src.managers.network.networks.network_2d import Network2D
        from src.managers.network.nodes.fixable_node_2d import FixableNode2D
        from src.managers.network.edges.edge_with_rest_length import EdgeWithRestLength

        engine = TwoDimensionalSpringForceDegradationEngineWithoutBiomechanics()

        def relax_impl(edges_snapshots: Sequence[Phase1EdgeSnapshot], k_eff_intact: Sequence[float], strain: float) -> Sequence[float]:
            # Adapter–solver reconciliation (Phase 1D):
            # The legacy solver operates on the network edge list we provide. During batch execution,
            # `k_eff_intact` corresponds to the *post-degradation* intact-edge list constructed inside
            # DegradationBatchStep, while `edges_snapshots` reflects the adapter's pre-batch snapshot.
            #
            # To ensure ordering/length invariance, we deterministically map `k_eff_intact` back onto
            # the correct edge subset by scanning edges once in a fixed order and consuming stiffness
            # entries in order.
            #
            # In Phase 1, each edge's strength either stays the same or decreases by `delta` in a batch,
            # so k_eff for a given edge must match either k0*S or k0*max(S-delta,0).
            intact_edges: list[Phase1EdgeSnapshot] = []
            k_eff_list: list[float] = []
            solver_edge_ids_in_order: list[Any] = []

            j = 0
            delta = getattr(self, "delta", None)
            delta = float(delta) if (delta is not None) else None

            for e in edges_snapshots:
                if j >= len(k_eff_intact):
                    break
                # Candidate effective stiffness values for this edge after a single batch:
                # - no-hit: k0*S
                # - hit: k0*max(S-delta,0)
                k0 = float(e.k0)
                s_old = float(e.S)
                # Stage 2 thickness-aware mechanics: stiffness scaling is part of the solver input.
                if getattr(self, "thickness_ref", None) is None or getattr(self, "thickness_alpha", None) is None:
                    thick_scale = 1.0
                else:
                    t_ref = float(getattr(self, "thickness_ref"))
                    if (not np.isfinite(t_ref)) or (t_ref <= 0.0):
                        raise ValueError("Invalid thickness_ref (must be finite and > 0).")
                    alpha = float(getattr(self, "thickness_alpha"))
                    thick_scale = (float(e.thickness) / t_ref) ** alpha
                    if not np.isfinite(thick_scale):
                        raise ValueError("Non-finite thickness stiffness scale factor.")

                cand_no_hit = (k0 * s_old) * float(thick_scale)
                if delta is None:
                    cand_hit = None
                else:
                    cand_hit = (k0 * max(s_old - delta, 0.0)) * float(thick_scale)

                k_target = float(k_eff_intact[j])
                tol_k = 1e-9 * max(1.0, abs(k_target), abs(cand_no_hit))
                matches = abs(k_target - cand_no_hit) <= tol_k
                if (not matches) and (cand_hit is not None):
                    matches = abs(k_target - cand_hit) <= tol_k

                if matches:
                    intact_edges.append(e)
                    k_eff_list.append(k_target)  # use caller-provided k_eff exactly
                    solver_edge_ids_in_order.append(e.edge_id)
                    j += 1
                else:
                    # This edge is not present in the post-degradation intact list for this call.
                    # (Either cleaved or skipped by the step's intact-edge filter.)
                    continue

            if j != len(k_eff_intact):
                raise ValueError("Solver input reconciliation failed: could not map k_eff_intact onto edge snapshots deterministically.")

            # Defensive assertion after remapping (required).
            assert len(k_eff_list) == len(intact_edges)

            if len(intact_edges) == 0:
                self._relaxed_node_coords = dict(self._initial_node_coords)
                return []

            # Existing solver supports per-edge stiffness via edge.spring_constant.
            # Global fallback remains available for edges without spring_constant.
            k0_global = float(k_eff_list[0]) if k_eff_list else 1.0

            # Rigid grips (uniaxial x-constraint only):
            # - All left boundary nodes share the SAME x = left_grip_x.
            # - All right boundary nodes share the SAME x = right_grip_x.
            # - y remains unconstrained.
            # - After Start, grip positions are frozen on the adapter (no per-node logic).
            fixed_left = set(self.left_boundary_node_ids)
            fixed_right = set(self.right_boundary_node_ids)
            if not fixed_left or not fixed_right:
                raise ValueError("Boundary nodes must be explicitly specified via is_left_boundary / is_right_boundary.")

            if (getattr(self, "left_grip_x", None) is not None) and (getattr(self, "right_grip_x", None) is not None):
                x_left_pole = float(getattr(self, "left_grip_x"))
                x_right_pole = float(getattr(self, "right_grip_x"))
            else:
                # Pre-Start fallback: derive provisional grips deterministically from ORIGINAL boundary-node x positions.
                left_xs0 = [float(self._initial_node_coords[int(nid)][0]) for nid in fixed_left]
                right_xs0 = [float(self._initial_node_coords[int(nid)][0]) for nid in fixed_right]
                left_grip_x0 = float(_median(left_xs0))
                right_grip_x0 = float(_median(right_xs0))
                base_width = float(right_grip_x0 - left_grip_x0)
                if (not np.isfinite(base_width)) or base_width <= 0.0:
                    raise ValueError("Invalid rigid-grip baseline width (must be > 0). Check boundary flags and node coordinates.")
                x_left_pole = float(left_grip_x0)
                x_right_pole = float(right_grip_x0) + float(strain) * float(base_width)
            if (not np.isfinite(x_left_pole)) or (not np.isfinite(x_right_pole)):
                raise ValueError("Non-finite rigid grip x positions (must be finite).")

            # Initial guess coordinates: use last relaxed if available for smooth motion.
            guess = self._relaxed_node_coords or self._initial_node_coords

            nodes = []
            for nid, (x0, y0) in self._initial_node_coords.items():
                gx, gy = guess.get(nid, (x0, y0))
                # Ignore any existing fixed-node semantics: we enforce rigid x-clamping explicitly.
                if nid in fixed_left:
                    gx = x_left_pole
                    gy = float(self.initial_boundary_y.get(int(nid), y0))
                elif nid in fixed_right:
                    gx = x_right_pole
                    gy = float(self.initial_boundary_y.get(int(nid), y0))
                nodes.append(
                    FixableNode2D(
                        {
                            "n_id": int(nid),
                            "n_x": float(gx),
                            "n_y": float(gy),
                            "is_fixed": False,
                        }
                    )
                )

            edges = []
            # Only intact edges are passed into the solver (cleaved excluded).
            for i, e in enumerate(intact_edges):
                edge_obj = (
                    EdgeWithRestLength(
                        {
                            "e_id": int(e.edge_id),
                            "n_from": int(e.n_from),
                            "n_to": int(e.n_to),
                            # Phase 2.3: solver must use effective rest length.
                            "rest_length": float(e.L_rest_effective),
                        }
                    )
                )
                # Phase 1D: per-edge stiffness k_i = k0 * S_i is provided transiently via attribute.
                # This does not change solver behavior when absent.
                setattr(edge_obj, "spring_constant", float(k_eff_list[i]))
                # Stage 1 thickness-aware modeling: propagate thickness as read-only data (no physics use).
                setattr(edge_obj, "thickness", float(e.thickness))
                edges.append(edge_obj)

            network = Network2D(
                {
                    "meta_data": {"spring_stiffness_constant": float(k0_global)},
                    "nodes": nodes,
                    "edges": edges,
                }
            )

            # Relax once (deterministic; no external side effects).
            engine.relax_network(network)

            # Final authority: enforce rigid clamp constraints (x AND y) after solver updates,
            # and before computing forces / exposing coordinates to visualization.
            for n in network.get_nodes():
                nid = int(n.n_id)
                if nid in fixed_left:
                    n.n_x = float(x_left_pole)
                    n.n_y = float(self.initial_boundary_y.get(int(nid), float(n.n_y)))
                elif nid in fixed_right:
                    n.n_x = float(x_right_pole)
                    n.n_y = float(self.initial_boundary_y.get(int(nid), float(n.n_y)))

            # Cache relaxed node coordinates (observable-only).
            relaxed_coords: dict[Any, tuple[float, float]] = {}
            for n in network.get_nodes():
                nid = int(n.n_id)
                x = float(n.n_x)
                y = float(n.n_y)
                relaxed_coords[nid] = (x, y)
            self._relaxed_node_coords = dict(relaxed_coords)

            # Compute per-edge scalar forces (tension) deterministically from relaxed geometry:
            # F = k * (length - rest_length)
            forces_by_edge_id: dict[Any, float] = {}
            for e in network.get_edges():
                n_from = network.get_node_by_id(e.n_from)
                n_to = network.get_node_by_id(e.n_to)
                p_from = (float(n_from.n_x), float(n_from.n_y))
                p_to = (float(n_to.n_x), float(n_to.n_y))
                length = _euclidean(p_from, p_to)
                k_edge = getattr(e, "spring_constant", float(k0_global))
                f = float(k_edge) * (float(length) - float(e.rest_length))
                forces_by_edge_id[int(e.get_id())] = f
                # Clear transient per-edge stiffness to prevent any accidental persistence.
                if hasattr(e, "spring_constant"):
                    try:
                        delattr(e, "spring_constant")
                    except Exception:
                        pass
            # Filter/remap solver forces to match adapter `intact_edges` ordering exactly.
            filtered_forces: list[float] = []
            for eid in solver_edge_ids_in_order:
                filtered_forces.append(float(forces_by_edge_id[int(eid)]))

            assert len(filtered_forces) == len(intact_edges)
            return filtered_forces

        return relax_impl

    def update_strengths_from_step_edges(self, step_edges: Sequence[Mapping[str, Any]]):
        """
        Apply DegradationBatchStep output (which persists only 'S') back into immutable snapshots.
        Index-based update only (no new behavior).
        """
        if len(step_edges) != len(self._edges):
            raise ValueError("Step edge list length must equal adapter edge count")
        new_edges: list[Phase1EdgeSnapshot] = []
        for snap, e in zip(self._edges, step_edges):
            new_edges.append(
                Phase1EdgeSnapshot(
                    edge_id=snap.edge_id,
                    n_from=snap.n_from,
                    n_to=snap.n_to,
                    k0=snap.k0,
                    original_rest_length=snap.original_rest_length,
                    L_rest_effective=snap.L_rest_effective,
                    M=snap.M,
                    S=float(e["S"]),
                    thickness=float(snap.thickness),
                    lysis_batch_index=snap.lysis_batch_index,
                    lysis_time=snap.lysis_time,
                )
            )
        self._edges = tuple(new_edges)

    def _as_step_edge_mappings(self) -> list[dict[str, Any]]:
        """
        Provide the edge mappings required by DegradationBatchStep without leaking solver internals.
        Forbidden persistent fields (is_cleaved/k_eff/force) are never included.
        """
        out: list[dict[str, Any]] = []
        for e in self._edges:
            out.append(
                {
                    "edge_id": e.edge_id,
                    "n_from": e.n_from,
                    "n_to": e.n_to,
                    "k0": float(e.k0),
                    # Phase 2.3: solver uses effective rest length; expose it here as well.
                    "rest_length": float(e.L_rest_effective),
                    "S": float(e.S),
                    "thickness": float(e.thickness),  # read-only experimental data (not used in Stage 1)
                }
            )
        return out

    def _forces_for_intact_edges_in_step_order(self) -> list[float]:
        """
        Start-of-batch forces for intact edges only (S>0), in the order expected by the step.
        """
        forces: list[float] = []
        for e in self._edges:
            if float(e.S) > 0.0:
                forces.append(float(self._forces_by_edge_id.get(e.edge_id, 0.0)))
        return forces

    def relax(self, strain: float) -> dict[Any, float]:
        """
        Public adapter API: compute per-edge forces at fixed global strain.
        Returns a mapping edge_id -> force (cleaved edges have force 0).
        """
        # Compute k_eff for intact edges only, transiently.
        k_eff_intact: list[float] = []
        intact_edge_ids: list[Any] = []
        for e in self._edges:
            if float(e.S) > 0.0:
                intact_edge_ids.append(e.edge_id)
                # Protofibril-based stiffness: k0 represents stiffness per protofibril;
                # fiber-level stiffness emerges via multiplication by N_pf.
                if FeatureFlags.USE_SPATIAL_PLASMIN and self.spatial_plasmin_params:
                    N_pf = float(self.spatial_plasmin_params.get("N_pf", 50))
                    k_base = float(e.k0) * N_pf * float(e.S)
                else:
                    k_base = float(e.k0) * float(e.S)
                # Stage 2 thickness-aware mechanics:
                # - thickness_ref computed once at Start; prior to Start we preserve legacy behavior.
                if self.thickness_ref is None or self.thickness_alpha is None:
                    k_eff = k_base
                else:
                    t_ref = float(self.thickness_ref)
                    if (not np.isfinite(t_ref)) or (t_ref <= 0.0):
                        raise ValueError("Invalid thickness_ref (must be finite and > 0).")
                    alpha = float(self.thickness_alpha)
                    scale = (float(e.thickness) / t_ref) ** alpha
                    k_eff = k_base * float(scale)
                if not np.isfinite(k_eff):
                    raise ValueError("Non-finite k_eff computed (thickness scaling).")
                # NUMERICAL STABILITY: Hard ceiling on effective stiffness (k_eff_max = 1e12 N/m)
                # Prevents overflow in stiff bundle configurations (N_pf × k0 × scale).
                # Biologically defensible: 50 protofibrils × realistic modulus → O(1e10–1e12).
                k_eff_max = 1e12  # N/m
                if k_eff > k_eff_max:
                    raise ValueError(
                        f"Effective stiffness overflow detected for edge {e.edge_id}:\n"
                        f"  k_eff = {k_eff:.3e} N/m exceeds k_eff_max = {k_eff_max:.3e} N/m.\n"
                        f"  k0 = {e.k0:.3e}, S = {e.S:.3f}, thickness = {e.thickness:.3e}.\n"
                        f"  Check stiffness parameters or thickness scaling (alpha = {alpha:.3f})."
                    )
                k_eff_intact.append(float(k_eff))

        forces_intact = list(self._relax_with_keff(k_eff_intact, float(strain)))
        if len(forces_intact) != len(intact_edge_ids):
            raise ValueError("Relaxation returned force list length != number of intact edges")

        forces_by_id: dict[Any, float] = {}
        for eid, f in zip(intact_edge_ids, forces_intact):
            forces_by_id[eid] = float(f)

        # Cleaved edges: force is 0 (derived, not stored on edges).
        for e in self._edges:
            if float(e.S) <= 0.0:
                forces_by_id[e.edge_id] = 0.0

        # Cache as last relaxation observable
        self._forces_by_edge_id = dict(forces_by_id)
        # Poster-grade stability: boundary nodes must match rigid clamp (x,y) after every relax (fatal on violation).
        self._assert_grip_invariants(where="relax()")
        return dict(forces_by_id)

    def _assert_grip_invariants(self, *, where: str):
        """
        Fatal invariant check (debug-level but loud):
        After a relaxation, all boundary nodes must have x == grip_x (within tolerance) AND
        y == initial_boundary_y[nid] (within tolerance).
        """
        if self.left_grip_x is None or self.right_grip_x is None:
            return
        if self._relaxed_node_coords is None:
            return
        if not isinstance(self.initial_boundary_y, dict) or not self.initial_boundary_y:
            raise RuntimeError("Rigid clamp invariant check failed: missing initial_boundary_y on adapter.")
        gxL = float(self.left_grip_x)
        gxR = float(self.right_grip_x)
        coords = self._relaxed_node_coords
        bad_left: list[tuple[int, float, float]] = []
        bad_right: list[tuple[int, float, float]] = []
        tolL = 1e-9 * max(1.0, abs(gxL))
        tolR = 1e-9 * max(1.0, abs(gxR))
        tolY = 1e-9 * max(1.0, abs(gxL), abs(gxR))
        for nid in sorted(int(x) for x in self.left_boundary_node_ids):
            xy = coords.get(int(nid))
            if xy is None:
                bad_left.append((int(nid), float("nan"), float("nan")))
            else:
                x, y = float(xy[0]), float(xy[1])
                y0 = float(self.initial_boundary_y.get(int(nid), float("nan")))
                if (not np.isfinite(y0)) or abs(y - y0) > tolY or abs(x - gxL) > tolL:
                    bad_left.append((int(nid), x, y))
        for nid in sorted(int(x) for x in self.right_boundary_node_ids):
            xy = coords.get(int(nid))
            if xy is None:
                bad_right.append((int(nid), float("nan"), float("nan")))
            else:
                x, y = float(xy[0]), float(xy[1])
                y0 = float(self.initial_boundary_y.get(int(nid), float("nan")))
                if (not np.isfinite(y0)) or abs(y - y0) > tolY or abs(x - gxR) > tolR:
                    bad_right.append((int(nid), x, y))
        if bad_left or bad_right:
            raise RuntimeError(
                "Rigid grip invariant violated.\n"
                f"Where: {where}\n"
                f"Expected left_grip_x={gxL}, right_grip_x={gxR}\n"
                f"(y is fixed per-node from initial_boundary_y)\n"
                f"Left boundary mismatches (node_id,x,y): {bad_left[:20]}\n"
                f"Right boundary mismatches (node_id,x,y): {bad_right[:20]}"
            )

    def _relax_with_keff(self, k_eff_intact: Sequence[float], strain: float) -> Sequence[float]:
        """
        Internal hook used by DegradationBatchStep (no solver leakage past adapter).
        """
        if self._relax_impl is None:
            raise NotImplementedError(
                "Phase 1 relaxation solver is not configured for Research Simulation.\n\n"
                "Phase1NetworkAdapter requires a relax_impl wrapper around the existing linear solver."
            )
        return self._relax_impl(self._edges, k_eff_intact, strain)


@runtime_checkable
class SimulationStep(Protocol):
    """
    Pure interface for a single, discrete *simulation step*.

    Intent
    - This protocol documents how future simulation logic will plug into the GUI-only
      controller/state shell without introducing hidden coupling.
    - It is deliberately *non-executable* and makes no assumptions about physics,
      degradation, cleavage, stochastic events, or time stepping.

    Required input (state snapshot)
    - A *read-only snapshot* of the current state. Implementations MUST treat inputs
      as immutable (i.e., do not mutate the provided snapshot object).
    - The snapshot is expected to include (at minimum) the fields present in
      `SimulationState` (loaded_network, strain_value, time, is_running, is_paused),
      plus any additional step-specific read-only context (e.g., loaded network data)
      provided out-of-band by the caller.

    Expected output (state delta)
    - Returns a "delta" describing proposed changes to apply to state *after* the step.
    - The delta MUST be expressible as a plain mapping of field names to new values.
      Example keys might include: "time", "loaded_network", "strain_value", etc.
    - The caller (controller) remains solely responsible for applying the delta and
      triggering re-rendering.

    Invariants (MUST NOT modify)
    - MUST NOT mutate Tkinter widgets, canvases, or any GUI objects.
    - MUST NOT call methods on `ResearchSimulationPage` or `TkinterView`.
    - MUST NOT mutate the controller or global/module state.
    - MUST NOT perform file I/O, networking, or randomness.
    - MUST NOT advance time implicitly (no "tick" semantics are assumed here).

    Execution semantics
    - One call represents one discrete event batch.
    - The step is pure with respect to its input snapshot: same snapshot => same delta.
    - How steps are scheduled (e.g., button press, external clock, queued events) is
      outside the scope of this protocol.
    """

    def __call__(self, state_snapshot: Any) -> Mapping[str, Any]:
        """
        Compute a state delta from a given state snapshot.

        Parameters:
            state_snapshot: A read-only snapshot object representing current state.

        Returns:
            Mapping[str, Any]: A state delta (field -> new value) to apply.
        """
        ...


@dataclass(frozen=True)
class DegradationBatchConfig:
    """
    Explicit parameter bundle for Phase 1 `DegradationBatchStep`.

    Parameters (all explicit; no implicit defaults beyond identity modifiers):
    - lambda_0: baseline degradation rate (units depend on model; not assumed here)
    - delta: degradation hit size applied to strength S (same units as S)
    - dt: batch duration Δt (seconds)
    - g_force: function g(F) mapping an edge force F to a nonnegative multiplier
    - modifier: optional multiplicative modifier hook (defaults to 1.0)

    Notes:
    - This config makes no physics assumptions about g(F) beyond "callable".
    - No stochastic assumptions are made beyond consuming uniform draws provided by rng_state.
    """
    lambda_0: float
    delta: float
    dt: float
    g_force: Callable[[float], float]
    modifier: Callable[[Mapping[str, Any]], float] = lambda _edge: 1.0


class DegradationBatchStep:
    """
    Phase 1 concrete SimulationStep: executes one discrete degradation batch.

    Strict batch order (per spec):
    1) Degradation phase (probabilities computed from START-of-batch forces)
    2) Cleavage phase (S_i <= 0 => cleaved)
    3) Mechanical relaxation (recompute equilibrium using injected linear solver)
    4) Metrics phase (time += dt, mean tension, counts, lysis %)

    Determinism:
    - Randomness is confined to degradation draws.
    - This step is deterministic given identical inputs, including rng_state.

    Input snapshot contract (documentation-only; no dependency on core code):
    - state_snapshot must provide:
      - edges: Sequence[Mapping[str, Any]]
        Required keys per edge mapping:
          - "S": strength (float)  [the ONLY evolving per-edge state persisted]
          - "k0": baseline stiffness (float)
        Optional keys are preserved if present EXCEPT forbidden persistent fields.
      - time: float (seconds)
      - strain_value: float (kept fixed; poles do not move)
      - forces: Sequence[float]
        Start-of-batch forces for intact edges ONLY, in the same order as scanning `edges`
        from left-to-right selecting edges with S_i > 0.
      - linear_solver: Callable[[Sequence[float], float], Sequence[float]]
        Must compute equilibrium forces for intact edges ONLY in the same order as the
        provided k_eff list, with fixed strain_value (poles do not move).

    Output delta (pure mapping):
    - "time": new time (seconds)
    - "edges": new list of edge mappings (with updated "S" only; no "is_cleaved", "k_eff", or "force")
    - "forces": relaxed forces for intact edges ONLY (in the same order as scanning `edges`
      selecting edges with S_i > 0 *after* the degradation/cleavage phase). This is returned
      to support subsequent batches without persisting per-edge force fields.
    - "metrics": mapping with:
        - "mean_tension"
        - "active_fibers"
        - "cleaved_fibers"
        - "lysis_fraction"  (0..1 per formula; no percent scaling applied here)
    """

    def __init__(self, config: DegradationBatchConfig, rng):
        self.config = config
        # Single RNG injected and seeded once per experiment.
        # Randomness is confined to degradation draws.
        self.rng = rng

    def __call__(self, state_snapshot: Any) -> Mapping[str, Any]:
        cfg = self.config

        # --- Snapshot extraction (read-only) ---
        edges_in: Sequence[Mapping[str, Any]] = getattr(state_snapshot, "edges")
        time_in: float = float(getattr(state_snapshot, "time"))
        strain_value: float = float(getattr(state_snapshot, "strain_value"))
        forces_start_intact: Sequence[float] = getattr(state_snapshot, "forces")
        linear_solver: Callable[[Sequence[float], float], Sequence[float]] = getattr(
            state_snapshot, "linear_solver"
        )

        # Identify intact edges at start of batch (S_i > 0), and align forces accordingly.
        intact_indices_start: list[int] = []
        for i, e in enumerate(edges_in):
            if float(e["S"]) > 0.0:
                intact_indices_start.append(i)
        if len(forces_start_intact) != len(intact_indices_start):
            raise ValueError("state_snapshot.forces length must equal number of intact edges at batch start (S>0)")

        # --- 1) Degradation phase ---
        edges_after_deg: list[dict[str, Any]] = []
        intact_force_cursor = 0
        for idx, e in enumerate(edges_in):
            # Copy edge mapping to avoid mutating snapshot structures.
            e_out: dict[str, Any] = dict(e)

            S_i = float(e_out["S"])
            if S_i > 0.0:
                # Use start-of-batch force field (aligned to intact edges only).
                F_i = float(forces_start_intact[intact_force_cursor])
                intact_force_cursor += 1

                g_val = float(cfg.g_force(F_i))
                mod_val = float(cfg.modifier(e_out))

                lambda_eff = float(cfg.lambda_0) * g_val * mod_val
                p_i = 1.0 - math.exp(-lambda_eff * float(cfg.dt))

                u = self.rng.random()
                if u < p_i:
                    S_i = max(S_i - float(cfg.delta), 0.0)
                    e_out["S"] = S_i
            # If S_i <= 0, do nothing in degradation phase (per spec: "for each intact edge")

            edges_after_deg.append(e_out)

        # --- 2) Cleavage phase ---
        # Cleavage is defined strictly as S_i <= 0 (derived, not persisted).
        # Effective stiffness is computed transiently as k_eff = k0 * S_i.
        intact_indices_after: list[int] = []
        k_eff_intact: list[float] = []
        for i, e_out in enumerate(edges_after_deg):
            S_i = float(e_out["S"])
            if S_i > 0.0:
                intact_indices_after.append(i)
                k_eff_intact.append(float(e_out["k0"]) * S_i)

        # --- 3) Mechanical relaxation phase ---
        # Must use injected existing linear solver (no placeholder physics here).
        # Exclude edges with S_i <= 0 entirely from solver input (per correction #4).
        forces_relaxed_intact = list(linear_solver(k_eff_intact, strain_value))
        if len(forces_relaxed_intact) != len(k_eff_intact):
            raise ValueError("linear_solver returned force list length != number of intact edges")

        # --- 4) Metrics phase ---
        time_out = time_in + float(cfg.dt)

        # Forces are obtained exclusively from solver output.
        # For cleaved edges (S<=0), force is treated as 0 transiently (not stored).
        active_forces: list[float] = []
        cleaved_count = 0

        sum_k0 = 0.0
        sum_keff = 0.0

        # Map relaxed forces back onto global edge indices for metrics computation only.
        force_by_edge_index: dict[int, float] = {}
        for i_idx, edge_idx in enumerate(intact_indices_after):
            force_by_edge_index[edge_idx] = float(forces_relaxed_intact[i_idx])

        for i, e_out in enumerate(edges_after_deg):
            S_i = float(e_out["S"])
            k0 = float(e_out["k0"])
            sum_k0 += k0
            if S_i > 0.0:
                sum_keff += (k0 * S_i)
            else:
                cleaved_count += 1

            F = float(force_by_edge_index.get(i, 0.0))
            if (S_i > 0.0) and (F > 0.0):
                active_forces.append(F)

        if sum_k0 == 0.0:
            raise ValueError("Cannot compute lysis_fraction: sum(k0) == 0")

        mean_tension = (sum(active_forces) / len(active_forces)) if active_forces else 0.0
        active_fibers = len(active_forces)
        lysis_fraction = 1.0 - (sum_keff / sum_k0)

        # Output edges: persist only evolving strength S_i; remove forbidden persistent fields.
        edges_out: list[dict[str, Any]] = []
        for e_out in edges_after_deg:
            cleaned = dict(e_out)
            cleaned.pop("ruptured", None)  # Backward compat: remove old key if present
            cleaned.pop("is_cleaved", None)  # Remove derived state (never persist)
            cleaned.pop("k_eff", None)
            cleaned.pop("force", None)
            edges_out.append(cleaned)

        return {
            "time": time_out,
            "edges": edges_out,
            "forces": list(forces_relaxed_intact),
            "metrics": {
                "mean_tension": mean_tension,
                "active_fibers": active_fibers,
                "cleaved_fibers": cleaved_count,
                "lysis_fraction": lysis_fraction,
            },
        }


class SimulationController:
    """
    GUI-only controller for Research Simulation.

    Responsibilities:
    - Own a SimulationState instance
    - Provide explicit methods for state transitions
    - Enforce valid transitions (e.g., cannot pause if not running)
    - Perform no physics, no numerical updates, no time evolution
    """

    def __init__(self, initial_state=None):
        self.state = initial_state or SimulationState(
            loaded_network=None,
            strain_value=0.0,
            time=0.0,
            is_running=False,
            is_paused=False,
        )
        # RNG policy (Option A — locked):
        # - Seed RNG ONCE when a network is loaded (Load action).
        # - Pause / Stop / Start must NOT reseed RNG.
        # - RNG reseeds ONLY when a new network is loaded.
        self.rng = random.Random(0)
        # Last batch observables (metrics) returned by the step; rendered by UI.
        self.last_metrics = None
        # Phase 3.8: last batch wall-clock duration (diagnostics only; not part of physics).
        self.last_batch_duration_sec: float | None = None
        # Phase 4.3: in-memory sweep results (deterministic; no auto-export).
        self.sweep_results: list[dict[str, Any]] = []
        # Phase 4.4: in-memory grid sweep results (deterministic; no auto-export).
        self.grid_sweep_results: list[dict[str, Any]] = []

    def load_network(self, path):
        """
        Load action for Research Simulation.

        Deterministic import bridge:
        - Parses CSV/XLSX into node table, edge table, and metadata (read-only).
        - Constructs an immutable Phase1NetworkAdapter with S=1.0 per edge and forces=0.0.
        - Seeds RNG ONCE per loaded network (Option A — locked).
        """
        p = "" if path is None else str(path).strip()
        if p == "":
            self.state.loaded_network = None
            return True

        if not os.path.exists(p):
            raise FileNotFoundError(f"Input file not found: {p}")

        ext = os.path.splitext(p)[1].lower()
        if ext == ".csv":
            tables = _parse_delimited_tables_from_csv(p)
        elif ext == ".xlsx":
            tables = _parse_delimited_tables_from_xlsx(p)
        else:
            raise ValueError(f"Unsupported file type for Research Simulation import: {ext}")

        if len(tables) < 2:
            raise ValueError("Input file must contain at least a nodes table and an edges table.")

        nodes_table = tables[0]
        edges_table = tables[1]
        meta_table = tables[2] if len(tables) >= 3 else {}

        # Row-index diagnostics (optional; present for stacked-table XLSX imports).
        nodes_row_idx = list(nodes_table.get("__row_index__", [])) if isinstance(nodes_table, dict) else []
        edges_row_idx = list(edges_table.get("__row_index__", [])) if isinstance(edges_table, dict) else []

        # Nodes: (node_id, x, y, boundary flags)
        n_id_col = _require_column(nodes_table, ["n_id", "node_id", "id"], table_name="nodes table")
        x_col = _require_column(nodes_table, ["n_x", "x"], table_name="nodes table")
        y_col = _require_column(nodes_table, ["n_y", "y"], table_name="nodes table")

        node_coords: dict[Any, tuple[float, float]] = {}
        node_ids: list[Any] = []

        # Explicit boundary flags are REQUIRED experimental input (no geometric heuristics).
        # Require node-level boolean columns: is_left_boundary / is_right_boundary.
        norm_nodes = {_normalize_column_name(k): k for k in nodes_table.keys()}
        has_left = "is_left_boundary" in norm_nodes
        has_right = "is_right_boundary" in norm_nodes

        # is_fixed is deprecated in Research Simulation; rigid grips supersede it.
        # If present in input, it is ignored for mechanics and visualization.
        if (not has_left) and (not has_right):
            raise ValueError(
                "Boundary nodes must be explicitly specified via is_left_boundary / is_right_boundary.\n\n"
                "Add boolean (True/False or 1/0) columns to the nodes table:\n"
                "- is_left_boundary\n"
                "- is_right_boundary\n"
            )
        if (not has_left) or (not has_right):
            missing = []
            if not has_left:
                missing.append("is_left_boundary")
            if not has_right:
                missing.append("is_right_boundary")
            raise ValueError(
                "Boundary nodes must be explicitly specified via is_left_boundary / is_right_boundary.\n\n"
                f"Missing required column(s) in nodes table: {missing}\n"
                "Add boolean (True/False or 1/0) columns to the nodes table."
            )
        left_flag_col = norm_nodes["is_left_boundary"]
        right_flag_col = norm_nodes["is_right_boundary"]

        left_nodes_set: set[int] = set()
        right_nodes_set: set[int] = set()
        seen_node_ids: set[int] = set()
        for i in range(len(nodes_table[n_id_col])):
            raw_ridx = nodes_row_idx[i] if (i < len(nodes_row_idx)) else None
            row_display = raw_ridx + 1 if raw_ridx is not None else i + 1
            nid = _coerce_int(nodes_table[n_id_col][i], sheet="nodes", row=row_display, column=n_id_col)
            x = _coerce_float(nodes_table[x_col][i], sheet="nodes", row=row_display, column=x_col)
            y = _coerce_float(nodes_table[y_col][i], sheet="nodes", row=row_display, column=y_col)
            if (not np.isfinite(float(x))) or (not np.isfinite(float(y))):
                raise ValueError(f"Invalid node coordinate for node_id {nid} at row {row_display}: (x={nodes_table[x_col][i]}, y={nodes_table[y_col][i]}) (must be finite)")
            if int(nid) in seen_node_ids:
                raw_ridx = nodes_row_idx[i] if (i < len(nodes_row_idx)) else None
                raise ValueError(f"Duplicate node_id detected: {nid} (row {raw_ridx})")
            seen_node_ids.add(int(nid))
            node_coords[nid] = (x, y)
            node_ids.append(nid)

            is_left = _coerce_bool_boundary_flag(nodes_table[left_flag_col][i], node_id=nid, column_name="is_left_boundary")
            is_right = _coerce_bool_boundary_flag(nodes_table[right_flag_col][i], node_id=nid, column_name="is_right_boundary")
            if is_left:
                left_nodes_set.add(int(nid))
            if is_right:
                right_nodes_set.add(int(nid))

        if not node_coords:
            raise ValueError("No nodes found in nodes table.")

        # Boundary validation (hard errors; deterministic).
        overlap = left_nodes_set.intersection(right_nodes_set)
        if overlap:
            both = ", ".join(str(int(x)) for x in sorted(overlap))
            raise ValueError(f"Invalid boundary specification: node(s) marked both left and right: [{both}]")
        if not left_nodes_set:
            raise ValueError("Invalid boundary specification: zero left boundary nodes (is_left_boundary == True).")
        if not right_nodes_set:
            raise ValueError("Invalid boundary specification: zero right boundary nodes (is_right_boundary == True).")

        # Load-time boundary consistency validation (Option A; no heuristic reassignment):
        # If a node lies within tolerance of the left/right x-extreme but is not flagged as that boundary,
        # fail loudly and list node_ids so the experimental input can be corrected.
        xs = [float(xy[0]) for xy in node_coords.values()]
        x_min = float(min(xs))
        x_max = float(max(xs))
        x_span = float(x_max - x_min)
        tol = 1e-6 * max(1.0, abs(x_span))
        near_left = sorted([int(nid) for nid, (x, _y) in node_coords.items() if abs(float(x) - x_min) <= tol])
        near_right = sorted([int(nid) for nid, (x, _y) in node_coords.items() if abs(float(x) - x_max) <= tol])
        missing_left_flags = sorted([nid for nid in near_left if nid not in left_nodes_set])
        missing_right_flags = sorted([nid for nid in near_right if nid not in right_nodes_set])
        if missing_left_flags or missing_right_flags:
            left_details = [(int(nid), float(node_coords[int(nid)][0])) for nid in missing_left_flags[:20] if int(nid) in node_coords]
            right_details = [(int(nid), float(node_coords[int(nid)][0])) for nid in missing_right_flags[:20] if int(nid) in node_coords]
            raise ValueError(
                "Boundary flag validation failed (no heuristic inference is performed).\n\n"
                f"File: {p}\n"
                f"x_min={x_min:.6g}, x_max={x_max:.6g}, tol={tol:.6g}\n"
                f"Nodes within tol of x_min but not flagged is_left_boundary (node_id,x): {left_details}\n"
                f"Nodes within tol of x_max but not flagged is_right_boundary (node_id,x): {right_details}\n\n"
                "Fix: set the appropriate boundary flag(s) for these node_id values:\n"
                "- is_left_boundary for near-x_min nodes\n"
                "- is_right_boundary for near-x_max nodes"
            )
        left_nodes = sorted(left_nodes_set)
        right_nodes = sorted(right_nodes_set)

        # Edges: (edge_id, n_from, n_to)
        e_id_col = _require_column(edges_table, ["e_id", "edge_id", "id"], table_name="edges table")
        n_from_col = _require_column(edges_table, ["n_from", "from", "source"], table_name="edges table")
        n_to_col = _require_column(edges_table, ["n_to", "to", "target"], table_name="edges table")

        # Thickness is required experimental data (Stage 1 thickness-aware modeling).
        # Must be a column named exactly "thickness" (no defaults, no inference).
        if "thickness" not in edges_table:
            cols = ", ".join(str(k) for k in edges_table.keys())
            raise ValueError(f"Missing required column 'thickness' in edges table. Found columns: [{cols}]")
        thickness_col = "thickness"

        # Rest lengths (Research Simulation correctness):
        # Always compute geometric rest length from imported node coordinates at load time.
        # Do NOT consume any rest_length column from input files; any such column is ignored.

        # Metadata spring constant (if available)
        k0 = None
        plasmin_mode = "saturating"
        n_plasmin = 1
        
        # v5.0 spatial plasmin parameters (optional; only used if USE_SPATIAL_PLASMIN=True)
        spatial_plasmin_params = {
            "L_seg": None,
            "N_pf": None,
            "sigma_site": None,
            "P_bulk": None,
            "k_on0": None,
            "k_off0": None,
            "alpha": None,
            "k_cat0": None,
            "beta": None,
            "epsilon": 0.1,  # Default cracked threshold
            "K_crit": None,
            # Unit conversion factors (Phase 1.5)
            "coord_to_m": 1.0,      # Default: coordinates already in meters
            "thickness_to_m": 1.0,  # Default: thickness already in meters
            "N_seg_max": 10000,     # Safety guard against segment explosion
            # Phase 2G: Stochastic seeding parameters
            "P_total_quanta": 100,  # Default: 100 plasmin quanta total
            "lambda_bind_total": 10.0,  # Default: 10 binding events/second rate
            # Phase 2D: Fracture threshold
            "n_crit_fraction": 0.1,  # Edge cleaves when min(n_i/N_pf) <= 0.1
        }
        
        if meta_table:
            mk = _require_column(meta_table, ["meta_key", "key"], table_name="meta_data table")
            mv = _require_column(meta_table, ["meta_value", "value"], table_name="meta_data table")
            for k, v in zip(meta_table[mk], meta_table[mv]):
                nk = _normalize_column_name(k)
                if nk == "spring_stiffness_constant":
                    k0 = _coerce_float(v)
                elif nk == "plasmin_mode":
                    plasmin_mode = str(v).strip().lower()
                elif nk in ("n_plasmin", "N_plasmin", "plasmin_n"):
                    try:
                        n_plasmin = int(float(v))
                    except Exception:
                        raise ValueError(f"Invalid N_plasmin in meta_data: {v} (must be an integer > 0)")
                # v5.0 spatial plasmin parameters
                elif nk == "l_seg":
                    spatial_plasmin_params["L_seg"] = _coerce_float(v)
                elif nk == "n_pf":
                    spatial_plasmin_params["N_pf"] = int(float(v))
                elif nk == "sigma_site":
                    spatial_plasmin_params["sigma_site"] = _coerce_float(v)
                elif nk == "p_bulk":
                    spatial_plasmin_params["P_bulk"] = _coerce_float(v)
                elif nk == "k_on0":
                    spatial_plasmin_params["k_on0"] = _coerce_float(v)
                elif nk == "k_off0":
                    spatial_plasmin_params["k_off0"] = _coerce_float(v)
                elif nk == "alpha":
                    spatial_plasmin_params["alpha"] = _coerce_float(v)
                elif nk == "k_cat0":
                    spatial_plasmin_params["k_cat0"] = _coerce_float(v)
                elif nk == "beta":
                    spatial_plasmin_params["beta"] = _coerce_float(v)
                elif nk == "epsilon":
                    spatial_plasmin_params["epsilon"] = _coerce_float(v)
                elif nk == "k_crit":
                    # Accept both k_crit and K_crit (normalize to K_crit internally)
                    if spatial_plasmin_params["K_crit"] is not None:
                        # Both k_crit and K_crit present; check consistency
                        existing = float(spatial_plasmin_params["K_crit"])
                        new_val = _coerce_float(v)
                        if abs(existing - new_val) > 1e-12:
                            raise ValueError(f"Conflicting K_crit values in meta_data: K_crit={existing}, k_crit={new_val}")
                    spatial_plasmin_params["K_crit"] = _coerce_float(v)
                # Phase 1.5: Unit conversion factors
                elif nk == "coord_to_m":
                    spatial_plasmin_params["coord_to_m"] = _coerce_float(v)
                elif nk == "thickness_to_m":
                    spatial_plasmin_params["thickness_to_m"] = _coerce_float(v)
                elif nk == "n_seg_max":
                    spatial_plasmin_params["N_seg_max"] = int(float(v))
                # Phase 2G: Stochastic seeding parameters
                elif nk == "p_total_quanta":
                    spatial_plasmin_params["P_total_quanta"] = int(float(v))
                elif nk == "lambda_bind_total":
                    spatial_plasmin_params["lambda_bind_total"] = _coerce_float(v)
                # Phase 2D: Fracture threshold
                elif nk == "n_crit_fraction":
                    spatial_plasmin_params["n_crit_fraction"] = _coerce_float(v)
        
        if k0 is None:
            raise ValueError("Missing required metadata spring_stiffness_constant (k0).")

        edges: list[Phase1EdgeSnapshot] = []
        seen_edge_ids: set[int] = set()
        bad_endpoint_edges: list[tuple[int, int, int, Any]] = []
        for i in range(len(edges_table[e_id_col])):
            raw_ridx = edges_row_idx[i] if (i < len(edges_row_idx)) else None

            # Deterministic guard: skip duplicated header rows accidentally included in the slice.
            try:
                eid_norm = _normalize_column_name(edges_table[e_id_col][i])
                nf_norm = _normalize_column_name(edges_table[n_from_col][i])
                nt_norm = _normalize_column_name(edges_table[n_to_col][i])
                if eid_norm in ("e_id", "edge_id", "id") and nf_norm in ("n_from", "from", "source") and nt_norm in ("n_to", "to", "target"):
                    continue
            except Exception:
                pass
            row_display = raw_ridx + 1 if raw_ridx is not None else i + 1
            try:
                eid = _coerce_int(edges_table[e_id_col][i], sheet="edges", row=row_display, column=e_id_col)
                n_from = _coerce_int(edges_table[n_from_col][i], sheet="edges", row=row_display, column=n_from_col)
                n_to = _coerce_int(edges_table[n_to_col][i], sheet="edges", row=row_display, column=n_to_col)
            except Exception as e:
                raise ValueError(f"Invalid edge row at row {row_display}: could not parse (edge_id, n_from, n_to)") from e
            if int(eid) in seen_edge_ids:
                raise ValueError(f"Duplicate edge_id detected: {eid} (row {raw_ridx})")
            seen_edge_ids.add(int(eid))
            if n_from not in node_coords or n_to not in node_coords:
                bad_endpoint_edges.append((int(eid), int(n_from), int(n_to), raw_ridx))
                continue

            rest_length = _euclidean(node_coords[n_from], node_coords[n_to])
            if (not np.isfinite(float(rest_length))) or float(rest_length) <= 0.0:
                raise ValueError(
                    f"Invalid computed rest length for edge {eid} at row {raw_ridx}: {rest_length} "
                    "(must be finite and > 0). Check node coordinates."
                )

            thickness = _coerce_float(edges_table[thickness_col][i], sheet="edges", row=row_display, column=thickness_col)
            if (not np.isfinite(thickness)) or (float(thickness) <= 0.0):
                raise ValueError(f"Invalid thickness for edge {eid} at row {row_display}: {edges_table[thickness_col][i]} (must be finite and > 0)")

            edges.append(
                Phase1EdgeSnapshot(
                    edge_id=eid,
                    n_from=n_from,
                    n_to=n_to,
                    k0=float(k0),
                    original_rest_length=float(rest_length),
                    L_rest_effective=float(rest_length),
                    M=0.0,
                    S=1.0,  # Will be updated for spatial mode below
                    thickness=float(thickness),
                    lysis_batch_index=None,
                    lysis_time=None,
                    segments=None,  # Will be initialized for spatial mode below
                )
            )

        # Hard validation: edges must reference existing nodes (prevents phantom endpoints and pole artifacts).
        if bad_endpoint_edges:
            missing_from = sum(1 for (_eid, n_from, _n_to, _r) in bad_endpoint_edges if n_from not in node_coords)
            missing_to = sum(1 for (_eid, _n_from, n_to, _r) in bad_endpoint_edges if n_to not in node_coords)
            examples = bad_endpoint_edges[:20]
            ex_str = ", ".join([f"(edge_id={eid}, n_from={nf}, n_to={nt}, row={r})" for (eid, nf, nt, r) in examples])
            raise ValueError(
                "Edge table references unknown node_id(s). This usually means the XLSX stacked-table parser ingested non-edge rows.\n\n"
                f"File: {p}\n"
                f"Unknown endpoints: missing_from={missing_from}, missing_to={missing_to}, total_bad_edges={len(bad_endpoint_edges)}\n"
                f"First examples (max 20): {ex_str}\n"
            )

        if not edges:
            raise ValueError("No edges found in edges table.")
        
        # v5.0 Spatial Plasmin Segment Initialization
        # ============================================
        # If USE_SPATIAL_PLASMIN is True, initialize segment-level state for each fiber.
        # Segments track localized binding (B_i) and protofibril damage (n_i).
        if FeatureFlags.USE_SPATIAL_PLASMIN:
            # Validate required parameters
            required_params = ["L_seg", "N_pf", "sigma_site"]
            missing = [p for p in required_params if spatial_plasmin_params[p] is None]
            if missing:
                raise ValueError(
                    f"USE_SPATIAL_PLASMIN is True but required spatial_plasmin_params are missing from meta_data: {missing}\n"
                    f"Add these parameters to the input file meta_data table."
                )
            
            L_seg = float(spatial_plasmin_params["L_seg"])
            N_pf = int(spatial_plasmin_params["N_pf"])
            sigma_site = float(spatial_plasmin_params["sigma_site"])
            coord_to_m = float(spatial_plasmin_params["coord_to_m"])
            thickness_to_m = float(spatial_plasmin_params["thickness_to_m"])
            N_seg_max = int(spatial_plasmin_params["N_seg_max"])
            
            if L_seg <= 0:
                raise ValueError(f"Invalid L_seg: {L_seg} (must be > 0)")
            if N_pf <= 0:
                raise ValueError(f"Invalid N_pf: {N_pf} (must be > 0)")
            if sigma_site <= 0:
                raise ValueError(f"Invalid sigma_site: {sigma_site} (must be > 0)")
            if coord_to_m <= 0:
                raise ValueError(f"Invalid coord_to_m: {coord_to_m} (must be > 0)")
            if thickness_to_m <= 0:
                raise ValueError(f"Invalid thickness_to_m: {thickness_to_m} (must be > 0)")
            if N_seg_max <= 0:
                raise ValueError(f"Invalid N_seg_max: {N_seg_max} (must be > 0)")
            
            # Initialize segments for each edge
            edges_with_segments = []
            for edge in edges:
                # Fiber length L (from geometry, converted to meters)
                # original_rest_length is in coordinate units; convert to meters
                L_coord = float(edge.original_rest_length)
                L = L_coord * coord_to_m  # [m]
                
                # Number of segments
                N_seg = int(math.ceil(L / L_seg))
                if N_seg <= 0:
                    N_seg = 1  # At least one segment
                
                # Phase 1.5: Hard safety guard against segment explosion
                if N_seg > N_seg_max:
                    raise ValueError(
                        f"Segment explosion detected for edge {edge.edge_id}:\n"
                        f"  L (meters) = {L:.6e}\n"
                        f"  L_seg = {L_seg:.6e}\n"
                        f"  N_seg = {N_seg}\n"
                        f"  N_seg_max = {N_seg_max}\n"
                        f"  coord_to_m = {coord_to_m}\n"
                        f"Suggestion: Check unit conversion factors (coord_to_m, thickness_to_m) or increase L_seg."
                    )
                
                # Fiber diameter D (from thickness column, converted to meters)
                D_raw = float(edge.thickness)
                D = D_raw * thickness_to_m  # [m]
                
                # Initialize segments
                segments = []
                for seg_idx in range(N_seg):
                    # Phase 1.5: Compute actual segment length (last segment may be shorter)
                    start = seg_idx * L_seg
                    L_i = min(L_seg, max(L - start, 0.0))
                    
                    if L_i <= 0:
                        # Do not create zero-length segments
                        continue
                    
                    # Surface area of segment (using actual length L_i)
                    A_surf = math.pi * D * L_i  # [m^2]
                    
                    # Max binding sites
                    S_i = A_surf / sigma_site  # [count]
                    S_i = max(1.0, S_i)  # At least 1 binding site
                    
                    # Initial state
                    seg = FiberSegment(
                        segment_index=int(seg_idx),
                        n_i=float(N_pf),  # Fully intact
                        B_i=0.0,          # No plasmin bound
                        S_i=float(S_i)
                    )
                    segments.append(seg)
                
                # Compute derived S (compatibility proxy): min(n_i / N_pf)
                # At initialization, all n_i = N_pf, so S = 1.0
                S_derived = 1.0
                
                # Create new edge with segments
                from dataclasses import replace
                edge_with_segments = replace(
                    edge,
                    segments=tuple(segments),
                    S=float(S_derived)  # Derived proxy
                )
                edges_with_segments.append(edge_with_segments)
            
            # Replace edges list with segmented version
            edges = edges_with_segments

        # Boundary attachment (load-time only; explicit input-driven membership):
        # - Membership is defined ONLY by node flags is_left_boundary / is_right_boundary.
        # - Membership NEVER changes after load.

        def _relax_impl_not_configured(_edges_snapshots, k_eff_intact, _strain):
            # Phase 1B import does not change/implement solver behavior.
            # A proper solver wrapper must be injected here without leaking it past the adapter boundary.
            raise NotImplementedError("Phase 1 relaxation solver hook is not configured for Research Simulation.")

        adapter = Phase1NetworkAdapter(
            path=p,
            node_coords=node_coords,
            left_boundary_node_ids=left_nodes,
            right_boundary_node_ids=right_nodes,
            relax_impl=None,
        )
        # Rigid clamp boundary condition: capture boundary node y-coordinates ONCE at load time.
        # These are fixed for the entire run (no vertical sliding along grips).
        initial_boundary_y: dict[int, float] = {}
        for nid in list(left_nodes) + list(right_nodes):
            if int(nid) not in adapter._initial_node_coords:
                raise ValueError(f"Boundary node_id {nid} missing from imported node coordinates.")
            y0 = float(adapter._initial_node_coords[int(nid)][1])
            if not np.isfinite(y0):
                raise ValueError(f"Invalid boundary node y-coordinate for node_id {nid}: {y0} (must be finite).")
            initial_boundary_y[int(nid)] = float(y0)
        adapter.initial_boundary_y = dict(initial_boundary_y)
        adapter.set_edges(edges)
        # Stage 5 defaults (frozen at Start): stored at load for traceability.
        adapter.plasmin_mode = str(plasmin_mode)
        adapter.N_plasmin = int(n_plasmin)
        
        # v5.0 spatial plasmin parameters (frozen at Start if spatial mode is active)
        adapter.spatial_plasmin_params = dict(spatial_plasmin_params)

        # Forces initialized to zero for all edges (observable only).
        adapter._forces_by_edge_id = {e.edge_id: 0.0 for e in adapter.edges}
        # Phase 2.2: previous-batch mechanical state is initialized at load (required).
        adapter.prev_mean_tension = None
        # Phase 3.1: reset experiment log on new network load (required).
        adapter.experiment_log = []
        # Phase 3.5: reset frozen params + provenance on new load only (required).
        adapter.frozen_params = None
        adapter.provenance_hash = None
        adapter.applied_strain = None
        # Phase 3.6: reset frozen RNG on new load only (required).
        adapter.frozen_rng_state = None
        adapter.frozen_rng_state_hash = None
        # Terminal-state reset on new load only.
        adapter.termination_reason = None
        adapter.termination_batch_index = None
        adapter.termination_time = None
        adapter.termination_cleavage_fraction = None
        # Phase 2D: reset fractured edge history on new load only.
        adapter.fractured_history = []

        # Phase 1C: configure relaxation to use the existing solver (no solver modifications).
        adapter.configure_existing_solver_relaxation()
        # Validate baseline mechanics at the *current* strain: relax once deterministically.
        adapter.relax(float(self.state.strain_value))

        # One-line deterministic load summary (for debugging/audit).
        left_grip_x0 = float(_median([float(adapter._initial_node_coords[nid][0]) for nid in left_nodes]))
        right_grip_x0 = float(_median([float(adapter._initial_node_coords[nid][0]) for nid in right_nodes]))
        print(
            f"Loaded nodes={len(node_coords)}, edges={len(adapter.edges)}, left_boundary={len(left_nodes)}, right_boundary={len(right_nodes)}; "
            f"grips=({left_grip_x0:.6g},{right_grip_x0:.6g})"
        )

        self.state.loaded_network = adapter

        # RNG reseeds ONLY when a new network is loaded.
        self.rng = random.Random(0)
        # Phase 3.6: adapter references controller RNG; it will be frozen at Start.
        adapter.rng = self.rng

        # Reset experiment observables on load (no physics).
        self.state.time = 0.0
        self.last_metrics = None
        return True

    def start(self):
        """
        Valid transitions:
        - stopped -> running (paused False)
        - running+paused -> running (paused False)
        - running+not paused -> no-op
        """
        if self.state.is_running and not self.state.is_paused:
            return False
        self.state.is_running = True
        self.state.is_paused = False
        return True

    def configure_phase1_parameters_from_ui(self, plasmin_concentration_str: str, time_step_str: str, max_time_str: str, applied_strain_str: str):
        """
        Phase 1 parameter wiring (deterministic):
        - lambda_0 := float(plasmin concentration)
        - dt := float(time step)
        - delta := fixed constant 0.05 (Phase 1 design)
        - g_force := lambda F: 1.0  (force-independent in Phase 1)

        Safety:
        - Requires a loaded Phase1NetworkAdapter
        - Validates lambda_0 > 0 and dt > 0
        - Does NOT silently default invalid values

        Notes:
        - max_time is read for traceability but unused in Phase 1 execution.
        """
        adapter = self.state.loaded_network
        if not isinstance(adapter, Phase1NetworkAdapter):
            raise ValueError("No network loaded for Research Simulation. Load a network before configuring Phase 1 parameters.")

        try:
            lambda_0 = float(str(plasmin_concentration_str).strip())
        except Exception:
            raise ValueError("Invalid plasmin concentration. Enter a positive number.")
        try:
            dt = float(str(time_step_str).strip())
        except Exception:
            raise ValueError("Invalid time step. Enter a positive number.")

        # Applied strain (fixed): read once at Start, must be finite and >= 0.
        try:
            applied_strain = float(str(applied_strain_str).strip())
        except Exception:
            raise ValueError("Invalid applied strain. Enter a finite number >= 0.")
        if not np.isfinite(applied_strain):
            raise ValueError("Applied strain must be finite.")
        if applied_strain < 0.0:
            raise ValueError("Applied strain must be >= 0.")

        if not (lambda_0 > 0.0):
            raise ValueError("Plasmin concentration must be > 0 (mapped to lambda_0).")
        if not (dt > 0.0):
            raise ValueError("Time step must be > 0 (mapped to Δt).")

        # Read (unused in Phase 1) — do not enforce yet.
        _ = str(max_time_str).strip()

        # Phase 3.5: parameter freeze hard rule.
        # If already frozen, disallow mutation; allow Start only if values match frozen_params.
        if isinstance(adapter, Phase1NetworkAdapter) and adapter.frozen_params is not None:
            frozen = adapter.frozen_params
            if float(frozen.get("lambda_0")) != float(lambda_0) or float(frozen.get("dt")) != float(dt):
                raise ValueError("Parameters are frozen for this experiment. Load a new network to change parameters.")
            if "applied_strain" in frozen and float(frozen.get("applied_strain")) != float(applied_strain):
                raise ValueError("Applied strain is frozen for this experiment. Load a new network to change strain.")
            # Boundary definitions are explicit experimental input and are frozen at Start.
            if "left_boundary_node_ids" in frozen or "right_boundary_node_ids" in frozen:
                left_now = [int(x) for x in sorted(adapter.left_boundary_node_ids)]
                right_now = [int(x) for x in sorted(adapter.right_boundary_node_ids)]
                if list(frozen.get("left_boundary_node_ids", [])) != left_now or list(frozen.get("right_boundary_node_ids", [])) != right_now:
                    raise ValueError("Boundary node definitions are frozen for this experiment. Load a new network to change boundaries.")
            # Rigid grip positions are frozen experimental inputs.
            if "left_grip_x" in frozen and "right_grip_x" in frozen:
                if adapter.left_grip_x is None or adapter.right_grip_x is None:
                    raise ValueError("Experiment is missing frozen grip positions on adapter. Load a new network and press Start.")
                if float(frozen.get("left_grip_x")) != float(adapter.left_grip_x) or float(frozen.get("right_grip_x")) != float(adapter.right_grip_x):
                    raise ValueError("Grip positions are frozen for this experiment. Load a new network to change grips.")
            # Stage 2 thickness-aware mechanics: thickness_ref and alpha are frozen (must exist post-Stage2).
            if "thickness_ref" not in frozen or "thickness_alpha" not in frozen:
                raise ValueError("Checkpoint/experiment is missing thickness_ref/thickness_alpha. Re-load and Start with Stage 2 enabled.")
            # Stage 3: beta/gamma are frozen (must exist post-Stage3).
            if "beta" not in frozen or "gamma" not in frozen:
                raise ValueError("Checkpoint/experiment is missing beta/gamma. Re-load and Start with Stage 3 enabled.")
            # Enforce that the deterministic thickness_ref derived from data matches the frozen one.
            t_ref_now = _median([float(e.thickness) for e in adapter.edges])
            if (not np.isfinite(t_ref_now)) or (t_ref_now <= 0.0):
                raise ValueError("Invalid thickness_ref computed from thickness data (must be finite and > 0).")
            if float(frozen.get("thickness_ref")) != float(t_ref_now):
                raise ValueError("Thickness reference is frozen for this experiment. Load a new network to change thickness data.")
            if float(frozen.get("thickness_alpha")) != 1.0:
                raise ValueError("Thickness alpha is frozen for this experiment. Load a new network to change.")
            if float(frozen.get("beta")) != 1.0 or float(frozen.get("gamma")) != 1.0:
                raise ValueError("Degradation beta/gamma are frozen for this experiment. Load a new network to change.")
            if "global_lysis_threshold" not in frozen:
                raise ValueError("Checkpoint/experiment is missing global_lysis_threshold. Re-load and Start with Stage 4 enabled.")
            if float(frozen.get("global_lysis_threshold")) != 0.9:
                raise ValueError("Global lysis threshold is frozen for this experiment. Load a new network to change.")
            # Stage 5: plasmin mode selection is frozen.
            if "plasmin_mode" not in frozen or "N_plasmin" not in frozen:
                raise ValueError("Checkpoint/experiment is missing plasmin_mode/N_plasmin. Re-load and Start with Stage 5 enabled.")
            return True

        adapter.lambda_0 = lambda_0
        adapter.dt = dt
        adapter.applied_strain = float(applied_strain)
        # Strain is fixed after Start; controller state uses this value.
        self.state.strain_value = float(applied_strain)
        # Phase 1: fixed by design (deterministic, documented constant).
        adapter.delta = 0.05
        # Phase 2.1: nonlinear, bounded mechanochemical coupling (deterministic; fixed constants, no UI).
        adapter.force_alpha = 1.0
        adapter.force_F0 = 1.0
        adapter.force_hill_n = 2.0
        adapter.g_force = adapter.phase2_1_g_force
        # Phase 2.2: strain-rate–aware multiplier constants (fixed, deterministic; no UI).
        # Setting rate_beta = 0.0 recovers Phase 2.1 behavior.
        adapter.rate_beta = 0.5
        adapter.rate_eps0 = 1.0
        # Phase 2.3: plastic rest-length remodeling constants (fixed, deterministic; no UI).
        # Setting plastic_rate = 0.0 recovers Phase 2.2 behavior exactly.
        adapter.plastic_F_threshold = 1.0
        adapter.plastic_rate = 0.01
        # Phase 2.4: force-driven rupture amplification constants (fixed, deterministic; no UI).
        # Setting rupture_gamma = 0.0 recovers Phase 2.3 behavior exactly.
        adapter.rupture_force_threshold = 3.0
        adapter.rupture_gamma = 0.5
        # Phase 2.5: energy-based failure gate constants (fixed, deterministic; no UI).
        # Setting fracture_eta = 0.0 recovers Phase 2.4 behavior exactly.
        adapter.fracture_Gc = 0.5
        adapter.fracture_eta = 0.3
        # Phase 2.6: topology-aware cooperativity gate constant (fixed, deterministic; no UI).
        # Setting coop_chi = 0.0 recovers Phase 2.5 behavior exactly.
        adapter.coop_chi = 0.5
        # Phase 2.7: stress-shielding / load redistribution saturation epsilon (fixed; no UI).
        adapter.shield_eps = 1e-6
        # Phase 2.8: temporal damage memory constants (fixed, deterministic; no UI).
        # Setting memory_rho = 0.0 recovers Phase 2.7 behavior exactly.
        adapter.memory_mu = 0.2
        adapter.memory_rho = 0.1
        # Phase 2.9: directional anisotropy gate constant (fixed, deterministic; no UI).
        # Setting aniso_kappa = 0.0 recovers Phase 2.8 behavior exactly.
        adapter.aniso_kappa = 0.5
        # Phase 3.0: execution guardrails (fixed, deterministic; no UI).
        adapter.g_max = 50.0
        adapter.cleavage_batch_cap = 0.2

        # Thickness reproducibility (Stage 1): compute deterministic per-edge thickness hash
        # over sorted (edge_id, thickness) pairs. This is immutable experimental data.
        thickness_pairs = [(int(e.edge_id), float(e.thickness)) for e in adapter.edges]
        thickness_pairs.sort(key=lambda t: t[0])
        thickness_hash = hashlib.sha256(json.dumps(thickness_pairs, sort_keys=True).encode("utf-8")).hexdigest()

        # Stage 2 thickness-aware mechanics: compute thickness_ref once at Start (deterministic median).
        thickness_ref = float(_median([float(e.thickness) for e in adapter.edges]))
        if (not np.isfinite(thickness_ref)) or (thickness_ref <= 0.0):
            raise ValueError("Invalid thickness_ref (median thickness) computed from input. Must be finite and > 0.")
        thickness_alpha = 1.0  # hardcoded for Stage 2 (no UI exposure)
        adapter.thickness_ref = float(thickness_ref)
        adapter.thickness_alpha = float(thickness_alpha)

        # Stage 3: thickness + tension dependent degradation (fixed constants; no UI).
        adapter.degradation_beta = 1.0
        adapter.degradation_gamma = 1.0

        # Stage 4: observational lysis tracking (fixed threshold; no UI).
        adapter.global_lysis_threshold = 0.9

        # Stage 5: plasmin-limited exposure (frozen; no UI). Defaults preserve legacy behavior.
        mode = str(getattr(adapter, "plasmin_mode", "saturating")).strip().lower()
        if mode not in ("saturating", "limited"):
            raise ValueError("Invalid plasmin_mode (must be 'saturating' or 'limited').")
        try:
            n_plasmin = int(getattr(adapter, "N_plasmin", 1))
        except Exception:
            raise ValueError("Invalid N_plasmin (must be an integer > 0).")
        if n_plasmin <= 0:
            raise ValueError("Invalid N_plasmin (must be an integer > 0).")
        adapter.plasmin_mode = mode
        adapter.N_plasmin = int(n_plasmin)

        # Rigid grips: compute grip x-positions ONCE at Start from original boundary-node x positions
        # and the fixed applied strain. Boundary nodes are x-clamped to these values for the entire run.
        left_ids = [int(x) for x in sorted(adapter.left_boundary_node_ids)]
        right_ids = [int(x) for x in sorted(adapter.right_boundary_node_ids)]
        if not left_ids or not right_ids:
            raise ValueError("Boundary nodes must be explicitly specified via is_left_boundary / is_right_boundary.")
        left_xs0 = [float(adapter._initial_node_coords[nid][0]) for nid in left_ids]
        right_xs0 = [float(adapter._initial_node_coords[nid][0]) for nid in right_ids]
        left_grip_x0 = float(_median(left_xs0))
        right_grip_x0 = float(_median(right_xs0))
        if (not np.isfinite(left_grip_x0)) or (not np.isfinite(right_grip_x0)):
            raise ValueError("Invalid boundary-node x positions (must be finite) for rigid grips.")
        base_width = float(right_grip_x0 - left_grip_x0)
        if not np.isfinite(base_width) or base_width <= 0.0:
            raise ValueError("Invalid rigid-grip baseline width (must be > 0). Check boundary flags and node coordinates.")
        adapter.left_grip_x = float(left_grip_x0)
        adapter.right_grip_x = float(right_grip_x0) + float(applied_strain) * float(base_width)
        if (not np.isfinite(adapter.left_grip_x)) or (not np.isfinite(adapter.right_grip_x)):
            raise ValueError("Non-finite rigid grip x positions computed at Start.")
        # Startup self-check (poster-safe): boundary/grip invariants.
        if not left_ids or not right_ids:
            raise ValueError("Invalid boundary sets at Start: left/right boundary node lists must be non-empty.")
        if not (float(adapter.left_grip_x) < float(adapter.right_grip_x)):
            raise ValueError("Invalid rigid grips at Start: left_grip_x must be < right_grip_x.")

        # Phase 3.5: freeze parameters + compute deterministic provenance hash at Start.
        # Deep-copy numeric parameters/constants into frozen_params and compute SHA256 over
        # JSON with sorted keys for deterministic reproducibility.
        left_boundary_node_ids = [int(x) for x in sorted(adapter.left_boundary_node_ids)]
        right_boundary_node_ids = [int(x) for x in sorted(adapter.right_boundary_node_ids)]
        if not left_boundary_node_ids or not right_boundary_node_ids:
            raise ValueError("Boundary nodes must be explicitly specified via is_left_boundary / is_right_boundary.")
        if set(left_boundary_node_ids).intersection(set(right_boundary_node_ids)):
            raise ValueError("Invalid boundary specification: a node is marked both left and right.")

        # Boundary y constraints: freeze imported boundary node y-coordinates deterministically.
        if not isinstance(getattr(adapter, "initial_boundary_y", None), dict):
            raise ValueError("Missing initial_boundary_y on adapter. Load a network before Start.")
        initial_by = dict(getattr(adapter, "initial_boundary_y"))
        if not initial_by:
            raise ValueError("Missing initial_boundary_y mapping (empty). Load a network before Start.")
        for nid in left_boundary_node_ids + right_boundary_node_ids:
            if int(nid) not in initial_by:
                raise ValueError(f"Missing initial boundary y-coordinate for node_id {nid}.")
            if not np.isfinite(float(initial_by[int(nid)])):
                raise ValueError(f"Invalid initial boundary y-coordinate for node_id {nid}: {initial_by[int(nid)]}")
        initial_boundary_y_pairs = [(int(nid), float(initial_by[int(nid)])) for nid in sorted(initial_by.keys())]
        frozen_params = {
            "lambda_0": float(adapter.lambda_0),
            "dt": float(adapter.dt),
            "applied_strain": float(adapter.applied_strain),
            "left_boundary_node_ids": list(left_boundary_node_ids),
            "right_boundary_node_ids": list(right_boundary_node_ids),
            "left_grip_x": float(adapter.left_grip_x),
            "right_grip_x": float(adapter.right_grip_x),
            "initial_boundary_y": list(initial_boundary_y_pairs),
            "thickness_hash": str(thickness_hash),
            "thickness_ref": float(adapter.thickness_ref),
            "thickness_alpha": float(adapter.thickness_alpha),
            "beta": float(adapter.degradation_beta),
            "gamma": float(adapter.degradation_gamma),
            "global_lysis_threshold": float(adapter.global_lysis_threshold),
            "plasmin_mode": str(adapter.plasmin_mode),
            "N_plasmin": int(adapter.N_plasmin),
            "delta": float(adapter.delta),
            "force_alpha": float(getattr(adapter, "force_alpha")),
            "force_F0": float(getattr(adapter, "force_F0")),
            "force_hill_n": float(getattr(adapter, "force_hill_n")),
            "rate_beta": float(getattr(adapter, "rate_beta")),
            "rate_eps0": float(getattr(adapter, "rate_eps0")),
            "plastic_F_threshold": float(getattr(adapter, "plastic_F_threshold")),
            "plastic_rate": float(getattr(adapter, "plastic_rate")),
            "rupture_force_threshold": float(getattr(adapter, "rupture_force_threshold")),
            "rupture_gamma": float(getattr(adapter, "rupture_gamma")),
            "fracture_Gc": float(getattr(adapter, "fracture_Gc")),
            "fracture_eta": float(getattr(adapter, "fracture_eta")),
            "coop_chi": float(getattr(adapter, "coop_chi")),
            "shield_eps": float(getattr(adapter, "shield_eps")),
            "memory_mu": float(getattr(adapter, "memory_mu")),
            "memory_rho": float(getattr(adapter, "memory_rho")),
            "aniso_kappa": float(getattr(adapter, "aniso_kappa")),
            "g_max": float(getattr(adapter, "g_max")),
            "cleavage_batch_cap": float(getattr(adapter, "cleavage_batch_cap")),
        }
        
        # Phase 2G: Initialize global plasmin pool (spatial mode only)
        if FeatureFlags.USE_SPATIAL_PLASMIN:
            if adapter.spatial_plasmin_params is None:
                raise ValueError("USE_SPATIAL_PLASMIN is True but spatial_plasmin_params not loaded.")
            
            # P_total_quanta: total plasmin quanta (from meta_data or default)
            # This is the supply limit - binding is sparse and stochastic
            P_total_quanta = int(adapter.spatial_plasmin_params.get("P_total_quanta", 100))
            if P_total_quanta < 0:
                raise ValueError(f"P_total_quanta must be >= 0, got {P_total_quanta}")
            
            adapter.P_total_quanta = P_total_quanta
            adapter.P_free_quanta = P_total_quanta  # All plasmin starts free
            
            # Add to frozen params for reproducibility
            frozen_params["P_total_quanta"] = int(P_total_quanta)
        
        frozen_json = json.dumps(frozen_params, sort_keys=True)
        adapter.frozen_params = copy.deepcopy(frozen_params)
        adapter.provenance_hash = hashlib.sha256(frozen_json.encode("utf-8")).hexdigest()

        # Phase 3.6: freeze RNG state at Start (capture once; do not reseed after this point).
        if adapter.rng is None:
            raise ValueError("RNG not available on adapter; load a network before Start.")
        rng_state = adapter.rng.getstate()
        if adapter.frozen_rng_state is not None:
            # Disallow any drift: frozen state must remain identical once set.
            if str(adapter.frozen_rng_state) != str(rng_state):
                raise ValueError("RNG state drift detected after Start. Load a new network to reset RNG.")
        else:
            adapter.frozen_rng_state = rng_state
            adapter.frozen_rng_state_hash = hashlib.sha256(str(rng_state).encode("utf-8")).hexdigest()
        return True

    def pause(self):
        """
        Valid transitions:
        - running+not paused -> paused
        - running+paused -> unpaused
        Invalid:
        - not running -> no-op
        """
        if not self.state.is_running:
            return False
        self.state.is_paused = not self.state.is_paused
        return True

    def stop(self):
        """
        Valid transitions:
        - any -> stopped (running False, paused False, time reset to 0)
        """
        changed = self.state.is_running or self.state.is_paused or (float(self.state.time) != 0.0)
        self.state.is_running = False
        self.state.is_paused = False
        self.state.time = 0.0
        return changed

    def set_strain(self, value):
        """
        Set normalized strain value (UI-only).
        Deterministic coercion + clamp to [0, 1].
        """
        try:
            v = float(value)
        except Exception:
            v = 0.0
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        changed = abs(float(self.state.strain_value) - v) > 1e-12
        self.state.strain_value = v

        # Phase 1C: static mechanics validation — relax once at the current strain.
        adapter = self.state.loaded_network
        if isinstance(adapter, Phase1NetworkAdapter):
            adapter.relax(float(v))
        return changed

    def advance_one_batch(self):
        # Phase 3.8: per-batch timing (read-only; diagnostics only).
        t0 = time.perf_counter()

        """
        Execute exactly one Phase 1 degradation batch.

        Preconditions (fail loudly via exceptions; UI shows message boxes):
        - Network must be loaded and provide required simulation data hooks
        - Simulation must be running and not paused

        Notes:
        - No looping, scheduling, timers, or threads.
        - Metrics are taken ONLY from the step's returned delta.
        """
        # Terminal-state: if the experiment already terminated (e.g., lost load-bearing capacity),
        # do not raise; just stop cleanly and allow log/snapshot export.
        if isinstance(self.state.loaded_network, Phase1NetworkAdapter) and getattr(self.state.loaded_network, "termination_reason", None) is not None:
            self.state.is_running = False
            self.state.is_paused = False
            return False

        if self.state.loaded_network is None:
            raise ValueError("No network loaded. Load a network before advancing.")
        if not self.state.is_running:
            raise ValueError("Simulation is not running. Press Start before advancing.")
        if self.state.is_paused:
            raise ValueError("Simulation is paused. Unpause before advancing.")

        adapter = self.state.loaded_network
        if not isinstance(adapter, Phase1NetworkAdapter):
            raise ValueError("Loaded network is not a Phase 1 adapter instance.")

        # Validate Phase 1 configuration (no inference or physics assumptions here).
        if len(adapter.edges) == 0:
            raise ValueError(
                "Loaded network has no Phase 1 edge snapshots configured.\n\n"
                "Phase1NetworkAdapter.edges must be populated by a future loader."
            )
        if adapter._relax_impl is None:
            raise ValueError(
                "Phase 1 solver is not configured.\n\n"
                "Phase1NetworkAdapter must wrap the existing linear solver via relax_impl."
            )
        # Phase 1D precondition: static relaxation must already have been performed
        # at the current strain (start-of-batch force field must exist).
        if adapter._relaxed_node_coords is None:
            raise ValueError(
                "Static relaxation has not been performed for the current strain.\n\n"
                "Press Start to apply the fixed strain and trigger relaxation before advancing."
            )
        for e in adapter.edges:
            if float(e.S) > 0.0 and e.edge_id not in adapter._forces_by_edge_id:
                raise ValueError(
                    "Start-of-batch forces are missing for one or more intact edges.\n\n"
                    "Press Start to trigger a deterministic relaxation at the fixed applied strain before advancing."
                )
        if adapter.g_force is None or adapter.lambda_0 is None or adapter.delta is None or adapter.dt is None:
            raise ValueError(
                "Phase 1 parameters are not configured on the adapter.\n\n"
                "Required: lambda_0, delta, dt, g_force."
            )

        # Phase 3.0 guardrails (pre-batch): edge state consistency checks.
        for e in adapter.edges:
            S = float(e.S)
            if not (0.0 <= S <= 1.0):
                raise ValueError(f"Invalid edge state: S out of bounds [0,1] for edge {e.edge_id}.")
            if float(e.L_rest_effective) < float(e.original_rest_length):
                raise ValueError(f"Invalid edge state: L_rest_effective < original_rest_length for edge {e.edge_id}.")
            if float(e.M) < 0.0:
                raise ValueError(f"Invalid edge state: M < 0 for edge {e.edge_id}.")

        # MECHANOCHEMICAL COUPLING FIX: Relax BEFORE chemistry to align forces with current strain.
        # This ensures force-dependent kinetics (k_cat, k_off) use forces that reflect the applied strain,
        # eliminating the one-batch lag that was suppressing strain-dependent lysis.
        adapter.relax(float(self.state.strain_value))

        # Phase 2.0 batch (force-dependent, deterministic; no RNG draws).
        # Batch order (exact):
        # a) Use post-relaxation forces from CURRENT strain (relaxed at start of batch)
        # b) Compute λ_i = λ0 * g_force(F_i) for intact edges
        # c) Update S_i deterministically over Δt
        # d) Remove edges with S_i <= 0 (represented by S=0; solver excludes S<=0)
        # e) Perform exactly one relaxation AFTER degradation (propagates stiffness/damage changes)

        if adapter.g_force is None:
            raise ValueError("Phase 2.0 requires g_force to be configured on the adapter.")

        # a) cached forces must exist for all intact edges
        intact_edges: list[Phase1EdgeSnapshot] = []
        force_list: list[float] = []
        for e in adapter.edges:
            if float(e.S) > 0.0:
                intact_edges.append(e)
                if e.edge_id not in adapter._forces_by_edge_id:
                    raise ValueError("Forces are missing for one or more intact edges. Relax at current strain before advancing.")
                force_list.append(float(adapter._forces_by_edge_id[e.edge_id]))

        # Safety: enforce mapping invariance
        assert len(intact_edges) == len(force_list)

        lambda_0 = float(adapter.lambda_0)
        dt = float(adapter.dt)

        if not (lambda_0 > 0.0) or not (dt > 0.0):
            raise ValueError("Invalid Phase 1/2 parameters: lambda_0 and dt must be > 0.")

        # 2) Compute mean tension and sigma_ref from cached pre-batch forces (tension only; nonnegative).
        # Stage 3: sigma_ref is the deterministic median tension across intact edges (computed once per batch).
        if len(intact_edges) == 0:
            mean_tension = 0.0
            sigma_ref = None
        else:
            tension_forces = [max(0.0, float(f)) for f in force_list]
            mean_tension = float(sum(tension_forces) / len(tension_forces))
            sigma_ref = float(_median(tension_forces))

            # Optional debug logging (enable via environment variable FIBRINET_DEBUG=1)
            import os
            if os.environ.get("FIBRINET_DEBUG") == "1":
                max_force = max(tension_forces) if tension_forces else 0.0
                batch_idx = len(adapter.experiment_log)
                print(f"[DEBUG] Batch {batch_idx}: sigma_ref={sigma_ref:.6e}, max_force={max_force:.6e}, mean_tension={mean_tension:.6e}", flush=True)
            
            # PHYSICS VALIDATION: In spatial mode, sigma_ref must be > 0 if used in division
            if FeatureFlags.USE_SPATIAL_PLASMIN and sigma_ref is not None and sigma_ref <= 0.0:
                # Spatial mode: zero tension is terminal for network (no load-bearing)
                # This is acceptable; treat as percolation failure (will be checked later)
                sigma_ref = None  # Clear for spatial mode to skip stress-factor calculations
            
            if (not np.isfinite(sigma_ref)) or (sigma_ref <= 0.0 and not FeatureFlags.USE_SPATIAL_PLASMIN):
                # Terminal-state handling (deterministic, model-side):
                # The network has lost load-bearing capacity (slack/collapsed), so sigma_ref is undefined.
                # Terminate cleanly: record reason in experiment_log and stop further batches.
                # NOTE: In spatial plasmin mode (v5.0), termination is by percolation only, not sigma_ref.
                reason = "network_lost_load_bearing_capacity"
                batch_index = int(len(adapter.experiment_log))
                # Termination should reflect end-of-batch time, consistent with normal stepping.
                # Note: dt_used not yet computed here, so use base dt for termination time
                time_val = float(self.state.time) + float(dt)
                cleaved_edges_total = sum(1 for e in adapter.edges if float(e.S) <= 0.0)
                total_edges = max(1, len(adapter.edges))
                cleavage_fraction = float(cleaved_edges_total) / float(total_edges)

                adapter.termination_reason = str(reason)
                adapter.termination_batch_index = int(batch_index)
                adapter.termination_time = float(time_val)
                adapter.termination_cleavage_fraction = float(cleavage_fraction)

                # Deterministic batch_hash for termination entry (no state mutation; sigma_ref=None).
                edges_sorted = sorted(adapter.edges, key=lambda ee: int(ee.edge_id))
                payload = {
                    "batch_index": int(batch_index),
                    "time": float(time_val),
                    "strain": float(self.state.strain_value),
                    "sigma_ref": None,
                    "plasmin_selected_edge_ids": [],
                    "termination_reason": str(reason),
                    "termination_batch_index": int(batch_index),
                    "termination_time": float(time_val),
                    "termination_cleavage_fraction": float(cleavage_fraction),
                    "edges": [
                        {
                            "edge_id": int(e.edge_id),
                            "S": float(e.S),
                            "M": float(e.M),
                            "original_rest_length": float(e.original_rest_length),
                            "L_rest_effective": float(e.L_rest_effective),
                            "thickness": float(e.thickness),
                            "lysis_batch_index": (int(e.lysis_batch_index) if e.lysis_batch_index is not None else None),
                            "lysis_time": (float(e.lysis_time) if e.lysis_time is not None else None),
                        }
                        for e in edges_sorted
                    ],
                    "frozen_params": copy.deepcopy(adapter.frozen_params),
                    "provenance_hash": adapter.provenance_hash,
                    "rng_state_hash": adapter.frozen_rng_state_hash,
                }
                batch_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

                intact_edges_now = sum(1 for e in adapter.edges if float(e.S) > 0.0)
                # Protofibril-based stiffness scaling for lysis fraction calculation
                if FeatureFlags.USE_SPATIAL_PLASMIN and adapter.spatial_plasmin_params:
                    N_pf = float(adapter.spatial_plasmin_params.get("N_pf", 50))
                    sum_k0 = float(sum(float(e.k0) * N_pf for e in adapter.edges))
                    sum_keff = float(sum(float(e.k0) * N_pf * float(e.S) for e in adapter.edges if float(e.S) > 0.0))
                else:
                    sum_k0 = float(sum(float(e.k0) for e in adapter.edges))
                    sum_keff = float(sum(float(e.k0) * float(e.S) for e in adapter.edges if float(e.S) > 0.0))
                eps = 1e-12  # numerical guard only (allowed)
                lysis_fraction = float(1.0 - (sum_keff / max(eps, sum_k0)))

                adapter.experiment_log.append(
                    {
                        "batch_index": int(batch_index),
                        "provenance_hash": adapter.provenance_hash,
                        "rng_state_hash": adapter.frozen_rng_state_hash,
                        "batch_hash": batch_hash,
                        "batch_duration_sec": float(time.perf_counter() - t0),
                        "thickness_hash": (adapter.frozen_params or {}).get("thickness_hash") if isinstance(adapter.frozen_params, dict) else None,
                        "plasmin_mode": str(getattr(adapter, "plasmin_mode", "saturating")),
                        "N_plasmin": int(getattr(adapter, "N_plasmin", 1)),
                        "plasmin_selected_edge_ids": [],
                        "newly_lysed_edge_ids": [],
                        "cumulative_lysed_edge_ids": sorted([int(e.edge_id) for e in adapter.edges if e.lysis_batch_index is not None]),
                        "global_lysis_batch_index": adapter.global_lysis_batch_index,
                        "global_lysis_time": adapter.global_lysis_time,
                        "time": float(time_val),
                        "strain": float(self.state.strain_value),
                        "branch_parent_batch_index": adapter.branch_parent_batch_index,
                        "branch_parent_batch_hash": adapter.branch_parent_batch_hash,
                        "sweep_param": adapter.sweep_param,
                        "sweep_value": adapter.sweep_value,
                        "grid_params": copy.deepcopy(adapter.grid_params),
                        "intact_edges": int(intact_edges_now),
                        "cleaved_edges_total": int(cleaved_edges_total),
                        "newly_cleaved": 0,
                        "mean_tension": float(mean_tension),
                        "lysis_fraction": float(lysis_fraction),
                        "dt_used": float(dt),  # Phase 2A/2B: use base dt in termination case
                        "n_min_frac": None,  # Phase 2B: not computed in termination case
                        "n_mean_frac": None,  # Phase 2B: not computed in termination case
                        "total_bound_plasmin": None,  # Phase 2B: not computed in termination case
                        "total_bound_this_batch": None,  # Phase 2G: not computed in termination case
                        "min_stiff_frac": None,  # Phase 2C: not computed in termination case
                        "mean_stiff_frac": None,  # Phase 2C: not computed in termination case
                        # Phase 2G: Stochastic plasmin seeding observables (not computed in termination case)
                        "P_total_quanta": int(adapter.P_total_quanta) if adapter.P_total_quanta is not None else None,
                        "P_free_quanta": int(adapter.P_free_quanta) if adapter.P_free_quanta is not None else None,
                        "bind_events_requested": None,
                        "bind_events_applied": None,
                        "total_unbound_this_batch": None,
                        "termination_reason": str(reason),
                        "termination_batch_index": int(batch_index),
                        "termination_time": float(time_val),
                        "termination_cleavage_fraction": float(cleavage_fraction),
                        "params": {
                            "lambda_0": float(adapter.lambda_0),
                            "dt": float(adapter.dt),
                            "delta": float(adapter.delta),
                            "force_alpha": float(getattr(adapter, "force_alpha")),
                            "force_F0": float(getattr(adapter, "force_F0")),
                            "force_hill_n": float(getattr(adapter, "force_hill_n")),
                            "rate_beta": float(getattr(adapter, "rate_beta")),
                            "plastic_rate": float(getattr(adapter, "plastic_rate")),
                            "rupture_gamma": float(getattr(adapter, "rupture_gamma")),
                            "fracture_Gc": float(getattr(adapter, "fracture_Gc")),
                            "fracture_eta": float(getattr(adapter, "fracture_eta")),
                            "coop_chi": float(getattr(adapter, "coop_chi")),
                            "aniso_kappa": float(getattr(adapter, "aniso_kappa")),
                            "memory_mu": float(getattr(adapter, "memory_mu")),
                            "memory_rho": float(getattr(adapter, "memory_rho")),
                        },
                    }
                )

                # Stop further batches cleanly (no exception).
                self.last_batch_duration_sec = float(adapter.experiment_log[-1]["batch_duration_sec"])
                # Advance controller time to the end-of-batch timestamp for consistency.
                self.state.time = float(time_val)
                self.state.is_running = False
                self.state.is_paused = False
                return False
        assert mean_tension >= 0.0
        assert np.isfinite(mean_tension)

        # Stage 5: plasmin-limited exposure (competition model), selection done once per batch.
        plasmin_mode = str(getattr(adapter, "plasmin_mode", "saturating")).strip().lower()
        if plasmin_mode not in ("saturating", "limited"):
            raise ValueError("Invalid plasmin_mode on adapter (must be 'saturating' or 'limited').")
        try:
            N_plasmin = int(getattr(adapter, "N_plasmin", 1))
        except Exception:
            raise ValueError("Invalid N_plasmin on adapter (must be an integer > 0).")
        if N_plasmin <= 0:
            raise ValueError("Invalid N_plasmin on adapter (must be an integer > 0).")

        # Attack weights are the Stage 3 rate multipliers without lambda_0.
        # attack_weight_i = (sigma/sigma_ref)^beta * (thickness_ref/thickness)^gamma
        attack_weight_by_id: dict[Any, float] = {}
        if len(intact_edges) == 0:
            selected_edge_ids: list[int] = []
            selected_edge_id_set: set[int] = set()
        else:
            # In spatial mode, sigma_ref may be None/0/non-finite (network slack); this is valid.
            # In legacy mode, sigma_ref must be valid for attack weights.
            if not FeatureFlags.USE_SPATIAL_PLASMIN:
                if sigma_ref is None or not np.isfinite(sigma_ref) or sigma_ref <= 0.0:
                    raise ValueError("Internal error: sigma_ref invalid for intact edges in legacy mode.")
            
            if adapter.thickness_ref is None:
                raise ValueError("Missing thickness_ref on adapter. Press Start to freeze parameters before advancing.")
            
            # In spatial mode with sigma_ref = 0/None/invalid (no tension), use uniform attack weights
            if FeatureFlags.USE_SPATIAL_PLASMIN and (sigma_ref is None or sigma_ref <= 0.0 or not np.isfinite(sigma_ref)):
                # Uniform weights (thickness-based only)
                gamma_d = float(getattr(adapter, "degradation_gamma", 1.0))
                for e in intact_edges:
                    w = (float(adapter.thickness_ref) / float(e.thickness)) ** float(gamma_d)
                    if not np.isfinite(w) or w < 0.0:
                        raise ValueError("Invalid attack weight computed (NaN/Inf/negative).")
                    attack_weight_by_id[e.edge_id] = float(w)
            else:
                # Legacy mode: tension-based weights (sigma_ref MUST be valid)
                # NUMERICAL STABILITY: Explicit guard against division by zero
                # FIX A1: Graceful termination if sigma_ref invalid (defense-in-depth)
                if sigma_ref is None or sigma_ref <= 0.0 or not np.isfinite(sigma_ref):
                    # This should NOT happen (early termination at line ~3966 should catch it)
                    # But if it does, terminate gracefully instead of crashing.
                    adapter.termination_reason = "sigma_ref_zero_legacy"
                    adapter.termination_batch_index = int(len(adapter.experiment_log))
                    adapter.termination_time = float(self.state.time)
                    adapter.termination_cleavage_fraction = 0.0
                    # Write minimal log entry (cannot compute full entry here; state incomplete)
                    adapter.experiment_log.append({
                        "batch_index": int(len(adapter.experiment_log)),
                        "time": float(self.state.time),
                        "termination_reason": "sigma_ref_zero_legacy",
                        "sigma_ref": (float(sigma_ref) if sigma_ref is not None else None),
                        "message": "Emergency termination: sigma_ref invalid in legacy attack weight computation",
                    })
                    self.state.is_running = False
                    self.state.is_paused = False
                    return False
                beta = float(getattr(adapter, "degradation_beta", 1.0))
                gamma_d = float(getattr(adapter, "degradation_gamma", 1.0))
                for e in intact_edges:
                    sigma = max(0.0, float(adapter._forces_by_edge_id.get(e.edge_id, 0.0)))
                    w = (float(sigma) / float(sigma_ref)) ** float(beta)
                    w *= (float(adapter.thickness_ref) / float(e.thickness)) ** float(gamma_d)
                    if not np.isfinite(w) or w < 0.0:
                        raise ValueError("Invalid attack weight computed (NaN/Inf/negative).")
                    attack_weight_by_id[e.edge_id] = float(w)

            # Uniform fallback: if N_plasmin >= intact edges, collapse exactly to saturating behavior.
            if plasmin_mode == "saturating" or N_plasmin >= len(intact_edges):
                selected_edge_ids = sorted([int(e.edge_id) for e in intact_edges])
                selected_edge_id_set = set(selected_edge_ids)
            else:
                # Deterministic but stochastic-capable selection WITHOUT replacement.
                # IMPORTANT: we do NOT advance adapter.rng state (reproducibility spine). Instead we
                # derive a per-batch RNG seed from the frozen RNG state hash + batch_index.
                batch_index_for_selection = int(len(adapter.experiment_log))
                if adapter.frozen_rng_state_hash is None:
                    raise ValueError("Missing frozen_rng_state_hash; press Start before advancing.")
                seed_material = f"{adapter.frozen_rng_state_hash}|plasmin_selection|{batch_index_for_selection}"
                seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
                local_rng = random.Random(seed)

                candidates = [(int(e.edge_id), float(attack_weight_by_id[e.edge_id])) for e in intact_edges]
                candidates.sort(key=lambda t: t[0])
                selected_edge_ids = []
                for _ in range(int(N_plasmin)):
                    total_w = float(sum(w for _, w in candidates))
                    if total_w <= 0.0 or (not np.isfinite(total_w)):
                        raise ValueError("Invalid attack weight total for plasmin selection (must be finite and > 0).")
                    r = local_rng.random() * total_w
                    cum = 0.0
                    pick_idx = None
                    for j, (eid, w) in enumerate(candidates):
                        cum += float(w)
                        if r <= cum:
                            pick_idx = j
                            break
                    if pick_idx is None:
                        pick_idx = len(candidates) - 1
                    selected_edge_ids.append(int(candidates[pick_idx][0]))
                    del candidates[pick_idx]
                selected_edge_ids = sorted(selected_edge_ids)
                selected_edge_id_set = set(selected_edge_ids)

        # 3) Update temporal damage memory M_i once per batch (pre-degradation).
        mu = float(getattr(adapter, "memory_mu"))
        rho = float(getattr(adapter, "memory_rho"))
        if not (0.0 <= mu <= 1.0):
            raise ValueError("memory_mu must be in [0, 1] for Phase 2.8.")
        if rho < 0.0:
            raise ValueError("memory_rho must be >= 0 for Phase 2.8.")
        M_next_by_id: dict[Any, float] = {}
        for e in adapter.edges:
            if float(e.S) > 0.0:
                F = float(adapter._forces_by_edge_id.get(e.edge_id, 0.0))
                M_new = (1.0 - mu) * float(e.M) + mu * max(float(F), 0.0)
                assert M_new >= 0.0
                assert np.isfinite(M_new)
                M_next_by_id[e.edge_id] = float(M_new)

        # Phase 2.2: strain-rate proxy computed from post-relaxation mean tension.
        # If no intact edges exist, factor is 1.0 (required).
        # Note: This is computed BEFORE binding update, so uses base dt (will be updated to dt_used for actual integration).
        if len(intact_edges) == 0:
            strain_rate_factor = 1.0
        else:
            strain_rate_factor = float(adapter.phase2_2_strain_rate_factor(mean_tension=mean_tension, dt=dt))
        assert np.isfinite(strain_rate_factor)

        # c) Update strengths deterministically; clamp S in [0, 1]
        new_edges: list[Phase1EdgeSnapshot] = []
        total_k0 = 0.0
        total_keff = 0.0
        cleaved = 0
        newly_lysed_edge_ids: list[int] = []

        # Build a quick lookup for forces by edge_id for this batch.
        force_by_id = {e.edge_id: f for e, f in zip(intact_edges, force_list)}
        
        # Phase 2G (v5.0): Supply-limited stochastic plasmin seeding for spatial mode ONLY.
        # Replaces the continuous Langmuir "binding everywhere" update.
        # Binding is now sparse, stochastic, and supply-limited.
        dt_used = dt  # default: use base dt (legacy behavior)
        
        # Phase 2G tracking variables (for logging)
        bind_events_requested = 0
        bind_events_applied = 0
        total_unbound_this_batch = 0

        if FeatureFlags.USE_SPATIAL_PLASMIN:
            from dataclasses import replace  # Import here for immutable update

            if adapter.spatial_plasmin_params is None:
                raise ValueError("USE_SPATIAL_PLASMIN is True but spatial_plasmin_params not loaded.")
            if adapter.P_total_quanta is None or adapter.P_free_quanta is None:
                raise ValueError("Plasmin pool not initialized. Press Start before advancing.")
            
            # Extract required parameters (FAIL explicitly if missing; no silent defaults)
            # CRITICAL PHYSICS FIX: k_cat0=0 silently disables lysis (unpublishable).
            if "k_cat0" not in adapter.spatial_plasmin_params:
                raise ValueError(
                    "Spatial plasmin mode requires 'k_cat0' in meta_data.\n"
                    "k_cat0 = basal cleavage rate (s⁻¹) for force-dependent lysis.\n"
                    "Add k_cat0 to your input file meta_data table (typical value: 0.1 - 10 s⁻¹)."
                )
            k_cat0 = float(adapter.spatial_plasmin_params["k_cat0"])
            if k_cat0 <= 0.0:
                raise ValueError(f"Invalid k_cat0 = {k_cat0} (must be > 0 for lysis to occur).")

            # Other required kinetic parameters
            if "beta" not in adapter.spatial_plasmin_params:
                raise ValueError(
                    "Spatial plasmin mode requires 'beta' in meta_data.\n"
                    "beta = force-coupling coefficient for k_cat(F) = k_cat0 * exp(beta*F).\n"
                    "Add beta to your input file meta_data table (typical value: 0.01 - 0.1 /pN)."
                )
            beta_cleave = float(adapter.spatial_plasmin_params["beta"])
            # PHYSICS VALIDATION: beta must be >= 0 (cleavage accelerates or is neutral with tension)
            if beta_cleave < 0.0:
                raise ValueError(
                    f"Invalid beta={beta_cleave} for cleavage kinetics (must be >= 0).\n"
                    f"beta < 0 would decrease cleavage under tension (unphysical for force-dependent enzymes).\n"
                    f"For fibrin, use beta >= 0 (tension accelerates plasmin-mediated lysis)."
                )

            # Unbinding parameters (can default; unbinding optional in some models)
            k_off0 = float(adapter.spatial_plasmin_params.get("k_off0", 0.0))
            alpha = float(adapter.spatial_plasmin_params.get("alpha", 0.0))
            # PHYSICS VALIDATION: alpha must be >= 0 (catch-bond or neutral, not slip-bond)
            if alpha < 0.0:
                raise ValueError(
                    f"Invalid alpha={alpha} for unbinding kinetics (must be >= 0).\n"
                    f"alpha < 0 would produce slip-bond (unbinding accelerates under tension).\n"
                    f"For fibrin, use alpha >= 0 (catch-bond: tension stabilizes binding)."
                )
            lambda_bind_total = float(adapter.spatial_plasmin_params.get("lambda_bind_total", 10.0))
            
            # dt_used: compute once per batch and use consistently for:
            # - stochastic unbinding (p_unbind depends on dt_used)
            # - stochastic binding (Poisson events scale with dt_used)
            # - explicit Euler cleavage update (stability constrained via dt_cleave_safe)
            dt_used = float(dt)
            dt_cleave_rates: list[float] = []
            for e in adapter.edges:
                if float(e.S) > 0.0 and e.segments is not None:
                    T_edge = max(0.0, float(force_by_id.get(e.edge_id, 0.0)))
                    k_cat = k_cat0 * math.exp(beta_cleave * T_edge)
                    for seg in e.segments:
                        # Rate = k_cat * S_i (max possible B_i)
                        rate = k_cat * float(seg.S_i)
                        if rate > 0.0:
                            dt_cleave_rates.append(rate)
            if dt_cleave_rates:
                dt_max_cleave = 1.0 / max(dt_cleave_rates)
                dt_cleave_safe = 0.1 * dt_max_cleave
                if math.isfinite(dt_cleave_safe) and dt_cleave_safe > 0.0:
                    dt_used = min(dt_used, float(dt_cleave_safe))

            # NUMERICAL STABILITY: Hard floor on timestep (dt_min = 1e-4 s)
            # Prevents ultra-small timesteps from exploding batch count (memory/performance collapse).
            # Standard for stiff spring-reaction systems; smaller = slower but no biological gain.
            dt_min = 1e-4  # seconds
            if dt_used < dt_min:
                raise ValueError(
                    f"Computed timestep dt_used = {dt_used:.3e} s is below minimum floor dt_min = {dt_min:.3e} s.\n"
                    f"This indicates extremely stiff cleavage kinetics (k_cat × B_i >> 1).\n"
                    f"  Base dt = {dt:.3e} s\n"
                    f"  Max cleavage rate = {max(dt_cleave_rates) if dt_cleave_rates else 0.0:.3e} s⁻¹\n"
                    f"Suggestion: Reduce k_cat0 or increase base dt to avoid simulation collapse."
                )
            
            # ========== STEP A: UNBINDING (stochastic) ==========
            # For each segment with B_i > 0:
            #   p_unbind = 1 - exp(-k_off(T_edge) * dt_used)
            #   U_i ~ Binomial(B_i, p_unbind)
            #   B_i -= U_i, P_free_quanta += U_i
            
            updated_edges_unbind: list[Phase1EdgeSnapshot] = []
            P_free = int(adapter.P_free_quanta)
            
            for e in adapter.edges:
                if float(e.S) > 0.0 and e.segments is not None:
                    T_edge = max(0.0, float(force_by_id.get(e.edge_id, 0.0)))
                    k_off = k_off0 * math.exp(-alpha * T_edge)
                    p_unbind = 1.0 - math.exp(-k_off * dt_used)
                    p_unbind = max(0.0, min(1.0, p_unbind))  # clamp for safety
                    
                    updated_segments: list[FiberSegment] = []
                    for seg in e.segments:
                        B_i = int(round(seg.B_i))  # Convert to integer for binomial
                        if B_i > 0 and p_unbind > 0.0:
                            # Sample unbinding events: U_i ~ Binomial(B_i, p_unbind)
                            U_i = 0
                            for _ in range(B_i):
                                if adapter.rng.random() < p_unbind:
                                    U_i += 1
                            
                            B_i_new = B_i - U_i
                            P_free += U_i
                            total_unbound_this_batch += U_i
                        else:
                            B_i_new = B_i
                        
                        updated_segments.append(
                            FiberSegment(
                                segment_index=seg.segment_index,
                                n_i=seg.n_i,
                                B_i=float(B_i_new),
                                S_i=seg.S_i,
                            )
                        )
                    updated_edges_unbind.append(replace(e, segments=tuple(updated_segments)))
                else:
                    updated_edges_unbind.append(e)
            
            adapter.set_edges(updated_edges_unbind)
            adapter.P_free_quanta = int(P_free)
            
            # ========== STEP B: BINDING (stochastic, supply-limited) ==========
            # Sample N_bind_events ~ Poisson(lambda_bind_total * dt_used)
            # Cap at P_free_quanta
            # Choose target segments with weighted sampling (weight = available = S_i - B_i)
            
            expected_events = lambda_bind_total * dt_used
            if expected_events > 0.0:
                # NUMERICAL STABILITY: Switch to numpy for large lambda to avoid underflow
                # Threshold: lambda > 100 (inverse CDF becomes unstable; exp(-100) ≈ 0)
                if expected_events > 100.0:
                    # Use numpy's robust Poisson sampler for large lambda
                    # Convert adapter.rng state to numpy-compatible seed
                    rng_state = adapter.rng.getstate()
                    seed = int.from_bytes(hashlib.sha256(str(rng_state).encode()).digest()[:4], 'big')
                    np_rng = np.random.RandomState(seed)
                    N_bind_events = int(np_rng.poisson(expected_events))
                    # Advance adapter.rng to maintain deterministic state consistency
                    adapter.rng.random()  # Single draw to advance state
                else:
                    # Use inverse CDF method for small lambda (< 100)
                    L = math.exp(-expected_events)
                    k = 0
                    p = 1.0
                    while p > L:
                        k += 1
                        p *= adapter.rng.random()
                    N_bind_events = k - 1
            else:
                N_bind_events = 0
            
            bind_events_requested = int(N_bind_events)
            N_bind_events = min(N_bind_events, int(adapter.P_free_quanta))

            # Build segment weight list: (edge_idx, seg_idx, weight)
            segment_weights: list[tuple[int, int, float]] = []
            for e_idx, e in enumerate(adapter.edges):
                if float(e.S) > 0.0 and e.segments is not None:
                    for s_idx, seg in enumerate(e.segments):
                        available = max(0.0, float(seg.S_i) - float(seg.B_i))
                        if available > 0.0:
                            segment_weights.append((e_idx, s_idx, available))

            # Process bind events
            edges_list = list(adapter.edges)
            for _ in range(N_bind_events):
                if not segment_weights:
                    break
                
                # Compute total weight
                total_weight = sum(w for _, _, w in segment_weights)
                if total_weight <= 0.0:
                    break
                
                # Weighted random selection
                r = adapter.rng.random() * total_weight
                cum = 0.0
                selected_idx = -1
                for i, (e_idx, s_idx, w) in enumerate(segment_weights):
                    cum += w
                    if r < cum:
                        selected_idx = i
                        break
                
                if selected_idx < 0:
                    selected_idx = len(segment_weights) - 1  # fallback
                
                e_idx, s_idx, _ = segment_weights[selected_idx]
                
                # Update the segment's B_i
                edge = edges_list[e_idx]
                seg = edge.segments[s_idx]
                B_i_new = float(seg.B_i) + 1.0
                
                # Clamp to S_i
                if B_i_new > float(seg.S_i):
                    # No available sites on this segment for a full quantum; do not apply event.
                    continue
                bind_events_applied += 1
                adapter.P_free_quanta -= 1
                
                # Create updated segment
                new_seg = FiberSegment(
                    segment_index=seg.segment_index,
                    n_i=seg.n_i,
                    B_i=B_i_new,
                    S_i=seg.S_i,
                )
                
                # Rebuild segments tuple for this edge
                new_segs = list(edge.segments)
                new_segs[s_idx] = new_seg
                edges_list[e_idx] = replace(edge, segments=tuple(new_segs))
                
                # Update weight for this segment
                new_available = max(0.0, float(new_seg.S_i) - B_i_new)
                if new_available > 0.0:
                    segment_weights[selected_idx] = (e_idx, s_idx, new_available)
                else:
                    # Remove from candidates
                    segment_weights.pop(selected_idx)
            
            # Commit updated edges
            adapter.set_edges(edges_list)
            
            # ========== STEP C: CONSERVATION CHECKS ==========
            # P_free_quanta + sum(B_i) == P_total_quanta (with tolerance for rounding)
            total_bound = 0
            for e in adapter.edges:
                if e.segments is not None:
                    for seg in e.segments:
                        total_bound += int(round(seg.B_i))

            expected_total = int(adapter.P_total_quanta)
            actual_total = int(adapter.P_free_quanta) + total_bound

            # NUMERICAL STABILITY: Use tolerance to handle float->int rounding errors
            # Tolerance = max(1 quantum, 1e-6 × P_total) scales correctly for all pool sizes
            tolerance = max(1, int(1e-6 * expected_total))
            if abs(actual_total - expected_total) > tolerance:
                raise ValueError(
                    f"Plasmin conservation violated beyond tolerance:\n"
                    f"  P_free={adapter.P_free_quanta} + sum(B_i)={total_bound} = {actual_total}\n"
                    f"  Expected P_total={expected_total}\n"
                    f"  Difference = {abs(actual_total - expected_total)} exceeds tolerance = {tolerance}"
                )
            
            # Phase 2B (v5.0): Cleavage kinetics (update n_i based on B_i and tension)
            # This updates ONLY n_i; S, k_eff, edge removal NOT changed in this phase.
            # Cleavage rate: dn_i/dt = -k_cat(T) * B_i, k_cat(T) = k_cat0 * exp(beta * T)
            
            # Extract cleavage parameters (dt_used already includes dt_cleave stability constraint above)
            N_pf = float(adapter.spatial_plasmin_params.get("N_pf", 50.0))
            
            # Update n_i for all edges with segments
            updated_edges_cleave: list[Phase1EdgeSnapshot] = []
            for e in adapter.edges:
                if e.segments is not None:
                    T_edge = max(0.0, float(force_by_id.get(e.edge_id, 0.0)))
                    k_cat = k_cat0 * math.exp(beta_cleave * T_edge)
                    
                    # Update each segment's n_i using cleavage kinetics
                    updated_segments_cleave: list[FiberSegment] = []
                    for seg in e.segments:
                        n_i_old = float(seg.n_i)
                        B_i = float(seg.B_i)
                        
                        # dn_i/dt = -k_cat * B_i
                        rate_cleave = -k_cat * B_i
                        n_i_new = n_i_old + dt_used * rate_cleave
                        
                        # Clamp to [0, N_pf]
                        n_i_new = max(0.0, min(N_pf, n_i_new))
                        
                        # Replace segment with updated n_i (B_i, S_i unchanged)
                        updated_segments_cleave.append(
                            FiberSegment(
                                segment_index=seg.segment_index,
                                n_i=n_i_new,
                                B_i=B_i,  # unchanged from binding update
                                S_i=float(seg.S_i),
                            )
                        )
                    
                    # Replace edge with updated segments tuple
                    updated_edges_cleave.append(replace(e, segments=tuple(updated_segments_cleave)))
                else:
                    # No segments: keep edge as-is
                    updated_edges_cleave.append(e)
            
            # Commit updated edges (immutable replacement)
            adapter.set_edges(updated_edges_cleave)
            
            # Phase 2C (v5.0): Stiffness coupling (first chemistry → mechanics feedback)
            # Compute per-edge stiffness fraction from segment protofibrils (weakest-link)
            # and update snapshot.S to feed solver k_eff = k0 * S.
            
            N_pf = float(adapter.spatial_plasmin_params.get("N_pf", 50.0))
            
            updated_edges_stiffness: list[Phase1EdgeSnapshot] = []
            for e in adapter.edges:
                if e.segments is not None and len(e.segments) > 0:
                    # Weakest-link: f_edge = min(n_i / N_pf) over all segments
                    n_fracs = [float(seg.n_i) / N_pf for seg in e.segments]
                    f_edge = min(n_fracs)
                    
                    # Clamp to [0, 1]
                    f_edge = max(0.0, min(1.0, f_edge))
                    
                    # Set S = f_edge (this feeds solver k_eff = k0 * S)
                    updated_edges_stiffness.append(replace(e, S=float(f_edge)))
                else:
                    # No segments: keep edge as-is
                    updated_edges_stiffness.append(e)
            
            # Commit updated edges (immutable replacement)
            adapter.set_edges(updated_edges_stiffness)

            # Phase 2D: Batched fracture detection and edge removal
            # Scan all edges for rupture criterion: min(n_i/N_pf) <= n_crit_fraction
            # Remove all fractured edges from topology and archive full segment state.

            # FIX B1: Zero-length segment filtering (numerical safety)
            # Segments with S_i ≈ 0 are effectively zero-length (numerical artifacts).
            # Exclude them from fracture detection to prevent false triggers.
            EPS_SITES = 1e-6  # Minimum viable binding sites (unitless)

            n_crit_fraction = float(adapter.spatial_plasmin_params.get("n_crit_fraction", 0.1))
            fractured_edge_ids: list[Any] = []

            # Collect all fractured edges (deterministic order: sorted by edge_id)
            for e in sorted(adapter.edges, key=lambda ee: int(ee.edge_id)):
                if e.segments is not None and len(e.segments) > 0:
                    # Filter out zero-length segments (S_i ≈ 0) before fracture check
                    valid_segments = [seg for seg in e.segments if float(seg.S_i) > EPS_SITES]
                    if not valid_segments:
                        # All segments are zero-length; skip fracture check (edge is invalid)
                        continue
                    # Compute minimum protofibril fraction (only from valid segments)
                    n_min_frac = min(float(seg.n_i) / N_pf for seg in valid_segments)

                    if n_min_frac <= n_crit_fraction:
                        fractured_edge_ids.append(e.edge_id)

            # Remove fractured edges and archive history
            if fractured_edge_ids:
                # Archive full state for each fractured edge
                batch_index = int(len(adapter.experiment_log))
                for edge_id in fractured_edge_ids:
                    # Find edge snapshot
                    edge_snapshot = next((e for e in adapter.edges if e.edge_id == edge_id), None)
                    if edge_snapshot is not None:
                        # Get tension at failure from cached forces
                        tension_at_failure = float(adapter._forces_by_edge_id.get(edge_id, 0.0))

                        # Compute edge-specific strain at failure
                        strain_at_failure = None
                        if adapter._relaxed_node_coords is not None:
                            p_from = adapter._relaxed_node_coords.get(edge_snapshot.n_from)
                            p_to = adapter._relaxed_node_coords.get(edge_snapshot.n_to)
                            if p_from is not None and p_to is not None:
                                L_current = _euclidean(p_from, p_to)
                                L_rest = float(edge_snapshot.L_rest_effective)
                                if L_rest > 0.0:
                                    strain_at_failure = float((L_current - L_rest) / L_rest)

                        adapter.fractured_history.append({
                            "edge_id": edge_id,
                            "batch_index": batch_index,
                            "n_from": edge_snapshot.n_from,  # Phase 3: needed for visualization
                            "n_to": edge_snapshot.n_to,      # Phase 3: needed for visualization
                            "segments": edge_snapshot.segments,  # Full FiberSegment tuple
                            "final_stiffness": float(edge_snapshot.S),  # Weakest-link fraction
                            "tension_at_failure": tension_at_failure,  # Edge tension at fracture
                            "strain_at_failure": strain_at_failure,    # Edge strain at fracture
                        })

                # Remove fractured edges from active graph
                remaining_edges = [e for e in adapter.edges if e.edge_id not in fractured_edge_ids]
                adapter.set_edges(remaining_edges)

                # Force redistribution: relax once after edge removal
                adapter.relax(float(self.state.strain_value))

        # Phase 2.6: build local neighborhood mapping once (no persistent graph state).
        # Neighbors are edges that share at least one node.
        s_by_edge_id = {e.edge_id: float(e.S) for e in adapter.edges}
        node_to_edge_ids: dict[Any, list[Any]] = {}
        for e in adapter.edges:
            node_to_edge_ids.setdefault(e.n_from, []).append(e.edge_id)
            node_to_edge_ids.setdefault(e.n_to, []).append(e.edge_id)

        # Pre-batch geometry for energy gate is taken from cached post-relaxation node positions.
        coords_pre = adapter._relaxed_node_coords
        if coords_pre is None:
            raise ValueError("Missing cached relaxed node positions for energy gate. Relax at current strain before advancing.")
        # Phase 3.0 guardrails: force cache validity check (must exist for all intact edges).
        for e in adapter.edges:
            if float(e.S) > 0.0 and e.edge_id not in adapter._forces_by_edge_id:
                raise ValueError("Cached forces missing for one or more intact edges. Press Start to re-run relaxation at the fixed applied strain.")

        for e in adapter.edges:
            # Protofibril-based stiffness scaling for total_k0
            if FeatureFlags.USE_SPATIAL_PLASMIN and adapter.spatial_plasmin_params:
                N_pf = float(adapter.spatial_plasmin_params.get("N_pf", 50))
                total_k0 += float(e.k0) * N_pf
            else:
                total_k0 += float(e.k0)
            S_old = float(e.S)
            L_eff = float(e.L_rest_effective)
            M_i = float(M_next_by_id.get(e.edge_id, e.M))
            
            # Phase 2A.2: Gate legacy scalar degradation path
            # In spatial mode (v5.0), S is NOT updated by legacy degradation.
            # Binding kinetics already updated B_i; cleavage (n_i) not implemented yet.
            # S remains frozen at initialized value (typically 1.0) until cleavage is implemented.
            if not FeatureFlags.USE_SPATIAL_PLASMIN and S_old > 0.0:
                F = float(force_by_id.get(e.edge_id, 0.0))

                # 4) APPLY plastic rest-length update (Phase 2.3), once per batch, before degradation.
                plastic_rate = float(getattr(adapter, "plastic_rate"))
                plastic_F_threshold = float(getattr(adapter, "plastic_F_threshold"))
                if plastic_rate < 0.0:
                    raise ValueError("plastic_rate must be >= 0 for Phase 2.3.")
                if F > plastic_F_threshold:
                    dL = plastic_rate * (F - plastic_F_threshold) * dt_used
                    assert np.isfinite(dL)
                    L_eff = L_eff + float(dL)
                # Safety: L_rest_effective must never drop below imported rest length.
                assert L_eff >= float(e.original_rest_length)
                assert np.isfinite(L_eff)

                # Phase 2.2: combine force response with strain-rate factor (bounded, deterministic).
                gF = float(adapter.g_force(F))
                # Phase 2.4: force-driven rupture amplification term r(F) (>= 1).
                F_crit = float(getattr(adapter, "rupture_force_threshold"))
                gamma = float(getattr(adapter, "rupture_gamma"))
                if F <= F_crit:
                    rF = 1.0
                else:
                    rF = 1.0 + gamma * (F - F_crit)
                assert rF >= 1.0
                assert np.isfinite(rF)

                # Phase 2.5: energy-based failure gate (fracture-toughness proxy), computed from pre-batch geometry.
                p_from = coords_pre.get(e.n_from)
                p_to = coords_pre.get(e.n_to)
                if p_from is None or p_to is None:
                    raise ValueError("Missing cached node coordinates for energy gate computation.")
                L = _euclidean(p_from, p_to)
                assert np.isfinite(L)
                # E = 0.5 * k0 * S * (L - L_rest_effective)^2
                # Protofibril-based stiffness scaling (spatial mode only)
                dL_geom = float(L) - float(L_eff)
                if FeatureFlags.USE_SPATIAL_PLASMIN and adapter.spatial_plasmin_params:
                    N_pf = float(adapter.spatial_plasmin_params.get("N_pf", 50))
                    E_i = 0.5 * float(e.k0) * N_pf * float(S_old) * (dL_geom * dL_geom)
                else:
                    E_i = 0.5 * float(e.k0) * float(S_old) * (dL_geom * dL_geom)
                assert E_i >= 0.0
                assert np.isfinite(E_i)
                Gc = float(getattr(adapter, "fracture_Gc"))
                eta = float(getattr(adapter, "fracture_eta"))
                if E_i <= Gc:
                    e_gate = 1.0
                else:
                    e_gate = 1.0 + eta * (E_i - Gc)
                assert e_gate >= 1.0
                assert np.isfinite(e_gate)

                # Phase 2.6: topology-aware cooperativity gate from neighbor damage (local, deterministic).
                neighbor_ids = set(node_to_edge_ids.get(e.n_from, [])) | set(node_to_edge_ids.get(e.n_to, []))
                if e.edge_id in neighbor_ids:
                    neighbor_ids.remove(e.edge_id)
                damage_terms: list[float] = []
                for nid in neighbor_ids:
                    Sj = float(s_by_edge_id.get(nid, 0.0))
                    if Sj > 0.0:
                        damage_terms.append(1.0 - Sj)
                if damage_terms:
                    D_local = float(sum(damage_terms) / len(damage_terms))
                else:
                    D_local = 0.0
                assert D_local >= 0.0
                assert np.isfinite(D_local)
                chi = float(getattr(adapter, "coop_chi"))
                c_gate = 1.0 + chi * D_local
                assert c_gate >= 1.0
                assert np.isfinite(c_gate)

                # Phase 2.7: stress-shielding / load redistribution saturation.
                eps = float(getattr(adapter, "shield_eps"))
                if eps <= 0.0:
                    raise ValueError("shield_eps must be > 0 for Phase 2.7.")
                F_tension = max(0.0, float(F))
                f_load = float(F_tension) / float(mean_tension + eps)
                assert f_load >= 0.0
                assert np.isfinite(f_load)
                if f_load >= 1.0:
                    s_gate = 1.0
                else:
                    s_gate = max(0.0, f_load)
                assert 0.0 <= s_gate <= 1.0
                assert np.isfinite(s_gate)

                # Phase 2.8: memory gate (computed from updated M_i).
                M_i = float(M_next_by_id.get(e.edge_id, float(e.M)))
                assert M_i >= 0.0
                assert np.isfinite(M_i)
                m_gate = 1.0 + rho * M_i
                assert m_gate >= 1.0
                assert np.isfinite(m_gate)

                # Phase 2.9: directional anisotropy gate (load-alignment sensitivity).
                # Uses cached pre-batch geometry; load direction is uniaxial x: u_load = (1, 0).
                p_from = coords_pre.get(e.n_from)
                p_to = coords_pre.get(e.n_to)
                if p_from is None or p_to is None:
                    raise ValueError("Missing cached node coordinates for anisotropy gate computation.")
                dx = float(p_to[0]) - float(p_from[0])
                dy = float(p_to[1]) - float(p_from[1])
                L_dir = float(math.sqrt(dx * dx + dy * dy))
                if L_dir <= 0.0:
                    a = 0.0
                else:
                    a = abs(dx / L_dir)
                assert 0.0 <= a <= 1.0
                assert np.isfinite(a)
                kappa = float(getattr(adapter, "aniso_kappa"))
                a_gate = 1.0 + kappa * a
                assert a_gate >= 1.0
                assert np.isfinite(a_gate)

                g_total = float(gF) * float(strain_rate_factor) * float(rF) * float(e_gate) * float(c_gate) * float(s_gate) * float(m_gate) * float(a_gate)
                assert np.isfinite(g_total)

                # Phase 3.0 numerical safety clamp (not physics).
                g_max = float(getattr(adapter, "g_max"))
                if g_total > g_max:
                    g_total = g_max
                # Stage 5: limited plasmin exposure.
                # - Only selected edges receive lambda_eff this batch; others get lambda_eff = 0.
                # - If selection covers all intact edges, this collapses exactly to saturating mode.
                if len(intact_edges) == 0:
                    lambda_eff = 0.0
                elif int(e.edge_id) not in selected_edge_id_set:
                    lambda_eff = 0.0
                else:
                    if sigma_ref is None:
                        raise ValueError("Internal error: sigma_ref missing for intact edges.")
                    if adapter.thickness_ref is None:
                        raise ValueError("Missing thickness_ref on adapter. Press Start to freeze parameters before advancing.")
                    
                    # In spatial mode with sigma_ref = 0, use uniform stress factor = 1.0
                    if FeatureFlags.USE_SPATIAL_PLASMIN and (sigma_ref <= 0.0 or not np.isfinite(sigma_ref)):
                        stress_factor = 1.0
                    else:
                        beta = float(getattr(adapter, "degradation_beta", 1.0))
                        sigma = max(0.0, float(F))
                        stress_factor = (float(sigma) / float(sigma_ref)) ** float(beta)
                    
                    gamma_d = float(getattr(adapter, "degradation_gamma", 1.0))
                    thickness_factor = (float(adapter.thickness_ref) / float(e.thickness)) ** float(gamma_d)
                    lambda_eff = float(lambda_0) * float(stress_factor) * float(thickness_factor)
                if not np.isfinite(lambda_eff):
                    raise ValueError("Invalid lambda_eff (NaN/Inf) computed for an edge.")
                if lambda_eff < 0.0:
                    raise ValueError("Invalid lambda_eff (negative) computed for an edge.")

                lam = float(lambda_eff) * float(g_total)
                S_new = S_old - lam * dt_used
                if S_new < 0.0:
                    S_new = 0.0
                elif S_new > 1.0:
                    S_new = 1.0
                
                if S_new <= 0.0:
                    cleaved += 1
            else:
                # Legacy mode: already cleaved edge (S <= 0)
                S_new = 0.0
                # Do not evolve plasticity for cleaved edges.
                # (L_eff, M_i already set above)
                
                if not FeatureFlags.USE_SPATIAL_PLASMIN:
                    # Only count as cleaved in legacy mode
                    if S_old > 0.0 and S_new <= 0.0:
                        cleaved += 1
            
            # Spatial mode: S remains unchanged (no legacy degradation)
            if FeatureFlags.USE_SPATIAL_PLASMIN:
                S_new = S_old
                # M_i and L_eff already set above

            # Protofibril-based stiffness scaling for total_keff
            if FeatureFlags.USE_SPATIAL_PLASMIN and adapter.spatial_plasmin_params:
                N_pf = float(adapter.spatial_plasmin_params.get("N_pf", 50))
                total_keff += float(e.k0) * N_pf * float(S_new)
            else:
                total_keff += float(e.k0) * float(S_new)

            # Stage 4: per-edge lysis tracking (observational only; set once).
            # In spatial mode, lysis is tracked by cleavage (n_i -> 0), not by S degradation.
            prev_lysis_batch = e.lysis_batch_index
            prev_lysis_time = e.lysis_time
            if not FeatureFlags.USE_SPATIAL_PLASMIN:
                if prev_lysis_batch is None and float(S_old) > 0.0 and float(S_new) <= 0.0:
                    # Lysis is attributed to this batch (batch_index = current log length), at end-of-batch time.
                    prev_lysis_batch = int(len(adapter.experiment_log))
                    prev_lysis_time = float(self.state.time) + float(dt_used)
                    newly_lysed_edge_ids.append(int(e.edge_id))

            new_edges.append(
                Phase1EdgeSnapshot(
                    edge_id=e.edge_id,
                    n_from=e.n_from,
                    n_to=e.n_to,
                    k0=float(e.k0),
                    original_rest_length=float(e.original_rest_length),
                    L_rest_effective=float(L_eff),
                    M=float(M_i),
                    S=float(S_new),
                    thickness=float(e.thickness),
                    lysis_batch_index=(int(prev_lysis_batch) if prev_lysis_batch is not None else None),
                    lysis_time=(float(prev_lysis_time) if prev_lysis_time is not None else None),
                    segments=e.segments,  # Phase 2A: preserve spatial plasmin segments (updated by binding kinetics)
                )
            )

        # Phase 3.0 cleavage density fail-safe (graceful termination before committing state).
        # ABORT BEHAVIOR FIX (Q5): Write log + stop gracefully (no re-raise).
        # In spatial mode, cleavage is by n_i -> 0 (not implemented yet), not by S -> 0.
        newly_cleaved = 0  # Initialize for both modes
        if not FeatureFlags.USE_SPATIAL_PLASMIN:
            intact_pre = sum(1 for e in adapter.edges if float(e.S) > 0.0)
            if intact_pre > 0:
                for e_old, e_new in zip(adapter.edges, new_edges):
                    if float(e_old.S) > 0.0 and float(e_new.S) <= 0.0:
                        newly_cleaved += 1
                frac = float(newly_cleaved) / float(intact_pre)
                cleavage_batch_cap = float(getattr(adapter, "cleavage_batch_cap"))
                if frac > cleavage_batch_cap:
                    # GRACEFUL TERMINATION: Write experiment_log entry and stop cleanly.
                    # Do NOT re-raise; this allows export of diagnostics.
                    reason = "cleavage_batch_cap_exceeded"
                    batch_index = int(len(adapter.experiment_log))
                    time_val = float(self.state.time) + dt_used
                    cleaved_edges_total = sum(1 for e in adapter.edges if float(e.S) <= 0.0)
                    total_edges = max(1, len(adapter.edges))
                    cleavage_fraction_total = float(cleaved_edges_total) / float(total_edges)

                    adapter.termination_reason = str(reason)
                    adapter.termination_batch_index = int(batch_index)
                    adapter.termination_time = float(time_val)
                    adapter.termination_cleavage_fraction = float(cleavage_fraction_total)

                    # Compute deterministic batch_hash for termination entry
                    edges_sorted = sorted(adapter.edges, key=lambda ee: int(ee.edge_id))
                    payload = {
                        "batch_index": int(batch_index),
                        "time": float(time_val),
                        "strain": float(self.state.strain_value),
                        "sigma_ref": (float(sigma_ref) if sigma_ref is not None else None),
                        "plasmin_selected_edge_ids": list(selected_edge_ids),
                        "termination_reason": str(reason),
                        "termination_batch_index": int(batch_index),
                        "termination_time": float(time_val),
                        "termination_cleavage_fraction": float(cleavage_fraction_total),
                        "cleavage_batch_frac_attempted": float(frac),
                        "cleavage_batch_cap": float(cleavage_batch_cap),
                        "edges": [
                            {
                                "edge_id": int(e.edge_id),
                                "S": float(e.S),
                                "M": float(e.M),
                                "original_rest_length": float(e.original_rest_length),
                                "L_rest_effective": float(e.L_rest_effective),
                                "thickness": float(e.thickness),
                                "lysis_batch_index": (int(e.lysis_batch_index) if e.lysis_batch_index is not None else None),
                                "lysis_time": (float(e.lysis_time) if e.lysis_time is not None else None),
                            }
                            for e in edges_sorted
                        ],
                        "frozen_params": copy.deepcopy(adapter.frozen_params),
                        "provenance_hash": adapter.provenance_hash,
                        "rng_state_hash": adapter.frozen_rng_state_hash,
                    }
                    batch_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

                    intact_edges_now = sum(1 for e in adapter.edges if float(e.S) > 0.0)
                    if FeatureFlags.USE_SPATIAL_PLASMIN and adapter.spatial_plasmin_params:
                        N_pf = float(adapter.spatial_plasmin_params.get("N_pf", 50))
                        sum_k0 = float(sum(float(e.k0) * N_pf for e in adapter.edges))
                        sum_keff = float(sum(float(e.k0) * N_pf * float(e.S) for e in adapter.edges if float(e.S) > 0.0))
                    else:
                        sum_k0 = float(sum(float(e.k0) for e in adapter.edges))
                        sum_keff = float(sum(float(e.k0) * float(e.S) for e in adapter.edges if float(e.S) > 0.0))
                    eps = 1e-12
                    lysis_fraction = float(1.0 - (sum_keff / max(eps, sum_k0)))

                    adapter.experiment_log.append({
                        "batch_index": int(batch_index),
                        "provenance_hash": adapter.provenance_hash,
                        "rng_state_hash": adapter.frozen_rng_state_hash,
                        "batch_hash": batch_hash,
                        "batch_duration_sec": float(time.perf_counter() - t0),
                        "thickness_hash": (adapter.frozen_params or {}).get("thickness_hash") if isinstance(adapter.frozen_params, dict) else None,
                        "plasmin_mode": str(getattr(adapter, "plasmin_mode", "saturating")),
                        "N_plasmin": int(getattr(adapter, "N_plasmin", 1)),
                        "plasmin_selected_edge_ids": list(selected_edge_ids),
                        "newly_lysed_edge_ids": [],
                        "cumulative_lysed_edge_ids": sorted([int(e.edge_id) for e in adapter.edges if e.lysis_batch_index is not None]),
                        "global_lysis_batch_index": adapter.global_lysis_batch_index,
                        "global_lysis_time": adapter.global_lysis_time,
                        "time": float(time_val),
                        "strain": float(self.state.strain_value),
                        "branch_parent_batch_index": adapter.branch_parent_batch_index,
                        "branch_parent_batch_hash": adapter.branch_parent_batch_hash,
                        "sweep_param": adapter.sweep_param,
                        "sweep_value": adapter.sweep_value,
                        "grid_params": copy.deepcopy(adapter.grid_params),
                        "intact_edges": int(intact_edges_now),
                        "cleaved_edges_total": int(cleaved_edges_total),
                        "newly_cleaved": int(newly_cleaved),
                        "mean_tension": float(mean_tension),
                        "lysis_fraction": float(lysis_fraction),
                        "dt_used": float(dt_used),
                        "termination_reason": str(reason),
                        "termination_batch_index": int(batch_index),
                        "termination_time": float(time_val),
                        "termination_cleavage_fraction": float(cleavage_fraction_total),
                        "cleavage_batch_frac_attempted": float(frac),
                        "cleavage_batch_cap": float(cleavage_batch_cap),
                        "params": {
                            "lambda_0": float(adapter.lambda_0),
                            "dt": float(adapter.dt),
                            "delta": float(adapter.delta),
                        },
                    })

                    # Stop cleanly (no exception).
                    self.last_batch_duration_sec = float(adapter.experiment_log[-1]["batch_duration_sec"])
                    self.state.time = float(time_val)
                    self.state.is_running = False
                    self.state.is_paused = False
                    return False

        # d) removal is represented by S=0; solver excludes S<=0 automatically
        adapter.set_edges(new_edges)

        # e) relax once after degradation (propagates stiffness/damage changes to forces)
        adapter.relax(float(self.state.strain_value))

        # Phase 3.0 post-relaxation sanity checks (abort).
        coords_post = adapter._relaxed_node_coords
        if coords_post is None:
            raise ValueError("Post-relaxation sanity check failed: missing relaxed node coordinates.")
        for nid, (x, y) in coords_post.items():
            if not (np.isfinite(x) and np.isfinite(y)):
                raise ValueError(f"Post-relaxation sanity check failed: NaN/Inf node position at node {nid}.")
        for eid, f in adapter._forces_by_edge_id.items():
            if not np.isfinite(f):
                raise ValueError(f"Post-relaxation sanity check failed: NaN/Inf force cached for edge {eid}.")
        post_intact_forces = [max(0.0, float(adapter._forces_by_edge_id[e.edge_id])) for e in adapter.edges if float(e.S) > 0.0]
        post_mean_tension = float(sum(post_intact_forces) / len(post_intact_forces)) if post_intact_forces else 0.0
        if not np.isfinite(post_mean_tension):
            raise ValueError("Post-relaxation sanity check failed: mean_tension is NaN/Inf.")

        # Boundary/grip invariants (fatal on violation): boundary nodes must share identical x == grip_x.
        if adapter.left_grip_x is not None and adapter.right_grip_x is not None:
            gxL = float(adapter.left_grip_x)
            gxR = float(adapter.right_grip_x)
            left_bad: list[tuple[int, float, float]] = []
            right_bad: list[tuple[int, float, float]] = []
            for nid in sorted(int(x) for x in adapter.left_boundary_node_ids):
                xy = coords_post.get(int(nid))
                if xy is None:
                    left_bad.append((int(nid), float("nan"), float("nan")))
                else:
                    x, y = float(xy[0]), float(xy[1])
                    if abs(x - gxL) > 1e-9 * max(1.0, abs(gxL)):
                        left_bad.append((int(nid), x, y))
            for nid in sorted(int(x) for x in adapter.right_boundary_node_ids):
                xy = coords_post.get(int(nid))
                if xy is None:
                    right_bad.append((int(nid), float("nan"), float("nan")))
                else:
                    x, y = float(xy[0]), float(xy[1])
                    if abs(x - gxR) > 1e-9 * max(1.0, abs(gxR)):
                        right_bad.append((int(nid), x, y))
            if left_bad or right_bad:
                raise RuntimeError(
                    "Rigid grip invariant violated after relaxation.\n"
                    f"Expected left_grip_x={gxL}, right_grip_x={gxR}\n"
                    f"Left boundary mismatches (node_id,x,y): {left_bad[:20]}\n"
                    f"Right boundary mismatches (node_id,x,y): {right_bad[:20]}"
                )

        # time advances only by dt_used (may differ from base dt in spatial mode)
        self.state.time = float(self.state.time) + dt_used

        # Phase 2.2: update prev_mean_tension at end of batch (after relaxation).
        if post_intact_forces:
            adapter.prev_mean_tension = float(sum(post_intact_forces) / len(post_intact_forces))
        else:
            adapter.prev_mean_tension = None

        if total_k0 == 0.0:
            raise ValueError("Cannot compute lysis_fraction: sum(k0) == 0")
        lysis_fraction = 1.0 - (total_keff / total_k0)

        # Stage 4: global lysis time (observational only; set once at first threshold crossing).
        threshold = float(getattr(adapter, "global_lysis_threshold", 0.9))
        if not np.isfinite(threshold):
            raise ValueError("Invalid global_lysis_threshold (must be finite).")
        if adapter.global_lysis_batch_index is None and float(lysis_fraction) >= float(threshold):
            adapter.global_lysis_batch_index = int(len(adapter.experiment_log))
            adapter.global_lysis_time = float(self.state.time)

        # Metrics returned as read-only observables
        self.last_metrics = {
            "cleaved_fibers": cleaved,
            "lysis_fraction": lysis_fraction,
        }

        # Phase 3.7: deterministic batch_hash (end-to-end integrity) computed after success.
        # Payload excludes forces/node positions/GUI state by design.
        batch_index = len(adapter.experiment_log)
        edges_sorted = sorted(adapter.edges, key=lambda ee: int(ee.edge_id))
        payload = {
            "batch_index": int(batch_index),
            "time": float(self.state.time),
            "strain": float(self.state.strain_value),
            "sigma_ref": (float(sigma_ref) if sigma_ref is not None else None),
            "plasmin_selected_edge_ids": list(selected_edge_ids),
            "global_lysis_batch_index": adapter.global_lysis_batch_index,
            "global_lysis_time": adapter.global_lysis_time,
            "edges": [
                {
                    "edge_id": int(e.edge_id),
                    "S": float(e.S),
                    "M": float(e.M),
                    "original_rest_length": float(e.original_rest_length),
                    "L_rest_effective": float(e.L_rest_effective),
                    "thickness": float(e.thickness),
                    "lysis_batch_index": (int(e.lysis_batch_index) if e.lysis_batch_index is not None else None),
                    "lysis_time": (float(e.lysis_time) if e.lysis_time is not None else None),
                }
                for e in edges_sorted
            ],
            "frozen_params": copy.deepcopy(adapter.frozen_params),
            "provenance_hash": adapter.provenance_hash,
            "rng_state_hash": adapter.frozen_rng_state_hash,
        }
        batch_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

        # Phase 3.1 deterministic experiment logging (append-only; no timestamps).
        # Append ONLY after a successful batch (i.e., after relaxation + sanity checks).
        intact_edges_post = sum(1 for e in adapter.edges if float(e.S) > 0.0)
        cleaved_edges_total = sum(1 for e in adapter.edges if float(e.S) <= 0.0)
        
        # Phase 2B/2C (v5.0): Compute spatial plasmin observables
        n_min_frac = None
        n_mean_frac = None
        total_bound_plasmin = None
        min_stiff_frac = None  # Phase 2C: min(S) over edges with segments
        mean_stiff_frac = None  # Phase 2C: mean(S) over edges with segments
        if FeatureFlags.USE_SPATIAL_PLASMIN and adapter.spatial_plasmin_params is not None:
            N_pf = float(adapter.spatial_plasmin_params.get("N_pf", 50.0))
            all_n_fracs = []
            all_B_i = []
            all_S_fracs = []  # Phase 2C: stiffness fractions
            for e in adapter.edges:
                if e.segments is not None:
                    for seg in e.segments:
                        all_n_fracs.append(float(seg.n_i) / N_pf)
                        all_B_i.append(float(seg.B_i))
                    # Phase 2C: S now reflects f_edge = min(n_i/N_pf)
                    all_S_fracs.append(float(e.S))
            if all_n_fracs:
                n_min_frac = float(min(all_n_fracs))
                n_mean_frac = float(sum(all_n_fracs) / len(all_n_fracs))
            if all_B_i:
                total_bound_plasmin = float(sum(all_B_i))
            if all_S_fracs:
                min_stiff_frac = float(min(all_S_fracs))
                mean_stiff_frac = float(sum(all_S_fracs) / len(all_S_fracs))

        # Phase 3E: Per-edge statistics (spatial mode only)
        # Compute detailed per-edge observables for JSON export
        per_edge_stats = {}
        edge_n_min_global = None
        edge_n_mean_global = None
        edge_B_total_sum = None
        if FeatureFlags.USE_SPATIAL_PLASMIN and adapter.spatial_plasmin_params is not None:
            N_pf = float(adapter.spatial_plasmin_params.get("N_pf", 50.0))
            all_edge_n_mins = []
            all_edge_n_means = []
            all_edge_B_totals = []

            for e in adapter.edges:
                if e.segments is not None and len(e.segments) > 0:
                    # Per-edge n_i/N_pf statistics
                    n_fracs_edge = [float(seg.n_i) / N_pf for seg in e.segments]
                    n_min_edge = float(min(n_fracs_edge))
                    n_mean_edge = float(sum(n_fracs_edge) / len(n_fracs_edge))

                    # Per-edge total bound plasmin
                    B_total_edge = float(sum(float(seg.B_i) for seg in e.segments))

                    # Store per-edge stats (for JSON export)
                    per_edge_stats[int(e.edge_id)] = {
                        "n_min_frac": n_min_edge,
                        "n_mean_frac": n_mean_edge,
                        "B_total": B_total_edge,
                    }

                    # Collect for global aggregates (for CSV export)
                    all_edge_n_mins.append(n_min_edge)
                    all_edge_n_means.append(n_mean_edge)
                    all_edge_B_totals.append(B_total_edge)

            # Global aggregates for CSV
            if all_edge_n_mins:
                edge_n_min_global = float(min(all_edge_n_mins))
            if all_edge_n_means:
                edge_n_mean_global = float(sum(all_edge_n_means) / len(all_edge_n_means))
            if all_edge_B_totals:
                edge_B_total_sum = float(sum(all_edge_B_totals))

        adapter.experiment_log.append(
            {
                "batch_index": int(batch_index),
                "provenance_hash": adapter.provenance_hash,
                "rng_state_hash": adapter.frozen_rng_state_hash,
                "batch_hash": batch_hash,
                "batch_duration_sec": float(time.perf_counter() - t0),
                "thickness_hash": (adapter.frozen_params or {}).get("thickness_hash") if isinstance(adapter.frozen_params, dict) else None,
                "plasmin_mode": str(getattr(adapter, "plasmin_mode", "saturating")),
                "N_plasmin": int(getattr(adapter, "N_plasmin", 1)),
                "plasmin_selected_edge_ids": list(selected_edge_ids),
                "newly_lysed_edge_ids": sorted([int(eid) for eid in newly_lysed_edge_ids]),
                "cumulative_lysed_edge_ids": sorted([int(e.edge_id) for e in adapter.edges if e.lysis_batch_index is not None]),
                "global_lysis_batch_index": adapter.global_lysis_batch_index,
                "global_lysis_time": adapter.global_lysis_time,
                "time": float(self.state.time),
                "strain": float(self.state.strain_value),
                "branch_parent_batch_index": adapter.branch_parent_batch_index,
                "branch_parent_batch_hash": adapter.branch_parent_batch_hash,
                "sweep_param": adapter.sweep_param,
                "sweep_value": adapter.sweep_value,
                "grid_params": copy.deepcopy(adapter.grid_params),
                "intact_edges": int(intact_edges_post),
                "cleaved_edges_total": int(cleaved_edges_total),
                "newly_cleaved": int(newly_cleaved),
                "mean_tension": float(post_mean_tension),
                "lysis_fraction": float(lysis_fraction),
                "dt_used": float(dt_used),  # Phase 2A: actual timestep used (may differ from base dt in spatial mode)
                "n_min_frac": n_min_frac,  # Phase 2B: min(n_i/N_pf) over all segments (spatial mode only)
                "n_mean_frac": n_mean_frac,  # Phase 2B: mean(n_i/N_pf) over all segments (spatial mode only)
                "total_bound_plasmin": total_bound_plasmin,  # Phase 2B: sum(B_i) over all segments (spatial mode only)
                "total_bound_this_batch": int(total_bound_plasmin) if (total_bound_plasmin is not None) else None,  # Phase 2G: sum(B_i) (integer quanta)
                "min_stiff_frac": min_stiff_frac,  # Phase 2C: min(S=f_edge) over edges with segments (spatial mode only)
                "mean_stiff_frac": mean_stiff_frac,  # Phase 2C: mean(S=f_edge) over edges with segments (spatial mode only)
                # Phase 2G: Stochastic plasmin seeding observables (spatial mode only)
                "P_total_quanta": int(adapter.P_total_quanta) if adapter.P_total_quanta is not None else None,
                "P_free_quanta": int(adapter.P_free_quanta) if adapter.P_free_quanta is not None else None,
                "bind_events_requested": int(bind_events_requested) if FeatureFlags.USE_SPATIAL_PLASMIN else None,
                "bind_events_applied": int(bind_events_applied) if FeatureFlags.USE_SPATIAL_PLASMIN else None,
                "total_unbound_this_batch": int(total_unbound_this_batch) if FeatureFlags.USE_SPATIAL_PLASMIN else None,
                # Phase 3E: Per-edge statistics (spatial mode only)
                "per_edge_stats": dict(per_edge_stats) if per_edge_stats else None,  # Full per-edge dict (JSON only)
                "edge_n_min_global": edge_n_min_global,  # Global min of per-edge n_min (CSV aggregate)
                "edge_n_mean_global": edge_n_mean_global,  # Global mean of per-edge n_mean (CSV aggregate)
                "edge_B_total_sum": edge_B_total_sum,  # Sum of per-edge total bound plasmin (CSV aggregate)
                "params": {
                    "lambda_0": float(adapter.lambda_0),
                    "dt": float(adapter.dt),
                    "delta": float(adapter.delta),
                    "force_alpha": float(getattr(adapter, "force_alpha")),
                    "force_F0": float(getattr(adapter, "force_F0")),
                    "force_hill_n": float(getattr(adapter, "force_hill_n")),
                    "rate_beta": float(getattr(adapter, "rate_beta")),
                    "plastic_rate": float(getattr(adapter, "plastic_rate")),
                    "rupture_gamma": float(getattr(adapter, "rupture_gamma")),
                    "fracture_Gc": float(getattr(adapter, "fracture_Gc")),
                    "fracture_eta": float(getattr(adapter, "fracture_eta")),
                    "coop_chi": float(getattr(adapter, "coop_chi")),
                    "aniso_kappa": float(getattr(adapter, "aniso_kappa")),
                    "memory_mu": float(getattr(adapter, "memory_mu")),
                    "memory_rho": float(getattr(adapter, "memory_rho")),
                },
            }
        )

        # Phase 3.8: performance envelope warnings (diagnostic only; never abort).
        self.last_batch_duration_sec = float(adapter.experiment_log[-1]["batch_duration_sec"])
        soft_warn_sec = 0.5
        hard_warn_sec = 2.0
        if self.last_batch_duration_sec > soft_warn_sec:
            print(f"[ResearchSimulation] Warning: batch duration {self.last_batch_duration_sec:.3f}s exceeded soft threshold {soft_warn_sec:.3f}s")
        # UI-level non-blocking dialog for hard threshold is handled by the page (no controller UI calls).

        # Phase 2E: Percolation-based termination (spatial mode only)
        # Check if network is still connected (left to right) after edge removal.
        # Legacy mode uses sigma_ref <= 0 termination (checked earlier).
        if FeatureFlags.USE_SPATIAL_PLASMIN:
            from ...managers.edge_evolution_engine import EdgeEvolutionEngine

            # Identify left and right boundary nodes
            left_boundary_ids = set(adapter.left_boundary_node_ids)
            right_boundary_ids = set(adapter.right_boundary_node_ids)

            # Check percolation using current edges and node coordinates
            is_connected = EdgeEvolutionEngine.check_percolation(
                edges=adapter.edges,
                left_boundary_ids=left_boundary_ids,
                right_boundary_ids=right_boundary_ids,
                node_coords=adapter.render_node_coords,
            )

            if not is_connected:
                # Network disconnected: terminate simulation
                reason = "network_percolation_failure"
                batch_index = int(len(adapter.experiment_log))
                time_val = float(self.state.time)  # Already advanced in experiment_log
                cleaved_edges_total = len(adapter.fractured_history)
                total_edges = len(adapter.edges) + cleaved_edges_total
                cleavage_fraction = float(cleaved_edges_total) / max(1, float(total_edges))

                adapter.termination_reason = str(reason)
                adapter.termination_batch_index = int(batch_index)
                adapter.termination_time = float(time_val)
                adapter.termination_cleavage_fraction = float(cleavage_fraction)

                print(f"[ResearchSimulation] Network disconnected at batch {batch_index} — terminating simulation")
                self.state.is_running = False
                self.state.is_paused = False
                return False

        return True

    def run_n_batches(self, n: int):
        """
        Phase 4.0 explicit multi-batch runner (deterministic, bounded, user-controlled).

        Rules:
        - n must be a positive integer
        - hard cap n_max = 100
        - executes exactly n sequential calls to advance_one_batch()
        - stops immediately on first failure (guardrail abort)
        - does not roll back completed batches
        """
        n_max = 100
        if not isinstance(n, int):
            raise ValueError("n must be an integer.")
        if n < 1 or n > n_max:
            raise ValueError(f"n must be between 1 and {n_max}.")

        completed = 0
        for i in range(n):
            try:
                ok = self.advance_one_batch()
            except Exception as e:
                # Report the failing batch index deterministically (1-based for users).
                raise RuntimeError(f"Batch {i + 1} failed after {completed} completed: {e}") from e
            if ok is False:
                # Terminal-state: stop early without error.
                break
            completed += 1
        return completed

    def resume_from_checkpoint(self, snapshot_path: str, log_path: str, resume_batch_index: int):
        """
        Phase 4.1 controller wiring: resume deterministically from snapshot + log at a batch index.
        On success, replaces loaded adapter state and updates controller time/strain.
        """
        adapter = self.state.loaded_network
        if not isinstance(adapter, Phase1NetworkAdapter):
            raise ValueError("No loaded Research Simulation network/adapter to resume into. Load a network first.")

        # Perform in-place resume only after all validations pass inside adapter.
        info = adapter.load_checkpoint(snapshot_path, log_path, int(resume_batch_index))

        # Sync controller state from checkpoint (time + strain), keep parameters frozen.
        self.state.time = float(info["resume_time"])
        self.state.strain_value = float(info["resume_strain"])
        # Allow immediate continuation (no Start needed; parameters already frozen).
        self.state.is_running = True
        self.state.is_paused = False
        # Ensure controller RNG uses the restored frozen RNG state and is referenced by adapter.
        if adapter.rng is None:
            adapter.rng = random.Random()
        self.rng = adapter.rng
        return info

    def fork_from_checkpoint(self, snapshot_path: str, log_path: str, resume_batch_index: int):
        """
        Phase 4.2 controller wiring: fork a new experimental branch from a checkpoint.
        Replaces the current adapter with the forked adapter (copy-on-write).
        """
        current = self.state.loaded_network
        if not isinstance(current, Phase1NetworkAdapter):
            raise ValueError("No loaded Research Simulation network/adapter to fork from. Load a network first.")

        forked = current.fork_from_checkpoint(snapshot_path, log_path, int(resume_batch_index))

        # Replace adapter
        self.state.loaded_network = forked

        # Sync controller time/strain to fork point (parent batch state).
        parent_entry = forked.experiment_log[int(resume_batch_index)]
        self.state.time = float(parent_entry["time"])
        self.state.strain_value = float(parent_entry.get("strain", 0.0))

        # Clear prior diagnostics/status
        self.last_metrics = None
        self.last_batch_duration_sec = None

        # Mark run as active (parameters remain frozen).
        self.state.is_running = True
        self.state.is_paused = False

        # Use forked RNG
        self.rng = forked.rng
        return {"fork_parent_batch_index": int(resume_batch_index), "fork_begin_batch_index": int(resume_batch_index) + 1, "time": float(self.state.time), "strain": float(self.state.strain_value)}

    def run_parameter_sweep_from_checkpoint(
        self,
        snapshot_path: str,
        log_path: str,
        resume_batch_index: int,
        param_name: str,
        values: list[float],
        batches_per_branch: int,
    ):
        """
        Phase 4.3: deterministic fan-out parameter sweep from a checkpoint.

        For each sweep value:
        - fork a fresh adapter from checkpoint (copy-on-write)
        - override exactly ONE parameter in frozen_params (param_name)
        - recompute provenance_hash for that branch
        - run exactly `batches_per_branch` sequential batches (hard cap 50)
        - collect {value, final_time, final_lysis_fraction, final_batch_hash}

        Constraints:
        - branches run sequentially
        - failures in one branch do not affect others
        - disallow changing dt
        - disallow changing RNG seed/state
        """
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError("values must be a non-empty list of floats.")
        if len(values) > 10:
            raise ValueError("values length must be <= 10.")
        if not isinstance(batches_per_branch, int) or batches_per_branch < 1 or batches_per_branch > 50:
            raise ValueError("batches_per_branch must be an int in [1, 50].")

        param_name = str(param_name).strip()
        if param_name == "":
            raise ValueError("param_name is required.")
        if param_name in ("dt", "thickness_ref", "thickness_alpha", "thickness_hash", "beta", "gamma"):
            raise ValueError(f"Changing {param_name} is not allowed in parameter sweeps.")

        current = self.state.loaded_network
        if not isinstance(current, Phase1NetworkAdapter):
            raise ValueError("No loaded Research Simulation network/adapter. Load a network first.")

        results: list[dict[str, Any]] = []
        failures: list[str] = []

        for v in values:
            try:
                sweep_value = float(v)
            except Exception:
                failures.append(str(v))
                continue

            try:
                forked = current.fork_from_checkpoint(snapshot_path, log_path, int(resume_batch_index))
                if forked.frozen_params is None:
                    raise ValueError("Forked adapter has no frozen_params.")
                if param_name not in forked.frozen_params:
                    raise ValueError(f"Parameter '{param_name}' not present in frozen_params.")

                # Override exactly one parameter (do not touch RNG state).
                forked.frozen_params[param_name] = float(sweep_value)

                # Apply the override to the adapter attribute used in computation (minimal mapping).
                if hasattr(forked, param_name):
                    setattr(forked, param_name, float(sweep_value))
                else:
                    # Handle known keys that map to different attribute names.
                    if param_name in ("force_alpha", "force_F0", "force_hill_n"):
                        setattr(forked, param_name, float(sweep_value))
                    else:
                        raise ValueError(f"Cannot apply override for '{param_name}' (no matching adapter attribute).")

                # Recompute provenance hash deterministically for the branch.
                frozen_json = json.dumps(forked.frozen_params, sort_keys=True)
                forked.provenance_hash = hashlib.sha256(frozen_json.encode("utf-8")).hexdigest()

                # Stamp sweep metadata into future log entries.
                forked.sweep_param = param_name
                forked.sweep_value = float(sweep_value)

                # Run batches on a temporary controller so the live controller is not mutated.
                temp_ctrl = SimulationController()
                temp_ctrl.state = SimulationState(
                    loaded_network=forked,
                    strain_value=float(self.state.strain_value),
                    time=float(forked.experiment_log[-1]["time"]) if forked.experiment_log else float(self.state.time),
                    is_running=True,
                    is_paused=False,
                )
                temp_ctrl.rng = forked.rng

                temp_ctrl.run_n_batches(int(batches_per_branch))

                final_entry = forked.experiment_log[-1]
                results.append(
                    {
                        "value": float(sweep_value),
                        "final_time": float(final_entry["time"]),
                        "final_lysis_fraction": float(final_entry["lysis_fraction"]),
                        "final_batch_hash": str(final_entry["batch_hash"]),
                    }
                )
            except Exception as e:
                failures.append(f"{sweep_value}: {e}")
                continue

        # Store in controller (in-memory only; deterministic).
        self.sweep_results = list(results)
        return {"results": results, "failures": failures}

    def run_grid_sweep_from_checkpoint(
        self,
        snapshot_path: str,
        log_path: str,
        resume_batch_index: int,
        param_grid: dict[str, list[float]],
        batches_per_branch: int,
    ):
        """
        Phase 4.4: deterministic multi-parameter grid sweep (Cartesian, bounded; sequential).

        Constraints:
        - up to 2 parameters
        - each value list length <= 5
        - total branches <= 25
        - batches_per_branch <= 30
        - disallow changing dt
        - do not change RNG seed/state
        """
        if not isinstance(param_grid, dict) or not param_grid:
            raise ValueError("param_grid must be a non-empty dict[str, list[float]].")
        if len(param_grid) > 2:
            raise ValueError("param_grid may contain at most 2 parameters.")
        if not isinstance(batches_per_branch, int) or batches_per_branch < 1 or batches_per_branch > 30:
            raise ValueError("batches_per_branch must be an int in [1, 30].")

        # Deterministic key ordering for Cartesian product
        keys = sorted(str(k).strip() for k in param_grid.keys())
        if any(k == "" for k in keys):
            raise ValueError("param_grid keys must be non-empty strings.")
        if any(k in ("dt", "thickness_ref", "thickness_alpha", "thickness_hash", "beta", "gamma") for k in keys):
            raise ValueError("Changing dt or thickness-derived keys is not allowed in grid sweeps.")

        values_lists: list[list[float]] = []
        for k in keys:
            vals = param_grid.get(k)
            if not isinstance(vals, list) or not vals:
                raise ValueError(f"param_grid['{k}'] must be a non-empty list of floats.")
            if len(vals) > 5:
                raise ValueError(f"param_grid['{k}'] length must be <= 5.")
            values_lists.append([float(v) for v in vals])

        total_branches = 1
        for vals in values_lists:
            total_branches *= len(vals)
        if total_branches > 25:
            raise ValueError("Total grid branches must be <= 25.")

        current = self.state.loaded_network
        if not isinstance(current, Phase1NetworkAdapter):
            raise ValueError("No loaded Research Simulation network/adapter. Load a network first.")

        results: list[dict[str, Any]] = []
        failures: list[str] = []

        # Cartesian product (deterministic nested loops for <= 2 params)
        combos: list[dict[str, float]] = []
        if len(keys) == 1:
            k0 = keys[0]
            for v0 in values_lists[0]:
                combos.append({k0: float(v0)})
        else:
            k0, k1 = keys[0], keys[1]
            for v0 in values_lists[0]:
                for v1 in values_lists[1]:
                    combos.append({k0: float(v0), k1: float(v1)})

        for combo in combos:
            try:
                forked = current.fork_from_checkpoint(snapshot_path, log_path, int(resume_batch_index))
                if forked.frozen_params is None:
                    raise ValueError("Forked adapter has no frozen_params.")

                # Validate keys exist in frozen_params and disallow dt
                for k, v in combo.items():
                    if k == "dt":
                        raise ValueError("Changing dt is not allowed.")
                    if k not in forked.frozen_params:
                        raise ValueError(f"Parameter '{k}' not present in frozen_params.")
                    forked.frozen_params[k] = float(v)
                    if hasattr(forked, k):
                        setattr(forked, k, float(v))
                    else:
                        # allow known keys that live as attributes
                        if k in ("force_alpha", "force_F0", "force_hill_n"):
                            setattr(forked, k, float(v))
                        else:
                            raise ValueError(f"Cannot apply override for '{k}' (no matching adapter attribute).")

                # Recompute provenance hash deterministically for the branch.
                frozen_json = json.dumps(forked.frozen_params, sort_keys=True)
                forked.provenance_hash = hashlib.sha256(frozen_json.encode("utf-8")).hexdigest()

                # Stamp grid metadata into future log entries (copied per entry by logger).
                forked.grid_params = dict(combo)
                forked.sweep_param = None
                forked.sweep_value = None

                # Run batches on a temporary controller so the live controller is not mutated.
                temp_ctrl = SimulationController()
                temp_ctrl.state = SimulationState(
                    loaded_network=forked,
                    strain_value=float(self.state.strain_value),
                    time=float(forked.experiment_log[-1]["time"]) if forked.experiment_log else float(self.state.time),
                    is_running=True,
                    is_paused=False,
                )
                temp_ctrl.rng = forked.rng

                temp_ctrl.run_n_batches(int(batches_per_branch))

                final_entry = forked.experiment_log[-1]
                results.append(
                    {
                        "grid_params": dict(combo),
                        "final_time": float(final_entry["time"]),
                        "final_lysis_fraction": float(final_entry["lysis_fraction"]),
                        "final_batch_hash": str(final_entry["batch_hash"]),
                    }
                )
            except Exception as e:
                failures.append(f"{combo}: {e}")
                continue

        self.grid_sweep_results = list(results)
        return {"results": results, "failures": failures}


class ResearchSimulationPage(TkinterView):
    """
    Deterministic Research Simulation (Batch-Based) GUI.
    """

    def __init__(self, view):
        Logger.log("start ResearchSimulationPage.__init__(self, view)")
        self.view = view

        # Style passthrough
        self.BG_COLOR = view.BG_COLOR
        self.FG_COLOR = view.FG_COLOR
        self.button_images = view.button_images
        self.PAGE_HEADING_FONT = view.HEADING_FONT
        self.PAGE_HEADING_BG = view.HEADING_BG
        self.PAGE_SUBHEADING_FONT = view.SUBHEADING_FONT
        self.PAGE_SUBHEADING_BG = view.SUBHEADING_BG
        self.SUBHEADING_2_FONT = view.SUBHEADING_2_FONT

        # UI state (no defaults that encode scientific assumptions)
        self.selected_network_path = tk.StringVar(value="")
        # Plasmin concentration: GUI-only parameter (never read from Excel metadata)
        self.plasmin_concentration = tk.StringVar(value="")
        self.time_step = tk.StringVar(value="")
        self.max_time = tk.StringVar(value="")
        # Applied strain is a fixed experimental parameter (read once at Start).
        self.applied_strain_fixed = tk.StringVar(value="0.0")
        # Phase 4.0: explicit multi-batch run control (validated; no automation)
        self.batches_to_run = tk.StringVar(value="1")
        # Phase 4.3: parameter sweep controls (validated; no async)
        self.sweep_param_name = tk.StringVar(value="")
        self.sweep_values_csv = tk.StringVar(value="")
        self.sweep_batches_per_branch = tk.StringVar(value="1")
        # Phase 4.4: grid sweep controls
        self.grid_batches_per_branch = tk.StringVar(value="1")
        self._grid_param_text = None

        # Controller owns state; UI may not mutate SimulationState directly.
        self.controller = SimulationController()
        # Entry widget reference for disabling after Start.
        self._applied_strain_entry = None

        # Metrics display (renders from SimulationState only)
        self.metric_time_min = tk.StringVar(value=self._format_minutes(self.controller.state.time))
        self.metric_lysis_pct = tk.StringVar(value="--")
        self.metric_active_fibers = tk.StringVar(value="--")
        self.metric_cleaved_fibers = tk.StringVar(value="--")
        self.metric_mean_tension = tk.StringVar(value="--")
        self.metric_max_tension = tk.StringVar(value="--")
        self.metric_running = tk.StringVar(value=self._format_running_state())
        self.metric_paused = tk.StringVar(value=self._format_paused_state())

        # Label widget reference for dynamic styling (force spike warning)
        self.metric_max_tension_label = None

        # Auto-pause on force spike controls
        self.enable_auto_pause = tk.BooleanVar(value=False)
        self.force_spike_threshold = tk.DoubleVar(value=1e-7)

        # Simulation speed control
        self.speed_multiplier = tk.DoubleVar(value=1.0)

        # Deterministic static "network" sketch (normalized coordinates)
        self._static_nodes = [
            (0.40, 0.30),
            (0.55, 0.25),
            (0.65, 0.35),
            (0.58, 0.55),
            (0.42, 0.60),
            (0.32, 0.45),
        ]
        self._static_edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (0, 3),
            (1, 4),
        ]

        # Canvas refs
        self._viz_canvas = None
        # Pre-load diagnostics (UI-only; no mutation of simulation state)
        self._boundary_preview: dict[str, Any] | None = None

        Logger.log("end ResearchSimulationPage.__init__(self, view)")

    def show_page(self, container):
        Logger.log("start ResearchSimulationPage.show_page(self, container)")

        # Top bar (navigation + title)
        top_bar = tk.Frame(container, bg=self.BG_COLOR)
        top_bar.pack(fill=tk.X, padx=12, pady=(12, 6))

        back_button = tk.Button(
            top_bar,
            image=self.button_images.get("Small_Left_Arrow"),
            bg=self.view.ICON_BUTTON_BG,
            border="0",
            cursor="hand2",
            command=self._on_back_to_home,
            padx=8,
            pady=8,
            activebackground=self.view.ACTIVE_BG_COLOR,
        )
        back_button.pack(side=tk.LEFT)

        title = tk.Label(
            top_bar,
            text="Research Simulation",
            foreground=self.FG_COLOR,
            font=(self.view.FONT_FAMILY, 24),
            background=self.PAGE_HEADING_BG,
        )
        title.pack(side=tk.LEFT, padx=(10, 0))

        subtitle = tk.Label(
            container,
            text="Deterministic Research Simulation (Batch-Based)",
            foreground=self.FG_COLOR,
            font=self.PAGE_SUBHEADING_FONT,
            background=self.PAGE_SUBHEADING_BG,
            wraplength=900,
            justify="left",
        )
        subtitle.pack(fill=tk.X, padx=16, pady=(0, 10))

        # Main split: left controls, right visualization
        main = tk.Frame(container, bg=self.BG_COLOR)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        main.grid_columnconfigure(0, weight=0, minsize=360)
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(0, weight=1)

        left = tk.Frame(main, bg=self.BG_COLOR)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        right = tk.Frame(main, bg=self.BG_COLOR)
        right.grid(row=0, column=1, sticky="nsew")

        # Left: sections
        self._build_network_loading_section(left)
        self._build_parameter_section(left)
        self._build_simulation_controls_section(left)
        self._build_metrics_section(left)

        # Right: visualization panel with poles + static network sketch
        self._build_visualization_section(right)

        Logger.log("end ResearchSimulationPage.show_page(self, container)")

    # ---------------------------
    # Section builders
    # ---------------------------
    def _build_network_loading_section(self, parent):
        lf = tk.LabelFrame(
            parent,
            text="Network Loading (CSV/XLSX)",
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=self.SUBHEADING_2_FONT,
            padx=10,
            pady=10,
        )
        lf.pack(fill=tk.X, pady=(0, 10))

        path_entry = tk.Entry(
            lf,
            textvariable=self.selected_network_path,
            bg="gray14",
            fg=self.FG_COLOR,
            insertbackground=self.FG_COLOR,
            relief=tk.FLAT,
        )
        path_entry.pack(fill=tk.X, pady=(0, 8))

        row = tk.Frame(lf, bg=self.BG_COLOR)
        row.pack(fill=tk.X)

        browse_btn = tk.Button(
            row,
            text="Browse…",
            bg="gray18",
            fg=self.FG_COLOR,
            activebackground="gray20",
            activeforeground=self.FG_COLOR,
            borderwidth=0,
            cursor="hand2",
            command=self._on_browse_network_file,
            padx=10,
            pady=6,
        )
        browse_btn.pack(side=tk.LEFT)

        load_btn = tk.Button(
            row,
            text="Load",
            bg="gray18",
            fg=self.FG_COLOR,
            activebackground="gray20",
            activeforeground=self.FG_COLOR,
            borderwidth=0,
            cursor="hand2",
            command=self._on_load_network_stub,
            padx=10,
            pady=6,
        )
        load_btn.pack(side=tk.LEFT, padx=(8, 0))

        preview_btn = tk.Button(
            row,
            text="Preview Boundaries",
            bg="gray18",
            fg=self.FG_COLOR,
            activebackground="gray20",
            activeforeground=self.FG_COLOR,
            borderwidth=0,
            cursor="hand2",
            command=self._on_preview_boundaries,
            padx=10,
            pady=6,
        )
        preview_btn.pack(side=tk.LEFT, padx=(8, 0))

    def _build_parameter_section(self, parent):
        lf = tk.LabelFrame(
            parent,
            text="Parameter Configuration",
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=self.SUBHEADING_2_FONT,
            padx=10,
            pady=10,
        )
        lf.pack(fill=tk.X, pady=(0, 10))

        row = tk.Frame(lf, bg=self.BG_COLOR)
        row.pack(fill=tk.X, pady=(0, 6))

        lbl = tk.Label(
            row,
            text="Applied Strain (fixed)",
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=self.SUBHEADING_2_FONT,
            width=22,
            anchor="w",
        )
        lbl.pack(side=tk.LEFT)

        self._applied_strain_entry = tk.Entry(
            row,
            textvariable=self.applied_strain_fixed,
            bg="gray14",
            fg=self.FG_COLOR,
            insertbackground=self.FG_COLOR,
            relief=tk.FLAT,
            width=12,
        )
        self._applied_strain_entry.pack(side=tk.LEFT, padx=(8, 0))

        # Plasmin concentration: GUI-controlled only (NOT read from Excel metadata)
        # Controls the cleavage rate λ₀ in Core V2 simulation
        self._kv_row(lf, "Plasmin concentration", self.plasmin_concentration)
        self._kv_row(lf, "Time step", self.time_step)
        # Max time: Simulation termination criterion (GUI-controlled, functional)
        self._kv_row(lf, "Max time", self.max_time)

    def _build_simulation_controls_section(self, parent):
        lf = tk.LabelFrame(
            parent,
            text="Simulation Controls",
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=self.SUBHEADING_2_FONT,
            padx=10,
            pady=10,
        )
        lf.pack(fill=tk.X, pady=(0, 10))

        row = tk.Frame(lf, bg=self.BG_COLOR)
        row.pack(fill=tk.X)

        start_btn = tk.Button(
            row,
            text="Start",
            bg="gray18",
            fg=self.FG_COLOR,
            activebackground="gray20",
            activeforeground=self.FG_COLOR,
            borderwidth=0,
            cursor="hand2",
            command=self._on_start,
            padx=12,
            pady=8,
        )
        start_btn.pack(side=tk.LEFT)

        pause_btn = tk.Button(
            row,
            text="Pause",
            bg="gray18",
            fg=self.FG_COLOR,
            activebackground="gray20",
            activeforeground=self.FG_COLOR,
            borderwidth=0,
            cursor="hand2",
            command=self._on_pause,
            padx=12,
            pady=8,
        )
        pause_btn.pack(side=tk.LEFT, padx=(8, 0))

        stop_btn = tk.Button(
            row,
            text="Stop",
            bg="gray18",
            fg=self.FG_COLOR,
            activebackground="gray20",
            activeforeground=self.FG_COLOR,
            borderwidth=0,
            cursor="hand2",
            command=self._on_stop,
            padx=12,
            pady=8,
        )
        stop_btn.pack(side=tk.LEFT, padx=(8, 0))

        advance_row = tk.Frame(lf, bg=self.BG_COLOR)
        advance_row.pack(fill=tk.X, pady=(10, 0))

        advance_btn = tk.Button(
            advance_row,
            text="Advance One Batch",
            bg="gray18",
            fg=self.FG_COLOR,
            activebackground="gray20",
            activeforeground=self.FG_COLOR,
            borderwidth=0,
            cursor="hand2",
            command=self._on_advance_one_batch,
            padx=12,
            pady=8,
        )
        advance_btn.pack(side=tk.LEFT)

        # Auto-pause on force spike controls
        auto_pause_frame = tk.Frame(lf, bg=self.BG_COLOR)
        auto_pause_frame.pack(fill=tk.X, pady=(10, 0))

        auto_pause_check = tk.Checkbutton(
            auto_pause_frame,
            text="Auto-pause on force spike",
            variable=self.enable_auto_pause,
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            selectcolor="gray18",
            activebackground=self.BG_COLOR,
            activeforeground=self.FG_COLOR,
        )
        auto_pause_check.pack(side=tk.LEFT)

        tk.Label(
            auto_pause_frame,
            text="Threshold:",
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=self.SUBHEADING_2_FONT,
        ).pack(side=tk.LEFT, padx=(10, 5))

        threshold_entry = tk.Entry(
            auto_pause_frame,
            textvariable=self.force_spike_threshold,
            width=10,
            bg="gray18",
            fg=self.FG_COLOR,
            insertbackground=self.FG_COLOR,
        )
        threshold_entry.pack(side=tk.LEFT)

        tk.Label(
            auto_pause_frame,
            text="N",
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=self.SUBHEADING_2_FONT,
        ).pack(side=tk.LEFT, padx=(5, 0))

        # Simulation speed control
        speed_frame = tk.Frame(lf, bg=self.BG_COLOR)
        speed_frame.pack(fill=tk.X, pady=(10, 0))

        tk.Label(
            speed_frame,
            text="Simulation Speed:",
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=self.SUBHEADING_2_FONT,
        ).pack(side=tk.LEFT)

        speed_slider = tk.Scale(
            speed_frame,
            from_=0.1,
            to=2.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.speed_multiplier,
            length=200,
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            activebackground="gray18",
            highlightthickness=0,
        )
        speed_slider.pack(side=tk.LEFT, padx=(10, 5))

        speed_value_label = tk.Label(
            speed_frame,
            textvariable=self.speed_multiplier,
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=self.SUBHEADING_2_FONT,
            width=4,
        )
        speed_value_label.pack(side=tk.LEFT)

        tk.Label(
            speed_frame,
            text="x",
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=self.SUBHEADING_2_FONT,
        ).pack(side=tk.LEFT)

        export_row = tk.Frame(lf, bg=self.BG_COLOR)
        export_row.pack(fill=tk.X, pady=(8, 0))

        export_btn = tk.Button(
            export_row,
            text="Export Experiment Log",
            bg="gray18",
            fg=self.FG_COLOR,
            activebackground="gray20",
            activeforeground=self.FG_COLOR,
            borderwidth=0,
            cursor="hand2",
            command=self._on_export_experiment_log,
            padx=12,
            pady=8,
        )
        export_btn.pack(side=tk.LEFT)

        snapshot_row = tk.Frame(lf, bg=self.BG_COLOR)
        snapshot_row.pack(fill=tk.X, pady=(8, 0))

        snapshot_btn = tk.Button(
            snapshot_row,
            text="Export Network Snapshot",
            bg="gray18",
            fg=self.FG_COLOR,
            activebackground="gray20",
            activeforeground=self.FG_COLOR,
            borderwidth=0,
            cursor="hand2",
            command=self._on_export_network_snapshot,
            padx=12,
            pady=8,
        )
        snapshot_btn.pack(side=tk.LEFT)

        fractured_row = tk.Frame(lf, bg=self.BG_COLOR)
        fractured_row.pack(fill=tk.X, pady=(8, 0))

        fractured_btn = tk.Button(
            fractured_row,
            text="Export Fractured History",
            bg="gray18",
            fg=self.FG_COLOR,
            activebackground="gray20",
            activeforeground=self.FG_COLOR,
            borderwidth=0,
            cursor="hand2",
            command=self._on_export_fractured_history,
            padx=12,
            pady=8,
        )
        fractured_btn.pack(side=tk.LEFT)

        degradation_row = tk.Frame(lf, bg=self.BG_COLOR)
        degradation_row.pack(fill=tk.X, pady=(8, 0))

        degradation_btn = tk.Button(
            degradation_row,
            text="Export Degradation Order",
            bg="gray18",
            fg=self.FG_COLOR,
            activebackground="gray20",
            activeforeground=self.FG_COLOR,
            borderwidth=0,
            cursor="hand2",
            command=self._on_export_degradation_order,
            padx=12,
            pady=8,
        )
        degradation_btn.pack(side=tk.LEFT)

        replay_row = tk.Frame(lf, bg=self.BG_COLOR)
        replay_row.pack(fill=tk.X, pady=(8, 0))

        replay_btn = tk.Button(
            replay_row,
            text="Replay Batch Check",
            bg="gray18",
            fg=self.FG_COLOR,
            activebackground="gray20",
            activeforeground=self.FG_COLOR,
            borderwidth=0,
            cursor="hand2",
            command=self._on_replay_batch_check,
            padx=12,
            pady=8,
        )
        replay_btn.pack(side=tk.LEFT)

        # Phase 4.0: bounded, explicit N-batch runner controls
        run_row = tk.Frame(lf, bg=self.BG_COLOR)
        run_row.pack(fill=tk.X, pady=(10, 0))

        run_label = tk.Label(
            run_row,
            text="Batches to Run",
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=self.SUBHEADING_2_FONT,
            anchor="w",
        )
        run_label.pack(side=tk.LEFT)

        run_entry = tk.Entry(
            run_row,
            textvariable=self.batches_to_run,
            bg="gray14",
            fg=self.FG_COLOR,
            insertbackground=self.FG_COLOR,
            relief=tk.FLAT,
            width=8,
        )
        run_entry.pack(side=tk.LEFT, padx=(10, 10))

        run_btn = tk.Button(
            run_row,
            text="Run N Batches",
            bg="gray18",
            fg=self.FG_COLOR,
            activebackground="gray20",
            activeforeground=self.FG_COLOR,
            borderwidth=0,
            cursor="hand2",
            command=self._on_run_n_batches,
            padx=12,
            pady=8,
        )
        run_btn.pack(side=tk.LEFT)

        # Phase 4.1: deterministic checkpoint resume (explicit action)
        resume_row = tk.Frame(lf, bg=self.BG_COLOR)
        resume_row.pack(fill=tk.X, pady=(10, 0))

        resume_btn = tk.Button(
            resume_row,
            text="Resume From Checkpoint",
            bg="gray18",
            fg=self.FG_COLOR,
            activebackground="gray20",
            activeforeground=self.FG_COLOR,
            borderwidth=0,
            cursor="hand2",
            command=self._on_resume_from_checkpoint,
            padx=12,
            pady=8,
        )
        resume_btn.pack(side=tk.LEFT)

        fork_row = tk.Frame(lf, bg=self.BG_COLOR)
        fork_row.pack(fill=tk.X, pady=(8, 0))

        fork_btn = tk.Button(
            fork_row,
            text="Fork From Checkpoint",
            bg="gray18",
            fg=self.FG_COLOR,
            activebackground="gray20",
            activeforeground=self.FG_COLOR,
            borderwidth=0,
            cursor="hand2",
            command=self._on_fork_from_checkpoint,
            padx=12,
            pady=8,
        )
        fork_btn.pack(side=tk.LEFT)

        # Phase 4.3: deterministic parameter sweep (fan-out)
        sweep_row = tk.Frame(lf, bg=self.BG_COLOR)
        sweep_row.pack(fill=tk.X, pady=(10, 0))

        sweep_param_label = tk.Label(
            sweep_row,
            text="Sweep Parameter",
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=self.SUBHEADING_2_FONT,
            anchor="w",
        )
        sweep_param_label.grid(row=0, column=0, sticky="w")

        sweep_param_entry = tk.Entry(
            sweep_row,
            textvariable=self.sweep_param_name,
            bg="gray14",
            fg=self.FG_COLOR,
            insertbackground=self.FG_COLOR,
            relief=tk.FLAT,
            width=18,
        )
        sweep_param_entry.grid(row=0, column=1, padx=(10, 0), sticky="w")

        sweep_vals_label = tk.Label(
            sweep_row,
            text="Sweep Values (comma-separated)",
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=self.SUBHEADING_2_FONT,
            anchor="w",
        )
        sweep_vals_label.grid(row=1, column=0, sticky="w", pady=(8, 0))

        sweep_vals_entry = tk.Entry(
            sweep_row,
            textvariable=self.sweep_values_csv,
            bg="gray14",
            fg=self.FG_COLOR,
            insertbackground=self.FG_COLOR,
            relief=tk.FLAT,
            width=28,
        )
        sweep_vals_entry.grid(row=1, column=1, padx=(10, 0), sticky="w", pady=(8, 0))

        sweep_batches_label = tk.Label(
            sweep_row,
            text="Batches / Branch",
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=self.SUBHEADING_2_FONT,
            anchor="w",
        )
        sweep_batches_label.grid(row=2, column=0, sticky="w", pady=(8, 0))

        sweep_batches_entry = tk.Entry(
            sweep_row,
            textvariable=self.sweep_batches_per_branch,
            bg="gray14",
            fg=self.FG_COLOR,
            insertbackground=self.FG_COLOR,
            relief=tk.FLAT,
            width=8,
        )
        sweep_batches_entry.grid(row=2, column=1, padx=(10, 0), sticky="w", pady=(8, 0))

        sweep_btn = tk.Button(
            sweep_row,
            text="Run Parameter Sweep",
            bg="gray18",
            fg=self.FG_COLOR,
            activebackground="gray20",
            activeforeground=self.FG_COLOR,
            borderwidth=0,
            cursor="hand2",
            command=self._on_run_parameter_sweep,
            padx=12,
            pady=8,
        )
        sweep_btn.grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))

        # Phase 4.4: grid sweep UI (JSON input)
        grid_row = tk.Frame(lf, bg=self.BG_COLOR)
        grid_row.pack(fill=tk.X, pady=(12, 0))

        grid_label = tk.Label(
            grid_row,
            text="Grid Parameters (JSON)",
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=self.SUBHEADING_2_FONT,
            anchor="w",
        )
        grid_label.pack(anchor="w")

        self._grid_param_text = tk.Text(
            grid_row,
            height=4,
            width=44,
            bg="gray14",
            fg=self.FG_COLOR,
            insertbackground=self.FG_COLOR,
            relief=tk.FLAT,
        )
        self._grid_param_text.pack(fill=tk.X, pady=(6, 0))
        # Example placeholder (deterministic string; user may overwrite)
        if self._grid_param_text.get("1.0", "end").strip() == "":
            self._grid_param_text.insert("1.0", '{"force_alpha":[0.5,1.0],"memory_mu":[0.1,0.3]}')

        grid_bp_row = tk.Frame(grid_row, bg=self.BG_COLOR)
        grid_bp_row.pack(fill=tk.X, pady=(8, 0))

        grid_bp_label = tk.Label(
            grid_bp_row,
            text="Batches / Branch",
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=self.SUBHEADING_2_FONT,
            anchor="w",
        )
        grid_bp_label.pack(side=tk.LEFT)

        grid_bp_entry = tk.Entry(
            grid_bp_row,
            textvariable=self.grid_batches_per_branch,
            bg="gray14",
            fg=self.FG_COLOR,
            insertbackground=self.FG_COLOR,
            relief=tk.FLAT,
            width=8,
        )
        grid_bp_entry.pack(side=tk.LEFT, padx=(10, 10))

        grid_btn = tk.Button(
            grid_bp_row,
            text="Run Grid Sweep",
            bg="gray18",
            fg=self.FG_COLOR,
            activebackground="gray20",
            activeforeground=self.FG_COLOR,
            borderwidth=0,
            cursor="hand2",
            command=self._on_run_grid_sweep,
            padx=12,
            pady=8,
        )
        grid_btn.pack(side=tk.LEFT)

    # Strain is a fixed experimental parameter (Applied Strain (fixed)) and no longer an interactive control.

    def _build_metrics_section(self, parent):
        lf = tk.LabelFrame(
            parent,
            text="Simulation Metrics",
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=self.SUBHEADING_2_FONT,
            padx=10,
            pady=10,
        )
        lf.pack(fill=tk.X)

        self._metric_row(lf, "Time (minutes)", self.metric_time_min)
        self._metric_row(lf, "Running", self.metric_running)
        self._metric_row(lf, "Paused", self.metric_paused)
        self._metric_row(lf, "Lysis %", self.metric_lysis_pct)
        self._metric_row(lf, "Active fibers", self.metric_active_fibers)
        self._metric_row(lf, "Cleaved fibers", self.metric_cleaved_fibers)
        self._metric_row(lf, "Mean tension", self.metric_mean_tension)

        # Max tension with dynamic color warning for force spikes
        max_tension_row = tk.Frame(lf, bg=self.BG_COLOR)
        max_tension_row.pack(fill=tk.X, pady=2)
        tk.Label(
            max_tension_row,
            text="Max tension",
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=self.SUBHEADING_2_FONT,
            anchor="w",
        ).pack(side=tk.LEFT)
        self.metric_max_tension_label = tk.Label(
            max_tension_row,
            textvariable=self.metric_max_tension,
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=self.SUBHEADING_2_FONT,
            anchor="e",
        )
        self.metric_max_tension_label.pack(side=tk.RIGHT)

    def _build_visualization_section(self, parent):
        lf = tk.LabelFrame(
            parent,
            text="Visualization",
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=self.SUBHEADING_2_FONT,
            padx=10,
            pady=10,
        )
        lf.pack(fill=tk.BOTH, expand=True)

        # Visualization mode variable
        self._viz_mode = tk.StringVar(value="strain")  # "strain" or "relaxed"

        # NOTE: Visualization mode toggle (Strain Heatmap vs Relaxed Network) temporarily hidden
        # Relaxed Network feature is preserved but UI access is disabled
        # Uncomment the section below to re-enable the toggle:
        #
        # toggle_frame = tk.Frame(lf, bg=self.BG_COLOR)
        # toggle_frame.pack(fill=tk.X, pady=(0, 5))
        #
        # tk.Radiobutton(
        #     toggle_frame,
        #     text="Strain Heatmap",
        #     variable=self._viz_mode,
        #     value="strain",
        #     command=self._on_viz_mode_change,
        #     bg=self.BG_COLOR,
        #     fg=self.FG_COLOR,
        #     selectcolor="gray14",
        #     activebackground=self.BG_COLOR,
        #     activeforeground=self.FG_COLOR,
        #     font=self.SUBHEADING_2_FONT
        # ).pack(side=tk.LEFT, padx=5)
        #
        # tk.Radiobutton(
        #     toggle_frame,
        #     text="Relaxed Network",
        #     variable=self._viz_mode,
        #     value="relaxed",
        #     command=self._on_viz_mode_change,
        #     bg=self.BG_COLOR,
        #     fg=self.FG_COLOR,
        #     selectcolor="gray14",
        #     activebackground=self.BG_COLOR,
        #     activeforeground=self.FG_COLOR,
        #     font=self.SUBHEADING_2_FONT
        # ).pack(side=tk.LEFT, padx=5)

        self._viz_canvas = tk.Canvas(
            lf,
            bg="gray8",
            highlightthickness=0,
        )
        self._viz_canvas.pack(fill=tk.BOTH, expand=True)

        # Hover tooltip for edge inspection (read-only, deterministic)
        self._tooltip_label = tk.Label(
            self._viz_canvas,
            bg="black",
            fg="white",
            font=("Consolas", 9),
            relief=tk.SOLID,
            borderwidth=1,
            padx=4,
            pady=2
        )
        self._tooltip_visible = False

        # Redraw deterministically on resize and on strain changes
        self._viz_canvas.bind("<Configure>", self._on_canvas_configure)
        self._viz_canvas.bind("<Motion>", self._on_canvas_motion)
        self._viz_canvas.bind("<Leave>", self._on_canvas_leave)
        self._redraw_visualization()

    # ---------------------------
    # Small UI helpers
    # ---------------------------
    def _kv_row(self, parent, label_text, var):
        row = tk.Frame(parent, bg=self.BG_COLOR)
        row.pack(fill=tk.X, pady=3)

        lbl = tk.Label(
            row,
            text=label_text,
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=self.SUBHEADING_2_FONT,
            width=18,
            anchor="w",
        )
        lbl.pack(side=tk.LEFT)

        entry = tk.Entry(
            row,
            textvariable=var,
            bg="gray14",
            fg=self.FG_COLOR,
            insertbackground=self.FG_COLOR,
            relief=tk.FLAT,
        )
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _metric_row(self, parent, label_text, var):
        row = tk.Frame(parent, bg=self.BG_COLOR)
        row.pack(fill=tk.X, pady=2)

        lbl = tk.Label(
            row,
            text=label_text,
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=self.SUBHEADING_2_FONT,
            anchor="w",
        )
        lbl.pack(side=tk.LEFT)

        val = tk.Label(
            row,
            textvariable=var,
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=self.SUBHEADING_2_FONT,
            anchor="e",
        )
        val.pack(side=tk.RIGHT)

    # ---------------------------
    # Callbacks (stubs)
    # ---------------------------
    def _on_back_to_home(self):
        Logger.log("ResearchSimulationPage: back to input page")
        self.view.show_page("input")

    def _on_browse_network_file(self):
        Logger.log("ResearchSimulationPage: browse network file")
        file_path = filedialog.askopenfilename(
            title="Select a Network Data File",
            filetypes=[
                ("Supported Files", "*.csv *.xlsx"),
                ("CSV Files", "*.csv"),
                ("Excel Files", "*.xlsx"),
            ],
        )
        if file_path:
            self.selected_network_path.set(file_path)

    def _on_load_network_stub(self):
        """
        Core V2: Load network using CoreV2GUIAdapter.
        """
        Logger.log("ResearchSimulationPage: load network (Core V2)")
        path = self.selected_network_path.get()

        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Please select a valid network file")
            return

        try:
            # Core V2 Integration
            adapter = CoreV2GUIAdapter()
            adapter.load_from_excel(path)

            # Store in controller state
            self.controller.state.loaded_network = adapter
            self.controller.state.time = 0.0
            self.controller.state.is_running = False
            self.controller.state.is_paused = False
            self.controller.last_metrics = None

            print(f"[Core V2] Network loaded: {len(adapter._edges_raw)} fibers")
            print(f"[Core V2] Unit scale: coord_to_m = {adapter.coord_to_m:.6e}")

            # Clear preview and render
            self._boundary_preview = None

            # Force canvas update to ensure proper dimensions before rendering
            if self._viz_canvas:
                self._viz_canvas.update_idletasks()

            self._render_from_state()

            messagebox.showinfo("Success", f"Network loaded\n{len(adapter._edges_raw)} fibers")

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load network:\n{str(e)}")
            import traceback
            traceback.print_exc()
            return

    def _on_preview_boundaries(self):
        """
        UI-only diagnostic mode:
        - Parses node coordinates from the selected file (read-only).
        - Highlights nodes within tol of x_min/x_max and indicates which are flagged.
        - Does NOT infer/reassign boundaries and does NOT modify simulation state.
        """
        p = str(self.selected_network_path.get() or "").strip()
        if p == "":
            messagebox.showerror("Preview Failed", "No file selected.")
            return
        try:
            ext = os.path.splitext(p)[1].lower()
            if ext == ".csv":
                tables = _parse_delimited_tables_from_csv(p)
            elif ext == ".xlsx":
                tables = _parse_delimited_tables_from_xlsx(p)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            if len(tables) < 1:
                raise ValueError("No tables detected in input.")
            nodes_table = tables[0]
            n_id_col = _require_column(nodes_table, ["n_id", "node_id", "id"], table_name="nodes table")
            x_col = _require_column(nodes_table, ["n_x", "x"], table_name="nodes table")
            y_col = _require_column(nodes_table, ["n_y", "y"], table_name="nodes table")

            norm_nodes = {_normalize_column_name(k): k for k in nodes_table.keys()}
            left_col = norm_nodes.get("is_left_boundary")
            right_col = norm_nodes.get("is_right_boundary")

            coords: dict[int, tuple[float, float]] = {}
            flagged_left: set[int] = set()
            flagged_right: set[int] = set()
            for i in range(len(nodes_table[n_id_col])):
                nid = _coerce_int(nodes_table[n_id_col][i])
                x = _coerce_float(nodes_table[x_col][i])
                y = _coerce_float(nodes_table[y_col][i])
                if not (np.isfinite(float(x)) and np.isfinite(float(y))):
                    continue
                coords[int(nid)] = (float(x), float(y))
                if left_col is not None:
                    if _coerce_bool_input_flag(nodes_table[left_col][i], node_id=nid, column_name="is_left_boundary"):
                        flagged_left.add(int(nid))
                if right_col is not None:
                    if _coerce_bool_input_flag(nodes_table[right_col][i], node_id=nid, column_name="is_right_boundary"):
                        flagged_right.add(int(nid))
            if not coords:
                raise ValueError("No valid node coordinates found for preview.")

            xs = [xy[0] for xy in coords.values()]
            x_min = float(min(xs))
            x_max = float(max(xs))
            x_span = float(x_max - x_min)
            tol = 1e-6 * max(1.0, abs(x_span))

            near_left = {nid for nid, (x, _y) in coords.items() if abs(float(x) - x_min) <= tol}
            near_right = {nid for nid, (x, _y) in coords.items() if abs(float(x) - x_max) <= tol}
            near_left_unflagged = sorted([nid for nid in near_left if nid not in flagged_left])
            near_right_unflagged = sorted([nid for nid in near_right if nid not in flagged_right])

            self._boundary_preview = {
                "path": p,
                "coords": coords,
                "x_min": x_min,
                "x_max": x_max,
                "tol": tol,
                "flagged_left": set(flagged_left),
                "flagged_right": set(flagged_right),
                "near_left": set(near_left),
                "near_right": set(near_right),
                "near_left_unflagged": list(near_left_unflagged),
                "near_right_unflagged": list(near_right_unflagged),
                "has_boundary_cols": (left_col is not None and right_col is not None),
            }
            self._redraw_visualization()
            msg = (
                f"Previewed boundaries for:\n{p}\n\n"
                f"x_min={x_min:.6g}, x_max={x_max:.6g}, tol={tol:.6g}\n"
                f"near_left={len(near_left)}, near_right={len(near_right)}\n"
                f"flagged_left={len(flagged_left)}, flagged_right={len(flagged_right)}\n"
                f"near_left_unflagged={near_left_unflagged[:20]}\n"
                f"near_right_unflagged={near_right_unflagged[:20]}\n"
            )
            messagebox.showinfo("Boundary Preview", msg)
        except Exception as e:
            messagebox.showerror("Preview Failed", str(e))
            return

    def _on_start(self):
        """Core V2: Start simulation with configured parameters."""
        Logger.log("ResearchSimulationPage: start (Core V2)")

        adapter = self.controller.state.loaded_network

        # Validate adapter type
        try:
            from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter
        except Exception:
            messagebox.showerror("Error", "Core V2 not available")
            return

        if adapter is None:
            messagebox.showerror("Error", "No network loaded. Load a network first.")
            return

        if not isinstance(adapter, CoreV2GUIAdapter):
            messagebox.showerror("Error", "Legacy adapter detected. Core V2 required.")
            return

        # Get parameters from GUI
        try:
            plasmin_conc = float(self.plasmin_concentration.get())
            dt = float(self.time_step.get())
            max_time = float(self.max_time.get())
            strain = float(self.applied_strain_fixed.get())
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter: {e}")
            return

        # Validate parameter bounds (prevent numerical instability)
        if dt <= 0 or dt > 0.1:
            messagebox.showerror(
                "Invalid Time Step",
                "Time step must be in (0, 0.1] seconds.\n\n"
                "Values >0.1s cause force singularities due to large position updates.\n"
                "Recommended: 0.01s for typical networks, 0.001s for high-strain cases."
            )
            return

        if plasmin_conc < 0.01 or plasmin_conc > 100:
            messagebox.showerror(
                "Invalid Plasmin Concentration",
                "Plasmin concentration should be in [0.01, 100].\n\n"
                "Values <0.01 result in extremely slow degradation.\n"
                "Values >100 may cause numerical issues in chemistry solver."
            )
            return

        if strain < 0 or strain > 2.0:
            messagebox.showerror(
                "Invalid Applied Strain",
                "Applied strain should be in [0, 2.0] (0-200%).\n\n"
                "Negative strain is non-physical.\n"
                "Strain >200% exceeds WLC model validity range."
            )
            return

        if max_time <= 0:
            messagebox.showerror(
                "Invalid Max Time",
                "Max time must be positive."
            )
            return

        # Soft warnings for unusual (but valid) parameters
        if dt < 0.001:
            result = messagebox.askyesno(
                "Very Small Timestep",
                f"Timestep dt={dt}s is very small and may slow simulation.\n"
                f"Typical value: 0.01s. Continue anyway?"
            )
            if not result:
                return

        if plasmin_conc > 10:
            result = messagebox.askyesno(
                "High Plasmin Concentration",
                f"Plasmin concentration={plasmin_conc} is very high and may cause rapid lysis.\n"
                f"Continue anyway?"
            )
            if not result:
                return

        if strain > 0.5:
            result = messagebox.askyesno(
                "High Strain Warning",
                f"Applied strain={strain*100:.0f}% is high and may cause numerical issues.\n"
                f"Monitor force metrics closely. Continue anyway?"
            )
            if not result:
                return

        # Configure and start Core V2
        try:
            adapter.configure_parameters(
                plasmin_concentration=plasmin_conc,
                time_step=dt,
                max_time=max_time,
                applied_strain=strain
            )
            adapter.start_simulation()

            self.controller.state.is_running = True
            self.controller.state.is_paused = False

            # Start non-blocking loop
            self._run_core_v2_step()

            print(f"[Core V2] Simulation started")

        except Exception as e:
            messagebox.showerror("Start Error", f"Failed to start:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _run_core_v2_step(self):
        """Non-blocking simulation loop for Core V2 (Catch C mitigation)."""
        if not self.controller.state.is_running or self.controller.state.is_paused:
            return

        adapter = self.controller.state.loaded_network
        if not isinstance(adapter, CoreV2GUIAdapter):
            return

        # Catch B: Batch physics steps per GUI frame (scaled by speed multiplier)
        batch_size = max(1, int(10 * self.speed_multiplier.get()))
        cleaved_this_batch = 0
        for _ in range(batch_size):
            continue_sim = adapter.advance_one_batch()

            if not continue_sim:
                # Simulation terminated
                self.controller.state.is_running = False
                print(f"[Core V2 Simulation] TERMINATED: {adapter.termination_reason}")
                print(f"[Core V2 Simulation] Final time: {adapter.get_current_time():.2f}s")
                print(f"[Core V2 Simulation] Clearance fraction: {adapter.get_lysis_fraction():.3f}")
                print(f"[Core V2 Simulation] Total fibers cleaved: {adapter.simulation.state.n_ruptured if adapter.simulation else 0}")

                # Final render
                self._render_core_v2_network()
                self._update_core_v2_metrics()

                messagebox.showinfo("Complete",
                    f"Simulation complete\n"
                    f"Reason: {adapter.termination_reason}\n"
                    f"Time: {adapter.get_current_time():.2f}s\n"
                    f"Clearance: {adapter.get_lysis_fraction():.1%}")
                return

        # Update GUI
        self._render_core_v2_network()
        self._update_core_v2_metrics()

        # Auto-pause on force spike check
        if self.enable_auto_pause.get():
            max_tension = adapter.get_max_tension()
            threshold = self.force_spike_threshold.get()
            if max_tension > threshold:
                self.controller.state.is_paused = True
                messagebox.showwarning(
                    "Force Spike Detected",
                    f"Simulation auto-paused due to high force:\n\n"
                    f"Max tension: {max_tension:.3e} N\n"
                    f"Threshold: {threshold:.3e} N\n"
                    f"Current time: {adapter.get_current_time():.2f}s\n\n"
                    f"Inspect network visualization and metrics.\n"
                    f"Click 'Pause' button again to resume, or adjust threshold."
                )
                return  # Stop loop

        # Heartbeat logging (every ~100 batches)
        if adapter.simulation and adapter.simulation.state.time % 1.0 < 0.01:  # Every ~1 second
            t = adapter.get_current_time()
            clearance = adapter.get_lysis_fraction()
            n_cleaved = adapter.simulation.state.n_ruptured
            print(f"[Core V2 Heartbeat] t={t:.2f}s, clearance={clearance:.1%}, cleaved={n_cleaved}")

        # Schedule next frame (Catch C: non-blocking)
        if self._viz_canvas and self._viz_canvas.winfo_exists():
            self._viz_canvas.after(0, self._run_core_v2_step)

    def _compute_strain_color(self, strain: float) -> str:
        """
        Map fiber strain to color using gradient: blue → cyan → yellow → orange → red.

        Color map:
          strain < 0.1: Blue (#4488FF)
          strain = 0.2: Cyan (#44FFFF)
          strain = 0.3: Yellow (#FFFF44)
          strain = 0.4: Orange (#FFAA00)
          strain > 0.5: Red (#FF4444)

        Uses linear interpolation between keypoints.
        """
        # Keypoints: (strain_threshold, (R, G, B))
        keypoints = [
            (0.0, (0x44, 0x88, 0xFF)),  # Blue
            (0.15, (0x44, 0xFF, 0xFF)),  # Cyan
            (0.25, (0xFF, 0xFF, 0x44)),  # Yellow
            (0.35, (0xFF, 0xAA, 0x00)),  # Orange
            (0.50, (0xFF, 0x44, 0x44)),  # Red
        ]

        # Clamp strain
        strain = max(0.0, min(strain, 1.0))

        # Find bounding keypoints
        if strain <= keypoints[0][0]:
            r, g, b = keypoints[0][1]
        elif strain >= keypoints[-1][0]:
            r, g, b = keypoints[-1][1]
        else:
            # Find the two keypoints that bound this strain
            for i in range(len(keypoints) - 1):
                s_low, (r_low, g_low, b_low) = keypoints[i]
                s_high, (r_high, g_high, b_high) = keypoints[i + 1]
                if s_low <= strain <= s_high:
                    # Linear interpolation
                    t = (strain - s_low) / (s_high - s_low)
                    r = int(r_low + t * (r_high - r_low))
                    g = int(g_low + t * (g_high - g_low))
                    b = int(b_low + t * (b_high - b_low))
                    break

        return f"#{r:02X}{g:02X}{b:02X}"

    def _render_core_v2_network(self):
        """Render network from Core V2 render data."""
        if self._viz_canvas is None:
            print("[Core V2 Render] ERROR: Canvas is None")
            return

        adapter = self.controller.state.loaded_network
        if not isinstance(adapter, CoreV2GUIAdapter):
            print("[Core V2 Render] ERROR: Adapter is not CoreV2GUIAdapter")
            return

        # Get render data (Catch B: only when actually rendering)
        render_data = adapter.get_render_data()
        nodes = render_data['nodes']  # {node_id: (x, y)} in abstract units
        edges = render_data['edges']  # [(edge_id, n_from, n_to, is_ruptured), ...]
        strains = render_data.get('strains', {})  # {fiber_id: strain} for coloring
        critical_fiber_id = render_data.get('critical_fiber_id', None)  # Fiber that triggered clearance
        plasmin_locations = render_data.get('plasmin_locations', {})  # {fiber_id: position (0-1)}

        print(f"[Core V2 Render] Render data: {len(nodes)} nodes, {len(edges)} edges, {len(plasmin_locations)} plasmin-active fibers")

        # Clear canvas
        self._viz_canvas.delete("network")

        # Get canvas dimensions for scaling
        canvas_width = self._viz_canvas.winfo_width()
        canvas_height = self._viz_canvas.winfo_height()

        print(f"[Core V2 Render] Canvas dimensions: {canvas_width}x{canvas_height}")

        if canvas_width <= 1 or canvas_height <= 1:
            print("[Core V2 Render] WARNING: Canvas too small, skipping render")
            return

        # Compute bounding box
        if not nodes:
            print("[Core V2 Render] WARNING: No nodes to render")
            return

        xs = [pos[0] for pos in nodes.values()]
        ys = [pos[1] for pos in nodes.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        x_span = x_max - x_min
        y_span = y_max - y_min

        if x_span == 0 or y_span == 0:
            return

        # Scale to canvas (with padding)
        padding = 40
        scale_x = (canvas_width - 2*padding) / x_span
        scale_y = (canvas_height - 2*padding) / y_span
        scale = min(scale_x, scale_y)

        def to_canvas(x, y):
            cx = padding + (x - x_min) * scale
            cy = canvas_height - (padding + (y - y_min) * scale)
            return cx, cy

        # Draw edges
        for edge_id, n_from, n_to, is_ruptured in edges:
            if n_from not in nodes or n_to not in nodes:
                continue

            x1, y1 = to_canvas(*nodes[n_from])
            x2, y2 = to_canvas(*nodes[n_to])

            # Check if this edge has plasmin attached
            has_plasmin = edge_id in plasmin_locations

            # Check if this is the critical fiber (triggered clearance)
            is_critical = (critical_fiber_id is not None and edge_id == critical_fiber_id)

            # Color by strain gradient (or special colors for ruptured/critical)
            if is_critical:
                color = "#FF00FF"  # Magenta (critical fiber) - highest priority
                width = 5  # Extra thick
            elif is_ruptured:
                color = "#FF4444"  # Red (ruptured)
                width = 1
            else:
                # Get strain for this fiber and compute color
                strain = strains.get(edge_id, 0.0)
                color = self._compute_strain_color(strain)
                # Highlight plasmin-active fibers with thicker width
                width = 3 if has_plasmin else 2

            self._viz_canvas.create_line(
                x1, y1, x2, y2,
                fill=color,
                width=width,
                tags="network"
            )

        # Draw plasmin dots (green dots showing enzyme locations)
        for fiber_id, position in plasmin_locations.items():
            # Find the edge corresponding to this fiber
            edge_info = None
            for edge_id, n_from, n_to, is_ruptured in edges:
                if edge_id == fiber_id:
                    edge_info = (n_from, n_to)
                    break

            if edge_info and edge_info[0] in nodes and edge_info[1] in nodes:
                n_from, n_to = edge_info
                x1, y1 = to_canvas(*nodes[n_from])
                x2, y2 = to_canvas(*nodes[n_to])

                # Interpolate position along edge (0.0 = n_from, 1.0 = n_to)
                px = x1 + position * (x2 - x1)
                py = y1 + position * (y2 - y1)

                # Draw green dot (plasmin molecule)
                dot_radius = 5
                self._viz_canvas.create_oval(
                    px - dot_radius, py - dot_radius,
                    px + dot_radius, py + dot_radius,
                    fill="#00FF00",  # Bright green
                    outline="#00AA00",  # Dark green outline
                    width=1,
                    tags="network"
                )

        # Draw nodes
        for node_id, (x, y) in nodes.items():
            cx, cy = to_canvas(x, y)
            self._viz_canvas.create_oval(
                cx-3, cy-3, cx+3, cy+3,
                fill="#CCCCCC",
                outline="",
                tags="network"
            )

        print(f"[Core V2 Render] SUCCESS: Rendered {len(edges)} edges and {len(nodes)} nodes")

    def _render_relaxed_core_v2_network(self):
        """Render mechanically relaxed network after percolation loss (Core V2)."""
        if self._viz_canvas is None:
            print("[Relaxed Render] ERROR: Canvas is None")
            return

        adapter = self.controller.state.loaded_network
        if not isinstance(adapter, CoreV2GUIAdapter):
            print("[Relaxed Render] ERROR: Adapter is not CoreV2GUIAdapter")
            return

        # Get relaxed network data
        relaxed_data = adapter.get_relaxed_network_data()

        if relaxed_data is None:
            # Percolation still intact - show message
            self._viz_canvas.delete("all")
            canvas_width = self._viz_canvas.winfo_width()
            canvas_height = self._viz_canvas.winfo_height()
            self._viz_canvas.create_text(
                canvas_width // 2,
                canvas_height // 2,
                fill="gray80",
                font=self.SUBHEADING_2_FONT,
                text="Network still percolating.\nRelaxed view available after left-right connectivity is lost.",
                justify=tk.CENTER
            )
            print("[Relaxed Render] Network still percolating - no relaxed state available")
            return

        # Extract data
        components = relaxed_data['components']
        node_positions = relaxed_data['node_positions']
        edges = relaxed_data['edges']

        print(f"[Relaxed Render] Rendering {len(components)} components, "
              f"{len(node_positions)} nodes, {len(edges)} edges")

        # Clear canvas
        self._viz_canvas.delete("all")

        # Get canvas dimensions
        canvas_width = self._viz_canvas.winfo_width()
        canvas_height = self._viz_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            print("[Relaxed Render] WARNING: Canvas too small, skipping render")
            return

        # Compute bounding box
        if not node_positions:
            print("[Relaxed Render] WARNING: No nodes to render")
            return

        xs = [pos[0] for pos in node_positions.values()]
        ys = [pos[1] for pos in node_positions.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        x_span = x_max - x_min if x_max != x_min else 1.0
        y_span = y_max - y_min if y_max != y_min else 1.0

        # Scale to canvas (with padding)
        padding = 40
        scale_x = (canvas_width - 2*padding) / x_span
        scale_y = (canvas_height - 2*padding) / y_span
        scale = min(scale_x, scale_y)

        def to_canvas(x, y):
            cx = padding + (x - x_min) * scale
            cy = canvas_height - (padding + (y - y_min) * scale)
            return cx, cy

        # Assign colors to component types
        component_colors = {
            'left_connected': '#4488FF',    # Blue
            'right_connected': '#FF4444',   # Red
            'isolated': '#44AA44',          # Green
            'spanning': '#FFAA00'           # Orange (shouldn't happen)
        }

        # Build component lookup for edges
        edge_to_component = {}
        for component in components:
            for edge_id in component.edge_ids:
                edge_to_component[edge_id] = component

        # Draw edges (fibers) grouped by component
        for edge_data in edges:
            edge_id = edge_data['edge_id']
            n_from = edge_data['n_from']
            n_to = edge_data['n_to']

            if n_from not in node_positions or n_to not in node_positions:
                continue

            x1, y1 = to_canvas(*node_positions[n_from])
            x2, y2 = to_canvas(*node_positions[n_to])

            # Determine color based on component type
            component = edge_to_component.get(edge_id)
            if component:
                color = component_colors.get(component.component_type, '#888888')
            else:
                color = '#888888'  # Gray for orphan edges

            self._viz_canvas.create_line(
                x1, y1, x2, y2,
                fill=color,
                width=2,
                tags="network"
            )

        # Draw nodes (crosslinks)
        for component in components:
            for node_id in component.node_ids:
                if node_id not in node_positions:
                    continue

                cx, cy = to_canvas(*node_positions[node_id])

                # Node styling based on type
                if node_id in component.fixed_nodes:
                    # Fixed boundary node
                    self._viz_canvas.create_oval(
                        cx-5, cy-5, cx+5, cy+5,
                        fill='#FF6600',  # Orange
                        outline='#CC4400',
                        width=2,
                        tags="network"
                    )
                else:
                    # Free node
                    self._viz_canvas.create_oval(
                        cx-3, cy-3, cx+3, cy+3,
                        fill='#CCCCCC',
                        outline='',
                        tags="network"
                    )

        # Draw legend
        legend_x = padding
        legend_y = padding
        legend_line_height = 20

        self._viz_canvas.create_text(
            legend_x, legend_y,
            anchor='nw',
            fill='white',
            font=('Arial', 10, 'bold'),
            text='Relaxed Network (Post-Percolation)'
        )

        legend_y += legend_line_height

        # Component type legend
        legend_items = [
            ('Left-connected', component_colors['left_connected']),
            ('Right-connected', component_colors['right_connected']),
            ('Isolated', component_colors['isolated']),
        ]

        for label, color in legend_items:
            legend_y += legend_line_height
            # Draw color sample line
            self._viz_canvas.create_line(
                legend_x, legend_y,
                legend_x + 30, legend_y,
                fill=color,
                width=3
            )
            # Draw label
            self._viz_canvas.create_text(
                legend_x + 35, legend_y,
                anchor='w',
                fill='white',
                font=('Arial', 9),
                text=label
            )

        legend_y += legend_line_height + 5

        # Node type legend
        self._viz_canvas.create_oval(
            legend_x, legend_y - 4,
            legend_x + 8, legend_y + 4,
            fill='#FF6600',
            outline='#CC4400',
            width=2
        )
        self._viz_canvas.create_text(
            legend_x + 12, legend_y,
            anchor='w',
            fill='white',
            font=('Arial', 9),
            text='Boundary node (fixed)'
        )

        legend_y += legend_line_height
        self._viz_canvas.create_oval(
            legend_x, legend_y - 3,
            legend_x + 6, legend_y + 3,
            fill='#CCCCCC',
            outline=''
        )
        self._viz_canvas.create_text(
            legend_x + 12, legend_y,
            anchor='w',
            fill='white',
            font=('Arial', 9),
            text='Crosslink node (free)'
        )

        print(f"[Relaxed Render] SUCCESS: Rendered relaxed network with {len(components)} components")

    def _update_core_v2_metrics(self):
        """Update metrics display from Core V2."""
        adapter = self.controller.state.loaded_network
        if not isinstance(adapter, CoreV2GUIAdapter):
            return

        t = adapter.get_current_time()
        lysis = adapter.get_lysis_fraction()

        self.metric_time_min.set(self._format_minutes(t))
        self.metric_lysis_pct.set(f"{lysis*100:.1f}%")

        if adapter.simulation:
            n_ruptured = adapter.simulation.state.n_ruptured
            n_total = len(adapter.simulation.state.fibers)
            self.metric_active_fibers.set(str(n_total - n_ruptured))
            self.metric_cleaved_fibers.set(str(n_ruptured))

        self.metric_running.set(self._format_running_state())
        self.metric_paused.set(self._format_paused_state())

        # Update tension metrics with force spike warning
        mean_tension = adapter.prev_mean_tension if adapter.prev_mean_tension is not None else 0.0
        max_tension = adapter.get_max_tension()

        self.metric_mean_tension.set(f"{mean_tension:.2e}")
        self.metric_max_tension.set(f"{max_tension:.2e}")

        # Color warning: red if max tension exceeds threshold
        if max_tension > 1e-7 and self.metric_max_tension_label:
            self.metric_max_tension_label.config(fg="red", font=(self.SUBHEADING_2_FONT[0], self.SUBHEADING_2_FONT[1], "bold"))
        elif self.metric_max_tension_label:
            self.metric_max_tension_label.config(fg=self.FG_COLOR, font=self.SUBHEADING_2_FONT)

    def _on_pause(self):
        """Core V2: Pause simulation."""
        Logger.log("ResearchSimulationPage: pause (Core V2)")
        if not self.controller.state.is_running:
            return

        self.controller.state.is_paused = True
        print("[Core V2] Paused")
        self._update_core_v2_metrics()

    def _on_resume(self):
        """Core V2: Resume simulation."""
        Logger.log("ResearchSimulationPage: resume (Core V2)")
        if not self.controller.state.is_running:
            return

        self.controller.state.is_paused = False
        print("[Core V2] Resumed")
        self._run_core_v2_step()

    def _on_stop(self):
        """
        Stop only mutates SimulationState.
        - Marks not running, not paused, and resets time to 0 seconds.
        """
        Logger.log("ResearchSimulationPage: stop (controller)")
        self.controller.stop()
        self._render_from_state()

    def _on_advance_one_batch(self):
        """
        One click = exactly one batch. No looping, timers, or scheduling.
        Preconditions and failures are shown as user-visible message boxes.
        """
        try:
            self.controller.advance_one_batch()
        except Exception as e:
            messagebox.showerror("Advance One Batch Failed", str(e))
            return
        self._render_from_state()

        # Phase 3.8: non-blocking warning dialog for hard performance threshold (diagnostic only).
        hard_warn_sec = 2.0
        dur = getattr(self.controller, "last_batch_duration_sec", None)
        if dur is not None and float(dur) > hard_warn_sec:
            win = tk.Toplevel(self.view.root)
            win.title("Performance Warning")
            win.configure(bg=self.BG_COLOR)
            win.transient(self.view.root)
            # Non-modal (no grab_set): user may ignore/continue.
            msg = tk.Label(
                win,
                text=f"Batch duration {float(dur):.3f}s exceeded hard threshold {hard_warn_sec:.3f}s.\n\n"
                     "This is a diagnostic warning only; results are unchanged.",
                bg=self.BG_COLOR,
                fg=self.FG_COLOR,
                font=self.SUBHEADING_2_FONT,
                justify="left",
                wraplength=420,
            )
            msg.pack(padx=14, pady=(14, 10))
            ok = tk.Button(
                win,
                text="OK",
                bg="gray18",
                fg=self.FG_COLOR,
                activebackground="gray20",
                activeforeground=self.FG_COLOR,
                borderwidth=0,
                cursor="hand2",
                command=win.destroy,
                padx=12,
                pady=8,
            )
            ok.pack(pady=(0, 14))

    def _on_run_n_batches(self):
        """
        Explicit user action: run a bounded number of batches (Phase 4.0).
        No timers/threads; this is a synchronous sequence of advance_one_batch() calls.
        """
        n_max = 100
        try:
            n = int(str(self.batches_to_run.get()).strip())
        except Exception:
            messagebox.showerror("Run N Batches Failed", "Batches to Run must be an integer.")
            return
        if n < 1 or n > n_max:
            messagebox.showerror("Run N Batches Failed", f"Batches to Run must be between 1 and {n_max}.")
            return

        try:
            completed = self.controller.run_n_batches(n)
        except Exception as e:
            messagebox.showerror("Run N Batches Failed", str(e))
            self._render_from_state()
            return

        self._render_from_state()
        # Completion dialog (deterministic values from current state/log).
        adapter = self.controller.state.loaded_network
        lysis_fraction = None
        if isinstance(adapter, Phase1NetworkAdapter) and getattr(adapter, "experiment_log", None):
            lysis_fraction = adapter.experiment_log[-1].get("lysis_fraction")
        messagebox.showinfo(
            "Run Complete",
            f"Batches completed: {completed}\n"
            f"Final time: {float(self.controller.state.time):.6f}\n"
            f"Final lysis_fraction: {float(lysis_fraction) if lysis_fraction is not None else 0.0:.12f}",
        )

    def _on_resume_from_checkpoint(self):
        """
        Explicit user action: resume deterministically from a checkpoint (snapshot + log + batch index).
        """
        snapshot_path = filedialog.askopenfilename(
            title="Select Network Snapshot (JSON)",
            filetypes=[("JSON", "*.json")],
        )
        if not snapshot_path:
            return

        log_path = filedialog.askopenfilename(
            title="Select Experiment Log (CSV or JSON)",
            filetypes=[("CSV", "*.csv"), ("JSON", "*.json")],
        )
        if not log_path:
            return

        idx = simpledialog.askinteger(
            "Resume Batch Index",
            "Enter resume batch index (0-based):",
            parent=self.view.root,
            minvalue=0,
        )
        if idx is None:
            return

        try:
            info = self.controller.resume_from_checkpoint(snapshot_path, log_path, int(idx))
        except Exception as e:
            messagebox.showerror("Resume Failed", str(e))
            return

        # Sync applied strain display deterministically and re-render.
        self.applied_strain_fixed.set(str(float(self.controller.state.strain_value)))
        self._render_from_state()

        messagebox.showinfo(
            "Resume Successful",
            f"Resumed at batch_index={int(info['resume_batch_index'])}\n"
            f"time={float(info['resume_time']):.6f}\n"
            f"strain={float(info['resume_strain']):.6f}",
        )

    def _on_fork_from_checkpoint(self):
        """
        Explicit user action: fork a new branch from a checkpoint (snapshot + log + batch index).
        """
        snapshot_path = filedialog.askopenfilename(
            title="Select Network Snapshot (JSON)",
            filetypes=[("JSON", "*.json")],
        )
        if not snapshot_path:
            return

        log_path = filedialog.askopenfilename(
            title="Select Experiment Log (CSV or JSON)",
            filetypes=[("CSV", "*.csv"), ("JSON", "*.json")],
        )
        if not log_path:
            return

        idx = simpledialog.askinteger(
            "Fork Batch Index",
            "Enter fork batch index (0-based):",
            parent=self.view.root,
            minvalue=0,
        )
        if idx is None:
            return

        try:
            info = self.controller.fork_from_checkpoint(snapshot_path, log_path, int(idx))
        except Exception as e:
            messagebox.showerror("Fork Failed", str(e))
            return

        # Sync applied strain display deterministically and re-render.
        self.applied_strain_fixed.set(str(float(self.controller.state.strain_value)))
        self._render_from_state()

        messagebox.showinfo(
            "Fork Successful",
            f"Forked from batch {int(info['fork_parent_batch_index'])}; new branch begins at batch {int(info['fork_begin_batch_index'])}\n"
            f"time={float(info['time']):.6f}\n"
            f"strain={float(info['strain']):.6f}",
        )

    def _on_run_parameter_sweep(self):
        """
        Explicit user action: run a bounded parameter sweep from a checkpoint (sequential fan-out).
        """
        snapshot_path = filedialog.askopenfilename(
            title="Select Network Snapshot (JSON)",
            filetypes=[("JSON", "*.json")],
        )
        if not snapshot_path:
            return

        log_path = filedialog.askopenfilename(
            title="Select Experiment Log (CSV or JSON)",
            filetypes=[("CSV", "*.csv"), ("JSON", "*.json")],
        )
        if not log_path:
            return

        idx = simpledialog.askinteger(
            "Sweep Parent Batch Index",
            "Enter parent batch index (0-based):",
            parent=self.view.root,
            minvalue=0,
        )
        if idx is None:
            return

        param_name = str(self.sweep_param_name.get()).strip()
        if param_name == "":
            messagebox.showerror("Sweep Failed", "Sweep Parameter is required.")
            return

        raw_vals = [s.strip() for s in str(self.sweep_values_csv.get()).split(",") if s.strip() != ""]
        if not raw_vals:
            messagebox.showerror("Sweep Failed", "Provide at least one sweep value.")
            return
        if len(raw_vals) > 10:
            messagebox.showerror("Sweep Failed", "At most 10 sweep values are allowed.")
            return

        try:
            values = [float(v) for v in raw_vals]
        except Exception:
            messagebox.showerror("Sweep Failed", "Sweep values must be comma-separated numbers.")
            return

        try:
            batches_per_branch = int(str(self.sweep_batches_per_branch.get()).strip())
        except Exception:
            messagebox.showerror("Sweep Failed", "Batches / Branch must be an integer.")
            return

        try:
            out = self.controller.run_parameter_sweep_from_checkpoint(
                snapshot_path=snapshot_path,
                log_path=log_path,
                resume_batch_index=int(idx),
                param_name=param_name,
                values=values,
                batches_per_branch=batches_per_branch,
            )
        except Exception as e:
            messagebox.showerror("Sweep Failed", str(e))
            return

        results = out.get("results", [])
        failures = out.get("failures", [])
        if results:
            lysis_vals = [float(r["final_lysis_fraction"]) for r in results]
            summary = (
                f"Branches completed: {len(results)}\n"
                f"Final lysis_fraction range: [{min(lysis_vals):.6f}, {max(lysis_vals):.6f}]\n"
            )
        else:
            summary = "No branches completed.\n"

        if failures:
            summary += "\nFailures:\n" + "\n".join(str(x) for x in failures)

        messagebox.showinfo("Sweep Complete", summary)

    def _on_run_grid_sweep(self):
        """
        Explicit user action: run a bounded Cartesian grid sweep from a checkpoint (sequential).
        """
        snapshot_path = filedialog.askopenfilename(
            title="Select Network Snapshot (JSON)",
            filetypes=[("JSON", "*.json")],
        )
        if not snapshot_path:
            return

        log_path = filedialog.askopenfilename(
            title="Select Experiment Log (CSV or JSON)",
            filetypes=[("CSV", "*.csv"), ("JSON", "*.json")],
        )
        if not log_path:
            return

        idx = simpledialog.askinteger(
            "Grid Parent Batch Index",
            "Enter parent batch index (0-based):",
            parent=self.view.root,
            minvalue=0,
        )
        if idx is None:
            return

        if self._grid_param_text is None:
            messagebox.showerror("Grid Sweep Failed", "Grid parameter input is not available.")
            return

        raw = self._grid_param_text.get("1.0", "end").strip()
        try:
            param_grid = json.loads(raw)
        except Exception:
            messagebox.showerror("Grid Sweep Failed", "Grid Parameters must be valid JSON.")
            return

        try:
            batches_per_branch = int(str(self.grid_batches_per_branch.get()).strip())
        except Exception:
            messagebox.showerror("Grid Sweep Failed", "Batches / Branch must be an integer.")
            return

        try:
            out = self.controller.run_grid_sweep_from_checkpoint(
                snapshot_path=snapshot_path,
                log_path=log_path,
                resume_batch_index=int(idx),
                param_grid=param_grid,
                batches_per_branch=batches_per_branch,
            )
        except Exception as e:
            messagebox.showerror("Grid Sweep Failed", str(e))
            return

        results = out.get("results", [])
        failures = out.get("failures", [])
        if results:
            lysis_vals = [float(r["final_lysis_fraction"]) for r in results]
            summary = (
                f"Branches completed: {len(results)}\n"
                f"Final lysis_fraction range: [{min(lysis_vals):.6f}, {max(lysis_vals):.6f}]\n"
            )
        else:
            summary = "No branches completed.\n"

        if failures:
            summary += "\nFailures:\n" + "\n".join(str(x) for x in failures)

        messagebox.showinfo("Grid Sweep Complete", summary)

    def _on_export_experiment_log(self):
        """
        Explicit user action: export Phase 3.1 experiment log to CSV or JSON.
        No simulation side effects.
        """
        adapter = self.controller.state.loaded_network
        if not isinstance(adapter, Phase1NetworkAdapter):
            messagebox.showerror("Export Failed", "No loaded Research Simulation network/adapter.")
            return
        if not getattr(adapter, "experiment_log", None):
            messagebox.showerror("Export Failed", "Experiment log is empty. Run at least one successful batch before exporting.")
            return

        path = filedialog.asksaveasfilename(
            title="Export Experiment Log",
            defaultextension=".csv",
            filetypes=[
                ("CSV", "*.csv"),
                ("JSON", "*.json"),
            ],
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".csv":
                adapter.export_experiment_log_csv(path)
            elif ext == ".json":
                adapter.export_experiment_log_json(path)
            else:
                messagebox.showerror("Export Failed", "Unsupported export type. Choose a .csv or .json filename.")
                return
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))
            return

    def _on_export_network_snapshot(self):
        """
        Explicit user action: export current network snapshot (nodes + edges) as JSON,
        or export per-edge lysis metadata as CSV (Stage 4).
        No simulation side effects.
        """
        adapter = self.controller.state.loaded_network
        if not isinstance(adapter, Phase1NetworkAdapter):
            messagebox.showerror("Export Failed", "No loaded Research Simulation network/adapter.")
            return
        if adapter._relaxed_node_coords is None:
            messagebox.showerror("Export Failed", "No relaxed geometry available. Run relaxation before exporting a snapshot.")
            return

        path = filedialog.asksaveasfilename(
            title="Export Network Snapshot",
            defaultextension=".json",
            filetypes=[
                ("JSON", "*.json"),
                ("CSV (Edge Lysis)", "*.csv"),
            ],
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".json":
                adapter.export_network_snapshot(path)
            elif ext == ".csv":
                adapter.export_edge_lysis_csv(path)
            else:
                messagebox.showerror("Export Failed", "Snapshot export must be a .json or .csv file.")
                return
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))
            return

    def _on_export_fractured_history(self):
        """
        Explicit user action: export fractured edge history to CSV.
        No simulation side effects.
        """
        adapter = self.controller.state.loaded_network
        if not isinstance(adapter, Phase1NetworkAdapter):
            messagebox.showerror("Export Failed", "No loaded Research Simulation network/adapter.")
            return
        if not getattr(adapter, "fractured_history", None):
            messagebox.showerror("No Fractures", "No fractured edges to export for this simulation.")
            return

        path = filedialog.asksaveasfilename(
            title="Export Fractured History",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return

        try:
            adapter.export_fractured_history_csv(path)
            messagebox.showinfo("Export Complete", f"Fractured history saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))
            return

    def _on_export_degradation_order(self):
        """
        Core V2: Export degradation order (sequence of fiber cleavages).
        Includes time, fiber ID, strain, and node endpoints for research analysis.
        """
        adapter = self.controller.state.loaded_network

        # Check if Core V2 adapter
        try:
            from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter
        except Exception:
            messagebox.showerror("Export Failed", "Core V2 not available")
            return

        if not isinstance(adapter, CoreV2GUIAdapter):
            messagebox.showerror("Export Failed", "This export is only available for Core V2 simulations")
            return

        if adapter.simulation is None:
            messagebox.showerror("Export Failed", "No simulation run yet. Start simulation first.")
            return

        path = filedialog.asksaveasfilename(
            title="Export Degradation Order",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return

        try:
            adapter.export_degradation_history(path)
            messagebox.showinfo("Export Complete", f"Degradation history saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))
            return

    def _on_replay_batch_check(self):
        """
        Explicit validation action: replay a single batch from a saved snapshot and compare to
        the latest experiment_log entry within tolerance.
        """
        adapter = self.controller.state.loaded_network
        if not isinstance(adapter, Phase1NetworkAdapter):
            messagebox.showerror("Replay Failed", "No loaded Research Simulation network/adapter.")
            return
        if not getattr(adapter, "experiment_log", None):
            messagebox.showerror("Replay Failed", "No experiment log entries. Run at least one successful batch first.")
            return

        snapshot_path = filedialog.askopenfilename(
            title="Select Network Snapshot (JSON)",
            filetypes=[("JSON", "*.json")],
        )
        if not snapshot_path:
            return

        try:
            result = adapter.replay_single_batch(snapshot_path)
        except Exception as e:
            messagebox.showerror("Replay Failed", str(e))
            return

        expected = adapter.experiment_log[-1]
        exp_newly = int(expected["newly_cleaved"])
        exp_mean = float(expected["mean_tension"])
        exp_lysis = float(expected["lysis_fraction"])

        got_newly = int(result["newly_cleaved"])
        got_mean = float(result["mean_tension"])
        got_lysis = float(result["lysis_fraction"])

        tol_mean = 1e-6
        tol_lysis = 1e-8

        ok_newly = (got_newly == exp_newly)
        ok_mean = abs(got_mean - exp_mean) <= tol_mean
        ok_lysis = abs(got_lysis - exp_lysis) <= tol_lysis

        if ok_newly and ok_mean and ok_lysis:
            messagebox.showinfo(
                "Replay Check Passed",
                "Replay matched the latest batch log entry within tolerance.",
            )
            return

        messagebox.showerror(
            "Replay Check Failed",
            "Replay did not match latest batch log entry.\n\n"
            f"newly_cleaved: expected {exp_newly}, got {got_newly}\n"
            f"mean_tension: expected {exp_mean:.10f}, got {got_mean:.10f} (tol {tol_mean})\n"
            f"lysis_fraction: expected {exp_lysis:.12f}, got {got_lysis:.12f} (tol {tol_lysis})\n",
        )

    def _on_canvas_configure(self, _event):
        self._redraw_visualization()

    def _on_canvas_motion(self, event):
        """
        Hover tooltip for edge inspection (read-only, deterministic).
        Displays edge_id, S_eff, tension, and segment damage if available.
        """
        adapter = self.controller.state.loaded_network
        if not isinstance(adapter, Phase1NetworkAdapter) or not adapter.edges:
            self._hide_tooltip()
            return

        # Get relaxed coordinates for edge lookup
        if adapter._relaxed_node_coords is None:
            self._hide_tooltip()
            return

        render_coords = dict(adapter._relaxed_node_coords)

        # Update boundary node positions with grips
        if adapter.left_grip_x is not None and adapter.right_grip_x is not None:
            for nid in adapter.left_boundary_node_ids:
                if nid in render_coords:
                    render_coords[nid] = (adapter.left_grip_x, render_coords[nid][1])
            for nid in adapter.right_boundary_node_ids:
                if nid in render_coords:
                    render_coords[nid] = (adapter.right_grip_x, render_coords[nid][1])

        # Find canvas-to-world transform (inverse of to_canvas in _redraw_visualization)
        w = int(self._viz_canvas.winfo_width())
        h = int(self._viz_canvas.winfo_height())
        if w <= 2 or h <= 2:
            self._hide_tooltip()
            return

        xs = [xy[0] for xy in render_coords.values()]
        ys = [xy[1] for xy in render_coords.values()]
        if not xs or not ys:
            self._hide_tooltip()
            return

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        pad = max(20, int(0.06 * min(w, h)))
        span_x = (x_max - x_min) if (x_max - x_min) != 0 else 1.0
        span_y = (y_max - y_min) if (y_max - y_min) != 0 else 1.0

        # Mouse position in canvas coordinates
        mx = event.x
        my = event.y

        # Find nearest edge to mouse cursor
        nearest_edge = None
        min_dist = float('inf')
        threshold_px = 10  # Hover sensitivity

        for e in adapter.edges:
            a = render_coords.get(e.n_from)
            b = render_coords.get(e.n_to)
            if a is None or b is None:
                continue

            # Convert world coords to canvas coords
            ax = int(pad + (a[0] - x_min) / span_x * (w - 2 * pad))
            ay = int(pad + (a[1] - y_min) / span_y * (h - 2 * pad))
            bx = int(pad + (b[0] - x_min) / span_x * (w - 2 * pad))
            by = int(pad + (b[1] - y_min) / span_y * (h - 2 * pad))

            # Distance from mouse to edge (point-to-line-segment distance)
            dist = self._point_to_segment_distance(mx, my, ax, ay, bx, by)
            if dist < min_dist:
                min_dist = dist
                nearest_edge = e

        if nearest_edge is not None and min_dist < threshold_px:
            # Build tooltip text
            edge_id = nearest_edge.edge_id
            S_eff = float(nearest_edge.S)
            tension = adapter._forces_by_edge_id.get(edge_id, 0.0)

            tooltip_lines = [
                f"Edge ID: {edge_id}",
                f"S_eff: {S_eff:.3f}",
                f"Tension: {tension:.2e} N"
            ]

            # Add segment damage info if in spatial mode
            if FeatureFlags.USE_SPATIAL_PLASMIN and nearest_edge.segments:
                N_pf = float(adapter.spatial_plasmin_params.get("N_pf", 50)) if adapter.spatial_plasmin_params else 50.0
                min_integrity = min(float(seg.n_i) / N_pf for seg in nearest_edge.segments)
                tooltip_lines.append(f"Min n/N_pf: {min_integrity:.3f}")

            tooltip_text = "\n".join(tooltip_lines)
            self._show_tooltip(mx, my, tooltip_text)
        else:
            self._hide_tooltip()

    def _on_canvas_leave(self, _event):
        """Hide tooltip when mouse leaves canvas."""
        self._hide_tooltip()

    def _point_to_segment_distance(self, px, py, ax, ay, bx, by):
        """
        Calculate minimum distance from point (px, py) to line segment (ax,ay)-(bx,by).
        Pure geometry calculation (read-only).
        """
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            # Degenerate segment (point)
            return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5

        # Parameter t for closest point on infinite line
        t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))

        # Closest point on segment
        closest_x = ax + t * dx
        closest_y = ay + t * dy

        # Distance to closest point
        return ((px - closest_x) ** 2 + (py - closest_y) ** 2) ** 0.5

    def _show_tooltip(self, x, y, text):
        """Display tooltip at canvas position (x, y)."""
        self._tooltip_label.config(text=text)
        self._tooltip_label.place(x=x + 10, y=y + 10)
        self._tooltip_visible = True

    def _hide_tooltip(self):
        """Hide tooltip."""
        if self._tooltip_visible:
            self._tooltip_label.place_forget()
            self._tooltip_visible = False

    # Strain is fixed: no interactive strain callback.

    # ---------------------------
    # Visualization (deterministic)
    # ---------------------------
    def _on_viz_mode_change(self):
        """Handle visualization mode toggle change."""
        Logger.log(f"Visualization mode changed to: {self._viz_mode.get()}")
        self._redraw_visualization()

    def _redraw_visualization(self):
        if self._viz_canvas is None:
            return

        w = int(self._viz_canvas.winfo_width())
        h = int(self._viz_canvas.winfo_height())
        if w <= 2 or h <= 2:
            return

        self._viz_canvas.delete("all")

        pole_w = max(10, int(0.02 * w))
        pole_h = max(50, int(0.60 * h))
        pole_y0 = int(0.5 * h - 0.5 * pole_h)
        pole_y1 = int(0.5 * h + 0.5 * pole_h)

        # Network visualization:
        # - If a Phase1NetworkAdapter is loaded, render the imported network deterministically.
        # - If a CoreV2GUIAdapter is loaded, delegate to Core V2 renderer.
        # - Otherwise fall back to the static placeholder sketch.
        adapter = self.controller.state.loaded_network

        # Core V2 integration: check visualization mode and delegate to appropriate renderer
        try:
            from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter
            if isinstance(adapter, CoreV2GUIAdapter):
                # Check visualization mode
                viz_mode = self._viz_mode.get() if hasattr(self, '_viz_mode') else "strain"

                if viz_mode == "relaxed":
                    # Render relaxed network (post-percolation)
                    self._render_relaxed_core_v2_network()
                else:
                    # Render strain heatmap (default)
                    self._render_core_v2_network()
                return
        except Exception as e:
            # If Core V2 import fails, continue with legacy rendering
            print(f"[Visualization] Core V2 rendering failed: {e}")
            pass

        # Pre-load boundary diagnostics (UI-only): show preview if present and no adapter is loaded.
        if (not isinstance(adapter, Phase1NetworkAdapter)) and isinstance(self._boundary_preview, dict):
            data = self._boundary_preview
            coords: dict[int, tuple[float, float]] = dict(data.get("coords", {}))
            if coords:
                xs = [xy[0] for xy in coords.values()]
                ys = [xy[1] for xy in coords.values()]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                pad = max(20, int(0.06 * min(w, h)))
                span_x = (x_max - x_min) if (x_max - x_min) != 0 else 1.0
                span_y = (y_max - y_min) if (y_max - y_min) != 0 else 1.0

                def to_canvas(pt):
                    x, y = pt
                    cx = pad + (x - x_min) / span_x * (w - 2 * pad)
                    cy = pad + (y - y_min) / span_y * (h - 2 * pad)
                    return int(cx), int(cy)

                # Extremes (diagnostic only): draw dashed vertical lines at x_min/x_max.
                left_x, _ = to_canvas((float(data.get("x_min", x_min)), y_min))
                right_x, _ = to_canvas((float(data.get("x_max", x_max)), y_min))
                self._viz_canvas.create_line(left_x, pole_y0, left_x, pole_y1, fill="red3", width=3, dash=(6, 4))
                self._viz_canvas.create_line(right_x, pole_y0, right_x, pole_y1, fill="red3", width=3, dash=(6, 4))

                flagged_left = set(data.get("flagged_left", set()))
                flagged_right = set(data.get("flagged_right", set()))
                near_left = set(data.get("near_left", set()))
                near_right = set(data.get("near_right", set()))
                # Nodes: color-code
                r = max(3, int(0.008 * min(w, h)))
                for nid, (x, y) in coords.items():
                    px, py = to_canvas((float(x), float(y)))
                    outline = "gray70"
                    fill = "white"
                    if int(nid) in flagged_left:
                        outline = "gold"
                    elif int(nid) in flagged_right:
                        outline = "orange"
                    elif int(nid) in near_left or int(nid) in near_right:
                        outline = "red"
                    self._viz_canvas.create_oval(px - r, py - r, px + r, py + r, fill=fill, outline=outline, width=2)
                # Small legend (diagnostic only)
                self._viz_canvas.create_text(
                    pad,
                    pad,
                    anchor="nw",
                    fill="gray80",
                    font=self.SUBHEADING_2_FONT,
                    text="Preview: gold=flagged left, orange=flagged right, red=near-boundary unflagged",
                )
                return

        if isinstance(adapter, Phase1NetworkAdapter) and adapter.edges:
            # Render semantics:
            # - Poles are x-constraint manifolds (vertical lines), not connectors.
            # - No lines are drawn between nodes and poles.
            # - Boundary nodes are rendered at x = grip_x with current y preserved (y is unconstrained).
            # After Start (parameters frozen), render from solver coordinates ONLY (no fallbacks).
            if adapter.frozen_params is not None:
                if adapter._relaxed_node_coords is None or adapter.left_grip_x is None or adapter.right_grip_x is None:
                    self._viz_canvas.create_text(
                        int(0.5 * w),
                        int(0.5 * h),
                        fill="gray80",
                        font=self.SUBHEADING_2_FONT,
                        text="Missing relaxed geometry / grips.\nLoad + Start, then relax.",
                    )
                    return
                x_left_pole = float(adapter.left_grip_x)
                x_right_pole = float(adapter.right_grip_x)
                relaxed_coords = dict(adapter._relaxed_node_coords)
            else:
                strain = float(self.controller.state.strain_value)
                # Use frozen rigid grips when available; otherwise derive provisional grips deterministically.
                if (getattr(adapter, "left_grip_x", None) is not None) and (getattr(adapter, "right_grip_x", None) is not None):
                    x_left_pole = float(getattr(adapter, "left_grip_x"))
                    x_right_pole = float(getattr(adapter, "right_grip_x"))
                else:
                    left_ids = [int(x) for x in sorted(adapter.left_boundary_node_ids)]
                    right_ids = [int(x) for x in sorted(adapter.right_boundary_node_ids)]
                    left_xs0 = [float(adapter._initial_node_coords[nid][0]) for nid in left_ids]
                    right_xs0 = [float(adapter._initial_node_coords[nid][0]) for nid in right_ids]
                    left_grip_x0 = float(_median(left_xs0))
                    right_grip_x0 = float(_median(right_xs0))
                    base_width = float(right_grip_x0 - left_grip_x0)
                    base_width = base_width if base_width != 0.0 else 1.0
                    x_left_pole = float(left_grip_x0)
                    x_right_pole = float(right_grip_x0) + strain * float(base_width)

                # Prefer relaxed coords if available; otherwise use imported coords (pre-Start only).
                relaxed_coords = dict(adapter._relaxed_node_coords or adapter.node_coords)

            render_coords: dict[Any, tuple[float, float]] = {}
            for nid, (x, y) in relaxed_coords.items():
                if nid in adapter.left_boundary_node_ids:
                    # Boundary: rigid grip x; y remains free (use current y).
                    render_coords[nid] = (x_left_pole, float(y))
                elif nid in adapter.right_boundary_node_ids:
                    render_coords[nid] = (x_right_pole, float(y))
                else:
                    render_coords[nid] = (float(x), float(y))

            xs = [xy[0] for xy in render_coords.values()]
            ys = [xy[1] for xy in render_coords.values()]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            pad = max(20, int(0.06 * min(w, h)))
            span_x = (x_max - x_min) if (x_max - x_min) != 0 else 1.0
            span_y = (y_max - y_min) if (y_max - y_min) != 0 else 1.0

            def to_canvas(pt):
                x, y = pt
                cx = pad + (x - x_min) / span_x * (w - 2 * pad)
                cy = pad + (y - y_min) / span_y * (h - 2 * pad)
                return int(cx), int(cy)

            left_x, _ = to_canvas((x_left_pole, y_min))
            right_x, _ = to_canvas((x_right_pole, y_min))

            # Poles: vertical lines only (constraint manifolds)
            self._viz_canvas.create_line(left_x, pole_y0, left_x, pole_y1, fill="red3", width=6)
            self._viz_canvas.create_line(right_x, pole_y0, right_x, pole_y1, fill="red3", width=6)

            # Edges
            for e in adapter.edges:
                a = render_coords.get(e.n_from)
                b = render_coords.get(e.n_to)
                if a is None or b is None:
                    continue
                ax, ay = to_canvas(a)
                bx, by = to_canvas(b)

                # Phase 3C: Edge stiffness visualization
                # Color and width proportional to weakest-link stiffness S
                S_eff = float(e.S)
                if S_eff >= 0.9:
                    edge_color = "deepskyblue2"  # High stiffness (intact)
                elif S_eff >= 0.5:
                    edge_color = "steelblue"     # Moderate stiffness
                elif S_eff >= 0.2:
                    edge_color = "gray"          # Low stiffness
                else:
                    edge_color = "red"           # Critical/near-failure

                # Visualization: edge width must reflect protofibril-scaled stiffness
                # In spatial mode: k_eff = k0 × N_pf × S (visual proxy only)
                if FeatureFlags.USE_SPATIAL_PLASMIN and adapter.spatial_plasmin_params:
                    N_pf = float(adapter.spatial_plasmin_params.get("N_pf", 50))
                    k_eff_visual = N_pf * S_eff  # Visual stiffness proxy
                    edge_width = max(1, int(0.04 * k_eff_visual))
                else:
                    edge_width = max(1, int(2 * S_eff))  # Legacy: weaker edges thinner
                self._viz_canvas.create_line(ax, ay, bx, by, fill=edge_color, width=edge_width)

            # Phase 4: Plasmin site visualization (feature-flagged)
            # Render spatial plasmin binding sites as small red circles at interpolated positions
            try:
                if FeatureFlags.USE_SPATIAL_PLASMIN:
                    plasmin_site_radius = max(2, int(0.004 * min(w, h)))
                    for e in adapter.edges:
                        plasmin_sites = getattr(e, "plasmin_sites", None)
                        if plasmin_sites:  # Non-empty tuple
                            # Get edge endpoints in world coordinates
                            n_from_pt = render_coords.get(e.n_from)
                            n_to_pt = render_coords.get(e.n_to)
                            if n_from_pt is None or n_to_pt is None:
                                continue
                            
                            # Render each plasmin site
                            for site in plasmin_sites:
                                try:
                                    # Parametric position: t ∈ [0, 1] along edge
                                    t = float(site.position_parametric)
                                    t = max(0.0, min(1.0, t))  # Clamp to [0, 1]
                                    
                                    # Interpolate site position along edge (world coords)
                                    x_site = float(n_from_pt[0]) + t * (float(n_to_pt[0]) - float(n_from_pt[0]))
                                    y_site = float(n_from_pt[1]) + t * (float(n_to_pt[1]) - float(n_from_pt[1]))
                                    
                                    # Convert to canvas coordinates
                                    sx, sy = to_canvas((x_site, y_site))
                                    
                                    # Color based on damage severity
                                    damage = float(site.damage_depth)
                                    critical = FeatureFlags.SPATIAL_PLASMIN_CRITICAL_DAMAGE
                                    if damage >= critical:
                                        # Lysed (critical damage reached)
                                        fill_color = "red"
                                        outline_color = "darkred"
                                    elif damage > 0.5 * critical:
                                        # Medium damage
                                        fill_color = "orange"
                                        outline_color = "darkorange"
                                    else:
                                        # Low damage
                                        fill_color = "yellow"
                                        outline_color = "gold"
                                    
                                    # Render plasmin site as small circle
                                    self._viz_canvas.create_oval(
                                        sx - plasmin_site_radius,
                                        sy - plasmin_site_radius,
                                        sx + plasmin_site_radius,
                                        sy + plasmin_site_radius,
                                        fill=fill_color,
                                        outline=outline_color,
                                        width=1
                                    )
                                except (AttributeError, ValueError, TypeError):
                                    # Graceful fallback: skip malformed site
                                    continue
            except Exception:
                # Silent fallback: if feature flag or rendering fails, continue without plasmin visualization
                pass

            # Phase 3A & 3B: Segment-level damage and binding visualization (v5.0 spatial mode)
            # Render segments as small circles colored by integrity (n_i/N_pf) and occupancy (B_i/S_i)
            try:
                if FeatureFlags.USE_SPATIAL_PLASMIN:
                    N_pf = adapter.spatial_plasmin_params.get("N_pf", 50.0) if adapter.spatial_plasmin_params else 50.0
                    segment_radius = max(3, int(0.005 * min(w, h)))  # Fixed radius ~3px

                    for e in adapter.edges:
                        segments = getattr(e, "segments", None)
                        if segments:  # Non-empty segments tuple
                            # Get edge endpoints in world coordinates
                            n_from_pt = render_coords.get(e.n_from)
                            n_to_pt = render_coords.get(e.n_to)
                            if n_from_pt is None or n_to_pt is None:
                                continue

                            num_segments = len(segments)
                            if num_segments == 0:
                                continue

                            # Render each segment
                            for seg in segments:
                                try:
                                    # Uniform spacing: t = segment_index / (num_segments - 1)
                                    # Handle single-segment edge: t = 0.5
                                    if num_segments == 1:
                                        t = 0.5
                                    else:
                                        t = float(seg.segment_index) / float(num_segments - 1)
                                    t = max(0.0, min(1.0, t))  # Clamp to [0, 1]

                                    # Interpolate segment position along edge (world coords)
                                    x_seg = float(n_from_pt[0]) + t * (float(n_to_pt[0]) - float(n_from_pt[0]))
                                    y_seg = float(n_from_pt[1]) + t * (float(n_to_pt[1]) - float(n_from_pt[1]))

                                    # Convert to canvas coordinates
                                    sx, sy = to_canvas((x_seg, y_seg))

                                    # Phase 3A: Color by damage (integrity = n_i / N_pf)
                                    integrity = float(seg.n_i) / float(N_pf)
                                    if integrity >= 0.9:
                                        damage_color = "green"       # Intact
                                    elif integrity >= 0.5:
                                        damage_color = "yellow"      # Moderate damage
                                    elif integrity >= 0.2:
                                        damage_color = "orange"      # Severe damage
                                    else:
                                        damage_color = "red"         # Near failure

                                    # Phase 3B: Binding occupancy overlay (optional: render as outline)
                                    occupancy = float(seg.B_i) / max(1.0, float(seg.S_i))
                                    if occupancy < 0.01:
                                        binding_outline = "blue"      # No binding
                                    elif occupancy < 0.3:
                                        binding_outline = "cyan"      # Low binding
                                    elif occupancy < 0.7:
                                        binding_outline = "purple"    # Medium binding
                                    else:
                                        binding_outline = "magenta"   # High binding

                                    # Render segment as circle: fill = damage, outline = binding
                                    self._viz_canvas.create_oval(
                                        sx - segment_radius,
                                        sy - segment_radius,
                                        sx + segment_radius,
                                        sy + segment_radius,
                                        fill=damage_color,
                                        outline=binding_outline,
                                        width=2
                                    )
                                except (AttributeError, ValueError, TypeError, ZeroDivisionError):
                                    # Graceful fallback: skip malformed segment
                                    continue
            except Exception:
                # Silent fallback: if feature flag or rendering fails, continue without segment visualization
                pass

            # Phase 3D: Fractured edge visualization (dashed gray lines)
            # Render removed edges from fractured_history
            try:
                if FeatureFlags.USE_SPATIAL_PLASMIN and adapter.fractured_history:
                    for record in adapter.fractured_history:
                        try:
                            n_from = record.get("n_from")
                            n_to = record.get("n_to")
                            if n_from is None or n_to is None:
                                continue

                            # Get node positions
                            a_frac = render_coords.get(n_from)
                            b_frac = render_coords.get(n_to)
                            if a_frac is None or b_frac is None:
                                continue

                            # Convert to canvas coordinates
                            ax_frac, ay_frac = to_canvas(a_frac)
                            bx_frac, by_frac = to_canvas(b_frac)

                            # Render as dashed gray line
                            self._viz_canvas.create_line(
                                ax_frac, ay_frac, bx_frac, by_frac,
                                fill="gray",
                                dash=(4, 4),  # Dashed pattern
                                width=1
                            )
                        except (AttributeError, KeyError, TypeError):
                            # Graceful fallback: skip malformed record
                            continue
            except Exception:
                # Silent fallback: if rendering fails, continue without fractured edge visualization
                pass

            # Legend panel (read-only visualization guide)
            legend_x = pad
            legend_y = pad
            legend_line_height = 16
            legend_font = ("Consolas", 9)

            # Edge color legend (always shown)
            self._viz_canvas.create_text(
                legend_x, legend_y,
                anchor="nw",
                fill="white",
                font=("Consolas", 9, "bold"),
                text="LEGEND"
            )
            legend_y += legend_line_height

            self._viz_canvas.create_text(
                legend_x, legend_y,
                anchor="nw",
                fill="deepskyblue2",
                font=legend_font,
                text="■ Edge: Intact (S≥0.9)"
            )
            legend_y += legend_line_height

            self._viz_canvas.create_text(
                legend_x, legend_y,
                anchor="nw",
                fill="steelblue",
                font=legend_font,
                text="■ Edge: Moderate (S≥0.5)"
            )
            legend_y += legend_line_height

            self._viz_canvas.create_text(
                legend_x, legend_y,
                anchor="nw",
                fill="gray",
                font=legend_font,
                text="■ Edge: Low (S≥0.2)"
            )
            legend_y += legend_line_height

            self._viz_canvas.create_text(
                legend_x, legend_y,
                anchor="nw",
                fill="red",
                font=legend_font,
                text="■ Edge: Critical (S<0.2)"
            )
            legend_y += legend_line_height

            # Spatial mode legend (segment-level)
            if FeatureFlags.USE_SPATIAL_PLASMIN:
                legend_y += 4  # Spacing
                self._viz_canvas.create_text(
                    legend_x, legend_y,
                    anchor="nw",
                    fill="white",
                    font=("Consolas", 9, "bold"),
                    text="SEGMENTS (spatial mode)"
                )
                legend_y += legend_line_height

                self._viz_canvas.create_text(
                    legend_x, legend_y,
                    anchor="nw",
                    fill="green",
                    font=legend_font,
                    text="● Damage: Intact (n/N≥0.9)"
                )
                legend_y += legend_line_height

                self._viz_canvas.create_text(
                    legend_x, legend_y,
                    anchor="nw",
                    fill="yellow",
                    font=legend_font,
                    text="● Damage: Moderate (n/N≥0.5)"
                )
                legend_y += legend_line_height

                self._viz_canvas.create_text(
                    legend_x, legend_y,
                    anchor="nw",
                    fill="orange",
                    font=legend_font,
                    text="● Damage: Severe (n/N≥0.2)"
                )
                legend_y += legend_line_height

                self._viz_canvas.create_text(
                    legend_x, legend_y,
                    anchor="nw",
                    fill="red",
                    font=legend_font,
                    text="● Damage: Critical (n/N<0.2)"
                )
                legend_y += legend_line_height

                legend_y += 4  # Spacing
                self._viz_canvas.create_text(
                    legend_x, legend_y,
                    anchor="nw",
                    fill="blue",
                    font=legend_font,
                    text="○ Binding: None (B/S<0.01)"
                )
                legend_y += legend_line_height

                self._viz_canvas.create_text(
                    legend_x, legend_y,
                    anchor="nw",
                    fill="cyan",
                    font=legend_font,
                    text="○ Binding: Low (B/S<0.3)"
                )
                legend_y += legend_line_height

                self._viz_canvas.create_text(
                    legend_x, legend_y,
                    anchor="nw",
                    fill="purple",
                    font=legend_font,
                    text="○ Binding: Medium (B/S<0.7)"
                )
                legend_y += legend_line_height

                self._viz_canvas.create_text(
                    legend_x, legend_y,
                    anchor="nw",
                    fill="magenta",
                    font=legend_font,
                    text="○ Binding: High (B/S≥0.7)"
                )
                legend_y += legend_line_height

                legend_y += 4  # Spacing
                self._viz_canvas.create_text(
                    legend_x, legend_y,
                    anchor="nw",
                    fill="gray",
                    font=legend_font,
                    text="--- Fractured (removed)"
                )

            # Nodes
            r = max(3, int(0.008 * min(w, h)))
            attached_left = set(adapter.left_attachment_node_ids)
            attached_right = set(adapter.right_attachment_node_ids)
            for nid, pt in render_coords.items():
                px, py = to_canvas(pt)
                fill = "white"
                outline = "gray70"
                if nid in attached_left:
                    outline = "gold"
                elif nid in attached_right:
                    outline = "orange"
                self._viz_canvas.create_oval(px - r, py - r, px + r, py + r, fill=fill, outline=outline, width=2)

            # Termination reason display (read-only, bottom-right)
            if adapter.termination_reason is not None:
                term_reason = str(adapter.termination_reason)
                term_batch = getattr(adapter, "termination_batch_index", None)
                term_time = getattr(adapter, "termination_time", None)

                # Format termination message
                term_lines = ["SIMULATION TERMINATED"]
                term_lines.append(f"Reason: {term_reason}")
                if term_batch is not None:
                    term_lines.append(f"Batch: {term_batch}")
                if term_time is not None:
                    term_lines.append(f"Time: {term_time:.2f} s")

                term_text = "\n".join(term_lines)

                # Display in bottom-right corner with background box
                text_x = w - pad - 10
                text_y = h - pad - 10

                # Create semi-transparent background box
                self._viz_canvas.create_rectangle(
                    text_x - 200, text_y - 60,
                    text_x + 10, text_y + 10,
                    fill="red3",
                    outline="white",
                    width=2
                )

                self._viz_canvas.create_text(
                    text_x - 95, text_y - 25,
                    anchor="center",
                    fill="white",
                    font=("Consolas", 10, "bold"),
                    text=term_text
                )
        else:
            # Poles for placeholder view (UI-only)
            strain = float(self.controller.state.strain_value)
            left_x = int((0.25 - 0.15 * strain) * w)
            right_x = int((0.75 + 0.15 * strain) * w)
            self._viz_canvas.create_line(left_x, pole_y0, left_x, pole_y1, fill="red3", width=6)
            self._viz_canvas.create_line(right_x, pole_y0, right_x, pole_y1, fill="red3", width=6)
            # Static placeholder sketch (normalized coords)
            points = []
            for (nx, ny) in self._static_nodes:
                px = int(nx * w)
                py = int(ny * h)
                points.append((px, py))

            # Fibers
            for a, b in self._static_edges:
                ax, ay = points[a]
                bx, by = points[b]
                self._viz_canvas.create_line(
                    ax, ay, bx, by,
                    fill="deepskyblue2",
                    width=2,
                )

            # Nodes
            r = max(3, int(0.01 * min(w, h)))
            for (px, py) in points:
                self._viz_canvas.create_oval(
                    px - r, py - r, px + r, py + r,
                    fill="white",
                    outline="gray70",
                )

    # ---------------------------
    # State -> UI rendering
    # ---------------------------
    def _render_from_state(self):
        """
        Single, explicit render point from SimulationState to UI.
        Keeps coupling readable: state changes -> render -> widgets update.
        """
        state = self.controller.state

        # Applied strain is fixed after Start (disable editing once frozen).
        adapter = state.loaded_network
        is_frozen = isinstance(adapter, Phase1NetworkAdapter) and (adapter.frozen_params is not None)
        if self._applied_strain_entry is not None:
            self._applied_strain_entry.configure(state=("disabled" if is_frozen else "normal"))
            if is_frozen:
                # Keep display aligned deterministically when frozen/resumed/forked.
                self.applied_strain_fixed.set(str(float(state.strain_value)))

        # Metrics from state only
        self.metric_time_min.set(self._format_minutes(state.time))
        self.metric_running.set(self._format_running_state())
        self.metric_paused.set(self._format_paused_state())

        # Metrics are displayed from the last batch delta only (no recomputation here).
        metrics = self.controller.last_metrics
        if isinstance(metrics, dict):
            # Formatting only; values originate from the step.
            self.metric_mean_tension.set(str(metrics.get("mean_tension", "--")))
            self.metric_active_fibers.set(str(metrics.get("active_fibers", "--")))
            self.metric_cleaved_fibers.set(str(metrics.get("cleaved_fibers", metrics.get("ruptured_fibers", "--"))))
            lysis_fraction = metrics.get("lysis_fraction", None)
            if lysis_fraction is None:
                self.metric_lysis_pct.set("--")
            else:
                # Display as percent for the UI label; deterministic unit conversion only.
                self.metric_lysis_pct.set(f"{float(lysis_fraction) * 100.0:.4f}")
        else:
            self.metric_mean_tension.set("--")
            self.metric_active_fibers.set("--")
            self.metric_cleaved_fibers.set("--")
            self.metric_lysis_pct.set("--")

        # Phase 1C static mechanics observables:
        # mean_tension and active_fibers are computed from the adapter's cached forces (read-only).
        adapter = state.loaded_network
        if isinstance(adapter, Phase1NetworkAdapter):
            forces = adapter.forces
            positive = [float(f) for f in forces.values() if float(f) > 0.0]
            mean_tension = (sum(positive) / len(positive)) if positive else 0.0
            active = len(positive)
            self.metric_mean_tension.set(f"{mean_tension:.6f}")
            self.metric_active_fibers.set(str(active))

        # Visualization depends only on state.strain_value (static network otherwise)
        self._redraw_visualization()

    def _format_minutes(self, seconds):
        """
        Format state time (seconds) into minutes for display.
        No evolution logic; deterministic conversion only.
        """
        try:
            sec = float(seconds)
        except Exception:
            sec = 0.0
        minutes = sec / 60.0
        # Fixed formatting for deterministic display
        return f"{minutes:.4f}"

    def _format_running_state(self):
        return "True" if bool(self.controller.state.is_running) else "False"

    def _format_paused_state(self):
        return "True" if bool(self.controller.state.is_paused) else "False"


