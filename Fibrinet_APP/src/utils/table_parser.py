"""
Table Parsing Utilities for FibriNet.

Deterministic, read-only parsers for CSV and XLSX network data files.
All functions are stateless and produce consistent output for identical input.

This module provides:
- Multi-table CSV/XLSX parsing (nodes, edges, meta_data)
- Type coercion with detailed error messages
- Column name normalization
- Utility functions for data manipulation
"""

import csv
import math
import os
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.utils.file_validation import validate_file_size, FileSizeError

# Default maximum file size for parsing (50 MB)
DEFAULT_MAX_PARSE_SIZE_MB = 50


def parse_delimited_tables_from_csv(
    path: str,
    max_size_mb: Optional[float] = None
) -> list[dict[str, list[Any]]]:
    """
    Deterministic, read-only CSV multi-table parser.

    Expected format (as used by existing test data):
    - Table 0: nodes (header row, then data rows)
    - blank line
    - Table 1: edges (header row, then data rows)
    - blank line
    - Table 2 (optional): meta_data (header row, then key/value rows)

    Args:
        path: Path to CSV file
        max_size_mb: Maximum file size in MB (default: 50 MB)

    Returns:
        List of table dictionaries, each mapping column names to value lists

    Raises:
        FileSizeError: If file exceeds maximum size
        FileNotFoundError: If file does not exist
        ValueError: If empty cells are detected in the CSV
    """
    # Validate file size before parsing
    validate_file_size(
        path,
        max_size_mb=max_size_mb or DEFAULT_MAX_PARSE_SIZE_MB
    )

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


def parse_delimited_tables_from_xlsx(
    path: str,
    max_size_mb: Optional[float] = None
) -> list[dict[str, list[Any]]]:
    """
    Deterministic, read-only XLSX multi-table parser.

    Supports:
    - Fast path: sheets named "nodes"/"edges" (optional "meta_data")
    - Fallback: multiple stacked tables in a single sheet, detected via header-row scanning
      and sliced deterministically without relying on blank rows.

    Critical safety goals:
    - Do not mix meta/header rows into the edges table.
    - Preserve raw cell values (dtype=object) so callers can validate strictly (no NaN->0 coercions).

    Args:
        path: Path to XLSX file
        max_size_mb: Maximum file size in MB (default: 50 MB)

    Returns:
        List of table dictionaries, each mapping column names to value lists

    Raises:
        FileSizeError: If file exceeds maximum size
        FileNotFoundError: If file does not exist
        ValueError: If required table headers cannot be detected
    """
    # Validate file size before parsing
    validate_file_size(
        path,
        max_size_mb=max_size_mb or DEFAULT_MAX_PARSE_SIZE_MB
    )

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
        return normalize_column_name(s)

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
                if not any(normalize_column_name(g) in present for g in group):
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
        sheet_names_norm = {normalize_column_name(s): s for s in xl.sheet_names}
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


def normalize_column_name(name: str) -> str:
    """Normalize column name to lowercase with underscores."""
    return str(name).strip().lower().replace(" ", "_")


def require_column(table: Mapping[str, list[Any]], candidates: Sequence[str], *, table_name: str) -> str:
    """
    Find a required column by checking candidate names.

    Args:
        table: Table dictionary mapping column names to values
        candidates: List of acceptable column names to check
        table_name: Name of table for error messages

    Returns:
        The actual column name found in the table

    Raises:
        ValueError: If none of the candidate columns exist
    """
    norm = {normalize_column_name(k): k for k in table.keys()}
    for c in candidates:
        key = norm.get(normalize_column_name(c))
        if key is not None:
            return key
    raise ValueError(f"Missing required column in {table_name}: one of {list(candidates)}")


def coerce_int(v: Any, *, sheet: str = None, row: int = None, column: str = None) -> int:
    """
    Coerce value to int with optional context for error messages.

    Args:
        v: Value to coerce
        sheet: Optional sheet name for error context
        row: Optional row number for error context
        column: Optional column name for error context

    Returns:
        Integer value

    Raises:
        ValueError: If value cannot be converted to int
    """
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


def coerce_float(v: Any, *, sheet: str = None, row: int = None, column: str = None) -> float:
    """
    Coerce value to float with optional context for error messages.

    Args:
        v: Value to coerce
        sheet: Optional sheet name for error context
        row: Optional row number for error context
        column: Optional column name for error context

    Returns:
        Float value

    Raises:
        ValueError: If value cannot be converted to float
    """
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


def coerce_bool_boundary_flag(v: Any, *, node_id: Any, column_name: str) -> bool:
    """
    Coerce explicit boundary flags from input tables.

    Accepted forms (per spec):
    - True/False
    - 1/0 (int/float) or "1"/"0"

    Args:
        v: Value to coerce
        node_id: Node ID for error context
        column_name: Column name for error context

    Returns:
        Boolean value

    Raises:
        ValueError: If value cannot be interpreted as boolean
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


def coerce_bool_input_flag(v: Any, *, node_id: Any, column_name: str) -> bool:
    """
    Coerce boolean-like input flags from input tables (strict, deterministic).

    Accepted forms:
    - True/False
    - 1/0 (int/float) or "1"/"0"

    Args:
        v: Value to coerce
        node_id: Node ID for error context
        column_name: Column name for error context

    Returns:
        Boolean value

    Raises:
        ValueError: If value cannot be interpreted as boolean
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


def euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two 2D points.

    Args:
        a: First point (x, y)
        b: Second point (x, y)

    Returns:
        Distance between points
    """
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return float(math.sqrt(dx * dx + dy * dy))


def jsonify(obj: Any) -> Any:
    """
    Convert tuples recursively into lists so objects are JSON-serializable.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, tuple):
        return [jsonify(x) for x in obj]
    if isinstance(obj, list):
        return [jsonify(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): jsonify(v) for k, v in obj.items()}
    return obj


def tuplify(obj: Any) -> Any:
    """
    Convert lists recursively into tuples (inverse of jsonify for RNG state restore).

    Args:
        obj: Object to convert

    Returns:
        Tuple version of the object
    """
    if isinstance(obj, list):
        return tuple(tuplify(x) for x in obj)
    if isinstance(obj, dict):
        return {k: tuplify(v) for k, v in obj.items()}
    return obj


def median(values: Sequence[float]) -> float:
    """
    Calculate deterministic median for a non-empty sequence of floats.

    Args:
        values: Sequence of numeric values

    Returns:
        Median value

    Raises:
        ValueError: If sequence is empty
    """
    xs = sorted(float(x) for x in values)
    n = len(xs)
    if n == 0:
        raise ValueError("median of empty sequence")
    mid = n // 2
    if (n % 2) == 1:
        return float(xs[mid])
    return 0.5 * (float(xs[mid - 1]) + float(xs[mid]))


# Legacy aliases with underscore prefix for backward compatibility
_parse_delimited_tables_from_csv = parse_delimited_tables_from_csv
_parse_delimited_tables_from_xlsx = parse_delimited_tables_from_xlsx
_normalize_column_name = normalize_column_name
_require_column = require_column
_coerce_int = coerce_int
_coerce_float = coerce_float
_coerce_bool_boundary_flag = coerce_bool_boundary_flag
_coerce_bool_input_flag = coerce_bool_input_flag
_euclidean = euclidean_distance
_jsonify = jsonify
_tuplify = tuplify
_median = median
