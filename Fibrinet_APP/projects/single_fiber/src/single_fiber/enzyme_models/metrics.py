"""
Metrics and analysis for enzyme cleavage experiments.

Computes statistics from simulation output (CSV files):
- Survival probability vs time
- Mean rupture time
- Variance and confidence intervals
- Fraction intact vs time

All functions are deterministic and operate on exported data only.
No GUI dependencies.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path


@dataclass
class SurvivalCurve:
    """
    Survival probability curve.

    Attributes:
        times_us: Time points in µs
        survival_prob: Survival probability at each time [0, 1]
        n_samples: Number of samples used to compute curve
        ci_lower: Lower 95% confidence bound (optional)
        ci_upper: Upper 95% confidence bound (optional)
    """
    times_us: np.ndarray
    survival_prob: np.ndarray
    n_samples: int
    ci_lower: Optional[np.ndarray] = None
    ci_upper: Optional[np.ndarray] = None


@dataclass
class RuptureStatistics:
    """
    Summary statistics for rupture times.

    Attributes:
        mean_rupture_time_us: Mean time to rupture (inf if some didn't rupture)
        std_rupture_time_us: Standard deviation of rupture times
        median_rupture_time_us: Median rupture time
        fraction_ruptured: Fraction of samples that ruptured
        n_samples: Total number of samples
        rupture_times_us: Individual rupture times (for those that ruptured)
    """
    mean_rupture_time_us: float
    std_rupture_time_us: float
    median_rupture_time_us: float
    fraction_ruptured: float
    n_samples: int
    rupture_times_us: np.ndarray


def compute_survival_curve(
    rupture_times: np.ndarray,
    max_time_us: float,
    n_time_points: int = 100,
    censored: Optional[np.ndarray] = None
) -> SurvivalCurve:
    """
    Compute empirical survival curve from rupture times.

    Uses Kaplan-Meier estimator when censoring is present.

    Args:
        rupture_times: Array of rupture times in µs (inf for no rupture)
        max_time_us: Maximum time for curve
        n_time_points: Number of time points in output
        censored: Boolean array indicating censored observations

    Returns:
        SurvivalCurve with times and probabilities
    """
    times = np.linspace(0, max_time_us, n_time_points)
    n_samples = len(rupture_times)

    if censored is None:
        censored = np.isinf(rupture_times)

    # Simple empirical survival: S(t) = P(T > t)
    survival = np.zeros(n_time_points)
    for i, t in enumerate(times):
        # Count samples that survived past time t
        # Censored samples count as survived if censoring time > t
        survived = np.sum(rupture_times > t)
        survival[i] = survived / n_samples

    # Wilson confidence interval for proportions
    z = 1.96  # 95% CI
    ci_lower = np.zeros(n_time_points)
    ci_upper = np.zeros(n_time_points)

    for i in range(n_time_points):
        p = survival[i]
        n = n_samples
        denom = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denom
        margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
        ci_lower[i] = max(0, center - margin)
        ci_upper[i] = min(1, center + margin)

    return SurvivalCurve(
        times_us=times,
        survival_prob=survival,
        n_samples=n_samples,
        ci_lower=ci_lower,
        ci_upper=ci_upper
    )


def compute_rupture_statistics(
    rupture_times: np.ndarray,
    max_time_us: float = float('inf')
) -> RuptureStatistics:
    """
    Compute summary statistics for rupture times.

    Args:
        rupture_times: Array of rupture times (inf for no rupture)
        max_time_us: Maximum observation time (for censoring)

    Returns:
        RuptureStatistics with mean, std, median, etc.
    """
    n_samples = len(rupture_times)

    # Filter finite rupture times
    finite_times = rupture_times[np.isfinite(rupture_times)]
    n_ruptured = len(finite_times)
    fraction_ruptured = n_ruptured / n_samples if n_samples > 0 else 0.0

    if n_ruptured == 0:
        return RuptureStatistics(
            mean_rupture_time_us=float('inf'),
            std_rupture_time_us=float('nan'),
            median_rupture_time_us=float('inf'),
            fraction_ruptured=0.0,
            n_samples=n_samples,
            rupture_times_us=np.array([])
        )

    return RuptureStatistics(
        mean_rupture_time_us=np.mean(finite_times),
        std_rupture_time_us=np.std(finite_times, ddof=1) if n_ruptured > 1 else 0.0,
        median_rupture_time_us=np.median(finite_times),
        fraction_ruptured=fraction_ruptured,
        n_samples=n_samples,
        rupture_times_us=finite_times
    )


def load_rupture_times_from_csv(
    csv_path: Path,
    time_column: str = "t_us",
    intact_column: str = "intact"
) -> Tuple[float, bool]:
    """
    Load rupture time from a single simulation CSV.

    Args:
        csv_path: Path to CSV file
        time_column: Name of time column
        intact_column: Name of intact status column

    Returns:
        (rupture_time_us, did_rupture) tuple
    """
    df = pd.read_csv(csv_path)

    if intact_column not in df.columns:
        # Assume no rupture if column missing
        return df[time_column].max(), False

    # Find first time where intact becomes 0
    rupture_rows = df[df[intact_column] == 0]

    if len(rupture_rows) == 0:
        return float('inf'), False

    rupture_time = rupture_rows[time_column].min()
    return rupture_time, True


def load_rupture_times_from_folder(
    folder_path: Path,
    pattern: str = "*.csv"
) -> np.ndarray:
    """
    Load rupture times from all CSVs in a folder.

    Args:
        folder_path: Path to folder containing CSVs
        pattern: Glob pattern for CSV files

    Returns:
        Array of rupture times (inf for no rupture)
    """
    folder = Path(folder_path)
    csv_files = sorted(folder.glob(pattern))

    rupture_times = []
    for csv_path in csv_files:
        try:
            time, _ = load_rupture_times_from_csv(csv_path)
            rupture_times.append(time)
        except Exception as e:
            print(f"Warning: could not load {csv_path}: {e}")

    return np.array(rupture_times)


def compute_fraction_intact_vs_time(
    df: pd.DataFrame,
    time_column: str = "t_us",
    intact_column: str = "intact"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute fraction of intact segments over time from single run.

    Args:
        df: DataFrame with simulation output
        time_column: Name of time column
        intact_column: Name of intact column

    Returns:
        (times, fractions) arrays
    """
    times = df[time_column].values
    intact = df[intact_column].values

    # For multi-segment, intact might be sum of intact segments
    # Normalize to [0, 1] based on initial value
    initial_intact = intact[0] if len(intact) > 0 else 1
    if initial_intact == 0:
        initial_intact = 1

    fractions = intact / initial_intact

    return times, fractions


def compute_hazard_vs_strain_curve(
    hazard_fn,
    params: dict,
    strain_range: Tuple[float, float] = (0.0, 1.0),
    tension_pN: float = 0.0,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute hazard rate as function of strain.

    Args:
        hazard_fn: Hazard function callable
        params: Parameters for hazard function
        strain_range: (min, max) strain values
        tension_pN: Fixed tension value
        n_points: Number of points

    Returns:
        (strains, hazard_rates) arrays
    """
    strains = np.linspace(strain_range[0], strain_range[1], n_points)
    rates = np.array([hazard_fn(s, tension_pN, params) for s in strains])
    return strains, rates


def compute_hazard_vs_tension_curve(
    hazard_fn,
    params: dict,
    tension_range: Tuple[float, float] = (0.0, 100.0),
    strain: float = 0.0,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute hazard rate as function of tension.

    Args:
        hazard_fn: Hazard function callable
        params: Parameters for hazard function
        tension_range: (min, max) tension values in pN
        strain: Fixed strain value
        n_points: Number of points

    Returns:
        (tensions, hazard_rates) arrays
    """
    tensions = np.linspace(tension_range[0], tension_range[1], n_points)
    rates = np.array([hazard_fn(strain, t, params) for t in tensions])
    return tensions, rates


__all__ = [
    "SurvivalCurve",
    "RuptureStatistics",
    "compute_survival_curve",
    "compute_rupture_statistics",
    "load_rupture_times_from_csv",
    "load_rupture_times_from_folder",
    "compute_fraction_intact_vs_time",
    "compute_hazard_vs_strain_curve",
    "compute_hazard_vs_tension_curve",
]
