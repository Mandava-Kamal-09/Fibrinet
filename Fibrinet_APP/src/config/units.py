"""Unit conversion system for FibriNet experimental parameters."""

from typing import Final


class Units:
    """Conversion factors between SI (internal) and experimental (display) units."""

    # Length conversions
    M_TO_UM: Final[float] = 1e6
    M_TO_NM: Final[float] = 1e9
    UM_TO_M: Final[float] = 1e-6
    NM_TO_M: Final[float] = 1e-9

    # Force conversions
    N_TO_PN: Final[float] = 1e12
    N_TO_NN: Final[float] = 1e9
    PN_TO_N: Final[float] = 1e-12
    NN_TO_N: Final[float] = 1e-9

    # Time conversions
    S_TO_MS: Final[float] = 1e3
    S_TO_MIN: Final[float] = 1.0 / 60.0
    MS_TO_S: Final[float] = 1e-3
    MIN_TO_S: Final[float] = 60.0

    # Energy conversions (k_B T at 37°C)
    J_TO_KBT: Final[float] = 1.0 / (1.380649e-23 * 310.15)
    KBT_TO_J: Final[float] = 1.380649e-23 * 310.15

    # Dimensionless conversions
    STRAIN_TO_PERCENT: Final[float] = 100.0
    PERCENT_TO_STRAIN: Final[float] = 0.01


def format_length_um(meters: float) -> str:
    """Format length in micrometers."""
    return f"{meters * Units.M_TO_UM:.2f}"


def format_length_nm(meters: float) -> str:
    """Format length in nanometers."""
    return f"{meters * Units.M_TO_NM:.1f}"


def format_force_pn(newtons: float) -> str:
    """Format force in picoNewtons."""
    return f"{newtons * Units.N_TO_PN:.1f}"


def format_force_nn(newtons: float) -> str:
    """Format force in nanoNewtons."""
    return f"{newtons * Units.N_TO_NN:.2f}"


def format_time_min(seconds: float) -> str:
    """Format time in minutes."""
    return f"{seconds * Units.S_TO_MIN:.2f}"


def format_time_s(seconds: float) -> str:
    """Format time in seconds."""
    return f"{seconds:.2f}"


def format_strain_percent(strain: float) -> str:
    """Format strain as percentage."""
    return f"{strain * Units.STRAIN_TO_PERCENT:.1f}"


def format_energy_kbt(joules: float) -> str:
    """Format energy in thermal energy units."""
    return f"{joules * Units.J_TO_KBT:.1f}"


# Display unit labels
UNIT_LABELS = {
    'strain': '%',
    'plasmin': 'µg/mL',
    'time_step': 'ms',
    'max_time': 'min',
    'length': 'µm',
    'thickness': 'nm',
    'force_fiber': 'pN',
    'force_network': 'nN',
    'time': 'min',
    'energy': 'k_B T',
    'lysis': '%',
}
