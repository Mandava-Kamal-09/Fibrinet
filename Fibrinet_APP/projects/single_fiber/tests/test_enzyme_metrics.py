"""
Tests for enzyme metrics and analysis functions.

Phase 4: Strain-Enzyme Coupling Lab

These tests verify:
- Survival curve computation
- Rupture statistics calculation
- Hazard vs strain/tension curves
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from projects.single_fiber.src.single_fiber.enzyme_models.metrics import (
    SurvivalCurve,
    RuptureStatistics,
    compute_survival_curve,
    compute_rupture_statistics,
    compute_hazard_vs_strain_curve,
    compute_hazard_vs_tension_curve,
)


class TestSurvivalCurve:
    """Tests for survival curve computation."""

    def test_survival_starts_at_one(self):
        """Survival probability starts at 1.0 at t=0."""
        rupture_times = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        curve = compute_survival_curve(rupture_times, max_time_us=100.0)
        assert curve.survival_prob[0] == pytest.approx(1.0)

    def test_survival_decreases_monotonically(self):
        """Survival probability is non-increasing over time."""
        rupture_times = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        curve = compute_survival_curve(rupture_times, max_time_us=100.0)

        for i in range(len(curve.survival_prob) - 1):
            assert curve.survival_prob[i] >= curve.survival_prob[i + 1]

    def test_survival_ends_at_zero_for_all_ruptured(self):
        """If all samples ruptured, survival ends at 0."""
        rupture_times = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        curve = compute_survival_curve(rupture_times, max_time_us=100.0)
        # After last rupture (t=50), survival should be 0
        assert curve.survival_prob[-1] == pytest.approx(0.0)

    def test_survival_with_censored(self):
        """Censored observations (inf) count as survived."""
        rupture_times = np.array([10.0, 20.0, np.inf, np.inf, np.inf])
        curve = compute_survival_curve(rupture_times, max_time_us=50.0)
        # 3 out of 5 never ruptured, so final survival = 0.6
        assert curve.survival_prob[-1] == pytest.approx(0.6)

    def test_confidence_intervals_bracket_survival(self):
        """CI lower <= survival <= CI upper."""
        rupture_times = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        curve = compute_survival_curve(rupture_times, max_time_us=100.0)

        for i in range(len(curve.survival_prob)):
            assert curve.ci_lower[i] <= curve.survival_prob[i]
            assert curve.survival_prob[i] <= curve.ci_upper[i]

    def test_n_samples_recorded(self):
        """Number of samples is recorded correctly."""
        rupture_times = np.array([10.0, 20.0, 30.0])
        curve = compute_survival_curve(rupture_times, max_time_us=100.0)
        assert curve.n_samples == 3


class TestRuptureStatistics:
    """Tests for rupture statistics computation."""

    def test_mean_rupture_time(self):
        """Mean rupture time is computed correctly."""
        rupture_times = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        stats = compute_rupture_statistics(rupture_times)
        assert stats.mean_rupture_time_us == pytest.approx(30.0)

    def test_median_rupture_time(self):
        """Median rupture time is computed correctly."""
        rupture_times = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        stats = compute_rupture_statistics(rupture_times)
        assert stats.median_rupture_time_us == pytest.approx(30.0)

    def test_std_rupture_time(self):
        """Standard deviation is computed correctly."""
        rupture_times = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        stats = compute_rupture_statistics(rupture_times)
        expected_std = np.std(rupture_times, ddof=1)
        assert stats.std_rupture_time_us == pytest.approx(expected_std)

    def test_fraction_ruptured(self):
        """Fraction ruptured is correct for mixed results."""
        rupture_times = np.array([10.0, 20.0, np.inf, np.inf, np.inf])
        stats = compute_rupture_statistics(rupture_times)
        assert stats.fraction_ruptured == pytest.approx(0.4)

    def test_all_ruptured(self):
        """Fraction ruptured = 1.0 when all samples ruptured."""
        rupture_times = np.array([10.0, 20.0, 30.0])
        stats = compute_rupture_statistics(rupture_times)
        assert stats.fraction_ruptured == pytest.approx(1.0)

    def test_none_ruptured(self):
        """Stats handle case where no samples ruptured."""
        rupture_times = np.array([np.inf, np.inf, np.inf])
        stats = compute_rupture_statistics(rupture_times)
        assert stats.fraction_ruptured == pytest.approx(0.0)
        assert stats.mean_rupture_time_us == float('inf')
        assert stats.median_rupture_time_us == float('inf')

    def test_rupture_times_array(self):
        """Individual rupture times are returned."""
        rupture_times = np.array([10.0, 20.0, np.inf, 30.0])
        stats = compute_rupture_statistics(rupture_times)
        # Should only include finite times
        assert len(stats.rupture_times_us) == 3
        assert 10.0 in stats.rupture_times_us
        assert 20.0 in stats.rupture_times_us
        assert 30.0 in stats.rupture_times_us


class TestHazardVsStrainCurve:
    """Tests for hazard rate vs strain curve computation."""

    def test_curve_shape(self):
        """Returns correct number of points."""
        def linear_hazard(strain, tension, params):
            return params["lambda0"] * (1 + params["alpha"] * strain)

        params = {"lambda0": 0.01, "alpha": 5.0}
        strains, rates = compute_hazard_vs_strain_curve(
            linear_hazard, params, strain_range=(0.0, 1.0), n_points=50
        )
        assert len(strains) == 50
        assert len(rates) == 50

    def test_strain_range(self):
        """Strain values span requested range."""
        def const_hazard(strain, tension, params):
            return params["lambda0"]

        params = {"lambda0": 0.01}
        strains, rates = compute_hazard_vs_strain_curve(
            const_hazard, params, strain_range=(0.2, 0.8), n_points=10
        )
        assert strains[0] == pytest.approx(0.2)
        assert strains[-1] == pytest.approx(0.8)

    def test_rates_computed_correctly(self):
        """Hazard rates match function output."""
        def exp_hazard(strain, tension, params):
            return params["lambda0"] * np.exp(params["alpha"] * strain)

        params = {"lambda0": 0.01, "alpha": 5.0}
        strains, rates = compute_hazard_vs_strain_curve(
            exp_hazard, params, strain_range=(0.0, 0.5), n_points=3
        )

        for s, r in zip(strains, rates):
            expected = params["lambda0"] * np.exp(params["alpha"] * s)
            assert r == pytest.approx(expected)


class TestHazardVsTensionCurve:
    """Tests for hazard rate vs tension curve computation."""

    def test_curve_shape(self):
        """Returns correct number of points."""
        def bell_hazard(strain, tension, params):
            return params["lambda0"] * np.exp(params["beta"] * tension)

        params = {"lambda0": 0.01, "beta": 0.1}
        tensions, rates = compute_hazard_vs_tension_curve(
            bell_hazard, params, tension_range=(0.0, 50.0), n_points=25
        )
        assert len(tensions) == 25
        assert len(rates) == 25

    def test_tension_range(self):
        """Tension values span requested range."""
        def const_hazard(strain, tension, params):
            return params["lambda0"]

        params = {"lambda0": 0.01}
        tensions, rates = compute_hazard_vs_tension_curve(
            const_hazard, params, tension_range=(10.0, 100.0), n_points=10
        )
        assert tensions[0] == pytest.approx(10.0)
        assert tensions[-1] == pytest.approx(100.0)

    def test_uses_fixed_strain(self):
        """Fixed strain value is passed to function."""
        def strain_dep_hazard(strain, tension, params):
            return strain * 10.0  # Rate proportional to strain

        params = {}
        tensions, rates = compute_hazard_vs_tension_curve(
            strain_dep_hazard, params,
            tension_range=(0.0, 50.0),
            strain=0.5,  # Fixed strain
            n_points=5
        )

        # All rates should equal 0.5 * 10 = 5.0
        for r in rates:
            assert r == pytest.approx(5.0)


class TestDataclasses:
    """Tests for dataclass structures."""

    def test_survival_curve_fields(self):
        """SurvivalCurve has expected fields."""
        curve = SurvivalCurve(
            times_us=np.array([0, 1, 2]),
            survival_prob=np.array([1.0, 0.5, 0.0]),
            n_samples=10,
            ci_lower=np.array([0.9, 0.4, 0.0]),
            ci_upper=np.array([1.0, 0.6, 0.1])
        )
        assert len(curve.times_us) == 3
        assert curve.n_samples == 10

    def test_rupture_statistics_fields(self):
        """RuptureStatistics has expected fields."""
        stats = RuptureStatistics(
            mean_rupture_time_us=25.0,
            std_rupture_time_us=10.0,
            median_rupture_time_us=20.0,
            fraction_ruptured=0.8,
            n_samples=100,
            rupture_times_us=np.array([10, 20, 30])
        )
        assert stats.mean_rupture_time_us == 25.0
        assert stats.n_samples == 100
