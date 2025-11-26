"""Tests for the nozzle module."""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rocket.engine import EngineInputs, design_engine
from rocket.nozzle import (
    NozzleContour,
    conical_contour,
    full_chamber_contour,
    generate_nozzle_from_geometry,
    rao_bell_contour,
)
from rocket.units import kelvin, megapascals, meters, newtons, pascals


class TestNozzleContour:
    """Test NozzleContour dataclass."""

    def test_create_valid_contour(self) -> None:
        """Test creating a valid contour."""
        x = np.array([0.0, 0.1, 0.2])
        y = np.array([0.05, 0.06, 0.08])

        contour = NozzleContour(x=x, y=y, contour_type="test")

        assert len(contour.x) == 3
        assert len(contour.y) == 3
        assert contour.contour_type == "test"

    def test_mismatched_lengths_raises(self) -> None:
        """Test that mismatched x and y lengths raise error."""
        x = np.array([0.0, 0.1, 0.2])
        y = np.array([0.05, 0.06])  # Different length!

        with pytest.raises(ValueError, match="same length"):
            NozzleContour(x=x, y=y, contour_type="test")

    def test_too_few_points_raises(self) -> None:
        """Test that contour with < 2 points raises error."""
        x = np.array([0.0])
        y = np.array([0.05])

        with pytest.raises(ValueError, match="at least 2 points"):
            NozzleContour(x=x, y=y, contour_type="test")

    def test_length_property(self) -> None:
        """Test length property."""
        x = np.array([0.0, 0.1, 0.2, 0.3])
        y = np.array([0.05, 0.06, 0.07, 0.08])

        contour = NozzleContour(x=x, y=y, contour_type="test")

        assert contour.length == pytest.approx(0.3)

    def test_throat_radius_property(self) -> None:
        """Test throat_radius property (minimum y)."""
        x = np.array([0.0, 0.1, 0.2, 0.3])
        y = np.array([0.06, 0.05, 0.07, 0.08])  # Minimum at index 1

        contour = NozzleContour(x=x, y=y, contour_type="test")

        assert contour.throat_radius == pytest.approx(0.05)

    def test_exit_radius_property(self) -> None:
        """Test exit_radius property (last y value)."""
        x = np.array([0.0, 0.1, 0.2, 0.3])
        y = np.array([0.05, 0.06, 0.07, 0.08])

        contour = NozzleContour(x=x, y=y, contour_type="test")

        assert contour.exit_radius == pytest.approx(0.08)

    def test_inlet_radius_property(self) -> None:
        """Test inlet_radius property (first y value)."""
        x = np.array([0.0, 0.1, 0.2, 0.3])
        y = np.array([0.05, 0.06, 0.07, 0.08])

        contour = NozzleContour(x=x, y=y, contour_type="test")

        assert contour.inlet_radius == pytest.approx(0.05)

    def test_to_arrays_mm(self) -> None:
        """Test conversion to millimeters."""
        x = np.array([0.0, 0.001, 0.002])  # meters
        y = np.array([0.001, 0.0015, 0.002])

        contour = NozzleContour(x=x, y=y, contour_type="test")
        x_mm, y_mm = contour.to_arrays_mm()

        assert x_mm[1] == pytest.approx(1.0)  # 0.001 m = 1 mm
        assert y_mm[0] == pytest.approx(1.0)

    def test_to_csv(self) -> None:
        """Test CSV export."""
        x = np.array([0.0, 0.01, 0.02])
        y = np.array([0.005, 0.006, 0.008])

        contour = NozzleContour(x=x, y=y, contour_type="test")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = Path(f.name)

        try:
            contour.to_csv(path)

            # Read and verify
            content = path.read_text()
            lines = content.strip().split("\n")

            assert len(lines) == 4  # Header + 3 data rows
            assert "x_m" in lines[0]
            assert "y_m" in lines[0]
        finally:
            path.unlink()


class TestConicalContour:
    """Test conical nozzle contour generation."""

    def test_conical_basic(self) -> None:
        """Test basic conical contour generation."""
        contour = conical_contour(
            throat_radius=meters(0.05),
            exit_radius=meters(0.15),
            half_angle=15.0,
            num_points=50,
        )

        assert contour.contour_type == "conical"
        assert len(contour.x) == 50

    def test_conical_starts_at_throat(self) -> None:
        """Test that conical contour starts at throat."""
        Rt = 0.05
        contour = conical_contour(
            throat_radius=meters(Rt),
            exit_radius=meters(0.15),
            half_angle=15.0,
        )

        assert contour.x[0] == pytest.approx(0.0)
        assert contour.y[0] == pytest.approx(Rt)

    def test_conical_ends_at_exit(self) -> None:
        """Test that conical contour ends at exit."""
        Re = 0.15
        contour = conical_contour(
            throat_radius=meters(0.05),
            exit_radius=meters(Re),
            half_angle=15.0,
        )

        assert contour.y[-1] == pytest.approx(Re)

    def test_conical_length_matches_geometry(self) -> None:
        """Test that conical length matches geometric calculation."""
        Rt = 0.05
        Re = 0.15
        half_angle = 15.0

        contour = conical_contour(
            throat_radius=meters(Rt),
            exit_radius=meters(Re),
            half_angle=half_angle,
        )

        expected_length = (Re - Rt) / math.tan(math.radians(half_angle))
        assert contour.length == pytest.approx(expected_length, rel=0.01)

    def test_conical_linear_profile(self) -> None:
        """Test that conical contour is linear."""
        contour = conical_contour(
            throat_radius=meters(0.05),
            exit_radius=meters(0.15),
            half_angle=15.0,
            num_points=100,
        )

        # Check linearity: y should increase linearly with x
        # dy/dx should be approximately constant (tan of half angle)
        expected_slope = math.tan(math.radians(15.0))

        for i in range(1, len(contour.x)):
            slope = (contour.y[i] - contour.y[i - 1]) / (contour.x[i] - contour.x[i - 1])
            assert slope == pytest.approx(expected_slope, rel=0.01)


class TestRaoBellContour:
    """Test Rao bell nozzle contour generation."""

    def test_rao_bell_basic(self) -> None:
        """Test basic Rao bell contour generation."""
        contour = rao_bell_contour(
            throat_radius=meters(0.05),
            exit_radius=meters(0.15),
            expansion_ratio=9.0,
            bell_fraction=0.8,
            num_points=100,
        )

        assert contour.contour_type == "rao_bell"
        assert len(contour.x) > 0

    def test_rao_bell_starts_near_throat(self) -> None:
        """Test that Rao bell starts near throat."""
        Rt = 0.05
        contour = rao_bell_contour(
            throat_radius=meters(Rt),
            exit_radius=meters(0.15),
            expansion_ratio=9.0,
        )

        # Should start at x=0 (throat)
        assert contour.x[0] == pytest.approx(0.0, abs=1e-6)
        # First y should be at throat radius
        assert contour.y[0] == pytest.approx(Rt, rel=0.01)

    def test_rao_bell_ends_at_exit(self) -> None:
        """Test that Rao bell ends at exit radius."""
        Re = 0.15
        contour = rao_bell_contour(
            throat_radius=meters(0.05),
            exit_radius=meters(Re),
            expansion_ratio=9.0,
        )

        assert contour.y[-1] == pytest.approx(Re, rel=0.01)

    def test_rao_bell_shorter_than_conical(self) -> None:
        """Test that 80% bell is shorter than 15° cone."""
        Rt = 0.05
        Re = 0.15

        conical = conical_contour(
            throat_radius=meters(Rt), exit_radius=meters(Re), half_angle=15.0
        )

        bell = rao_bell_contour(
            throat_radius=meters(Rt),
            exit_radius=meters(Re),
            expansion_ratio=9.0,
            bell_fraction=0.8,
        )

        assert bell.length < conical.length

    def test_rao_bell_monotonic_y(self) -> None:
        """Test that y increases monotonically (no wiggles after throat arc)."""
        contour = rao_bell_contour(
            throat_radius=meters(0.05),
            exit_radius=meters(0.15),
            expansion_ratio=9.0,
        )

        # After the initial throat arc, y should generally increase
        # (allowing for small numerical variations)
        mid_idx = len(contour.y) // 4  # Start checking after throat region
        for i in range(mid_idx, len(contour.y) - 1):
            assert contour.y[i + 1] >= contour.y[i] - 1e-10


class TestGenerateNozzleFromGeometry:
    """Test convenience function to generate nozzle from geometry."""

    @pytest.fixture
    def sample_geometry(self):
        """Create sample engine geometry."""
        inputs = EngineInputs(
            thrust=newtons(5000),
            chamber_pressure=megapascals(2.0),
            chamber_temp=kelvin(3200),
            exit_pressure=pascals(101325),
            molecular_weight=22.0,
            gamma=1.2,
            lstar=meters(1.0),
            mixture_ratio=2.0,
        )
        _, geometry = design_engine(inputs)
        return geometry

    def test_generates_valid_contour(self, sample_geometry) -> None:
        """Test that function generates valid contour."""
        contour = generate_nozzle_from_geometry(sample_geometry)

        assert isinstance(contour, NozzleContour)
        assert contour.contour_type == "rao_bell"

    def test_matches_geometry_dimensions(self, sample_geometry) -> None:
        """Test that contour matches geometry dimensions."""
        contour = generate_nozzle_from_geometry(sample_geometry)

        Rt = sample_geometry.throat_diameter.to("m").value / 2
        Re = sample_geometry.exit_diameter.to("m").value / 2

        # Throat radius should match
        assert contour.throat_radius == pytest.approx(Rt, rel=0.05)
        # Exit radius should match
        assert contour.exit_radius == pytest.approx(Re, rel=0.05)


class TestFullChamberContour:
    """Test full chamber contour generation."""

    @pytest.fixture
    def sample_design(self):
        """Create sample engine design."""
        inputs = EngineInputs(
            thrust=newtons(5000),
            chamber_pressure=megapascals(2.0),
            chamber_temp=kelvin(3200),
            exit_pressure=pascals(101325),
            molecular_weight=22.0,
            gamma=1.2,
            lstar=meters(1.0),
            mixture_ratio=2.0,
        )
        perf, geometry = design_engine(inputs)
        return inputs, geometry

    def test_full_contour_longer(self, sample_design) -> None:
        """Test that full contour is longer than nozzle alone."""
        inputs, geometry = sample_design

        nozzle_only = generate_nozzle_from_geometry(geometry)
        full = full_chamber_contour(inputs, geometry, nozzle_only)

        assert full.length > nozzle_only.length

    def test_full_contour_starts_at_chamber(self, sample_design) -> None:
        """Test that full contour starts at chamber radius."""
        inputs, geometry = sample_design

        nozzle_only = generate_nozzle_from_geometry(geometry)
        full = full_chamber_contour(inputs, geometry, nozzle_only)

        Rc = geometry.chamber_diameter.to("m").value / 2

        # First point should be near chamber radius
        assert full.inlet_radius == pytest.approx(Rc, rel=0.1)

    def test_full_contour_includes_throat(self, sample_design) -> None:
        """Test that full contour includes throat region."""
        inputs, geometry = sample_design

        nozzle_only = generate_nozzle_from_geometry(geometry)
        full = full_chamber_contour(inputs, geometry, nozzle_only)

        Rt = geometry.throat_diameter.to("m").value / 2

        # Minimum radius should be approximately throat radius
        assert full.throat_radius == pytest.approx(Rt, rel=0.05)

    def test_full_contour_ends_at_exit(self, sample_design) -> None:
        """Test that full contour ends at exit radius."""
        inputs, geometry = sample_design

        nozzle_only = generate_nozzle_from_geometry(geometry)
        full = full_chamber_contour(inputs, geometry, nozzle_only)

        Re = geometry.exit_diameter.to("m").value / 2

        assert full.exit_radius == pytest.approx(Re, rel=0.05)

