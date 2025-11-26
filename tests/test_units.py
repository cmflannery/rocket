"""Tests for the units module."""


import pytest

from rocket.units import (
    Quantity,
    atmospheres,
    bar,
    centimeters,
    dimensionless,
    feet,
    feet_per_second,
    inches,
    kelvin,
    kg_per_second,
    kilograms,
    kilonewtons,
    kilopascals,
    megapascals,
    meters,
    meters_per_second,
    millimeters,
    newtons,
    pascals,
    pounds_force,
    pounds_mass,
    psi,
    rankine,
    seconds,
    square_meters,
)


class TestQuantityCreation:
    """Test Quantity creation and validation."""

    def test_create_basic_quantity(self) -> None:
        """Test creating a basic quantity."""
        q = Quantity(10.0, "m", "length")
        assert q.value == 10.0
        assert q.unit == "m"
        assert q.dimension == "length"

    def test_invalid_unit_raises_error(self) -> None:
        """Test that invalid units raise ValueError."""
        with pytest.raises(ValueError, match="Unknown unit"):
            Quantity(10.0, "invalid_unit", "length")

    def test_mismatched_dimension_raises_error(self) -> None:
        """Test that mismatched dimension raises ValueError."""
        with pytest.raises(ValueError, match="has dimension"):
            Quantity(10.0, "m", "pressure")  # m is length, not pressure

    def test_frozen_dataclass(self) -> None:
        """Test that Quantity is immutable."""
        q = meters(10.0)
        with pytest.raises(AttributeError):
            q.value = 20.0  # type: ignore


class TestFactoryFunctions:
    """Test factory functions for creating quantities."""

    def test_meters(self) -> None:
        q = meters(5.0)
        assert q.value == 5.0
        assert q.unit == "m"
        assert q.dimension == "length"

    def test_centimeters(self) -> None:
        q = centimeters(100.0)
        assert q.value == 100.0
        assert q.unit == "cm"

    def test_millimeters(self) -> None:
        q = millimeters(1000.0)
        assert q.value == 1000.0
        assert q.unit == "mm"

    def test_feet(self) -> None:
        q = feet(3.281)
        assert q.value == pytest.approx(3.281)
        assert q.unit == "ft"

    def test_inches(self) -> None:
        q = inches(12.0)
        assert q.value == 12.0
        assert q.unit == "in"

    def test_kilograms(self) -> None:
        q = kilograms(100.0)
        assert q.unit == "kg"
        assert q.dimension == "mass"

    def test_pounds_mass(self) -> None:
        q = pounds_mass(2.205)
        assert q.unit == "lbm"

    def test_newtons(self) -> None:
        q = newtons(1000.0)
        assert q.unit == "N"
        assert q.dimension == "force"

    def test_kilonewtons(self) -> None:
        q = kilonewtons(1.0)
        assert q.unit == "kN"

    def test_pounds_force(self) -> None:
        q = pounds_force(224.8)
        assert q.unit == "lbf"

    def test_pascals(self) -> None:
        q = pascals(101325.0)
        assert q.unit == "Pa"
        assert q.dimension == "pressure"

    def test_kilopascals(self) -> None:
        q = kilopascals(101.325)
        assert q.unit == "kPa"

    def test_megapascals(self) -> None:
        q = megapascals(0.101325)
        assert q.unit == "MPa"

    def test_bar(self) -> None:
        q = bar(1.01325)
        assert q.unit == "bar"

    def test_psi(self) -> None:
        q = psi(14.696)
        assert q.unit == "psi"

    def test_atmospheres(self) -> None:
        q = atmospheres(1.0)
        assert q.unit == "atm"

    def test_kelvin(self) -> None:
        q = kelvin(300.0)
        assert q.unit == "K"
        assert q.dimension == "temperature"

    def test_rankine(self) -> None:
        q = rankine(540.0)
        assert q.unit == "R"

    def test_meters_per_second(self) -> None:
        q = meters_per_second(340.0)
        assert q.unit == "m/s"
        assert q.dimension == "velocity"

    def test_feet_per_second(self) -> None:
        q = feet_per_second(1116.0)
        assert q.unit == "ft/s"

    def test_seconds(self) -> None:
        q = seconds(60.0)
        assert q.unit == "s"
        assert q.dimension == "time"

    def test_square_meters(self) -> None:
        q = square_meters(1.0)
        assert q.unit == "m^2"
        assert q.dimension == "area"

    def test_kg_per_second(self) -> None:
        q = kg_per_second(10.0)
        assert q.unit == "kg/s"
        assert q.dimension == "mass_flow"

    def test_dimensionless(self) -> None:
        q = dimensionless(3.14159)
        assert q.unit == "1"
        assert q.dimension == "dimensionless"


class TestUnitConversion:
    """Test unit conversion functionality."""

    def test_meters_to_feet(self) -> None:
        q = meters(1.0)
        q_ft = q.to("ft")
        assert q_ft.value == pytest.approx(3.28084, rel=1e-4)
        assert q_ft.unit == "ft"
        assert q_ft.dimension == "length"

    def test_feet_to_meters(self) -> None:
        q = feet(3.28084)
        q_m = q.to("m")
        assert q_m.value == pytest.approx(1.0, rel=1e-4)

    def test_meters_to_centimeters(self) -> None:
        q = meters(1.0)
        q_cm = q.to("cm")
        assert q_cm.value == pytest.approx(100.0)

    def test_pascals_to_psi(self) -> None:
        q = pascals(101325.0)
        q_psi = q.to("psi")
        assert q_psi.value == pytest.approx(14.696, rel=1e-3)

    def test_psi_to_pascals(self) -> None:
        q = psi(14.696)
        q_pa = q.to("Pa")
        assert q_pa.value == pytest.approx(101325.0, rel=1e-3)

    def test_newtons_to_lbf(self) -> None:
        q = newtons(4.44822)
        q_lbf = q.to("lbf")
        assert q_lbf.value == pytest.approx(1.0, rel=1e-4)

    def test_lbf_to_newtons(self) -> None:
        q = pounds_force(1.0)
        q_n = q.to("N")
        assert q_n.value == pytest.approx(4.44822, rel=1e-4)

    def test_to_si(self) -> None:
        q = feet(10.0)
        q_si = q.to_si()
        assert q_si.unit == "m"
        assert q_si.value == pytest.approx(3.048, rel=1e-4)

    def test_si_value_property(self) -> None:
        q = feet(10.0)
        assert q.si_value == pytest.approx(3.048, rel=1e-4)

    def test_conversion_incompatible_dimensions(self) -> None:
        q = meters(1.0)
        with pytest.raises(ValueError, match="Cannot convert between different dimensions"):
            q.to("Pa")

    def test_round_trip_conversion(self) -> None:
        """Test that converting back and forth preserves value."""
        original = meters(123.456)
        converted = original.to("ft").to("in").to("cm").to("m")
        assert converted.value == pytest.approx(original.value, rel=1e-9)


class TestArithmetic:
    """Test arithmetic operations on quantities."""

    def test_add_same_units(self) -> None:
        q1 = meters(5.0)
        q2 = meters(3.0)
        result = q1 + q2
        assert result.value == 8.0
        assert result.unit == "m"

    def test_add_different_units_same_dimension(self) -> None:
        q1 = meters(1.0)
        q2 = centimeters(50.0)
        result = q1 + q2
        assert result.value == pytest.approx(1.5)
        assert result.unit == "m"

    def test_add_different_dimensions_raises(self) -> None:
        q1 = meters(1.0)
        q2 = pascals(100.0)
        with pytest.raises(ValueError, match="different dimensions"):
            q1 + q2

    def test_subtract_same_units(self) -> None:
        q1 = meters(5.0)
        q2 = meters(3.0)
        result = q1 - q2
        assert result.value == 2.0

    def test_subtract_different_units_same_dimension(self) -> None:
        q1 = meters(2.0)
        q2 = centimeters(50.0)
        result = q1 - q2
        assert result.value == pytest.approx(1.5)

    def test_multiply_by_scalar(self) -> None:
        q = meters(5.0)
        result = q * 2.0
        assert result.value == 10.0
        assert result.unit == "m"

    def test_rmul_scalar(self) -> None:
        q = meters(5.0)
        result = 2.0 * q
        assert result.value == 10.0

    def test_divide_by_scalar(self) -> None:
        q = meters(10.0)
        result = q / 2.0
        assert result.value == 5.0
        assert result.unit == "m"

    def test_negate(self) -> None:
        q = meters(5.0)
        result = -q
        assert result.value == -5.0

    def test_abs(self) -> None:
        q = meters(-5.0)
        result = abs(q)
        assert result.value == 5.0

    def test_multiply_length_by_length(self) -> None:
        """Test that length * length = area."""
        q1 = meters(2.0)
        q2 = meters(3.0)
        result = q1 * q2
        assert result.dimension == "area"
        assert result.value == pytest.approx(6.0)

    def test_divide_area_by_length(self) -> None:
        """Test that area / length = length."""
        area = square_meters(10.0)
        length = meters(2.0)
        result = area / length
        assert result.dimension == "length"
        assert result.value == pytest.approx(5.0)


class TestComparison:
    """Test comparison operations."""

    def test_equality_same_units(self) -> None:
        q1 = meters(5.0)
        q2 = meters(5.0)
        assert q1 == q2

    def test_equality_different_units(self) -> None:
        q1 = meters(1.0)
        q2 = centimeters(100.0)
        assert q1 == q2

    def test_inequality(self) -> None:
        q1 = meters(5.0)
        q2 = meters(3.0)
        assert q1 != q2

    def test_less_than(self) -> None:
        q1 = meters(3.0)
        q2 = meters(5.0)
        assert q1 < q2

    def test_less_than_different_units(self) -> None:
        q1 = meters(0.5)
        q2 = centimeters(100.0)
        assert q1 < q2

    def test_greater_than(self) -> None:
        q1 = meters(5.0)
        q2 = meters(3.0)
        assert q1 > q2

    def test_less_than_or_equal(self) -> None:
        q1 = meters(3.0)
        q2 = meters(3.0)
        assert q1 <= q2

    def test_greater_than_or_equal(self) -> None:
        q1 = meters(5.0)
        q2 = meters(5.0)
        assert q1 >= q2

    def test_compare_different_dimensions_raises(self) -> None:
        q1 = meters(5.0)
        q2 = pascals(5.0)
        with pytest.raises(ValueError, match="different dimensions"):
            _ = q1 < q2


class TestStringRepresentation:
    """Test string representations."""

    def test_repr(self) -> None:
        q = meters(5.123456)
        assert "5.12346" in repr(q)
        assert "m" in repr(q)

    def test_str(self) -> None:
        q = meters(5.123456)
        s = str(q)
        assert "5.12346" in s
        assert "m" in s


class TestHashing:
    """Test hash behavior for use in sets/dicts."""

    def test_hash_equal_quantities(self) -> None:
        """Equal quantities should have the same hash."""
        q1 = meters(5.0)
        q2 = meters(5.0)
        assert hash(q1) == hash(q2)

    def test_hash_equivalent_quantities(self) -> None:
        """Quantities equal in SI should have the same hash."""
        q1 = meters(1.0)
        q2 = centimeters(100.0)
        assert hash(q1) == hash(q2)

    def test_use_in_set(self) -> None:
        """Test that quantities can be used in sets."""
        s = {meters(1.0), meters(2.0), meters(1.0)}
        assert len(s) == 2

