"""Units module for Rocket.

Provides a Quantity class for type-safe physical quantities with unit conversion.
All physical values in the library should use Quantity, never bare floats.

Design principles:
- Explicit over implicit: all conversions require calling .to()
- Type safe: beartype checks at runtime
- Immutable: frozen dataclasses prevent accidental mutation
- No magic: clear, predictable behavior
"""

import math
from dataclasses import dataclass

from beartype import beartype

# =============================================================================
# Dimension and Unit Definitions
# =============================================================================

# Dimensions represent physical quantities (length, mass, time, etc.)
# Each dimension has a base SI unit

DIMENSIONS = {
    "length": "m",
    "mass": "kg",
    "time": "s",
    "temperature": "K",
    "force": "N",
    "pressure": "Pa",
    "velocity": "m/s",
    "area": "m^2",
    "volume": "m^3",
    "mass_flow": "kg/s",
    "density": "kg/m^3",
    "specific_impulse": "s",
    "dimensionless": "1",
}

# Conversion factors TO base SI unit
# e.g., 1 ft = 0.3048 m, so CONVERSIONS["ft"] = 0.3048
CONVERSIONS: dict[str, tuple[float, str]] = {
    # Length
    "m": (1.0, "length"),
    "cm": (0.01, "length"),
    "mm": (0.001, "length"),
    "km": (1000.0, "length"),
    "ft": (0.3048, "length"),
    "in": (0.0254, "length"),
    "inch": (0.0254, "length"),
    "inches": (0.0254, "length"),
    # Mass
    "kg": (1.0, "mass"),
    "g": (0.001, "mass"),
    "lbm": (0.453592, "mass"),
    "slug": (14.5939, "mass"),
    # Time
    "s": (1.0, "time"),
    "ms": (0.001, "time"),
    "min": (60.0, "time"),
    "hr": (3600.0, "time"),
    # Temperature (special - requires offset handling)
    "K": (1.0, "temperature"),
    "R": (5 / 9, "temperature"),  # Rankine to Kelvin (multiply only, offset handled separately)
    # Force
    "N": (1.0, "force"),
    "kN": (1000.0, "force"),
    "MN": (1e6, "force"),
    "lbf": (4.44822, "force"),
    "kgf": (9.80665, "force"),
    # Pressure
    "Pa": (1.0, "pressure"),
    "kPa": (1000.0, "pressure"),
    "MPa": (1e6, "pressure"),
    "bar": (1e5, "pressure"),
    "atm": (101325.0, "pressure"),
    "psi": (6894.76, "pressure"),
    "psia": (6894.76, "pressure"),
    # Velocity
    "m/s": (1.0, "velocity"),
    "km/s": (1000.0, "velocity"),
    "ft/s": (0.3048, "velocity"),
    # Area
    "m^2": (1.0, "area"),
    "cm^2": (1e-4, "area"),
    "mm^2": (1e-6, "area"),
    "ft^2": (0.092903, "area"),
    "in^2": (0.00064516, "area"),
    # Volume
    "m^3": (1.0, "volume"),
    "L": (0.001, "volume"),
    "cm^3": (1e-6, "volume"),
    "ft^3": (0.0283168, "volume"),
    "in^3": (1.6387e-5, "volume"),
    # Mass flow rate
    "kg/s": (1.0, "mass_flow"),
    "lbm/s": (0.453592, "mass_flow"),
    # Density
    "kg/m^3": (1.0, "density"),
    "lbm/ft^3": (16.0185, "density"),
    # Power
    "W": (1.0, "power"),
    "kW": (1000.0, "power"),
    "MW": (1e6, "power"),
    "hp": (745.7, "power"),  # Mechanical horsepower
    # Specific impulse (time dimension but special meaning)
    # Note: Isp in seconds is the same in SI and Imperial
    # Dimensionless
    "1": (1.0, "dimensionless"),
    "": (1.0, "dimensionless"),
}


def _get_dimension(unit: str) -> str:
    """Get the dimension for a unit string."""
    if unit not in CONVERSIONS:
        raise ValueError(f"Unknown unit: {unit!r}")
    return CONVERSIONS[unit][1]


def _get_conversion_factor(unit: str) -> float:
    """Get the conversion factor to SI base unit."""
    if unit not in CONVERSIONS:
        raise ValueError(f"Unknown unit: {unit!r}")
    return CONVERSIONS[unit][0]


def _convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a value between units of the same dimension."""
    from_dim = _get_dimension(from_unit)
    to_dim = _get_dimension(to_unit)

    if from_dim != to_dim:
        raise ValueError(
            f"Cannot convert between different dimensions: {from_dim} and {to_dim}"
        )

    # Convert to SI base, then to target
    si_value = value * _get_conversion_factor(from_unit)
    return si_value / _get_conversion_factor(to_unit)


# =============================================================================
# Quantity Class
# =============================================================================


@beartype
@dataclass(frozen=True, slots=True)
class Quantity:
    """A physical quantity with value, unit, and dimension.

    Quantities are immutable and support arithmetic operations that respect
    dimensional analysis.

    Examples:
        >>> thrust = Quantity(50000, "N", "force")
        >>> thrust_lbf = thrust.to("lbf")
        >>> print(thrust_lbf)
        Quantity(11240.45 lbf)

        >>> length = meters(2.5)
        >>> area = length * length  # Returns Quantity with area dimension
    """

    value: float | int
    unit: str
    dimension: str

    def __post_init__(self) -> None:
        """Validate that unit matches dimension."""
        if self.unit not in CONVERSIONS:
            raise ValueError(f"Unknown unit: {self.unit!r}")
        expected_dim = CONVERSIONS[self.unit][1]
        if self.dimension != expected_dim:
            raise ValueError(
                f"Unit {self.unit!r} has dimension {expected_dim!r}, "
                f"but {self.dimension!r} was specified"
            )

    def to(self, target_unit: str) -> "Quantity":
        """Convert to a different unit of the same dimension.

        Args:
            target_unit: The unit to convert to

        Returns:
            A new Quantity with the converted value and new unit

        Raises:
            ValueError: If target_unit is incompatible dimension
        """
        new_value = _convert(self.value, self.unit, target_unit)
        return Quantity(new_value, target_unit, self.dimension)

    def to_si(self) -> "Quantity":
        """Convert to SI base unit for this dimension."""
        si_unit = DIMENSIONS[self.dimension]
        return self.to(si_unit)

    @property
    def si_value(self) -> float:
        """Get the value in SI base units without creating new Quantity."""
        return self.value * _get_conversion_factor(self.unit)

    def __repr__(self) -> str:
        return f"Quantity({self.value:.6g} {self.unit})"

    def __str__(self) -> str:
        return f"{self.value:.6g} {self.unit}"

    # -------------------------------------------------------------------------
    # Arithmetic Operations
    # -------------------------------------------------------------------------

    def __add__(self, other: "Quantity") -> "Quantity":
        """Add two quantities of the same dimension."""
        if not isinstance(other, Quantity):
            raise TypeError(f"Cannot add Quantity and {type(other).__name__}")
        if self.dimension != other.dimension:
            raise ValueError(
                f"Cannot add quantities with different dimensions: "
                f"{self.dimension} and {other.dimension}"
            )
        # Convert other to same unit as self, then add
        other_converted = other.to(self.unit)
        return Quantity(self.value + other_converted.value, self.unit, self.dimension)

    def __sub__(self, other: "Quantity") -> "Quantity":
        """Subtract two quantities of the same dimension."""
        if not isinstance(other, Quantity):
            raise TypeError(f"Cannot subtract Quantity and {type(other).__name__}")
        if self.dimension != other.dimension:
            raise ValueError(
                f"Cannot subtract quantities with different dimensions: "
                f"{self.dimension} and {other.dimension}"
            )
        other_converted = other.to(self.unit)
        return Quantity(self.value - other_converted.value, self.unit, self.dimension)

    def __mul__(self, other: "Quantity | float | int") -> "Quantity":
        """Multiply by a scalar or another quantity."""
        if isinstance(other, (int, float)):
            return Quantity(self.value * other, self.unit, self.dimension)
        if isinstance(other, Quantity):
            # Dimensional multiplication - result dimension depends on operands
            new_dim, new_unit = _multiply_dimensions(
                self.dimension, self.unit, other.dimension, other.unit
            )
            new_value = self.si_value * other.si_value
            # Convert back from SI to the derived unit
            new_value = new_value / _get_conversion_factor(new_unit)
            return Quantity(new_value, new_unit, new_dim)
        raise TypeError(f"Cannot multiply Quantity by {type(other).__name__}")

    def __rmul__(self, other: float | int) -> "Quantity":
        """Right multiply by scalar."""
        if isinstance(other, (int, float)):
            return Quantity(self.value * other, self.unit, self.dimension)
        raise TypeError(f"Cannot multiply {type(other).__name__} by Quantity")

    def __truediv__(self, other: "Quantity | float | int") -> "Quantity":
        """Divide by a scalar or another quantity."""
        if isinstance(other, (int, float)):
            return Quantity(self.value / other, self.unit, self.dimension)
        if isinstance(other, Quantity):
            new_dim, new_unit = _divide_dimensions(
                self.dimension, self.unit, other.dimension, other.unit
            )
            new_value = self.si_value / other.si_value
            new_value = new_value / _get_conversion_factor(new_unit)
            return Quantity(new_value, new_unit, new_dim)
        raise TypeError(f"Cannot divide Quantity by {type(other).__name__}")

    def __rtruediv__(self, other: float | int) -> "Quantity":
        """Right division (scalar / Quantity) - returns inverse dimension."""
        if isinstance(other, (int, float)):
            # This would create an inverse dimension which we don't fully support
            # For now, raise an error
            raise TypeError(
                "Division of scalar by Quantity not supported. "
                "Use explicit inverse units instead."
            )
        raise TypeError(f"Cannot divide {type(other).__name__} by Quantity")

    def __neg__(self) -> "Quantity":
        """Negate the quantity."""
        return Quantity(-self.value, self.unit, self.dimension)

    def __pos__(self) -> "Quantity":
        """Positive (returns copy)."""
        return Quantity(self.value, self.unit, self.dimension)

    def __abs__(self) -> "Quantity":
        """Absolute value."""
        return Quantity(abs(self.value), self.unit, self.dimension)

    # -------------------------------------------------------------------------
    # Comparison Operations
    # -------------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Check equality (compares SI values for same dimension)."""
        if not isinstance(other, Quantity):
            return NotImplemented
        if self.dimension != other.dimension:
            return False
        # Compare in SI units to handle unit differences
        return math.isclose(self.si_value, other.si_value, rel_tol=1e-9)

    def __lt__(self, other: "Quantity") -> bool:
        if not isinstance(other, Quantity):
            raise TypeError(f"Cannot compare Quantity with {type(other).__name__}")
        if self.dimension != other.dimension:
            raise ValueError("Cannot compare quantities with different dimensions")
        return self.si_value < other.si_value

    def __le__(self, other: "Quantity") -> bool:
        return self == other or self < other

    def __gt__(self, other: "Quantity") -> bool:
        if not isinstance(other, Quantity):
            raise TypeError(f"Cannot compare Quantity with {type(other).__name__}")
        if self.dimension != other.dimension:
            raise ValueError("Cannot compare quantities with different dimensions")
        return self.si_value > other.si_value

    def __ge__(self, other: "Quantity") -> bool:
        return self == other or self > other

    def __hash__(self) -> int:
        """Hash based on SI value and dimension for consistency."""
        return hash((round(self.si_value, 9), self.dimension))


# =============================================================================
# Dimension Algebra
# =============================================================================

# Multiplication table for dimensions
_MULT_TABLE: dict[tuple[str, str], str] = {
    ("length", "length"): "area",
    ("area", "length"): "volume",
    ("length", "area"): "volume",
    ("velocity", "time"): "length",
    ("mass", "velocity"): "force",  # Approximation: momentum
    ("force", "length"): "force",  # Work/energy - simplified
    ("pressure", "area"): "force",
    ("mass_flow", "velocity"): "force",
    ("mass_flow", "time"): "mass",
    ("density", "volume"): "mass",
    ("dimensionless", "length"): "length",
    ("dimensionless", "mass"): "mass",
    ("dimensionless", "force"): "force",
    ("dimensionless", "pressure"): "pressure",
    ("dimensionless", "velocity"): "velocity",
    ("dimensionless", "area"): "area",
    ("dimensionless", "volume"): "volume",
    ("dimensionless", "time"): "time",
    ("dimensionless", "temperature"): "temperature",
    ("dimensionless", "mass_flow"): "mass_flow",
    ("dimensionless", "density"): "density",
    ("dimensionless", "dimensionless"): "dimensionless",
}

# Division table for dimensions
_DIV_TABLE: dict[tuple[str, str], str] = {
    ("area", "length"): "length",
    ("volume", "length"): "area",
    ("volume", "area"): "length",
    ("length", "time"): "velocity",
    ("velocity", "time"): "velocity",  # acceleration - simplified
    ("mass", "volume"): "density",
    ("mass", "time"): "mass_flow",
    ("force", "area"): "pressure",
    ("force", "mass"): "velocity",  # acceleration simplified
    ("force", "pressure"): "area",
    ("force", "velocity"): "mass_flow",
    ("length", "length"): "dimensionless",
    ("mass", "mass"): "dimensionless",
    ("force", "force"): "dimensionless",
    ("pressure", "pressure"): "dimensionless",
    ("area", "area"): "dimensionless",
    ("volume", "volume"): "dimensionless",
    ("velocity", "velocity"): "dimensionless",
    ("time", "time"): "dimensionless",
    ("dimensionless", "dimensionless"): "dimensionless",
}


def _multiply_dimensions(
    dim1: str, unit1: str, dim2: str, unit2: str
) -> tuple[str, str]:
    """Determine result dimension and unit for multiplication."""
    # Check both orderings
    key = (dim1, dim2)
    if key in _MULT_TABLE:
        result_dim = _MULT_TABLE[key]
    elif (dim2, dim1) in _MULT_TABLE:
        result_dim = _MULT_TABLE[(dim2, dim1)]
    else:
        raise ValueError(
            f"Multiplication of {dim1} and {dim2} not supported. "
            "Result dimension is ambiguous."
        )

    result_unit = DIMENSIONS[result_dim]
    return result_dim, result_unit


def _divide_dimensions(dim1: str, unit1: str, dim2: str, unit2: str) -> tuple[str, str]:
    """Determine result dimension and unit for division."""
    key = (dim1, dim2)
    if key in _DIV_TABLE:
        result_dim = _DIV_TABLE[key]
    else:
        raise ValueError(
            f"Division of {dim1} by {dim2} not supported. "
            "Result dimension is ambiguous."
        )

    result_unit = DIMENSIONS[result_dim]
    return result_dim, result_unit


# =============================================================================
# Factory Functions - Clear, Explicit Quantity Creation
# =============================================================================


@beartype
def meters(value: float | int) -> Quantity:
    """Create a length quantity in meters."""
    return Quantity(value, "m", "length")


@beartype
def centimeters(value: float | int) -> Quantity:
    """Create a length quantity in centimeters."""
    return Quantity(value, "cm", "length")


@beartype
def millimeters(value: float | int) -> Quantity:
    """Create a length quantity in millimeters."""
    return Quantity(value, "mm", "length")


@beartype
def feet(value: float | int) -> Quantity:
    """Create a length quantity in feet."""
    return Quantity(value, "ft", "length")


@beartype
def inches(value: float | int) -> Quantity:
    """Create a length quantity in inches."""
    return Quantity(value, "in", "length")


@beartype
def kilograms(value: float | int) -> Quantity:
    """Create a mass quantity in kilograms."""
    return Quantity(value, "kg", "mass")


@beartype
def pounds_mass(value: float | int) -> Quantity:
    """Create a mass quantity in pounds-mass."""
    return Quantity(value, "lbm", "mass")


@beartype
def seconds(value: float | int) -> Quantity:
    """Create a time quantity in seconds."""
    return Quantity(value, "s", "time")


@beartype
def kelvin(value: float | int) -> Quantity:
    """Create a temperature quantity in Kelvin."""
    return Quantity(value, "K", "temperature")


@beartype
def rankine(value: float | int) -> Quantity:
    """Create a temperature quantity in Rankine."""
    return Quantity(value, "R", "temperature")


@beartype
def newtons(value: float | int) -> Quantity:
    """Create a force quantity in Newtons."""
    return Quantity(value, "N", "force")


@beartype
def kilonewtons(value: float | int) -> Quantity:
    """Create a force quantity in kilonewtons."""
    return Quantity(value, "kN", "force")


@beartype
def pounds_force(value: float | int) -> Quantity:
    """Create a force quantity in pounds-force."""
    return Quantity(value, "lbf", "force")


@beartype
def pascals(value: float | int) -> Quantity:
    """Create a pressure quantity in Pascals."""
    return Quantity(value, "Pa", "pressure")


@beartype
def kilopascals(value: float | int) -> Quantity:
    """Create a pressure quantity in kilopascals."""
    return Quantity(value, "kPa", "pressure")


@beartype
def megapascals(value: float | int) -> Quantity:
    """Create a pressure quantity in megapascals."""
    return Quantity(value, "MPa", "pressure")


@beartype
def bar(value: float | int) -> Quantity:
    """Create a pressure quantity in bar."""
    return Quantity(value, "bar", "pressure")


@beartype
def atmospheres(value: float | int) -> Quantity:
    """Create a pressure quantity in atmospheres."""
    return Quantity(value, "atm", "pressure")


@beartype
def psi(value: float | int) -> Quantity:
    """Create a pressure quantity in psi."""
    return Quantity(value, "psi", "pressure")


@beartype
def meters_per_second(value: float | int) -> Quantity:
    """Create a velocity quantity in m/s."""
    return Quantity(value, "m/s", "velocity")


@beartype
def feet_per_second(value: float | int) -> Quantity:
    """Create a velocity quantity in ft/s."""
    return Quantity(value, "ft/s", "velocity")


@beartype
def km_per_second(value: float | int) -> Quantity:
    """Create a velocity quantity in km/s."""
    return Quantity(value, "km/s", "velocity")


@beartype
def square_meters(value: float | int) -> Quantity:
    """Create an area quantity in m^2."""
    return Quantity(value, "m^2", "area")


@beartype
def square_centimeters(value: float | int) -> Quantity:
    """Create an area quantity in cm^2."""
    return Quantity(value, "cm^2", "area")


@beartype
def square_inches(value: float | int) -> Quantity:
    """Create an area quantity in in^2."""
    return Quantity(value, "in^2", "area")


@beartype
def cubic_meters(value: float | int) -> Quantity:
    """Create a volume quantity in m^3."""
    return Quantity(value, "m^3", "volume")


@beartype
def liters(value: float | int) -> Quantity:
    """Create a volume quantity in liters."""
    return Quantity(value, "L", "volume")


@beartype
def kg_per_second(value: float | int) -> Quantity:
    """Create a mass flow rate quantity in kg/s."""
    return Quantity(value, "kg/s", "mass_flow")


@beartype
def lbm_per_second(value: float | int) -> Quantity:
    """Create a mass flow rate quantity in lbm/s."""
    return Quantity(value, "lbm/s", "mass_flow")


@beartype
def kg_per_cubic_meter(value: float | int) -> Quantity:
    """Create a density quantity in kg/m^3."""
    return Quantity(value, "kg/m^3", "density")


@beartype
def dimensionless(value: float | int) -> Quantity:
    """Create a dimensionless quantity."""
    return Quantity(value, "1", "dimensionless")


@beartype
def watts(value: float | int) -> Quantity:
    """Create a power quantity in Watts."""
    return Quantity(value, "W", "power")


@beartype
def kilowatts(value: float | int) -> Quantity:
    """Create a power quantity in kilowatts."""
    return Quantity(value, "kW", "power")


@beartype
def megawatts(value: float | int) -> Quantity:
    """Create a power quantity in megawatts."""
    return Quantity(value, "MW", "power")


@beartype
def horsepower(value: float | int) -> Quantity:
    """Create a power quantity in horsepower."""
    return Quantity(value, "hp", "power")


# =============================================================================
# Constants
# =============================================================================

# Standard gravity
G0_SI = meters_per_second(9.80665)
G0_IMP = feet_per_second(32.174)

# Standard atmospheric pressure
ATM_SI = pascals(101325.0)
ATM_IMP = psi(14.696)

# Universal gas constant
R_UNIVERSAL_SI = 8314.46  # J/(kmol·K) - stored as float, used in calculations
R_UNIVERSAL_IMP = 1545.35  # ft·lbf/(lbmol·R)

