"""Mass properties modeling for rocket vehicles.

Provides time-varying mass, center of gravity, and inertia tensor
calculations as propellant is consumed.

Example:
    >>> from rocket.vehicle import MassProperties, Vehicle
    >>>
    >>> # Create vehicle
    >>> vehicle = Vehicle(
    ...     dry_mass=MassProperties(mass=500, cg=[0, 0, 2.5], inertia=I_dry),
    ...     propellant_mass=4500,
    ...     propellant_cg=[0, 0, 1.5],
    ...     engine_performance=engine_perf,
    ... )
    >>>
    >>> # Get current properties at 50% propellant remaining
    >>> props = vehicle.at_propellant_fraction(0.5)
    >>> print(f"Mass: {props.mass:.0f} kg, CG: {props.cg[2]:.2f} m")
"""

from dataclasses import dataclass, field

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

from rocket.engine import EnginePerformance

# =============================================================================
# Mass Properties
# =============================================================================


@beartype
@dataclass
class MassProperties:
    """Mass properties for a rigid body.

    Attributes:
        mass: Total mass [kg]
        cg: Center of gravity in body frame [x, y, z] [m]
        inertia: 3x3 inertia tensor about CG in body frame [kg*m^2]
    """
    mass: float
    cg: NDArray[np.float64]
    inertia: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate and convert inputs."""
        self.cg = np.asarray(self.cg, dtype=np.float64)
        self.inertia = np.asarray(self.inertia, dtype=np.float64)

        if self.cg.shape != (3,):
            raise ValueError(f"CG must be shape (3,), got {self.cg.shape}")
        if self.inertia.shape != (3, 3):
            raise ValueError(f"Inertia must be shape (3, 3), got {self.inertia.shape}")
        if self.mass <= 0:
            raise ValueError(f"Mass must be positive, got {self.mass}")

    @classmethod
    def from_principal(
        cls,
        mass: float,
        cg: list[float] | NDArray[np.float64],
        Ixx: float,
        Iyy: float,
        Izz: float,
    ) -> "MassProperties":
        """Create from principal moments of inertia.

        Args:
            mass: Total mass [kg]
            cg: Center of gravity [m]
            Ixx, Iyy, Izz: Principal moments of inertia [kg*m^2]
        """
        inertia = np.diag([Ixx, Iyy, Izz])
        return cls(mass=mass, cg=np.asarray(cg), inertia=inertia)

    def translate_inertia(self, offset: NDArray[np.float64]) -> NDArray[np.float64]:
        """Translate inertia tensor to a new reference point.

        Uses parallel axis theorem: I_new = I_cg + m * (d^2 * I - d ⊗ d)

        Args:
            offset: Translation vector from CG to new point [m]

        Returns:
            Inertia tensor about new point
        """
        d = np.asarray(offset)
        d_sq = np.dot(d, d)
        I_offset = self.mass * (d_sq * np.eye(3) - np.outer(d, d))
        return self.inertia + I_offset

    def __add__(self, other: "MassProperties") -> "MassProperties":
        """Combine two mass properties (for staging, etc.)."""
        total_mass = self.mass + other.mass

        # Combined CG
        combined_cg = (self.mass * self.cg + other.mass * other.cg) / total_mass

        # Translate both inertias to combined CG
        offset_self = combined_cg - self.cg
        offset_other = combined_cg - other.cg

        I_self = self.translate_inertia(offset_self)
        I_other = other.translate_inertia(offset_other)

        combined_inertia = I_self + I_other

        return MassProperties(
            mass=total_mass,
            cg=combined_cg,
            inertia=combined_inertia,
        )


# =============================================================================
# Inertia Utilities
# =============================================================================


@beartype
def compute_inertia_cylinder(
    mass: float,
    radius: float,
    length: float,
    hollow: bool = False,
    inner_radius: float = 0.0,
) -> NDArray[np.float64]:
    """Compute inertia tensor for a cylinder.

    Assumes cylinder axis is along Z (body axis).

    Args:
        mass: Total mass [kg]
        radius: Outer radius [m]
        length: Length [m]
        hollow: Whether cylinder is hollow
        inner_radius: Inner radius for hollow cylinder [m]

    Returns:
        3x3 inertia tensor about geometric center [kg*m^2]
    """
    r = radius
    L = length

    if hollow and inner_radius > 0:
        r_i = inner_radius
        # Hollow cylinder
        Ixx = (mass / 12) * (3 * (r**2 + r_i**2) + L**2)
        Iyy = Ixx
        Izz = (mass / 2) * (r**2 + r_i**2)
    else:
        # Solid cylinder
        Ixx = (mass / 12) * (3 * r**2 + L**2)
        Iyy = Ixx
        Izz = (mass / 2) * r**2

    return np.diag([Ixx, Iyy, Izz])


@beartype
def compute_inertia_sphere(
    mass: float,
    radius: float,
    hollow: bool = False,
) -> NDArray[np.float64]:
    """Compute inertia tensor for a sphere.

    Args:
        mass: Total mass [kg]
        radius: Radius [m]
        hollow: Whether sphere is hollow (thin shell)

    Returns:
        3x3 inertia tensor about center [kg*m^2]
    """
    inertia = 2 / 3 * mass * radius ** 2 if hollow else 2 / 5 * mass * radius ** 2

    return np.diag([inertia, inertia, inertia])


# =============================================================================
# Vehicle Model
# =============================================================================


@beartype
@dataclass
class Vehicle:
    """Complete rocket vehicle model.

    Combines dry mass, propellant, and engine to provide time-varying
    mass properties as propellant is consumed.

    Attributes:
        dry_mass: Dry mass properties (structure, engine, payload)
        initial_propellant_mass: Initial propellant mass [kg]
        propellant_cg: Propellant center of gravity (full) [m]
        propellant_inertia: Propellant inertia tensor (full) [kg*m^2]
        engine: Engine performance data
        reference_area: Aerodynamic reference area [m^2]
        reference_length: Aerodynamic reference length [m]
    """
    dry_mass: MassProperties
    initial_propellant_mass: float
    propellant_cg: NDArray[np.float64] = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    propellant_inertia: NDArray[np.float64] | None = None
    engine: EnginePerformance | None = None
    reference_area: float = 1.0  # [m^2]
    reference_length: float = 1.0  # [m]

    _current_propellant: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize propellant tracking."""
        self.propellant_cg = np.asarray(self.propellant_cg, dtype=np.float64)
        self._current_propellant = self.initial_propellant_mass

        # Default propellant inertia if not specified (assume cylinder)
        if self.propellant_inertia is None:
            # Rough estimate: treat as cylinder with radius ~ sqrt(area/pi)
            r = np.sqrt(self.reference_area / np.pi)
            L = self.reference_length * 0.5
            self.propellant_inertia = compute_inertia_cylinder(
                self.initial_propellant_mass, r, L
            )
        else:
            self.propellant_inertia = np.asarray(self.propellant_inertia, dtype=np.float64)

    @property
    def propellant_mass(self) -> float:
        """Current propellant mass [kg]."""
        return self._current_propellant

    @propellant_mass.setter
    def propellant_mass(self, value: float) -> None:
        """Set current propellant mass."""
        self._current_propellant = max(0.0, min(value, self.initial_propellant_mass))

    @property
    def propellant_fraction(self) -> float:
        """Fraction of propellant remaining (0-1)."""
        if self.initial_propellant_mass <= 0:
            return 0.0
        return self._current_propellant / self.initial_propellant_mass

    @property
    def total_mass(self) -> float:
        """Total current mass [kg]."""
        return self.dry_mass.mass + self._current_propellant

    @property
    def burnout_mass(self) -> float:
        """Mass at propellant depletion [kg]."""
        return self.dry_mass.mass

    @property
    def mass_ratio(self) -> float:
        """Initial mass / burnout mass."""
        if self.dry_mass.mass <= 0:
            return float('inf')
        return (self.dry_mass.mass + self.initial_propellant_mass) / self.dry_mass.mass

    def at_propellant_fraction(self, fraction: float) -> MassProperties:
        """Get mass properties at a given propellant fraction.

        Args:
            fraction: Propellant remaining (0 = empty, 1 = full)

        Returns:
            MassProperties at that propellant state
        """
        fraction = max(0.0, min(1.0, fraction))
        prop_mass = fraction * self.initial_propellant_mass

        if prop_mass <= 0:
            return self.dry_mass

        # Scale propellant inertia with mass (linear approximation)
        prop_inertia = self.propellant_inertia * fraction

        propellant = MassProperties(
            mass=prop_mass,
            cg=self.propellant_cg,
            inertia=prop_inertia,
        )

        return self.dry_mass + propellant

    def at_time(self, t: float, mdot: float) -> MassProperties:
        """Get mass properties at time t with constant mass flow.

        Args:
            t: Time since ignition [s]
            mdot: Mass flow rate (positive) [kg/s]

        Returns:
            MassProperties at time t
        """
        consumed = mdot * t
        remaining = max(0.0, self.initial_propellant_mass - consumed)
        fraction = remaining / self.initial_propellant_mass if self.initial_propellant_mass > 0 else 0
        return self.at_propellant_fraction(fraction)

    @property
    def current_properties(self) -> MassProperties:
        """Get mass properties at current propellant state."""
        return self.at_propellant_fraction(self.propellant_fraction)

    def consume_propellant(self, mass: float) -> float:
        """Consume propellant and return actual amount consumed.

        Args:
            mass: Mass to consume [kg]

        Returns:
            Actual mass consumed (may be less if tank empty) [kg]
        """
        actual = min(mass, self._current_propellant)
        self._current_propellant -= actual
        return actual

    def burn_time(self, mdot: float) -> float:
        """Calculate burn time at given mass flow rate.

        Args:
            mdot: Mass flow rate [kg/s]

        Returns:
            Burn time [s]
        """
        if mdot <= 0:
            return float('inf')
        return self.initial_propellant_mass / mdot

    def delta_v(self, isp: float) -> float:
        """Calculate ideal delta-V (Tsiolkovsky equation).

        Args:
            isp: Specific impulse [s]

        Returns:
            Delta-V [m/s]
        """
        g0 = 9.80665
        return g0 * isp * np.log(self.mass_ratio)


# =============================================================================
# Multi-Stage Vehicle
# =============================================================================


@beartype
@dataclass
class Stage:
    """Single stage of a multi-stage rocket.

    Attributes:
        vehicle: Stage vehicle model
        separation_delay: Time delay after burnout before separation [s]
        name: Stage identifier
    """
    vehicle: Vehicle
    separation_delay: float = 0.0
    name: str = ""


@beartype
class MultiStageVehicle:
    """Multi-stage launch vehicle.

    Models staging sequence with mass properties that change
    discretely at stage separation events.

    Example:
        >>> first = Stage(vehicle=stage1_vehicle, name="S1")
        >>> second = Stage(vehicle=stage2_vehicle, name="S2")
        >>> rocket = MultiStageVehicle([first, second])
        >>>
        >>> # Get total delta-V
        >>> print(f"Total delta-V: {rocket.total_delta_v(isp=300):.0f} m/s")
    """

    def __init__(self, stages: list[Stage]) -> None:
        """Initialize multi-stage vehicle.

        Args:
            stages: List of stages from bottom to top (first stage first)
        """
        self.stages = stages
        self._current_stage_idx = 0

    @property
    def current_stage(self) -> Stage:
        """Get current active stage."""
        return self.stages[self._current_stage_idx]

    @property
    def num_stages(self) -> int:
        """Number of stages."""
        return len(self.stages)

    @property
    def stages_remaining(self) -> int:
        """Number of stages remaining (including current)."""
        return len(self.stages) - self._current_stage_idx

    def separate_stage(self) -> bool:
        """Separate current stage. Returns True if successful."""
        if self._current_stage_idx < len(self.stages) - 1:
            self._current_stage_idx += 1
            return True
        return False

    @property
    def total_mass(self) -> float:
        """Total mass of remaining stages."""
        return sum(
            s.vehicle.total_mass
            for s in self.stages[self._current_stage_idx:]
        )

    @property
    def payload_mass(self) -> float:
        """Payload mass (final stage dry mass)."""
        return self.stages[-1].vehicle.dry_mass.mass

    def total_delta_v(self, isp: float | list[float]) -> float:
        """Calculate total ideal delta-V for all stages.

        Args:
            isp: Specific impulse(s) - single value or per-stage [s]

        Returns:
            Total delta-V [m/s]
        """
        isps = [isp] * len(self.stages) if isinstance(isp, (int, float)) else list(isp)

        g0 = 9.80665
        dv_total = 0.0
        payload = 0.0

        # Work backwards from payload
        for i in range(len(self.stages) - 1, -1, -1):
            stage = self.stages[i]
            v = stage.vehicle

            m_initial = v.dry_mass.mass + v.initial_propellant_mass + payload
            m_final = v.dry_mass.mass + payload

            if m_final > 0:
                dv = g0 * isps[i] * np.log(m_initial / m_final)
                dv_total += dv

            payload = m_initial

        return dv_total

