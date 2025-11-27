"""Aerodynamics models for rocket vehicles.

Provides drag and lift coefficient models as functions of Mach number
and angle of attack for rocket vehicle simulation.

The models include:
- SimpleAero: Mach-dependent drag with optional angle of attack effects
- Cd tables: Interpolated from empirical data

Example:
    >>> from rocket.vehicle import SimpleAero
    >>> from rocket.environment import Atmosphere
    >>>
    >>> aero = SimpleAero(Cd0=0.3, reference_area=1.0)
    >>>
    >>> # Get drag force
    >>> velocity = 500  # m/s
    >>> altitude = 10000  # m
    >>> atm = Atmosphere()
    >>> q = atm.dynamic_pressure(altitude, velocity)
    >>>
    >>> drag = aero.drag_force(q, mach=1.5)
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

# =============================================================================
# Aerodynamics Protocol
# =============================================================================


class AerodynamicsModel(Protocol):
    """Protocol for aerodynamics models."""

    def drag_coefficient(self, mach: float, alpha: float = 0.0) -> float:
        """Get drag coefficient at Mach number and angle of attack."""
        ...

    def lift_coefficient(self, mach: float, alpha: float) -> float:
        """Get lift coefficient at Mach number and angle of attack."""
        ...


# =============================================================================
# Simple Aerodynamics Model
# =============================================================================


@beartype
@dataclass
class SimpleAero:
    """Simple aerodynamics model with Mach-dependent drag.

    Uses a piecewise model for drag coefficient:
    - Subsonic (M < 0.8): Cd = Cd0
    - Transonic (0.8 < M < 1.2): Cd rises to peak at M=1
    - Supersonic (M > 1.2): Cd decreases with Mach

    Attributes:
        Cd0: Zero-lift drag coefficient at subsonic speeds
        reference_area: Aerodynamic reference area [m^2]
        transonic_peak: Multiplier for Cd at M=1 (typically 1.3-2.0)
        Cl_alpha: Lift curve slope [1/rad] (for angle of attack effects)
    """
    Cd0: float = 0.3
    reference_area: float = 1.0  # [m^2]
    transonic_peak: float = 1.5  # Cd multiplier at M=1
    Cl_alpha: float = 2.0  # [1/rad]
    Cd_alpha: float = 0.5  # Drag increase per rad^2 of alpha

    @beartype
    def drag_coefficient(self, mach: float, alpha: float = 0.0) -> float:
        """Get drag coefficient as function of Mach and angle of attack.

        Args:
            mach: Mach number
            alpha: Angle of attack [rad]

        Returns:
            Drag coefficient
        """
        # Base drag coefficient vs Mach
        if mach < 0.8:
            # Subsonic - constant
            Cd = self.Cd0
        elif mach < 1.0:
            # Transonic rise - linear interpolation to peak
            frac = (mach - 0.8) / 0.2
            Cd = self.Cd0 * (1 + frac * (self.transonic_peak - 1))
        elif mach < 1.2:
            # Transonic fall from peak
            frac = (mach - 1.0) / 0.2
            Cd = self.Cd0 * self.transonic_peak * (1 - 0.3 * frac)
        else:
            # Supersonic - gradual decrease
            Cd = self.Cd0 * self.transonic_peak * 0.7 / np.sqrt(mach**2 - 1 + 0.1)

        # Angle of attack effect (induced drag)
        Cd += self.Cd_alpha * alpha**2

        return Cd

    @beartype
    def lift_coefficient(self, mach: float, alpha: float) -> float:
        """Get lift coefficient as function of Mach and angle of attack.

        Args:
            mach: Mach number
            alpha: Angle of attack [rad]

        Returns:
            Lift coefficient
        """
        # Simple linear lift model with Mach correction
        if mach < 1.0:
            # Subsonic - Prandtl-Glauert correction
            beta = np.sqrt(1 - mach**2 + 0.01)
            Cl = self.Cl_alpha * alpha / beta
        else:
            # Supersonic - Ackeret approximation
            beta = np.sqrt(mach**2 - 1 + 0.01)
            Cl = 4 * alpha / beta

        return Cl

    @beartype
    def drag_force(
        self,
        dynamic_pressure: float,
        mach: float = 0.0,
        alpha: float = 0.0,
    ) -> float:
        """Calculate drag force.

        Args:
            dynamic_pressure: Dynamic pressure q = 0.5 * rho * v^2 [Pa]
            mach: Mach number
            alpha: Angle of attack [rad]

        Returns:
            Drag force [N]
        """
        Cd = self.drag_coefficient(mach, alpha)
        return dynamic_pressure * self.reference_area * Cd

    @beartype
    def lift_force(
        self,
        dynamic_pressure: float,
        mach: float,
        alpha: float,
    ) -> float:
        """Calculate lift force.

        Args:
            dynamic_pressure: Dynamic pressure [Pa]
            mach: Mach number
            alpha: Angle of attack [rad]

        Returns:
            Lift force [N]
        """
        Cl = self.lift_coefficient(mach, alpha)
        return dynamic_pressure * self.reference_area * Cl

    @beartype
    def forces_body(
        self,
        velocity_body: NDArray[np.float64],
        density: float,
        speed_of_sound: float,
    ) -> NDArray[np.float64]:
        """Calculate aerodynamic forces in body frame.

        Assumes velocity is primarily along body X axis.

        Args:
            velocity_body: Velocity in body frame [vx, vy, vz] [m/s]
            density: Air density [kg/m^3]
            speed_of_sound: Speed of sound [m/s]

        Returns:
            Force vector in body frame [Fx, Fy, Fz] [N]
        """
        v = np.linalg.norm(velocity_body)
        if v < 1.0:
            return np.array([0.0, 0.0, 0.0])

        mach = v / speed_of_sound
        q = 0.5 * density * v**2

        # Angle of attack from velocity components
        vx, vy, vz = velocity_body
        alpha = np.arctan2(-vz, vx)  # Pitch plane
        beta = np.arctan2(vy, vx)    # Yaw plane

        # Total angle of attack
        alpha_total = np.sqrt(alpha**2 + beta**2)

        # Drag (opposes velocity)
        drag = self.drag_force(q, mach, alpha_total)

        # Lift (perpendicular to velocity in pitch plane)
        lift = self.lift_force(q, mach, alpha_total)

        # Convert to body frame forces
        # Drag acts opposite to velocity direction
        v_hat = velocity_body / v
        F_drag = -drag * v_hat

        # Lift acts perpendicular to velocity
        # In the pitch plane (XZ), perpendicular to velocity
        if abs(alpha) > 1e-6:
            lift_direction = np.array([np.sin(alpha), 0, np.cos(alpha)])
            F_lift = lift * lift_direction * np.sign(-alpha)
        else:
            F_lift = np.array([0.0, 0.0, 0.0])

        return F_drag + F_lift

    @beartype
    def moments_body(
        self,
        velocity_body: NDArray[np.float64],
        angular_velocity: NDArray[np.float64],
        density: float,
        speed_of_sound: float,
        reference_length: float,
    ) -> NDArray[np.float64]:
        """Calculate aerodynamic moments in body frame.

        Includes:
        - Static stability (from fins/body) - restoring moment at angle of attack
        - Pitch damping - opposes rotation rate

        Args:
            velocity_body: Velocity in body frame [m/s]
            angular_velocity: Angular velocity in body frame [rad/s]
            density: Air density [kg/m^3]
            speed_of_sound: Speed of sound [m/s]
            reference_length: Reference length for moment coefficients [m]

        Returns:
            Moment vector in body frame [Mx, My, Mz] [N*m]
        """
        v = np.linalg.norm(velocity_body)
        if v < 1.0:
            return np.array([0.0, 0.0, 0.0])

        q = 0.5 * density * v**2

        # Angle of attack from velocity components
        vx, vy, vz = velocity_body
        alpha = np.arctan2(-vz, vx)  # Pitch angle of attack
        beta = np.arctan2(vy, vx)    # Sideslip angle

        # Static stability coefficient (Cm_alpha)
        # Negative = stable (nose-down moment when AoA positive)
        # Moderate stability to allow gravity turn while preventing tumbling
        Cm_alpha = -2.0  # Moderate static stability [1/rad]

        # Pitch damping coefficient (Cm_q)
        # Negative = damping (opposes rotation)
        # Moderate damping to reduce oscillations
        Cmq = -5.0  # Moderate pitch damping [1/rad]

        # Non-dimensional pitch rate: q * L / V
        p, qb, r = angular_velocity  # Body rates
        q_hat = qb * reference_length / v if v > 1.0 else 0.0
        r_hat = r * reference_length / v if v > 1.0 else 0.0

        # Pitching moment (about Y axis)
        Cm = Cm_alpha * alpha + Cmq * q_hat
        My = q * self.reference_area * reference_length * Cm

        # Yawing moment (about Z axis) - similar stability for sideslip
        Cn_beta = -2.0  # Moderate yaw static stability
        Cnr = -5.0  # Moderate yaw damping
        Cn = Cn_beta * beta + Cnr * r_hat
        Mz = q * self.reference_area * reference_length * Cn

        # Roll damping (about X axis)
        Clp = -5.0  # Roll damping
        p_hat = p * reference_length / v if v > 1.0 else 0.0
        Mx = q * self.reference_area * reference_length * Clp * p_hat

        return np.array([Mx, My, Mz])


# =============================================================================
# Tabulated Aerodynamics
# =============================================================================


@beartype
@dataclass
class TabulatedAero:
    """Aerodynamics model with tabulated Cd vs Mach.

    Uses linear interpolation between data points.

    Example:
        >>> mach_table = [0.0, 0.8, 1.0, 1.2, 2.0, 3.0]
        >>> cd_table = [0.3, 0.3, 0.45, 0.4, 0.35, 0.3]
        >>> aero = TabulatedAero(mach_table, cd_table, reference_area=1.0)
    """
    mach_values: list[float]
    Cd_values: list[float]
    reference_area: float = 1.0
    Cl_alpha: float = 2.0

    def __post_init__(self) -> None:
        """Validate inputs."""
        if len(self.mach_values) != len(self.Cd_values):
            raise ValueError("Mach and Cd tables must have same length")
        if len(self.mach_values) < 2:
            raise ValueError("Need at least 2 data points")

    @beartype
    def drag_coefficient(self, mach: float, alpha: float = 0.0) -> float:
        """Get drag coefficient by interpolation."""
        Cd = np.interp(mach, self.mach_values, self.Cd_values)
        # Add alpha effect
        Cd += 0.5 * alpha**2
        return float(Cd)

    @beartype
    def lift_coefficient(self, mach: float, alpha: float) -> float:
        """Get lift coefficient."""
        return self.Cl_alpha * alpha

    @beartype
    def drag_force(
        self,
        dynamic_pressure: float,
        mach: float = 0.0,
        alpha: float = 0.0,
    ) -> float:
        """Calculate drag force [N]."""
        return dynamic_pressure * self.reference_area * self.drag_coefficient(mach, alpha)


# =============================================================================
# Common Vehicle Profiles
# =============================================================================


def typical_rocket_aero(
    diameter: float,
    nose_fineness: float = 3.0,
) -> SimpleAero:
    """Create typical rocket aerodynamics model.

    Args:
        diameter: Vehicle diameter [m]
        nose_fineness: Nose cone length / diameter ratio

    Returns:
        SimpleAero model with typical coefficients
    """
    area = np.pi * (diameter / 2) ** 2

    # Cd0 depends on fineness ratio
    if nose_fineness < 2:
        Cd0 = 0.5  # Blunt nose
    elif nose_fineness < 4:
        Cd0 = 0.3  # Moderate
    else:
        Cd0 = 0.2  # Sharp nose

    return SimpleAero(
        Cd0=Cd0,
        reference_area=area,
        transonic_peak=1.5,
    )

