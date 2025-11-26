"""Gravity models for rocket vehicle simulation.

Provides spherical and J2 (oblateness) gravity models for Earth.

Models available:
- Spherical: Simple 1/r^2 gravity, adequate for most launch simulations
- J2: Includes Earth's oblateness effect, important for orbital mechanics
- Point mass: For flat-Earth approximations

Reference:
- WGS84 ellipsoid parameters
- EGM96 geopotential model (J2 term only)

Example:
    >>> from rocket.environment import Gravity, GravityModel
    >>>
    >>> # Simple spherical gravity
    >>> grav = Gravity(model=GravityModel.SPHERICAL)
    >>> g = grav.acceleration(position)  # [gx, gy, gz] in m/s^2
    >>>
    >>> # With J2 oblateness
    >>> grav_j2 = Gravity(model=GravityModel.J2)
    >>> g_j2 = grav_j2.acceleration(position)
"""

from enum import Enum, auto

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

# =============================================================================
# Constants
# =============================================================================

# WGS84 Earth parameters
MU_EARTH = 3.986004418e14  # Gravitational parameter [m^3/s^2]
R_EARTH_EQ = 6378137.0  # Equatorial radius [m]
R_EARTH_POLAR = 6356752.314245  # Polar radius [m]
J2 = 1.08262668e-3  # Second zonal harmonic (oblateness)
OMEGA_EARTH = 7.2921159e-5  # Earth rotation rate [rad/s]

# Standard gravity at sea level
G0 = 9.80665  # [m/s^2]


# =============================================================================
# Gravity Model Enum
# =============================================================================


class GravityModel(Enum):
    """Available gravity models."""

    CONSTANT = auto()   # Constant g (flat Earth)
    SPHERICAL = auto()  # Point mass (1/r^2)
    J2 = auto()         # Spherical + J2 oblateness


# =============================================================================
# Gravity Class
# =============================================================================


@beartype
class Gravity:
    """Gravity model for rocket vehicle simulation.

    Supports multiple fidelity levels from constant gravity to J2 perturbations.

    Example:
        >>> grav = Gravity(model=GravityModel.SPHERICAL)
        >>>
        >>> # Position in ECI frame [m]
        >>> position = np.array([6.5e6, 0, 0])
        >>>
        >>> # Get acceleration [m/s^2]
        >>> g = grav.acceleration(position)
    """

    def __init__(
        self,
        model: GravityModel = GravityModel.SPHERICAL,
        g0: float = G0,
    ) -> None:
        """Initialize gravity model.

        Args:
            model: Gravity model type
            g0: Reference gravity for CONSTANT model [m/s^2]
        """
        self.model = model
        self.g0 = g0

    @beartype
    def acceleration(
        self,
        position: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute gravitational acceleration at position.

        Args:
            position: Position vector in ECI frame [x, y, z] [m]

        Returns:
            Acceleration vector in ECI frame [ax, ay, az] [m/s^2]
        """
        position = np.asarray(position, dtype=np.float64)

        if self.model == GravityModel.CONSTANT:
            return self._constant_gravity(position)
        elif self.model == GravityModel.SPHERICAL:
            return self._spherical_gravity(position)
        elif self.model == GravityModel.J2:
            return self._j2_gravity(position)
        else:
            raise ValueError(f"Unknown gravity model: {self.model}")

    def _constant_gravity(
        self,
        position: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Constant gravity pointing toward Earth center.

        Used for flat-Earth approximation with Z pointing up.
        """
        # Gravity points radially inward (toward Earth center)
        r = np.linalg.norm(position)
        if r < 1.0:
            return np.array([0.0, 0.0, -self.g0])

        r_hat = position / r
        return -self.g0 * r_hat

    def _spherical_gravity(
        self,
        position: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Spherical (point mass) gravity model.

        g = -mu/r^2 * r_hat
        """
        r = np.linalg.norm(position)
        if r < 1e3:  # Avoid singularity near center
            r = 1e3

        r_hat = position / r
        g_mag = MU_EARTH / (r ** 2)

        return -g_mag * r_hat

    def _j2_gravity(
        self,
        position: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """J2 gravity model including Earth's oblateness.

        Includes the dominant J2 zonal harmonic which accounts for
        Earth's equatorial bulge.
        """
        x, y, z = position
        r = np.linalg.norm(position)

        if r < 1e3:  # Avoid singularity
            r = 1e3

        # Point mass term
        r ** 2
        r3 = r ** 3

        # J2 perturbation
        Re_r = R_EARTH_EQ / r
        z_r = z / r

        factor = 1.5 * J2 * Re_r ** 2

        # Acceleration components
        common = MU_EARTH / r3

        ax = -common * x * (1 - factor * (5 * z_r ** 2 - 1))
        ay = -common * y * (1 - factor * (5 * z_r ** 2 - 1))
        az = -common * z * (1 - factor * (5 * z_r ** 2 - 3))

        return np.array([ax, ay, az])

    @beartype
    def magnitude(self, position: NDArray[np.float64]) -> float:
        """Get gravity magnitude at position.

        Args:
            position: Position vector [m]

        Returns:
            Gravity magnitude [m/s^2]
        """
        g = self.acceleration(position)
        return float(np.linalg.norm(g))

    @beartype
    def potential(self, position: NDArray[np.float64]) -> float:
        """Get gravitational potential energy per unit mass.

        Args:
            position: Position vector [m]

        Returns:
            Gravitational potential [J/kg]
        """
        r = np.linalg.norm(position)
        if r < 1e3:
            r = 1e3

        if self.model == GravityModel.CONSTANT:
            # For flat Earth, potential = g * h
            return self.g0 * (r - R_EARTH_EQ)

        # Spherical potential
        U = -MU_EARTH / r

        if self.model == GravityModel.J2:
            # Add J2 contribution
            z = position[2]
            sin_lat = z / r
            U += MU_EARTH * J2 * R_EARTH_EQ ** 2 / (2 * r ** 3) * (3 * sin_lat ** 2 - 1)

        return U


# =============================================================================
# Convenience Functions
# =============================================================================


@beartype
def gravity_at_altitude(altitude: float) -> float:
    """Get gravity magnitude at altitude above Earth's surface.

    Args:
        altitude: Altitude above sea level [m]

    Returns:
        Gravity magnitude [m/s^2]
    """
    r = R_EARTH_EQ + altitude
    return MU_EARTH / (r ** 2)


@beartype
def escape_velocity(altitude: float = 0.0) -> float:
    """Get escape velocity at altitude.

    Args:
        altitude: Altitude above sea level [m]

    Returns:
        Escape velocity [m/s]
    """
    r = R_EARTH_EQ + altitude
    return np.sqrt(2 * MU_EARTH / r)


@beartype
def orbital_velocity(altitude: float) -> float:
    """Get circular orbital velocity at altitude.

    Args:
        altitude: Altitude above sea level [m]

    Returns:
        Circular orbital velocity [m/s]
    """
    r = R_EARTH_EQ + altitude
    return np.sqrt(MU_EARTH / r)


@beartype
def orbital_period(altitude: float) -> float:
    """Get orbital period for circular orbit at altitude.

    Args:
        altitude: Altitude above sea level [m]

    Returns:
        Orbital period [s]
    """
    r = R_EARTH_EQ + altitude
    return 2 * np.pi * np.sqrt(r ** 3 / MU_EARTH)

