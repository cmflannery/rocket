"""Gravity models for rocket vehicle simulation.

Provides spherical and J2 (oblateness) gravity models for Earth.
Core functions are numba-compiled for performance.

Models available:
- Spherical: Simple 1/r^2 gravity, adequate for most launch simulations
- J2: Includes Earth's oblateness effect, important for orbital mechanics
- Constant: Flat-Earth approximation

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
from numba import njit
from numpy.typing import NDArray

# =============================================================================
# Constants
# =============================================================================

# WGS84 Earth parameters
MU_EARTH: float = 3.986004418e14  # Gravitational parameter [m^3/s^2]
R_EARTH_EQ: float = 6378137.0  # Equatorial radius [m]
R_EARTH_POLAR: float = 6356752.314245  # Polar radius [m]
J2: float = 1.08262668e-3  # Second zonal harmonic (oblateness)
OMEGA_EARTH: float = 7.2921159e-5  # Earth rotation rate [rad/s]

# Standard gravity at sea level
G0: float = 9.80665  # [m/s^2]


# =============================================================================
# Gravity Model Enum
# =============================================================================


class GravityModel(Enum):
    """Available gravity models."""

    CONSTANT = auto()   # Constant g (flat Earth)
    SPHERICAL = auto()  # Point mass (1/r^2)
    J2 = auto()         # Spherical + J2 oblateness


# =============================================================================
# Numba-Optimized Core Functions
# =============================================================================


@njit(cache=True, fastmath=True)
def _spherical_gravity(
    x: float, y: float, z: float,
    mu: float = MU_EARTH,
) -> tuple[float, float, float]:
    """Numba-optimized spherical gravity.

    g = -mu/r^2 * r_hat
    """
    r_sq = x*x + y*y + z*z
    r = np.sqrt(r_sq)

    if r < 1e3:  # Avoid singularity near center
        r = 1e3
        r_sq = r * r

    g_over_r = mu / (r_sq * r)

    return (-g_over_r * x, -g_over_r * y, -g_over_r * z)


@njit(cache=True, fastmath=True)
def _j2_gravity(
    x: float, y: float, z: float,
    mu: float = MU_EARTH,
    r_eq: float = R_EARTH_EQ,
    j2: float = J2,
) -> tuple[float, float, float]:
    """Numba-optimized J2 gravity.

    Includes Earth's oblateness perturbation.
    """
    r_sq = x*x + y*y + z*z
    r = np.sqrt(r_sq)

    if r < 1e3:
        r = 1e3
        r_sq = r * r

    r3 = r * r_sq

    # J2 perturbation factors
    Re_r = r_eq / r
    z_r = z / r
    z_r_sq = z_r * z_r

    factor = 1.5 * j2 * Re_r * Re_r

    # Acceleration components
    common = mu / r3

    ax = -common * x * (1.0 - factor * (5.0 * z_r_sq - 1.0))
    ay = -common * y * (1.0 - factor * (5.0 * z_r_sq - 1.0))
    az = -common * z * (1.0 - factor * (5.0 * z_r_sq - 3.0))

    return (ax, ay, az)


@njit(cache=True, fastmath=True)
def _constant_gravity(
    x: float, y: float, z: float,
    g0: float = G0,
) -> tuple[float, float, float]:
    """Constant gravity pointing toward Earth center."""
    r_sq = x*x + y*y + z*z
    r = np.sqrt(r_sq)

    if r < 1.0:
        return (0.0, 0.0, -g0)

    inv_r = 1.0 / r
    return (-g0 * x * inv_r, -g0 * y * inv_r, -g0 * z * inv_r)


@njit(cache=True, fastmath=True)
def gravity_magnitude_at_altitude(altitude: float, mu: float = MU_EARTH, r_eq: float = R_EARTH_EQ) -> float:
    """Get gravity magnitude at altitude above surface."""
    r = r_eq + altitude
    return mu / (r * r)


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
        x, y, z = float(position[0]), float(position[1]), float(position[2])

        if self.model == GravityModel.CONSTANT:
            gx, gy, gz = _constant_gravity(x, y, z, self.g0)
        elif self.model == GravityModel.SPHERICAL:
            gx, gy, gz = _spherical_gravity(x, y, z)
        elif self.model == GravityModel.J2:
            gx, gy, gz = _j2_gravity(x, y, z)
        else:
            raise ValueError(f"Unknown gravity model: {self.model}")

        return np.array([gx, gy, gz])

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
    return gravity_magnitude_at_altitude(altitude, MU_EARTH, R_EARTH_EQ)


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
