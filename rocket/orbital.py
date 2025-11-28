"""Orbital mechanics utilities with numba optimization.

Provides fast, accurate orbital mechanics computations for simulation.
All core functions are numba-compiled for performance in tight simulation loops.

Key functions:
- compute_orbital_elements: Classical orbital elements from state vectors
- eci_to_lla: ECI to geodetic coordinates
- lla_to_eci: Geodetic to ECI coordinates  
- launch_azimuth: Compute launch azimuth for target inclination

Example:
    >>> from rocket.orbital import compute_orbital_elements, eci_to_lla
    >>> 
    >>> # Get orbital elements from position/velocity
    >>> elements = compute_orbital_elements(position, velocity)
    >>> print(f"Apogee: {elements.apogee_alt/1000:.1f} km")
    >>> 
    >>> # Convert ECI to lat/lon/alt
    >>> lat, lon, alt = eci_to_lla(position)
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

# =============================================================================
# Constants
# =============================================================================

MU_EARTH: float = 3.986004418e14  # Gravitational parameter [m^3/s^2]
R_EARTH_EQ: float = 6378137.0  # Equatorial radius [m]
R_EARTH_POLAR: float = 6356752.314245  # Polar radius [m]
OMEGA_EARTH: float = 7.2921159e-5  # Earth rotation rate [rad/s]
J2: float = 1.08262668e-3  # J2 oblateness coefficient
FLATTENING: float = 1.0 / 298.257223563  # WGS84 flattening


# =============================================================================
# Data Classes
# =============================================================================


class OrbitalElements(NamedTuple):
    """Classical orbital elements.
    
    Attributes:
        semi_major_axis: Semi-major axis [m]
        eccentricity: Orbital eccentricity [-]
        inclination: Inclination [rad]
        raan: Right ascension of ascending node [rad]
        arg_periapsis: Argument of periapsis [rad]
        true_anomaly: True anomaly [rad]
        apogee_alt: Apogee altitude above surface [m]
        perigee_alt: Perigee altitude above surface [m]
        period: Orbital period [s]
        specific_energy: Specific orbital energy [J/kg]
    """
    semi_major_axis: float
    eccentricity: float
    inclination: float
    raan: float
    arg_periapsis: float
    true_anomaly: float
    apogee_alt: float
    perigee_alt: float
    period: float
    specific_energy: float


@dataclass(frozen=True)
class GeodeticCoords:
    """Geodetic coordinates.
    
    Attributes:
        latitude: Geodetic latitude [rad]
        longitude: Longitude [rad]
        altitude: Altitude above ellipsoid [m]
    """
    latitude: float
    longitude: float
    altitude: float

    @property
    def latitude_deg(self) -> float:
        """Latitude in degrees."""
        return np.degrees(self.latitude)

    @property
    def longitude_deg(self) -> float:
        """Longitude in degrees."""
        return np.degrees(self.longitude)


# =============================================================================
# Core Numba Functions
# =============================================================================


@njit(cache=True, fastmath=True)
def _clamp(x: float, lo: float, hi: float) -> float:
    """Clamp scalar to range [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@njit(cache=True, fastmath=True)
def _compute_orbital_elements_core(
    rx: float, ry: float, rz: float,
    vx: float, vy: float, vz: float,
    mu: float = MU_EARTH,
    r_body: float = R_EARTH_EQ,
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    """Numba-optimized orbital elements computation.
    
    Returns tuple of:
        (sma, ecc, inc, raan, arg_pe, true_anom, apogee_alt, perigee_alt, period, energy)
    """
    # Position and velocity magnitudes
    r = np.sqrt(rx*rx + ry*ry + rz*rz)
    v = np.sqrt(vx*vx + vy*vy + vz*vz)

    # Specific orbital energy
    energy = v*v / 2.0 - mu / r

    # Semi-major axis (negative for hyperbolic)
    if abs(energy) < 1e-10:
        # Parabolic - use large value
        sma = 1e12
    else:
        sma = -mu / (2.0 * energy)

    # Angular momentum vector h = r x v
    hx = ry * vz - rz * vy
    hy = rz * vx - rx * vz
    hz = rx * vy - ry * vx
    h = np.sqrt(hx*hx + hy*hy + hz*hz)

    # Inclination
    if h > 1e-10:
        inc = np.arccos(_clamp(hz / h, -1.0, 1.0))
    else:
        inc = 0.0

    # Node vector n = k x h (k = [0, 0, 1])
    nx = -hy
    ny = hx
    n = np.sqrt(nx*nx + ny*ny)

    # Right ascension of ascending node
    if n > 1e-10:
        raan = np.arccos(_clamp(nx / n, -1.0, 1.0))
        if ny < 0:
            raan = 2.0 * np.pi - raan
    else:
        raan = 0.0

    # Eccentricity vector e = (v x h) / mu - r / |r|
    # First compute v x h
    vxh_x = vy * hz - vz * hy
    vxh_y = vz * hx - vx * hz
    vxh_z = vx * hy - vy * hx

    ex = vxh_x / mu - rx / r
    ey = vxh_y / mu - ry / r
    ez = vxh_z / mu - rz / r
    ecc = np.sqrt(ex*ex + ey*ey + ez*ez)

    # Argument of periapsis
    if n > 1e-10 and ecc > 1e-10:
        cos_omega = (nx * ex + ny * ey) / (n * ecc)
        arg_pe = np.arccos(_clamp(cos_omega, -1.0, 1.0))
        if ez < 0:
            arg_pe = 2.0 * np.pi - arg_pe
    else:
        arg_pe = 0.0

    # True anomaly
    if ecc > 1e-10:
        cos_nu = (ex * rx + ey * ry + ez * rz) / (ecc * r)
        true_anom = np.arccos(_clamp(cos_nu, -1.0, 1.0))
        # Check if we're past periapsis (r dot v > 0 means approaching apoapsis)
        rdotv = rx * vx + ry * vy + rz * vz
        if rdotv < 0:
            true_anom = 2.0 * np.pi - true_anom
    else:
        # Circular orbit - use argument of latitude
        true_anom = 0.0

    # Apogee and perigee
    if ecc < 1.0:
        apogee_alt = sma * (1.0 + ecc) - r_body
        perigee_alt = sma * (1.0 - ecc) - r_body
        period = 2.0 * np.pi * np.sqrt(sma**3 / mu)
    else:
        # Hyperbolic/parabolic
        apogee_alt = 1e12
        perigee_alt = sma * (1.0 - ecc) - r_body if sma > 0 else -sma * (ecc - 1.0) - r_body
        period = 0.0

    return (sma, ecc, inc, raan, arg_pe, true_anom, apogee_alt, perigee_alt, period, energy)


@njit(cache=True, fastmath=True)
def _eci_to_lla_core(
    x: float, y: float, z: float,
    a: float = R_EARTH_EQ,
    f: float = FLATTENING,
) -> tuple[float, float, float]:
    """Numba-optimized ECI to geodetic conversion.
    
    Uses iterative algorithm for accuracy.
    
    Returns:
        (latitude_rad, longitude_rad, altitude_m)
    """
    # Longitude is straightforward
    lon = np.arctan2(y, x)

    # Semi-minor axis
    b = a * (1.0 - f)

    # Distance from Z axis
    p = np.sqrt(x*x + y*y)

    # Iterative latitude computation (Bowring's method)
    # Initial guess using spherical approximation
    r = np.sqrt(x*x + y*y + z*z)
    lat = np.arcsin(z / r) if r > 0 else 0.0

    # Iterate to convergence
    e2 = 2.0 * f - f * f  # First eccentricity squared
    for _ in range(10):
        sin_lat = np.sin(lat)
        N = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
        lat_new = np.arctan2(z + e2 * N * sin_lat, p)
        if abs(lat_new - lat) < 1e-12:
            break
        lat = lat_new

    # Altitude
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)

    if abs(cos_lat) > 1e-10:
        alt = p / cos_lat - N
    else:
        alt = abs(z) - b

    return (lat, lon, alt)


@njit(cache=True, fastmath=True)
def _lla_to_eci_core(
    lat: float, lon: float, alt: float,
    a: float = R_EARTH_EQ,
    f: float = FLATTENING,
) -> tuple[float, float, float]:
    """Numba-optimized geodetic to ECI conversion.
    
    Returns:
        (x, y, z) in meters
    """
    e2 = 2.0 * f - f * f  # First eccentricity squared

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    # Radius of curvature in prime vertical
    N = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)

    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1.0 - e2) + alt) * sin_lat

    return (x, y, z)


@njit(cache=True, fastmath=True)
def _circular_velocity(altitude: float, mu: float = MU_EARTH, r_body: float = R_EARTH_EQ) -> float:
    """Circular orbital velocity at altitude."""
    r = r_body + altitude
    return np.sqrt(mu / r)


@njit(cache=True, fastmath=True)
def _escape_velocity(altitude: float, mu: float = MU_EARTH, r_body: float = R_EARTH_EQ) -> float:
    """Escape velocity at altitude."""
    r = r_body + altitude
    return np.sqrt(2.0 * mu / r)


@njit(cache=True, fastmath=True)
def _orbital_period(semi_major_axis: float, mu: float = MU_EARTH) -> float:
    """Orbital period from semi-major axis."""
    if semi_major_axis <= 0:
        return 0.0
    return 2.0 * np.pi * np.sqrt(semi_major_axis**3 / mu)


@njit(cache=True, fastmath=True)
def _time_to_apogee(
    rx: float, ry: float, rz: float,
    vx: float, vy: float, vz: float,
    sma: float, ecc: float,
    mu: float = MU_EARTH,
) -> float:
    """Compute time to next apogee passage."""
    if ecc < 1e-6 or ecc >= 1.0:
        return 0.0

    r = np.sqrt(rx*rx + ry*ry + rz*rz)

    # Mean motion
    n = np.sqrt(mu / sma**3)

    # Eccentric anomaly from r
    cos_E = (1.0 - r / sma) / ecc
    cos_E = _clamp(cos_E, -1.0, 1.0)
    E = np.arccos(cos_E)

    # Determine sign based on radial velocity
    rdotv = rx * vx + ry * vy + rz * vz
    if rdotv < 0:
        E = 2.0 * np.pi - E

    # Mean anomaly
    M = E - ecc * np.sin(E)

    # Time to apogee (M = pi at apogee)
    if np.pi >= M:
        return (np.pi - M) / n
    else:
        return (3.0 * np.pi - M) / n


@njit(cache=True, fastmath=True)
def _launch_azimuth_core(
    launch_lat: float,
    target_inc: float,
) -> tuple[float, float]:
    """Compute launch azimuth for target inclination.
    
    Returns:
        (azimuth_ascending, azimuth_descending) in radians from North
        
    For inclination < latitude, returns (nan, nan) as it's impossible.
    """
    cos_inc = np.cos(target_inc)
    cos_lat = np.cos(launch_lat)

    if abs(cos_lat) < 1e-10:
        # At pole - any azimuth works for polar orbit
        return (0.0, np.pi)

    sin_az = cos_inc / cos_lat

    if abs(sin_az) > 1.0:
        # Impossible - inclination less than latitude
        return (np.nan, np.nan)

    # Azimuth for ascending node pass (northeast launch)
    az_asc = np.arcsin(sin_az)
    # Azimuth for descending node pass (southeast launch)
    az_desc = np.pi - az_asc

    return (az_asc, az_desc)


# =============================================================================
# Python API Functions
# =============================================================================


def compute_orbital_elements(
    position: NDArray[np.float64],
    velocity: NDArray[np.float64],
    mu: float = MU_EARTH,
    r_body: float = R_EARTH_EQ,
) -> OrbitalElements:
    """Compute classical orbital elements from state vectors.
    
    Args:
        position: Position vector in inertial frame [x, y, z] [m]
        velocity: Velocity vector in inertial frame [vx, vy, vz] [m/s]
        mu: Gravitational parameter [m^3/s^2]
        r_body: Central body radius [m]
        
    Returns:
        OrbitalElements named tuple with all classical elements
        
    Example:
        >>> pos = np.array([6778137.0, 0.0, 0.0])  # ~400km altitude
        >>> vel = np.array([0.0, 7672.0, 0.0])     # Circular velocity
        >>> elements = compute_orbital_elements(pos, vel)
        >>> print(f"Period: {elements.period/60:.1f} min")
    """
    result = _compute_orbital_elements_core(
        position[0], position[1], position[2],
        velocity[0], velocity[1], velocity[2],
        mu, r_body,
    )
    return OrbitalElements(*result)


def eci_to_lla(
    position: NDArray[np.float64],
) -> GeodeticCoords:
    """Convert ECI position to geodetic coordinates.
    
    Args:
        position: Position vector in ECI frame [x, y, z] [m]
        
    Returns:
        GeodeticCoords with latitude, longitude, altitude
        
    Example:
        >>> pos = np.array([6778137.0, 0.0, 0.0])
        >>> coords = eci_to_lla(pos)
        >>> print(f"Lat: {coords.latitude_deg:.2f}°, Alt: {coords.altitude/1000:.1f} km")
    """
    lat, lon, alt = _eci_to_lla_core(position[0], position[1], position[2])
    return GeodeticCoords(latitude=lat, longitude=lon, altitude=alt)


def eci_to_lla_array(
    position: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Convert ECI to (lat_rad, lon_rad, alt_m) tuple - faster for tight loops."""
    return _eci_to_lla_core(position[0], position[1], position[2])


def lla_to_eci(
    latitude: float,
    longitude: float,
    altitude: float,
) -> NDArray[np.float64]:
    """Convert geodetic coordinates to ECI position.
    
    Args:
        latitude: Geodetic latitude [rad]
        longitude: Longitude [rad]
        altitude: Altitude above ellipsoid [m]
        
    Returns:
        Position vector in ECI frame [x, y, z] [m]
        
    Example:
        >>> pos = lla_to_eci(np.radians(28.5), np.radians(-80.6), 0.0)
        >>> print(f"Position: {pos}")
    """
    x, y, z = _lla_to_eci_core(latitude, longitude, altitude)
    return np.array([x, y, z])


def circular_velocity(altitude: float, mu: float = MU_EARTH) -> float:
    """Get circular orbital velocity at altitude.
    
    Args:
        altitude: Altitude above surface [m]
        mu: Gravitational parameter [m^3/s^2]
        
    Returns:
        Circular orbital velocity [m/s]
    """
    return _circular_velocity(altitude, mu, R_EARTH_EQ)


def escape_velocity(altitude: float, mu: float = MU_EARTH) -> float:
    """Get escape velocity at altitude.
    
    Args:
        altitude: Altitude above surface [m]
        mu: Gravitational parameter [m^3/s^2]
        
    Returns:
        Escape velocity [m/s]
    """
    return _escape_velocity(altitude, mu, R_EARTH_EQ)


def orbital_period(semi_major_axis: float, mu: float = MU_EARTH) -> float:
    """Get orbital period from semi-major axis.
    
    Args:
        semi_major_axis: Semi-major axis [m]
        mu: Gravitational parameter [m^3/s^2]
        
    Returns:
        Orbital period [s]
    """
    return _orbital_period(semi_major_axis, mu)


def time_to_apogee(
    position: NDArray[np.float64],
    velocity: NDArray[np.float64],
    elements: OrbitalElements | None = None,
) -> float:
    """Compute time until next apogee passage.
    
    Args:
        position: Position vector [m]
        velocity: Velocity vector [m/s]
        elements: Pre-computed orbital elements (optional, for performance)
        
    Returns:
        Time to apogee [s]
    """
    if elements is None:
        elements = compute_orbital_elements(position, velocity)

    return _time_to_apogee(
        position[0], position[1], position[2],
        velocity[0], velocity[1], velocity[2],
        elements.semi_major_axis, elements.eccentricity,
    )


def launch_azimuth(
    launch_latitude: float,
    target_inclination: float,
    ascending: bool = True,
) -> float:
    """Compute launch azimuth for target orbital inclination.
    
    Args:
        launch_latitude: Launch site latitude [rad]
        target_inclination: Target orbital inclination [rad]
        ascending: True for ascending node pass (typical), False for descending
        
    Returns:
        Launch azimuth from North [rad]
        
    Raises:
        ValueError: If target inclination is less than launch latitude
        
    Example:
        >>> # Cape Canaveral to ISS inclination
        >>> az = launch_azimuth(np.radians(28.5), np.radians(51.6))
        >>> print(f"Azimuth: {np.degrees(az):.1f}°")
    """
    az_asc, az_desc = _launch_azimuth_core(launch_latitude, target_inclination)

    if np.isnan(az_asc):
        raise ValueError(
            f"Target inclination ({np.degrees(target_inclination):.1f}°) must be >= "
            f"launch latitude ({np.degrees(launch_latitude):.1f}°)"
        )

    return az_asc if ascending else az_desc


def earth_rotation_velocity(position: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute velocity due to Earth rotation at position.
    
    Args:
        position: Position vector in ECI [m]
        
    Returns:
        Velocity vector due to Earth rotation [m/s]
    """
    omega = np.array([0.0, 0.0, OMEGA_EARTH])
    return np.cross(omega, position)


# =============================================================================
# Batch Operations (for trajectory analysis)
# =============================================================================


@njit(cache=True, parallel=True, fastmath=True)
def compute_orbital_elements_batch(
    positions: NDArray[np.float64],
    velocities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute orbital elements for multiple states in parallel.
    
    Args:
        positions: Nx3 array of position vectors [m]
        velocities: Nx3 array of velocity vectors [m/s]
        
    Returns:
        Nx10 array with columns:
        [sma, ecc, inc, raan, arg_pe, true_anom, apogee_alt, perigee_alt, period, energy]
    """
    n = positions.shape[0]
    result = np.empty((n, 10), dtype=np.float64)

    for i in range(n):
        elements = _compute_orbital_elements_core(
            positions[i, 0], positions[i, 1], positions[i, 2],
            velocities[i, 0], velocities[i, 1], velocities[i, 2],
            MU_EARTH, R_EARTH_EQ,
        )
        for j in range(10):
            result[i, j] = elements[j]

    return result


@njit(cache=True, parallel=True, fastmath=True)
def eci_to_lla_batch(
    positions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert multiple ECI positions to geodetic coordinates in parallel.
    
    Args:
        positions: Nx3 array of position vectors [m]
        
    Returns:
        Nx3 array with columns [lat_rad, lon_rad, alt_m]
    """
    n = positions.shape[0]
    result = np.empty((n, 3), dtype=np.float64)

    for i in range(n):
        lat, lon, alt = _eci_to_lla_core(
            positions[i, 0], positions[i, 1], positions[i, 2]
        )
        result[i, 0] = lat
        result[i, 1] = lon
        result[i, 2] = alt

    return result

