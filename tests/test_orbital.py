"""Unit tests for orbital mechanics utilities.

Tests the numba-optimized orbital mechanics functions for accuracy.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from rocket.orbital import (
    MU_EARTH,
    OMEGA_EARTH,
    R_EARTH_EQ,
    circular_velocity,
    compute_orbital_elements,
    compute_orbital_elements_batch,
    earth_rotation_velocity,
    eci_to_lla,
    eci_to_lla_array,
    eci_to_lla_batch,
    escape_velocity,
    launch_azimuth,
    lla_to_eci,
    orbital_period,
    time_to_apogee,
)

# =============================================================================
# Orbital Elements Tests
# =============================================================================


class TestOrbitalElements:
    """Test orbital element computation."""

    def test_circular_orbit_400km(self):
        """Test circular orbit at 400 km (ISS-like)."""
        alt = 400e3
        r = R_EARTH_EQ + alt
        v = np.sqrt(MU_EARTH / r)

        # Position at equator, velocity East
        position = np.array([r, 0.0, 0.0])
        velocity = np.array([0.0, v, 0.0])

        elements = compute_orbital_elements(position, velocity)

        # Check semi-major axis
        assert_allclose(elements.semi_major_axis, r, rtol=1e-6)

        # Check eccentricity (should be ~0 for circular)
        assert elements.eccentricity < 1e-6

        # Check inclination (equatorial = 0)
        assert_allclose(elements.inclination, 0.0, atol=1e-6)

        # Check altitudes
        assert_allclose(elements.apogee_alt, alt, rtol=1e-4)
        assert_allclose(elements.perigee_alt, alt, rtol=1e-4)

        # Check period (should be ~92 minutes)
        expected_period = 2 * np.pi * np.sqrt(r**3 / MU_EARTH)
        assert_allclose(elements.period, expected_period, rtol=1e-6)

    def test_elliptical_orbit(self):
        """Test elliptical orbit with known parameters."""
        # Create orbit with specific apogee/perigee
        perigee_alt = 200e3
        apogee_alt = 800e3

        r_p = R_EARTH_EQ + perigee_alt
        r_a = R_EARTH_EQ + apogee_alt

        sma = (r_p + r_a) / 2
        ecc = (r_a - r_p) / (r_a + r_p)

        # At perigee: r = r_p, v = v_p (perpendicular)
        v_p = np.sqrt(MU_EARTH * (2/r_p - 1/sma))

        position = np.array([r_p, 0.0, 0.0])
        velocity = np.array([0.0, v_p, 0.0])

        elements = compute_orbital_elements(position, velocity)

        assert_allclose(elements.semi_major_axis, sma, rtol=1e-5)
        assert_allclose(elements.eccentricity, ecc, rtol=1e-5)
        assert_allclose(elements.apogee_alt, apogee_alt, rtol=1e-3)
        assert_allclose(elements.perigee_alt, perigee_alt, rtol=1e-3)

    def test_inclined_orbit(self):
        """Test orbit with inclination."""
        alt = 400e3
        inc = np.radians(51.6)  # ISS inclination

        r = R_EARTH_EQ + alt
        v = np.sqrt(MU_EARTH / r)

        # Position at equator
        position = np.array([r, 0.0, 0.0])

        # Velocity inclined (has Z component)
        velocity = np.array([0.0, v * np.cos(inc), v * np.sin(inc)])

        elements = compute_orbital_elements(position, velocity)

        assert_allclose(elements.inclination, inc, rtol=1e-4)
        assert elements.eccentricity < 1e-5

    def test_specific_energy_conservation(self):
        """Test that specific energy is computed correctly."""
        alt = 500e3
        r = R_EARTH_EQ + alt
        v = np.sqrt(MU_EARTH / r)

        position = np.array([r, 0.0, 0.0])
        velocity = np.array([0.0, v, 0.0])

        elements = compute_orbital_elements(position, velocity)

        # Expected energy for circular orbit
        expected_energy = -MU_EARTH / (2 * r)
        assert_allclose(elements.specific_energy, expected_energy, rtol=1e-6)

    def test_batch_computation(self):
        """Test batch orbital element computation."""
        n = 100
        alts = np.linspace(200e3, 1000e3, n)

        positions = np.zeros((n, 3))
        velocities = np.zeros((n, 3))

        for i, alt in enumerate(alts):
            r = R_EARTH_EQ + alt
            v = np.sqrt(MU_EARTH / r)
            positions[i] = [r, 0, 0]
            velocities[i] = [0, v, 0]

        results = compute_orbital_elements_batch(positions, velocities)

        # Check shape
        assert results.shape == (n, 10)

        # Check eccentricity is near zero for all
        assert np.all(results[:, 1] < 1e-5)

        # Check altitudes
        for i, alt in enumerate(alts):
            assert_allclose(results[i, 6], alt, rtol=1e-3)  # apogee_alt
            assert_allclose(results[i, 7], alt, rtol=1e-3)  # perigee_alt


# =============================================================================
# Coordinate Transform Tests
# =============================================================================


class TestCoordinateTransforms:
    """Test ECI <-> geodetic coordinate conversions."""

    def test_eci_to_lla_equator(self):
        """Test conversion at equator."""
        alt = 400e3
        r = R_EARTH_EQ + alt

        position = np.array([r, 0.0, 0.0])
        coords = eci_to_lla(position)

        assert_allclose(coords.latitude, 0.0, atol=1e-10)
        assert_allclose(coords.longitude, 0.0, atol=1e-10)
        assert_allclose(coords.altitude, alt, rtol=1e-3)

    def test_eci_to_lla_pole(self):
        """Test conversion at North pole."""
        alt = 400e3
        r = R_EARTH_EQ + alt

        # Note: At pole, we're closer to center due to flattening
        # but we're using spherical approximation for now
        position = np.array([0.0, 0.0, r])
        coords = eci_to_lla(position)

        assert_allclose(coords.latitude, np.pi/2, atol=1e-4)
        # Longitude undefined at pole, but should not be NaN
        assert not np.isnan(coords.longitude)

    def test_lla_to_eci_roundtrip(self):
        """Test roundtrip conversion LLA -> ECI -> LLA."""
        lat = np.radians(28.5)  # Cape Canaveral
        lon = np.radians(-80.6)
        alt = 0.0

        eci = lla_to_eci(lat, lon, alt)
        lat2, lon2, alt2 = eci_to_lla_array(eci)

        assert_allclose(lat2, lat, atol=1e-6)
        assert_allclose(lon2, lon, atol=1e-6)
        assert_allclose(alt2, alt, atol=100)  # ~100m accuracy

    def test_eci_to_lla_batch(self):
        """Test batch ECI to LLA conversion."""
        n = 50
        positions = np.zeros((n, 3))

        for i in range(n):
            theta = 2 * np.pi * i / n
            r = R_EARTH_EQ + 400e3
            positions[i] = [r * np.cos(theta), r * np.sin(theta), 0]

        results = eci_to_lla_batch(positions)

        # All should be at ~0 latitude (equator)
        assert_allclose(results[:, 0], 0.0, atol=1e-6)

        # All should be at ~400km altitude
        assert_allclose(results[:, 2], 400e3, rtol=1e-3)


# =============================================================================
# Orbital Velocity Tests
# =============================================================================


class TestOrbitalVelocities:
    """Test orbital velocity calculations."""

    def test_circular_velocity_sea_level(self):
        """Test circular velocity at sea level."""
        v = circular_velocity(0.0)
        expected = np.sqrt(MU_EARTH / R_EARTH_EQ)
        assert_allclose(v, expected, rtol=1e-10)

    def test_circular_velocity_400km(self):
        """Test circular velocity at 400 km."""
        v = circular_velocity(400e3)
        r = R_EARTH_EQ + 400e3
        expected = np.sqrt(MU_EARTH / r)
        assert_allclose(v, expected, rtol=1e-10)

    def test_escape_velocity(self):
        """Test escape velocity."""
        v_esc = escape_velocity(0.0)
        v_circ = circular_velocity(0.0)

        # Escape velocity = sqrt(2) * circular velocity
        assert_allclose(v_esc, v_circ * np.sqrt(2), rtol=1e-10)

    def test_orbital_period(self):
        """Test orbital period calculation."""
        alt = 400e3
        r = R_EARTH_EQ + alt

        period = orbital_period(r)  # Takes semi-major axis
        expected = 2 * np.pi * np.sqrt(r**3 / MU_EARTH)

        assert_allclose(period, expected, rtol=1e-10)

        # Should be ~92 minutes for ISS
        assert 90 * 60 < period < 95 * 60


# =============================================================================
# Launch Azimuth Tests
# =============================================================================


class TestLaunchAzimuth:
    """Test launch azimuth calculation."""

    def test_polar_orbit_from_equator(self):
        """Polar orbit (90°) from equator requires due north launch."""
        az = launch_azimuth(0.0, np.radians(90.0), ascending=True)
        assert_allclose(az, 0.0, atol=1e-10)

    def test_equatorial_orbit_from_equator(self):
        """Equatorial orbit from equator requires due east launch."""
        az = launch_azimuth(0.0, 0.0, ascending=True)
        assert_allclose(az, np.pi/2, atol=1e-10)

    def test_iss_from_cape_canaveral(self):
        """ISS inclination (51.6°) from Cape Canaveral (28.5°N)."""
        lat = np.radians(28.5)
        inc = np.radians(51.6)

        az = launch_azimuth(lat, inc, ascending=True)

        # Should be roughly NE (between 0 and 90 degrees)
        assert 0 < az < np.pi/2
        # Approximately 45° from North for ascending (not 35° - that was a typo)
        # sin(az) = cos(inc)/cos(lat), az ≈ 45° for these values
        assert_allclose(np.degrees(az), 45.0, atol=2.0)

    def test_impossible_inclination_raises(self):
        """Should raise for inclination < latitude."""
        lat = np.radians(45.0)
        inc = np.radians(30.0)  # Can't reach 30° from 45°N

        with pytest.raises(ValueError):
            launch_azimuth(lat, inc)


# =============================================================================
# Time to Apogee Tests
# =============================================================================


class TestTimeToApogee:
    """Test time to apogee calculation."""

    def test_at_perigee(self):
        """Time to apogee should be half period when at perigee."""
        perigee_alt = 200e3
        apogee_alt = 800e3

        r_p = R_EARTH_EQ + perigee_alt
        r_a = R_EARTH_EQ + apogee_alt
        sma = (r_p + r_a) / 2

        v_p = np.sqrt(MU_EARTH * (2/r_p - 1/sma))

        position = np.array([r_p, 0.0, 0.0])
        velocity = np.array([0.0, v_p, 0.0])

        t_apogee = time_to_apogee(position, velocity)

        # Should be half the orbital period
        period = 2 * np.pi * np.sqrt(sma**3 / MU_EARTH)
        assert_allclose(t_apogee, period / 2, rtol=1e-4)

    def test_at_apogee(self):
        """Time to apogee should be ~period when just passed apogee."""
        perigee_alt = 200e3
        apogee_alt = 800e3

        r_p = R_EARTH_EQ + perigee_alt
        r_a = R_EARTH_EQ + apogee_alt
        sma = (r_p + r_a) / 2

        v_a = np.sqrt(MU_EARTH * (2/r_a - 1/sma))

        # At apogee with positive velocity = approaching apogee
        position = np.array([r_a, 0.0, 0.0])
        velocity = np.array([0.0, v_a, 0.0])  # Positive = ascending toward apogee

        t_apogee = time_to_apogee(position, velocity)

        # Should be close to zero (at apogee)
        assert t_apogee < 10.0  # Within 10 seconds of apogee

    def test_circular_orbit(self):
        """For circular orbit, time to apogee is ill-defined but shouldn't crash."""
        r = R_EARTH_EQ + 400e3
        v = np.sqrt(MU_EARTH / r)

        position = np.array([r, 0.0, 0.0])
        velocity = np.array([0.0, v, 0.0])

        # Should return 0 for circular orbit
        t_apogee = time_to_apogee(position, velocity)
        assert t_apogee == 0.0


# =============================================================================
# Earth Rotation Tests
# =============================================================================


class TestEarthRotation:
    """Test Earth rotation velocity calculations."""

    def test_equator_rotation_velocity(self):
        """Test rotation velocity at equator."""
        position = np.array([R_EARTH_EQ, 0.0, 0.0])
        v_rot = earth_rotation_velocity(position)

        # Should be ~465 m/s eastward
        expected_speed = OMEGA_EARTH * R_EARTH_EQ
        assert_allclose(np.linalg.norm(v_rot), expected_speed, rtol=1e-6)

        # Direction should be +Y (East when X points toward vernal equinox)
        assert v_rot[1] > 0
        assert_allclose(v_rot[0], 0.0, atol=1e-6)
        assert_allclose(v_rot[2], 0.0, atol=1e-6)

    def test_pole_rotation_velocity(self):
        """Test rotation velocity at pole (should be zero)."""
        position = np.array([0.0, 0.0, R_EARTH_EQ])
        v_rot = earth_rotation_velocity(position)

        assert_allclose(v_rot, [0.0, 0.0, 0.0], atol=1e-10)

