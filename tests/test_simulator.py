"""Unit tests for the step-driven Simulator.

Tests the simulation infrastructure for accuracy and physical correctness.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from rocket.environment.gravity import MU_EARTH, R_EARTH_EQ
from rocket.simulation import (
    SimulationResult,
    Simulator,
    ThrustCommand,
)

# =============================================================================
# Simulator Initialization Tests
# =============================================================================


class TestSimulatorInit:
    """Test simulator initialization."""

    def test_from_launch_pad_default(self):
        """Test default launch pad initialization (Cape Canaveral)."""
        sim = Simulator.from_launch_pad()

        state = sim.get_state()

        # Should be near Earth surface
        r = np.linalg.norm(state.position)
        assert_allclose(r, R_EARTH_EQ, rtol=1e-3)

        # Should have Earth rotation velocity (~465 m/s at equator)
        speed = np.linalg.norm(state.velocity)
        assert 300 < speed < 500  # Cape Canaveral is at 28.5°N

        # Time should be zero
        assert state.time == 0.0

    def test_from_launch_pad_custom_location(self):
        """Test launch pad at custom location."""
        sim = Simulator.from_launch_pad(
            latitude=0.0,
            longitude=0.0,
            vehicle_mass=50000.0,
        )

        state = sim.get_state()

        # At equator, rotation velocity should be ~465 m/s
        speed = np.linalg.norm(state.velocity)
        expected = 7.2921159e-5 * R_EARTH_EQ
        assert_allclose(speed, expected, rtol=1e-3)

        # Mass should match
        assert state.mass == 50000.0

    def test_from_orbit(self):
        """Test initialization in circular orbit."""
        alt = 400e3
        sim = Simulator.from_orbit(altitude=alt, vehicle_mass=1000.0)

        state = sim.get_state()

        # Should be at correct altitude
        r = np.linalg.norm(state.position)
        assert_allclose(r, R_EARTH_EQ + alt, rtol=1e-6)

        # Should have correct orbital velocity
        v = np.linalg.norm(state.velocity)
        expected_v = np.sqrt(MU_EARTH / (R_EARTH_EQ + alt))
        assert_allclose(v, expected_v, rtol=1e-6)


# =============================================================================
# Free Fall Tests
# =============================================================================


class TestFreeFall:
    """Test gravity-only propagation."""

    def test_free_fall_acceleration(self):
        """Test that free fall gives correct acceleration magnitude."""
        sim = Simulator.from_orbit(altitude=400e3, vehicle_mass=1000.0)

        # Get initial state
        state0 = sim.get_state()
        pos0 = state0.position.copy()
        vel0 = state0.velocity.copy()

        # Step with no thrust
        dt = 0.1
        sim.step(thrust=0.0, dt=dt)

        state1 = sim.get_state()

        # Compute average acceleration magnitude
        dv = state1.velocity - vel0
        accel_mag = np.linalg.norm(dv) / dt

        # Should match gravity magnitude
        r = np.linalg.norm(pos0)
        expected_g_mag = MU_EARTH / r**2

        # Allow some error due to position change during step
        assert_allclose(accel_mag, expected_g_mag, rtol=0.01)


# =============================================================================
# Circular Orbit Tests
# =============================================================================


class TestCircularOrbit:
    """Test circular orbit propagation."""

    def test_orbit_stays_circular(self):
        """Circular orbit should maintain altitude."""
        alt = 400e3
        sim = Simulator.from_orbit(altitude=alt)

        initial_r = np.linalg.norm(sim.get_state().position)

        # Propagate for a few minutes
        dt = 1.0
        for _ in range(300):  # 5 minutes
            sim.step(thrust=0.0, dt=dt)

        final_r = np.linalg.norm(sim.get_state().position)

        # Altitude should be very close
        assert_allclose(final_r, initial_r, rtol=1e-5)

    def test_orbit_period(self):
        """Test that orbit period is correct by checking radius returns."""
        alt = 400e3
        sim = Simulator.from_orbit(altitude=alt)

        initial_r = np.linalg.norm(sim.get_state().position)

        # Expected period
        r = R_EARTH_EQ + alt
        period = 2 * np.pi * np.sqrt(r**3 / MU_EARTH)

        # Propagate for one period
        dt = 1.0
        n_steps = int(period / dt)
        for _ in range(n_steps):
            sim.step(thrust=0.0, dt=dt)

        final_r = np.linalg.norm(sim.get_state().position)

        # Radius should return to same value after one orbit
        assert_allclose(final_r, initial_r, rtol=0.001)


# =============================================================================
# Energy/Momentum Conservation Tests
# =============================================================================


class TestConservation:
    """Test conservation laws."""

    def test_energy_conservation_free_fall(self):
        """Orbital energy should be conserved in free fall."""
        sim = Simulator.from_orbit(altitude=400e3)

        def compute_energy(state):
            r = np.linalg.norm(state.position)
            v = np.linalg.norm(state.velocity)
            return 0.5 * v**2 - MU_EARTH / r

        initial_energy = compute_energy(sim.get_state())

        # Propagate
        dt = 1.0
        energies = [initial_energy]
        for _ in range(100):
            sim.step(thrust=0.0, dt=dt)
            energies.append(compute_energy(sim.get_state()))

        energies = np.array(energies)

        # Energy should be constant to within numerical precision
        assert_allclose(energies, initial_energy, rtol=1e-6)

    def test_angular_momentum_conservation(self):
        """Angular momentum should be conserved in central force field."""
        sim = Simulator.from_orbit(altitude=400e3)

        def compute_angular_momentum(state):
            return np.cross(state.position, state.velocity)

        initial_h = compute_angular_momentum(sim.get_state())

        # Propagate
        dt = 1.0
        for _ in range(100):
            sim.step(thrust=0.0, dt=dt)

        final_h = compute_angular_momentum(sim.get_state())

        # Angular momentum should be conserved
        assert_allclose(final_h, initial_h, rtol=1e-5)


# =============================================================================
# Thrust Tests
# =============================================================================


class TestThrust:
    """Test thrust application."""

    def test_thrust_accelerates(self):
        """Applying thrust should accelerate the vehicle."""
        sim = Simulator.from_orbit(altitude=400e3, vehicle_mass=1000.0)

        initial_speed = np.linalg.norm(sim.get_state().velocity)

        # Apply prograde thrust
        thrust = 10000.0  # 10 kN
        dt = 1.0

        for _ in range(10):
            sim.step(thrust=thrust, dt=dt)

        final_speed = np.linalg.norm(sim.get_state().velocity)

        # Speed should increase
        assert final_speed > initial_speed

    def test_thrust_command_object(self):
        """Test using ThrustCommand object."""
        sim = Simulator.from_orbit(altitude=400e3)

        cmd = ThrustCommand(magnitude=10000.0, gimbal_pitch=0.01, gimbal_yaw=0.0)

        # Should not raise
        sim.step(thrust=cmd, dt=0.1)

        # State should have advanced
        assert sim.get_state().time == pytest.approx(0.1)

    def test_mass_consumption(self):
        """Mass should decrease when mass_rate is specified."""
        sim = Simulator.from_launch_pad(vehicle_mass=10000.0)

        initial_mass = sim.get_state().mass

        # Simulate with mass flow
        mdot = -100.0  # 100 kg/s consumption
        dt = 1.0
        for _ in range(10):
            sim.step(thrust=100000.0, mass_rate=mdot, dt=dt)

        final_mass = sim.get_state().mass

        # Mass should decrease by mdot * t
        expected_mass = initial_mass + mdot * 10.0
        assert_allclose(final_mass, expected_mass, rtol=1e-4)


# =============================================================================
# Environment Data Tests
# =============================================================================


class TestEnvironment:
    """Test environment data retrieval."""

    def test_get_environment_altitude(self):
        """Test environment altitude calculation."""
        alt = 400e3
        sim = Simulator.from_orbit(altitude=alt)

        env = sim.get_environment()

        assert_allclose(env.altitude, alt, rtol=1e-3)

    def test_get_environment_gravity(self):
        """Test environment gravity calculation."""
        alt = 400e3
        sim = Simulator.from_orbit(altitude=alt)

        env = sim.get_environment()

        # Gravity magnitude
        r = R_EARTH_EQ + alt
        expected_g = MU_EARTH / r**2
        assert_allclose(env.gravity_magnitude, expected_g, rtol=1e-4)

    def test_atmosphere_below_karman_line(self):
        """Test that atmosphere is present below 100 km."""
        sim = Simulator.from_launch_pad()

        env = sim.get_environment()

        assert env.atmosphere is not None
        assert env.atmosphere.density > 0

    def test_no_atmosphere_in_orbit(self):
        """Test that atmosphere model returns None above 150 km."""
        sim = Simulator.from_orbit(altitude=400e3)

        env = sim.get_environment()

        assert env.atmosphere is None


# =============================================================================
# History and Results Tests
# =============================================================================


class TestHistory:
    """Test state history recording."""

    def test_history_recorded(self):
        """Test that history is recorded by default."""
        sim = Simulator.from_orbit(altitude=400e3)

        for _ in range(10):
            sim.step(dt=1.0)

        history = sim.get_history()

        # Should have initial + 10 steps
        assert len(history) == 11

    def test_simulation_result(self):
        """Test SimulationResult creation."""
        sim = Simulator.from_orbit(altitude=400e3)

        for _ in range(10):
            sim.step(dt=1.0)

        result = SimulationResult.from_simulator(sim)

        assert len(result.time) == 11
        assert result.position.shape == (11, 3)
        assert result.velocity.shape == (11, 3)

    def test_clear_history(self):
        """Test clearing history."""
        sim = Simulator.from_orbit(altitude=400e3)

        for _ in range(10):
            sim.step(dt=1.0)

        sim.clear_history()
        history = sim.get_history()

        # Should only have current state
        assert len(history) == 1


# =============================================================================
# Attitude Tests
# =============================================================================


class TestAttitude:
    """Test attitude handling."""

    def test_set_attitude(self):
        """Test setting attitude directly."""
        sim = Simulator.from_orbit(altitude=400e3)

        # Set to identity quaternion
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        sim.set_attitude(identity)

        state = sim.get_state()
        assert_allclose(state.quaternion, identity, atol=1e-10)

    def test_attitude_propagates(self):
        """Test that attitude is propagated during step."""
        sim = Simulator.from_orbit(altitude=400e3)

        # Set some angular velocity
        sim.state.angular_velocity = np.array([0.1, 0.0, 0.0])

        initial_q = sim.get_state().quaternion.copy()

        # Step forward
        sim.step(dt=1.0)

        final_q = sim.get_state().quaternion

        # Quaternion should have changed
        assert not np.allclose(final_q, initial_q)

