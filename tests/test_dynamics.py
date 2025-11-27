"""Unit tests for dynamics module - state, quaternions, rigid body dynamics.

These tests verify the fundamental mechanics of the 6DOF simulation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from rocket.dynamics.rigid_body import (
    DynamicsConfig,
    RigidBodyDynamics,
    euler_rotational_dynamics,
    quaternion_derivative,
    rigid_body_derivatives,
    rk4_step,
)
from rocket.dynamics.state import (
    State,
    dcm_to_quaternion,
    euler_to_quaternion,
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_dcm,
    quaternion_to_euler,
)
from rocket.environment.gravity import MU_EARTH, R_EARTH_EQ, Gravity, GravityModel

# =============================================================================
# Quaternion Tests
# =============================================================================

class TestQuaternionOperations:
    """Test quaternion math operations."""

    def test_normalize_identity(self):
        """Identity quaternion should remain identity after normalization."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        q_norm = normalize_quaternion(q)
        assert_allclose(q_norm, q, atol=1e-10)

    def test_normalize_unit_length(self):
        """Normalized quaternion should have unit length."""
        q = np.array([1.0, 2.0, 3.0, 4.0])
        q_norm = normalize_quaternion(q)
        assert_allclose(np.linalg.norm(q_norm), 1.0, atol=1e-10)

    def test_quaternion_multiply_identity(self):
        """Multiplying by identity should give same quaternion."""
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        q = np.array([0.707, 0.707, 0.0, 0.0])
        q = normalize_quaternion(q)

        result = quaternion_multiply(identity, q)
        assert_allclose(result, q, atol=1e-10)

        result2 = quaternion_multiply(q, identity)
        assert_allclose(result2, q, atol=1e-10)

    def test_quaternion_conjugate_inverse(self):
        """Quaternion times its conjugate should give identity."""
        q = normalize_quaternion(np.array([1.0, 2.0, 3.0, 4.0]))
        q_conj = quaternion_conjugate(q)

        result = quaternion_multiply(q, q_conj)
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        assert_allclose(result, identity, atol=1e-10)

    def test_dcm_roundtrip(self):
        """Converting quaternion to DCM and back should preserve quaternion."""
        q_original = normalize_quaternion(np.array([0.5, 0.5, 0.5, 0.5]))

        dcm = quaternion_to_dcm(q_original)
        q_recovered = dcm_to_quaternion(dcm)

        # Quaternion may have opposite sign (same rotation)
        if np.dot(q_original, q_recovered) < 0:
            q_recovered = -q_recovered

        assert_allclose(q_recovered, q_original, atol=1e-10)

    def test_dcm_orthonormal(self):
        """DCM should be orthonormal (R @ R.T = I, det(R) = 1)."""
        q = normalize_quaternion(np.array([1.0, 2.0, 3.0, 4.0]))
        dcm = quaternion_to_dcm(q)

        # Check orthonormality
        assert_allclose(dcm @ dcm.T, np.eye(3), atol=1e-10)

        # Check determinant
        assert_allclose(np.linalg.det(dcm), 1.0, atol=1e-10)

    def test_euler_roundtrip(self):
        """Converting Euler angles to quaternion and back should preserve angles."""
        roll, pitch, yaw = np.radians([30, 45, 60])

        q = euler_to_quaternion(roll, pitch, yaw)
        roll2, pitch2, yaw2 = quaternion_to_euler(q)

        assert_allclose([roll2, pitch2, yaw2], [roll, pitch, yaw], atol=1e-10)

    def test_euler_identity(self):
        """Zero Euler angles should give identity quaternion."""
        q = euler_to_quaternion(0.0, 0.0, 0.0)
        assert_allclose(q, [1, 0, 0, 0], atol=1e-10)

    def test_euler_pitch_90(self):
        """90 degree pitch should align body X with inertial Z."""
        q = euler_to_quaternion(0.0, np.pi/2, 0.0)
        dcm = quaternion_to_dcm(q)

        # Body X axis in inertial frame
        body_x_inertial = dcm.T @ np.array([1, 0, 0])

        # Should be close to inertial Z
        assert_allclose(body_x_inertial, [0, 0, 1], atol=1e-10)


# =============================================================================
# State Tests
# =============================================================================

class TestStateInitialization:
    """Test State class initialization methods."""

    def test_flat_earth_vertical(self):
        """Flat Earth state with 90° pitch should have body X pointing up."""
        state = State.from_flat_earth(z=100.0, pitch_deg=90.0, mass_kg=1000.0)

        # Body X axis in inertial frame (using DCM columns)
        dcm = state.dcm_body_to_inertial
        body_x = dcm[:, 0]

        # Should point in +Z direction (up)
        assert_allclose(body_x, [0, 0, 1], atol=1e-10)

    def test_flat_earth_horizontal(self):
        """Flat Earth state with 0° pitch should have body X horizontal."""
        state = State.from_flat_earth(pitch_deg=0.0, yaw_deg=0.0, mass_kg=1000.0)

        dcm = state.dcm_body_to_inertial
        body_x = dcm[:, 0]

        # Should point in +X direction (horizontal)
        assert_allclose(body_x, [1, 0, 0], atol=1e-10)

    def test_launch_pad_position(self):
        """Launch pad state should have correct position on Earth surface."""
        lat, lon = 28.5, -80.6  # Cape Canaveral
        state = State.from_launch_pad(latitude_deg=lat, longitude_deg=lon)

        # Should be at Earth's surface
        r = np.linalg.norm(state.position)
        assert_allclose(r, R_EARTH_EQ, rtol=1e-6)

        # Check latitude/longitude
        recovered_lat = np.degrees(np.arcsin(state.position[2] / r))
        recovered_lon = np.degrees(np.arctan2(state.position[1], state.position[0]))

        assert_allclose(recovered_lat, lat, atol=0.1)
        assert_allclose(recovered_lon, lon, atol=0.1)

    def test_launch_pad_earth_rotation_velocity(self):
        """Launch pad velocity should match Earth rotation."""
        state = State.from_launch_pad(latitude_deg=0.0)  # Equator

        # At equator, velocity should be ~465 m/s eastward
        expected_v = 7.2921159e-5 * R_EARTH_EQ  # omega * r

        assert_allclose(state.speed, expected_v, rtol=0.01)

    def test_launch_pad_body_x_points_up(self):
        """At launch pad, body X axis should point radially outward (up)."""
        state = State.from_launch_pad(latitude_deg=28.5, longitude_deg=-80.6)

        # Radial direction (up)
        r_hat = state.position / np.linalg.norm(state.position)

        # Body X axis
        dcm = state.dcm_body_to_inertial
        body_x = dcm[:, 0]

        # Dot product should be close to 1 (parallel)
        dot = np.dot(body_x, r_hat)
        assert_allclose(abs(dot), 1.0, atol=0.01), f"Body X not aligned with up, dot={dot}"

    def test_altitude_flat_earth(self):
        """Altitude property should return z for flat Earth."""
        state = State.from_flat_earth(z=5000.0, mass_kg=1000.0)
        assert_allclose(state.altitude, 5000, atol=1e-10)

    def test_altitude_spherical_earth(self):
        """Altitude property should return distance from Earth center minus radius."""
        # Create state at Earth surface
        state = State.from_launch_pad(altitude_m=0.0)
        assert_allclose(state.altitude, 0, atol=100)  # Within 100m of surface

        # Create state 100km up
        state2 = State.from_launch_pad(altitude_m=100000.0)
        assert_allclose(state2.altitude, 100000, atol=100)


# =============================================================================
# Gravity Tests
# =============================================================================

class TestGravityModel:
    """Test gravity model calculations."""

    def test_constant_gravity_direction(self):
        """Constant gravity should point radially inward."""
        grav = Gravity(model=GravityModel.CONSTANT)

        # Test at various positions
        positions = [
            np.array([R_EARTH_EQ, 0, 0]),
            np.array([0, R_EARTH_EQ, 0]),
            np.array([0, 0, R_EARTH_EQ]),
        ]

        for pos in positions:
            g = grav.acceleration(pos)
            r_hat = pos / np.linalg.norm(pos)
            g_hat = g / np.linalg.norm(g)

            # Should point opposite to r_hat (toward center)
            assert_allclose(g_hat, -r_hat, atol=1e-10)

    def test_spherical_gravity_magnitude(self):
        """Spherical gravity magnitude should follow inverse square law."""
        grav = Gravity(model=GravityModel.SPHERICAL)

        # At Earth surface
        pos_surface = np.array([R_EARTH_EQ, 0, 0])
        g_surface = grav.magnitude(pos_surface)
        expected_g = MU_EARTH / R_EARTH_EQ**2
        assert_allclose(g_surface, expected_g, rtol=1e-6)

        # At 200 km altitude
        r_200km = R_EARTH_EQ + 200000
        pos_200km = np.array([r_200km, 0, 0])
        g_200km = grav.magnitude(pos_200km)
        expected_g_200km = MU_EARTH / r_200km**2
        assert_allclose(g_200km, expected_g_200km, rtol=1e-6)

        # Gravity should be less at higher altitude
        assert g_200km < g_surface

    def test_spherical_gravity_direction(self):
        """Spherical gravity should point toward Earth center."""
        grav = Gravity(model=GravityModel.SPHERICAL)

        pos = np.array([R_EARTH_EQ + 100000, 50000, 30000])
        g = grav.acceleration(pos)

        r_hat = pos / np.linalg.norm(pos)
        g_hat = g / np.linalg.norm(g)

        # Should point toward center (opposite to r_hat)
        assert_allclose(g_hat, -r_hat, atol=1e-10)


# =============================================================================
# Rigid Body Dynamics Tests
# =============================================================================

class TestRigidBodyDynamics:
    """Test rigid body equations of motion."""

    def test_pure_thrust_acceleration(self):
        """Pure thrust along body axis should accelerate in that direction."""
        # Simple state: body aligned with inertial, at origin
        state = State(
            position=np.array([R_EARTH_EQ, 0.0, 0.0], dtype=np.float64),
            velocity=np.zeros(3, dtype=np.float64),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),  # Identity
            angular_velocity=np.zeros(3, dtype=np.float64),
            mass=1000.0,
        )

        # Thrust along body X (= inertial X when identity quaternion)
        thrust_body = np.array([10000.0, 0.0, 0.0], dtype=np.float64)  # 10 kN

        inertia = np.diag([1000.0, 1000.0, 500.0])

        deriv = rigid_body_derivatives(
            state=state,
            force_body=thrust_body,
            moment_body=np.zeros(3),
            inertia=inertia,
        )

        # Expected acceleration = F/m = 10000/1000 = 10 m/s²
        expected_accel = np.array([10, 0, 0])
        assert_allclose(deriv.velocity_dot, expected_accel, atol=1e-10)

    def test_pure_moment_angular_acceleration(self):
        """Pure moment should cause angular acceleration."""
        state = State(
            position=np.zeros(3, dtype=np.float64),
            velocity=np.zeros(3, dtype=np.float64),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            angular_velocity=np.zeros(3, dtype=np.float64),
            mass=1000.0,
        )

        # Moment about Y axis
        moment_body = np.array([0.0, 1000.0, 0.0], dtype=np.float64)  # 1000 N·m

        inertia = np.diag([100.0, 200.0, 100.0])  # kg·m²

        deriv = rigid_body_derivatives(
            state=state,
            force_body=np.zeros(3),
            moment_body=moment_body,
            inertia=inertia,
        )

        # Expected angular acceleration = M/I = 1000/200 = 5 rad/s²
        expected_alpha = np.array([0, 5, 0])
        assert_allclose(deriv.angular_velocity_dot, expected_alpha, atol=1e-10)

    def test_euler_equations_gyroscopic(self):
        """Test gyroscopic coupling in Euler equations."""
        # Spinning about X, apply moment about Y
        omega = np.array([10.0, 0.0, 0.0], dtype=np.float64)  # Spinning at 10 rad/s about X
        moment = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # No external moment
        inertia = np.diag([100.0, 200.0, 300.0])  # Different inertias

        alpha = euler_rotational_dynamics(omega, moment, inertia)

        # With different inertias, gyroscopic coupling should be present
        # omega x (I * omega) = [10,0,0] x [1000,0,0] = [0,0,0]
        # So no coupling when spinning about principal axis
        assert_allclose(alpha, np.zeros(3), atol=1e-10)

    def test_quaternion_derivative_rotation(self):
        """Quaternion derivative should represent rotation."""
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # Identity
        omega = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # Rotating about Z at 1 rad/s

        q_dot = quaternion_derivative(q, omega)

        # For identity quaternion with omega_z = 1:
        # q_dot = 0.5 * omega_matrix @ q
        expected = 0.5 * np.array([0.0, 0.0, 0.0, 1.0])
        assert_allclose(q_dot, expected, atol=1e-10)


class TestRK4Integration:
    """Test RK4 integration step."""

    def test_rk4_constant_velocity(self):
        """With no forces, constant velocity should continue."""
        state = State(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            velocity=np.array([100.0, 0.0, 0.0], dtype=np.float64),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            angular_velocity=np.zeros(3, dtype=np.float64),
            mass=1000.0,
            flat_earth=True,
        )

        def derivatives_fn(s):
            from rocket.dynamics.state import StateDerivative
            return StateDerivative(
                position_dot=s.velocity,
                velocity_dot=np.zeros(3),
                quaternion_dot=np.zeros(4),
                angular_velocity_dot=np.zeros(3),
                mass_dot=0.0,
            )

        dt = 0.1
        new_state = rk4_step(state, dt, derivatives_fn)

        # Position should advance by v*dt = 100*0.1 = 10 m
        assert_allclose(new_state.position, [10, 0, 0], atol=1e-10)
        assert_allclose(new_state.velocity, [100, 0, 0], atol=1e-10)

    def test_rk4_constant_acceleration(self):
        """With constant force, should integrate correctly."""
        state = State(
            position=np.zeros(3, dtype=np.float64),
            velocity=np.zeros(3, dtype=np.float64),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            angular_velocity=np.zeros(3, dtype=np.float64),
            mass=1000.0,
            flat_earth=True,
        )

        accel = np.array([10.0, 0.0, 0.0], dtype=np.float64)  # 10 m/s² in X

        def derivatives_fn(s):
            from rocket.dynamics.state import StateDerivative
            return StateDerivative(
                position_dot=s.velocity,
                velocity_dot=accel,
                quaternion_dot=np.zeros(4),
                angular_velocity_dot=np.zeros(3),
                mass_dot=0.0,
            )

        dt = 1.0
        new_state = rk4_step(state, dt, derivatives_fn)

        # After 1s with 10 m/s² accel starting from rest:
        # v = a*t = 10 m/s
        # x = 0.5*a*t² = 5 m
        assert_allclose(new_state.velocity, [10, 0, 0], atol=1e-6)
        assert_allclose(new_state.position, [5, 0, 0], atol=1e-6)


class TestDynamicsWithGravity:
    """Test complete dynamics model with gravity."""

    def test_freefall_flat_earth(self):
        """Object in freefall should accelerate at g."""
        state = State.from_flat_earth(z=1000.0, mass_kg=100.0)

        inertia = np.diag([10.0, 10.0, 5.0])
        config = DynamicsConfig(gravity_model=GravityModel.CONSTANT)
        dynamics = RigidBodyDynamics(inertia=inertia, config=config)

        # No thrust
        deriv = dynamics.derivatives(
            state=state,
            thrust_body=np.zeros(3),
            moment_body=np.zeros(3),
        )

        # Should accelerate downward at ~9.8 m/s²
        # Gravity force is computed and transformed to body frame
        accel_mag = np.linalg.norm(deriv.velocity_dot)
        assert_allclose(accel_mag, 9.80665, rtol=0.01)

    def test_hovering(self):
        """Thrust equal to weight should result in zero acceleration."""
        mass = 1000.0  # kg
        state = State.from_flat_earth(z=1000.0, pitch_deg=90.0, mass_kg=mass)

        inertia = np.diag([1000.0, 1000.0, 500.0])
        config = DynamicsConfig(gravity_model=GravityModel.CONSTANT)
        dynamics = RigidBodyDynamics(inertia=inertia, config=config)

        # Thrust equal to weight, along body X (which points up)
        weight = mass * 9.80665
        thrust_body = np.array([weight, 0, 0])

        deriv = dynamics.derivatives(
            state=state,
            thrust_body=thrust_body,
            moment_body=np.zeros(3),
        )

        # Should have ~zero acceleration (hovering)
        assert_allclose(deriv.velocity_dot, np.zeros(3), atol=0.1)


# =============================================================================
# Gimbal Model Tests
# =============================================================================

class TestGimbalModel:
    """Test gimbal/TVC model calculations."""

    def test_no_gimbal_thrust_along_x(self):
        """With no gimbal, thrust should be along +X."""
        from rocket.propulsion.throttle_model import GimbalModel

        gimbal = GimbalModel(max_gimbal_angle=np.radians(5))

        thrust = gimbal.thrust_vector(10000.0, 0.0, 0.0)

        # Should be [10000, 0, 0]
        assert_allclose(thrust, [10000, 0, 0], atol=1e-10)

    def test_pitch_gimbal_deflection(self):
        """Pitch gimbal should deflect thrust in XZ plane."""
        from rocket.propulsion.throttle_model import GimbalModel

        gimbal = GimbalModel(max_gimbal_angle=np.radians(10))

        # 5 degree pitch deflection
        pitch_angle = np.radians(5)
        thrust = gimbal.thrust_vector(10000.0, pitch_angle, 0.0)

        # Fx = T * cos(pitch), Fz = -T * sin(pitch)
        expected_fx = 10000 * np.cos(pitch_angle)
        expected_fz = -10000 * np.sin(pitch_angle)

        assert_allclose(thrust[0], expected_fx, rtol=1e-6)
        assert_allclose(thrust[2], expected_fz, rtol=1e-6)
        assert_allclose(thrust[1], 0, atol=1e-10)

    def test_gimbal_moment_zero_at_zero_angle(self):
        """With no gimbal deflection, moment should be zero."""
        from rocket.propulsion.throttle_model import GimbalModel

        gimbal = GimbalModel(max_gimbal_angle=np.radians(5), gimbal_moment_arm=2.0)

        moment = gimbal.moment(10000.0, 0.0, 0.0)

        # Thrust along X, engine at -X: moment = r x F = [-2,0,0] x [10000,0,0] = 0
        assert_allclose(moment, np.zeros(3), atol=1e-10)

    def test_gimbal_moment_pitch(self):
        """Pitch gimbal should create moment about Y axis."""
        from rocket.propulsion.throttle_model import GimbalModel

        gimbal = GimbalModel(max_gimbal_angle=np.radians(10), gimbal_moment_arm=2.0)

        pitch_angle = np.radians(5)
        moment = gimbal.moment(10000.0, pitch_angle, 0.0)

        # Engine at r = [-2, 0, 0]
        # Thrust F = [T*cos(p), 0, -T*sin(p)]  (Fz negative for positive pitch)
        # Moment = r x F, My = rz*Fx - rx*Fz = 0 - (-2)*(-T*sin(p)) = -2*T*sin(p)
        # Negative My = nose pitches UP (correct for positive pitch gimbal)
        expected_my = -2.0 * 10000.0 * np.sin(pitch_angle)

        assert_allclose(moment[1], expected_my, rtol=1e-6)


# =============================================================================
# Integration Tests
# =============================================================================

class TestVerticalLaunch:
    """Integration test: simple vertical launch."""

    def test_vertical_ascent(self):
        """Rocket pointing up with thrust > weight should ascend."""
        mass = 1000.0
        state = State.from_flat_earth(z=0.0, pitch_deg=90.0, mass_kg=mass)

        inertia = np.diag([500.0, 500.0, 100.0])
        config = DynamicsConfig(gravity_model=GravityModel.CONSTANT)
        dynamics = RigidBodyDynamics(inertia=inertia, config=config)

        # Thrust = 2x weight
        weight = mass * 9.80665
        thrust_body = np.array([2 * weight, 0, 0])

        # Simulate for 1 second
        dt = 0.01
        for _ in range(100):
            deriv = dynamics.derivatives(
                state=state,
                thrust_body=thrust_body,
                moment_body=np.zeros(3),
            )
            state = rk4_step(state, dt, lambda s, d=deriv: d)

        # Should have positive altitude
        assert state.altitude > 0

        # Should have positive vertical velocity
        assert state.velocity[2] > 0

        # Expected: net accel = g upward, v ≈ 9.8 m/s, z ≈ 4.9 m
        assert_allclose(state.velocity[2], 9.80665, rtol=0.1)
        assert_allclose(state.altitude, 4.9, rtol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

