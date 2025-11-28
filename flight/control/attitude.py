"""Attitude control for rocket vehicles.

Provides quaternion-based attitude control using thrust vector control (TVC).
Computes gimbal commands to achieve target attitude or track target direction.

This is flight software - designed to run on the vehicle.

Example:
    >>> from flight.control import AttitudeController
    >>> from rocket.simulation import Simulator, ThrustCommand
    >>>
    >>> controller = AttitudeController(kp=2.0, kd=0.5)
    >>> target_attitude = guidance.compute(state).target_attitude
    >>>
    >>> gimbal = controller.compute(state, target_attitude)
    >>> thrust = ThrustCommand(magnitude=1e6, gimbal_pitch=gimbal[0], gimbal_yaw=gimbal[1])
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rocket.dynamics.state import State, quaternion_conjugate, quaternion_multiply

# =============================================================================
# Attitude Controller
# =============================================================================


@dataclass
class AttitudeController:
    """Quaternion-based attitude controller with TVC.

    Uses proportional-derivative control on attitude error.
    Computes gimbal commands to achieve target attitude.

    Attributes:
        kp: Proportional gain for attitude error
        kd: Derivative gain for angular rate
        max_gimbal: Maximum gimbal angle [rad]
        rate_limit: Maximum gimbal rate [rad/s]
    """
    kp: float = 2.0
    kd: float = 0.5
    max_gimbal: float = np.radians(6.0)
    rate_limit: float = np.radians(20.0)

    # Internal state for rate limiting
    _last_gimbal: tuple[float, float] = (0.0, 0.0)
    _last_time: float = -1.0

    def compute(
        self,
        state: State,
        target_attitude: NDArray[np.float64] | None = None,
        target_direction: NDArray[np.float64] | None = None,
        dt: float = 0.02,
    ) -> tuple[float, float]:
        """Compute gimbal commands.

        Args:
            state: Current vehicle state
            target_attitude: Target quaternion [q0, q1, q2, q3]
            target_direction: Alternative: target thrust direction (inertial)
            dt: Time step for rate limiting [s]

        Returns:
            (gimbal_pitch, gimbal_yaw) in radians
        """
        if target_attitude is None and target_direction is None:
            return (0.0, 0.0)

        if target_attitude is None:
            # Convert direction to attitude
            target_attitude = self._direction_to_attitude(state, target_direction)

        # Compute attitude error quaternion
        # q_error = q_target * q_current^(-1)
        q_current = state.quaternion
        q_target = target_attitude

        q_error = quaternion_multiply(q_target, quaternion_conjugate(q_current))

        # Ensure short path (positive scalar part)
        if q_error[0] < 0:
            q_error = -q_error

        # Small angle approximation: error angles ≈ 2 * vector part
        error_body = 2.0 * q_error[1:4]

        # Angular rate (already in body frame)
        omega = state.angular_velocity

        # PD control law
        # Gimbal pitch (rotation about body Y) corrects roll/pitch errors
        # Gimbal yaw (rotation about body Z) corrects yaw errors
        gimbal_pitch = self.kp * error_body[1] - self.kd * omega[1]
        gimbal_yaw = -self.kp * error_body[2] + self.kd * omega[2]

        # Apply limits
        gimbal_pitch = np.clip(gimbal_pitch, -self.max_gimbal, self.max_gimbal)
        gimbal_yaw = np.clip(gimbal_yaw, -self.max_gimbal, self.max_gimbal)

        # Rate limiting
        if self._last_time >= 0 and dt > 0:
            max_delta = self.rate_limit * dt
            dp = gimbal_pitch - self._last_gimbal[0]
            dy = gimbal_yaw - self._last_gimbal[1]

            dp = np.clip(dp, -max_delta, max_delta)
            dy = np.clip(dy, -max_delta, max_delta)

            gimbal_pitch = self._last_gimbal[0] + dp
            gimbal_yaw = self._last_gimbal[1] + dy

        self._last_gimbal = (gimbal_pitch, gimbal_yaw)
        self._last_time = state.time

        return (gimbal_pitch, gimbal_yaw)

    def _direction_to_attitude(
        self,
        state: State,
        direction: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert target direction to attitude quaternion."""
        from rocket.dynamics.state import dcm_to_quaternion

        # Body X = target direction
        body_x = direction / np.linalg.norm(direction)

        # Body Z = nadir
        if state.flat_earth:
            nadir = np.array([0.0, 0.0, -1.0])
        else:
            nadir = -state.position / np.linalg.norm(state.position)

        # Orthogonalize
        body_z = nadir - np.dot(nadir, body_x) * body_x
        z_mag = np.linalg.norm(body_z)
        if z_mag > 0.01:
            body_z = body_z / z_mag
        else:
            body_z = np.array([0.0, 0.0, 1.0])
            body_z = body_z - np.dot(body_z, body_x) * body_x
            body_z = body_z / np.linalg.norm(body_z)

        body_y = np.cross(body_z, body_x)

        dcm = np.column_stack([body_x, body_y, body_z])
        return dcm_to_quaternion(dcm.T)

    def reset(self) -> None:
        """Reset controller state."""
        self._last_gimbal = (0.0, 0.0)
        self._last_time = -1.0


# =============================================================================
# Rate Controller (for landing)
# =============================================================================


@dataclass
class RateController:
    """Angular rate controller.

    Lower-level controller that tracks commanded angular rates.
    Used as inner loop for attitude control or direct rate commands.

    Attributes:
        kp: Proportional gain
        max_gimbal: Maximum gimbal angle [rad]
    """
    kp: float = 1.0
    max_gimbal: float = np.radians(6.0)

    def compute(
        self,
        state: State,
        target_rate: NDArray[np.float64],
    ) -> tuple[float, float]:
        """Compute gimbal to achieve target angular rate.

        Args:
            state: Current vehicle state
            target_rate: Target angular rate [p, q, r] in body frame [rad/s]

        Returns:
            (gimbal_pitch, gimbal_yaw) in radians
        """
        omega = state.angular_velocity
        rate_error = target_rate - omega

        gimbal_pitch = self.kp * rate_error[1]
        gimbal_yaw = -self.kp * rate_error[2]

        gimbal_pitch = np.clip(gimbal_pitch, -self.max_gimbal, self.max_gimbal)
        gimbal_yaw = np.clip(gimbal_yaw, -self.max_gimbal, self.max_gimbal)

        return (gimbal_pitch, gimbal_yaw)

