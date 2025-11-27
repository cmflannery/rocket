"""Thrust Vector Control (TVC) controller.

Provides attitude control using engine gimbaling for rockets.
Combines PID control with gimbal dynamics and limits.

Example:
    >>> from rocket.gnc.control import TVCController
    >>> from rocket.dynamics import State
    >>>
    >>> tvc = TVCController(
    ...     pitch_gains=PIDGains(kp=2.0, ki=0.1, kd=0.5),
    ...     yaw_gains=PIDGains(kp=2.0, ki=0.1, kd=0.5),
    ...     max_gimbal=np.radians(6),
    ... )
    >>>
    >>> # Compute gimbal commands
    >>> gimbal_cmd = tvc.update(state, target_attitude, dt)
"""

from dataclasses import dataclass, field

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

from rocket.dynamics.state import State
from rocket.gnc.control.pid import PIDController, PIDGains

# =============================================================================
# TVC Controller
# =============================================================================


@beartype
@dataclass
class TVCController:
    """Thrust Vector Control (TVC) attitude controller.

    Uses PID control to track commanded attitude angles
    by computing gimbal angle commands.

    Two control modes:
    1. Attitude mode: Track commanded pitch/yaw angles
    2. Rate mode: Track commanded pitch/yaw rates

    Attributes:
        pitch_gains: PID gains for pitch control
        yaw_gains: PID gains for yaw control
        roll_gains: PID gains for roll control (if RCS available)
        max_gimbal: Maximum gimbal angle [rad]
        gimbal_rate_limit: Maximum gimbal rate [rad/s]
    """
    # Default gains tuned for typical rocket dynamics
    # Conservative values prioritizing stability over responsiveness
    # Low integral gain to prevent windup during gravity turns
    pitch_gains: PIDGains = field(default_factory=lambda: PIDGains(kp=0.3, ki=0.0, kd=0.1))
    yaw_gains: PIDGains = field(default_factory=lambda: PIDGains(kp=0.3, ki=0.0, kd=0.1))
    roll_gains: PIDGains = field(default_factory=lambda: PIDGains(kp=0.3, ki=0.0, kd=0.1))
    max_gimbal: float = np.radians(6.0)  # ±6 degrees
    gimbal_rate_limit: float = np.radians(10.0)  # 10 deg/s (conservative)

    # Internal controllers
    _pitch_ctrl: PIDController = field(init=False, repr=False)
    _yaw_ctrl: PIDController = field(init=False, repr=False)
    _roll_ctrl: PIDController = field(init=False, repr=False)

    # State
    _prev_gimbal: tuple[float, float] = field(default=(0.0, 0.0), init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize internal controllers."""
        self._pitch_ctrl = PIDController.from_gains(
            self.pitch_gains,
            output_limits=(-self.max_gimbal, self.max_gimbal)
        )
        self._yaw_ctrl = PIDController.from_gains(
            self.yaw_gains,
            output_limits=(-self.max_gimbal, self.max_gimbal)
        )
        self._roll_ctrl = PIDController.from_gains(
            self.roll_gains,
            output_limits=(-0.5, 0.5)  # Roll moment coefficient
        )

    @beartype
    def reset(self) -> None:
        """Reset controller state."""
        self._pitch_ctrl.reset()
        self._yaw_ctrl.reset()
        self._roll_ctrl.reset()
        self._prev_gimbal = (0.0, 0.0)

    @beartype
    def update_attitude(
        self,
        state: State,
        target_pitch: float,
        target_yaw: float,
        dt: float,
    ) -> tuple[float, float]:
        """Compute gimbal commands to track target attitude.

        Uses primarily rate damping for stability during gravity turns.
        Small proportional gain helps with initial kick maneuver.

        Args:
            state: Current vehicle state
            target_pitch: Target pitch angle [rad]
            target_yaw: Target yaw angle [rad]
            dt: Time step [s]

        Returns:
            Tuple of (pitch_gimbal, yaw_gimbal) commands [rad]
        """
        # Get current attitude and rates
        roll, pitch, yaw = state.euler_angles
        p, q, r = state.angular_velocity  # Body rates

        # Check for NaN in state (numerical instability protection)
        if np.any(np.isnan([roll, pitch, yaw, p, q, r])):
            return self._prev_gimbal

        # Compute errors
        pitch_error = target_pitch - pitch
        yaw_error = target_yaw - yaw

        # Wrap errors to [-pi, pi]
        pitch_error = np.arctan2(np.sin(pitch_error), np.cos(pitch_error))
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

        # For near-vertical flight (pitch > 60°), disable yaw control
        # Yaw is undefined when vertical
        if abs(pitch) > np.radians(60):
            yaw_error = 0.0

        # During gravity turn (pitch < 80°), use rate-damping only
        # This prevents the controller from fighting the natural turn
        if abs(pitch) < np.radians(80):
            # Pure rate damping - just oppose angular velocity
            rate_gain = 0.01  # Gentle damping
            pitch_cmd = -rate_gain * q
            yaw_cmd = -rate_gain * r
        else:
            # Near-vertical: small proportional control + rate damping
            # Limit error to prevent aggressive corrections
            max_error = np.radians(3.0)
            pitch_error = np.clip(pitch_error, -max_error, max_error)
            yaw_error = np.clip(yaw_error, -max_error, max_error)

            # Small proportional gain with rate damping
            kp = 0.1  # Small proportional gain
            kd = 0.02  # Rate damping

            pitch_cmd = kp * pitch_error - kd * q
            yaw_cmd = kp * yaw_error - kd * r

        # Apply rate limiting
        pitch_cmd, yaw_cmd = self._rate_limit((pitch_cmd, yaw_cmd), dt)

        self._prev_gimbal = (pitch_cmd, yaw_cmd)
        return (pitch_cmd, yaw_cmd)

    @beartype
    def update_rate(
        self,
        state: State,
        target_pitch_rate: float,
        target_yaw_rate: float,
        dt: float,
    ) -> tuple[float, float]:
        """Compute gimbal commands to track target rates.

        Args:
            state: Current vehicle state
            target_pitch_rate: Target pitch rate [rad/s]
            target_yaw_rate: Target yaw rate [rad/s]
            dt: Time step [s]

        Returns:
            Tuple of (pitch_gimbal, yaw_gimbal) commands [rad]
        """
        # Get current rates (in body frame)
        p, q, r = state.angular_velocity

        # Compute rate errors (q is pitch rate, r is yaw rate in body frame)
        pitch_rate_error = target_pitch_rate - q
        yaw_rate_error = target_yaw_rate - r

        # PID control
        pitch_cmd = self._pitch_ctrl.update(pitch_rate_error, dt)
        yaw_cmd = self._yaw_ctrl.update(yaw_rate_error, dt)

        # Apply rate limiting
        pitch_cmd, yaw_cmd = self._rate_limit((pitch_cmd, yaw_cmd), dt)

        self._prev_gimbal = (pitch_cmd, yaw_cmd)
        return (pitch_cmd, yaw_cmd)

    def _rate_limit(
        self,
        target: tuple[float, float],
        dt: float,
    ) -> tuple[float, float]:
        """Apply rate limiting to gimbal commands."""
        max_change = self.gimbal_rate_limit * dt

        pitch_delta = np.clip(
            target[0] - self._prev_gimbal[0],
            -max_change, max_change
        )
        yaw_delta = np.clip(
            target[1] - self._prev_gimbal[1],
            -max_change, max_change
        )

        pitch_cmd = np.clip(
            self._prev_gimbal[0] + pitch_delta,
            -self.max_gimbal, self.max_gimbal
        )
        yaw_cmd = np.clip(
            self._prev_gimbal[1] + yaw_delta,
            -self.max_gimbal, self.max_gimbal
        )

        return (float(pitch_cmd), float(yaw_cmd))

    @beartype
    def roll_moment(
        self,
        state: State,
        target_roll: float,
        dt: float,
    ) -> float:
        """Compute roll moment coefficient for RCS.

        TVC cannot control roll, so this is for RCS systems.

        Args:
            state: Current state
            target_roll: Target roll angle [rad]
            dt: Time step [s]

        Returns:
            Roll moment coefficient [-1 to 1]
        """
        roll, _, _ = state.euler_angles
        roll_error = target_roll - roll
        roll_error = np.arctan2(np.sin(roll_error), np.cos(roll_error))
        return self._roll_ctrl.update(roll_error, dt)


# =============================================================================
# Full Attitude Controller
# =============================================================================


@beartype
@dataclass
class AttitudeController:
    """Complete attitude controller with TVC and optional RCS.

    Manages both gimbal (pitch/yaw) and reaction control (roll).

    Example:
        >>> ctrl = AttitudeController(max_gimbal=np.radians(6))
        >>>
        >>> # Target attitude
        >>> target = np.array([0, np.radians(85), 0])  # roll, pitch, yaw
        >>>
        >>> gimbal, rcs = ctrl.update(state, target, dt)
    """
    tvc: TVCController = field(default_factory=TVCController)
    has_rcs: bool = False

    @beartype
    def reset(self) -> None:
        """Reset controller state."""
        self.tvc.reset()

    @beartype
    def update(
        self,
        state: State,
        target_attitude: NDArray[np.float64],
        dt: float,
    ) -> tuple[tuple[float, float], float]:
        """Compute control outputs for target attitude.

        Args:
            state: Current vehicle state
            target_attitude: Target [roll, pitch, yaw] [rad]
            dt: Time step [s]

        Returns:
            Tuple of ((pitch_gimbal, yaw_gimbal), roll_rcs)
        """
        target_roll, target_pitch, target_yaw = target_attitude

        gimbal = self.tvc.update_attitude(state, target_pitch, target_yaw, dt)

        rcs = 0.0
        if self.has_rcs:
            rcs = self.tvc.roll_moment(state, target_roll, dt)

        return gimbal, rcs

