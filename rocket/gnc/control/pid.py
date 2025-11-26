"""PID controller implementation.

Provides a general-purpose PID controller with:
- Anti-windup for integral term
- Derivative filtering
- Output saturation
- Gain scheduling support

Example:
    >>> from rocket.gnc.control import PIDController
    >>>
    >>> # Create pitch rate controller
    >>> ctrl = PIDController(kp=1.0, ki=0.1, kd=0.5, output_limits=(-0.1, 0.1))
    >>>
    >>> # Compute control output
    >>> error = target_pitch_rate - actual_pitch_rate
    >>> command = ctrl.update(error, dt=0.01)
"""

from dataclasses import dataclass, field

import numpy as np
from beartype import beartype

# =============================================================================
# PID Gains
# =============================================================================


@beartype
@dataclass
class PIDGains:
    """PID controller gains.

    Attributes:
        kp: Proportional gain
        ki: Integral gain
        kd: Derivative gain
    """
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0

    def scale(self, factor: float) -> "PIDGains":
        """Scale all gains by a factor."""
        return PIDGains(
            kp=self.kp * factor,
            ki=self.ki * factor,
            kd=self.kd * factor,
        )


# =============================================================================
# PID Controller
# =============================================================================


@beartype
@dataclass
class PIDController:
    """General-purpose PID controller.

    Implements the parallel PID form:
        u = kp * e + ki * integral(e) + kd * de/dt

    Features:
    - Anti-windup with integral clamping
    - First-order derivative filter
    - Output saturation

    Attributes:
        kp: Proportional gain
        ki: Integral gain
        kd: Derivative gain
        output_limits: (min, max) output limits
        integral_limits: (min, max) integral term limits (anti-windup)
        derivative_filter: Low-pass filter coefficient for derivative (0-1)
    """
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    output_limits: tuple[float, float] | None = None
    integral_limits: tuple[float, float] | None = None
    derivative_filter: float = 0.1  # Filter coefficient (0 = no filter)

    # Internal state
    _integral: float = field(default=0.0, init=False, repr=False)
    _prev_error: float | None = field(default=None, init=False, repr=False)
    _prev_derivative: float = field(default=0.0, init=False, repr=False)

    @classmethod
    def from_gains(
        cls,
        gains: PIDGains,
        output_limits: tuple[float, float] | None = None,
    ) -> "PIDController":
        """Create controller from PIDGains object."""
        return cls(
            kp=gains.kp,
            ki=gains.ki,
            kd=gains.kd,
            output_limits=output_limits,
        )

    @beartype
    def reset(self) -> None:
        """Reset controller state (integral and derivative history)."""
        self._integral = 0.0
        self._prev_error = None
        self._prev_derivative = 0.0

    @beartype
    def update(self, error: float, dt: float) -> float:
        """Compute PID control output.

        Args:
            error: Current error (setpoint - measurement)
            dt: Time step [s]

        Returns:
            Control output
        """
        if dt <= 0:
            return 0.0

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self._integral += error * dt
        if self.integral_limits:
            self._integral = np.clip(
                self._integral,
                self.integral_limits[0],
                self.integral_limits[1]
            )
        i_term = self.ki * self._integral

        # Derivative term with filtering
        if self._prev_error is not None:
            raw_derivative = (error - self._prev_error) / dt
            # Low-pass filter
            alpha = self.derivative_filter
            self._prev_derivative = (
                alpha * raw_derivative +
                (1 - alpha) * self._prev_derivative
            )
        d_term = self.kd * self._prev_derivative

        self._prev_error = error

        # Total output
        output = p_term + i_term + d_term

        # Output saturation
        if self.output_limits:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])

        return float(output)

    @beartype
    def update_with_derivative(
        self,
        error: float,
        error_derivative: float,
        dt: float,
    ) -> float:
        """Compute PID output with externally computed derivative.

        Useful when derivative is available from sensors (e.g., rate gyro).

        Args:
            error: Current error
            error_derivative: Rate of change of error
            dt: Time step [s]

        Returns:
            Control output
        """
        if dt <= 0:
            return 0.0

        # Proportional
        p_term = self.kp * error

        # Integral with anti-windup
        self._integral += error * dt
        if self.integral_limits:
            self._integral = np.clip(
                self._integral,
                self.integral_limits[0],
                self.integral_limits[1]
            )
        i_term = self.ki * self._integral

        # Derivative (use provided rate directly)
        d_term = -self.kd * error_derivative  # Negative for "derivative on measurement"

        output = p_term + i_term + d_term

        if self.output_limits:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])

        return float(output)

    @property
    def gains(self) -> PIDGains:
        """Get current gains as PIDGains object."""
        return PIDGains(kp=self.kp, ki=self.ki, kd=self.kd)

    @gains.setter
    def gains(self, value: PIDGains) -> None:
        """Set gains from PIDGains object."""
        self.kp = value.kp
        self.ki = value.ki
        self.kd = value.kd


# =============================================================================
# Cascaded PID Controller
# =============================================================================


@beartype
@dataclass
class CascadedPID:
    """Cascaded (inner/outer loop) PID controller.

    Common architecture for attitude control:
    - Outer loop: angle error -> rate command
    - Inner loop: rate error -> actuator command

    Example:
        >>> cascade = CascadedPID(
        ...     outer=PIDController(kp=2.0),
        ...     inner=PIDController(kp=1.0, kd=0.1),
        ... )
        >>>
        >>> # Control loop
        >>> angle_error = target_angle - current_angle
        >>> rate_cmd = cascade.outer.update(angle_error, dt)
        >>> rate_error = rate_cmd - current_rate
        >>> actuator_cmd = cascade.inner.update(rate_error, dt)
    """
    outer: PIDController
    inner: PIDController

    @beartype
    def reset(self) -> None:
        """Reset both loops."""
        self.outer.reset()
        self.inner.reset()

    @beartype
    def update(
        self,
        outer_error: float,
        inner_measurement: float,
        dt: float,
    ) -> float:
        """Compute cascaded output.

        Args:
            outer_error: Error for outer loop (e.g., angle error)
            inner_measurement: Measurement for inner loop (e.g., rate)
            dt: Time step [s]

        Returns:
            Final control output
        """
        # Outer loop: error -> inner loop setpoint
        inner_setpoint = self.outer.update(outer_error, dt)

        # Inner loop: setpoint - measurement -> output
        inner_error = inner_setpoint - inner_measurement
        return self.inner.update(inner_error, dt)

