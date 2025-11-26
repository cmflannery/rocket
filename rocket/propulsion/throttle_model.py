"""Throttle and gimbal models for dynamic engine simulation.

Bridges static engine design (EnginePerformance) to dynamic simulation
by computing thrust and mass flow at various throttle settings and altitudes.

Example:
    >>> from rocket import design_engine, EngineInputs
    >>> from rocket.propulsion import ThrottleModel
    >>>
    >>> # Design engine
    >>> inputs = EngineInputs.from_propellants("LOX", "CH4", ...)
    >>> perf, geom = design_engine(inputs)
    >>>
    >>> # Create throttle model
    >>> throttle = ThrottleModel(perf, min_throttle=0.4)
    >>>
    >>> # Get thrust at 80% throttle, 10km altitude
    >>> thrust_N, mdot = throttle.at(throttle=0.8, altitude=10000)
"""

from dataclasses import dataclass

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

from rocket.engine import EnginePerformance
from rocket.environment.atmosphere import Atmosphere

# Standard gravity
G0 = 9.80665


# =============================================================================
# Throttle Model
# =============================================================================


@beartype
@dataclass
class ThrottleModel:
    """Dynamic throttle model for engine simulation.

    Computes thrust and mass flow as functions of:
    - Throttle setting (0 to 1)
    - Altitude (for atmospheric effects)

    Uses a simple model:
    - Thrust at sea level = mdot * Isp_sl * g0
    - Thrust in vacuum = mdot * Isp_vac * g0
    - At intermediate altitudes, linearly interpolate based on pressure

    Attributes:
        engine: Static engine performance data
        min_throttle: Minimum stable throttle setting (e.g., 0.4 for 40%)
        max_throttle: Maximum throttle setting (typically 1.0)
        throttle_rate: Maximum throttle change rate [1/s]
    """
    engine: EnginePerformance
    min_throttle: float = 0.0
    max_throttle: float = 1.0
    throttle_rate: float = 1.0  # Full range in 1 second

    _atmosphere: Atmosphere | None = None
    _thrust_sl: float = 0.0
    _thrust_vac: float = 0.0

    def __post_init__(self) -> None:
        """Initialize atmosphere model and compute base thrust values."""
        self._atmosphere = Atmosphere()

        # Compute thrust at sea level and vacuum
        mdot = self.engine.mdot.value
        isp_sl = self.engine.isp.value
        isp_vac = self.engine.isp_vac.value

        self._thrust_sl = mdot * isp_sl * G0
        self._thrust_vac = mdot * isp_vac * G0

    @beartype
    def thrust(
        self,
        throttle: float,
        altitude: float = 0.0,
    ) -> float:
        """Get thrust at throttle setting and altitude.

        Args:
            throttle: Throttle setting (0 to 1)
            altitude: Altitude [m]

        Returns:
            Thrust [N]
        """
        # Clamp throttle to valid range
        throttle = max(self.min_throttle, min(self.max_throttle, throttle))

        # Get atmospheric pressure
        if self._atmosphere is None:
            self._atmosphere = Atmosphere()

        pressure = self._atmosphere.pressure(altitude)
        p_sl = 101325.0  # Sea level pressure [Pa]

        # Interpolate between sea level and vacuum thrust
        # At p_sl, use thrust_sl; at 0, use thrust_vac
        if pressure >= p_sl:
            base_thrust = self._thrust_sl
        elif pressure <= 0:
            base_thrust = self._thrust_vac
        else:
            # Linear interpolation
            frac = pressure / p_sl
            base_thrust = self._thrust_vac + frac * (self._thrust_sl - self._thrust_vac)

        # Scale by throttle (linear approximation)
        return throttle * base_thrust

    @beartype
    def mass_flow(
        self,
        throttle: float,
        altitude: float = 0.0,
    ) -> float:
        """Get mass flow rate at throttle setting.

        Args:
            throttle: Throttle setting (0 to 1)
            altitude: Altitude [m] (currently unused, future: altitude effects)

        Returns:
            Mass flow rate [kg/s]
        """
        throttle = max(self.min_throttle, min(self.max_throttle, throttle))

        # Mass flow scales linearly with throttle
        return throttle * self.engine.mdot.value

    @beartype
    def at(
        self,
        throttle: float,
        altitude: float = 0.0,
    ) -> tuple[float, float]:
        """Get thrust and mass flow at operating point.

        Args:
            throttle: Throttle setting (0 to 1)
            altitude: Altitude [m]

        Returns:
            Tuple of (thrust [N], mass_flow [kg/s])
        """
        return (
            self.thrust(throttle, altitude),
            self.mass_flow(throttle, altitude),
        )

    @beartype
    def isp(
        self,
        throttle: float,
        altitude: float = 0.0,
    ) -> float:
        """Get specific impulse at operating point.

        Args:
            throttle: Throttle setting
            altitude: Altitude [m]

        Returns:
            Specific impulse [s]
        """
        F, mdot = self.at(throttle, altitude)
        if mdot <= 0:
            return 0.0
        return F / (mdot * G0)

    @beartype
    def burn_time(
        self,
        propellant_mass: float,
        throttle: float = 1.0,
    ) -> float:
        """Calculate burn time for given propellant mass.

        Args:
            propellant_mass: Available propellant [kg]
            throttle: Throttle setting

        Returns:
            Burn time [s]
        """
        mdot = self.mass_flow(throttle)
        if mdot <= 0:
            return float('inf')
        return propellant_mass / mdot

    @beartype
    def rate_limit_throttle(
        self,
        current: float,
        target: float,
        dt: float,
    ) -> float:
        """Apply rate limiting to throttle command.

        Args:
            current: Current throttle setting
            target: Target throttle setting
            dt: Time step [s]

        Returns:
            Rate-limited throttle setting
        """
        max_change = self.throttle_rate * dt
        delta = target - current
        delta = np.clip(delta, -max_change, max_change)
        return current + delta


# =============================================================================
# Gimbal Model
# =============================================================================


@beartype
@dataclass
class GimbalModel:
    """Thrust vector control (TVC) gimbal model.

    Models engine gimbaling for attitude control, computing
    thrust vector direction and resulting moments.

    Attributes:
        max_gimbal_angle: Maximum gimbal angle [rad]
        gimbal_rate: Maximum gimbal rate [rad/s]
        gimbal_moment_arm: Distance from CG to gimbal point [m]
    """
    max_gimbal_angle: float = np.radians(6.0)  # ±6 degrees typical
    gimbal_rate: float = np.radians(20.0)  # 20 deg/s typical
    gimbal_moment_arm: float = 2.0  # [m]

    @beartype
    def thrust_vector(
        self,
        thrust_magnitude: float,
        gimbal_pitch: float,
        gimbal_yaw: float,
    ) -> NDArray[np.float64]:
        """Compute thrust vector in body frame.

        Args:
            thrust_magnitude: Thrust force magnitude [N]
            gimbal_pitch: Pitch gimbal angle [rad] (rotation about Y)
            gimbal_yaw: Yaw gimbal angle [rad] (rotation about Z)

        Returns:
            Thrust vector [Fx, Fy, Fz] in body frame [N]
            (Nominal thrust is along +X in body frame - rocket forward axis)
        """
        # Clamp gimbal angles
        pitch = np.clip(gimbal_pitch, -self.max_gimbal_angle, self.max_gimbal_angle)
        yaw = np.clip(gimbal_yaw, -self.max_gimbal_angle, self.max_gimbal_angle)

        # Thrust direction (nominal along +X, deflected by gimbal)
        # Pitch deflects in XZ plane, yaw deflects in XY plane
        Fx = thrust_magnitude * np.cos(pitch) * np.cos(yaw)
        Fy = thrust_magnitude * np.sin(yaw)
        Fz = -thrust_magnitude * np.sin(pitch)

        return np.array([Fx, Fy, Fz])

    @beartype
    def moment(
        self,
        thrust_magnitude: float,
        gimbal_pitch: float,
        gimbal_yaw: float,
    ) -> NDArray[np.float64]:
        """Compute moment from gimbaled thrust.

        Args:
            thrust_magnitude: Thrust force magnitude [N]
            gimbal_pitch: Pitch gimbal angle [rad]
            gimbal_yaw: Yaw gimbal angle [rad]

        Returns:
            Moment vector [Mx, My, Mz] in body frame [N*m]
        """
        thrust = self.thrust_vector(thrust_magnitude, gimbal_pitch, gimbal_yaw)

        # Engine position relative to CG (assumed along -X axis, at rear of vehicle)
        engine_pos = np.array([-self.gimbal_moment_arm, 0.0, 0.0])

        # Moment = r x F
        moment = np.cross(engine_pos, thrust)

        return moment

    @beartype
    def rate_limit_gimbal(
        self,
        current: tuple[float, float],
        target: tuple[float, float],
        dt: float,
    ) -> tuple[float, float]:
        """Apply rate limiting to gimbal command.

        Args:
            current: Current (pitch, yaw) angles [rad]
            target: Target (pitch, yaw) angles [rad]
            dt: Time step [s]

        Returns:
            Rate-limited (pitch, yaw) angles [rad]
        """
        max_change = self.gimbal_rate * dt

        pitch_delta = np.clip(target[0] - current[0], -max_change, max_change)
        yaw_delta = np.clip(target[1] - current[1], -max_change, max_change)

        new_pitch = np.clip(current[0] + pitch_delta,
                          -self.max_gimbal_angle, self.max_gimbal_angle)
        new_yaw = np.clip(current[1] + yaw_delta,
                        -self.max_gimbal_angle, self.max_gimbal_angle)

        return (new_pitch, new_yaw)


# =============================================================================
# Combined Propulsion System
# =============================================================================


@beartype
@dataclass
class PropulsionSystem:
    """Complete propulsion system with throttle and gimbal.

    Combines throttle and gimbal models for full thrust vector control.

    Example:
        >>> prop = PropulsionSystem(throttle_model, gimbal_model)
        >>>
        >>> # Get thrust and moment at current state
        >>> thrust, moment, mdot = prop.forces_and_moments(
        ...     throttle=0.8,
        ...     gimbal=(0.02, -0.01),  # pitch, yaw in rad
        ...     altitude=5000,
        ... )
    """
    throttle: ThrottleModel
    gimbal: GimbalModel

    @beartype
    def forces_and_moments(
        self,
        throttle: float,
        gimbal: tuple[float, float],
        altitude: float = 0.0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """Compute thrust force, moment, and mass flow.

        Args:
            throttle: Throttle setting (0 to 1)
            gimbal: Gimbal angles (pitch, yaw) [rad]
            altitude: Altitude [m]

        Returns:
            Tuple of:
            - Thrust vector in body frame [N]
            - Moment vector in body frame [N*m]
            - Mass flow rate [kg/s]
        """
        thrust_mag = self.throttle.thrust(throttle, altitude)
        mdot = self.throttle.mass_flow(throttle, altitude)

        pitch, yaw = gimbal
        thrust_vec = self.gimbal.thrust_vector(thrust_mag, pitch, yaw)
        moment = self.gimbal.moment(thrust_mag, pitch, yaw)

        return thrust_vec, moment, mdot
