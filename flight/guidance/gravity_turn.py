"""Gravity turn ascent guidance.

Implements efficient ascent trajectories for launch vehicles using
the gravity turn technique, which minimizes steering losses by
allowing gravity to naturally pitch the vehicle over.

Phases:
1. Vertical rise: Clear the launch pad and build initial velocity
2. Pitch kick: Small pitch maneuver to initiate the turn
3. Gravity turn: Follow velocity vector (zero angle of attack)
4. Insertion: Coast or burn to target orbit

This is flight software - designed to run on the vehicle.
Test against the simulation in rocket/.

Example:
    >>> from flight.guidance import GravityTurnGuidance
    >>> from rocket.simulation import Simulator
    >>>
    >>> sim = Simulator.from_launch_pad(...)
    >>> guidance = GravityTurnGuidance(
    ...     target_altitude=200e3,
    ...     target_inclination=28.5,
    ... )
    >>>
    >>> while not guidance.is_complete(sim.get_state()):
    ...     state = sim.get_state()
    ...     cmd = guidance.compute(state)
    ...     sim.step(cmd.thrust, gimbal=(cmd.gimbal_pitch, cmd.gimbal_yaw), dt=0.02)
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from rocket.dynamics.state import State
from rocket.environment.gravity import R_EARTH_EQ
from rocket.orbital import OMEGA_EARTH

# =============================================================================
# Guidance Output
# =============================================================================


class GuidanceCommand(NamedTuple):
    """Output from guidance algorithm.

    Attributes:
        thrust: Commanded thrust magnitude [N]
        gimbal_pitch: Commanded gimbal pitch angle [rad]
        gimbal_yaw: Commanded gimbal yaw angle [rad]
        target_attitude: Target body attitude quaternion (optional)
        phase: Current guidance phase
    """
    thrust: float
    gimbal_pitch: float = 0.0
    gimbal_yaw: float = 0.0
    target_attitude: NDArray[np.float64] | None = None
    phase: int = 0


# =============================================================================
# Gravity Turn Guidance
# =============================================================================


@dataclass
class GravityTurnGuidance:
    """Gravity turn ascent guidance.

    Computes thrust and attitude commands for an efficient ascent
    trajectory from launch to target altitude.

    The gravity turn minimizes losses by:
    1. Flying through the atmosphere at near-zero angle of attack
    2. Using gravity to naturally pitch the vehicle over
    3. Achieving orbital velocity along the velocity vector

    Attributes:
        target_altitude: Target insertion altitude [m]
        max_thrust: Maximum thrust at full throttle [N]
        vertical_rise_time: Time to fly vertical before pitch kick [s]
        pitch_kick_duration: Duration of pitch kick maneuver [s]
        pitch_kick_angle: Pitch kick magnitude [rad]
        throttle: Throttle setting during powered flight (0-1)
    """
    target_altitude: float = 200e3
    max_thrust: float = 1e6
    vertical_rise_time: float = 10.0
    pitch_kick_duration: float = 10.0
    pitch_kick_angle: float = np.radians(5.0)
    launch_azimuth: float = np.radians(90.0)  # Azimuth from North [rad]
    throttle: float = 1.0

    # Internal state
    _phase: int = 1  # 1=vertical, 2=kick, 3=gravity turn, 4=insertion

    def compute(self, state: State) -> GuidanceCommand:
        """Compute guidance command for current state.

        Args:
            state: Current vehicle state

        Returns:
            GuidanceCommand with thrust and attitude commands
        """
        t = state.time
        alt = self._get_altitude(state)

        # Phase transitions
        if t < self.vertical_rise_time:
            self._phase = 1
        elif t < self.vertical_rise_time + self.pitch_kick_duration:
            self._phase = 2
        elif alt < self.target_altitude:
            self._phase = 3
        else:
            self._phase = 4

        # Compute target direction
        target_dir = self._compute_target_direction(state)

        # Compute gimbal commands to track target
        gimbal_pitch, gimbal_yaw = self._compute_gimbal(state, target_dir)

        # Throttle
        if self._phase == 4:
            thrust = 0.0  # Engine cutoff
        else:
            thrust = self.max_thrust * self.throttle

        return GuidanceCommand(
            thrust=thrust,
            gimbal_pitch=gimbal_pitch,
            gimbal_yaw=gimbal_yaw,
            target_attitude=self._direction_to_quaternion(state, target_dir),
            phase=self._phase,
        )

    def _get_altitude(self, state: State) -> float:
        """Get altitude from state."""
        if state.flat_earth:
            return state.position[2]
        return np.linalg.norm(state.position) - R_EARTH_EQ

    def _compute_target_direction(self, state: State) -> NDArray[np.float64]:
        """Compute target thrust direction in inertial frame."""
        pos = state.position
        vel = state.velocity

        # Radial (up) direction
        if state.flat_earth:
            r_hat = np.array([0.0, 0.0, 1.0])
        else:
            r_hat = pos / np.linalg.norm(pos)

        if self._phase == 1:
            # Vertical rise - point straight up
            return r_hat

        elif self._phase == 2:
            # Pitch kick - tilt slightly from vertical
            # Compute kick progress (0 to 1)
            kick_start = self.vertical_rise_time
            kick_progress = (state.time - kick_start) / self.pitch_kick_duration
            kick_progress = np.clip(kick_progress, 0.0, 1.0)

            # Kick direction: Based on launch azimuth
            # North is roughly Z cross R in ECI? No, Z is North Pole.
            # East = (0,0,1) x r_hat
            # North = r_hat x East
            z_axis = np.array([0.0, 0.0, 1.0])
            
            # Local East vector
            east = np.cross(z_axis, r_hat)
            east_mag = np.linalg.norm(east)
            if east_mag < 1e-6:
                # At pole - arbitrary
                east = np.array([1.0, 0.0, 0.0])
            else:
                east = east / east_mag
                
            # Local North vector
            north = np.cross(r_hat, east)
            
            # Target kick direction based on azimuth (measured from North)
            # kick_dir = North * cos(az) + East * sin(az)
            c_az = np.cos(self.launch_azimuth)
            s_az = np.sin(self.launch_azimuth)
            
            kick_dir = north * c_az + east * s_az
            kick_dir = kick_dir / np.linalg.norm(kick_dir)

            # Blend from vertical to kicked
            kick_angle = self.pitch_kick_angle * kick_progress
            target = np.cos(kick_angle) * r_hat + np.sin(kick_angle) * kick_dir
            return target / np.linalg.norm(target)

        else:  # Phase 3 or 4 - Gravity turn (follow relative velocity)
            # Use velocity relative to rotating Earth (ECEF velocity)
            # This ensures we fly a constant heading relative to the ground (approx)
            # and accounts for Earth rotation naturally.
            omega = np.array([0.0, 0.0, OMEGA_EARTH])
            v_rel = vel - np.cross(omega, pos)
            
            speed_rel = np.linalg.norm(v_rel)
            if speed_rel > 10.0:
                return v_rel / speed_rel
            else:
                # Low velocity - maintain current direction
                return r_hat

    def _compute_gimbal(
        self,
        state: State,
        target_dir: NDArray[np.float64],
    ) -> tuple[float, float]:
        """Compute gimbal angles to achieve target direction.

        This is a simple proportional controller. Real flight software
        would use a more sophisticated control law.
        """
        # Get current body X axis (thrust direction) in inertial frame
        dcm = state.dcm_body_to_inertial
        body_x = dcm[:, 0]

        # Error between current and target (in body frame)
        error_inertial = target_dir - body_x
        error_body = state.dcm_inertial_to_body @ error_inertial

        # Gimbal commands (proportional to error)
        # Gimbal pitch rotates about body Y
        # Gimbal yaw rotates about body Z
        kp = 2.0  # Proportional gain

        gimbal_pitch = kp * error_body[2]  # Error in Z -> pitch correction
        gimbal_yaw = -kp * error_body[1]   # Error in Y -> yaw correction

        # Limit gimbal angles
        max_gimbal = np.radians(6.0)
        gimbal_pitch = np.clip(gimbal_pitch, -max_gimbal, max_gimbal)
        gimbal_yaw = np.clip(gimbal_yaw, -max_gimbal, max_gimbal)

        return gimbal_pitch, gimbal_yaw

    def _direction_to_quaternion(
        self,
        state: State,
        target_dir: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert target direction to attitude quaternion.

        Body X points along target direction.
        Body Z points toward Earth (nadir).
        """
        from rocket.dynamics.state import dcm_to_quaternion

        # Body X = target direction
        body_x = target_dir / np.linalg.norm(target_dir)

        # Body Z = nadir (toward Earth center)
        if state.flat_earth:
            nadir = np.array([0.0, 0.0, -1.0])
        else:
            nadir = -state.position / np.linalg.norm(state.position)

        # Make orthogonal
        body_z = nadir - np.dot(nadir, body_x) * body_x
        z_mag = np.linalg.norm(body_z)
        if z_mag > 0.01:
            body_z = body_z / z_mag
        else:
            # Degenerate case - pick arbitrary perpendicular
            body_z = np.array([0.0, 0.0, 1.0])
            body_z = body_z - np.dot(body_z, body_x) * body_x
            body_z = body_z / np.linalg.norm(body_z)

        # Body Y completes right-hand system
        body_y = np.cross(body_z, body_x)

        # DCM from body to inertial (columns are body axes in inertial)
        dcm = np.column_stack([body_x, body_y, body_z])

        # Convert to quaternion (need inertial-to-body)
        return dcm_to_quaternion(dcm.T)

    def is_complete(self, state: State) -> bool:
        """Check if guidance objectives are achieved."""
        alt = self._get_altitude(state)
        return alt >= self.target_altitude

    @property
    def phase(self) -> int:
        """Current guidance phase."""
        return self._phase

    @property
    def phase_name(self) -> str:
        """Human-readable phase name."""
        names = {
            1: "Vertical Rise",
            2: "Pitch Kick",
            3: "Gravity Turn",
            4: "Insertion",
        }
        return names.get(self._phase, "Unknown")

