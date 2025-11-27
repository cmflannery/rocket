"""Gravity turn guidance for rocket ascent.

Implements simple gravity turn ascent profiles for rocket vehicles.
The gravity turn is an efficient trajectory that minimizes steering losses
by allowing gravity to naturally pitch the vehicle over.

Phases:
1. Vertical rise: Vehicle climbs vertically until pitch-over altitude
2. Pitch-over: Small pitch kick initiates the gravity turn
3. Gravity turn: Vehicle follows velocity vector (zero angle of attack)
4. Coast/Insertion: Engine cutoff for orbital insertion

Example:
    >>> from rocket.gnc.guidance import GravityTurnGuidance
    >>>
    >>> guidance = GravityTurnGuidance(
    ...     pitch_kick=np.radians(2),  # 2 degree initial kick
    ...     pitch_kick_time=10,  # At T+10s
    ... )
    >>>
    >>> # Get commanded pitch at current state
    >>> pitch_cmd = guidance.pitch_command(state)
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

from rocket.dynamics.state import State

# =============================================================================
# Pitch Program
# =============================================================================


@beartype
@dataclass
class PitchProgram:
    """Time-based pitch program for ascent guidance.

    Defines pitch angle vs time as piecewise linear segments.

    Example:
        >>> # Start vertical, pitch to 45 deg by T+60s, then to 0 by T+120s
        >>> program = PitchProgram(
        ...     times=[0, 10, 60, 120],
        ...     pitches=[90, 88, 45, 0],  # degrees
        ... )
        >>> pitch = program.at_time(30)  # Interpolated pitch
    """
    times: list[float]  # Time points [s]
    pitches: list[float]  # Pitch angles [degrees]

    def __post_init__(self) -> None:
        """Validate inputs."""
        if len(self.times) != len(self.pitches):
            raise ValueError("times and pitches must have same length")
        if len(self.times) < 2:
            raise ValueError("Need at least 2 points")

    @beartype
    def at_time(self, t: float) -> float:
        """Get commanded pitch at time t.

        Args:
            t: Time [s]

        Returns:
            Pitch angle [rad]
        """
        pitch_deg = np.interp(t, self.times, self.pitches)
        return np.radians(pitch_deg)

    @classmethod
    def gravity_turn(
        cls,
        vertical_time: float = 10.0,
        pitch_kick: float = 2.0,  # degrees
        target_pitch: float = 0.0,  # degrees (horizontal)
        turn_time: float = 120.0,  # time to reach target
    ) -> "PitchProgram":
        """Create a standard gravity turn program.

        Args:
            vertical_time: Time to fly vertical before pitch-over [s]
            pitch_kick: Initial pitch kick magnitude [degrees]
            target_pitch: Final pitch angle [degrees]
            turn_time: Time to complete turn [s]
        """
        return cls(
            times=[0, vertical_time, vertical_time + 1, turn_time],
            pitches=[90, 90, 90 - pitch_kick, target_pitch],
        )


# =============================================================================
# Gravity Turn Guidance
# =============================================================================


@beartype
@dataclass
class GravityTurnGuidance:
    """Gravity turn ascent guidance.

    Implements a gravity turn trajectory with:
    1. Vertical rise phase
    2. Pitch-over kick
    3. Zero-alpha (velocity-following) phase

    The guidance outputs commanded attitude for the control system.

    Attributes:
        pitch_kick: Initial pitch-over angle [rad]
        pitch_kick_time: Time to initiate pitch-over [s]
        target_altitude: Target altitude for insertion [m]
        heading: Launch heading (0=north, 90=east) [rad]
        mode: Guidance mode ('pitch_program' or 'velocity_follow')
    """
    pitch_kick: float = np.radians(2.0)  # 2 degrees
    pitch_kick_time: float = 10.0  # [s]
    pitch_kick_duration: float = 5.0  # [s]
    target_altitude: float = 200000  # 200 km
    heading: float = np.radians(90)  # Due east
    mode: Literal["pitch_program", "velocity_follow"] = "velocity_follow"

    # Optional pitch program
    pitch_program: PitchProgram | None = None

    @beartype
    def pitch_command(self, state: State) -> float:
        """Get commanded pitch angle.

        For a gravity turn, the commanded pitch follows the flight path angle
        (velocity vector direction). This maintains zero angle of attack,
        letting gravity naturally pitch the vehicle over.

        Args:
            state: Current vehicle state

        Returns:
            Commanded pitch angle [rad] (0 = horizontal, pi/2 = vertical)
        """
        if self.mode == "pitch_program" and self.pitch_program:
            return self.pitch_program.at_time(state.time)

        # Velocity-following mode
        t = state.time

        # Phase 1: Vertical rise - maintain 90° pitch
        if t < self.pitch_kick_time:
            return np.pi / 2

        # Get current flight path angle from velocity
        v = state.velocity
        v_horiz = np.sqrt(v[0]**2 + v[1]**2)
        v_vert = v[2]  # Positive up in flat-earth frame
        speed = np.sqrt(v_horiz**2 + v_vert**2)

        # If velocity is too small, maintain vertical
        if speed < 1.0:
            return np.pi / 2

        # Flight path angle (angle of velocity vector from horizontal)
        flight_path_angle = np.arctan2(v_vert, v_horiz)

        # Phase 2: Pitch kick - command pitch slightly below FPA to initiate turn
        if t < self.pitch_kick_time + self.pitch_kick_duration:
            frac = (t - self.pitch_kick_time) / self.pitch_kick_duration
            # Offset from FPA by kick angle (times completion fraction)
            pitch_cmd = flight_path_angle - self.pitch_kick * frac
            return float(np.clip(pitch_cmd, 0.0, np.pi / 2))

        # Phase 3: True gravity turn - follow velocity vector exactly
        # This is the "zero-lift" gravity turn where we maintain zero AoA
        return float(np.clip(flight_path_angle, 0.0, np.pi / 2))

    @beartype
    def yaw_command(self, state: State) -> float:
        """Get commanded yaw angle.

        For a gravity turn, yaw follows the velocity heading.
        During near-vertical flight, returns 0 (yaw is undefined when vertical).

        Args:
            state: Current vehicle state

        Returns:
            Commanded yaw angle [rad]
        """
        v = state.velocity
        v_horiz = np.sqrt(v[0]**2 + v[1]**2)

        # During near-vertical flight, yaw is undefined/meaningless
        # Return 0 to avoid control fighting
        if v_horiz < 10.0:  # Less than 10 m/s horizontal
            return 0.0

        # Follow velocity heading once horizontal velocity is significant
        return np.arctan2(v[1], v[0])

    @beartype
    def attitude_command(self, state: State) -> NDArray[np.float64]:
        """Get full attitude command.

        Args:
            state: Current vehicle state

        Returns:
            Commanded attitude [roll, pitch, yaw] [rad]
        """
        pitch = self.pitch_command(state)
        yaw = self.yaw_command(state)
        roll = 0.0  # No roll during gravity turn

        return np.array([roll, pitch, yaw])

    @beartype
    def throttle_command(self, state: State) -> float:
        """Get throttle command.

        Simple logic: full throttle until target altitude.

        Args:
            state: Current vehicle state

        Returns:
            Throttle setting (0 to 1)
        """
        # Check altitude
        alt = state.altitude

        if alt >= self.target_altitude:
            return 0.0  # Engine cutoff

        return 1.0  # Full throttle

    @beartype
    def is_complete(self, state: State) -> bool:
        """Check if guidance objectives are met.

        Args:
            state: Current vehicle state

        Returns:
            True if target altitude reached
        """
        return bool(state.altitude >= self.target_altitude)

