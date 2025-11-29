"""Orbital insertion guidance for second stage.

This guidance targets a specific circular orbit by computing
the required velocity at apogee and timing the circularization burn.

For a circular orbit at altitude h:
    v_circular = sqrt(mu / (R_earth + h))

The guidance:
1. Follows gravity turn until reaching target apogee
2. Coasts to apogee
3. Executes circularization burn prograde


This is flight software - designed to run on the vehicle.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from rocket.dynamics.state import State
from rocket.environment.gravity import MU_EARTH, R_EARTH_EQ


class OrbitalInsertionPhase(IntEnum):
    """Orbital insertion guidance phases."""
    VERTICAL_RISE = 1
    PITCH_KICK = 2
    GRAVITY_TURN = 3
    COAST_TO_APOGEE = 4
    CIRCULARIZE = 5
    ORBIT = 6


class InsertionCommand(NamedTuple):
    """Output from orbital insertion guidance.

    Attributes:
        thrust: Commanded thrust magnitude [N]
        target_attitude: Target body attitude quaternion
        phase: Current guidance phase
    """
    thrust: float
    target_attitude: NDArray[np.float64] | None = None
    phase: int = 1


@njit(cache=True)
def _compute_circular_velocity(altitude: float, mu: float = MU_EARTH, r_body: float = R_EARTH_EQ) -> float:
    """Compute circular orbital velocity at given altitude."""
    r = r_body + altitude
    return np.sqrt(mu / r)


@njit(cache=True)
def _compute_apogee_from_state(
    rx: float, ry: float, rz: float,
    vx: float, vy: float, vz: float,
    mu: float = MU_EARTH,
) -> float:
    """Compute apogee altitude from current state."""
    r = np.sqrt(rx*rx + ry*ry + rz*rz)
    v = np.sqrt(vx*vx + vy*vy + vz*vz)

    # Specific orbital energy
    epsilon = v*v/2 - mu/r

    if epsilon >= 0:
        return np.inf  # Escape trajectory

    # Semi-major axis
    a = -mu / (2 * epsilon)

    # Angular momentum magnitude
    hx = ry*vz - rz*vy
    hy = rz*vx - rx*vz
    hz = rx*vy - ry*vx
    h = np.sqrt(hx*hx + hy*hy + hz*hz)

    # Eccentricity
    e = np.sqrt(1 + 2*epsilon*h*h/(mu*mu))
    e = max(0.0, min(e, 0.9999))  # Clamp for numerical stability

    # Apogee distance
    r_apogee = a * (1 + e)

    return r_apogee - R_EARTH_EQ


@njit(cache=True)
def _compute_perigee_from_state(
    rx: float, ry: float, rz: float,
    vx: float, vy: float, vz: float,
    mu: float = MU_EARTH,
) -> float:
    """Compute perigee altitude from current state."""
    r = np.sqrt(rx*rx + ry*ry + rz*rz)
    v = np.sqrt(vx*vx + vy*vy + vz*vz)

    # Specific orbital energy
    epsilon = v*v/2 - mu/r

    if epsilon >= 0:
        return np.inf  # Escape trajectory (no perigee in usual sense)

    # Semi-major axis
    a = -mu / (2 * epsilon)

    # Angular momentum magnitude
    hx = ry*vz - rz*vy
    hy = rz*vx - rx*vz
    hz = rx*vy - ry*vx
    h = np.sqrt(hx*hx + hy*hy + hz*hz)

    # Eccentricity
    e = np.sqrt(1 + 2*epsilon*h*h/(mu*mu))
    e = max(0.0, min(e, 0.9999))  # Clamp for numerical stability

    # Perigee distance
    r_perigee = a * (1 - e)

    return r_perigee - R_EARTH_EQ


@dataclass
class OrbitalInsertionGuidance:
    """Second stage orbital insertion guidance.

    Guides the second stage from separation to circular orbit:
    1. Continue gravity turn to build apogee
    2. Coast to apogee
    3. Circularize with prograde burn

    Attributes:
        target_altitude: Target circular orbit altitude [m]
        max_thrust: Maximum thrust [N]
        vertical_rise_time: Time to fly vertical (from separation) [s]
        pitch_kick_duration: Duration of pitch kick [s]
        pitch_kick_angle: Pitch kick magnitude [rad]
        apogee_tolerance: How close to apogee before circularization [m]
    """
    target_altitude: float = 300e3
    max_thrust: float = 1e6
    vertical_rise_time: float = 0.0  # Already past vertical rise at separation
    pitch_kick_duration: float = 0.0  # Already past pitch kick at separation
    pitch_kick_angle: float = 0.0
    apogee_tolerance: float = 5000.0  # 5 km from apogee

    # Internal state
    _phase: OrbitalInsertionPhase = field(default=OrbitalInsertionPhase.GRAVITY_TURN)
    _separation_time: float = field(default=0.0)
    _main_engine_cutoff: bool = field(default=False)
    _circularize_start_time: float = field(default=0.0)

    def initialize(self, state: State, separation_time: float):
        """Initialize guidance at stage separation.

        Args:
            state: State at separation
            separation_time: Time of stage separation
        """
        self._separation_time = separation_time
        self._phase = OrbitalInsertionPhase.GRAVITY_TURN
        self._main_engine_cutoff = False
        self._circularize_start_time = 0.0

    def compute(self, state: State) -> InsertionCommand:
        """Compute guidance command.

        Args:
            state: Current vehicle state

        Returns:
            InsertionCommand with thrust and attitude
        """
        alt = self._get_altitude(state)
        apogee = self._get_apogee(state)

        # Update phase
        self._update_phase(state, alt, apogee)

        # Compute target direction based on phase
        if self._phase in (
            OrbitalInsertionPhase.VERTICAL_RISE,
            OrbitalInsertionPhase.PITCH_KICK,
            OrbitalInsertionPhase.GRAVITY_TURN,
        ):
            return self._ascent_command(state, alt, apogee)

        elif self._phase == OrbitalInsertionPhase.COAST_TO_APOGEE:
            return self._coast_command(state)

        elif self._phase == OrbitalInsertionPhase.CIRCULARIZE:
            return self._circularize_command(state)

        else:  # ORBIT
            return InsertionCommand(thrust=0.0, phase=self._phase.value)

    def _update_phase(self, state: State, alt: float, apogee: float):
        """Update guidance phase."""
        if self._phase == OrbitalInsertionPhase.GRAVITY_TURN:
            # We want to reach a stable orbit where we can coast to apogee.
            #
            # The key insight: we can only coast to apogee if perigee is
            # above the atmosphere. Otherwise we'll re-enter before reaching apogee.
            #
            # MECO conditions:
            # 1. Apogee >= target altitude
            # 2. Perigee > 150km (above atmosphere, can complete orbit)
            # 3. Not about to escape

            perigee = self._get_perigee(state)
            speed = np.linalg.norm(state.velocity)
            r = np.linalg.norm(state.position)
            v_escape = np.sqrt(2 * MU_EARTH / r)

            # Check if we're about to escape - emergency cutoff
            if speed >= 0.95 * v_escape:
                self._main_engine_cutoff = True
                self._phase = OrbitalInsertionPhase.COAST_TO_APOGEE
                return

            # Normal MECO: apogee at target altitude
            # We cut off when apogee reaches target, then coast to apogee and circularize
            # Perigee will be raised during circularization burn
            if apogee >= self.target_altitude:
                self._main_engine_cutoff = True
                self._phase = OrbitalInsertionPhase.COAST_TO_APOGEE

        elif self._phase == OrbitalInsertionPhase.COAST_TO_APOGEE:
            # Check if we are near apogee and ready to circularize.
            #
            # The earlier logic was overly strict and could miss the apogee
            # window entirely, leaving the stage in a long "coast" on an
            # eccentric, Earth‑intersecting trajectory. Here we use a more
            # robust condition:
            #   - Apogee is at (or above) the target
            #   - Current altitude is within `apogee_tolerance` of apogee
            #   - Radial velocity is small (near the top of the orbit)
            r_hat = state.position / np.linalg.norm(state.position)
            v_radial = np.dot(state.velocity, r_hat)

            close_to_apogee = abs(apogee - alt) <= self.apogee_tolerance
            apogee_high_enough = apogee >= 0.9 * self.target_altitude
            near_turnaround = abs(v_radial) < 50.0

            if apogee_high_enough and close_to_apogee and near_turnaround:
                self._phase = OrbitalInsertionPhase.CIRCULARIZE
                self._circularize_start_time = state.time

        elif self._phase == OrbitalInsertionPhase.CIRCULARIZE:
            # Check if orbit is circular enough around the target altitude.
            #
            # We use:
            #   - Speed close to ideal circular speed at target altitude
            #   - Apogee close to target altitude
            #   - Altitude safely in space (avoid declaring ORBIT while sub‑surface)
            v_circular = _compute_circular_velocity(self.target_altitude)
            speed = np.linalg.norm(state.velocity)

            speed_close = abs(speed - v_circular) < 30.0  # tighter band
            apogee_close = abs(apogee - self.target_altitude) < 10_000.0  # 10 km
            altitude_ok = alt > 0.8 * self.target_altitude

            if speed_close and apogee_close and altitude_ok:
                self._phase = OrbitalInsertionPhase.ORBIT

    def _ascent_command(self, state: State, alt: float, apogee: float) -> InsertionCommand:
        """Gravity turn ascent - but never pitch below horizontal!

        This guidance follows the velocity vector BUT prevents the vehicle from
        pitching down toward Earth. If velocity is below local horizontal, we
        steer toward horizontal to keep building orbital energy.
        """
        pos = state.position
        vel = state.velocity
        r = np.linalg.norm(pos)
        speed = np.linalg.norm(vel)

        # Get local vertical (up)
        r_hat = pos / r

        # Compute local horizontal velocity component
        v_radial = np.dot(vel, r_hat)
        v_horizontal = vel - v_radial * r_hat
        v_horiz_mag = np.linalg.norm(v_horizontal)

        if speed < 50.0:
            # Low speed - point up
            target_dir = r_hat
        else:
            # Check if velocity is pointing down (negative radial component)
            # We want to stay at or above horizontal
            flight_path_angle = np.arcsin(np.clip(v_radial / speed, -1, 1))

            # Minimum flight path angle - don't pitch below this
            # Start with 5° above horizontal, relax as we gain altitude
            # CRITICAL FIX: Never allow negative FPA (falling) until orbit insertion
            min_fpa = np.radians(5.0) if alt < 150e3 else np.radians(0.0)
            
            # If we are falling (v_radial < 0), strictly pitch up
            if v_radial < -10.0:
                min_fpa = np.radians(10.0)

            if flight_path_angle < min_fpa:
                # Velocity is too low/falling - steer above velocity vector
                # Target direction is pitched up relative to horizon
                # We compute horizontal vector
                if v_horiz_mag > 1.0:
                    h_hat = v_horizontal / v_horiz_mag
                else:
                    # If moving vertically, horizontal is arbitrary (use X)
                    h_hat = np.array([1.0, 0.0, 0.0])
                    h_hat = h_hat - np.dot(h_hat, r_hat) * r_hat
                    h_hat = h_hat / np.linalg.norm(h_hat)
                    
                target_dir = h_hat * np.cos(min_fpa) + r_hat * np.sin(min_fpa)
            else:
                # Velocity is acceptable - follow it (gravity turn)
                target_dir = vel / speed

        attitude = self._direction_to_quaternion(state, target_dir)

        # Full thrust during ascent
        thrust = self.max_thrust

        return InsertionCommand(
            thrust=thrust,
            target_attitude=attitude,
            phase=self._phase.value,
        )

    def _coast_command(self, state: State) -> InsertionCommand:
        """Coast to apogee."""
        # Point prograde for circularization
        vel = state.velocity
        speed = np.linalg.norm(vel)
        target_dir = vel / speed if speed > 10.0 else state.position / np.linalg.norm(state.position)

        attitude = self._direction_to_quaternion(state, target_dir)

        return InsertionCommand(
            thrust=0.0,
            target_attitude=attitude,
            phase=self._phase.value,
        )

    def _circularize_command(self, state: State) -> InsertionCommand:
        """Circularization burn at apogee."""
        # Point prograde
        vel = state.velocity
        speed = np.linalg.norm(vel)

        if speed > 10.0:
            target_dir = vel / speed
        else:
            # Very low speed - use orbital prograde direction
            r_hat = state.position / np.linalg.norm(state.position)
            h = np.cross(state.position, state.velocity)
            h_hat = h / np.linalg.norm(h)
            target_dir = np.cross(h_hat, r_hat)
            target_dir = target_dir / np.linalg.norm(target_dir)

        attitude = self._direction_to_quaternion(state, target_dir)

        return InsertionCommand(
            thrust=self.max_thrust,
            target_attitude=attitude,
            phase=self._phase.value,
        )

    def _get_altitude(self, state: State) -> float:
        """Get current altitude."""
        if state.flat_earth:
            return state.position[2]
        return np.linalg.norm(state.position) - R_EARTH_EQ

    def _get_apogee(self, state: State) -> float:
        """Get current apogee altitude."""
        pos = state.position
        vel = state.velocity
        return _compute_apogee_from_state(
            pos[0], pos[1], pos[2],
            vel[0], vel[1], vel[2],
        )

    def _get_perigee(self, state: State) -> float:
        """Get current perigee altitude."""
        pos = state.position
        vel = state.velocity
        return _compute_perigee_from_state(
            pos[0], pos[1], pos[2],
            vel[0], vel[1], vel[2],
        )

    def _direction_to_quaternion(
        self,
        state: State,
        target_dir: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert target direction to attitude quaternion."""
        from rocket.dynamics.state import dcm_to_quaternion

        body_x = target_dir / np.linalg.norm(target_dir)

        if state.flat_earth:
            nadir = np.array([0.0, 0.0, -1.0])
        else:
            nadir = -state.position / np.linalg.norm(state.position)

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

    @property
    def phase(self) -> OrbitalInsertionPhase:
        """Current phase."""
        return self._phase

    @property
    def phase_name(self) -> str:
        """Human-readable phase name."""
        names = {
            OrbitalInsertionPhase.VERTICAL_RISE: "Vertical Rise",
            OrbitalInsertionPhase.PITCH_KICK: "Pitch Kick",
            OrbitalInsertionPhase.GRAVITY_TURN: "Gravity Turn",
            OrbitalInsertionPhase.COAST_TO_APOGEE: "Coast to Apo",
            OrbitalInsertionPhase.CIRCULARIZE: "Circularize",
            OrbitalInsertionPhase.ORBIT: "Orbit",
        }
        return names.get(self._phase, "Unknown")

    def is_complete(self, state: State) -> bool:
        """Check if orbital insertion is complete."""
        return self._phase == OrbitalInsertionPhase.ORBIT

