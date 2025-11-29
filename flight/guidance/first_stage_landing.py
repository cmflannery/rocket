"""First stage return-to-launch-site (RTLS) guidance.

Implements SpaceX-style first stage recovery using G-FOLD for landing.

The guidance has three main phases:
1. Boostback - Reverse trajectory to head back toward landing site
2. Coast/Entry - Ballistic coast, optional entry burn for speed reduction  
3. Landing - G-FOLD optimal powered descent

The key insight: G-FOLD handles the landing burn timing and profile automatically.
We don't need to hardcode "start landing burn at X altitude" - G-FOLD solves
for when to start burning based on current state and fuel optimality.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from flight.guidance.gfold import GFOLDGuidance
from rocket.dynamics.state import State
from rocket.environment.gravity import R_EARTH_EQ
from rocket.orbital import OMEGA_EARTH


class LandingPhase(IntEnum):
    """Landing guidance phases."""
    SEPARATION = 0     # Just separated, brief coast
    BOOSTBACK = 1      # Burning to reverse trajectory
    COAST = 2          # Ballistic coast toward landing site
    ENTRY_BURN = 3     # Optional speed reduction burn
    DESCENT = 4        # G-FOLD controlled descent (may include coast)
    LANDING_BURN = 5   # G-FOLD active thrusting
    LANDED = 6         # On the ground


class LandingCommand(NamedTuple):
    """Output from landing guidance."""
    thrust: float
    target_attitude: NDArray[np.float64] | None = None
    phase: int = 0
    target_position: NDArray[np.float64] | None = None


@dataclass
class FirstStageLandingGuidance:
    """First stage RTLS landing guidance using G-FOLD.
    
    Attributes:
        initial_landing_site_eci: Initial (t=0) target landing position in ECI [m]
        max_thrust: Maximum thrust [N] (all engines for boostback)
        min_thrust: Minimum thrust [N] (single engine throttled)
        dry_mass: Stage dry mass [kg]
        isp: Engine specific impulse [s]
    """
    initial_landing_site_eci: NDArray[np.float64]
    max_thrust: float
    min_thrust: float
    dry_mass: float
    isp: float = 282.0

    # Internal state
    _phase: LandingPhase = field(default=LandingPhase.SEPARATION)
    _separation_time: float = field(default=0.0)
    _boostback_start_time: float = field(default=0.0)
    _gfold: GFOLDGuidance | None = field(default=None, repr=False)
    _entry_burn_complete: bool = field(default=False)
    _last_gfold_check: float = field(default=0.0)
    _gfold_check_interval: float = field(default=5.0)  # Check every 5 seconds
    _current_target_eci: NDArray[np.float64] | None = field(default=None)

    def __post_init__(self):
        # Initialize G-FOLD for landing
        # We will update target position dynamically in compute()
        self._current_target_eci = self.initial_landing_site_eci.copy()
        self._gfold = GFOLDGuidance(
            target_position=self._current_target_eci,
            max_thrust=self.max_thrust,
            min_thrust=self.min_thrust,
            dry_mass=self.dry_mass,
            isp=self.isp,
            glideslope_tan=3.0,  # Allow fairly shallow approaches for RTLS
            max_tilt=np.radians(70.0),  # Allow significant tilt for trajectory correction
        )

    def initialize(self, state: State, separation_time: float):
        """Initialize guidance at stage separation."""
        self._separation_time = separation_time
        self._phase = LandingPhase.SEPARATION
        self._update_target_eci(state.time)

    @property
    def landing_site_eci(self) -> NDArray[np.float64]:
        """Get current landing site ECI position."""
        if self._current_target_eci is None:
            return self.initial_landing_site_eci
        return self._current_target_eci

    def _update_target_eci(self, time: float):
        """Update landing target based on Earth rotation."""
        # Rotate initial position by omega * time around Z axis
        theta = OMEGA_EARTH * time
        c, s = np.cos(theta), np.sin(theta)
        
        # Initial position
        x0, y0, z0 = self.initial_landing_site_eci
        
        # Rotated position
        x = x0 * c - y0 * s
        y = x0 * s + y0 * c
        z = z0
        
        self._current_target_eci = np.array([x, y, z])
        
        # Update G-FOLD target
        if self._gfold is not None:
            self._gfold.update_target(self._current_target_eci)

    @property
    def phase(self) -> LandingPhase:
        return self._phase

    @property
    def phase_name(self) -> str:
        return self._phase.name

    def compute(self, state: State) -> LandingCommand:
        """Compute guidance command for current state."""
        # Update target position for Earth rotation
        self._update_target_eci(state.time)
        
        alt = self._get_altitude(state)
        speed = np.linalg.norm(state.velocity)

        # Update phase based on current state
        self._update_phase(state, alt, speed)

        # Compute command based on phase
        if self._phase == LandingPhase.SEPARATION:
            return self._coast_command(state)
        elif self._phase == LandingPhase.BOOSTBACK:
            return self._boostback_command(state)
        elif self._phase == LandingPhase.COAST:
            return self._coast_command(state)
        elif self._phase == LandingPhase.ENTRY_BURN:
            return self._entry_burn_command(state)
        elif self._phase in (LandingPhase.DESCENT, LandingPhase.LANDING_BURN):
            return self._gfold_command(state)
        else:
            return LandingCommand(thrust=0.0, phase=self._phase.value)

    def _get_altitude(self, state: State) -> float:
        """Get altitude above Earth surface."""
        return np.linalg.norm(state.position) - R_EARTH_EQ

    def _get_distance_to_site(self, state: State) -> tuple[float, float]:
        """Get horizontal and vertical distance to landing site."""
        target = self.landing_site_eci
        r_hat = target / np.linalg.norm(target)
        rel_pos = state.position - target
        v_dist = np.dot(rel_pos, r_hat)  # Vertical (above site)
        h_vec = rel_pos - v_dist * r_hat
        h_dist = np.linalg.norm(h_vec)   # Horizontal
        return h_dist, v_dist

    def _get_velocity_toward_site(self, state: State) -> float:
        """Get velocity component toward landing site (positive = toward)."""
        target = self.landing_site_eci
        r_hat = target / np.linalg.norm(target)
        to_site = target - state.position
        to_site_horiz = to_site - np.dot(to_site, r_hat) * r_hat
        to_site_mag = np.linalg.norm(to_site_horiz)
        if to_site_mag < 100:
            return 0.0
        to_site_hat = to_site_horiz / to_site_mag

        # Horizontal velocity
        v_radial = np.dot(state.velocity, r_hat)
        v_horiz = state.velocity - v_radial * r_hat

        return np.dot(v_horiz, to_site_hat)

    def _update_phase(self, state: State, alt: float, speed: float):
        """Update guidance phase based on state."""
        time_since_sep = state.time - self._separation_time
        h_dist, v_dist = self._get_distance_to_site(state)
        v_toward = self._get_velocity_toward_site(state)

        if self._phase == LandingPhase.SEPARATION:
            # Brief coast after separation (3 seconds)
            if time_since_sep > 3.0:
                self._phase = LandingPhase.BOOSTBACK
                self._boostback_start_time = state.time

        elif self._phase == LandingPhase.BOOSTBACK:
            # Boostback complete when:
            # 1. Moving toward site fast enough to reach it, OR
            # 2. Burned for too long (safety limit)

            # Estimate required return speed based on trajectory
            # Simple model: need to cover horizontal distance while falling
            g = 9.81
            t_fall_est = np.sqrt(2 * max(v_dist, 1000) / g) * 1.5  # Time to fall with margin
            required_speed = h_dist / max(t_fall_est, 10.0)

            # Add margin
            target_speed = required_speed * 1.2
            target_speed = np.clip(target_speed, 400.0, 1500.0)

            time_since_boostback = state.time - self._boostback_start_time

            if v_toward >= target_speed or time_since_boostback > 200.0:
                self._phase = LandingPhase.COAST

        elif self._phase == LandingPhase.COAST:
            # Transition to entry burn or directly to G-FOLD descent
            # Entry burn is optional - used to reduce speed before atmosphere

            # Use simple heuristic for landing burn timing instead of solving G-FOLD every step
            # Start landing burn when we need to decelerate to land safely
            # Rough estimate: need to kill vertical velocity before hitting ground
            # v^2 = 2 * a * h => h = v^2 / (2*a)
            # With a = (T/m - g), we can estimate required altitude to start burn
            
            v_radial = np.dot(state.velocity, state.position / np.linalg.norm(state.position))
            g = 9.81
            a_max = self.max_thrust / state.mass - g  # Net deceleration
            
            if a_max > 0:
                # Height needed to stop from current vertical speed
                stopping_height = (v_radial ** 2) / (2 * a_max) if v_radial < 0 else 0
                # Add margin
                trigger_alt = stopping_height * 2.0 + 5000  # 2x margin + 5km buffer
                
                if alt < trigger_alt and alt < 50000:  # Allow starting G-FOLD higher
                    self._phase = LandingPhase.DESCENT
            
            # Entry burn for high-speed descent
            if alt < 70000 and speed > 1500 and not self._entry_burn_complete:
                self._phase = LandingPhase.ENTRY_BURN

        elif self._phase == LandingPhase.ENTRY_BURN:
            # Entry burn complete when speed reduced
            # Continue burning until 400 m/s, or if we get too low (safety)
            # But allow burning lower if we are still going fast
            if speed < 400 or alt < 15000:
                self._entry_burn_complete = True
                self._phase = LandingPhase.COAST

        elif self._phase == LandingPhase.DESCENT:
            # Check for landing - don't call get_command here, just check altitude
            if alt < 10.0 and speed < 10.0:
                self._phase = LandingPhase.LANDED

        elif self._phase == LandingPhase.LANDING_BURN:
            # Check for landing
            if alt < 10.0 and speed < 10.0:
                self._phase = LandingPhase.LANDED

    def _direction_to_quaternion(self, state: State, direction: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert thrust direction to attitude quaternion."""
        # Body +X should point in thrust direction
        x_body = direction / np.linalg.norm(direction)

        # Choose a reasonable up direction
        r_hat = state.position / np.linalg.norm(state.position)

        # Y perpendicular to X and radial
        y_body = np.cross(r_hat, x_body)
        y_norm = np.linalg.norm(y_body)
        if y_norm < 0.1:
            # X is nearly radial, use velocity for reference
            v_hat = state.velocity / max(np.linalg.norm(state.velocity), 1.0)
            y_body = np.cross(v_hat, x_body)
            y_norm = np.linalg.norm(y_body)

        if y_norm > 0.01:
            y_body = y_body / y_norm
        else:
            y_body = np.array([0.0, 1.0, 0.0])

        z_body = np.cross(x_body, y_body)

        # Build rotation matrix and convert to quaternion
        # dcm columns are body axes in inertial frame -> dcm is Body to Inertial
        dcm = np.column_stack([x_body, y_body, z_body])
        # State expects quaternion for Inertial to Body rotation
        # So we need to convert dcm.T (Inertial to Body) to quaternion
        return self._dcm_to_quaternion(dcm.T)

    def _dcm_to_quaternion(self, dcm: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert DCM to quaternion [w, x, y, z]."""
        trace = np.trace(dcm)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (dcm[2, 1] - dcm[1, 2]) * s
            y = (dcm[0, 2] - dcm[2, 0]) * s
            z = (dcm[1, 0] - dcm[0, 1]) * s
        elif dcm[0, 0] > dcm[1, 1] and dcm[0, 0] > dcm[2, 2]:
            s = 2.0 * np.sqrt(1.0 + dcm[0, 0] - dcm[1, 1] - dcm[2, 2])
            w = (dcm[2, 1] - dcm[1, 2]) / s
            x = 0.25 * s
            y = (dcm[0, 1] + dcm[1, 0]) / s
            z = (dcm[0, 2] + dcm[2, 0]) / s
        elif dcm[1, 1] > dcm[2, 2]:
            s = 2.0 * np.sqrt(1.0 + dcm[1, 1] - dcm[0, 0] - dcm[2, 2])
            w = (dcm[0, 2] - dcm[2, 0]) / s
            x = (dcm[0, 1] + dcm[1, 0]) / s
            y = 0.25 * s
            z = (dcm[1, 2] + dcm[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + dcm[2, 2] - dcm[0, 0] - dcm[1, 1])
            w = (dcm[1, 0] - dcm[0, 1]) / s
            x = (dcm[0, 2] + dcm[2, 0]) / s
            y = (dcm[1, 2] + dcm[2, 1]) / s
            z = 0.25 * s

        q = np.array([w, x, y, z])
        return q / np.linalg.norm(q)

    def _coast_command(self, state: State) -> LandingCommand:
        """Coast phase - point retrograde for next burn."""
        vel = state.velocity
        speed = np.linalg.norm(vel)
        if speed > 10.0:
            direction = -vel / speed
        else:
            direction = state.position / np.linalg.norm(state.position)

        attitude = self._direction_to_quaternion(state, direction)
        return LandingCommand(
            thrust=0.0,
            target_attitude=attitude,
            phase=self._phase.value,
            target_position=self.landing_site_eci,
        )

    def _boostback_command(self, state: State) -> LandingCommand:
        """Boostback burn - thrust toward landing site horizontally."""
        # Get radial direction
        r_hat = state.position / np.linalg.norm(state.position)

        # Direction to landing site (horizontal only)
        to_site = self.landing_site_eci - state.position
        to_site_horiz = to_site - np.dot(to_site, r_hat) * r_hat
        to_site_mag = np.linalg.norm(to_site_horiz)

        if to_site_mag > 100:
            thrust_dir = to_site_horiz / to_site_mag
        else:
            # Very close - just point up
            thrust_dir = r_hat

        attitude = self._direction_to_quaternion(state, thrust_dir)
        return LandingCommand(
            thrust=self.max_thrust,
            target_attitude=attitude,
            phase=self._phase.value,
            target_position=self.landing_site_eci,
        )

    def _entry_burn_command(self, state: State) -> LandingCommand:
        """Entry burn - retrograde to reduce speed."""
        vel = state.velocity
        speed = np.linalg.norm(vel)

        if speed > 10.0:
            direction = -vel / speed
        else:
            direction = state.position / np.linalg.norm(state.position)

        attitude = self._direction_to_quaternion(state, direction)
        return LandingCommand(
            thrust=self.max_thrust,
            target_attitude=attitude,
            phase=self._phase.value,
            target_position=self.landing_site_eci,
        )

    def _gfold_command(self, state: State) -> LandingCommand:
        """G-FOLD controlled descent/landing."""
        cmd = self._gfold.get_command(state)

        attitude = self._direction_to_quaternion(state, cmd["direction"])

        # Update phase based on G-FOLD output
        if cmd["phase"] == "BURN":
            phase = LandingPhase.LANDING_BURN.value
        elif cmd["phase"] == "LANDED":
            phase = LandingPhase.LANDED.value
        else:
            phase = LandingPhase.DESCENT.value

        return LandingCommand(
            thrust=cmd["thrust"],
            target_attitude=attitude,
            phase=phase,
            target_position=self.landing_site_eci,
        )

    def is_complete(self, state: State) -> bool:
        """Check if landing is complete."""
        alt = self._get_altitude(state)
        speed = np.linalg.norm(state.velocity)
        return alt < 10.0 and speed < 10.0
