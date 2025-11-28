"""First stage return-to-launch-site (RTLS) guidance.

Implements SpaceX-style first stage recovery trajectory:
1. Coast after separation
2. Boostback burn - reverse direction to head back to launch site
3. Entry burn - slow down for atmospheric entry
4. Ballistic descent
5. Landing burn - final deceleration for soft touchdown

This is a simplified guidance law, not the optimal convex trajectory
that SpaceX uses, but demonstrates the key phases and targeting.

Reference:
    SpaceX Falcon 9 User's Guide
    "The Physics of Reusable Rockets" - Various analyses
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from rocket.dynamics.state import State
from rocket.environment.gravity import R_EARTH_EQ


class FirstStageLandingPhase(IntEnum):
    """Landing guidance phases."""
    COAST = 1          # Coasting after separation
    BOOSTBACK = 2      # Burn to reverse trajectory
    ENTRY_COAST = 3    # Coast before entry burn
    ENTRY_BURN = 4     # Slow down for atmosphere
    DESCENT = 5        # Ballistic fall through atmosphere
    LANDING_BURN = 6   # Final deceleration
    LANDED = 7         # On the ground


class LandingCommand(NamedTuple):
    """Output from landing guidance.

    Attributes:
        thrust: Commanded thrust magnitude [N]
        target_attitude: Target body attitude quaternion
        phase: Current guidance phase
        target_position: Current target position
    """
    thrust: float
    target_attitude: NDArray[np.float64] | None = None
    phase: int = 1
    target_position: NDArray[np.float64] | None = None


@dataclass
class FirstStageLandingGuidance:
    """First stage RTLS landing guidance.

    Simplified guidance for returning a first stage to the launch site.
    Uses polynomial guidance for the landing burn phase.

    Attributes:
        landing_site_eci: Target landing position in ECI [m]
        max_thrust: Maximum thrust [N]
        min_throttle: Minimum throttle (for throttle-down capability)
        dry_mass: Stage dry mass [kg] (for thrust-to-weight)
        boostback_dv: Target delta-V for boostback burn [m/s]
        entry_burn_start_alt: Altitude to start entry burn [m]
        entry_burn_target_speed: Target speed after entry burn [m/s]
        landing_burn_start_alt: Altitude to start landing burn [m]
    """
    landing_site_eci: NDArray[np.float64]
    max_thrust: float
    min_throttle: float = 0.4  # Falcon 9 can't throttle below ~40%
    dry_mass: float = 25000.0
    boostback_dv: float = 300.0  # ~300 m/s for RTLS
    entry_burn_start_alt: float = 70000.0  # 70 km
    entry_burn_target_speed: float = 500.0  # Target ~500 m/s after entry burn
    landing_burn_start_alt: float = 3000.0  # Start final burn at 3 km

    # Internal state
    _phase: FirstStageLandingPhase = field(default=FirstStageLandingPhase.COAST)
    _boostback_start_time: float = field(default=0.0)
    _separation_time: float = field(default=0.0)
    _boostback_complete: bool = field(default=False)
    _entry_burn_complete: bool = field(default=False)

    def initialize(self, state: State, separation_time: float):
        """Initialize guidance at stage separation.

        Args:
            state: State at separation
            separation_time: Time of stage separation
        """
        self._separation_time = separation_time
        self._phase = FirstStageLandingPhase.COAST
        self._boostback_complete = False
        self._entry_burn_complete = False
        self._boostback_start_time = 0.0

    def compute(self, state: State) -> LandingCommand:
        """Compute landing guidance command.

        Args:
            state: Current vehicle state

        Returns:
            LandingCommand with thrust and attitude
        """
        alt = self._get_altitude(state)
        speed = np.linalg.norm(state.velocity)

        # Phase transitions
        self._update_phase(state, alt, speed)

        # Compute target direction and thrust
        if self._phase == FirstStageLandingPhase.COAST:
            return self._coast_command(state)

        elif self._phase == FirstStageLandingPhase.BOOSTBACK:
            return self._boostback_command(state)

        elif self._phase == FirstStageLandingPhase.ENTRY_COAST:
            return self._coast_command(state)

        elif self._phase == FirstStageLandingPhase.ENTRY_BURN:
            return self._entry_burn_command(state)

        elif self._phase == FirstStageLandingPhase.DESCENT:
            return self._descent_command(state)

        elif self._phase == FirstStageLandingPhase.LANDING_BURN:
            return self._landing_burn_command(state, alt)

        else:  # LANDED
            return LandingCommand(thrust=0.0, phase=self._phase.value)

    def _update_phase(self, state: State, alt: float, speed: float):
        """Update guidance phase based on state."""
        time_since_sep = state.time - self._separation_time

        if self._phase == FirstStageLandingPhase.COAST:
            # Start boostback after brief coast (2-3 seconds)
            if time_since_sep > 3.0:
                self._phase = FirstStageLandingPhase.BOOSTBACK
                self._boostback_start_time = state.time

        elif self._phase == FirstStageLandingPhase.BOOSTBACK:
            # Check if boostback is complete (velocity pointing back toward site)
            if self._boostback_complete:
                self._phase = FirstStageLandingPhase.ENTRY_COAST

        elif self._phase == FirstStageLandingPhase.ENTRY_COAST:
            # Start entry burn at specified altitude
            if alt < self.entry_burn_start_alt and not self._entry_burn_complete:
                self._phase = FirstStageLandingPhase.ENTRY_BURN

        elif self._phase == FirstStageLandingPhase.ENTRY_BURN:
            if self._entry_burn_complete or speed < self.entry_burn_target_speed:
                self._entry_burn_complete = True
                self._phase = FirstStageLandingPhase.DESCENT

        elif self._phase == FirstStageLandingPhase.DESCENT:
            # Start landing burn at specified altitude
            if alt < self.landing_burn_start_alt:
                self._phase = FirstStageLandingPhase.LANDING_BURN

        elif self._phase == FirstStageLandingPhase.LANDING_BURN:
            if alt < 10.0 and speed < 2.0:  # Touchdown
                self._phase = FirstStageLandingPhase.LANDED

    def _coast_command(self, state: State) -> LandingCommand:
        """Command during coast phases - point retrograde."""
        # Point body +X retrograde (for next burn)
        vel = state.velocity
        speed = np.linalg.norm(vel)
        if speed > 10.0:
            retrograde = -vel / speed
        else:
            retrograde = np.array([0.0, 0.0, 1.0])

        attitude = self._direction_to_quaternion(state, retrograde)
        return LandingCommand(
            thrust=0.0,
            target_attitude=attitude,
            phase=self._phase.value,
            target_position=self.landing_site_eci,
        )

    def _boostback_command(self, state: State) -> LandingCommand:
        """Boostback burn - reverse horizontal trajectory to return to landing site.

        Real RTLS boostback burns are primarily HORIZONTAL - we want to:
        1. Kill the downrange velocity
        2. Add return velocity toward the landing site
        3. NOT add vertical velocity (that wastes fuel and time)
        """
        vel = state.velocity
        speed = np.linalg.norm(vel)

        # Get radial direction (up)
        r_hat = state.position / np.linalg.norm(state.position)

        # Decompose velocity into radial and horizontal components
        v_radial_mag = np.dot(vel, r_hat)
        v_horizontal = vel - v_radial_mag * r_hat
        v_horiz_mag = np.linalg.norm(v_horizontal)

        # Direction back toward landing site (horizontal only)
        to_site = self.landing_site_eci - state.position
        to_site_horiz = to_site - np.dot(to_site, r_hat) * r_hat
        to_site_horiz_mag = np.linalg.norm(to_site_horiz)

        if to_site_horiz_mag > 100:
            to_site_hat = to_site_horiz / to_site_horiz_mag
        else:
            # Very close to site horizontally
            to_site_hat = -v_horizontal / v_horiz_mag if v_horiz_mag > 10 else r_hat

        # Check if boostback should complete
        # Complete when: (1) moving toward site with sufficient speed, OR (2) time limit
        time_since_start = state.time - self._boostback_start_time if self._boostback_start_time > 0 else 0
        
        # Calculate velocity towards landing site
        v_towards_site = np.dot(v_horizontal, to_site_hat)
        
        # Target return speed (tunable parameter)
        # We need enough speed to cover the downrange distance while falling
        target_return_speed = 450.0 

        # Complete if we have enough velocity towards the site
        if v_towards_site > target_return_speed:
            self._boostback_complete = True
            return self._coast_command(state)

        # Complete after max burn time (don't burn forever)
        if time_since_start > 150:
            self._boostback_complete = True
            return self._coast_command(state)

        # Boostback thrust direction: Always towards the landing site (horizontally)
        # This simultaneously kills downrange velocity and builds return velocity
        target_dir = to_site_hat

        # Keep thrust purely horizontal (no vertical component)
        # This is key for efficient RTLS - we coast up/down, not thrust up
        target_dir = target_dir - np.dot(target_dir, r_hat) * r_hat
        target_dir_mag = np.linalg.norm(target_dir)
        if target_dir_mag > 0.1:
            target_dir = target_dir / target_dir_mag
        else:
            target_dir = to_site_hat

        attitude = self._direction_to_quaternion(state, target_dir)

        return LandingCommand(
            thrust=self.max_thrust,
            target_attitude=attitude,
            phase=self._phase.value,
            target_position=self.landing_site_eci,
        )

    def _entry_burn_command(self, state: State) -> LandingCommand:
        """Entry burn - slow down for atmosphere."""
        # Point retrograde
        vel = state.velocity
        speed = np.linalg.norm(vel)

        if speed < self.entry_burn_target_speed:
            self._entry_burn_complete = True
            return self._coast_command(state)

        retrograde = -vel / speed if speed > 10.0 else np.array([0.0, 0.0, 1.0])
        attitude = self._direction_to_quaternion(state, retrograde)

        return LandingCommand(
            thrust=self.max_thrust,
            target_attitude=attitude,
            phase=self._phase.value,
            target_position=self.landing_site_eci,
        )

    def _descent_command(self, state: State) -> LandingCommand:
        """Ballistic descent - prepare for landing burn."""
        # Point retrograde for drag deceleration (if we had drag)
        vel = state.velocity
        speed = np.linalg.norm(vel)
        if speed > 10.0:
            retrograde = -vel / speed
        else:
            # Point up (local vertical)
            retrograde = state.position / np.linalg.norm(state.position)

        attitude = self._direction_to_quaternion(state, retrograde)

        return LandingCommand(
            thrust=0.0,
            target_attitude=attitude,
            phase=self._phase.value,
            target_position=self.landing_site_eci,
        )

    def _landing_burn_command(self, state: State, alt: float) -> LandingCommand:
        """Landing burn - aggressive constant-deceleration landing."""
        vel = state.velocity
        speed = np.linalg.norm(vel)

        # Vertical velocity component (positive = away from Earth)
        r_hat = state.position / np.linalg.norm(state.position)
        v_vertical = np.dot(vel, r_hat)
        
        # Horizontal velocity
        v_horiz = vel - v_vertical * r_hat
        v_horiz_mag = np.linalg.norm(v_horiz)

        # Surface gravity
        g = 9.81

        # For landing, we need to kill all velocity
        # Use full thrust until we're nearly stopped
        if speed > 10.0:
            # High speed - full thrust retrograde
            # Bias slightly towards killing horizontal velocity if we are low
            if alt < 1000 and v_horiz_mag > 10:
                 # Mix retrograde with extra horizontal killing
                 target_dir = -vel - v_horiz * 1.0
                 target_dir = target_dir / np.linalg.norm(target_dir)
                 thrust = self.max_thrust
            else:
                 target_dir = -vel / speed
                 thrust = self.max_thrust
        elif alt > 50 and (abs(v_vertical) > 5.0 or speed > 20.0):
            # Medium altitude, still moving - moderate thrust
            # Required decel to stop at ground: a = v^2 / (2*h)
            # Use total speed to ensure we kill horizontal velocity too
            required_decel = (speed**2) / (2 * max(alt, 10)) * 1.5 + g
            required_thrust = state.mass * required_decel
            thrust = np.clip(required_thrust, self.max_thrust * self.min_throttle, self.max_thrust)
            
            # Direction: Retrograde
            target_dir = -vel / speed
            
        elif alt > 10:
            # Low altitude - controlled descent
            # Target 5 m/s descent rate
            target_descent = -5.0
            error = v_vertical - target_descent
            # Fix: Subtract error (negative error = too fast down => add thrust)
            thrust = state.mass * (g - error * 2.0)  # PD controller
            thrust = np.clip(thrust, self.max_thrust * self.min_throttle, self.max_thrust)
            
            # Kill horizontal velocity aggressively here
            if v_horiz_mag > 1.0:
                target_dir = -v_vertical*r_hat - v_horiz*2.0
                target_dir = target_dir / np.linalg.norm(target_dir)
            else:
                target_dir = r_hat # Up
                
        else:
            # Very low - hover thrust
            thrust = state.mass * g
            target_dir = r_hat

        attitude = self._direction_to_quaternion(state, target_dir)

        return LandingCommand(
            thrust=thrust,
            target_attitude=attitude,
            phase=self._phase.value,
            target_position=self.landing_site_eci,
        )

    def _get_altitude(self, state: State) -> float:
        """Get altitude above surface."""
        if state.flat_earth:
            return state.position[2]
        return np.linalg.norm(state.position) - R_EARTH_EQ

    def _direction_to_quaternion(
        self,
        state: State,
        target_dir: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert target direction to attitude quaternion."""
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
            body_z = np.array([0.0, 0.0, 1.0])
            body_z = body_z - np.dot(body_z, body_x) * body_x
            body_z = body_z / np.linalg.norm(body_z)

        body_y = np.cross(body_z, body_x)

        dcm = np.column_stack([body_x, body_y, body_z])
        return dcm_to_quaternion(dcm.T)

    @property
    def phase(self) -> FirstStageLandingPhase:
        """Current landing phase."""
        return self._phase

    @property
    def phase_name(self) -> str:
        """Human-readable phase name."""
        names = {
            FirstStageLandingPhase.COAST: "Coast",
            FirstStageLandingPhase.BOOSTBACK: "Boostback",
            FirstStageLandingPhase.ENTRY_COAST: "Entry Coast",
            FirstStageLandingPhase.ENTRY_BURN: "Entry Burn",
            FirstStageLandingPhase.DESCENT: "Descent",
            FirstStageLandingPhase.LANDING_BURN: "Landing Burn",
            FirstStageLandingPhase.LANDED: "Landed",
        }
        return names.get(self._phase, "Unknown")

    def is_complete(self, state: State) -> bool:
        """Check if landing is complete (on ground or crashed)."""
        alt = self._get_altitude(state)
        return alt < 5.0 or self._phase == FirstStageLandingPhase.LANDED

