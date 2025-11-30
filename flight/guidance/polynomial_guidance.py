"""Polynomial guidance for powered landing.

A simpler, faster alternative to G-FOLD that uses polynomial trajectory
shaping to compute thrust commands. This is similar to the Apollo lunar
module guidance.

The key idea: given current position/velocity and target position/velocity,
compute the required acceleration profile as a polynomial in time.
"""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from rocket.dynamics.state import State


@dataclass
class PolynomialGuidance:
    """Polynomial guidance for powered descent.
    
    Uses a cubic polynomial to shape the trajectory from current state
    to target state.
    
    Attributes:
        target_position: Landing site position [m]
        target_velocity: Target velocity at landing [m/s] (usually zero)
        max_thrust: Maximum thrust [N]
        min_thrust: Minimum thrust [N]
        dry_mass: Vehicle dry mass [kg]
    """
    target_position: NDArray[np.float64]
    target_velocity: NDArray[np.float64] = None
    max_thrust: float = 1e6
    min_thrust: float = 1e5
    dry_mass: float = 25000.0
    
    def __post_init__(self):
        if self.target_velocity is None:
            self.target_velocity = np.zeros(3)
        
        # Compute gravity vector
        r_target = np.linalg.norm(self.target_position)
        g_mag = 9.80665 * (6378137.0 / r_target) ** 2
        self._g_vec = -self.target_position / r_target * g_mag
        self._up = -self._g_vec / np.linalg.norm(self._g_vec)
    
    def get_command(self, state: State, t_go: float = None) -> dict:
        """Compute guidance command.
        
        Args:
            state: Current vehicle state
            t_go: Time to go until landing. If None, estimated automatically.
            
        Returns:
            dict with 'thrust', 'direction', 'phase', 't_go'
        """
        r = state.position
        v = state.velocity
        m = state.mass
        
        r_target = self.target_position
        v_target = self.target_velocity
        
        # Relative state
        dr = r - r_target  # Position error (want to drive to zero)
        dv = v - v_target  # Velocity error (want to drive to zero)
        
        # Altitude above target
        alt = np.dot(dr, self._up)
        
        # Check if landed
        if alt < 10.0 and np.linalg.norm(dv) < 5.0:
            return {"thrust": 0.0, "direction": self._up, "phase": "LANDED", "t_go": 0.0}
        
        # Check if crashed
        if alt < 0:
            return {"thrust": 0.0, "direction": self._up, "phase": "CRASHED", "t_go": 0.0}
        
        # Estimate time-to-go if not provided
        if t_go is None:
            t_go = self._estimate_tgo(dr, dv, m)
        
        t_go = max(t_go, 1.0)  # Minimum 1 second
        
        # Compute required acceleration using polynomial guidance
        # For a cubic polynomial trajectory:
        # r(t) = a0 + a1*t + a2*t^2 + a3*t^3
        # With boundary conditions:
        #   r(0) = r_current, r(t_go) = r_target
        #   v(0) = v_current, v(t_go) = v_target
        #
        # The required acceleration at t=0 is:
        # a_cmd = 6*(r_target - r_current)/t_go^2 - 4*v_current/t_go - 2*v_target/t_go
        
        a_cmd = (
            6.0 * (-dr) / (t_go ** 2)
            - 4.0 * dv / t_go
            - 2.0 * (v_target - v_target) / t_go  # This is zero for v_target=0
        )
        
        # Add gravity compensation
        a_cmd = a_cmd - self._g_vec
        
        # Compute thrust magnitude and direction
        a_mag = np.linalg.norm(a_cmd)
        
        if a_mag > 0.1:
            direction = a_cmd / a_mag
        else:
            direction = self._up
        
        # Convert acceleration to thrust
        thrust = a_mag * m
        
        # Clamp thrust to limits
        if thrust > self.max_thrust:
            thrust = self.max_thrust
        elif thrust < self.min_thrust:
            # If required thrust is below minimum, either coast or use minimum
            if a_mag < 0.5:  # Very low acceleration needed - coast
                thrust = 0.0
                phase = "COAST"
            else:
                thrust = self.min_thrust
                phase = "BURN"
        else:
            phase = "BURN"
        
        if thrust == 0.0:
            phase = "COAST"
        else:
            phase = "BURN"
        
        return {
            "thrust": thrust,
            "direction": direction,
            "phase": phase,
            "t_go": t_go,
        }
    
    def _estimate_tgo(self, dr: NDArray, dv: NDArray, mass: float) -> float:
        """Estimate time-to-go based on current state."""
        # Vertical component
        alt = np.dot(dr, self._up)
        v_vert = np.dot(dv, self._up)
        
        # Available acceleration
        g = np.linalg.norm(self._g_vec)
        a_max = self.max_thrust / mass
        a_net = a_max - g  # Net upward acceleration when thrusting up
        
        if a_net <= 0:
            # Can't hover - use simple kinematic estimate
            return max(1.0, alt / max(abs(v_vert), 1.0))
        
        # Estimate based on stopping distance
        # If descending (v_vert < 0), need to decelerate
        if v_vert < 0:
            # Time to stop: t = |v| / a
            t_stop = abs(v_vert) / a_net
            # Distance during deceleration: d = v^2 / (2a)
            d_stop = v_vert ** 2 / (2 * a_net)
            
            if alt > d_stop:
                # Have margin - coast then burn
                # Time to fall to burn altitude
                coast_alt = alt - d_stop * 1.5  # Start burn with 50% margin
                if coast_alt > 0:
                    # Kinematic: coast_alt = |v_vert| * t + 0.5 * g * t^2
                    # Solve for t
                    disc = v_vert**2 + 2 * g * coast_alt
                    if disc > 0:
                        t_coast = (-v_vert + np.sqrt(disc)) / g
                    else:
                        t_coast = 0
                    return t_coast + t_stop * 1.5
                else:
                    return t_stop * 1.5
            else:
                # Need to burn now
                return t_stop * 1.2
        else:
            # Ascending - will come back down
            t_up = v_vert / g
            alt_apex = alt + 0.5 * v_vert * t_up
            t_down = np.sqrt(2 * alt_apex / g)
            return t_up + t_down


