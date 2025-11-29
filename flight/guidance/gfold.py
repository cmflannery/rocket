"""G-FOLD (Fuel Optimal Large Divert) Powered Descent Guidance.

Implements the lossless convexification algorithm for fuel-optimal powered landing.

This is the actual algorithm used by SpaceX for Falcon 9 landings, based on:
    Acikmese, B., & Ploen, S. R. (2007). "Convex Programming Approach to 
    Powered Descent Guidance for Mars Landing"
    
    Blackmore, L., Acikmese, B., & Scharf, D. P. (2010). "Minimum-Landing-Error 
    Powered-Descent Guidance for Mars Landing Using Convex Optimization"

The key insight: by using a change of variables (log mass), the nonconvex 
rocket landing problem becomes convex and can be solved to global optimality.

Usage:
    The guidance runs continuously. At each timestep:
    1. Solve G-FOLD from current state to target
    2. If solution says "burn now", execute the thrust command
    3. If solution says "coast", coast and re-solve next step
"""

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from rocket.dynamics.state import State


@dataclass
class GFOLDGuidance:
    """G-FOLD powered descent guidance.
    
    Solves the fuel-optimal landing problem in real-time.
    
    Attributes:
        target_position: Landing site position in ECI [m]
        target_velocity: Target velocity at landing [m/s] (usually zero)
        max_thrust: Maximum thrust magnitude [N]
        min_thrust: Minimum thrust magnitude [N] (engine can't throttle to zero)
        dry_mass: Vehicle dry mass [kg]
        g_vec: Gravity vector [m/s^2] (computed from target position)
        glideslope_tan: Tangent of glide slope angle (0 = vertical only)
        max_tilt: Maximum thrust vector tilt from vertical [rad]
    """
    target_position: NDArray[np.float64]
    max_thrust: float
    min_thrust: float  # Falcon 9 Merlin can't throttle below ~40%
    dry_mass: float
    isp: float = 282.0  # Sea level Isp [s]
    glideslope_tan: float = 5.0  # Allow steep approaches for RTLS
    max_tilt: float = np.radians(80.0)  # Allow significant tilt
    n_nodes: int = 25  # Fewer nodes for faster solving

    # State
    _solution: dict | None = field(default=None, repr=False)
    _solution_time: float = field(default=0.0, repr=False)
    _burn_started: bool = field(default=False, repr=False)
    _burn_start_time: float = field(default=0.0, repr=False)
    _last_solve_attempt: float = field(default=-1.0, repr=False)

    def __post_init__(self):
        # Compute gravity vector (pointing toward Earth center from target)
        r_target = np.linalg.norm(self.target_position)
        g_mag = 9.80665 * (6378137.0 / r_target) ** 2
        self._g_vec = -self.target_position / r_target * g_mag
        self._up = -self._g_vec / np.linalg.norm(self._g_vec)
        
        # Precompute projection matrix for glideslope constraint
        # P = I - up @ up^T
        # h_vec = P @ (r - target)
        self._P_horiz = np.eye(3) - np.outer(self._up, self._up)

    def update_target(self, target_position: NDArray[np.float64]):
        """Update landing target and internal vectors.
        
        Used to compensate for Earth rotation or retargeting.
        """
        self.target_position = target_position
        
        # Recompute gravity vector
        r_target = np.linalg.norm(self.target_position)
        g_mag = 9.80665 * (6378137.0 / r_target) ** 2
        self._g_vec = -self.target_position / r_target * g_mag
        self._up = -self._g_vec / np.linalg.norm(self._g_vec)
        
        # Recompute projection matrix
        self._P_horiz = np.eye(3) - np.outer(self._up, self._up)

    def solve(self, state: State) -> dict | None:
        """Solve G-FOLD from current state.
        
        Returns optimal trajectory or None if infeasible.
        """
        g0 = 9.80665
        alpha = 1.0 / (self.isp * g0)  # Mass flow rate per unit thrust

        # Current state
        r0 = state.position
        v0 = state.velocity
        m0 = state.mass

        # Estimate time of flight
        # Use energy-based estimate: need to kill velocity and fall to ground
        r_rel = r0 - self.target_position
        alt = np.dot(r_rel, self._up)
        v_vert = np.dot(v0, self._up)
        v_horiz = np.linalg.norm(v0 - v_vert * self._up)

        # Time to fall (simplified)
        g = np.linalg.norm(self._g_vec)
        if v_vert >= 0:
            # Going up - time to apex + time to fall
            t_up = v_vert / g
            alt_apex = alt + 0.5 * v_vert * t_up
            t_down = np.sqrt(2 * max(alt_apex, 0) / g)
            t_fall = t_up + t_down
        else:
            # Going down
            # Solve: alt + v_vert*t - 0.5*g*t^2 = 0
            disc = v_vert**2 + 2*g*alt
            if disc < 0:
                t_fall = 30.0
            else:
                t_fall = (-v_vert + np.sqrt(disc)) / g

        # Estimate burn time needed (conservative)
        # Need to remove velocity + fight gravity
        dv_needed = np.linalg.norm(v0) + g * t_fall * 0.5 # Rough estimate
        a_max_approx = self.max_thrust / m0
        t_burn = dv_needed / a_max_approx
        
        # Use the larger of fall time or burn time as baseline
        t_guess = max(t_fall, t_burn) * 1.1
        t_guess = np.clip(t_guess, 5.0, 120.0)

        # Try multiple time horizons
        # Order: try best guess, then slightly longer (conservative), then shorter
        # Reduced set to avoid stalling the simulation
        self._last_solve_attempt = state.time
        for tf_mult in [1.0, 1.3, 0.9]:
            tf = t_guess * tf_mult
            result = self._solve_fixed_time(state, tf)
            if result is not None:
                self._solution = result
                self._solution_time = state.time
                return result

        return None

    def _solve_fixed_time(self, state: State, tf: float) -> dict | None:
        """Solve G-FOLD with fixed final time."""
        N = self.n_nodes
        dt = tf / (N - 1)
        g0 = 9.80665
        alpha = 1.0 / (self.isp * g0)

        r0 = state.position
        v0 = state.velocity
        m0 = state.mass

        # Decision variables
        r = cp.Variable((N, 3))  # Position
        v = cp.Variable((N, 3))  # Velocity
        u = cp.Variable((N, 3))  # Thrust acceleration (T/m)
        z = cp.Variable(N)       # Log mass: z = ln(m)
        sigma = cp.Variable(N)   # Thrust magnitude slack

        # Initial log mass
        z0 = np.log(m0)

        # Mass bounds
        z_min = np.log(self.dry_mass)
        z_max = z0

        constraints = []

        # Initial conditions
        constraints += [
            r[0] == r0,
            v[0] == v0,
            z[0] == z0,
        ]

        # Terminal conditions
        constraints += [
            r[N-1] == self.target_position,
            v[N-1] == np.zeros(3),  # Soft landing
        ]

        # Dynamics (Euler integration)
        for k in range(N - 1):
            constraints += [
                r[k+1] == r[k] + dt * v[k] + 0.5 * dt**2 * (self._g_vec + u[k]),
                v[k+1] == v[k] + dt * (self._g_vec + u[k]),
            ]

            # Mass dynamics: dm/dt = -alpha * T, so dz/dt = -alpha * T / m = -alpha * sigma
            # Using sigma as thrust magnitude
            constraints += [
                z[k+1] == z[k] - alpha * sigma[k] * dt,
            ]

        # Thrust constraints (lossless convexification)
        # The key insight: ||u|| <= sigma and thrust bounds on sigma
        # This is a second-order cone constraint
        rho_min = self.min_thrust
        rho_max = self.max_thrust

        for k in range(N):
            # Mass bounds
            constraints += [z[k] >= z_min, z[k] <= z_max]

            # Thrust acceleration magnitude
            constraints += [cp.norm(u[k]) <= sigma[k]]

            # Thrust bounds (converted to acceleration using mass)
            # T_min / m <= sigma <= T_max / m
            # Using first-order Taylor expansion: 1/m ≈ (1/m0) * exp(z0 - z)
            # For convexity, we use: exp(z0 - z) >= 1 + (z0 - z) (lower bound)
            #                        exp(z0 - z) <= exp(z0 - z_ref) * (1 + (z_ref - z)) (upper bound)

            # Simplified: use average mass estimate for bounds
            # This is less accurate but keeps problem tractable
            m_est = m0 * np.exp(-alpha * rho_max * 0.7 * k * dt / m0)
            m_est = max(m_est, self.dry_mass)

            constraints += [
                sigma[k] >= rho_min / m_est,
                sigma[k] <= rho_max / m_est,
            ]

            # Pointing constraint (thrust mostly upward)
            # u · up >= cos(max_tilt) * ||u||
            cos_tilt = np.cos(self.max_tilt)
            constraints += [u[k] @ self._up >= cos_tilt * sigma[k]]

        # Glide slope constraint (stay within cone above target)
        for k in range(N):
            rel_pos = r[k] - self.target_position
            
            # Use precomputed projection matrix to simplify expression graph
            # h_dist = || P @ rel_pos ||
            h_dist_vec = self._P_horiz @ rel_pos
            
            v_dist = rel_pos @ self._up  # Vertical component

            constraints += [
                cp.norm(h_dist_vec) <= self.glideslope_tan * v_dist,
                v_dist >= 0,  # Stay above ground
            ]

        # Objective: minimize fuel = integral of thrust magnitude
        objective = cp.Minimize(cp.sum(sigma) * dt)

        # Solve
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.CLARABEL, verbose=False, time_limit=0.05)
        except (cp.SolverError, AttributeError, Exception) as e:
            # Catch AttributeError specifically to handle cvxpy recursion issues
            return None

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return None

        # Extract solution
        times = np.linspace(state.time, state.time + tf, N)
        masses = np.exp(z.value)
        thrust_acc = u.value
        thrust_mag = sigma.value * masses  # Convert acceleration to force

        return {
            "time": times,
            "position": r.value,
            "velocity": v.value,
            "thrust_acceleration": thrust_acc,
            "thrust_magnitude": thrust_mag,
            "mass": masses,
            "tf": tf,
        }

    def get_command(self, state: State) -> dict:
        """Get guidance command for current state.
        
        This is the main interface called by the flight computer each timestep.
        
        Returns:
            dict with 'thrust' (magnitude) and 'direction' (unit vector)
        """
        # Compute distance/time to target
        r_rel = state.position - self.target_position
        alt = np.dot(r_rel, self._up)
        dist = np.linalg.norm(r_rel)
        speed = np.linalg.norm(state.velocity)

        # Check if we've landed
        if alt < 10.0 and speed < 5.0:
            return {"thrust": 0.0, "direction": self._up, "phase": "LANDED"}

        # Check if below ground (crashed)
        if alt < 0:
            return {"thrust": 0.0, "direction": self._up, "phase": "CRASHED"}

        # Rate limit solve attempts to prevent simulation hang when infeasible
        # If we have no solution, only try to solve once per second
        if self._solution is None and state.time - self._last_solve_attempt < 1.0:
             pass # Skip solve, use fallback
        else:
            # Solve G-FOLD if we don't have a recent solution
            need_resolve = (
                self._solution is None or
                state.time - self._solution_time > 5.0 or  # Re-solve every 5s
                (self._burn_started and state.time - self._solution_time > 1.0)  # More frequent during burn
            )

            if need_resolve:
                self.solve(state)

        if self._solution is None:
            # No solution - use heuristic fallback (point retrograde, full thrust)
            if speed > 1.0:
                direction = -state.velocity / speed
            else:
                direction = self._up

            # Only thrust if we're getting close to ground
            if alt < 60000 and speed > 100:
                return {"thrust": self.max_thrust, "direction": direction, "phase": "FALLBACK"}
            else:
                return {"thrust": 0.0, "direction": direction, "phase": "COAST_FALLBACK"}

        # Interpolate solution to current time
        sol = self._solution
        t_rel = state.time - sol["time"][0]

        if t_rel < 0:
            # Before solution start - coast
            return {"thrust": 0.0, "direction": self._up, "phase": "COAST"}

        if t_rel >= sol["tf"]:
            # Past solution end - should have landed, use fallback
            if speed > 1.0:
                direction = -state.velocity / speed
            else:
                direction = self._up
            return {"thrust": self.max_thrust * 0.5, "direction": direction, "phase": "FINAL"}

        # Interpolate thrust
        idx = np.searchsorted(sol["time"] - sol["time"][0], t_rel)
        idx = min(idx, len(sol["time"]) - 1)

        thrust_mag = sol["thrust_magnitude"][idx]
        thrust_acc = sol["thrust_acceleration"][idx]

        # Get direction from thrust acceleration
        acc_mag = np.linalg.norm(thrust_acc)
        if acc_mag > 0.1:
            direction = thrust_acc / acc_mag
        else:
            direction = self._up

        # Determine if we should be burning
        # G-FOLD gives us the optimal trajectory - if thrust > min_thrust, burn
        if thrust_mag >= self.min_thrust * 0.9:
            if not self._burn_started:
                self._burn_started = True
                self._burn_start_time = state.time
            return {"thrust": thrust_mag, "direction": direction, "phase": "BURN"}
        else:
            return {"thrust": 0.0, "direction": direction, "phase": "COAST"}

    def should_start_burn(self, state: State) -> bool:
        """Check if we should start the landing burn now.
        
        This is determined by solving G-FOLD and checking if the optimal
        trajectory starts with thrust.
        """
        if self._burn_started:
            return True

        result = self.solve(state)
        if result is None:
            return False

        # Check if first few nodes have significant thrust
        avg_thrust = np.mean(result["thrust_magnitude"][:5])
        return avg_thrust >= self.min_thrust * 0.5
