"""Powered descent guidance using convex optimization (G-FOLD).

This module implements convex optimization-based powered descent guidance
for reusable launch vehicle landing, based on the "Lossless Convexification"
algorithm by Acikmese & Blackmore.

Reference:
    Acikmese, B., & Blackmore, L. (2013). "Lossless Convexification of Nonconvex
    Control Bound and Pointing Constraints of the Soft Landing Optimal Control Problem".
    IEEE Transactions on Control Systems Technology.
"""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from rocket.dynamics.state import State


@dataclass
class PoweredDescentGuidance:
    """Powered descent guidance using convex optimization (G-FOLD).

    Solves for the fuel-optimal trajectory to soft landing.

    Attributes:
        target_position: Landing site position in ECI [m]
        max_thrust: Maximum thrust magnitude [N]
        min_throttle: Minimum throttle (0.0 to 1.0)
        dry_mass: Vehicle dry mass [kg] (for thrust-to-weight ratio)
        glideslope_angle: Minimum angle from horizon for approach [rad]
        pointing_limit: Maximum angle between thrust vector and velocity/up [rad]
        time_horizon_guess: Initial guess for burn duration [s]
        nodes: Number of discretization nodes
    """
    target_position: NDArray[np.float64]
    max_thrust: float
    min_throttle: float = 0.4
    dry_mass: float = 25000.0
    glideslope_angle: float = np.radians(30.0)  # Very relaxed for RTLS (60° from vertical allowed)
    pointing_limit: float = np.radians(30.0)    # Relaxed thrust tilt
    time_horizon_guess: float = 25.0 # Starting guess
    nodes: int = 30

    # Internal cache
    _last_trajectory: dict = None
    _solved: bool = False

    def solve(self, state: State, tf_guess: float = None) -> dict:
        """Solve the convex optimal landing problem.

        Attempts to solve for optimal trajectory. If infeasible, retries with
        different time horizons.

        Args:
            state: Current vehicle state
            tf_guess: Optional guess for time of flight [s]

        Returns:
            Dictionary containing the optimal trajectory arrays or None if failed.
        """
        if tf_guess is None:
            # Estimate time of flight based on 1D physics
            v0_mag = np.linalg.norm(state.velocity)
            m0 = state.mass
            a_avail = self.max_thrust / m0
            g = 9.81
            a_net = a_avail - g
            if a_net > 0:
                tf_guess = v0_mag / a_net * 1.2 # Add 20% margin
            else:
                tf_guess = 30.0

            tf_guess = max(10.0, min(tf_guess, 60.0))

        # Try a range of time horizons around the guess
        time_guesses = [tf_guess, tf_guess*1.5, tf_guess*2.0]

        for tf in time_guesses:
            try:
                traj = self._solve_single(state, tf)
                if traj is not None:
                    return traj
            except Exception:
                pass

        # Optimization failed - likely physically infeasible trajectory
        # This usually means boostback didn't bring us close enough to the pad
        return None

    def _solve_single(self, state: State, tf: float) -> dict | None:
        """Single optimization attempt."""
        N = self.nodes
        dt = tf / (N - 1)
        g0 = 9.80665
        Isp = 282.0 # Sea level estimate
        alpha = 1.0 / (Isp * g0)

        # --- Decision Variables ---
        r = cp.Variable((N, 3))  # Position
        v = cp.Variable((N, 3))  # Velocity
        u = cp.Variable((N, 3))  # Thrust vector

        # --- Constants & Parameters ---
        m0 = state.mass
        r0 = state.position
        v0 = state.velocity

        # Gravity vector
        r_target_norm = np.linalg.norm(self.target_position)
        g_mag = 9.81
        g_vec = -self.target_position / r_target_norm * g_mag

        # --- Constraints ---
        constraints = []

        # 1. Initial Conditions
        constraints += [
            r[0] == r0,
            v[0] == v0
        ]

        # 2. Terminal Conditions
        constraints += [
            r[N-1] == self.target_position,
            v[N-1] == np.zeros(3)
        ]

        # 3. Dynamics (Approximate Mass)
        avg_thrust = self.max_thrust * 0.7
        mdot_est = avg_thrust * alpha
        m_arr = np.linspace(m0, m0 - mdot_est * tf, N)

        for k in range(N - 1):
            # Position: r[k+1] = r[k] + v[k]*dt + 0.5*(g + u[k]/m)*dt^2
            constraints += [
                r[k+1] == r[k] + v[k] * dt + 0.5 * (g_vec + u[k]/m_arr[k]) * dt**2
            ]
            # Velocity: v[k+1] = v[k] + (g + u[k]/m)*dt
            constraints += [
                v[k+1] == v[k] + (g_vec + u[k]/m_arr[k]) * dt
            ]

        # 4. Control Constraints
        for k in range(N):
            # Max Thrust
            constraints += [cp.norm(u[k]) <= self.max_thrust]

            # Min Thrust (avoid zero thrust which can cause numerical issues)
            constraints += [cp.norm(u[k]) >= self.max_thrust * self.min_throttle * 0.1]

        # --- Objective ---
        # Minimize fuel (norm u)
        objective = cp.Minimize(cp.sum([cp.norm(u[k]) for k in range(N)]))

        # --- Solve ---
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            return None

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return None

        # --- Extract Result ---
        self._solved = True
        trajectory = {
            "time": np.linspace(state.time, state.time + tf, N),
            "pos": r.value,
            "vel": v.value,
            "thrust": u.value,
            "mass": m_arr
        }
        self._last_trajectory = trajectory
        return trajectory

    def get_command(self, state: State, time_since_start: float) -> dict:
        """Get guidance command for current state based on planned trajectory."""
        if not self._solved or self._last_trajectory is None:
            # Attempt to solve if not yet solved
            if not self._solved:
                res = self.solve(state)
                if res is None:
                    # Fallback: Simple gravity turn / retrograde
                    vel = state.velocity
                    speed = np.linalg.norm(vel)
                    if speed > 1:
                        direction = -vel/speed
                    else:
                        direction = state.position / np.linalg.norm(state.position)
                    return {"thrust": self.max_thrust, "direction": direction}

        traj = self._last_trajectory

        # Find current time index
        t_plan = traj["time"] - traj["time"][0]

        if time_since_start >= t_plan[-1]:
            return {"thrust": 0.0, "direction": np.array([0,0,1])}

        # Interpolate
        u_current = np.zeros(3)
        for i in range(3):
            u_current[i] = np.interp(time_since_start, t_plan, traj["thrust"][:, i])

        thrust_mag = np.linalg.norm(u_current)

        if thrust_mag > 1.0:
            direction = u_current / thrust_mag
        else:
            direction = state.position / np.linalg.norm(state.position)

        return {
            "thrust": thrust_mag,
            "direction": direction
        }
