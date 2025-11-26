"""6DOF rigid body equations of motion.

Implements the equations of motion for a rigid body rocket vehicle,
computing state derivatives from forces, moments, and current state.

The equations use:
- Newton's second law for translational motion: F = m * a
- Euler's equations for rotational motion: M = I * alpha + omega x (I * omega)
- Quaternion kinematics for attitude propagation

Example:
    >>> from rocket.dynamics import State, rigid_body_derivatives
    >>> from rocket.environment import Gravity, Atmosphere
    >>>
    >>> # Current state
    >>> state = State.from_launch_pad(mass_kg=5000)
    >>>
    >>> # Forces in body frame
    >>> thrust = np.array([0, 0, -50000])  # 50 kN thrust along body Z
    >>>
    >>> # Compute derivatives
    >>> inertia = np.diag([1000, 1000, 500])  # kg*m^2
    >>> state_dot = rigid_body_derivatives(state, thrust, moments, inertia, mdot)
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

from rocket.dynamics.state import (
    State,
    StateDerivative,
    normalize_quaternion,
    quaternion_to_dcm,
)
from rocket.environment.atmosphere import Atmosphere
from rocket.environment.gravity import Gravity, GravityModel

# =============================================================================
# Force/Moment Protocols
# =============================================================================


class ForceModel(Protocol):
    """Protocol for force models."""

    def compute(
        self,
        state: State,
        time: float,
    ) -> NDArray[np.float64]:
        """Compute force vector in body frame [N]."""
        ...


class MomentModel(Protocol):
    """Protocol for moment/torque models."""

    def compute(
        self,
        state: State,
        time: float,
    ) -> NDArray[np.float64]:
        """Compute moment vector in body frame [N*m]."""
        ...


# =============================================================================
# Rigid Body Dynamics
# =============================================================================


@beartype
def quaternion_derivative(
    q: NDArray[np.float64],
    omega: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute quaternion time derivative from angular velocity.

    Args:
        q: Current quaternion [q0, q1, q2, q3]
        omega: Angular velocity in body frame [p, q, r] [rad/s]

    Returns:
        Quaternion derivative dq/dt
    """
    q0, q1, q2, q3 = q
    p, qb, r = omega  # Using qb to avoid confusion with quaternion q

    # Quaternion kinematics matrix
    omega_matrix = np.array([
        [0, -p, -qb, -r],
        [p, 0, r, -qb],
        [qb, -r, 0, p],
        [r, qb, -p, 0],
    ])

    return 0.5 * omega_matrix @ q


@beartype
def euler_rotational_dynamics(
    omega: NDArray[np.float64],
    moment: NDArray[np.float64],
    inertia: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute angular acceleration from Euler's equations.

    Euler's equations: I * omega_dot = M - omega x (I * omega)

    Args:
        omega: Angular velocity in body frame [p, q, r] [rad/s]
        moment: Applied moment in body frame [Mx, My, Mz] [N*m]
        inertia: 3x3 inertia tensor [kg*m^2]

    Returns:
        Angular acceleration [p_dot, q_dot, r_dot] [rad/s^2]
    """
    # Gyroscopic term: omega x (I * omega)
    I_omega = inertia @ omega
    gyroscopic = np.cross(omega, I_omega)

    # Solve I * omega_dot = M - gyroscopic
    omega_dot = np.linalg.solve(inertia, moment - gyroscopic)

    return omega_dot


@beartype
def rigid_body_derivatives(
    state: State,
    force_body: NDArray[np.float64],
    moment_body: NDArray[np.float64],
    inertia: NDArray[np.float64],
    mass_rate: float = 0.0,
) -> StateDerivative:
    """Compute state derivatives for 6DOF rigid body motion.

    Args:
        state: Current vehicle state
        force_body: Total force in body frame [Fx, Fy, Fz] [N]
        moment_body: Total moment in body frame [Mx, My, Mz] [N*m]
        inertia: 3x3 inertia tensor in body frame [kg*m^2]
        mass_rate: Mass flow rate (negative for propellant consumption) [kg/s]

    Returns:
        State derivatives for integration
    """
    # Position derivative = velocity
    position_dot = state.velocity.copy()

    # Transform force to inertial frame
    dcm_body_to_inertial = quaternion_to_dcm(
        np.array([state.quaternion[0], -state.quaternion[1],
                  -state.quaternion[2], -state.quaternion[3]])
    )
    force_inertial = dcm_body_to_inertial @ force_body

    # Velocity derivative = acceleration = F/m
    velocity_dot = force_inertial / state.mass

    # Quaternion derivative from kinematics
    quaternion_dot = quaternion_derivative(state.quaternion, state.angular_velocity)

    # Angular velocity derivative from Euler's equations
    angular_velocity_dot = euler_rotational_dynamics(
        state.angular_velocity,
        moment_body,
        inertia,
    )

    return StateDerivative(
        position_dot=position_dot,
        velocity_dot=velocity_dot,
        quaternion_dot=quaternion_dot,
        angular_velocity_dot=angular_velocity_dot,
        mass_dot=mass_rate,
    )


# =============================================================================
# Integration
# =============================================================================


@beartype
def rk4_step(
    state: State,
    dt: float,
    derivatives_fn,
) -> State:
    """Perform one RK4 integration step.

    Args:
        state: Current state
        dt: Time step [s]
        derivatives_fn: Function that computes StateDerivative from State

    Returns:
        State at t + dt
    """
    # Get state as array
    y0 = state.to_array()
    t0 = state.time
    flat_earth = state.flat_earth

    def f(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        s = State.from_array(y, t, flat_earth=flat_earth)
        return derivatives_fn(s).to_array()

    # RK4 stages
    k1 = f(t0, y0)
    k2 = f(t0 + dt/2, y0 + dt/2 * k1)
    k3 = f(t0 + dt/2, y0 + dt/2 * k2)
    k4 = f(t0 + dt, y0 + dt * k3)

    # Update
    y1 = y0 + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    # Create new state and normalize quaternion
    new_state = State.from_array(y1, t0 + dt, flat_earth=flat_earth)
    new_state.quaternion = normalize_quaternion(new_state.quaternion)

    return new_state


@beartype
def integrate(
    initial_state: State,
    derivatives_fn,
    t_final: float,
    dt: float = 0.01,
    max_steps: int = 1000000,
) -> list[State]:
    """Integrate equations of motion over time.

    Args:
        initial_state: Initial state
        derivatives_fn: Function computing StateDerivative from State
        t_final: Final simulation time [s]
        dt: Time step [s]
        max_steps: Maximum number of steps

    Returns:
        List of states at each time step
    """
    states = [initial_state]
    state = initial_state.copy()

    n_steps = int(t_final / dt)
    n_steps = min(n_steps, max_steps)

    for _ in range(n_steps):
        state = rk4_step(state, dt, derivatives_fn)
        states.append(state)

        if state.time >= t_final:
            break

    return states


# =============================================================================
# Complete Dynamics Model
# =============================================================================


@beartype
@dataclass
class DynamicsConfig:
    """Configuration for dynamics simulation.

    Attributes:
        gravity_model: Gravity model type
        include_atmosphere: Whether to include atmospheric effects
        include_j2: Whether to include J2 perturbation
    """
    gravity_model: GravityModel = GravityModel.SPHERICAL
    include_atmosphere: bool = True

    def create_gravity(self) -> Gravity:
        """Create gravity model from config."""
        return Gravity(model=self.gravity_model)

    def create_atmosphere(self) -> Atmosphere | None:
        """Create atmosphere model from config."""
        return Atmosphere() if self.include_atmosphere else None


@beartype
class RigidBodyDynamics:
    """Complete 6DOF rigid body dynamics model.

    Combines gravity, aerodynamics, and propulsion into a unified
    dynamics model for rocket vehicle simulation.

    Example:
        >>> dynamics = RigidBodyDynamics(
        ...     inertia=np.diag([1000, 1000, 500]),
        ... )
        >>>
        >>> state = State.from_flat_earth(z=0, mass_kg=5000)
        >>> thrust = np.array([0, 0, -50000])
        >>>
        >>> state_dot = dynamics.derivatives(state, thrust, moments, mdot)
    """

    def __init__(
        self,
        inertia: NDArray[np.float64],
        config: DynamicsConfig | None = None,
    ) -> None:
        """Initialize dynamics model.

        Args:
            inertia: 3x3 inertia tensor in body frame [kg*m^2]
            config: Dynamics configuration
        """
        self.inertia = np.asarray(inertia, dtype=np.float64)
        self.config = config or DynamicsConfig()

        self.gravity = self.config.create_gravity()
        self.atmosphere = self.config.create_atmosphere()

    @beartype
    def gravity_force(self, state: State) -> NDArray[np.float64]:
        """Compute gravity force in inertial frame [N]."""
        g = self.gravity.acceleration(state.position)
        return state.mass * g

    @beartype
    def derivatives(
        self,
        state: State,
        thrust_body: NDArray[np.float64],
        moment_body: NDArray[np.float64],
        mass_rate: float = 0.0,
        aero_force_body: NDArray[np.float64] | None = None,
        aero_moment_body: NDArray[np.float64] | None = None,
    ) -> StateDerivative:
        """Compute state derivatives.

        Args:
            state: Current vehicle state
            thrust_body: Thrust force in body frame [N]
            moment_body: Control moments in body frame [N*m]
            mass_rate: Propellant mass flow rate (negative) [kg/s]
            aero_force_body: Aerodynamic force in body frame [N]
            aero_moment_body: Aerodynamic moment in body frame [N*m]

        Returns:
            State derivatives
        """
        # Total forces in body frame
        total_force_body = thrust_body.copy()
        total_moment_body = moment_body.copy()

        # Add aerodynamic forces if provided
        if aero_force_body is not None:
            total_force_body += aero_force_body
        if aero_moment_body is not None:
            total_moment_body += aero_moment_body

        # Add gravity (convert to body frame)
        gravity_inertial = self.gravity_force(state)
        dcm_inertial_to_body = state.dcm_inertial_to_body
        gravity_body = dcm_inertial_to_body @ gravity_inertial
        total_force_body += gravity_body

        return rigid_body_derivatives(
            state=state,
            force_body=total_force_body,
            moment_body=total_moment_body,
            inertia=self.inertia,
            mass_rate=mass_rate,
        )

    @beartype
    def simulate(
        self,
        initial_state: State,
        thrust_fn,
        moment_fn,
        mass_rate_fn,
        t_final: float,
        dt: float = 0.01,
    ) -> list[State]:
        """Run simulation with time-varying inputs.

        Args:
            initial_state: Initial state
            thrust_fn: Function(state, t) -> thrust_body [N]
            moment_fn: Function(state, t) -> moment_body [N*m]
            mass_rate_fn: Function(state, t) -> mass_rate [kg/s]
            t_final: Final time [s]
            dt: Time step [s]

        Returns:
            List of states over time
        """
        def derivatives_fn(state: State) -> StateDerivative:
            t = state.time
            thrust = thrust_fn(state, t)
            moment = moment_fn(state, t)
            mdot = mass_rate_fn(state, t)
            return self.derivatives(state, thrust, moment, mdot)

        return integrate(initial_state, derivatives_fn, t_final, dt)

