"""Dynamics module for 6DOF rigid body simulation.

This module provides the equations of motion and state representation
for simulating rocket vehicle dynamics.

Example:
    >>> from rocket.dynamics import State, RigidBodyDynamics
    >>> import numpy as np
    >>>
    >>> state = State.from_flat_earth(z=0, mass_kg=5000)
    >>> dynamics = RigidBodyDynamics(inertia=np.diag([1000, 1000, 500]))
    >>> thrust = np.array([0, 0, -50000])
    >>> moment = np.array([0, 0, 0])
    >>> state_dot = dynamics.derivatives(state, thrust, moment, mdot=-10)
"""

from rocket.dynamics.rigid_body import (
    DynamicsConfig,
    RigidBodyDynamics,
    euler_rotational_dynamics,
    integrate,
    quaternion_derivative,
    rigid_body_derivatives,
    rk4_step,
)
from rocket.dynamics.state import (
    State,
    StateDerivative,
    dcm_to_quaternion,
    euler_to_quaternion,
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_dcm,
    quaternion_to_euler,
)

__all__ = [
    # State
    "State",
    "StateDerivative",
    # Quaternion utilities
    "quaternion_to_dcm",
    "dcm_to_quaternion",
    "euler_to_quaternion",
    "quaternion_to_euler",
    "quaternion_multiply",
    "quaternion_conjugate",
    "normalize_quaternion",
    # Rigid body dynamics
    "RigidBodyDynamics",
    "DynamicsConfig",
    "rigid_body_derivatives",
    "quaternion_derivative",
    "euler_rotational_dynamics",
    # Integration
    "integrate",
    "rk4_step",
]
