"""Vehicle modeling for rocket simulation.

Provides mass properties, aerodynamics, and staging models for
complete vehicle representation.

Example:
    >>> from rocket.vehicle import MassProperties, Vehicle
    >>>
    >>> # Define mass properties
    >>> mass = MassProperties(
    ...     mass=5000,
    ...     cg=np.array([0, 0, 2.5]),
    ...     inertia=np.diag([1000, 1000, 500]),
    ... )
    >>>
    >>> # Create vehicle
    >>> vehicle = Vehicle(
    ...     dry_mass=mass,
    ...     propellant_mass=4000,
    ...     engine=engine_perf,
    ... )
"""

from rocket.vehicle.aerodynamics import (
    AerodynamicsModel,
    SimpleAero,
)
from rocket.vehicle.mass import (
    MassProperties,
    Vehicle,
    compute_inertia_cylinder,
)

__all__ = [
    # Mass properties
    "MassProperties",
    "Vehicle",
    "compute_inertia_cylinder",
    # Aerodynamics
    "AerodynamicsModel",
    "SimpleAero",
]

