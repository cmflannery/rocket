"""Environment models for rocket vehicle simulation.

Provides atmospheric, gravitational, and wind models for flight simulation.

Example:
    >>> from rocket.environment import Atmosphere, Gravity
    >>>
    >>> atm = Atmosphere()
    >>> rho = atm.density(altitude=10000)  # kg/m^3
    >>>
    >>> grav = Gravity()
    >>> g = grav.acceleration(position)  # m/s^2
"""

from rocket.environment.atmosphere import (
    Atmosphere,
    AtmosphereResult,
)
from rocket.environment.gravity import (
    Gravity,
    GravityModel,
)

__all__ = [
    "Atmosphere",
    "AtmosphereResult",
    "Gravity",
    "GravityModel",
]

