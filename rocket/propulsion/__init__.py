"""Propulsion module for dynamic engine modeling.

Bridges static engine design to dynamic simulation with throttle
and gimbal control.

Example:
    >>> from rocket import design_engine, EngineInputs
    >>> from rocket.propulsion import ThrottleModel
    >>>
    >>> # Design engine
    >>> inputs = EngineInputs.from_propellants("LOX", "CH4", ...)
    >>> perf, geom = design_engine(inputs)
    >>>
    >>> # Create throttle model
    >>> throttle = ThrottleModel(perf, min_throttle=0.4)
    >>>
    >>> # Get thrust at 80% throttle, 10km altitude
    >>> thrust, mdot = throttle.at(throttle=0.8, altitude=10000)
"""

from rocket.propulsion.throttle_model import (
    GimbalModel,
    ThrottleModel,
)

__all__ = [
    "ThrottleModel",
    "GimbalModel",
]

