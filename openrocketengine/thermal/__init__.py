"""Thermal analysis module for Rocket.

This module provides tools for analyzing thermal loads on rocket engine
components and evaluating cooling system feasibility.

Key capabilities:
- Heat flux estimation using Bartz correlation
- Regenerative cooling feasibility screening
- Wall temperature prediction
- Coolant property database

Example:
    >>> from openrocketengine import EngineInputs, design_engine
    >>> from openrocketengine.thermal import estimate_heat_flux, check_cooling_feasibility
    >>>
    >>> inputs = EngineInputs.from_propellants("LOX", "CH4", ...)
    >>> performance, geometry = design_engine(inputs)
    >>>
    >>> q_throat = estimate_heat_flux(inputs, performance, geometry, location="throat")
    >>> print(f"Throat heat flux: {q_throat.value/1e6:.1f} MW/m²")
    >>>
    >>> cooling = check_cooling_feasibility(
    ...     inputs, performance, geometry,
    ...     coolant="CH4",
    ...     max_wall_temp=kelvin(800),
    ... )
    >>> print(f"Cooling feasible: {cooling.feasible}")
"""

from openrocketengine.thermal.heat_flux import (
    adiabatic_wall_temperature,
    bartz_heat_flux,
    estimate_heat_flux,
    heat_flux_profile,
    recovery_factor,
)
from openrocketengine.thermal.regenerative import (
    CoolingFeasibility,
    CoolantProperties,
    check_cooling_feasibility,
    get_coolant_properties,
)

__all__ = [
    # Heat flux estimation
    "adiabatic_wall_temperature",
    "bartz_heat_flux",
    "estimate_heat_flux",
    "heat_flux_profile",
    "recovery_factor",
    # Regenerative cooling
    "CoolingFeasibility",
    "CoolantProperties",
    "check_cooling_feasibility",
    "get_coolant_properties",
]

