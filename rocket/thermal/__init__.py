"""Thermal analysis module for Rocket.

This module provides tools for analyzing thermal loads on rocket engine
components and evaluating cooling system feasibility.

Key capabilities:
- Heat flux estimation using Bartz correlation
- Regenerative cooling feasibility screening
- Transient thermal simulation (startup, shutdown, duty cycles)
- Material properties database
- Wall temperature prediction
- Coolant property database

Example:
    >>> from rocket import EngineInputs, design_engine
    >>> from rocket.thermal import estimate_heat_flux, check_cooling_feasibility
    >>>
    >>> inputs = EngineInputs.from_propellants("LOX", "CH4", ...)
    >>> performance, geometry = design_engine(inputs)
    >>>
    >>> q_throat = estimate_heat_flux(inputs, performance, geometry, location="throat")
    >>> print(f"Throat heat flux: {q_throat.value/1e6:.1f} MW/m²")
    >>>
    >>> # Transient simulation
    >>> from rocket.thermal import simulate_startup
    >>> result = simulate_startup(inputs, performance, geometry,
    ...     wall_thickness=meters(0.003), wall_material="copper")
    >>> print(f"Peak wall temp: {result.peak_wall_temp:.0f} K")
"""

from rocket.thermal.heat_flux import (
    adiabatic_wall_temperature,
    bartz_heat_flux,
    estimate_heat_flux,
    heat_flux_profile,
    recovery_factor,
)
from rocket.thermal.materials import (
    check_material_limits,
    compare_materials,
    get_material_properties,
    get_specific_heat,
    get_thermal_conductivity,
    get_thermal_diffusivity,
    list_materials,
)
from rocket.thermal.regenerative import (
    CoolantProperties,
    CoolingFeasibility,
    check_cooling_feasibility,
    get_coolant_properties,
)
from rocket.thermal.transient import (
    TransientThermalResult,
    simulate_duty_cycle,
    simulate_shutdown,
    simulate_startup,
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
    # Materials
    "get_material_properties",
    "get_thermal_conductivity",
    "get_specific_heat",
    "get_thermal_diffusivity",
    "check_material_limits",
    "compare_materials",
    "list_materials",
    # Transient thermal
    "TransientThermalResult",
    "simulate_startup",
    "simulate_shutdown",
    "simulate_duty_cycle",
]

