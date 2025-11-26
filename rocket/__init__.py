"""Rocket - Tools for rocket vehicle design and analysis.

This package provides a comprehensive toolkit for designing and analyzing
rocket vehicles, including propulsion, tanks, and structures.

Example:
    >>> from rocket import EngineInputs, design_engine
    >>> from rocket.units import kilonewtons, megapascals
    >>>
    >>> inputs = EngineInputs.from_propellants(
    ...     oxidizer="LOX",
    ...     fuel="RP1",
    ...     thrust=kilonewtons(100),
    ...     chamber_pressure=megapascals(7),
    ...     mixture_ratio=2.7,
    ... )
    >>> performance, geometry = design_engine(inputs)
    >>> print(f"Isp: {performance.isp.value:.1f} s")
"""

__version__ = "0.3.0"

# Core engine design
from rocket.engine import (
    EngineGeometry,
    EngineInputs,
    EnginePerformance,
    compute_geometry,
    compute_performance,
    design_engine,
    format_geometry_summary,
    format_performance_summary,
    isp_at_altitude,
    thrust_at_altitude,
)

# Nozzle contour generation
from rocket.nozzle import (
    NozzleContour,
    conical_contour,
    full_chamber_contour,
    generate_nozzle_from_geometry,
    rao_bell_contour,
)

# Visualization
from rocket.plotting import (
    plot_engine_cross_section,
    plot_engine_dashboard,
    plot_mass_breakdown,
    plot_nozzle_contour,
    plot_performance_vs_altitude,
)

# Propellants and thermochemistry
from rocket.propellants import (
    CombustionProperties,
    get_combustion_properties,
    get_optimal_mixture_ratio,
    is_cea_available,
)

# Tank sizing
from rocket.tanks import (
    PropellantRequirements,
    TankGeometry,
    format_tank_summary,
    get_propellant_density,
    list_materials,
    list_propellants,
    size_propellant,
    size_tank,
)

__all__ = [
    # Version
    "__version__",
    # Engine dataclasses
    "EngineInputs",
    "EnginePerformance",
    "EngineGeometry",
    # Engine computation
    "compute_performance",
    "compute_geometry",
    "design_engine",
    "thrust_at_altitude",
    "isp_at_altitude",
    "format_performance_summary",
    "format_geometry_summary",
    # Nozzle
    "NozzleContour",
    "rao_bell_contour",
    "conical_contour",
    "full_chamber_contour",
    "generate_nozzle_from_geometry",
    # Plotting
    "plot_engine_cross_section",
    "plot_nozzle_contour",
    "plot_performance_vs_altitude",
    "plot_engine_dashboard",
    "plot_mass_breakdown",
    # Propellants
    "CombustionProperties",
    "get_combustion_properties",
    "get_optimal_mixture_ratio",
    "is_cea_available",
    # Tanks
    "PropellantRequirements",
    "TankGeometry",
    "size_propellant",
    "size_tank",
    "get_propellant_density",
    "list_propellants",
    "list_materials",
    "format_tank_summary",
]
