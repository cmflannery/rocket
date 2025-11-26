"""OpenRocketEngine - Tools for liquid rocket engine design and analysis.

This package provides a comprehensive toolkit for designing and analyzing
liquid propellant rocket engines using isentropic flow equations.

Example:
    >>> from openrocketengine import EngineInputs, design_engine
    >>> from openrocketengine.units import newtons, megapascals, kelvin, meters, pascals
    >>>
    >>> inputs = EngineInputs(
    ...     thrust=newtons(5000),
    ...     chamber_pressure=megapascals(2.0),
    ...     chamber_temp=kelvin(3200),
    ...     exit_pressure=pascals(101325),
    ...     molecular_weight=22.0,
    ...     gamma=1.2,
    ...     lstar=meters(1.0),
    ...     mixture_ratio=2.0,
    ... )
    >>> performance, geometry = design_engine(inputs)
    >>> print(f"Isp: {performance.isp.value:.1f} s")
"""

__version__ = "0.2.0"

# Core engine design
from openrocketengine.engine import (
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
from openrocketengine.nozzle import (
    NozzleContour,
    conical_contour,
    full_chamber_contour,
    generate_nozzle_from_geometry,
    rao_bell_contour,
)

# Visualization
from openrocketengine.plotting import (
    plot_engine_cross_section,
    plot_engine_dashboard,
    plot_nozzle_contour,
    plot_performance_vs_altitude,
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
]
