"""OpenRocketEngine - Tools for liquid rocket engine design and analysis.

This package provides a comprehensive toolkit for designing and analyzing
liquid propellant rocket engines using isentropic flow equations.

Example:
    >>> from rocket import EngineInputs, design_engine
    >>> from rocket.units import newtons, megapascals, kelvin, meters, pascals
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
    plot_cycle_comparison_bars,
    plot_cycle_radar,
    plot_cycle_tradeoff,
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
    list_database_propellants,
)

# Output management
from rocket.output import (
    OutputContext,
    clean_outputs,
    get_default_output_dir,
    list_outputs,
)

# Analysis framework
from rocket.analysis import (
    Distribution,
    LogNormal,
    MultiObjectiveOptimizer,
    Normal,
    ParametricStudy,
    ParetoResults,
    Range,
    StudyResults,
    Triangular,
    UncertaintyAnalysis,
    UncertaintyResults,
    Uniform,
)

# System-level design
from rocket.system import (
    EngineSystemResult,
    design_engine_system,
    format_system_summary,
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
    "plot_cycle_comparison_bars",
    "plot_cycle_radar",
    "plot_cycle_tradeoff",
    # Propellants
    "CombustionProperties",
    "get_combustion_properties",
    "get_optimal_mixture_ratio",
    "is_cea_available",
    "list_database_propellants",
    # Output management
    "OutputContext",
    "get_default_output_dir",
    "list_outputs",
    "clean_outputs",
    # Analysis framework
    "ParametricStudy",
    "UncertaintyAnalysis",
    "MultiObjectiveOptimizer",
    "StudyResults",
    "UncertaintyResults",
    "ParetoResults",
    "Range",
    "Distribution",
    "Normal",
    "Uniform",
    "Triangular",
    "LogNormal",
    # System-level design
    "EngineSystemResult",
    "design_engine_system",
    "format_system_summary",
]
