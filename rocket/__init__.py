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

# Output management
from rocket.output import (
    OutputContext,
    clean_outputs,
    get_default_output_dir,
    list_outputs,
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

# System-level design
from rocket.system import (
    EngineSystemResult,
    design_engine_system,
    format_system_summary,
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

# Validation against real engines
from rocket.validation import (
    ReferenceEngine,
    ValidationResult,
    describe_reference,
    get_reference,
    list_reference_engines,
    run_all_validations,
    validate_against,
    validation_report,
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
    # Tanks
    "PropellantRequirements",
    "TankGeometry",
    "size_propellant",
    "size_tank",
    "get_propellant_density",
    "list_propellants",
    "list_materials",
    "format_tank_summary",
    # Validation
    "ReferenceEngine",
    "ValidationResult",
    "validate_against",
    "validation_report",
    "run_all_validations",
    "get_reference",
    "describe_reference",
    "list_reference_engines",
]
