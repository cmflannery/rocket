"""Engine cycle analysis module for Rocket.

This module provides analysis tools for different rocket engine cycles:
- Pressure-fed (simplest)
- Gas generator (turbopump-fed with separate combustion)
- Expander (turbine driven by heated propellant)
- Staged combustion (preburner exhaust into main chamber)

Each cycle type has different performance characteristics, complexity,
and feasibility constraints.

Example:
    >>> from rocket import EngineInputs, design_engine
    >>> from rocket.cycles import GasGeneratorCycle, analyze_cycle
    >>>
    >>> inputs = EngineInputs.from_propellants("LOX", "CH4", ...)
    >>> performance, geometry = design_engine(inputs)
    >>> 
    >>> cycle = GasGeneratorCycle(
    ...     turbine_inlet_temp=kelvin(900),
    ...     pump_efficiency=0.70,
    ... )
    >>> result = analyze_cycle(inputs, performance, geometry, cycle)
    >>> print(f"Net Isp: {result.net_isp.value:.1f} s")
"""

from rocket.cycles.base import (
    CycleConfiguration,
    CyclePerformance,
    CycleType,
    analyze_cycle,
    format_cycle_summary,
)
from rocket.cycles.gas_generator import GasGeneratorCycle
from rocket.cycles.pressure_fed import PressureFedCycle
from rocket.cycles.staged_combustion import (
    FullFlowStagedCombustion,
    StagedCombustionCycle,
)

__all__ = [
    # Base types
    "CycleConfiguration",
    "CyclePerformance",
    "CycleType",
    # Cycle configurations
    "PressureFedCycle",
    "GasGeneratorCycle",
    "StagedCombustionCycle",
    "FullFlowStagedCombustion",
    # Analysis function
    "analyze_cycle",
    "format_cycle_summary",
]

