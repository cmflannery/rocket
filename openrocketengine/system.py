"""High-level system design API for Rocket.

This module provides the main entry point for complete engine system design,
integrating:
- Engine performance and geometry
- Cycle analysis (pressure-fed, gas-generator, etc.)
- Thermal/cooling feasibility

Example:
    >>> from openrocketengine import EngineInputs
    >>> from openrocketengine.system import design_engine_system
    >>> from openrocketengine.cycles import GasGeneratorCycle
    >>> from openrocketengine.units import kilonewtons, megapascals, kelvin
    >>>
    >>> inputs = EngineInputs.from_propellants(
    ...     oxidizer="LOX",
    ...     fuel="CH4",
    ...     thrust=kilonewtons(2000),
    ...     chamber_pressure=megapascals(30),
    ... )
    >>>
    >>> result = design_engine_system(
    ...     inputs=inputs,
    ...     cycle=GasGeneratorCycle(turbine_inlet_temp=kelvin(900)),
    ...     check_cooling=True,
    ... )
    >>>
    >>> print(f"Net Isp: {result.cycle.net_isp.value:.1f} s")
    >>> print(f"Cooling feasible: {result.cooling.feasible}")
"""

from dataclasses import dataclass
from typing import Any

from beartype import beartype

from openrocketengine.engine import (
    EngineGeometry,
    EngineInputs,
    EnginePerformance,
    compute_geometry,
    compute_performance,
)
from openrocketengine.cycles.base import CycleConfiguration, CyclePerformance, analyze_cycle
from openrocketengine.thermal.regenerative import CoolingFeasibility, check_cooling_feasibility
from openrocketengine.units import kelvin


@beartype
@dataclass(frozen=True)
class EngineSystemResult:
    """Complete engine system design result.

    Contains all analysis results from engine performance through
    cycle analysis and thermal assessment.

    Attributes:
        inputs: Original engine inputs
        performance: Ideal engine performance (before cycle losses)
        geometry: Engine geometry
        cycle: Cycle analysis results (if cycle provided)
        cooling: Cooling feasibility results (if check_cooling=True)
        feasible: Overall feasibility (cycle closes AND cooling ok)
        warnings: Combined warnings from all analyses
    """

    inputs: EngineInputs
    performance: EnginePerformance
    geometry: EngineGeometry
    cycle: CyclePerformance | None
    cooling: CoolingFeasibility | None
    feasible: bool
    warnings: list[str]


@beartype
def design_engine_system(
    inputs: EngineInputs,
    cycle: CycleConfiguration | None = None,
    check_cooling: bool = True,
    coolant: str | None = None,
    max_wall_temp: Any | None = None,
) -> EngineSystemResult:
    """Design a complete engine system with cycle and thermal analysis.

    This is the main entry point for rocket engine system design. It:
    1. Computes ideal engine performance and geometry
    2. Analyzes the engine cycle (if specified)
    3. Checks cooling feasibility (if requested)
    4. Returns a comprehensive result with all analyses

    Args:
        inputs: Engine input parameters (from EngineInputs.from_propellants or manual)
        cycle: Engine cycle configuration (PressureFedCycle, GasGeneratorCycle, etc.)
               If None, only basic performance is computed.
        check_cooling: Whether to perform cooling feasibility analysis
        coolant: Coolant name for cooling analysis. If None, uses fuel.
        max_wall_temp: Maximum allowed wall temperature [K]. If None, uses defaults.

    Returns:
        EngineSystemResult with all analysis results

    Example:
        >>> # Basic design without cycle analysis
        >>> result = design_engine_system(inputs)
        >>> print(f"Isp: {result.performance.isp.value:.1f} s")
        >>>
        >>> # Full system design with gas generator cycle
        >>> from openrocketengine.cycles import GasGeneratorCycle
        >>> result = design_engine_system(
        ...     inputs=inputs,
        ...     cycle=GasGeneratorCycle(turbine_inlet_temp=kelvin(900)),
        ...     check_cooling=True,
        ... )
    """
    all_warnings: list[str] = []

    # Step 1: Compute basic engine performance and geometry
    performance = compute_performance(inputs)
    geometry = compute_geometry(inputs, performance)

    # Step 2: Cycle analysis (if cycle provided)
    cycle_result: CyclePerformance | None = None
    if cycle is not None:
        cycle_result = analyze_cycle(inputs, performance, geometry, cycle)
        all_warnings.extend(cycle_result.warnings)

    # Step 3: Cooling feasibility (if requested)
    cooling_result: CoolingFeasibility | None = None
    if check_cooling:
        # Determine coolant (default to fuel)
        if coolant is None:
            # Try to infer fuel from engine name
            coolant = _infer_coolant(inputs)

        cooling_result = check_cooling_feasibility(
            inputs=inputs,
            performance=performance,
            geometry=geometry,
            coolant=coolant,
            max_wall_temp=max_wall_temp,
        )
        all_warnings.extend(cooling_result.warnings)

    # Determine overall feasibility
    feasible = True
    if cycle_result is not None and not cycle_result.feasible:
        feasible = False
    if cooling_result is not None and not cooling_result.feasible:
        feasible = False

    return EngineSystemResult(
        inputs=inputs,
        performance=performance,
        geometry=geometry,
        cycle=cycle_result,
        cooling=cooling_result,
        feasible=feasible,
        warnings=all_warnings,
    )


def _infer_coolant(inputs: EngineInputs) -> str:
    """Infer coolant from engine inputs."""
    if inputs.name:
        name_upper = inputs.name.upper()
        if "CH4" in name_upper or "METHANE" in name_upper or "METHALOX" in name_upper:
            return "CH4"
        elif "LH2" in name_upper or "HYDROGEN" in name_upper or "HYDROLOX" in name_upper:
            return "LH2"
        elif "RP1" in name_upper or "KEROSENE" in name_upper or "KEROLOX" in name_upper:
            return "RP1"
        elif "ETHANOL" in name_upper:
            return "Ethanol"

    # Default to RP-1
    return "RP1"


@beartype
def format_system_summary(result: EngineSystemResult) -> str:
    """Format complete system design results as readable string.

    Args:
        result: EngineSystemResult from design_engine_system()

    Returns:
        Formatted multi-line string summary
    """
    name = result.inputs.name or "Unnamed Engine"
    status = "✓ FEASIBLE" if result.feasible else "✗ NOT FEASIBLE"

    lines = [
        "=" * 70,
        f"ENGINE SYSTEM DESIGN: {name}",
        f"Overall Status: {status}",
        "=" * 70,
        "",
        "PERFORMANCE (Ideal):",
        f"  Thrust (SL):     {result.inputs.thrust.to('kN').value:.1f} kN",
        f"  Isp (SL):        {result.performance.isp.value:.1f} s",
        f"  Isp (Vac):       {result.performance.isp_vac.value:.1f} s",
        f"  C*:              {result.performance.cstar.value:.0f} m/s",
        f"  Mass Flow:       {result.performance.mdot.value:.2f} kg/s",
        "",
        "GEOMETRY:",
        f"  Throat Dia:      {result.geometry.throat_diameter.to('m').value*100:.1f} cm",
        f"  Exit Dia:        {result.geometry.exit_diameter.to('m').value*100:.1f} cm",
        f"  Chamber Dia:     {result.geometry.chamber_diameter.to('m').value*100:.1f} cm",
        f"  Expansion Ratio: {result.geometry.expansion_ratio:.1f}",
    ]

    if result.cycle is not None:
        lines.extend([
            "",
            f"CYCLE ({result.cycle.cycle_type.name}):",
            f"  Net Isp:         {result.cycle.net_isp.value:.1f} s",
            f"  Cycle Efficiency:{result.cycle.cycle_efficiency*100:.1f}%",
            f"  Turbine Power:   {result.cycle.turbine_power.value/1000:.0f} kW",
            f"  GG/Turbine Flow: {result.cycle.turbine_mass_flow.value:.2f} kg/s",
            f"  Tank P (Ox):     {result.cycle.tank_pressure_ox.to('bar').value:.1f} bar",
            f"  Tank P (Fuel):   {result.cycle.tank_pressure_fuel.to('bar').value:.1f} bar",
        ])

    if result.cooling is not None:
        lines.extend([
            "",
            "COOLING:",
            f"  Throat Heat Flux:{result.cooling.throat_heat_flux.value/1e6:.1f} MW/m²",
            f"  Max Wall Temp:   {result.cooling.max_wall_temp.value:.0f} K",
            f"  Allowed Temp:    {result.cooling.max_allowed_temp.value:.0f} K",
            f"  Coolant ΔT:      {result.cooling.coolant_temp_rise.value:.0f} K",
            f"  Flow Margin:     {result.cooling.flow_margin:.2f}x",
            f"  Pressure Drop:   {result.cooling.pressure_drop.to('bar').value:.1f} bar",
        ])

    if result.warnings:
        lines.extend(["", "WARNINGS:"])
        for warning in result.warnings:
            lines.append(f"  ⚠ {warning}")

    lines.append("=" * 70)

    return "\n".join(lines)

