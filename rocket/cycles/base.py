"""Base types for engine cycle analysis.

This module defines the common interfaces and data structures used by
all engine cycle types.
"""

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, runtime_checkable

from beartype import beartype

from rocket.engine import EngineGeometry, EngineInputs, EnginePerformance
from rocket.units import Quantity, pascals


class CycleType(Enum):
    """Engine cycle type enumeration."""

    PRESSURE_FED = auto()
    GAS_GENERATOR = auto()
    EXPANDER = auto()
    STAGED_COMBUSTION = auto()
    FULL_FLOW_STAGED = auto()
    ELECTRIC_PUMP = auto()


@beartype
@dataclass(frozen=True, slots=True)
class CyclePerformance:
    """Performance results from engine cycle analysis.

    Captures the system-level performance including losses from
    turbine drive systems, pump power requirements, and pressure margins.

    Attributes:
        net_isp: Effective Isp after accounting for cycle losses [s]
        net_thrust: Delivered thrust after losses [N]
        cycle_efficiency: Ratio of net Isp to ideal Isp [-]
        pump_power_ox: Oxidizer pump power requirement [W]
        pump_power_fuel: Fuel pump power requirement [W]
        turbine_power: Total turbine power available [W]
        turbine_mass_flow: Mass flow through turbine [kg/s]
        tank_pressure_ox: Required oxidizer tank pressure [Pa]
        tank_pressure_fuel: Required fuel tank pressure [Pa]
        npsh_margin_ox: Net Positive Suction Head margin for ox pump [Pa]
        npsh_margin_fuel: NPSH margin for fuel pump [Pa]
        cycle_type: Type of cycle analyzed
        feasible: Whether the cycle closes (power balance satisfied)
        warnings: List of any warnings or marginal conditions
    """

    net_isp: Quantity
    net_thrust: Quantity
    cycle_efficiency: float
    pump_power_ox: Quantity
    pump_power_fuel: Quantity
    turbine_power: Quantity
    turbine_mass_flow: Quantity
    tank_pressure_ox: Quantity
    tank_pressure_fuel: Quantity
    npsh_margin_ox: Quantity
    npsh_margin_fuel: Quantity
    cycle_type: CycleType
    feasible: bool
    warnings: list[str]


@runtime_checkable
class CycleConfiguration(Protocol):
    """Protocol that all cycle configurations must implement.

    This protocol ensures that any cycle configuration can be used
    with the generic analyze_cycle() function.
    """

    @property
    def cycle_type(self) -> CycleType:
        """Return the cycle type."""
        ...

    def analyze(
        self,
        inputs: EngineInputs,
        performance: EnginePerformance,
        geometry: EngineGeometry,
    ) -> CyclePerformance:
        """Analyze the cycle and return performance results.

        Args:
            inputs: Engine input parameters
            performance: Computed engine performance
            geometry: Computed engine geometry

        Returns:
            CyclePerformance with system-level results
        """
        ...


@beartype
def analyze_cycle(
    inputs: EngineInputs,
    performance: EnginePerformance,
    geometry: EngineGeometry,
    cycle: CycleConfiguration,
) -> CyclePerformance:
    """Analyze an engine cycle configuration.

    This is the main entry point for cycle analysis. It delegates to
    the specific cycle implementation's analyze() method.

    Args:
        inputs: Engine input parameters
        performance: Computed engine performance (from design_engine)
        geometry: Computed engine geometry (from design_engine)
        cycle: Cycle configuration (PressureFedCycle, GasGeneratorCycle, etc.)

    Returns:
        CyclePerformance with net performance and feasibility assessment

    Example:
        >>> inputs = EngineInputs.from_propellants("LOX", "CH4", ...)
        >>> performance, geometry = design_engine(inputs)
        >>> cycle = GasGeneratorCycle(turbine_inlet_temp=kelvin(900), ...)
        >>> result = analyze_cycle(inputs, performance, geometry, cycle)
    """
    return cycle.analyze(inputs, performance, geometry)


# =============================================================================
# Utility Functions
# =============================================================================


@beartype
def pump_power(
    mass_flow: Quantity,
    pressure_rise: Quantity,
    density: float,
    efficiency: float,
) -> Quantity:
    """Calculate pump power requirement.

    Uses the basic hydraulic power equation:
        P = (mdot * delta_P) / (rho * eta)

    Args:
        mass_flow: Mass flow rate through pump [kg/s]
        pressure_rise: Pressure rise across pump [Pa]
        density: Fluid density [kg/m³]
        efficiency: Pump efficiency (0-1)

    Returns:
        Pump power in Watts
    """
    mdot = mass_flow.to("kg/s").value
    dp = pressure_rise.to("Pa").value

    # Volumetric flow rate
    Q = mdot / density  # m³/s

    # Hydraulic power
    P_hydraulic = Q * dp  # W

    # Shaft power (accounting for efficiency)
    P_shaft = P_hydraulic / efficiency

    return Quantity(P_shaft, "W", "power")


@beartype
def turbine_power(
    mass_flow: Quantity,
    inlet_temp: Quantity,
    pressure_ratio: float,
    gamma: float,
    efficiency: float,
    R: float = 287.0,  # J/(kg·K), approximate for combustion products
) -> Quantity:
    """Calculate turbine power output.

    Uses isentropic turbine equations with efficiency factor.

    Args:
        mass_flow: Mass flow through turbine [kg/s]
        inlet_temp: Turbine inlet temperature [K]
        pressure_ratio: Inlet pressure / outlet pressure [-]
        gamma: Ratio of specific heats for turbine gas [-]
        efficiency: Turbine isentropic efficiency (0-1)
        R: Specific gas constant [J/(kg·K)]

    Returns:
        Turbine power output in Watts
    """
    mdot = mass_flow.to("kg/s").value
    T_in = inlet_temp.to("K").value

    # Isentropic temperature ratio
    T_ratio_ideal = pressure_ratio ** ((gamma - 1) / gamma)

    # Actual temperature drop
    delta_T_ideal = T_in * (1 - 1 / T_ratio_ideal)
    delta_T_actual = delta_T_ideal * efficiency

    # Specific heat at constant pressure
    cp = gamma * R / (gamma - 1)

    # Turbine power
    P = mdot * cp * delta_T_actual

    return Quantity(P, "W", "power")


@beartype
def npsh_available(
    tank_pressure: Quantity,
    fluid_density: float,
    vapor_pressure: Quantity,
    inlet_height: float = 0.0,
    line_losses: Quantity | None = None,
) -> Quantity:
    """Calculate Net Positive Suction Head available at pump inlet.

    NPSH_a = (P_tank - P_vapor) / (rho * g) + h - losses

    Args:
        tank_pressure: Tank ullage pressure [Pa]
        fluid_density: Propellant density [kg/m³]
        vapor_pressure: Propellant vapor pressure [Pa]
        inlet_height: Height of fluid above pump inlet [m]
        line_losses: Pressure losses in feed lines [Pa]

    Returns:
        NPSH available in Pascals (pressure equivalent)
    """
    g = 9.80665  # m/s²

    P_tank = tank_pressure.to("Pa").value
    P_vapor = vapor_pressure.to("Pa").value
    losses = line_losses.to("Pa").value if line_losses else 0.0

    # NPSH in meters of head
    npsh_m = (P_tank - P_vapor) / (fluid_density * g) + inlet_height - losses / (fluid_density * g)

    # Convert to pressure equivalent
    npsh_pa = npsh_m * fluid_density * g

    return pascals(npsh_pa)


@beartype
def estimate_line_losses(
    mass_flow: Quantity,
    density: float,
    pipe_diameter: float,
    pipe_length: float,
    num_elbows: int = 2,
    num_valves: int = 2,
) -> Quantity:
    """Estimate pressure losses in feed lines.

    Uses Darcy-Weisbach equation with loss coefficients for fittings.

    Args:
        mass_flow: Mass flow rate [kg/s]
        density: Fluid density [kg/m³]
        pipe_diameter: Pipe inner diameter [m]
        pipe_length: Total pipe length [m]
        num_elbows: Number of 90° elbows
        num_valves: Number of valves

    Returns:
        Total pressure loss [Pa]
    """
    mdot = mass_flow.to("kg/s").value
    D = pipe_diameter
    L = pipe_length

    # Calculate velocity
    A = math.pi * (D / 2) ** 2
    V = mdot / (density * A)

    # Dynamic pressure
    q = 0.5 * density * V ** 2

    # Friction factor (assuming turbulent flow, smooth pipe)
    # Using Blasius correlation as approximation
    Re = density * V * D / 1e-3  # Approximate viscosity
    f = 0.316 / Re ** 0.25 if Re > 2300 else 64 / Re

    # Pipe friction losses
    dp_pipe = f * (L / D) * q

    # Fitting losses (K-factors)
    K_elbow = 0.3  # 90° elbow
    K_valve = 0.2  # Gate valve (open)

    dp_fittings = (num_elbows * K_elbow + num_valves * K_valve) * q

    return pascals(dp_pipe + dp_fittings)


@beartype
def format_cycle_summary(result: CyclePerformance) -> str:
    """Format cycle analysis results as readable string.

    Args:
        result: CyclePerformance from analyze_cycle()

    Returns:
        Formatted multi-line string
    """
    status = "✓ FEASIBLE" if result.feasible else "✗ INFEASIBLE"

    lines = [
        f"{'=' * 60}",
        f"CYCLE ANALYSIS: {result.cycle_type.name}",
        f"Status: {status}",
        f"{'=' * 60}",
        "",
        "PERFORMANCE:",
        f"  Net Isp:           {result.net_isp.value:.1f} s",
        f"  Net Thrust:        {result.net_thrust.to('kN').value:.2f} kN",
        f"  Cycle Efficiency:  {result.cycle_efficiency * 100:.1f}%",
        "",
        "POWER BALANCE:",
        f"  Turbine Power:     {result.turbine_power.value / 1000:.1f} kW",
        f"  Pump Power (Ox):   {result.pump_power_ox.value / 1000:.1f} kW",
        f"  Pump Power (Fuel): {result.pump_power_fuel.value / 1000:.1f} kW",
        f"  Turbine Flow:      {result.turbine_mass_flow.value:.3f} kg/s",
        "",
        "TANK REQUIREMENTS:",
        f"  Ox Tank Pressure:  {result.tank_pressure_ox.to('bar').value:.1f} bar",
        f"  Fuel Tank Pressure:{result.tank_pressure_fuel.to('bar').value:.1f} bar",
        "",
        "NPSH MARGINS:",
        f"  Ox NPSH Margin:    {result.npsh_margin_ox.to('bar').value:.2f} bar",
        f"  Fuel NPSH Margin:  {result.npsh_margin_fuel.to('bar').value:.2f} bar",
    ]

    if result.warnings:
        lines.extend(["", "WARNINGS:"])
        for warning in result.warnings:
            lines.append(f"  ⚠ {warning}")

    lines.append(f"{'=' * 60}")

    return "\n".join(lines)

