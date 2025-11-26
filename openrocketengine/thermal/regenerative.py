"""Regenerative cooling feasibility analysis.

Regenerative cooling uses the fuel (or oxidizer) as a coolant, flowing
through channels in the chamber/nozzle wall before injection. This is
the most common cooling method for high-performance rocket engines.

This module provides screening-level analysis to determine if a given
engine design can be regeneratively cooled within material limits.

Key considerations:
- Heat flux at the throat (highest)
- Coolant heat capacity and flow rate
- Wall material temperature limits
- Coolant-side pressure drop

References:
    - Huzel & Huang, Chapter 4
    - Sutton & Biblarz, Chapter 8
"""

import math
from dataclasses import dataclass

from beartype import beartype

from openrocketengine.engine import EngineGeometry, EngineInputs, EnginePerformance
from openrocketengine.thermal.heat_flux import estimate_heat_flux
from openrocketengine.units import Quantity, kelvin, pascals


# =============================================================================
# Coolant Properties Database
# =============================================================================


@beartype
@dataclass(frozen=True, slots=True)
class CoolantProperties:
    """Thermophysical properties of a coolant.

    Properties are approximate values at typical operating conditions.
    For detailed design, use property tables or CoolProp.

    Attributes:
        name: Coolant name
        density: Liquid density [kg/m³]
        specific_heat: Specific heat capacity [J/(kg·K)]
        thermal_conductivity: Thermal conductivity [W/(m·K)]
        viscosity: Dynamic viscosity [Pa·s]
        boiling_point: Boiling point at 1 atm [K]
        max_temp: Maximum recommended temperature before decomposition [K]
    """

    name: str
    density: float
    specific_heat: float
    thermal_conductivity: float
    viscosity: float
    boiling_point: float
    max_temp: float


# Coolant property database
# Values are approximate at typical inlet conditions
COOLANT_DATABASE: dict[str, CoolantProperties] = {
    "RP1": CoolantProperties(
        name="RP-1 (Kerosene)",
        density=810.0,
        specific_heat=2000.0,
        thermal_conductivity=0.12,
        viscosity=0.0015,
        boiling_point=490.0,
        max_temp=600.0,  # Coking limit
    ),
    "CH4": CoolantProperties(
        name="Liquid Methane",
        density=422.0,
        specific_heat=3500.0,
        thermal_conductivity=0.19,
        viscosity=0.00012,
        boiling_point=111.0,
        max_temp=500.0,  # Before significant decomposition
    ),
    "LH2": CoolantProperties(
        name="Liquid Hydrogen",
        density=70.8,
        specific_heat=14300.0,
        thermal_conductivity=0.10,
        viscosity=0.000013,
        boiling_point=20.0,
        max_temp=300.0,  # Stays liquid/supercritical
    ),
    "Ethanol": CoolantProperties(
        name="Ethanol",
        density=789.0,
        specific_heat=2440.0,
        thermal_conductivity=0.17,
        viscosity=0.0011,
        boiling_point=351.0,
        max_temp=500.0,
    ),
    "N2O4": CoolantProperties(
        name="Nitrogen Tetroxide",
        density=1450.0,
        specific_heat=1560.0,
        thermal_conductivity=0.12,
        viscosity=0.0004,
        boiling_point=294.0,
        max_temp=400.0,
    ),
    "MMH": CoolantProperties(
        name="Monomethylhydrazine",
        density=878.0,
        specific_heat=2920.0,
        thermal_conductivity=0.22,
        viscosity=0.0008,
        boiling_point=360.0,
        max_temp=450.0,
    ),
}


@beartype
def get_coolant_properties(coolant: str) -> CoolantProperties:
    """Get properties for a coolant.

    Args:
        coolant: Coolant name (e.g., "RP1", "CH4", "LH2")

    Returns:
        CoolantProperties for the coolant

    Raises:
        ValueError: If coolant not found in database
    """
    # Normalize name
    name = coolant.upper().replace("-", "").replace(" ", "")

    for key, props in COOLANT_DATABASE.items():
        if key.upper() == name:
            return props

    available = list(COOLANT_DATABASE.keys())
    raise ValueError(f"Unknown coolant '{coolant}'. Available: {available}")


# =============================================================================
# Cooling Feasibility Analysis
# =============================================================================


@beartype
@dataclass(frozen=True, slots=True)
class CoolingFeasibility:
    """Results of regenerative cooling feasibility analysis.

    Attributes:
        feasible: Whether cooling is feasible within constraints
        max_wall_temp: Maximum predicted wall temperature [K]
        max_allowed_temp: Maximum allowed wall temperature [K]
        coolant_temp_rise: Temperature rise of coolant [K]
        coolant_outlet_temp: Predicted coolant outlet temperature [K]
        throat_heat_flux: Heat flux at throat [W/m²]
        total_heat_load: Total heat load to coolant [W]
        required_coolant_flow: Minimum required coolant flow [kg/s]
        available_coolant_flow: Available coolant flow (fuel or ox) [kg/s]
        flow_margin: Ratio of available/required flow [-]
        pressure_drop: Estimated coolant-side pressure drop [Pa]
        channel_velocity: Estimated coolant velocity in channels [m/s]
        warnings: List of warnings or concerns
    """

    feasible: bool
    max_wall_temp: Quantity
    max_allowed_temp: Quantity
    coolant_temp_rise: Quantity
    coolant_outlet_temp: Quantity
    throat_heat_flux: Quantity
    total_heat_load: Quantity
    required_coolant_flow: Quantity
    available_coolant_flow: Quantity
    flow_margin: float
    pressure_drop: Quantity
    channel_velocity: float
    warnings: list[str]


@beartype
def check_cooling_feasibility(
    inputs: EngineInputs,
    performance: EnginePerformance,
    geometry: EngineGeometry,
    coolant: str,
    coolant_inlet_temp: Quantity | None = None,
    max_wall_temp: Quantity | None = None,
    wall_material: str = "copper_alloy",
    num_channels: int | None = None,
    channel_aspect_ratio: float = 3.0,
) -> CoolingFeasibility:
    """Check if regenerative cooling is feasible for this engine design.

    This is a screening-level analysis that estimates whether the engine
    can be cooled within material limits using the available coolant flow.

    Args:
        inputs: Engine input parameters
        performance: Computed engine performance
        geometry: Computed engine geometry
        coolant: Coolant name (typically the fuel: "RP1", "CH4", "LH2")
        coolant_inlet_temp: Coolant inlet temperature [K]. Defaults to storage temp.
        max_wall_temp: Maximum allowed wall temperature [K]. Defaults based on material.
        wall_material: Wall material for thermal limits
        num_channels: Number of cooling channels. If None, estimated.
        channel_aspect_ratio: Channel height/width ratio

    Returns:
        CoolingFeasibility assessment
    """
    warnings: list[str] = []

    # Get coolant properties
    try:
        coolant_props = get_coolant_properties(coolant)
    except ValueError:
        # Use generic properties if unknown
        coolant_props = CoolantProperties(
            name=coolant,
            density=800.0,
            specific_heat=2000.0,
            thermal_conductivity=0.15,
            viscosity=0.001,
            boiling_point=300.0,
            max_temp=500.0,
        )
        warnings.append(f"Unknown coolant '{coolant}', using generic properties")

    # Set default inlet temperature (storage temperature)
    if coolant_inlet_temp is None:
        # Cryogenic propellants near boiling point, storables at ~300K
        if coolant.upper() in ["LH2", "CH4", "LOX"]:
            T_inlet = coolant_props.boiling_point + 10  # Just above boiling
        else:
            T_inlet = 300.0  # Room temperature
    else:
        T_inlet = coolant_inlet_temp.to("K").value

    # Set default max wall temperature based on material
    if max_wall_temp is None:
        wall_temps = {
            "copper_alloy": 800,     # GRCop-84, NARloy-Z
            "nickel_alloy": 1000,    # Inconel 718
            "steel": 700,            # Stainless steel
            "niobium": 1500,         # Refractory metal
        }
        T_wall_max = wall_temps.get(wall_material, 800)
    else:
        T_wall_max = max_wall_temp.to("K").value

    # Estimate heat flux at throat (worst case)
    q_throat = estimate_heat_flux(
        inputs, performance, geometry,
        location="throat",
        wall_temp=kelvin(T_wall_max * 0.9),  # Assume wall near limit
    )

    # Also get chamber and exit heat fluxes for total heat load
    q_chamber = estimate_heat_flux(inputs, performance, geometry, location="chamber")
    q_exit = estimate_heat_flux(inputs, performance, geometry, location="exit")

    # Estimate total heat load
    # Simplified: use average heat flux × total surface area
    Dt = geometry.throat_diameter.to("m").value
    Dc = geometry.chamber_diameter.to("m").value
    De = geometry.exit_diameter.to("m").value
    Lc = geometry.chamber_length.to("m").value
    Ln = geometry.nozzle_length.to("m").value

    # Surface areas (approximate)
    A_chamber = math.pi * Dc * Lc
    A_convergent = math.pi * (Dc + Dt) / 2 * (Dc - Dt) / (2 * math.tan(math.radians(45))) * 0.5
    A_throat = math.pi * Dt * Dt * 0.1  # Small throat region
    A_divergent = math.pi * (Dt + De) / 2 * Ln * 0.7  # Approximate bell surface

    # Average heat fluxes for each region (throat is highest)
    q_avg_chamber = q_chamber.value * 0.3  # Lower than throat
    q_avg_convergent = (q_chamber.value + q_throat.value) / 2
    q_avg_throat = q_throat.value
    q_avg_divergent = (q_throat.value + q_exit.value) / 2

    # Total heat load
    Q_total = (
        q_avg_chamber * A_chamber +
        q_avg_convergent * A_convergent +
        q_avg_throat * A_throat +
        q_avg_divergent * A_divergent
    )

    # Available coolant flow (assume fuel is coolant)
    mdot_coolant_available = performance.mdot_fuel.to("kg/s").value

    # Required coolant flow to absorb heat without exceeding temperature limit
    # Q = mdot * cp * delta_T
    # delta_T_max = T_coolant_max - T_inlet
    T_coolant_max = min(coolant_props.max_temp, T_wall_max - 100)  # Stay below wall
    delta_T_max = T_coolant_max - T_inlet

    if delta_T_max <= 0:
        warnings.append("Coolant inlet temperature exceeds maximum allowable")
        delta_T_max = 100  # Use minimum for calculation

    mdot_coolant_required = Q_total / (coolant_props.specific_heat * delta_T_max)

    # Flow margin
    flow_margin = mdot_coolant_available / mdot_coolant_required if mdot_coolant_required > 0 else float('inf')

    # Actual temperature rise with available flow
    if mdot_coolant_available > 0:
        delta_T_actual = Q_total / (mdot_coolant_available * coolant_props.specific_heat)
    else:
        delta_T_actual = float('inf')

    T_coolant_out = T_inlet + delta_T_actual

    # Estimate wall temperature
    # T_wall = T_coolant + Q/(h_coolant * A)
    # For screening, use correlation: T_wall ~ T_coolant + q * (t_wall / k_wall + 1/h_coolant)
    # Simplified: wall runs ~100-200K above coolant
    T_wall_estimate = T_coolant_out + 150  # K above coolant

    # Pressure drop estimation
    # Number of channels (estimate if not provided)
    if num_channels is None:
        # Roughly 1 channel per mm of circumference at throat
        num_channels = max(20, int(math.pi * Dt * 1000))

    # Channel dimensions (rough estimate)
    channel_width = (math.pi * Dt) / num_channels * 0.6  # 60% channel, 40% rib
    channel_height = channel_width * channel_aspect_ratio
    channel_area = channel_width * channel_height

    # Coolant velocity
    total_channel_area = num_channels * channel_area
    v_coolant = mdot_coolant_available / (coolant_props.density * total_channel_area)

    # Pressure drop (Darcy-Weisbach approximation)
    # ΔP = f * (L/D_h) * (ρ * v²/2)
    D_h = 4 * channel_area / (2 * (channel_width + channel_height))  # Hydraulic diameter
    Re = coolant_props.density * v_coolant * D_h / coolant_props.viscosity
    f = 0.316 / Re**0.25 if Re > 2300 else 64 / max(Re, 100)  # Friction factor

    L_total = Lc + Ln  # Total cooled length
    dp = f * (L_total / D_h) * (coolant_props.density * v_coolant**2 / 2)

    # Add losses for bends, manifolds
    dp *= 1.5

    # Feasibility assessment
    feasible = True
    
    if T_wall_estimate > T_wall_max:
        feasible = False
        warnings.append(
            f"Estimated wall temp {T_wall_estimate:.0f}K exceeds limit {T_wall_max:.0f}K"
        )

    if flow_margin < 1.0:
        feasible = False
        warnings.append(
            f"Insufficient coolant flow: need {mdot_coolant_required:.2f} kg/s, "
            f"have {mdot_coolant_available:.2f} kg/s"
        )

    if T_coolant_out > coolant_props.max_temp:
        warnings.append(
            f"Coolant outlet temp {T_coolant_out:.0f}K exceeds max {coolant_props.max_temp:.0f}K"
        )

    if v_coolant > 50:
        warnings.append(f"High coolant velocity {v_coolant:.1f} m/s may cause erosion")

    if dp > 5e6:
        warnings.append(f"High pressure drop {dp/1e6:.1f} MPa")

    # Heat flux at throat check
    if q_throat.value > 50e6:  # > 50 MW/m²
        warnings.append(
            f"Very high throat heat flux {q_throat.value/1e6:.1f} MW/m² - "
            "film cooling may be needed"
        )

    return CoolingFeasibility(
        feasible=feasible,
        max_wall_temp=kelvin(T_wall_estimate),
        max_allowed_temp=kelvin(T_wall_max),
        coolant_temp_rise=kelvin(delta_T_actual),
        coolant_outlet_temp=kelvin(T_coolant_out),
        throat_heat_flux=q_throat,
        total_heat_load=Quantity(Q_total, "N", "force"),  # W, using N as proxy
        required_coolant_flow=Quantity(mdot_coolant_required, "kg/s", "mass_flow"),
        available_coolant_flow=Quantity(mdot_coolant_available, "kg/s", "mass_flow"),
        flow_margin=flow_margin,
        pressure_drop=pascals(dp),
        channel_velocity=v_coolant,
        warnings=warnings,
    )


@beartype
def format_cooling_summary(result: CoolingFeasibility) -> str:
    """Format cooling feasibility results as readable string.

    Args:
        result: CoolingFeasibility from check_cooling_feasibility()

    Returns:
        Formatted multi-line string
    """
    status = "✓ FEASIBLE" if result.feasible else "✗ NOT FEASIBLE"

    lines = [
        f"{'=' * 60}",
        f"REGENERATIVE COOLING ANALYSIS",
        f"Status: {status}",
        f"{'=' * 60}",
        "",
        "THERMAL:",
        f"  Throat Heat Flux:      {result.throat_heat_flux.value/1e6:.1f} MW/m²",
        f"  Total Heat Load:       {result.total_heat_load.value/1e6:.2f} MW",
        f"  Max Wall Temp:         {result.max_wall_temp.value:.0f} K",
        f"  Allowed Wall Temp:     {result.max_allowed_temp.value:.0f} K",
        "",
        "COOLANT:",
        f"  Temperature Rise:      {result.coolant_temp_rise.value:.0f} K",
        f"  Outlet Temperature:    {result.coolant_outlet_temp.value:.0f} K",
        f"  Required Flow:         {result.required_coolant_flow.value:.2f} kg/s",
        f"  Available Flow:        {result.available_coolant_flow.value:.2f} kg/s",
        f"  Flow Margin:           {result.flow_margin:.2f}x",
        "",
        "HYDRAULICS:",
        f"  Channel Velocity:      {result.channel_velocity:.1f} m/s",
        f"  Pressure Drop:         {result.pressure_drop.to('bar').value:.1f} bar",
    ]

    if result.warnings:
        lines.extend(["", "WARNINGS:"])
        for warning in result.warnings:
            lines.append(f"  ⚠ {warning}")

    lines.append(f"{'=' * 60}")

    return "\n".join(lines)

