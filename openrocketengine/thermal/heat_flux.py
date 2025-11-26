"""Heat flux estimation for rocket engine combustion chambers and nozzles.

This module implements the Bartz correlation and related methods for
estimating convective heat transfer in rocket engines.

The Bartz correlation is the industry-standard method for preliminary
heat flux estimation, derived from turbulent pipe flow correlations
modified for rocket engine conditions.

References:
    - Bartz, D.R., "A Simple Equation for Rapid Estimation of Rocket
      Nozzle Convective Heat Transfer Coefficients", Jet Propulsion, 1957
    - Huzel & Huang, "Modern Engineering for Design of Liquid-Propellant
      Rocket Engines", Chapter 4
"""

import math

import numpy as np
from beartype import beartype

from openrocketengine.engine import EngineGeometry, EngineInputs, EnginePerformance
from openrocketengine.units import Quantity, kelvin


@beartype
def recovery_factor(prandtl: float, laminar: bool = False) -> float:
    """Calculate the recovery factor for adiabatic wall temperature.

    The recovery factor accounts for the difference between the
    stagnation temperature and the actual adiabatic wall temperature
    due to boundary layer effects.

    Args:
        prandtl: Prandtl number of the gas [-]
        laminar: If True, use laminar correlation; else turbulent

    Returns:
        Recovery factor r [-], typically 0.85-0.92 for turbulent flow
    """
    if laminar:
        return math.sqrt(prandtl)  # r = Pr^0.5 for laminar
    else:
        return prandtl ** (1/3)  # r = Pr^(1/3) for turbulent


@beartype
def adiabatic_wall_temperature(
    stagnation_temp: Quantity,
    mach: float,
    gamma: float,
    recovery_factor: float,
) -> Quantity:
    """Calculate adiabatic wall temperature.

    The adiabatic wall temperature is the temperature the wall would
    reach if there were no heat transfer (perfectly insulated wall).
    It's used as the driving temperature difference for heat flux.

    T_aw = T_0 * [1 + r * (gamma-1)/2 * M^2] / [1 + (gamma-1)/2 * M^2]

    Args:
        stagnation_temp: Chamber/stagnation temperature [K]
        mach: Local Mach number [-]
        gamma: Ratio of specific heats [-]
        recovery_factor: Recovery factor r [-]

    Returns:
        Adiabatic wall temperature [K]
    """
    T0 = stagnation_temp.to("K").value
    gm1 = gamma - 1

    # Temperature ratio T/T0 from isentropic relations
    T_ratio = 1 / (1 + gm1/2 * mach**2)

    # Static temperature
    T_static = T0 * T_ratio

    # Adiabatic wall temperature
    # T_aw = T_static + r * (T0 - T_static)
    T_aw = T_static + recovery_factor * (T0 - T_static)

    return kelvin(T_aw)


@beartype
def bartz_heat_flux(
    chamber_pressure: Quantity,
    chamber_temp: Quantity,
    throat_diameter: Quantity,
    local_diameter: Quantity,
    characteristic_velocity: Quantity,
    gamma: float,
    molecular_weight: float,
    local_mach: float,
    wall_temp: Quantity | None = None,
) -> Quantity:
    """Calculate convective heat flux using the Bartz correlation.

    The Bartz equation estimates the convective heat transfer coefficient:

    h = (0.026/D_t^0.2) * (mu^0.2 * cp / Pr^0.6) * (p_c / c*)^0.8 *
        (D_t/R_c)^0.1 * (A_t/A)^0.9 * sigma

    where sigma is a correction factor for property variations.

    Args:
        chamber_pressure: Chamber pressure [Pa]
        chamber_temp: Chamber temperature [K]
        throat_diameter: Throat diameter [m]
        local_diameter: Local diameter at evaluation point [m]
        characteristic_velocity: c* [m/s]
        gamma: Ratio of specific heats [-]
        molecular_weight: Molecular weight [kg/kmol]
        local_mach: Local Mach number [-]
        wall_temp: Wall temperature [K]. If None, estimates at 600K.

    Returns:
        Heat flux [W/m²]
    """
    # Extract values in SI
    pc = chamber_pressure.to("Pa").value
    Tc = chamber_temp.to("K").value
    Dt = throat_diameter.to("m").value
    D = local_diameter.to("m").value
    cstar = characteristic_velocity.to("m/s").value
    MW = molecular_weight

    # Wall temperature (estimate if not provided)
    Tw = wall_temp.to("K").value if wall_temp else 600.0

    # Gas properties
    R_specific = 8314.46 / MW  # J/(kg·K)
    cp = gamma * R_specific / (gamma - 1)  # J/(kg·K)

    # Estimate viscosity using Sutherland's law
    # Reference: air at 273K, mu0 = 1.71e-5 Pa·s
    # For combustion products, use higher reference
    mu_ref = 4e-5  # Pa·s at Tc_ref
    Tc_ref = 3000  # K
    S = 200  # Sutherland constant (approximate for combustion products)

    mu = mu_ref * (Tc / Tc_ref) ** 1.5 * (Tc_ref + S) / (Tc + S)

    # Prandtl number
    # Pr = mu * cp / k, approximate k from cp and Pr ~ 0.7-0.9
    Pr = 0.8  # Typical for combustion products

    # Area ratio
    area_ratio = (D / Dt) ** 2

    # Sigma correction factor for property variation across boundary layer
    # sigma = 1 / [(Tw/Tc * (1 + (gamma-1)/2 * M^2) + 0.5)^0.68 *
    #              (1 + (gamma-1)/2 * M^2)^0.12]
    gm1 = gamma - 1
    temp_factor = Tw / Tc * (1 + gm1/2 * local_mach**2) + 0.5
    sigma = 1 / (temp_factor ** 0.68 * (1 + gm1/2 * local_mach**2) ** 0.12)

    # Bartz correlation for heat transfer coefficient
    # h = (0.026 / Dt^0.2) * (mu^0.2 * cp / Pr^0.6) * (pc/cstar)^0.8 *
    #     (Dt/Dt)^0.1 * (At/A)^0.9 * sigma
    h = (0.026 / Dt**0.2) * (mu**0.2 * cp / Pr**0.6) * \
        (pc / cstar)**0.8 * (1/area_ratio)**0.9 * sigma

    # Calculate adiabatic wall temperature
    r = recovery_factor(Pr)
    T_aw = Tc * (1 + r * gm1/2 * local_mach**2) / (1 + gm1/2 * local_mach**2)

    # Heat flux
    q = h * (T_aw - Tw)

    return Quantity(q, "Pa", "pressure")  # W/m² = Pa·m/s, using Pa as proxy


@beartype
def estimate_heat_flux(
    inputs: EngineInputs,
    performance: EnginePerformance,
    geometry: EngineGeometry,
    location: str = "throat",
    wall_temp: Quantity | None = None,
) -> Quantity:
    """Estimate heat flux at a specific location in the engine.

    Provides a simplified interface to the Bartz correlation.

    Args:
        inputs: Engine input parameters
        performance: Computed engine performance
        geometry: Computed engine geometry
        location: Location to evaluate: "throat", "chamber", or "exit"
        wall_temp: Wall temperature [K]. If None, estimates based on location.

    Returns:
        Heat flux [W/m²]
    """
    # Determine local conditions based on location
    if location == "throat":
        local_diameter = geometry.throat_diameter
        local_mach = 1.0
        default_wall_temp = 700  # K, hot at throat
    elif location == "chamber":
        local_diameter = geometry.chamber_diameter
        local_mach = 0.1  # Low Mach in chamber
        default_wall_temp = 600  # K
    elif location == "exit":
        local_diameter = geometry.exit_diameter
        local_mach = performance.exit_mach
        default_wall_temp = 400  # K, cooler at exit
    else:
        raise ValueError(f"Unknown location: {location}. Use 'throat', 'chamber', or 'exit'")

    if wall_temp is None:
        wall_temp = kelvin(default_wall_temp)

    return bartz_heat_flux(
        chamber_pressure=inputs.chamber_pressure,
        chamber_temp=inputs.chamber_temp,
        throat_diameter=geometry.throat_diameter,
        local_diameter=local_diameter,
        characteristic_velocity=performance.cstar,
        gamma=inputs.gamma,
        molecular_weight=inputs.molecular_weight,
        local_mach=local_mach,
        wall_temp=wall_temp,
    )


@beartype
def heat_flux_profile(
    inputs: EngineInputs,
    performance: EnginePerformance,
    geometry: EngineGeometry,
    n_points: int = 50,
    wall_temp: Quantity | None = None,
) -> tuple[list[float], list[float]]:
    """Calculate heat flux profile along the engine.

    Returns heat flux from chamber through throat to exit.

    Args:
        inputs: Engine input parameters
        performance: Computed engine performance
        geometry: Computed engine geometry
        n_points: Number of points in profile
        wall_temp: Wall temperature (constant along length)

    Returns:
        Tuple of (x_positions, heat_fluxes) where x is normalized (0=chamber, 1=exit)
    """
    # Get key dimensions
    Dc = geometry.chamber_diameter.to("m").value
    Dt = geometry.throat_diameter.to("m").value
    De = geometry.exit_diameter.to("m").value

    # Generate normalized positions
    x_norm = np.linspace(0, 1, n_points)

    # Estimate local diameter and Mach number along engine
    # Simplified: linear convergent, bell divergent
    diameters = []
    machs = []

    for x in x_norm:
        if x < 0.3:
            # Chamber region
            D = Dc
            M = 0.1 + x * 0.3  # Low, slowly increasing
        elif x < 0.5:
            # Convergent section
            frac = (x - 0.3) / 0.2
            D = Dc - frac * (Dc - Dt)
            M = 0.1 + frac * 0.9
        elif x < 0.55:
            # Throat region
            D = Dt
            M = 1.0
        else:
            # Divergent section
            frac = (x - 0.55) / 0.45
            D = Dt + frac * (De - Dt)
            # Simple approximation for supersonic Mach
            M = 1.0 + frac * (performance.exit_mach - 1.0)

        diameters.append(D)
        machs.append(max(0.1, M))

    # Calculate heat flux at each point
    heat_fluxes = []
    for D, M in zip(diameters, machs, strict=True):
        q = bartz_heat_flux(
            chamber_pressure=inputs.chamber_pressure,
            chamber_temp=inputs.chamber_temp,
            throat_diameter=geometry.throat_diameter,
            local_diameter=Quantity(D, "m", "length"),
            characteristic_velocity=performance.cstar,
            gamma=inputs.gamma,
            molecular_weight=inputs.molecular_weight,
            local_mach=M,
            wall_temp=wall_temp or kelvin(600),
        )
        heat_fluxes.append(q.value)

    return list(x_norm), heat_fluxes

