"""Isentropic flow equations for rocket engine analysis.

This module contains the core thermodynamic calculations for rocket engine
performance analysis. All functions are pure (no side effects) and
numba-accelerated for performance.

The equations are based on isentropic flow relations for ideal gases,
which form the foundation of rocket propulsion analysis.

References:
    - Sutton & Biblarz, "Rocket Propulsion Elements", 9th Ed.
    - Huzel & Huang, "Modern Engineering for Design of Liquid-Propellant Rocket Engines"
    - Hill & Peterson, "Mechanics and Thermodynamics of Propulsion", 2nd Ed.
"""

import math

import numba
import numpy as np
from numpy.typing import NDArray

# =============================================================================
# Constants
# =============================================================================

# Standard gravity acceleration
G0_SI: float = 9.80665  # m/s^2
G0_IMP: float = 32.174  # ft/s^2

# Universal gas constant
R_UNIVERSAL_SI: float = 8314.46  # J/(kmol·K)
R_UNIVERSAL_IMP: float = 1545.35  # ft·lbf/(lbmol·R)


# =============================================================================
# Core Isentropic Flow Functions (Numba Accelerated)
# =============================================================================


@numba.njit(cache=True)
def specific_gas_constant(molecular_weight: float) -> float:
    """Calculate specific gas constant from molecular weight.

    Args:
        molecular_weight: Molecular weight of the gas [kg/kmol]

    Returns:
        Specific gas constant R [J/(kg·K)]
    """
    return R_UNIVERSAL_SI / molecular_weight


@numba.njit(cache=True)
def characteristic_velocity(gamma: float, R: float, Tc: float) -> float:
    """Calculate characteristic velocity (c*).

    c* is a measure of the energy available from the combustion process,
    independent of nozzle performance.

    Args:
        gamma: Ratio of specific heats (Cp/Cv) [-]
        R: Specific gas constant [J/(kg·K)]
        Tc: Chamber (stagnation) temperature [K]

    Returns:
        Characteristic velocity [m/s]
    """
    # c* = sqrt(gamma * R * Tc) / (gamma * sqrt((2/(gamma+1))^((gamma+1)/(gamma-1))))
    term1 = math.sqrt(gamma * R * Tc)
    term2 = gamma * math.sqrt((2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0)))
    return term1 / term2


@numba.njit(cache=True)
def thrust_coefficient(
    gamma: float, pe_pc: float, pa_pc: float, expansion_ratio: float
) -> float:
    """Calculate thrust coefficient (Cf).

    Cf characterizes the nozzle's ability to convert thermal energy
    into directed kinetic energy.

    Args:
        gamma: Ratio of specific heats [-]
        pe_pc: Exit pressure / chamber pressure ratio [-]
        pa_pc: Ambient pressure / chamber pressure ratio [-]
        expansion_ratio: Nozzle exit area / throat area (Ae/At) [-]

    Returns:
        Thrust coefficient [-]
    """
    # Momentum thrust term
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    exponent = gm1 / gamma

    term1 = 2.0 * gamma**2 / gm1
    term2 = (2.0 / gp1) ** (gp1 / gm1)
    term3 = 1.0 - pe_pc**exponent

    Cf_momentum = math.sqrt(term1 * term2 * term3)

    # Pressure thrust term
    Cf_pressure = (pe_pc - pa_pc) * expansion_ratio

    return Cf_momentum + Cf_pressure


@numba.njit(cache=True)
def thrust_coefficient_vacuum(gamma: float, pe_pc: float, expansion_ratio: float) -> float:
    """Calculate vacuum thrust coefficient (Cf_vac).

    Args:
        gamma: Ratio of specific heats [-]
        pe_pc: Exit pressure / chamber pressure ratio [-]
        expansion_ratio: Nozzle exit area / throat area (Ae/At) [-]

    Returns:
        Vacuum thrust coefficient [-]
    """
    return thrust_coefficient(gamma, pe_pc, 0.0, expansion_ratio)


@numba.njit(cache=True)
def specific_impulse(cstar: float, Cf: float, g0: float = G0_SI) -> float:
    """Calculate specific impulse (Isp).

    Isp is the key performance metric for rocket engines, representing
    the thrust produced per unit weight flow rate of propellant.

    Args:
        cstar: Characteristic velocity [m/s]
        Cf: Thrust coefficient [-]
        g0: Standard gravity [m/s^2], default 9.80665

    Returns:
        Specific impulse [s]
    """
    return cstar * Cf / g0


@numba.njit(cache=True)
def exhaust_velocity(gamma: float, R: float, Tc: float, pe_pc: float) -> float:
    """Calculate exhaust velocity (ue).

    This is the velocity of the exhaust gases at the nozzle exit
    for isentropic expansion.

    Args:
        gamma: Ratio of specific heats [-]
        R: Specific gas constant [J/(kg·K)]
        Tc: Chamber temperature [K]
        pe_pc: Exit pressure / chamber pressure ratio [-]

    Returns:
        Exhaust velocity [m/s]
    """
    gm1 = gamma - 1.0
    exponent = gm1 / gamma

    term1 = 2.0 * gamma * R * Tc / gm1
    term2 = 1.0 - pe_pc**exponent

    return math.sqrt(term1 * term2)


@numba.njit(cache=True)
def mass_flow_rate(thrust: float, Isp: float, g0: float = G0_SI) -> float:
    """Calculate total mass flow rate from thrust and Isp.

    Args:
        thrust: Engine thrust [N]
        Isp: Specific impulse [s]
        g0: Standard gravity [m/s^2]

    Returns:
        Mass flow rate [kg/s]
    """
    return thrust / (Isp * g0)


@numba.njit(cache=True)
def mass_flow_rate_from_throat(
    pc: float, At: float, gamma: float, R: float, Tc: float
) -> float:
    """Calculate mass flow rate from throat conditions.

    Uses the choked flow condition at the throat.

    Args:
        pc: Chamber pressure [Pa]
        At: Throat area [m^2]
        gamma: Ratio of specific heats [-]
        R: Specific gas constant [J/(kg·K)]
        Tc: Chamber temperature [K]

    Returns:
        Mass flow rate [kg/s]
    """
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0

    term1 = pc * At
    term2 = gamma / (R * Tc)
    term3 = (2.0 / gp1) ** (gp1 / gm1)

    return term1 * math.sqrt(term2 * term3)


@numba.njit(cache=True)
def throat_area(mdot: float, cstar: float, pc: float) -> float:
    """Calculate required throat area.

    Args:
        mdot: Mass flow rate [kg/s]
        cstar: Characteristic velocity [m/s]
        pc: Chamber pressure [Pa]

    Returns:
        Throat area [m^2]
    """
    return mdot * cstar / pc


@numba.njit(cache=True)
def area_from_diameter(diameter: float) -> float:
    """Calculate circular area from diameter.

    Args:
        diameter: Diameter [m]

    Returns:
        Area [m^2]
    """
    return math.pi * (diameter / 2.0) ** 2


@numba.njit(cache=True)
def diameter_from_area(area: float) -> float:
    """Calculate diameter from circular area.

    Args:
        area: Area [m^2]

    Returns:
        Diameter [m]
    """
    return 2.0 * math.sqrt(area / math.pi)


# =============================================================================
# Mach Number Relations
# =============================================================================


@numba.njit(cache=True)
def mach_from_pressure_ratio(pc_p: float, gamma: float) -> float:
    """Calculate Mach number from stagnation-to-static pressure ratio.

    Args:
        pc_p: Chamber (stagnation) pressure / local static pressure [-]
        gamma: Ratio of specific heats [-]

    Returns:
        Mach number [-]
    """
    gm1 = gamma - 1.0
    exponent = gm1 / gamma

    return math.sqrt((2.0 / gm1) * (pc_p**exponent - 1.0))


@numba.njit(cache=True)
def pressure_ratio_from_mach(M: float, gamma: float) -> float:
    """Calculate stagnation-to-static pressure ratio from Mach number.

    Args:
        M: Mach number [-]
        gamma: Ratio of specific heats [-]

    Returns:
        pc/p ratio [-]
    """
    gm1 = gamma - 1.0
    exponent = gamma / gm1

    return (1.0 + gm1 / 2.0 * M**2) ** exponent


@numba.njit(cache=True)
def temperature_ratio_from_mach(M: float, gamma: float) -> float:
    """Calculate stagnation-to-static temperature ratio from Mach number.

    Args:
        M: Mach number [-]
        gamma: Ratio of specific heats [-]

    Returns:
        Tc/T ratio [-]
    """
    return 1.0 + (gamma - 1.0) / 2.0 * M**2


@numba.njit(cache=True)
def area_ratio_from_mach(M: float, gamma: float) -> float:
    """Calculate area ratio (A/A*) from Mach number.

    A* is the critical (sonic) area, i.e., the throat area for choked flow.

    Args:
        M: Mach number [-]
        gamma: Ratio of specific heats [-]

    Returns:
        Area ratio A/A* [-]
    """
    if M <= 0.0:
        return float("inf")

    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    exponent = gp1 / (2.0 * gm1)

    term1 = 1.0 / M
    term2 = (2.0 / gp1) * (1.0 + gm1 / 2.0 * M**2)

    return term1 * term2**exponent


@numba.njit(cache=True)
def mach_from_area_ratio_supersonic(area_ratio: float, gamma: float) -> float:
    """Calculate supersonic Mach number from area ratio using Newton-Raphson.

    For a given A/A* > 1, there are two solutions: subsonic and supersonic.
    This function returns the supersonic solution (M > 1).

    Args:
        area_ratio: Area ratio A/A* [-], must be >= 1
        gamma: Ratio of specific heats [-]

    Returns:
        Supersonic Mach number [-]
    """
    if area_ratio < 1.0:
        return 1.0  # At throat

    # Initial guess for supersonic flow
    M = 2.0 + area_ratio / 5.0

    # Newton-Raphson iteration
    for _ in range(50):
        f = area_ratio_from_mach(M, gamma) - area_ratio

        # Numerical derivative
        dM = 1e-8
        df = (area_ratio_from_mach(M + dM, gamma) - area_ratio_from_mach(M - dM, gamma)) / (
            2.0 * dM
        )

        if abs(df) < 1e-12:
            break

        M_new = M - f / df

        if M_new < 1.0:
            M_new = 1.0 + 0.1

        if abs(M_new - M) < 1e-10:
            break

        M = M_new

    return M


@numba.njit(cache=True)
def mach_from_area_ratio_subsonic(area_ratio: float, gamma: float) -> float:
    """Calculate subsonic Mach number from area ratio using Newton-Raphson.

    For a given A/A* > 1, there are two solutions: subsonic and supersonic.
    This function returns the subsonic solution (M < 1).

    Args:
        area_ratio: Area ratio A/A* [-], must be >= 1
        gamma: Ratio of specific heats [-]

    Returns:
        Subsonic Mach number [-]
    """
    if area_ratio < 1.0:
        return 1.0

    # Initial guess for subsonic flow
    M = 0.5

    # Newton-Raphson iteration
    for _ in range(50):
        f = area_ratio_from_mach(M, gamma) - area_ratio

        # Numerical derivative
        dM = 1e-8
        df = (area_ratio_from_mach(M + dM, gamma) - area_ratio_from_mach(M - dM, gamma)) / (
            2.0 * dM
        )

        if abs(df) < 1e-12:
            break

        M_new = M - f / df

        if M_new > 1.0:
            M_new = 0.99
        if M_new < 0.0:
            M_new = 0.01

        if abs(M_new - M) < 1e-10:
            break

        M = M_new

    return M


# =============================================================================
# Throat and Exit Conditions
# =============================================================================


@numba.njit(cache=True)
def throat_temperature(Tc: float, gamma: float) -> float:
    """Calculate throat (critical) temperature.

    Args:
        Tc: Chamber temperature [K]
        gamma: Ratio of specific heats [-]

    Returns:
        Throat temperature [K]
    """
    return Tc / (1.0 + (gamma - 1.0) / 2.0)


@numba.njit(cache=True)
def throat_pressure(pc: float, gamma: float) -> float:
    """Calculate throat (critical) pressure.

    Args:
        pc: Chamber pressure [Pa]
        gamma: Ratio of specific heats [-]

    Returns:
        Throat pressure [Pa]
    """
    exponent = gamma / (gamma - 1.0)
    return pc * (2.0 / (gamma + 1.0)) ** exponent


@numba.njit(cache=True)
def expansion_ratio_from_pressure_ratio(pc_pe: float, gamma: float) -> float:
    """Calculate nozzle expansion ratio from chamber-to-exit pressure ratio.

    Args:
        pc_pe: Chamber pressure / exit pressure [-]
        gamma: Ratio of specific heats [-]

    Returns:
        Expansion ratio (Ae/At) [-]
    """
    # First get exit Mach number
    Me = mach_from_pressure_ratio(pc_pe, gamma)
    # Then get area ratio
    return area_ratio_from_mach(Me, gamma)


@numba.njit(cache=True)
def exit_pressure_from_expansion_ratio(
    expansion_ratio: float, pc: float, gamma: float
) -> float:
    """Calculate exit pressure from expansion ratio.

    Args:
        expansion_ratio: Nozzle expansion ratio (Ae/At) [-]
        pc: Chamber pressure [Pa]
        gamma: Ratio of specific heats [-]

    Returns:
        Exit pressure [Pa]
    """
    # Get exit Mach number (supersonic solution)
    Me = mach_from_area_ratio_supersonic(expansion_ratio, gamma)
    # Get pressure ratio
    pc_pe = pressure_ratio_from_mach(Me, gamma)
    return pc / pc_pe


# =============================================================================
# Chamber Geometry
# =============================================================================


@numba.njit(cache=True)
def chamber_volume(lstar: float, At: float) -> float:
    """Calculate chamber volume from L* and throat area.

    L* (characteristic length) is defined as the chamber volume divided
    by the throat area: L* = Vc / At

    Args:
        lstar: Characteristic length [m]
        At: Throat area [m^2]

    Returns:
        Chamber volume [m^3]
    """
    return lstar * At


@numba.njit(cache=True)
def cylindrical_chamber_length(
    Vc: float, Ac: float, Rc: float, Rt: float, contraction_angle: float
) -> float:
    """Calculate length of cylindrical section of chamber.

    Accounts for the converging section geometry.

    Args:
        Vc: Chamber volume [m^3]
        Ac: Chamber cross-sectional area [m^2]
        Rc: Chamber radius [m]
        Rt: Throat radius [m]
        contraction_angle: Convergence half-angle [radians]

    Returns:
        Cylindrical section length [m]
    """
    # Volume of converging cone section
    cone_length = (Rc - Rt) / math.tan(contraction_angle)
    # Approximate cylindrical length (subtract converging section contribution)
    return Vc / Ac - 0.5 * cone_length


@numba.njit(cache=True)
def conical_nozzle_length(Rt: float, Re: float, half_angle: float) -> float:
    """Calculate length of a conical nozzle.

    Args:
        Rt: Throat radius [m]
        Re: Exit radius [m]
        half_angle: Nozzle half-angle [radians]

    Returns:
        Nozzle length [m]
    """
    return (Re - Rt) / math.tan(half_angle)


@numba.njit(cache=True)
def bell_nozzle_length(
    Rt: float, Re: float, bell_fraction: float = 0.8, reference_angle: float = 0.2618
) -> float:
    """Calculate length of a bell (parabolic) nozzle.

    Bell nozzles are typically specified as a percentage of the length
    of a 15-degree conical nozzle with the same expansion ratio.

    Args:
        Rt: Throat radius [m]
        Re: Exit radius [m]
        bell_fraction: Length as fraction of 15° cone (e.g., 0.8 for 80% bell)
        reference_angle: Reference cone half-angle [radians], default 15° = 0.2618

    Returns:
        Nozzle length [m]
    """
    conical_length = conical_nozzle_length(Rt, Re, reference_angle)
    return conical_length * bell_fraction


# =============================================================================
# Vectorized Functions for Parametric Studies
# =============================================================================


@numba.njit(cache=True, parallel=True)
def thrust_coefficient_sweep(
    gamma: float,
    pe_pc: float,
    pa_pc_array: NDArray[np.float64],
    expansion_ratio: float,
) -> NDArray[np.float64]:
    """Calculate thrust coefficient for array of ambient pressures.

    Useful for altitude performance analysis.

    Args:
        gamma: Ratio of specific heats [-]
        pe_pc: Exit pressure / chamber pressure ratio [-]
        pa_pc_array: Array of ambient pressure / chamber pressure ratios [-]
        expansion_ratio: Nozzle expansion ratio [-]

    Returns:
        Array of thrust coefficients [-]
    """
    n = len(pa_pc_array)
    result = np.empty(n, dtype=np.float64)

    for i in numba.prange(n):
        result[i] = thrust_coefficient(gamma, pe_pc, pa_pc_array[i], expansion_ratio)

    return result


@numba.njit(cache=True)
def area_ratio_sweep(
    mach_array: NDArray[np.float64], gamma: float
) -> NDArray[np.float64]:
    """Calculate area ratios for array of Mach numbers.

    Args:
        mach_array: Array of Mach numbers [-]
        gamma: Ratio of specific heats [-]

    Returns:
        Array of area ratios [-]
    """
    n = len(mach_array)
    result = np.empty(n, dtype=np.float64)

    for i in range(n):
        result[i] = area_ratio_from_mach(mach_array[i], gamma)

    return result

