"""Propellant thermochemistry module for Rocket.

This module provides combustion thermochemistry calculations using NASA CEA
via RocketCEA. It computes chamber temperature, molecular weight, gamma,
and other properties needed for rocket engine performance analysis.

Example:
    >>> from rocket.propellants import get_combustion_properties
    >>> props = get_combustion_properties(
    ...     oxidizer="LOX",
    ...     fuel="RP1",
    ...     mixture_ratio=2.7,
    ...     chamber_pressure_pa=7e6,
    ... )
    >>> print(f"Tc = {props.chamber_temp_k:.0f} K")
"""

from dataclasses import dataclass
from typing import Literal

from beartype import beartype
from rocketcea.cea_obj import CEA_Obj


# =============================================================================
# Data Structures
# =============================================================================


@beartype
@dataclass(frozen=True, slots=True)
class CombustionProperties:
    """Thermochemical properties from combustion analysis.

    These properties are needed to compute rocket engine performance
    using isentropic flow equations.

    Attributes:
        chamber_temp_k: Adiabatic flame temperature in chamber [K]
        molecular_weight: Mean molecular weight of combustion products [kg/kmol]
        gamma: Ratio of specific heats (Cp/Cv) [-]
        specific_heat_cp: Specific heat at constant pressure [J/(kg·K)]
        characteristic_velocity: Theoretical c* [m/s]
        oxidizer: Oxidizer name
        fuel: Fuel name
        mixture_ratio: Oxidizer-to-fuel mass ratio [-]
        chamber_pressure_pa: Chamber pressure [Pa]
        source: Data source ("rocketcea" or "database")
    """

    chamber_temp_k: float | int
    molecular_weight: float | int
    gamma: float | int
    specific_heat_cp: float | int
    characteristic_velocity: float | int
    oxidizer: str
    fuel: str
    mixture_ratio: float | int
    chamber_pressure_pa: float | int
    source: str


# =============================================================================
# Propellant Name Mapping
# =============================================================================

# Map common names to RocketCEA names
OXIDIZER_NAMES: dict[str, str] = {
    "LOX": "LOX",
    "LO2": "LOX",
    "O2": "LOX",
    "OXYGEN": "LOX",
    "N2O4": "N2O4",
    "NTO": "N2O4",
    "N2O": "N2O",
    "NITROUS": "N2O",
    "NITROUSOXIDE": "N2O",
    "H2O2": "H2O2",
    "HTP": "H2O2",
    "PEROXIDE": "H2O2",
    "MON25": "MON25",
    "MON3": "MON3",
    "IRFNA": "IRFNA",
    "RFNA": "IRFNA",
    "CLF5": "CLF5",
    "F2": "F2",
    "FLUORINE": "F2",
}

FUEL_NAMES: dict[str, str] = {
    "LH2": "LH2",
    "H2": "LH2",
    "HYDROGEN": "LH2",
    "RP1": "RP1",
    "RP-1": "RP1",
    "KEROSENE": "RP1",
    "JET-A": "Jet-A",
    "JETA": "Jet-A",
    "CH4": "CH4",
    "METHANE": "CH4",
    "LCH4": "CH4",
    "C2H5OH": "Ethanol",
    "ETHANOL": "Ethanol",
    "C3H8O": "IPA",
    "IPA": "IPA",
    "ISOPROPANOL": "IPA",
    "MMH": "MMH",
    "UDMH": "UDMH",
    "N2H4": "N2H4",
    "HYDRAZINE": "N2H4",
    "A50": "A-50",
    "A-50": "A-50",
    "AEROZINE50": "A-50",
}


def _normalize_propellant_name(name: str, is_oxidizer: bool) -> str:
    """Normalize propellant name to RocketCEA format."""
    normalized = name.upper().replace(" ", "").replace("-", "")
    lookup = OXIDIZER_NAMES if is_oxidizer else FUEL_NAMES

    if normalized in lookup:
        return lookup[normalized]

    # Try original name (RocketCEA might accept it)
    return name


# =============================================================================
# RocketCEA Integration
# =============================================================================


def _get_properties_from_cea(
    oxidizer: str,
    fuel: str,
    mixture_ratio: float,
    chamber_pressure_pa: float,
) -> CombustionProperties:
    """Get combustion properties using RocketCEA."""

    # Convert pressure to psia (RocketCEA default)
    pc_psia = chamber_pressure_pa / 6894.76

    # Normalize propellant names
    ox_name = _normalize_propellant_name(oxidizer, is_oxidizer=True)
    fuel_name = _normalize_propellant_name(fuel, is_oxidizer=False)

    # Create CEA object
    cea = CEA_Obj(oxName=ox_name, fuelName=fuel_name)

    # Get chamber properties
    # Note: RocketCEA returns (Mw, gamma) from get_Chamber_MolWt_gamma
    Tc = cea.get_Tcomb(Pc=pc_psia, MR=mixture_ratio)  # Chamber temp in R
    Tc_K = Tc * 5 / 9  # Convert Rankine to Kelvin

    mw_gamma = cea.get_Chamber_MolWt_gamma(Pc=pc_psia, MR=mixture_ratio, eps=1.0)
    MW = mw_gamma[0]  # Molecular weight
    gamma = mw_gamma[1]  # Gamma

    # Get c* in ft/s, convert to m/s
    cstar_fts = cea.get_Cstar(Pc=pc_psia, MR=mixture_ratio)
    cstar_ms = cstar_fts * 0.3048

    # Calculate Cp from gamma and MW
    R_universal = 8314.46  # J/(kmol·K)
    R_specific = R_universal / MW  # J/(kg·K)
    Cp = gamma * R_specific / (gamma - 1)

    return CombustionProperties(
        chamber_temp_k=Tc_K,
        molecular_weight=MW,
        gamma=gamma,
        specific_heat_cp=Cp,
        characteristic_velocity=cstar_ms,
        oxidizer=oxidizer,
        fuel=fuel,
        mixture_ratio=mixture_ratio,
        chamber_pressure_pa=chamber_pressure_pa,
        source="rocketcea",
    )


# =============================================================================
# Public API
# =============================================================================


@beartype
def get_combustion_properties(
    oxidizer: str,
    fuel: str,
    mixture_ratio: float,
    chamber_pressure_pa: float,
) -> CombustionProperties:
    """Get combustion thermochemistry properties for a propellant combination.

    This function returns the thermochemical properties needed for rocket engine
    performance calculations using NASA CEA via RocketCEA.

    Args:
        oxidizer: Oxidizer name (e.g., "LOX", "N2O4", "N2O", "H2O2")
        fuel: Fuel name (e.g., "RP1", "LH2", "CH4", "Ethanol", "MMH")
        mixture_ratio: Oxidizer-to-fuel mass ratio (O/F)
        chamber_pressure_pa: Chamber pressure in Pascals

    Returns:
        CombustionProperties containing Tc, MW, gamma, Cp, c*

    Example:
        >>> props = get_combustion_properties(
        ...     oxidizer="LOX",
        ...     fuel="RP1",
        ...     mixture_ratio=2.7,
        ...     chamber_pressure_pa=7e6,
        ... )
        >>> print(f"Tc = {props.chamber_temp_k:.0f} K, gamma = {props.gamma:.3f}")
    """
    return _get_properties_from_cea(oxidizer, fuel, mixture_ratio, chamber_pressure_pa)


@beartype
def is_cea_available() -> bool:
    """Check if RocketCEA is installed and available.

    Returns:
        Always True (RocketCEA is a required dependency)
    """
    return True


@beartype
def get_optimal_mixture_ratio(
    oxidizer: str,
    fuel: str,
    chamber_pressure_pa: float,
    expansion_ratio: float = 40.0,
    metric: Literal["isp", "cstar", "density_isp"] = "isp",
) -> tuple[float, float]:
    """Find the optimal mixture ratio for maximum performance.

    Searches for the mixture ratio that maximizes the specified metric.

    Args:
        oxidizer: Oxidizer name
        fuel: Fuel name
        chamber_pressure_pa: Chamber pressure in Pascals
        expansion_ratio: Nozzle expansion ratio for Isp calculation
        metric: Optimization target:
            - "isp": Maximize specific impulse
            - "cstar": Maximize characteristic velocity
            - "density_isp": Maximize density * Isp (important for volume-limited vehicles)

    Returns:
        Tuple of (optimal_mixture_ratio, maximum_metric_value)
    """
    pc_psia = chamber_pressure_pa / 6894.76
    ox_name = _normalize_propellant_name(oxidizer, is_oxidizer=True)
    fuel_name = _normalize_propellant_name(fuel, is_oxidizer=False)

    cea = CEA_Obj(oxName=ox_name, fuelName=fuel_name)

    # Search over mixture ratios
    best_mr = 1.0
    best_value = 0.0

    # Determine search range based on propellant type
    if ox_name == "LOX" and fuel_name == "LH2":
        mr_range = [x / 10 for x in range(30, 90, 2)]  # 3.0 to 9.0
    elif ox_name == "LOX":
        mr_range = [x / 10 for x in range(15, 40, 2)]  # 1.5 to 4.0
    else:
        mr_range = [x / 10 for x in range(10, 50, 2)]  # 1.0 to 5.0

    for mr in mr_range:
        try:
            if metric == "isp":
                value = cea.get_Isp(Pc=pc_psia, MR=mr, eps=expansion_ratio)
            elif metric == "cstar":
                value = cea.get_Cstar(Pc=pc_psia, MR=mr)
            elif metric == "density_isp":
                isp = cea.get_Isp(Pc=pc_psia, MR=mr, eps=expansion_ratio)
                # Approximate density Isp (would need propellant densities for accuracy)
                value = isp  # Simplified - use Isp as proxy

            if value > best_value:
                best_value = value
                best_mr = mr
        except Exception:
            continue

    return best_mr, best_value

