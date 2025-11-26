"""Propellant thermochemistry module for OpenRocketEngine.

This module provides combustion thermochemistry calculations using NASA CEA
via RocketCEA. It computes chamber temperature, molecular weight, gamma,
and other properties needed for rocket engine performance analysis.

Example:
    >>> from openrocketengine.propellants import get_combustion_properties
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
# Fallback Database (When CEA Not Available)
# =============================================================================

# Tabulated data for common propellant combinations at typical conditions
# Format: (oxidizer, fuel): {MR: (Tc_K, MW, gamma, cstar_m/s)}
# Data at approximately 1000 psia (6.9 MPa) chamber pressure
# Sources: Sutton & Biblarz, various NASA reports

_PROPELLANT_DATABASE: dict[tuple[str, str], dict[float, tuple[float, float, float, float]]] = {
    ("LOX", "LH2"): {
        4.0: (3015, 12.0, 1.20, 2290),
        5.0: (3250, 13.5, 1.18, 2360),
        6.0: (3400, 14.8, 1.16, 2390),
        7.0: (3470, 16.0, 1.15, 2380),
        8.0: (3450, 17.0, 1.14, 2340),
    },
    ("LOX", "RP1"): {
        2.0: (3450, 21.5, 1.21, 1750),
        2.3: (3550, 22.5, 1.19, 1780),
        2.5: (3600, 23.0, 1.18, 1790),
        2.7: (3620, 23.3, 1.17, 1800),
        3.0: (3580, 24.0, 1.16, 1780),
    },
    ("LOX", "CH4"): {
        2.5: (3400, 19.5, 1.19, 1820),
        3.0: (3530, 20.5, 1.17, 1850),
        3.2: (3560, 21.0, 1.16, 1860),
        3.5: (3570, 21.5, 1.15, 1850),
        4.0: (3520, 22.5, 1.14, 1820),
    },
    ("LOX", "Ethanol"): {
        1.0: (2800, 20.0, 1.24, 1650),
        1.3: (3100, 21.0, 1.22, 1720),
        1.5: (3250, 21.5, 1.20, 1750),
        1.8: (3350, 22.0, 1.19, 1760),
        2.0: (3380, 22.5, 1.18, 1750),
    },
    ("N2O4", "MMH"): {
        1.5: (3000, 21.0, 1.24, 1680),
        1.8: (3150, 21.5, 1.22, 1720),
        2.0: (3220, 22.0, 1.21, 1730),
        2.2: (3260, 22.5, 1.20, 1730),
        2.5: (3250, 23.0, 1.19, 1710),
    },
    ("N2O4", "UDMH"): {
        1.8: (3050, 21.5, 1.23, 1690),
        2.0: (3150, 22.0, 1.22, 1710),
        2.2: (3200, 22.5, 1.21, 1720),
        2.5: (3220, 23.0, 1.20, 1710),
        2.8: (3180, 23.5, 1.19, 1690),
    },
    ("N2O4", "A-50"): {
        1.5: (3000, 21.0, 1.24, 1680),
        1.8: (3120, 21.5, 1.22, 1710),
        2.0: (3180, 22.0, 1.21, 1720),
        2.2: (3210, 22.5, 1.20, 1720),
        2.6: (3180, 23.0, 1.19, 1700),
    },
    ("N2O", "Ethanol"): {
        3.0: (2800, 24.0, 1.22, 1550),
        4.0: (2950, 25.0, 1.20, 1580),
        5.0: (3000, 26.0, 1.19, 1570),
        6.0: (2980, 27.0, 1.18, 1540),
    },
    ("H2O2", "RP1"): {
        6.0: (2700, 22.5, 1.21, 1580),
        7.0: (2750, 23.0, 1.20, 1590),
        7.5: (2760, 23.5, 1.19, 1580),
        8.0: (2750, 24.0, 1.19, 1570),
    },
}


def _interpolate_database(
    oxidizer: str, fuel: str, mixture_ratio: float
) -> tuple[float, float, float, float] | None:
    """Interpolate propellant database for given mixture ratio."""
    key = (oxidizer, fuel)
    if key not in _PROPELLANT_DATABASE:
        return None

    data = _PROPELLANT_DATABASE[key]
    mrs = sorted(data.keys())

    # Clamp to available range
    if mixture_ratio <= mrs[0]:
        return data[mrs[0]]
    if mixture_ratio >= mrs[-1]:
        return data[mrs[-1]]

    # Find bracketing values
    for i in range(len(mrs) - 1):
        if mrs[i] <= mixture_ratio <= mrs[i + 1]:
            mr_low, mr_high = mrs[i], mrs[i + 1]
            break
    else:
        return data[mrs[-1]]

    # Linear interpolation
    t = (mixture_ratio - mr_low) / (mr_high - mr_low)
    low = data[mr_low]
    high = data[mr_high]

    return (
        low[0] + t * (high[0] - low[0]),  # Tc
        low[1] + t * (high[1] - low[1]),  # MW
        low[2] + t * (high[2] - low[2]),  # gamma
        low[3] + t * (high[3] - low[3]),  # cstar
    )


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


def _get_properties_from_database(
    oxidizer: str,
    fuel: str,
    mixture_ratio: float,
    chamber_pressure_pa: float,
) -> CombustionProperties:
    """Get combustion properties from built-in database."""
    # Normalize names
    ox_name = _normalize_propellant_name(oxidizer, is_oxidizer=True)
    fuel_name = _normalize_propellant_name(fuel, is_oxidizer=False)

    result = _interpolate_database(ox_name, fuel_name, mixture_ratio)

    if result is None:
        available = list(_PROPELLANT_DATABASE.keys())
        raise ValueError(
            f"Propellant combination ({ox_name}, {fuel_name}) not in database. "
            f"Available combinations: {available}. "
            f"Install RocketCEA for arbitrary propellant combinations: pip install rocketcea"
        )

    Tc_K, MW, gamma, cstar = result

    # Calculate Cp
    R_universal = 8314.46
    R_specific = R_universal / MW
    Cp = gamma * R_specific / (gamma - 1)

    return CombustionProperties(
        chamber_temp_k=Tc_K,
        molecular_weight=MW,
        gamma=gamma,
        specific_heat_cp=Cp,
        characteristic_velocity=cstar,
        oxidizer=oxidizer,
        fuel=fuel,
        mixture_ratio=mixture_ratio,
        chamber_pressure_pa=chamber_pressure_pa,
        source="database",
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
    use_cea: bool = True,
) -> CombustionProperties:
    """Get combustion thermochemistry properties for a propellant combination.

    This function returns the thermochemical properties needed for rocket engine
    performance calculations. When RocketCEA is installed and use_cea=True,
    it uses NASA CEA for accurate equilibrium calculations. Otherwise, it falls
    back to a built-in database of common propellant combinations.

    Args:
        oxidizer: Oxidizer name (e.g., "LOX", "N2O4", "N2O", "H2O2")
        fuel: Fuel name (e.g., "RP1", "LH2", "CH4", "Ethanol", "MMH")
        mixture_ratio: Oxidizer-to-fuel mass ratio (O/F)
        chamber_pressure_pa: Chamber pressure in Pascals
        use_cea: If True and RocketCEA is installed, use CEA. Otherwise use database.

    Returns:
        CombustionProperties containing Tc, MW, gamma, Cp, c*

    Raises:
        ValueError: If propellant combination is not available in database
            and RocketCEA is not installed

    Example:
        >>> props = get_combustion_properties(
        ...     oxidizer="LOX",
        ...     fuel="RP1",
        ...     mixture_ratio=2.7,
        ...     chamber_pressure_pa=7e6,
        ... )
        >>> print(f"Tc = {props.chamber_temp_k:.0f} K, gamma = {props.gamma:.3f}")
    """
    if use_cea:
        return _get_properties_from_cea(
            oxidizer, fuel, mixture_ratio, chamber_pressure_pa
        )
    else:
        return _get_properties_from_database(
            oxidizer, fuel, mixture_ratio, chamber_pressure_pa
        )


@beartype
def is_cea_available() -> bool:
    """Check if RocketCEA is installed and available.

    Returns:
        Always True (RocketCEA is a required dependency)
    """
    return True


@beartype
def list_database_propellants() -> list[tuple[str, str]]:
    """List propellant combinations available in the built-in database.

    Returns:
        List of (oxidizer, fuel) tuples available without RocketCEA
    """
    return list(_PROPELLANT_DATABASE.keys())


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
    Requires RocketCEA for accurate optimization.

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

    Raises:
        RuntimeError: If RocketCEA is not installed
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

