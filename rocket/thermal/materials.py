"""Material properties database for thermal analysis.

This module provides thermal and mechanical properties for common
materials used in rocket engine construction.

Example:
    >>> from rocket.thermal.materials import get_material_properties, list_materials
    >>>
    >>> # Get properties for copper
    >>> props = get_material_properties("copper")
    >>> print(f"Thermal conductivity: {props['k']} W/m·K")
    >>>
    >>> # Get temperature-dependent properties
    >>> k_at_temp = get_thermal_conductivity("inconel718", temperature=800)
"""

from typing import Any

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

# =============================================================================
# Material Property Database
# =============================================================================

# Properties at reference temperature (300 K unless noted)
# k = thermal conductivity [W/m·K]
# rho = density [kg/m³]
# cp = specific heat [J/kg·K]
# max_temp = maximum service temperature [K]
# yield_strength = yield strength at room temp [MPa]
# melting_point = melting point [K]

MATERIALS: dict[str, dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # High Conductivity Materials (for regenerative cooling liners)
    # -------------------------------------------------------------------------
    "copper": {
        "name": "Pure Copper (OFHC)",
        "k": 401.0,
        "rho": 8960.0,
        "cp": 385.0,
        "max_temp": 700,  # Softens significantly above this
        "yield_strength": 70.0,  # MPa, annealed
        "melting_point": 1358,
        "k_coeffs": [420.0, -0.06],  # k(T) = a + b*T linear approximation
        "cp_coeffs": [355.0, 0.1],   # cp(T) = a + b*T
    },
    "copper_zirconium": {
        "name": "Copper-Zirconium (CuCrZr)",
        "k": 320.0,
        "rho": 8900.0,
        "cp": 385.0,
        "max_temp": 800,  # Higher than pure Cu due to precipitation hardening
        "yield_strength": 350.0,  # MPa
        "melting_point": 1350,
        "k_coeffs": [340.0, -0.05],
        "cp_coeffs": [360.0, 0.08],
    },
    "grcop84": {
        "name": "GRCop-84 (NASA Cu alloy)",
        "k": 295.0,  # At 300 K
        "rho": 8820.0,
        "cp": 390.0,
        "max_temp": 920,  # Excellent high-temp strength
        "yield_strength": 280.0,  # MPa at RT
        "melting_point": 1360,
        "k_coeffs": [320.0, -0.04],
        "cp_coeffs": [365.0, 0.07],
    },

    # -------------------------------------------------------------------------
    # Nickel Superalloys (for hot gas path, injectors)
    # -------------------------------------------------------------------------
    "inconel718": {
        "name": "Inconel 718",
        "k": 11.4,  # Much lower than copper
        "rho": 8190.0,
        "cp": 435.0,
        "max_temp": 980,  # Precipitation hardened
        "yield_strength": 1035.0,  # MPa
        "melting_point": 1609,
        "k_coeffs": [9.0, 0.015],  # k increases with T for nickel alloys
        "cp_coeffs": [420.0, 0.05],
    },
    "inconel625": {
        "name": "Inconel 625",
        "k": 9.8,
        "rho": 8440.0,
        "cp": 410.0,
        "max_temp": 1090,  # Solid solution strengthened
        "yield_strength": 490.0,  # MPa
        "melting_point": 1623,
        "k_coeffs": [8.0, 0.012],
        "cp_coeffs": [400.0, 0.04],
    },
    "haynes230": {
        "name": "Haynes 230",
        "k": 8.9,
        "rho": 8970.0,
        "cp": 397.0,
        "max_temp": 1150,  # Excellent oxidation resistance
        "yield_strength": 390.0,  # MPa
        "melting_point": 1573,
        "k_coeffs": [7.5, 0.01],
        "cp_coeffs": [385.0, 0.035],
    },

    # -------------------------------------------------------------------------
    # Stainless Steels
    # -------------------------------------------------------------------------
    "ss304": {
        "name": "304 Stainless Steel",
        "k": 16.2,
        "rho": 8000.0,
        "cp": 500.0,
        "max_temp": 870,
        "yield_strength": 215.0,  # MPa
        "melting_point": 1673,
        "k_coeffs": [14.0, 0.012],
        "cp_coeffs": [480.0, 0.06],
    },
    "ss316": {
        "name": "316 Stainless Steel",
        "k": 14.0,
        "rho": 8000.0,
        "cp": 500.0,
        "max_temp": 870,
        "yield_strength": 290.0,  # MPa
        "melting_point": 1673,
        "k_coeffs": [12.5, 0.01],
        "cp_coeffs": [480.0, 0.055],
    },

    # -------------------------------------------------------------------------
    # Refractory Metals (for extreme temperatures)
    # -------------------------------------------------------------------------
    "niobium": {
        "name": "Niobium (Columbium)",
        "k": 53.7,
        "rho": 8570.0,
        "cp": 265.0,
        "max_temp": 1500,  # With protective coating
        "yield_strength": 105.0,  # MPa
        "melting_point": 2750,
        "k_coeffs": [55.0, -0.005],
        "cp_coeffs": [255.0, 0.02],
    },
    "molybdenum": {
        "name": "Molybdenum",
        "k": 138.0,
        "rho": 10220.0,
        "cp": 251.0,
        "max_temp": 1650,  # In inert atmosphere
        "yield_strength": 500.0,  # MPa
        "melting_point": 2896,
        "k_coeffs": [142.0, -0.008],
        "cp_coeffs": [245.0, 0.015],
    },
    "tungsten": {
        "name": "Tungsten",
        "k": 173.0,
        "rho": 19300.0,
        "cp": 132.0,
        "max_temp": 2000,  # Highest melting point metal
        "yield_strength": 750.0,  # MPa
        "melting_point": 3695,
        "k_coeffs": [180.0, -0.01],
        "cp_coeffs": [128.0, 0.008],
    },

    # -------------------------------------------------------------------------
    # Ablatives and Ceramics
    # -------------------------------------------------------------------------
    "carbon_carbon": {
        "name": "Carbon-Carbon Composite",
        "k": 50.0,  # Highly directional, this is through-thickness
        "rho": 1800.0,
        "cp": 710.0,
        "max_temp": 2500,  # In inert atmosphere
        "yield_strength": 100.0,  # MPa, varies with layup
        "melting_point": 3820,  # Sublimes
        "k_coeffs": [40.0, 0.02],
        "cp_coeffs": [700.0, 0.03],
    },
    "silica_phenolic": {
        "name": "Silica-Phenolic Ablative",
        "k": 0.5,  # Very low - that's the point
        "rho": 1700.0,
        "cp": 1050.0,
        "max_temp": 2200,  # Ablates
        "yield_strength": 50.0,
        "melting_point": 2000,  # Decomposes
        "k_coeffs": [0.4, 0.0003],
        "cp_coeffs": [1000.0, 0.15],
    },
}


# =============================================================================
# Property Access Functions
# =============================================================================


@beartype
def list_materials() -> list[str]:
    """List all available materials.

    Returns:
        List of material identifiers
    """
    return list(MATERIALS.keys())


@beartype
def get_material_properties(material: str) -> dict[str, Any]:
    """Get all properties for a material at reference temperature.

    Args:
        material: Material identifier (e.g., "copper", "inconel718")

    Returns:
        Dict with all material properties

    Raises:
        ValueError: If material not found
    """
    material_lower = material.lower().replace(" ", "_").replace("-", "")

    # Try exact match first
    if material_lower in MATERIALS:
        return MATERIALS[material_lower].copy()

    # Try partial match
    for key in MATERIALS:
        if material_lower in key or key in material_lower:
            return MATERIALS[key].copy()

    available = ", ".join(list_materials())
    raise ValueError(f"Unknown material: {material}. Available: {available}")


@beartype
def get_thermal_conductivity(
    material: str,
    temperature: float | NDArray[np.float64] | None = None,
) -> float | NDArray[np.float64]:
    """Get thermal conductivity, optionally at specific temperature(s).

    Args:
        material: Material identifier
        temperature: Temperature [K]. If None, returns reference value.

    Returns:
        Thermal conductivity [W/m·K]
    """
    props = get_material_properties(material)

    if temperature is None:
        return props["k"]

    # Temperature-dependent calculation
    coeffs = props.get("k_coeffs", [props["k"], 0.0])
    a, b = coeffs[0], coeffs[1]

    # Linear approximation: k(T) = a + b*(T - 300)
    return a + b * (np.asarray(temperature) - 300)


@beartype
def get_specific_heat(
    material: str,
    temperature: float | NDArray[np.float64] | None = None,
) -> float | NDArray[np.float64]:
    """Get specific heat, optionally at specific temperature(s).

    Args:
        material: Material identifier
        temperature: Temperature [K]. If None, returns reference value.

    Returns:
        Specific heat [J/kg·K]
    """
    props = get_material_properties(material)

    if temperature is None:
        return props["cp"]

    # Temperature-dependent calculation
    coeffs = props.get("cp_coeffs", [props["cp"], 0.0])
    a, b = coeffs[0], coeffs[1]

    return a + b * (np.asarray(temperature) - 300)


@beartype
def get_thermal_diffusivity(
    material: str,
    temperature: float | None = None,
) -> float:
    """Get thermal diffusivity.

    Args:
        material: Material identifier
        temperature: Temperature [K]. If None, uses reference value.

    Returns:
        Thermal diffusivity [m²/s]
    """
    props = get_material_properties(material)

    k = get_thermal_conductivity(material, temperature)
    cp = get_specific_heat(material, temperature)
    rho = props["rho"]

    return float(k / (rho * cp))


@beartype
def check_material_limits(
    material: str,
    temperature: float | NDArray[np.float64],
) -> dict[str, Any]:
    """Check if temperature exceeds material limits.

    Args:
        material: Material identifier
        temperature: Temperature(s) to check [K]

    Returns:
        Dict with:
            - is_safe: bool, True if within limits
            - max_temp: float, maximum service temperature
            - melting_point: float, melting temperature
            - margin: float, distance to max service temp (positive = safe)
            - warnings: list of warning strings
    """
    props = get_material_properties(material)
    temp_array = np.atleast_1d(temperature)

    max_temp = props["max_temp"]
    melting_point = props["melting_point"]

    margin = max_temp - np.max(temp_array)
    is_safe = bool(np.all(temp_array <= max_temp))

    warnings = []
    if np.any(temp_array > max_temp):
        warnings.append(f"Temperature exceeds max service temp ({max_temp} K)")
    if np.any(temp_array > 0.9 * max_temp):
        warnings.append("Temperature above 90% of max service temp")
    if np.any(temp_array > melting_point):
        warnings.append(f"Temperature exceeds melting point ({melting_point} K)!")

    return {
        "is_safe": is_safe,
        "max_temp": max_temp,
        "melting_point": melting_point,
        "margin": float(margin),
        "warnings": warnings,
    }


@beartype
def compare_materials(
    materials: list[str],
    criteria: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compare multiple materials across criteria.

    Args:
        materials: List of material identifiers to compare
        criteria: Properties to compare. Default: ["k", "max_temp", "rho"]

    Returns:
        Dict mapping criteria to {material: value} dicts
    """
    if criteria is None:
        criteria = ["k", "max_temp", "rho", "cp"]

    result: dict[str, dict[str, float]] = {c: {} for c in criteria}

    for mat in materials:
        props = get_material_properties(mat)
        for criterion in criteria:
            if criterion in props:
                result[criterion][mat] = props[criterion]

    return result

