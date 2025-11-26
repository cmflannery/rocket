"""Tank sizing and propellant utilities for OpenRocketEngine.

This module provides propellant density data and tank sizing utilities.
"""

from beartype import beartype

# Propellant densities at ~1 atm and typical storage temperatures [kg/m³]
PROPELLANT_DENSITIES = {
    # Oxidizers
    "LOX": 1141.0,  # Liquid oxygen @ 90K
    "N2O4": 1440.0,  # Nitrogen tetroxide
    "N2O": 1220.0,  # Nitrous oxide @ 0°C
    "H2O2": 1450.0,  # High-test hydrogen peroxide (90%)
    "IRFNA": 1550.0,  # Inhibited red fuming nitric acid
    # Fuels
    "LH2": 70.8,  # Liquid hydrogen @ 20K
    "RP1": 810.0,  # Kerosene (RP-1)
    "CH4": 422.8,  # Liquid methane @ 111K
    "LCH4": 422.8,  # Alias for liquid methane
    "ETHANOL": 789.0,  # Ethanol
    "MMH": 880.0,  # Monomethylhydrazine
    "UDMH": 791.0,  # Unsymmetrical dimethylhydrazine
    "N2H4": 1021.0,  # Hydrazine
    "METHANOL": 792.0,  # Methanol
    "ISOPROPANOL": 786.0,  # Isopropyl alcohol
    "JET-A": 820.0,  # Jet fuel
}


@beartype
def get_propellant_density(propellant: str) -> float:
    """Get the density of a propellant in kg/m³.

    Args:
        propellant: Propellant name (e.g., "LOX", "CH4", "RP1")

    Returns:
        Density in kg/m³

    Raises:
        ValueError: If propellant is not found in database
    """
    propellant_upper = propellant.upper()

    if propellant_upper in PROPELLANT_DENSITIES:
        return PROPELLANT_DENSITIES[propellant_upper]

    # Try common aliases
    aliases = {
        "O2": "LOX",
        "OXYGEN": "LOX",
        "METHANE": "CH4",
        "KEROSENE": "RP1",
        "HYDROGEN": "LH2",
        "H2": "LH2",
    }

    if propellant_upper in aliases:
        return PROPELLANT_DENSITIES[aliases[propellant_upper]]

    available = ", ".join(sorted(PROPELLANT_DENSITIES.keys()))
    raise ValueError(
        f"Unknown propellant: {propellant}. Available: {available}"
    )


@beartype
def list_propellants() -> list[str]:
    """List all available propellants in the database.

    Returns:
        List of propellant names
    """
    return sorted(PROPELLANT_DENSITIES.keys())

