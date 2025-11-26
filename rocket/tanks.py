"""Tank sizing module for Rocket.

This module provides tools for sizing propellant tanks and calculating
propellant requirements based on mission delta-V and engine performance.

Features:
- Propellant density database (LOX, LH2, RP-1, CH4, etc.)
- Rocket equation calculations for propellant mass
- Tank geometry sizing (cylindrical with elliptical domes)
- Structural mass estimation

Example:
    >>> from rocket.tanks import size_propellant, size_tank
    >>> from rocket.units import km_per_second, kilograms
    >>>
    >>> prop = size_propellant(
    ...     isp_s=300,
    ...     delta_v=km_per_second(3),
    ...     dry_mass=kilograms(500),
    ... )
    >>> print(f"Total propellant: {prop.total_propellant}")
"""

import math
from dataclasses import dataclass

from beartype import beartype

from rocket.units import (
    Quantity,
    cubic_meters,
    kilograms,
    meters,
    pascals,
    seconds,
)


# =============================================================================
# Propellant Database
# =============================================================================

# Propellant densities at typical storage conditions [kg/m³]
# Sources: Sutton & Biblarz, propellant datasheets
PROPELLANT_DENSITIES: dict[str, float] = {
    # Oxidizers
    "LOX": 1141.0,      # Liquid oxygen at -183°C
    "LO2": 1141.0,      # Alias
    "N2O4": 1450.0,     # Nitrogen tetroxide at 20°C
    "N2O": 1220.0,      # Nitrous oxide at -88°C (liquid)
    "H2O2": 1450.0,     # High-test peroxide (90%)
    "MON25": 1400.0,    # Mixed oxides of nitrogen
    "IRFNA": 1560.0,    # Inhibited red fuming nitric acid

    # Fuels
    "LH2": 70.8,        # Liquid hydrogen at -253°C
    "RP1": 810.0,       # RP-1 kerosene at 20°C
    "RP-1": 810.0,      # Alias
    "CH4": 422.6,       # Liquid methane at -161°C
    "LCH4": 422.6,      # Alias
    "Ethanol": 789.0,   # Ethanol at 20°C
    "C2H5OH": 789.0,    # Alias
    "MMH": 878.0,       # Monomethylhydrazine at 20°C
    "UDMH": 793.0,      # Unsymmetrical dimethylhydrazine at 20°C
    "N2H4": 1004.0,     # Hydrazine at 20°C
    "A-50": 903.0,      # Aerozine-50 (50% UDMH, 50% N2H4)
    "IPA": 786.0,       # Isopropyl alcohol at 20°C
    "Jet-A": 804.0,     # Jet fuel at 15°C
}

# Tank material properties
TANK_MATERIALS: dict[str, dict[str, float]] = {
    "Al2219": {
        "density": 2840.0,          # kg/m³
        "yield_strength": 290e6,     # Pa (T87 temper)
        "ultimate_strength": 400e6,  # Pa
    },
    "Al2195": {
        "density": 2710.0,          # kg/m³ (Al-Li alloy)
        "yield_strength": 455e6,     # Pa (T8 temper)
        "ultimate_strength": 530e6,  # Pa
    },
    "Al6061": {
        "density": 2700.0,          # kg/m³
        "yield_strength": 276e6,     # Pa (T6 temper)
        "ultimate_strength": 310e6,  # Pa
    },
    "SS301": {
        "density": 7880.0,          # kg/m³ (stainless steel)
        "yield_strength": 965e6,     # Pa (full hard)
        "ultimate_strength": 1275e6, # Pa
    },
    "Ti6Al4V": {
        "density": 4430.0,          # kg/m³
        "yield_strength": 880e6,     # Pa
        "ultimate_strength": 950e6,  # Pa
    },
    "CFRP": {
        "density": 1600.0,          # kg/m³ (carbon fiber composite)
        "yield_strength": 600e6,     # Pa (conservative)
        "ultimate_strength": 1000e6, # Pa
    },
}


# =============================================================================
# Data Structures
# =============================================================================


@beartype
@dataclass(frozen=True, slots=True)
class PropellantRequirements:
    """Propellant masses required for a given mission.

    Calculated from the rocket equation based on required delta-V,
    specific impulse, and vehicle dry mass.

    Attributes:
        oxidizer_mass: Mass of oxidizer required [kg]
        fuel_mass: Mass of fuel required [kg]
        total_propellant: Total propellant mass [kg]
        burn_time: Estimated burn time [s]
        mass_ratio: Wet mass / dry mass [-]
    """

    oxidizer_mass: Quantity
    fuel_mass: Quantity
    total_propellant: Quantity
    burn_time: Quantity
    mass_ratio: float


@beartype
@dataclass(frozen=True, slots=True)
class TankGeometry:
    """Tank dimensions and properties.

    Represents a cylindrical tank with elliptical end domes.
    All dimensions are for the inner wall (propellant volume).

    Attributes:
        volume: Internal tank volume [m³]
        diameter: Tank outer diameter [m]
        barrel_length: Cylindrical section length [m]
        dome_height: Height of each elliptical dome [m]
        total_length: Total tank length including domes [m]
        wall_thickness: Tank wall thickness [m]
        dry_mass: Tank structural mass [kg]
        propellant: Propellant name
        material: Tank material name
    """

    volume: Quantity
    diameter: Quantity
    barrel_length: Quantity
    dome_height: Quantity
    total_length: Quantity
    wall_thickness: Quantity
    dry_mass: Quantity
    propellant: str
    material: str


# =============================================================================
# Propellant Sizing
# =============================================================================


@beartype
def get_propellant_density(propellant: str) -> float:
    """Get density of a propellant in kg/m³.

    Args:
        propellant: Propellant name (e.g., "LOX", "RP1", "LH2")

    Returns:
        Density in kg/m³

    Raises:
        ValueError: If propellant not found in database
    """
    # Normalize name
    name = propellant.upper().replace("-", "").replace(" ", "")

    # Check direct match
    if propellant in PROPELLANT_DENSITIES:
        return PROPELLANT_DENSITIES[propellant]

    # Check normalized
    for key, value in PROPELLANT_DENSITIES.items():
        if key.upper().replace("-", "") == name:
            return value

    available = list(PROPELLANT_DENSITIES.keys())
    raise ValueError(f"Unknown propellant '{propellant}'. Available: {available}")


@beartype
def size_propellant(
    isp_s: float | int,
    delta_v: Quantity,
    dry_mass: Quantity,
    mixture_ratio: float | int = 1.0,
    mdot: Quantity | None = None,
) -> PropellantRequirements:
    """Calculate propellant mass required for a given delta-V.

    Uses the Tsiolkovsky rocket equation:
        delta_v = Isp * g0 * ln(m_wet / m_dry)

    Args:
        isp_s: Specific impulse in seconds
        delta_v: Required velocity change [velocity]
        dry_mass: Vehicle dry mass (structure, payload, etc.) [mass]
        mixture_ratio: Oxidizer/fuel mass ratio. Default 1.0 (no split).
        mdot: Mass flow rate for burn time calculation [mass/time].
            If None, burn time is estimated from typical thrust-to-weight.

    Returns:
        PropellantRequirements with oxidizer, fuel, and total masses

    Example:
        >>> prop = size_propellant(
        ...     isp_s=300,
        ...     delta_v=km_per_second(3),
        ...     dry_mass=kilograms(500),
        ...     mixture_ratio=2.7,  # LOX/RP-1
        ... )
    """
    # Validate dimensions
    if delta_v.dimension != "velocity":
        raise ValueError(f"delta_v must be velocity, got {delta_v.dimension}")
    if dry_mass.dimension != "mass":
        raise ValueError(f"dry_mass must be mass, got {dry_mass.dimension}")

    # Convert to SI
    dv = delta_v.to("m/s").value
    m_dry = dry_mass.to("kg").value
    g0 = 9.80665  # m/s²

    # Rocket equation: dv = Isp * g0 * ln(m_wet/m_dry)
    # => m_wet/m_dry = exp(dv / (Isp * g0))
    exhaust_velocity = isp_s * g0
    mass_ratio = math.exp(dv / exhaust_velocity)
    m_wet = m_dry * mass_ratio
    m_propellant = m_wet - m_dry

    # Split into oxidizer and fuel
    if mixture_ratio > 0:
        m_oxidizer = m_propellant * mixture_ratio / (1 + mixture_ratio)
        m_fuel = m_propellant / (1 + mixture_ratio)
    else:
        m_oxidizer = 0.0
        m_fuel = m_propellant

    # Estimate burn time
    if mdot is not None:
        if mdot.dimension != "mass_flow":
            raise ValueError(f"mdot must be mass_flow, got {mdot.dimension}")
        mdot_kg_s = mdot.to("kg/s").value
        burn_time_s = m_propellant / mdot_kg_s
    else:
        # Estimate from typical thrust-to-weight ratio (~1.2 for first stage)
        # F = mdot * Isp * g0, T/W = F / (m_wet * g0) = 1.2
        # => mdot = 1.2 * m_wet / Isp
        mdot_est = 1.2 * m_wet / isp_s
        burn_time_s = m_propellant / mdot_est

    return PropellantRequirements(
        oxidizer_mass=kilograms(m_oxidizer),
        fuel_mass=kilograms(m_fuel),
        total_propellant=kilograms(m_propellant),
        burn_time=seconds(burn_time_s),
        mass_ratio=mass_ratio,
    )


# =============================================================================
# Tank Sizing
# =============================================================================


@beartype
def size_tank(
    propellant_mass: Quantity,
    propellant: str,
    tank_pressure: Quantity,
    material: str = "Al2219",
    dome_ratio: float | int = 0.7071,  # sqrt(2)/2 for 2:1 ellipse
    safety_factor: float | int = 1.5,
    ullage_fraction: float | int = 0.03,
    diameter: Quantity | None = None,
) -> TankGeometry:
    """Size a propellant tank for given mass and pressure.

    Designs a cylindrical tank with elliptical end domes.
    Wall thickness is based on hoop stress from internal pressure.

    Args:
        propellant_mass: Mass of propellant to store [mass]
        propellant: Propellant name (for density lookup)
        tank_pressure: Maximum expected operating pressure [pressure]
        material: Tank material name. Default "Al2219".
        dome_ratio: Dome height / radius ratio. Default 0.707 (2:1 ellipse).
        safety_factor: Structural safety factor. Default 1.5.
        ullage_fraction: Volume fraction for ullage. Default 0.03 (3%).
        diameter: Fixed tank diameter [length]. If None, sized for L/D ~ 2.

    Returns:
        TankGeometry with dimensions and mass estimate

    Example:
        >>> tank = size_tank(
        ...     propellant_mass=kilograms(10000),
        ...     propellant="LOX",
        ...     tank_pressure=pascals(500000),  # 5 bar
        ...     material="Al2219",
        ... )
    """
    # Validate inputs
    if propellant_mass.dimension != "mass":
        raise ValueError(f"propellant_mass must be mass, got {propellant_mass.dimension}")
    if tank_pressure.dimension != "pressure":
        raise ValueError(f"tank_pressure must be pressure, got {tank_pressure.dimension}")

    # Get propellant density
    rho_prop = get_propellant_density(propellant)

    # Get material properties
    if material not in TANK_MATERIALS:
        available = list(TANK_MATERIALS.keys())
        raise ValueError(f"Unknown material '{material}'. Available: {available}")
    mat = TANK_MATERIALS[material]
    rho_mat = mat["density"]
    sigma_yield = mat["yield_strength"]

    # Convert to SI
    m_prop = propellant_mass.to("kg").value
    p = tank_pressure.to("Pa").value

    # Calculate required volume (with ullage)
    v_prop = m_prop / rho_prop
    v_total = v_prop * (1 + ullage_fraction)

    # Determine diameter
    if diameter is not None:
        if diameter.dimension != "length":
            raise ValueError(f"diameter must be length, got {diameter.dimension}")
        d = diameter.to("m").value
    else:
        # Size for L/D ~ 2 (barrel only, not counting domes)
        # V_barrel = pi * r² * L = pi * r² * (2 * 2r) = 4 * pi * r³
        # V_dome (2 elliptical domes) = 2 * (2/3) * pi * r² * (dome_ratio * r)
        #                             = (4/3) * pi * r³ * dome_ratio
        # V_total = 4*pi*r³ + (4/3)*pi*r³*dome_ratio = pi*r³*(4 + 4*dome_ratio/3)
        coeff = 4 + 4 * dome_ratio / 3
        r = (v_total / (math.pi * coeff)) ** (1 / 3)
        d = 2 * r

    r = d / 2

    # Calculate dome volume (two elliptical domes)
    # V_dome = (2/3) * pi * r² * h_dome for each dome
    h_dome = dome_ratio * r
    v_domes = 2 * (2 / 3) * math.pi * r**2 * h_dome

    # Calculate barrel length
    v_barrel = v_total - v_domes
    if v_barrel < 0:
        # Domes alone provide enough volume
        v_barrel = 0
        # Recalculate dome height
        # 2 * (2/3) * pi * r² * h = v_total
        h_dome = v_total / ((4 / 3) * math.pi * r**2)
        v_domes = v_total

    l_barrel = v_barrel / (math.pi * r**2) if v_barrel > 0 else 0

    # Total length
    l_total = l_barrel + 2 * h_dome

    # Wall thickness from hoop stress
    # sigma = p * r / t => t = p * r / (sigma / SF)
    sigma_allow = sigma_yield / safety_factor
    t = p * r / sigma_allow

    # Minimum gauge (manufacturing limit)
    t = max(t, 0.001)  # 1mm minimum

    # Tank mass estimate
    # Barrel: 2 * pi * r * L * t * rho
    # Domes: approximate as 2 * (surface area of ellipsoid cap) * t * rho
    #        Surface ≈ 2 * pi * r * (r + h) for hemisphere, less for ellipse
    #        Use factor of 0.8 for 2:1 ellipse
    a_barrel = 2 * math.pi * r * l_barrel
    a_domes = 2 * 2 * math.pi * r * (r + h_dome) * 0.8  # Two domes
    a_total = a_barrel + a_domes
    m_tank = a_total * t * rho_mat

    # Add mass for welds, fittings, etc. (typically 15-20%)
    m_tank *= 1.15

    return TankGeometry(
        volume=cubic_meters(v_total),
        diameter=meters(d),
        barrel_length=meters(l_barrel),
        dome_height=meters(h_dome),
        total_length=meters(l_total),
        wall_thickness=meters(t),
        dry_mass=kilograms(m_tank),
        propellant=propellant,
        material=material,
    )


# =============================================================================
# Convenience Functions
# =============================================================================


@beartype
def list_propellants() -> list[str]:
    """List available propellants in the density database.

    Returns:
        List of propellant names
    """
    return list(PROPELLANT_DENSITIES.keys())


@beartype
def list_materials() -> list[str]:
    """List available tank materials.

    Returns:
        List of material names
    """
    return list(TANK_MATERIALS.keys())


@beartype
def format_tank_summary(tank: TankGeometry) -> str:
    """Format tank geometry as a readable string.

    Args:
        tank: Tank geometry to summarize

    Returns:
        Multi-line string summary
    """
    lines = [
        f"Tank: {tank.propellant} ({tank.material})",
        "=" * 40,
        f"Volume:         {tank.volume.to('m^3').value:.3f} m³",
        f"                ({tank.volume.to('m^3').value * 1000:.1f} L)",
        f"Diameter:       {tank.diameter.to('m').value:.3f} m",
        f"                ({tank.diameter.to('m').value * 100:.1f} cm)",
        f"Barrel length:  {tank.barrel_length.to('m').value:.3f} m",
        f"Dome height:    {tank.dome_height.to('m').value:.3f} m (each)",
        f"Total length:   {tank.total_length.to('m').value:.3f} m",
        f"Wall thickness: {tank.wall_thickness.to('m').value * 1000:.2f} mm",
        f"Dry mass:       {tank.dry_mass.to('kg').value:.1f} kg",
    ]
    return "\n".join(lines)

