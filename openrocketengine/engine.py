"""Engine module for OpenRocketEngine.

This module provides the core data structures and computation functions for
rocket engine design and analysis.

Design principles:
- Immutable dataclasses for all engine parameters
- Pure functions for all computations (no side effects)
- Explicit data flow: inputs -> performance -> geometry
- Type safety with beartype runtime checking
"""

import math
from dataclasses import dataclass

from beartype import beartype

from openrocketengine.isentropic import (
    G0_SI,
    area_ratio_from_mach,
    bell_nozzle_length,
    chamber_volume,
    characteristic_velocity,
    cylindrical_chamber_length,
    diameter_from_area,
    mach_from_pressure_ratio,
    mass_flow_rate,
    specific_gas_constant,
    specific_impulse,
    throat_area,
    thrust_coefficient,
    thrust_coefficient_vacuum,
)
from openrocketengine.units import (
    Quantity,
    kelvin,
    kg_per_second,
    meters,
    meters_per_second,
    pascals,
    seconds,
    square_meters,
)

# =============================================================================
# Input Data Structures
# =============================================================================


@beartype
@dataclass(frozen=True, slots=True)
class EngineInputs:
    """All inputs required to define a rocket engine.

    This immutable dataclass contains all the parameters needed to compute
    engine performance and geometry. All physical quantities use the Quantity
    class for type safety and unit handling.

    Attributes:
        thrust: Sea-level thrust [force]
        chamber_pressure: Chamber (stagnation) pressure [pressure]
        chamber_temp: Chamber (stagnation) temperature [temperature]
        exit_pressure: Nozzle exit pressure [pressure]
        ambient_pressure: Ambient pressure for performance calculation [pressure]
        molecular_weight: Molecular weight of exhaust gases [kg/kmol]
        gamma: Ratio of specific heats (Cp/Cv) [-]
        lstar: Characteristic chamber length [length]
        mixture_ratio: Oxidizer to fuel mass ratio [-]
        contraction_ratio: Chamber area / throat area [-]
        contraction_angle: Chamber convergence half-angle [degrees]
        bell_fraction: Bell nozzle length as fraction of 15° cone [-]
        name: Optional engine name for identification
    """

    thrust: Quantity  # Sea-level thrust
    chamber_pressure: Quantity  # pc
    chamber_temp: Quantity  # Tc
    exit_pressure: Quantity  # pe
    molecular_weight: float  # kg/kmol
    gamma: float  # Cp/Cv, dimensionless
    lstar: Quantity  # L*, characteristic length
    mixture_ratio: float  # O/F ratio, dimensionless
    ambient_pressure: Quantity | None = None  # pa, defaults to pe
    contraction_ratio: float = 4.0  # Ac/At, dimensionless
    contraction_angle: float = 45.0  # degrees
    bell_fraction: float = 0.8  # fraction of 15° cone length
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate inputs after initialization."""
        # Validate dimensions
        if self.thrust.dimension != "force":
            raise ValueError(f"thrust must be force, got {self.thrust.dimension}")
        if self.chamber_pressure.dimension != "pressure":
            raise ValueError(
                f"chamber_pressure must be pressure, got {self.chamber_pressure.dimension}"
            )
        if self.chamber_temp.dimension != "temperature":
            raise ValueError(
                f"chamber_temp must be temperature, got {self.chamber_temp.dimension}"
            )
        if self.exit_pressure.dimension != "pressure":
            raise ValueError(
                f"exit_pressure must be pressure, got {self.exit_pressure.dimension}"
            )
        if self.lstar.dimension != "length":
            raise ValueError(f"lstar must be length, got {self.lstar.dimension}")
        if self.ambient_pressure is not None and self.ambient_pressure.dimension != "pressure":
            raise ValueError(
                f"ambient_pressure must be pressure, got {self.ambient_pressure.dimension}"
            )

        # Validate physical constraints
        if self.gamma <= 1.0:
            raise ValueError(f"gamma must be > 1, got {self.gamma}")
        if self.molecular_weight <= 0:
            raise ValueError(f"molecular_weight must be > 0, got {self.molecular_weight}")
        if self.mixture_ratio <= 0:
            raise ValueError(f"mixture_ratio must be > 0, got {self.mixture_ratio}")
        if self.contraction_ratio < 1.0:
            raise ValueError(f"contraction_ratio must be >= 1, got {self.contraction_ratio}")
        if not (0 < self.contraction_angle < 90):
            raise ValueError(
                f"contraction_angle must be between 0 and 90 degrees, got {self.contraction_angle}"
            )
        if not (0 < self.bell_fraction <= 1.0):
            raise ValueError(f"bell_fraction must be between 0 and 1, got {self.bell_fraction}")

    @property
    def effective_ambient_pressure(self) -> Quantity:
        """Return ambient pressure, defaulting to exit pressure if not specified."""
        if self.ambient_pressure is not None:
            return self.ambient_pressure
        return self.exit_pressure

    @classmethod
    def from_propellants(
        cls,
        oxidizer: str,
        fuel: str,
        thrust: Quantity,
        chamber_pressure: Quantity,
        mixture_ratio: float | None = None,
        exit_pressure: Quantity | None = None,
        lstar: Quantity | None = None,
        ambient_pressure: Quantity | None = None,
        contraction_ratio: float = 4.0,
        contraction_angle: float = 45.0,
        bell_fraction: float = 0.8,
        name: str | None = None,
        use_cea: bool = True,
    ) -> "EngineInputs":
        """Create EngineInputs from propellant names, automatically computing thermochemistry.

        This factory method uses RocketCEA (if available) or a built-in database
        to determine the combustion properties (chamber temperature, molecular weight,
        and gamma) from the specified propellant combination.

        Args:
            oxidizer: Oxidizer name (e.g., "LOX", "N2O4", "N2O", "H2O2")
            fuel: Fuel name (e.g., "RP1", "LH2", "CH4", "Ethanol", "MMH")
            thrust: Sea-level thrust
            chamber_pressure: Chamber pressure
            mixture_ratio: O/F mass ratio. If None, uses optimal ratio for max Isp.
            exit_pressure: Nozzle exit pressure. Defaults to 1 atm (101325 Pa).
            lstar: Characteristic length. Defaults to 1.0 m (typical for biprop).
            ambient_pressure: Ambient pressure for performance calc. Defaults to exit_pressure.
            contraction_ratio: Chamber/throat area ratio. Default 4.0.
            contraction_angle: Convergent section half-angle [deg]. Default 45.
            bell_fraction: Bell length as fraction of 15° cone. Default 0.8.
            name: Optional engine name.
            use_cea: If True, use RocketCEA when available. Default True.

        Returns:
            EngineInputs with thermochemistry computed from propellant combination.

        Example:
            >>> inputs = EngineInputs.from_propellants(
            ...     oxidizer="LOX",
            ...     fuel="RP1",
            ...     thrust=kilonewtons(100),
            ...     chamber_pressure=megapascals(7),
            ...     mixture_ratio=2.7,
            ... )
            >>> print(f"Tc = {inputs.chamber_temp}")
        """
        from openrocketengine.propellants import (
            get_combustion_properties,
            get_optimal_mixture_ratio,
            is_cea_available,
            _normalize_propellant_name,
        )

        # Default exit pressure to 1 atm
        if exit_pressure is None:
            exit_pressure = pascals(101325)

        # Default L* to 1.0 m
        if lstar is None:
            lstar = meters(1.0)

        # Get chamber pressure in Pa for CEA
        pc_pa = chamber_pressure.to("Pa").value

        # Find optimal mixture ratio if not specified
        if mixture_ratio is None:
            if is_cea_available() and use_cea:
                mixture_ratio, _ = get_optimal_mixture_ratio(
                    oxidizer=oxidizer,
                    fuel=fuel,
                    chamber_pressure_pa=pc_pa,
                    metric="isp",
                )
            else:
                # Use typical values for common propellants
                defaults = {
                    ("LOX", "LH2"): 6.0,
                    ("LOX", "RP1"): 2.7,
                    ("LOX", "CH4"): 3.2,
                    ("LOX", "Ethanol"): 1.5,
                    ("N2O4", "MMH"): 2.0,
                    ("N2O4", "UDMH"): 2.2,
                    ("N2O4", "A-50"): 2.0,
                    ("N2O", "Ethanol"): 5.0,
                    ("H2O2", "RP1"): 7.5,
                }
                ox_norm = _normalize_propellant_name(oxidizer, is_oxidizer=True)
                fuel_norm = _normalize_propellant_name(fuel, is_oxidizer=False)
                mixture_ratio = defaults.get((ox_norm, fuel_norm), 2.5)

        # Get combustion properties
        props = get_combustion_properties(
            oxidizer=oxidizer,
            fuel=fuel,
            mixture_ratio=mixture_ratio,
            chamber_pressure_pa=pc_pa,
            use_cea=use_cea,
        )

        # Generate name if not provided
        if name is None:
            name = f"{oxidizer}/{fuel} Engine"

        return cls(
            thrust=thrust,
            chamber_pressure=chamber_pressure,
            chamber_temp=kelvin(props.chamber_temp_k),
            exit_pressure=exit_pressure,
            molecular_weight=props.molecular_weight,
            gamma=props.gamma,
            lstar=lstar,
            mixture_ratio=mixture_ratio,
            ambient_pressure=ambient_pressure,
            contraction_ratio=contraction_ratio,
            contraction_angle=contraction_angle,
            bell_fraction=bell_fraction,
            name=name,
        )


# =============================================================================
# Output Data Structures
# =============================================================================


@beartype
@dataclass(frozen=True, slots=True)
class EnginePerformance:
    """Computed performance metrics for a rocket engine.

    All values are computed from EngineInputs using isentropic flow equations.

    Attributes:
        isp: Specific impulse at sea level [s]
        isp_vac: Specific impulse in vacuum [s]
        cstar: Characteristic velocity [m/s]
        exhaust_velocity: Nozzle exit velocity [m/s]
        thrust_coeff: Thrust coefficient at sea level [-]
        thrust_coeff_vac: Vacuum thrust coefficient [-]
        mdot: Total mass flow rate [kg/s]
        mdot_ox: Oxidizer mass flow rate [kg/s]
        mdot_fuel: Fuel mass flow rate [kg/s]
        expansion_ratio: Nozzle expansion ratio Ae/At [-]
        exit_mach: Mach number at nozzle exit [-]
    """

    isp: Quantity  # seconds
    isp_vac: Quantity  # seconds
    cstar: Quantity  # m/s
    exhaust_velocity: Quantity  # m/s
    thrust_coeff: float  # dimensionless
    thrust_coeff_vac: float  # dimensionless
    mdot: Quantity  # kg/s
    mdot_ox: Quantity  # kg/s
    mdot_fuel: Quantity  # kg/s
    expansion_ratio: float  # dimensionless
    exit_mach: float  # dimensionless


@beartype
@dataclass(frozen=True, slots=True)
class EngineGeometry:
    """Computed geometry for a rocket engine.

    All dimensions are computed from EngineInputs and EnginePerformance.

    Attributes:
        throat_area: Throat cross-sectional area [m^2]
        throat_diameter: Throat diameter [m]
        exit_area: Nozzle exit area [m^2]
        exit_diameter: Nozzle exit diameter [m]
        chamber_area: Chamber cross-sectional area [m^2]
        chamber_diameter: Chamber diameter [m]
        chamber_volume: Total chamber volume [m^3]
        chamber_length: Cylindrical chamber length [m]
        nozzle_length: Nozzle length (from throat to exit) [m]
        expansion_ratio: Ae/At [-]
        contraction_ratio: Ac/At [-]
    """

    throat_area: Quantity
    throat_diameter: Quantity
    exit_area: Quantity
    exit_diameter: Quantity
    chamber_area: Quantity
    chamber_diameter: Quantity
    chamber_volume: Quantity
    chamber_length: Quantity
    nozzle_length: Quantity
    expansion_ratio: float
    contraction_ratio: float


# =============================================================================
# Computation Functions
# =============================================================================


@beartype
def compute_performance(inputs: EngineInputs) -> EnginePerformance:
    """Compute engine performance from inputs.

    This is a pure function that takes engine inputs and returns computed
    performance metrics using isentropic flow equations.

    Args:
        inputs: Engine input parameters

    Returns:
        Computed performance metrics
    """
    # Extract values in SI units
    thrust_N = inputs.thrust.to("N").value
    pc_Pa = inputs.chamber_pressure.to("Pa").value
    Tc_K = inputs.chamber_temp.to("K").value
    pe_Pa = inputs.exit_pressure.to("Pa").value
    pa_Pa = inputs.effective_ambient_pressure.to("Pa").value
    MW = inputs.molecular_weight
    gamma = inputs.gamma
    MR = inputs.mixture_ratio

    # Compute gas properties
    R = specific_gas_constant(MW)

    # Pressure ratios
    pe_pc = pe_Pa / pc_Pa
    pa_pc = pa_Pa / pc_Pa

    # Exit Mach number from pressure ratio
    exit_mach = mach_from_pressure_ratio(pc_Pa / pe_Pa, gamma)

    # Expansion ratio from exit Mach
    expansion_ratio = area_ratio_from_mach(exit_mach, gamma)

    # Characteristic velocity
    cstar_val = characteristic_velocity(gamma, R, Tc_K)

    # Thrust coefficients
    Cf = thrust_coefficient(gamma, pe_pc, pa_pc, expansion_ratio)
    Cf_vac = thrust_coefficient_vacuum(gamma, pe_pc, expansion_ratio)

    # Specific impulse
    Isp_val = specific_impulse(cstar_val, Cf, G0_SI)
    Isp_vac_val = specific_impulse(cstar_val, Cf_vac, G0_SI)

    # Mass flow rate
    mdot_val = mass_flow_rate(thrust_N, Isp_val, G0_SI)
    mdot_ox_val = mdot_val * MR / (MR + 1.0)
    mdot_fuel_val = mdot_val / (MR + 1.0)

    # Exhaust velocity
    ue_val = Isp_val * G0_SI

    return EnginePerformance(
        isp=seconds(Isp_val),
        isp_vac=seconds(Isp_vac_val),
        cstar=meters_per_second(cstar_val),
        exhaust_velocity=meters_per_second(ue_val),
        thrust_coeff=Cf,
        thrust_coeff_vac=Cf_vac,
        mdot=kg_per_second(mdot_val),
        mdot_ox=kg_per_second(mdot_ox_val),
        mdot_fuel=kg_per_second(mdot_fuel_val),
        expansion_ratio=expansion_ratio,
        exit_mach=exit_mach,
    )


@beartype
def compute_geometry(inputs: EngineInputs, performance: EnginePerformance) -> EngineGeometry:
    """Compute engine geometry from inputs and performance.

    This is a pure function that takes engine inputs and computed performance
    to determine physical dimensions.

    Args:
        inputs: Engine input parameters
        performance: Computed performance metrics

    Returns:
        Computed geometry
    """
    # Extract values
    pc_Pa = inputs.chamber_pressure.to("Pa").value
    mdot_val = performance.mdot.to("kg/s").value
    cstar_val = performance.cstar.to("m/s").value
    lstar_m = inputs.lstar.to("m").value
    expansion_ratio = performance.expansion_ratio
    contraction_ratio = inputs.contraction_ratio
    contraction_angle_rad = math.radians(inputs.contraction_angle)
    bell_fraction = inputs.bell_fraction

    # Throat geometry
    At = throat_area(mdot_val, cstar_val, pc_Pa)
    Dt = diameter_from_area(At)
    Rt = Dt / 2.0

    # Exit geometry
    Ae = At * expansion_ratio
    De = diameter_from_area(Ae)
    Re = De / 2.0

    # Chamber geometry
    Ac = At * contraction_ratio
    Dc = diameter_from_area(Ac)
    Rc = Dc / 2.0

    # Chamber volume from L*
    Vc = chamber_volume(lstar_m, At)

    # Cylindrical chamber length
    Lcyl = cylindrical_chamber_length(Vc, Ac, Rc, Rt, contraction_angle_rad)
    # Ensure positive length
    Lcyl = max(Lcyl, 0.0)

    # Nozzle length (bell nozzle)
    Ln = bell_nozzle_length(Rt, Re, bell_fraction)

    return EngineGeometry(
        throat_area=square_meters(At),
        throat_diameter=meters(Dt),
        exit_area=square_meters(Ae),
        exit_diameter=meters(De),
        chamber_area=square_meters(Ac),
        chamber_diameter=meters(Dc),
        chamber_volume=Quantity(Vc, "m^3", "volume"),
        chamber_length=meters(Lcyl),
        nozzle_length=meters(Ln),
        expansion_ratio=expansion_ratio,
        contraction_ratio=contraction_ratio,
    )


@beartype
def design_engine(inputs: EngineInputs) -> tuple[EnginePerformance, EngineGeometry]:
    """Complete engine design from inputs.

    Convenience function that computes both performance and geometry.

    Args:
        inputs: Engine input parameters

    Returns:
        Tuple of (performance, geometry)
    """
    performance = compute_performance(inputs)
    geometry = compute_geometry(inputs, performance)
    return performance, geometry


# =============================================================================
# Analysis Functions
# =============================================================================


@beartype
def thrust_at_altitude(
    inputs: EngineInputs,
    performance: EnginePerformance,
    geometry: EngineGeometry,
    ambient_pressure: Quantity,
) -> Quantity:
    """Calculate thrust at a given ambient pressure (altitude).

    Args:
        inputs: Engine input parameters
        performance: Computed performance (used for expansion ratio)
        geometry: Computed geometry (used for exit area)
        ambient_pressure: Ambient pressure at altitude

    Returns:
        Thrust at the specified altitude
    """
    pc_Pa = inputs.chamber_pressure.to("Pa").value
    pe_Pa = inputs.exit_pressure.to("Pa").value
    pa_Pa = ambient_pressure.to("Pa").value
    gamma = inputs.gamma
    cstar_val = performance.cstar.to("m/s").value
    mdot_val = performance.mdot.to("kg/s").value
    expansion_ratio = performance.expansion_ratio

    pe_pc = pe_Pa / pc_Pa
    pa_pc = pa_Pa / pc_Pa

    Cf = thrust_coefficient(gamma, pe_pc, pa_pc, expansion_ratio)
    thrust_N = mdot_val * cstar_val * Cf

    return Quantity(thrust_N, "N", "force")


@beartype
def isp_at_altitude(
    inputs: EngineInputs,
    performance: EnginePerformance,
    ambient_pressure: Quantity,
) -> Quantity:
    """Calculate specific impulse at a given ambient pressure (altitude).

    Args:
        inputs: Engine input parameters
        performance: Computed performance
        ambient_pressure: Ambient pressure at altitude

    Returns:
        Specific impulse at the specified altitude
    """
    pc_Pa = inputs.chamber_pressure.to("Pa").value
    pe_Pa = inputs.exit_pressure.to("Pa").value
    pa_Pa = ambient_pressure.to("Pa").value
    gamma = inputs.gamma
    cstar_val = performance.cstar.to("m/s").value
    expansion_ratio = performance.expansion_ratio

    pe_pc = pe_Pa / pc_Pa
    pa_pc = pa_Pa / pc_Pa

    Cf = thrust_coefficient(gamma, pe_pc, pa_pc, expansion_ratio)
    Isp = specific_impulse(cstar_val, Cf, G0_SI)

    return seconds(Isp)


# =============================================================================
# Summary and Display
# =============================================================================


@beartype
def format_performance_summary(inputs: EngineInputs, performance: EnginePerformance) -> str:
    """Format a human-readable performance summary.

    Args:
        inputs: Engine input parameters
        performance: Computed performance

    Returns:
        Formatted string summary
    """
    name = inputs.name or "Unnamed Engine"
    lines = [
        f"{'=' * 60}",
        f"ENGINE PERFORMANCE SUMMARY: {name}",
        f"{'=' * 60}",
        "",
        "INPUTS:",
        f"  Thrust (SL):        {inputs.thrust}",
        f"  Chamber Pressure:   {inputs.chamber_pressure}",
        f"  Chamber Temp:       {inputs.chamber_temp}",
        f"  Exit Pressure:      {inputs.exit_pressure}",
        f"  Molecular Weight:   {inputs.molecular_weight:.2f} kg/kmol",
        f"  Gamma:              {inputs.gamma:.3f}",
        f"  Mixture Ratio:      {inputs.mixture_ratio:.2f}",
        "",
        "PERFORMANCE:",
        f"  Isp (SL):           {performance.isp.value:.1f} s",
        f"  Isp (Vac):          {performance.isp_vac.value:.1f} s",
        f"  C*:                 {performance.cstar.value:.1f} m/s",
        f"  Exit Velocity:      {performance.exhaust_velocity.value:.1f} m/s",
        f"  Thrust Coeff (SL):  {performance.thrust_coeff:.3f}",
        f"  Thrust Coeff (Vac): {performance.thrust_coeff_vac:.3f}",
        f"  Exit Mach:          {performance.exit_mach:.2f}",
        "",
        "MASS FLOW:",
        f"  Total:              {performance.mdot.value:.3f} kg/s",
        f"  Oxidizer:           {performance.mdot_ox.value:.3f} kg/s",
        f"  Fuel:               {performance.mdot_fuel.value:.3f} kg/s",
        "",
        f"  Expansion Ratio:    {performance.expansion_ratio:.2f}",
        f"{'=' * 60}",
    ]
    return "\n".join(lines)


@beartype
def format_geometry_summary(inputs: EngineInputs, geometry: EngineGeometry) -> str:
    """Format a human-readable geometry summary.

    Args:
        inputs: Engine input parameters
        geometry: Computed geometry

    Returns:
        Formatted string summary
    """
    name = inputs.name or "Unnamed Engine"
    lines = [
        f"{'=' * 60}",
        f"ENGINE GEOMETRY SUMMARY: {name}",
        f"{'=' * 60}",
        "",
        "THROAT:",
        f"  Area:               {geometry.throat_area.value * 1e4:.4f} cm^2",
        f"  Diameter:           {geometry.throat_diameter.value * 100:.3f} cm",
        "",
        "EXIT:",
        f"  Area:               {geometry.exit_area.value * 1e4:.2f} cm^2",
        f"  Diameter:           {geometry.exit_diameter.value * 100:.2f} cm",
        "",
        "CHAMBER:",
        f"  Area:               {geometry.chamber_area.value * 1e4:.2f} cm^2",
        f"  Diameter:           {geometry.chamber_diameter.value * 100:.2f} cm",
        f"  Volume:             {geometry.chamber_volume.value * 1e6:.1f} cm^3",
        f"  Length (cyl):       {geometry.chamber_length.value * 100:.2f} cm",
        "",
        "NOZZLE:",
        f"  Length:             {geometry.nozzle_length.value * 100:.2f} cm",
        "",
        "RATIOS:",
        f"  Expansion (Ae/At):  {geometry.expansion_ratio:.2f}",
        f"  Contraction (Ac/At):{geometry.contraction_ratio:.2f}",
        f"{'=' * 60}",
    ]
    return "\n".join(lines)

