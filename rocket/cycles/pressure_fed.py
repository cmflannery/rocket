"""Pressure-fed engine cycle analysis.

Pressure-fed engines use high-pressure gas (typically helium) to push
propellants from tanks into the combustion chamber. They are the simplest
cycle type but require heavy tanks to contain the high pressures.

Advantages:
- Simplicity and reliability (no turbopumps)
- Fewer failure modes
- Lower development cost

Disadvantages:
- Heavy tanks (must withstand full chamber pressure + margins)
- Limited chamber pressure (~3 MPa practical limit)
- Lower performance (limited Isp due to pressure constraints)

Typical applications:
- Upper stages
- Spacecraft thrusters
- Student/amateur rockets
"""

from dataclasses import dataclass

from beartype import beartype

from rocket.cycles.base import (
    CyclePerformance,
    CycleType,
    estimate_line_losses,
    npsh_available,
)
from rocket.engine import EngineGeometry, EngineInputs, EnginePerformance
from rocket.tanks import get_propellant_density
from rocket.units import Quantity, kg_per_second, pascals

# Typical vapor pressures for common propellants [Pa]
# At nominal storage temperatures
VAPOR_PRESSURES: dict[str, float] = {
    "LOX": 101325,      # ~1 atm at -183°C
    "LH2": 101325,      # ~1 atm at -253°C
    "CH4": 101325,      # ~1 atm at -161°C
    "RP1": 1000,        # Very low at 20°C
    "Ethanol": 5900,    # ~0.06 atm at 20°C
    "N2O4": 96000,      # ~0.95 atm at 20°C
    "MMH": 4800,        # Low at 20°C
    "N2O": 5200000,     # ~51 atm at 20°C (self-pressurizing)
}


def _get_vapor_pressure(propellant: str) -> float:
    """Get vapor pressure for a propellant."""
    # Normalize name
    name = propellant.upper().replace("-", "").replace(" ", "")

    for key, value in VAPOR_PRESSURES.items():
        if key.upper() == name:
            return value

    # Default to low vapor pressure if unknown
    return 1000.0


@beartype
@dataclass(frozen=True, slots=True)
class PressureFedCycle:
    """Configuration for a pressure-fed engine cycle.

    In a pressure-fed system, the tank pressure must exceed the chamber
    pressure plus all line losses and injector pressure drop.

    Attributes:
        injector_dp_fraction: Injector pressure drop as fraction of Pc (typically 0.15-0.25)
        line_loss_fraction: Feed line losses as fraction of Pc (typically 0.05-0.10)
        tank_pressure_margin: Safety margin on tank pressure (typically 1.1-1.2)
        pressurant: Pressurant gas type (typically "helium" or "nitrogen")
        ox_line_diameter: Oxidizer feed line diameter [m]
        fuel_line_diameter: Fuel feed line diameter [m]
        ox_line_length: Oxidizer feed line length [m]
        fuel_line_length: Fuel feed line length [m]
    """

    injector_dp_fraction: float = 0.20
    line_loss_fraction: float = 0.05
    tank_pressure_margin: float = 1.15
    pressurant: str = "helium"
    ox_line_diameter: float = 0.05   # m
    fuel_line_diameter: float = 0.04  # m
    ox_line_length: float = 2.0       # m
    fuel_line_length: float = 2.0     # m

    @property
    def cycle_type(self) -> CycleType:
        return CycleType.PRESSURE_FED

    def analyze(
        self,
        inputs: EngineInputs,
        performance: EnginePerformance,
        geometry: EngineGeometry,
    ) -> CyclePerformance:
        """Analyze pressure-fed cycle and determine tank pressures.

        Args:
            inputs: Engine input parameters
            performance: Computed engine performance
            geometry: Computed engine geometry

        Returns:
            CyclePerformance with pressure requirements and feasibility
        """
        warnings: list[str] = []

        # Extract values
        pc = inputs.chamber_pressure.to("Pa").value
        mdot_ox = performance.mdot_ox.to("kg/s").value
        mdot_fuel = performance.mdot_fuel.to("kg/s").value

        # Get propellant densities
        # Extract propellant names from engine name or use defaults
        ox_name = "LOX"  # Default
        fuel_name = "RP1"  # Default
        if inputs.name:
            name_upper = inputs.name.upper()
            if "LOX" in name_upper or "LO2" in name_upper:
                ox_name = "LOX"
            if "CH4" in name_upper or "METHANE" in name_upper:
                fuel_name = "CH4"
            elif "RP1" in name_upper or "KEROSENE" in name_upper:
                fuel_name = "RP1"
            elif "ETHANOL" in name_upper:
                fuel_name = "Ethanol"
            elif "LH2" in name_upper or "HYDROGEN" in name_upper:
                fuel_name = "LH2"

        try:
            rho_ox = get_propellant_density(ox_name)
        except ValueError:
            rho_ox = 1141.0  # Default LOX
            warnings.append(f"Unknown oxidizer, assuming LOX density: {rho_ox} kg/m³")

        try:
            rho_fuel = get_propellant_density(fuel_name)
        except ValueError:
            rho_fuel = 810.0  # Default RP-1
            warnings.append(f"Unknown fuel, assuming RP-1 density: {rho_fuel} kg/m³")

        # Calculate pressure budget
        # Tank pressure must overcome: chamber + injector drop + line losses
        dp_injector = pc * self.injector_dp_fraction
        dp_lines_ox = pc * self.line_loss_fraction
        dp_lines_fuel = pc * self.line_loss_fraction

        # More detailed line loss estimate
        dp_lines_ox_calc = estimate_line_losses(
            mass_flow=kg_per_second(mdot_ox),
            density=rho_ox,
            pipe_diameter=self.ox_line_diameter,
            pipe_length=self.ox_line_length,
        ).to("Pa").value

        dp_lines_fuel_calc = estimate_line_losses(
            mass_flow=kg_per_second(mdot_fuel),
            density=rho_fuel,
            pipe_diameter=self.fuel_line_diameter,
            pipe_length=self.fuel_line_length,
        ).to("Pa").value

        # Use maximum of estimated and calculated
        dp_lines_ox = max(dp_lines_ox, dp_lines_ox_calc)
        dp_lines_fuel = max(dp_lines_fuel, dp_lines_fuel_calc)

        # Required tank pressures
        p_tank_ox = (pc + dp_injector + dp_lines_ox) * self.tank_pressure_margin
        p_tank_fuel = (pc + dp_injector + dp_lines_fuel) * self.tank_pressure_margin

        # NPSH analysis
        p_vapor_ox = _get_vapor_pressure(ox_name)
        p_vapor_fuel = _get_vapor_pressure(fuel_name)

        npsh_ox = npsh_available(
            tank_pressure=pascals(p_tank_ox),
            fluid_density=rho_ox,
            vapor_pressure=pascals(p_vapor_ox),
            line_losses=pascals(dp_lines_ox),
        )

        npsh_fuel = npsh_available(
            tank_pressure=pascals(p_tank_fuel),
            fluid_density=rho_fuel,
            vapor_pressure=pascals(p_vapor_fuel),
            line_losses=pascals(dp_lines_fuel),
        )

        # Check feasibility
        feasible = True

        # Pressure-fed practical limit is ~3-4 MPa
        if pc > 4e6:
            warnings.append(
                f"Chamber pressure {pc/1e6:.1f} MPa exceeds typical pressure-fed limit (~3-4 MPa)"
            )

        # Tank pressure feasibility
        if p_tank_ox > 6e6:
            warnings.append(
                f"Ox tank pressure {p_tank_ox/1e6:.1f} MPa is very high for pressure-fed"
            )
        if p_tank_fuel > 6e6:
            warnings.append(
                f"Fuel tank pressure {p_tank_fuel/1e6:.1f} MPa is very high for pressure-fed"
            )

        # For pressure-fed, there are no turbopumps, so no pump power
        # All "pumping" is done by the pressurized tanks
        pump_power_ox = Quantity(0.0, "W", "power")
        pump_power_fuel = Quantity(0.0, "W", "power")
        turbine_power = Quantity(0.0, "W", "power")
        turbine_flow = kg_per_second(0.0)

        # Net performance equals ideal performance (no turbine drive losses)
        net_isp = performance.isp
        net_thrust = inputs.thrust
        cycle_efficiency = 1.0  # No cycle losses for pressure-fed

        return CyclePerformance(
            net_isp=net_isp,
            net_thrust=net_thrust,
            cycle_efficiency=cycle_efficiency,
            pump_power_ox=pump_power_ox,
            pump_power_fuel=pump_power_fuel,
            turbine_power=turbine_power,
            turbine_mass_flow=turbine_flow,
            tank_pressure_ox=pascals(p_tank_ox),
            tank_pressure_fuel=pascals(p_tank_fuel),
            npsh_margin_ox=npsh_ox,
            npsh_margin_fuel=npsh_fuel,
            cycle_type=self.cycle_type,
            feasible=feasible,
            warnings=warnings,
        )


@beartype
def estimate_pressurant_mass(
    propellant_volume: Quantity,
    tank_pressure: Quantity,
    pressurant: str = "helium",
    initial_temp: float = 300.0,  # K
    blowdown_ratio: float = 2.0,
) -> Quantity:
    """Estimate pressurant gas mass required.

    Uses ideal gas law with blowdown consideration.
    In blowdown mode, tank pressure drops as propellant is expelled.

    Args:
        propellant_volume: Volume of propellant to expel [m³]
        tank_pressure: Initial tank pressure [Pa]
        pressurant: Gas type ("helium" or "nitrogen")
        initial_temp: Pressurant initial temperature [K]
        blowdown_ratio: Initial/final pressure ratio for blowdown

    Returns:
        Required pressurant mass [kg]
    """
    # Gas constants
    R_helium = 2077.0  # J/(kg·K)
    R_nitrogen = 296.8  # J/(kg·K)

    R = R_helium if pressurant.lower() == "helium" else R_nitrogen

    V = propellant_volume.to("m^3").value
    P = tank_pressure.to("Pa").value

    # For pressure-regulated system: m = P * V / (R * T)
    # For blowdown: need to account for pressure decay
    # Simplified: assume average pressure
    P_avg = P / (1 + 1/blowdown_ratio) * 2

    mass = P_avg * V / (R * initial_temp)

    # Add margin for residuals and cooling
    mass *= 1.2

    return Quantity(mass, "kg", "mass")

