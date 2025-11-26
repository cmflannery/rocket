"""Gas generator engine cycle analysis.

The gas generator (GG) cycle is the most common turbopump-fed cycle.
A small portion of propellants is burned in a separate gas generator
to drive the turbine, then exhausted overboard (or through a secondary nozzle).

Advantages:
- Proven, reliable technology
- Simpler than staged combustion
- Lower turbine temperatures
- Decoupled turbine from main chamber

Disadvantages:
- GG exhaust is "wasted" (reduces effective Isp by 1-3%)
- Limited chamber pressure compared to staged combustion
- Requires separate GG and associated plumbing

Examples:
- SpaceX Merlin (LOX/RP-1)
- Rocketdyne F-1 (LOX/RP-1)  
- RS-68 (LOX/LH2)
- Vulcain (LOX/LH2)
"""

import math
from dataclasses import dataclass

from beartype import beartype

from rocket.cycles.base import (
    CyclePerformance,
    CycleType,
    estimate_line_losses,
    npsh_available,
    pump_power,
    turbine_power,
)
from rocket.engine import EngineGeometry, EngineInputs, EnginePerformance
from rocket.tanks import get_propellant_density
from rocket.units import Quantity, kelvin, kg_per_second, pascals, seconds


# Typical vapor pressures for common propellants [Pa]
VAPOR_PRESSURES: dict[str, float] = {
    "LOX": 101325,
    "LH2": 101325,
    "CH4": 101325,
    "RP1": 1000,
    "Ethanol": 5900,
}


def _get_vapor_pressure(propellant: str) -> float:
    """Get vapor pressure for a propellant."""
    name = propellant.upper().replace("-", "").replace(" ", "")
    for key, value in VAPOR_PRESSURES.items():
        if key.upper() == name:
            return value
    return 1000.0


@beartype
@dataclass(frozen=True, slots=True)
class GasGeneratorCycle:
    """Configuration for a gas generator engine cycle.

    The gas generator produces hot gas to drive the turbines that power
    the propellant pumps. The GG exhaust is typically dumped overboard.

    Attributes:
        turbine_inlet_temp: Gas generator combustion temperature [K]
            Typically 700-1000K to protect turbine blades
        pump_efficiency_ox: Oxidizer pump efficiency (0.6-0.75 typical)
        pump_efficiency_fuel: Fuel pump efficiency (0.6-0.75 typical)
        turbine_efficiency: Turbine isentropic efficiency (0.5-0.7 typical)
        turbine_pressure_ratio: Turbine inlet/outlet pressure ratio (2-6 typical)
        gg_mixture_ratio: GG O/F ratio (fuel-rich, typically 0.3-0.5)
        mechanical_efficiency: Mechanical losses in turbopump (0.95-0.98)
        tank_pressure_ox: Oxidizer tank pressure [Pa]
        tank_pressure_fuel: Fuel tank pressure [Pa]
    """

    turbine_inlet_temp: Quantity = None  # type: ignore  # Will validate in __post_init__
    pump_efficiency_ox: float = 0.70
    pump_efficiency_fuel: float = 0.70
    turbine_efficiency: float = 0.60
    turbine_pressure_ratio: float = 4.0
    gg_mixture_ratio: float = 0.4  # Fuel-rich
    mechanical_efficiency: float = 0.97
    tank_pressure_ox: Quantity | None = None
    tank_pressure_fuel: Quantity | None = None

    def __post_init__(self) -> None:
        """Validate inputs."""
        if self.turbine_inlet_temp is None:
            object.__setattr__(self, 'turbine_inlet_temp', kelvin(900))

    @property
    def cycle_type(self) -> CycleType:
        return CycleType.GAS_GENERATOR

    def analyze(
        self,
        inputs: EngineInputs,
        performance: EnginePerformance,
        geometry: EngineGeometry,
    ) -> CyclePerformance:
        """Analyze gas generator cycle and compute net performance.

        The key equations are:
        1. Pump power = mdot * delta_P / (rho * eta)
        2. Turbine power = mdot_gg * cp * delta_T * eta
        3. Power balance: P_turbine = P_pump_ox + P_pump_fuel
        4. Net Isp = (F_main - mdot_gg * ue_gg) / (mdot_total * g0)

        Args:
            inputs: Engine input parameters
            performance: Computed engine performance
            geometry: Computed engine geometry

        Returns:
            CyclePerformance with net performance and power balance
        """
        warnings: list[str] = []

        # Extract values
        pc = inputs.chamber_pressure.to("Pa").value
        mdot_total = performance.mdot.to("kg/s").value
        mdot_ox = performance.mdot_ox.to("kg/s").value
        mdot_fuel = performance.mdot_fuel.to("kg/s").value
        isp = performance.isp.value
        thrust = inputs.thrust.to("N").value

        # Get propellant properties
        # Try to determine from engine name
        ox_name = "LOX"
        fuel_name = "RP1"
        if inputs.name:
            name_upper = inputs.name.upper()
            if "CH4" in name_upper or "METHANE" in name_upper or "METHALOX" in name_upper:
                fuel_name = "CH4"
            elif "LH2" in name_upper or "HYDROGEN" in name_upper or "HYDROLOX" in name_upper:
                fuel_name = "LH2"

        try:
            rho_ox = get_propellant_density(ox_name)
        except ValueError:
            rho_ox = 1141.0
            warnings.append(f"Unknown oxidizer, assuming LOX density: {rho_ox} kg/m³")

        try:
            rho_fuel = get_propellant_density(fuel_name)
        except ValueError:
            rho_fuel = 810.0
            warnings.append(f"Unknown fuel, assuming RP-1 density: {rho_fuel} kg/m³")

        # Determine tank pressures
        if self.tank_pressure_ox is not None:
            p_tank_ox = self.tank_pressure_ox.to("Pa").value
        else:
            # Typical tank pressure for turbopump-fed: 2-5 bar
            p_tank_ox = 300000  # 3 bar default

        if self.tank_pressure_fuel is not None:
            p_tank_fuel = self.tank_pressure_fuel.to("Pa").value
        else:
            p_tank_fuel = 250000  # 2.5 bar default

        # Calculate pump pressure rise
        # Pump must raise from tank pressure to chamber pressure + injector drop + margins
        injector_dp = pc * 0.20  # 20% pressure drop across injector
        p_pump_outlet = pc + injector_dp

        dp_ox = p_pump_outlet - p_tank_ox
        dp_fuel = p_pump_outlet - p_tank_fuel

        # Pump power requirements
        P_pump_ox = pump_power(
            mass_flow=kg_per_second(mdot_ox),
            pressure_rise=pascals(dp_ox),
            density=rho_ox,
            efficiency=self.pump_efficiency_ox,
        ).value

        P_pump_fuel = pump_power(
            mass_flow=kg_per_second(mdot_fuel),
            pressure_rise=pascals(dp_fuel),
            density=rho_fuel,
            efficiency=self.pump_efficiency_fuel,
        ).value

        # Total pump power (accounting for mechanical losses)
        P_pump_total = (P_pump_ox + P_pump_fuel) / self.mechanical_efficiency

        # Gas generator analysis
        # Turbine power must equal pump power
        P_turbine_required = P_pump_total

        # GG exhaust properties (fuel-rich combustion products)
        # Approximate gamma and R for fuel-rich GG
        gamma_gg = 1.25  # Lower gamma for fuel-rich
        R_gg = 350.0  # J/(kg·K), approximate for fuel-rich products
        T_gg = self.turbine_inlet_temp.to("K").value

        # Turbine specific work
        # w = cp * T_in * eta * (1 - 1/PR^((gamma-1)/gamma))
        cp_gg = gamma_gg * R_gg / (gamma_gg - 1)
        T_ratio = self.turbine_pressure_ratio ** ((gamma_gg - 1) / gamma_gg)
        w_turbine = cp_gg * T_gg * self.turbine_efficiency * (1 - 1/T_ratio)

        # GG mass flow required
        mdot_gg = P_turbine_required / w_turbine

        # Check GG flow is reasonable (typically 1-5% of total)
        gg_fraction = mdot_gg / mdot_total
        if gg_fraction > 0.10:
            warnings.append(
                f"GG flow is {gg_fraction*100:.1f}% of total - unusually high"
            )

        # GG propellant split
        mdot_gg_ox = mdot_gg * self.gg_mixture_ratio / (1 + self.gg_mixture_ratio)
        mdot_gg_fuel = mdot_gg / (1 + self.gg_mixture_ratio)

        # Net performance calculation
        # The GG exhaust has much lower velocity than main chamber
        # Approximate GG exhaust velocity
        # For low MR (fuel-rich): Isp_gg ~ 200-250s
        isp_gg = 220.0  # s, approximate for fuel-rich GG exhaust
        g0 = 9.80665
        ue_gg = isp_gg * g0

        # GG exhaust thrust (negative contribution to net thrust)
        F_gg = mdot_gg * ue_gg

        # Net thrust and Isp
        # Main chamber produces full thrust
        # But we've "spent" mdot_gg propellant for low-Isp exhaust
        F_main = thrust
        net_thrust = F_main  # GG exhaust typically dumps to atmosphere
        
        # Effective total mass flow (main + GG)
        mdot_effective = mdot_total  # GG flow comes from same tanks

        # Net Isp considering the GG "loss"
        # Two ways to think about it:
        # 1. All propellant flows through main chamber at full Isp
        # 2. GG flow produces low-Isp exhaust
        # Net: weighted average of Isp
        net_isp = (F_main + F_gg * 0.3) / (mdot_effective * g0)  # GG contributes ~30% of its thrust

        # Alternative: simple debit approach
        # net_isp = isp * (1 - gg_fraction) + isp_gg * gg_fraction
        net_isp_alt = isp * (1 - gg_fraction * 0.7)  # ~70% loss on GG flow

        # Use the more conservative estimate
        net_isp = min(net_isp, net_isp_alt)

        cycle_efficiency = net_isp / isp

        # NPSH analysis
        p_vapor_ox = _get_vapor_pressure(ox_name)
        p_vapor_fuel = _get_vapor_pressure(fuel_name)

        npsh_ox = npsh_available(
            tank_pressure=pascals(p_tank_ox),
            fluid_density=rho_ox,
            vapor_pressure=pascals(p_vapor_ox),
        )

        npsh_fuel = npsh_available(
            tank_pressure=pascals(p_tank_fuel),
            fluid_density=rho_fuel,
            vapor_pressure=pascals(p_vapor_fuel),
        )

        # Feasibility checks
        feasible = True

        if npsh_ox.to("Pa").value < 50000:  # < 0.5 bar
            warnings.append("Low NPSH margin for oxidizer pump - risk of cavitation")

        if npsh_fuel.to("Pa").value < 50000:
            warnings.append("Low NPSH margin for fuel pump - risk of cavitation")

        if T_gg > 1100:
            warnings.append(
                f"Turbine inlet temp {T_gg:.0f}K exceeds typical limit (~1000K)"
            )

        if gg_fraction > 0.05:
            warnings.append(
                f"GG fraction {gg_fraction*100:.1f}% is high - consider staged combustion"
            )

        return CyclePerformance(
            net_isp=seconds(net_isp),
            net_thrust=Quantity(net_thrust, "N", "force"),
            cycle_efficiency=cycle_efficiency,
            pump_power_ox=Quantity(P_pump_ox, "W", "power"),
            pump_power_fuel=Quantity(P_pump_fuel, "W", "power"),
            turbine_power=Quantity(P_turbine_required, "W", "power"),
            turbine_mass_flow=kg_per_second(mdot_gg),
            tank_pressure_ox=pascals(p_tank_ox),
            tank_pressure_fuel=pascals(p_tank_fuel),
            npsh_margin_ox=npsh_ox,
            npsh_margin_fuel=npsh_fuel,
            cycle_type=self.cycle_type,
            feasible=feasible,
            warnings=warnings,
        )


@beartype
def estimate_turbopump_mass(
    pump_power: Quantity,
    turbine_power: Quantity,
    propellant_type: str = "LOX/RP1",
) -> Quantity:
    """Estimate turbopump mass from power requirements.

    Uses historical correlations from existing engines.

    Args:
        pump_power: Total pump power [W]
        turbine_power: Turbine power [W]
        propellant_type: Propellant combination for correlation selection

    Returns:
        Estimated turbopump mass [kg]
    """
    P = max(pump_power.value, turbine_power.value)

    # Historical correlation: mass ~ k * P^0.6
    # k varies by propellant type and technology level
    if "LH2" in propellant_type.upper():
        k = 0.015  # LH2 pumps are larger due to low density
    else:
        k = 0.008  # LOX/RP-1, LOX/CH4

    mass = k * P ** 0.6

    # Minimum mass for small turbopumps
    mass = max(mass, 5.0)

    return Quantity(mass, "kg", "mass")

