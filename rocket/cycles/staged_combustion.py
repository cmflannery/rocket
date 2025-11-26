"""Staged combustion engine cycle analysis.

Staged combustion is the highest-performance liquid engine cycle. Unlike
gas generators where turbine exhaust is dumped overboard, staged combustion
routes all turbine exhaust into the main combustion chamber.

Variants:
- Oxidizer-rich staged combustion (ORSC): Preburner runs oxidizer-rich
  Example: RD-180, RD-191, NK-33
- Fuel-rich staged combustion (FRSC): Preburner runs fuel-rich
  Example: RS-25 (SSME), BE-4
- Full-flow staged combustion (FFSC): Both ox-rich AND fuel-rich preburners
  Example: SpaceX Raptor

Advantages:
- Highest Isp (all propellant goes through main chamber)
- High chamber pressure capability
- High thrust-to-weight ratio

Disadvantages:
- Most complex cycle
- Expensive development
- Challenging turbine environments (especially ORSC)

References:
    - Sutton & Biblarz, Chapter 6
    - Humble, Henry & Larson, "Space Propulsion Analysis and Design"
"""

import math
from dataclasses import dataclass

from beartype import beartype

from rocket.cycles.base import (
    CyclePerformance,
    CycleType,
    npsh_available,
    pump_power,
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
class StagedCombustionCycle:
    """Configuration for a staged combustion engine cycle.

    In staged combustion, the preburner exhaust (which drove the turbine)
    is routed into the main combustion chamber, so no propellant is wasted.

    Attributes:
        preburner_temp: Preburner combustion temperature [K]
            Typically 700-900K for fuel-rich, 500-700K for ox-rich
        pump_efficiency_ox: Oxidizer pump efficiency (0.7-0.8 typical)
        pump_efficiency_fuel: Fuel pump efficiency (0.7-0.8 typical)
        turbine_efficiency: Turbine isentropic efficiency (0.6-0.75 typical)
        turbine_pressure_ratio: Turbine pressure ratio (1.5-3.0 typical)
        preburner_mixture_ratio: Preburner O/F ratio
            Fuel-rich: 0.3-0.6 (for SSME-type)
            Ox-rich: 50-100 (for RD-180-type)
        oxidizer_rich: If True, uses ox-rich preburner (ORSC)
        mechanical_efficiency: Mechanical losses (0.95-0.98)
        tank_pressure_ox: Oxidizer tank pressure [Pa]
        tank_pressure_fuel: Fuel tank pressure [Pa]
    """

    preburner_temp: Quantity | None = None
    pump_efficiency_ox: float = 0.75
    pump_efficiency_fuel: float = 0.75
    turbine_efficiency: float = 0.70
    turbine_pressure_ratio: float = 2.0
    preburner_mixture_ratio: float = 0.5  # Fuel-rich default
    oxidizer_rich: bool = False
    mechanical_efficiency: float = 0.97
    tank_pressure_ox: Quantity | None = None
    tank_pressure_fuel: Quantity | None = None

    @property
    def cycle_type(self) -> CycleType:
        return CycleType.STAGED_COMBUSTION

    def analyze(
        self,
        inputs: EngineInputs,
        performance: EnginePerformance,
        geometry: EngineGeometry,
    ) -> CyclePerformance:
        """Analyze staged combustion cycle.

        The key difference from gas generator is that turbine exhaust
        goes to the main chamber, so there's no Isp penalty from
        dumping low-energy gases.

        The power balance is more complex because the preburner
        operates at a pressure higher than the main chamber.
        """
        warnings: list[str] = []

        # Extract values
        pc = inputs.chamber_pressure.to("Pa").value
        mdot_total = performance.mdot.to("kg/s").value
        mdot_ox = performance.mdot_ox.to("kg/s").value
        mdot_fuel = performance.mdot_fuel.to("kg/s").value
        isp = performance.isp.value

        # Set default preburner temperature
        if self.preburner_temp is not None:
            T_pb = self.preburner_temp.to("K").value
        else:
            # Default based on cycle type
            T_pb = 600.0 if self.oxidizer_rich else 800.0

        # Get propellant properties
        ox_name = "LOX"
        fuel_name = "RP1"
        if inputs.name:
            name_upper = inputs.name.upper()
            if "CH4" in name_upper or "METHANE" in name_upper:
                fuel_name = "CH4"
            elif "LH2" in name_upper or "HYDROGEN" in name_upper:
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

        # Tank pressures
        if self.tank_pressure_ox is not None:
            p_tank_ox = self.tank_pressure_ox.to("Pa").value
        else:
            p_tank_ox = 400000  # 4 bar typical for staged combustion

        if self.tank_pressure_fuel is not None:
            p_tank_fuel = self.tank_pressure_fuel.to("Pa").value
        else:
            p_tank_fuel = 350000  # 3.5 bar

        # Preburner pressure must be higher than chamber pressure
        # Turbine pressure drop + injector losses
        p_preburner = pc * 1.3  # 30% higher than chamber

        # Pump pressure rises
        # Pumps must deliver to preburner pressure (higher than PC)
        injector_dp = pc * 0.20
        p_pump_outlet = p_preburner + injector_dp

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

        # Total pump power
        P_pump_total = (P_pump_ox + P_pump_fuel) / self.mechanical_efficiency

        # Preburner/turbine analysis
        # In staged combustion, ALL propellant eventually goes through main chamber
        # Preburner flow drives turbine, then goes to main chamber

        if self.oxidizer_rich:
            # Ox-rich: most of oxidizer through preburner with small fuel
            # mdot_pb = mdot_ox + mdot_pb_fuel
            # Preburner MR is very high (50-100), so mdot_pb_fuel is small
            mdot_pb_fuel = mdot_ox / self.preburner_mixture_ratio
            mdot_pb = mdot_ox + mdot_pb_fuel
            # Remaining fuel goes directly to main chamber
            mdot_direct_fuel = mdot_fuel - mdot_pb_fuel

            if mdot_direct_fuel < 0:
                warnings.append("Preburner consumes more fuel than available - infeasible")
                mdot_direct_fuel = 0

            # Preburner exhaust is mostly oxygen with some combustion products
            gamma_pb = 1.30  # Higher gamma for ox-rich
            R_pb = 280.0  # J/(kg·K)

        else:
            # Fuel-rich: most of fuel through preburner with small oxidizer
            mdot_pb_ox = mdot_fuel * self.preburner_mixture_ratio
            mdot_pb = mdot_fuel + mdot_pb_ox
            # Remaining oxidizer goes directly to main chamber
            mdot_direct_ox = mdot_ox - mdot_pb_ox

            if mdot_direct_ox < 0:
                warnings.append("Preburner consumes more oxidizer than available - infeasible")
                mdot_direct_ox = 0

            # Preburner exhaust is fuel-rich combustion products
            gamma_pb = 1.20  # Lower gamma for fuel-rich
            R_pb = 400.0  # J/(kg·K), higher for lighter products

        # Turbine power available
        cp_pb = gamma_pb * R_pb / (gamma_pb - 1)
        T_ratio = self.turbine_pressure_ratio ** ((gamma_pb - 1) / gamma_pb)
        w_turbine = cp_pb * T_pb * self.turbine_efficiency * (1 - 1/T_ratio)

        P_turbine_available = mdot_pb * w_turbine

        # Power balance check
        power_margin = P_turbine_available / P_pump_total if P_pump_total > 0 else float('inf')

        if power_margin < 1.0:
            warnings.append(
                f"Power balance not achieved: turbine provides {P_turbine_available/1e6:.1f} MW, "
                f"pumps need {P_pump_total/1e6:.1f} MW"
            )

        # Net performance
        # In staged combustion, ALL propellant goes through main chamber
        # at (nearly) full Isp, so cycle efficiency is very high
        # Small losses from:
        # 1. Preburner inefficiency
        # 2. Slightly different combustion from preburned products

        # Estimate efficiency loss (typically 1-3%)
        efficiency_loss = 0.02  # 2% loss typical for staged combustion
        net_isp = isp * (1 - efficiency_loss)
        cycle_efficiency = 1 - efficiency_loss

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

        # Feasibility assessment
        feasible = power_margin >= 0.95  # Allow small margin

        if power_margin < 1.0:
            feasible = False

        if self.oxidizer_rich and T_pb > 700:
            warnings.append(
                f"Ox-rich preburner at {T_pb:.0f}K - requires specialized turbine materials"
            )

        if not self.oxidizer_rich and T_pb > 1000:
            warnings.append(
                f"Fuel-rich preburner temp {T_pb:.0f}K is high"
            )

        if pc > 25e6:
            warnings.append(
                f"Chamber pressure {pc/1e6:.0f} MPa is very high - "
                "typical staged combustion limit ~30 MPa"
            )

        return CyclePerformance(
            net_isp=seconds(net_isp),
            net_thrust=inputs.thrust,  # Staged combustion delivers full thrust
            cycle_efficiency=cycle_efficiency,
            pump_power_ox=Quantity(P_pump_ox, "W", "power"),
            pump_power_fuel=Quantity(P_pump_fuel, "W", "power"),
            turbine_power=Quantity(P_turbine_available, "W", "power"),
            turbine_mass_flow=kg_per_second(mdot_pb),
            tank_pressure_ox=pascals(p_tank_ox),
            tank_pressure_fuel=pascals(p_tank_fuel),
            npsh_margin_ox=npsh_ox,
            npsh_margin_fuel=npsh_fuel,
            cycle_type=self.cycle_type,
            feasible=feasible,
            warnings=warnings,
        )


@beartype
@dataclass(frozen=True, slots=True)
class FullFlowStagedCombustion:
    """Configuration for full-flow staged combustion (FFSC).

    FFSC uses TWO preburners:
    - Fuel-rich preburner: drives fuel turbopump
    - Ox-rich preburner: drives oxidizer turbopump

    This provides the highest possible performance and allows
    independent control of each turbopump.

    Example: SpaceX Raptor

    Attributes:
        fuel_preburner_temp: Fuel-rich preburner temperature [K]
        ox_preburner_temp: Ox-rich preburner temperature [K]
        pump_efficiency: Pump efficiency for both pumps
        turbine_efficiency: Turbine efficiency for both turbines
        fuel_turbine_pr: Fuel turbine pressure ratio
        ox_turbine_pr: Ox turbine pressure ratio
    """

    fuel_preburner_temp: Quantity | None = None
    ox_preburner_temp: Quantity | None = None
    pump_efficiency: float = 0.77
    turbine_efficiency: float = 0.72
    fuel_turbine_pr: float = 1.8
    ox_turbine_pr: float = 1.5
    tank_pressure_ox: Quantity | None = None
    tank_pressure_fuel: Quantity | None = None

    @property
    def cycle_type(self) -> CycleType:
        return CycleType.FULL_FLOW_STAGED

    def analyze(
        self,
        inputs: EngineInputs,
        performance: EnginePerformance,
        geometry: EngineGeometry,
    ) -> CyclePerformance:
        """Analyze full-flow staged combustion cycle."""
        warnings: list[str] = []

        # Extract values
        pc = inputs.chamber_pressure.to("Pa").value
        mdot_ox = performance.mdot_ox.to("kg/s").value
        mdot_fuel = performance.mdot_fuel.to("kg/s").value
        isp = performance.isp.value

        # Default preburner temperatures
        T_fuel_pb = self.fuel_preburner_temp.to("K").value if self.fuel_preburner_temp else 800.0
        T_ox_pb = self.ox_preburner_temp.to("K").value if self.ox_preburner_temp else 600.0

        # Get propellant properties
        try:
            rho_ox = get_propellant_density("LOX")
        except ValueError:
            rho_ox = 1141.0

        try:
            rho_fuel = get_propellant_density("CH4")  # FFSC typically uses methane
        except ValueError:
            rho_fuel = 422.0

        # Tank pressures
        p_tank_ox = self.tank_pressure_ox.to("Pa").value if self.tank_pressure_ox else 500000
        p_tank_fuel = self.tank_pressure_fuel.to("Pa").value if self.tank_pressure_fuel else 450000

        # Preburner pressures (higher than chamber)
        p_preburner = pc * 1.25

        # Pump requirements
        dp_ox = p_preburner - p_tank_ox + pc * 0.2
        dp_fuel = p_preburner - p_tank_fuel + pc * 0.2

        P_pump_ox = pump_power(
            mass_flow=kg_per_second(mdot_ox),
            pressure_rise=pascals(dp_ox),
            density=rho_ox,
            efficiency=self.pump_efficiency,
        ).value

        P_pump_fuel = pump_power(
            mass_flow=kg_per_second(mdot_fuel),
            pressure_rise=pascals(dp_fuel),
            density=rho_fuel,
            efficiency=self.pump_efficiency,
        ).value

        # In FFSC, all oxidizer goes through ox-rich preburner
        # All fuel goes through fuel-rich preburner
        # Each preburner drives its respective turbopump

        # Fuel-rich preburner (drives fuel pump)
        # Small amount of ox mixed with all fuel
        gamma_fuel_pb = 1.18
        R_fuel_pb = 450.0
        cp_fuel_pb = gamma_fuel_pb * R_fuel_pb / (gamma_fuel_pb - 1)
        T_ratio_fuel = self.fuel_turbine_pr ** ((gamma_fuel_pb - 1) / gamma_fuel_pb)
        w_fuel_turbine = cp_fuel_pb * T_fuel_pb * self.turbine_efficiency * (1 - 1/T_ratio_fuel)

        # Ox-rich preburner (drives ox pump)
        gamma_ox_pb = 1.30
        R_ox_pb = 280.0
        cp_ox_pb = gamma_ox_pb * R_ox_pb / (gamma_ox_pb - 1)
        T_ratio_ox = self.ox_turbine_pr ** ((gamma_ox_pb - 1) / gamma_ox_pb)
        w_ox_turbine = cp_ox_pb * T_ox_pb * self.turbine_efficiency * (1 - 1/T_ratio_ox)

        # Power available from each turbine
        # In FFSC, all propellant flows through preburners
        P_fuel_turbine = mdot_fuel * w_fuel_turbine
        P_ox_turbine = mdot_ox * w_ox_turbine

        # Check power balance
        fuel_margin = P_fuel_turbine / P_pump_fuel if P_pump_fuel > 0 else float('inf')
        ox_margin = P_ox_turbine / P_pump_ox if P_pump_ox > 0 else float('inf')

        feasible = True
        if fuel_margin < 0.95:
            warnings.append(f"Fuel turbopump power margin low: {fuel_margin:.2f}")
            feasible = False
        if ox_margin < 0.95:
            warnings.append(f"Ox turbopump power margin low: {ox_margin:.2f}")
            feasible = False

        # FFSC has minimal Isp loss (all propellant to main chamber)
        efficiency_loss = 0.01  # ~1% loss
        net_isp = isp * (1 - efficiency_loss)
        cycle_efficiency = 1 - efficiency_loss

        # NPSH
        p_vapor_ox = _get_vapor_pressure("LOX")
        p_vapor_fuel = _get_vapor_pressure("CH4")

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

        # Warnings
        if pc > 30e6:
            warnings.append(f"Chamber pressure {pc/1e6:.0f} MPa is at FFSC limit")

        if T_ox_pb > 700:
            warnings.append("Ox-rich preburner temp requires advanced turbine materials")

        return CyclePerformance(
            net_isp=seconds(net_isp),
            net_thrust=inputs.thrust,
            cycle_efficiency=cycle_efficiency,
            pump_power_ox=Quantity(P_pump_ox, "W", "power"),
            pump_power_fuel=Quantity(P_pump_fuel, "W", "power"),
            turbine_power=Quantity(P_fuel_turbine + P_ox_turbine, "W", "power"),
            turbine_mass_flow=kg_per_second(mdot_ox + mdot_fuel),  # All flow through preburners
            tank_pressure_ox=pascals(p_tank_ox),
            tank_pressure_fuel=pascals(p_tank_fuel),
            npsh_margin_ox=npsh_ox,
            npsh_margin_fuel=npsh_fuel,
            cycle_type=self.cycle_type,
            feasible=feasible,
            warnings=warnings,
        )

