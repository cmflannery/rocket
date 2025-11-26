#!/usr/bin/env python
"""Vehicle sizing example for Rocket.

This example demonstrates the complete workflow from mission requirements
to vehicle sizing with professional visualizations:
1. Define mission delta-V
2. Size engine from propellants
3. Calculate propellant mass
4. Size propellant tanks
5. Generate vehicle stack and mass breakdown visualizations
"""

from rocket import (
    EngineInputs,
    design_engine,
    size_propellant,
    size_tank,
)
from rocket.nozzle import full_chamber_contour, generate_nozzle_from_geometry
from rocket.plotting import (
    plot_engine_dashboard,
    plot_mass_breakdown,
)
from rocket.units import kilograms, kilonewtons, km_per_second, megapascals, pascals


def print_header(text: str) -> None:
    """Print a formatted section header."""
    print()
    print("┌" + "─" * 68 + "┐")
    print(f"│ {text:<66} │")
    print("└" + "─" * 68 + "┘")


def print_table(rows: list[tuple[str, str]], title: str = "") -> None:
    """Print a formatted table."""
    if title:
        print(f"\n  {title}")
        print("  " + "─" * 40)
    for label, value in rows:
        print(f"  {label:<24} {value:>14}")


def main() -> None:
    """Run the vehicle sizing example."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "ROCKET VEHICLE SIZING TOOL" + " " * 24 + "║")
    print("║" + " " * 20 + "LOX/CH4 SSTO Concept" + " " * 28 + "║")
    print("╚" + "═" * 68 + "╝")

    # =========================================================================
    # Mission Requirements
    # =========================================================================

    print_header("MISSION REQUIREMENTS")

    delta_v = km_per_second(9.5)
    payload_mass = kilograms(1000)

    print_table([
        ("Target orbit", "LEO (400 km)"),
        ("Delta-V requirement", f"{delta_v.to('km/s').value:.1f} km/s"),
        ("Payload mass", f"{payload_mass.to('kg').value:,.0f} kg"),
    ])

    # =========================================================================
    # Engine Design
    # =========================================================================

    print_header("ENGINE DESIGN")

    engine_inputs = EngineInputs.from_propellants(
        oxidizer="LOX",
        fuel="CH4",
        thrust=kilonewtons(500),
        chamber_pressure=megapascals(10),
        mixture_ratio=3.2,
        name="Methalox-500",
    )

    performance, geometry = design_engine(engine_inputs)

    print_table([
        ("Engine name", engine_inputs.name),
        ("Propellants", "LOX / CH4"),
        ("Thrust", f"{engine_inputs.thrust.to('kN').value:.0f} kN"),
        ("Chamber pressure", f"{engine_inputs.chamber_pressure.to('MPa').value:.0f} MPa"),
        ("Chamber temperature", f"{engine_inputs.chamber_temp.to('K').value:.0f} K"),
        ("Mixture ratio (O/F)", f"{engine_inputs.mixture_ratio}"),
    ], "Configuration")

    print_table([
        ("Isp (sea level)", f"{performance.isp.value:.1f} s"),
        ("Isp (vacuum)", f"{performance.isp_vac.value:.1f} s"),
        ("Thrust coefficient", f"{performance.thrust_coeff:.3f}"),
        ("c*", f"{performance.cstar.to('m/s').value:.0f} m/s"),
        ("Mass flow", f"{performance.mdot.value:.1f} kg/s"),
    ], "Performance")

    print_table([
        ("Throat diameter", f"{geometry.throat_diameter.to('m').value * 100:.1f} cm"),
        ("Exit diameter", f"{geometry.exit_diameter.to('m').value * 100:.1f} cm"),
        ("Expansion ratio", f"{geometry.expansion_ratio:.1f}"),
        ("Chamber diameter", f"{geometry.chamber_diameter.to('m').value * 100:.1f} cm"),
        ("Chamber length", f"{geometry.chamber_length.to('m').value * 100:.1f} cm"),
    ], "Geometry")

    # =========================================================================
    # Propellant Sizing
    # =========================================================================

    print_header("PROPELLANT SIZING")

    # Estimate structure mass
    structure_mass_estimate = kilograms(3000)
    dry_mass = kilograms(payload_mass.value + structure_mass_estimate.value)
    isp_avg = (performance.isp.value + performance.isp_vac.value) / 2

    propellant = size_propellant(
        isp_s=isp_avg,
        delta_v=delta_v,
        dry_mass=dry_mass,
        mixture_ratio=engine_inputs.mixture_ratio,
        mdot=performance.mdot,
    )

    print_table([
        ("Average Isp used", f"{isp_avg:.1f} s"),
        ("Initial dry mass est.", f"{dry_mass.to('kg').value:,.0f} kg"),
    ], "Assumptions")

    print_table([
        ("Oxidizer (LOX)", f"{propellant.oxidizer_mass.to('kg').value:,.0f} kg"),
        ("Fuel (CH4)", f"{propellant.fuel_mass.to('kg').value:,.0f} kg"),
        ("Total propellant", f"{propellant.total_propellant.to('kg').value:,.0f} kg"),
        ("Mass ratio", f"{propellant.mass_ratio:.2f}"),
        ("Burn time", f"{propellant.burn_time.to('s').value:.0f} s"),
    ], "Propellant Requirements")

    # =========================================================================
    # Tank Sizing
    # =========================================================================

    print_header("TANK SIZING")

    lox_tank = size_tank(
        propellant_mass=propellant.oxidizer_mass,
        propellant="LOX",
        tank_pressure=pascals(300000),
        material="Al2195",
    )

    print_table([
        ("Volume", f"{lox_tank.volume.to('m^3').value:.2f} m³"),
        ("Diameter", f"{lox_tank.diameter.to('m').value:.2f} m"),
        ("Total length", f"{lox_tank.total_length.to('m').value:.2f} m"),
        ("Wall thickness", f"{lox_tank.wall_thickness.to('m').value * 1000:.1f} mm"),
        ("Tank mass", f"{lox_tank.dry_mass.to('kg').value:.0f} kg"),
    ], "LOX Tank (Al2195)")

    ch4_tank = size_tank(
        propellant_mass=propellant.fuel_mass,
        propellant="CH4",
        tank_pressure=pascals(250000),
        material="Al2195",
    )

    print_table([
        ("Volume", f"{ch4_tank.volume.to('m^3').value:.2f} m³"),
        ("Diameter", f"{ch4_tank.diameter.to('m').value:.2f} m"),
        ("Total length", f"{ch4_tank.total_length.to('m').value:.2f} m"),
        ("Wall thickness", f"{ch4_tank.wall_thickness.to('m').value * 1000:.1f} mm"),
        ("Tank mass", f"{ch4_tank.dry_mass.to('kg').value:.0f} kg"),
    ], "CH4 Tank (Al2195)")

    # =========================================================================
    # Vehicle Mass Summary
    # =========================================================================

    print_header("VEHICLE MASS SUMMARY")

    # Component masses
    engine_mass = 300.0
    avionics_mass = 100.0
    misc_structure = 500.0

    masses = {
        "Payload": payload_mass.value,
        "Engine": engine_mass,
        "LOX Tank": lox_tank.dry_mass.value,
        "CH4 Tank": ch4_tank.dry_mass.value,
        "Avionics": avionics_mass,
        "Structure": misc_structure,
        "LOX": propellant.oxidizer_mass.value,
        "CH4": propellant.fuel_mass.value,
    }

    total_dry = sum(v for k, v in masses.items() if k not in ["LOX", "CH4"])
    total_wet = sum(masses.values())

    print_table([
        ("Payload", f"{masses['Payload']:,.0f} kg"),
        ("Engine", f"{masses['Engine']:,.0f} kg"),
        ("LOX Tank", f"{masses['LOX Tank']:,.0f} kg"),
        ("CH4 Tank", f"{masses['CH4 Tank']:,.0f} kg"),
        ("Avionics", f"{masses['Avionics']:,.0f} kg"),
        ("Structure", f"{masses['Structure']:,.0f} kg"),
    ], "Dry Mass Components")

    print()
    print(f"  {'─' * 40}")
    print(f"  {'TOTAL DRY MASS':<24} {total_dry:>14,.0f} kg")
    print(f"  {'Propellant (LOX + CH4)':<24} {propellant.total_propellant.value:>14,.0f} kg")
    print(f"  {'─' * 40}")
    print(f"  {'TOTAL WET MASS':<24} {total_wet:>14,.0f} kg")

    # Key metrics
    twr = engine_inputs.thrust.to("N").value / (total_wet * 9.80665)
    structure_fraction = (total_dry - payload_mass.value) / propellant.total_propellant.value

    print_table([
        ("Thrust-to-weight ratio", f"{twr:.2f}"),
        ("Structure fraction", f"{structure_fraction * 100:.1f}%"),
        ("Propellant fraction", f"{propellant.total_propellant.value / total_wet * 100:.1f}%"),
    ], "Performance Metrics")

    # =========================================================================
    # Generate Visualizations
    # =========================================================================

    print_header("GENERATING VISUALIZATIONS")

    # 1. Engine dashboard
    print("  [1/2] Engine dashboard...")
    nozzle = generate_nozzle_from_geometry(geometry)
    contour = full_chamber_contour(engine_inputs, geometry, nozzle)
    fig_engine = plot_engine_dashboard(engine_inputs, performance, geometry, contour)
    fig_engine.savefig("engine_dashboard.png", dpi=150, bbox_inches="tight")
    print("        → Saved: engine_dashboard.png")

    # 2. Mass breakdown
    print("  [2/2] Mass breakdown...")
    fig_mass = plot_mass_breakdown(masses, title="Vehicle Mass Breakdown")
    fig_mass.savefig("mass_breakdown.png", dpi=150, bbox_inches="tight")
    print("        → Saved: mass_breakdown.png")

    # Final summary
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 26 + "DESIGN COMPLETE" + " " * 27 + "║")
    print("╠" + "═" * 68 + "╣")
    print("║  Vehicle: Methalox SSTO                                           ║")
    print(f"║  Payload: {payload_mass.value:,.0f} kg to LEO                                        ║")
    print(f"║  Wet Mass: {total_wet:,.0f} kg                                           ║")
    print(f"║  T/W: {twr:.2f}                                                        ║")
    print("╠" + "═" * 68 + "╣")
    print("║  Output files:                                                     ║")
    print("║    • engine_dashboard.png                                          ║")
    print("║    • mass_breakdown.png                                            ║")
    print("╚" + "═" * 68 + "╝")
    print()


if __name__ == "__main__":
    main()
