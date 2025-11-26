#!/usr/bin/env python
"""Thermal analysis example for Rocket.

This example demonstrates thermal screening to determine if an engine
design can be regeneratively cooled:

1. Estimate heat flux using the Bartz correlation
2. Check cooling feasibility for different coolants
3. Understand the relationship between chamber pressure and cooling

High chamber pressure engines (like Raptor) face severe thermal challenges.
This analysis helps catch infeasible designs early.
"""

from openrocketengine import (
    EngineInputs,
    OutputContext,
    design_engine,
)
from openrocketengine.thermal import (
    check_cooling_feasibility,
    estimate_heat_flux,
    heat_flux_profile,
)
from openrocketengine.units import kelvin, kilonewtons, megapascals


def print_header(text: str) -> None:
    """Print a formatted section header."""
    print()
    print("┌" + "─" * 68 + "┐")
    print(f"│ {text:<66} │")
    print("└" + "─" * 68 + "┘")


def main() -> None:
    """Run the thermal analysis example."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "THERMAL FEASIBILITY ANALYSIS" + " " * 22 + "║")
    print("║" + " " * 15 + "Can This Engine Be Regeneratively Cooled?" + " " * 12 + "║")
    print("╚" + "═" * 68 + "╝")

    # =========================================================================
    # Design a High-Performance Engine
    # =========================================================================

    print_header("ENGINE DESIGN")

    # High chamber pressure like Raptor
    inputs = EngineInputs.from_propellants(
        oxidizer="LOX",
        fuel="CH4",
        thrust=kilonewtons(500),
        chamber_pressure=megapascals(25),  # High pressure!
        mixture_ratio=3.2,
        name="High-Pc-Methalox",
    )

    performance, geometry = design_engine(inputs)

    print(f"  Engine: {inputs.name}")
    print(f"  Thrust: {inputs.thrust.to('kN').value:.0f} kN")
    print(f"  Chamber pressure: {inputs.chamber_pressure.to('MPa').value:.0f} MPa")
    print(f"  Chamber temperature: {inputs.chamber_temp.to('K').value:.0f} K")
    print()
    print(f"  Isp (vac): {performance.isp_vac.value:.1f} s")
    print(f"  Throat diameter: {geometry.throat_diameter.to('m').value*100:.2f} cm")

    # =========================================================================
    # Heat Flux Estimation
    # =========================================================================

    print_header("HEAT FLUX ANALYSIS")

    print("  Using Bartz correlation for convective heat transfer...")
    print()

    # Estimate heat flux at key locations
    locations = ["chamber", "throat", "exit"]

    print(f"  {'Location':<15} {'Heat Flux (MW/m²)':<20}")
    print("  " + "-" * 35)

    max_q = 0.0
    max_location = ""

    for location in locations:
        q = estimate_heat_flux(
            inputs=inputs,
            performance=performance,
            geometry=geometry,
            location=location,
        )
        q_mw = q.to("W/m^2").value / 1e6

        if q_mw > max_q:
            max_q = q_mw
            max_location = location

        print(f"  {location:<15} {q_mw:<20.1f}")

    print()
    print(f"  Maximum heat flux: {max_q:.1f} MW/m² at {max_location}")

    # =========================================================================
    # Heat Flux Profile Along Nozzle
    # =========================================================================

    print_header("AXIAL HEAT FLUX PROFILE")

    x_positions, heat_fluxes = heat_flux_profile(
        inputs=inputs,
        performance=performance,
        geometry=geometry,
        n_points=11,
    )

    print(f"  {'x/L':<10} {'Heat Flux (MW/m²)':<20}")
    print("  " + "-" * 30)

    for x, q in zip(x_positions, heat_fluxes, strict=True):
        q_mw = q / 1e6  # heat_flux_profile returns W/m²
        bar = "█" * int(q_mw / 5)  # Simple bar chart
        print(f"  {x:<10.2f} {q_mw:<10.1f} {bar}")

    # =========================================================================
    # Cooling Feasibility Check - Methane
    # =========================================================================

    print_header("COOLING FEASIBILITY: METHANE (CH4)")

    print("  Checking if methane can cool this engine...")
    print()

    ch4_cooling = check_cooling_feasibility(
        inputs=inputs,
        performance=performance,
        geometry=geometry,
        coolant="CH4",
        coolant_inlet_temp=kelvin(110),  # Near boiling point
        max_wall_temp=kelvin(920),  # Material limit
    )

    if ch4_cooling.feasible:
        print("  ✓ FEASIBLE with methane cooling!")
    else:
        print("  ✗ NOT FEASIBLE with methane cooling")

    print()
    print(f"  Max wall temperature:    {ch4_cooling.max_wall_temp.to('K').value:.0f} K (limit: {ch4_cooling.max_allowed_temp.to('K').value:.0f} K)")
    print(f"  Throat heat flux:        {ch4_cooling.throat_heat_flux.to('W/m^2').value/1e6:.1f} MW/m²")
    print(f"  Coolant outlet temp:     {ch4_cooling.coolant_outlet_temp.to('K').value:.0f} K")
    print(f"  Flow margin:             {ch4_cooling.flow_margin:.2f}x")
    if ch4_cooling.warnings:
        print(f"  Warnings:")
        for w in ch4_cooling.warnings:
            print(f"    ⚠ {w}")

    # =========================================================================
    # Cooling Feasibility Check - RP-1 (for comparison)
    # =========================================================================

    print_header("COOLING FEASIBILITY: RP-1 (KEROSENE)")

    # Design a kerosene engine for comparison
    rp1_inputs = EngineInputs.from_propellants(
        oxidizer="LOX",
        fuel="RP1",
        thrust=kilonewtons(500),
        chamber_pressure=megapascals(25),
        mixture_ratio=2.7,
        name="High-Pc-Kerolox",
    )
    rp1_perf, rp1_geom = design_engine(rp1_inputs)

    print("  Checking RP-1 cooling for LOX/RP-1 engine...")
    print()

    rp1_cooling = check_cooling_feasibility(
        inputs=rp1_inputs,
        performance=rp1_perf,
        geometry=rp1_geom,
        coolant="RP1",
        coolant_inlet_temp=kelvin(300),  # Room temperature
        max_wall_temp=kelvin(920),
    )

    if rp1_cooling.feasible:
        print("  ✓ FEASIBLE with RP-1 cooling!")
    else:
        print("  ✗ NOT FEASIBLE with RP-1 cooling")

    print()
    print(f"  Max wall temperature:    {rp1_cooling.max_wall_temp.to('K').value:.0f} K")
    print(f"  Coolant outlet temp:     {rp1_cooling.coolant_outlet_temp.to('K').value:.0f} K")
    print(f"  Flow margin:             {rp1_cooling.flow_margin:.2f}x")

    # =========================================================================
    # Chamber Pressure Impact Study
    # =========================================================================

    print_header("CHAMBER PRESSURE vs. COOLING FEASIBILITY")

    print("  How does Pc affect cooling feasibility?")
    print()
    print(f"  {'Pc (MPa)':<12} {'Throat q (MW/m²)':<20} {'Coolable?':<12}")
    print("  " + "-" * 44)

    for pc_mpa in [5, 10, 15, 20, 25, 30]:
        test_inputs = EngineInputs.from_propellants(
            oxidizer="LOX",
            fuel="CH4",
            thrust=kilonewtons(500),
            chamber_pressure=megapascals(pc_mpa),
            mixture_ratio=3.2,
            name=f"Test-{pc_mpa}MPa",
        )
        test_perf, test_geom = design_engine(test_inputs)

        q_throat = estimate_heat_flux(test_inputs, test_perf, test_geom, location="throat")
        q_mw = q_throat.to("W/m^2").value / 1e6

        cooling = check_cooling_feasibility(
            test_inputs, test_perf, test_geom,
            coolant="CH4",
            coolant_inlet_temp=kelvin(110),
            max_wall_temp=kelvin(920),
        )

        status = "✓ Yes" if cooling.feasible else "✗ No"
        print(f"  {pc_mpa:<12} {q_mw:<20.1f} {status:<12}")

    # =========================================================================
    # Save Results
    # =========================================================================

    print_header("SAVING RESULTS")

    with OutputContext("thermal_analysis", include_timestamp=True) as ctx:
        ctx.save_summary({
            "engine": {
                "name": inputs.name,
                "thrust_kN": inputs.thrust.to("kN").value,
                "chamber_pressure_MPa": inputs.chamber_pressure.to("MPa").value,
                "chamber_temp_K": inputs.chamber_temp.to("K").value,
            },
            "heat_flux": {
                "max_MW_m2": max_q,
                "max_location": max_location,
            },
            "ch4_cooling": {
                "feasible": ch4_cooling.feasible,
                "max_wall_temp_K": ch4_cooling.max_wall_temp.value,
                "flow_margin": ch4_cooling.flow_margin,
            },
            "rp1_cooling": {
                "feasible": rp1_cooling.feasible,
                "max_wall_temp_K": rp1_cooling.max_wall_temp.value,
                "flow_margin": rp1_cooling.flow_margin,
            },
        })

        print()
        print(f"  Results saved to: {ctx.output_dir}")

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 24 + "ANALYSIS COMPLETE" + " " * 27 + "║")
    print("╚" + "═" * 68 + "╝")
    print()


if __name__ == "__main__":
    main()
