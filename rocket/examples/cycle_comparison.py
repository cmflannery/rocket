#!/usr/bin/env python
"""Engine cycle comparison example for Rocket.

This example demonstrates how to compare different engine cycle architectures
for a given set of requirements:

1. Pressure-fed (simplest, lowest performance)
2. Gas generator (most common, good balance)
3. Staged combustion (highest performance, most complex)

Understanding cycle tradeoffs is critical for:
- Selecting the right architecture for your mission
- Understanding performance vs. complexity tradeoffs
- Estimating system-level impacts (tank pressure, turbomachinery)
"""

from rocket import (
    EngineInputs,
    OutputContext,
    design_engine,
    plot_cycle_comparison_bars,
    plot_cycle_radar,
    plot_cycle_tradeoff,
)
from rocket.cycles import (
    GasGeneratorCycle,
    PressureFedCycle,
    StagedCombustionCycle,
)
from rocket.units import kelvin, kilonewtons, megapascals


def print_header(text: str) -> None:
    """Print a formatted section header."""
    print()
    print("┌" + "─" * 68 + "┐")
    print(f"│ {text:<66} │")
    print("└" + "─" * 68 + "┘")


def main() -> None:
    """Run the engine cycle comparison example."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 16 + "ENGINE CYCLE COMPARISON STUDY" + " " * 23 + "║")
    print("║" + " " * 18 + "LOX/CH4 Methalox Engine" + " " * 27 + "║")
    print("╚" + "═" * 68 + "╝")

    # =========================================================================
    # Common Engine Requirements
    # =========================================================================

    print_header("ENGINE REQUIREMENTS")

    # Base engine thermochemistry (same for all cycles)
    base_inputs = EngineInputs.from_propellants(
        oxidizer="LOX",
        fuel="CH4",
        thrust=kilonewtons(500),
        chamber_pressure=megapascals(15),
        mixture_ratio=3.2,
        name="Methalox-500",
    )

    print(f"  Thrust target:      {base_inputs.thrust.to('kN').value:.0f} kN")
    print(f"  Chamber pressure:   {base_inputs.chamber_pressure.to('MPa').value:.0f} MPa")
    print("  Propellants:        LOX / CH4")
    print(f"  Mixture ratio:      {base_inputs.mixture_ratio}")
    print()

    # Get baseline performance and geometry
    performance, geometry = design_engine(base_inputs)

    print(f"  Ideal Isp (vac):    {performance.isp_vac.value:.1f} s")
    print(f"  Ideal c*:           {performance.cstar.to('m/s').value:.0f} m/s")
    print(f"  Mass flow:          {performance.mdot.to('kg/s').value:.1f} kg/s")

    # Store results for comparison
    results: list[dict] = []

    # =========================================================================
    # Cycle 1: Pressure-Fed
    # =========================================================================

    print_header("CYCLE 1: PRESSURE-FED")

    print("  Architecture:")
    print("    - Propellants pushed by tank pressure (no pumps)")
    print("    - Requires high tank pressure → heavy tanks")
    print("    - Simplest system, highest reliability")
    print()

    pressure_fed = PressureFedCycle(
        tank_pressure_margin=1.3,
        line_loss_fraction=0.05,
        injector_dp_fraction=0.15,
    )

    pf_result = pressure_fed.analyze(base_inputs, performance, geometry)

    print("  Results:")
    print(f"    Tank pressure (ox):      {pf_result.tank_pressure_ox.to('MPa').value:.1f} MPa")
    print(f"    Tank pressure (fuel):    {pf_result.tank_pressure_fuel.to('MPa').value:.1f} MPa")
    print(f"    Net Isp:                 {pf_result.net_isp.to('s').value:.1f} s")
    print(f"    Net thrust:              {pf_result.net_thrust.to('kN').value:.0f} kN")
    print(f"    Cycle efficiency:        {pf_result.cycle_efficiency*100:.1f}%")
    if pf_result.warnings:
        print(f"    Warnings: {len(pf_result.warnings)}")
        for w in pf_result.warnings:
            print(f"      ⚠ {w}")

    results.append({
        "name": "Pressure-Fed",
        "net_isp": pf_result.net_isp.to("s").value,
        "tank_pressure_MPa": pf_result.tank_pressure_ox.to("MPa").value,
        "pump_power_kW": 0,
        "efficiency": pf_result.cycle_efficiency,
        "simplicity": 1.0,
        "reliability": 0.95,
    })

    # =========================================================================
    # Cycle 2: Gas Generator
    # =========================================================================

    print_header("CYCLE 2: GAS GENERATOR")

    print("  Architecture:")
    print("    - Small combustor (GG) drives turbine")
    print("    - Turbine exhaust dumped overboard (Isp loss)")
    print("    - Moderate complexity, proven technology")
    print()

    gas_generator = GasGeneratorCycle(
        turbine_inlet_temp=kelvin(900),
        pump_efficiency_ox=0.70,
        pump_efficiency_fuel=0.70,
        turbine_efficiency=0.65,
        gg_mixture_ratio=0.4,
    )

    gg_result = gas_generator.analyze(base_inputs, performance, geometry)

    total_pump_power_gg = (
        gg_result.pump_power_ox.to("kW").value +
        gg_result.pump_power_fuel.to("kW").value
    )

    print("  Results:")
    print(f"    Turbine mass flow:       {gg_result.turbine_mass_flow.to('kg/s').value:.1f} kg/s")
    print(f"    Net Isp:                 {gg_result.net_isp.to('s').value:.1f} s")
    print(f"    Net thrust:              {gg_result.net_thrust.to('kN').value:.0f} kN")
    print(f"    Cycle efficiency:        {gg_result.cycle_efficiency*100:.1f}%")
    print(f"    Isp loss vs ideal:       {performance.isp_vac.value - gg_result.net_isp.to('s').value:.1f} s")
    if gg_result.warnings:
        print(f"    Warnings: {len(gg_result.warnings)}")
        for w in gg_result.warnings:
            print(f"      ⚠ {w}")

    results.append({
        "name": "Gas Generator",
        "net_isp": gg_result.net_isp.to("s").value,
        "tank_pressure_MPa": gg_result.tank_pressure_ox.to("MPa").value if gg_result.tank_pressure_ox else 0.5,
        "pump_power_kW": total_pump_power_gg,
        "efficiency": gg_result.cycle_efficiency,
        "simplicity": 0.6,
        "reliability": 0.90,
    })

    # =========================================================================
    # Cycle 3: Staged Combustion (Oxidizer-Rich)
    # =========================================================================

    print_header("CYCLE 3: STAGED COMBUSTION (OX-RICH)")

    print("  Architecture:")
    print("    - Preburner runs oxygen-rich")
    print("    - ALL flow goes through main chamber (no dump)")
    print("    - Highest performance, most complex")
    print()

    staged_combustion = StagedCombustionCycle(
        preburner_temp=kelvin(750),
        pump_efficiency_ox=0.75,
        pump_efficiency_fuel=0.75,
        turbine_efficiency=0.70,
        preburner_mixture_ratio=50.0,
        oxidizer_rich=True,
    )

    sc_result = staged_combustion.analyze(base_inputs, performance, geometry)

    total_pump_power_sc = (
        sc_result.pump_power_ox.to("kW").value +
        sc_result.pump_power_fuel.to("kW").value
    )

    print("  Results:")
    print(f"    Turbine mass flow:       {sc_result.turbine_mass_flow.to('kg/s').value:.1f} kg/s")
    print(f"    Net Isp:                 {sc_result.net_isp.to('s').value:.1f} s")
    print(f"    Net thrust:              {sc_result.net_thrust.to('kN').value:.0f} kN")
    print(f"    Cycle efficiency:        {sc_result.cycle_efficiency*100:.1f}%")
    if sc_result.warnings:
        print(f"    Warnings: {len(sc_result.warnings)}")
        for w in sc_result.warnings:
            print(f"      ⚠ {w}")

    results.append({
        "name": "Staged Combustion",
        "net_isp": sc_result.net_isp.to("s").value,
        "tank_pressure_MPa": sc_result.tank_pressure_ox.to("MPa").value if sc_result.tank_pressure_ox else 0.5,
        "pump_power_kW": total_pump_power_sc,
        "efficiency": sc_result.cycle_efficiency,
        "simplicity": 0.3,
        "reliability": 0.85,
    })

    # =========================================================================
    # Comparison Summary
    # =========================================================================

    print_header("COMPARISON SUMMARY")

    print()
    print(f"  {'Cycle':<20} {'Net Isp':<12} {'Efficiency':<12} {'Tank P (MPa)':<12}")
    print("  " + "-" * 56)

    for r in results:
        isp_str = f"{r['net_isp']:.1f} s"
        eff_str = f"{r['efficiency']*100:.1f}%"
        tank_str = f"{r['tank_pressure_MPa']:.1f}"
        print(f"  {r['name']:<20} {isp_str:<12} {eff_str:<12} {tank_str:<12}")

    # Compute comparison from actual results
    if len(results) >= 3:
        isp_gain = results[2]["net_isp"] - results[1]["net_isp"]
        isp_gain_pct = 100 * isp_gain / results[1]["net_isp"] if results[1]["net_isp"] > 0 else 0

        print()
        print("  Staged combustion vs Gas Generator:")
        print(f"    Isp gain: {isp_gain:.1f} s ({isp_gain_pct:.1f}%)")

    print()

    # =========================================================================
    # Save Results
    # =========================================================================

    print_header("SAVING RESULTS")

    with OutputContext("cycle_comparison", include_timestamp=True) as ctx:
        # Generate visualizations
        print()
        print("  Generating visualizations...")

        # 1. Bar chart comparison
        fig_bars = plot_cycle_comparison_bars(
            results,
            metrics=["net_isp", "efficiency", "tank_pressure_MPa", "pump_power_kW"],
            title="LOX/CH4 Engine Cycle Comparison",
        )
        fig_bars.savefig(ctx.path("cycle_comparison_bars.png"), dpi=150, bbox_inches="tight")
        print("    - cycle_comparison_bars.png")

        # 2. Radar chart
        fig_radar = plot_cycle_radar(
            results,
            metrics=["net_isp", "efficiency", "simplicity", "reliability"],
            title="Cycle Trade-offs (Normalized)",
        )
        fig_radar.savefig(ctx.path("cycle_radar.png"), dpi=150, bbox_inches="tight")
        print("    - cycle_radar.png")

        # 3. Trade-off scatter plot
        fig_tradeoff = plot_cycle_tradeoff(
            results,
            x_metric="net_isp",
            y_metric="simplicity",
            size_metric="pump_power_kW",
            title="Performance vs Simplicity Trade Space",
        )
        fig_tradeoff.savefig(ctx.path("cycle_tradeoff.png"), dpi=150, bbox_inches="tight")
        print("    - cycle_tradeoff.png")

        # Save summary data
        ctx.save_summary({
            "requirements": {
                "thrust_kN": base_inputs.thrust.to("kN").value,
                "chamber_pressure_MPa": base_inputs.chamber_pressure.to("MPa").value,
                "propellants": "LOX/CH4",
                "mixture_ratio": base_inputs.mixture_ratio,
            },
            "ideal_performance": {
                "isp_vac_s": performance.isp_vac.value,
                "cstar_m_s": performance.cstar.value,
                "mdot_kg_s": performance.mdot.value,
            },
            "cycles": results,
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
