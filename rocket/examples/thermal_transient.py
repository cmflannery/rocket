#!/usr/bin/env python
"""Transient thermal analysis example for Rocket.

This example demonstrates time-varying thermal simulations:

1. Engine startup transient (wall temp rise to steady state)
2. Engine shutdown analysis (thermal soak-back)
3. Duty cycle simulation (multiple burn/coast cycles)
4. Material comparison for liner selection

These are compute-intensive analyses that model the time evolution
of wall temperatures under varying heat flux conditions.

All outputs are saved to an organized directory structure.
"""

import polars as pl

from rocket import OutputContext
from rocket.engine import EngineInputs, design_engine
from rocket.plotting import (
    plot_duty_cycle_thermal,
    plot_thermal_margin,
    plot_thermal_transient,
)
from rocket.thermal import (
    get_material_properties,
    list_materials,
    simulate_duty_cycle,
    simulate_shutdown,
    simulate_startup,
)
from rocket.units import kelvin, megapascals, meters, newtons


def main() -> None:
    """Run transient thermal analysis examples."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "TRANSIENT THERMAL ANALYSIS".center(68) + "║")
    print("║" + "Time-Varying Heat Transfer During Engine Operation".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # =========================================================================
    # Design the Engine
    # =========================================================================

    print("┌" + "─" * 68 + "┐")
    print("│ ENGINE DESIGN".ljust(69) + "│")
    print("└" + "─" * 68 + "┘")

    # High-performance methalox engine (like Raptor-class)
    inputs = EngineInputs.from_propellants(
        oxidizer="LOX",
        fuel="CH4",
        thrust=newtons(500_000),  # 500 kN
        chamber_pressure=megapascals(20.0),  # 20 MPa
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
    print()

    # Wall parameters
    wall_thickness = meters(0.003)  # 3 mm copper liner
    coolant_temp = kelvin(110.0)  # LCH4 inlet temp

    print(f"  Wall thickness: {wall_thickness.to('m').value*1000:.1f} mm")
    print(f"  Coolant inlet temp: {coolant_temp.to('K').value:.0f} K")
    print()

    with OutputContext("thermal_transient_results", include_timestamp=True) as ctx:
        ctx.add_metadata("engine_name", inputs.name)
        ctx.add_metadata("thrust_kN", inputs.thrust.to("kN").value)
        ctx.add_metadata("chamber_pressure_MPa", inputs.chamber_pressure.to("MPa").value)

        # =====================================================================
        # Study 1: Engine Startup Transient
        # =====================================================================

        print("┌" + "─" * 68 + "┐")
        print("│ STUDY 1: ENGINE STARTUP TRANSIENT".ljust(69) + "│")
        print("└" + "─" * 68 + "┘")
        print()

        print("  Simulating throat thermal response during ignition...")
        print("  - Heat flux ramps from 0 to steady state over 2 seconds")
        print("  - Then runs at steady state for 3 seconds")
        print()

        startup_result = simulate_startup(
            inputs, performance, geometry,
            wall_thickness=wall_thickness,
            wall_material="grcop84",  # NASA copper alloy
            coolant_temp=coolant_temp,
            startup_time=2.0,
            steady_time=3.0,
            dt=0.001,
            location="throat",
            startup_profile="ramp",
        )

        print("  Results:")
        print(f"    Total simulation time:  {startup_result.duration:.1f} s")
        print(f"    Peak inner wall temp:   {startup_result.peak_wall_temp:.0f} K")
        print(f"    Material limit:         {startup_result.max_material_temp:.0f} K")
        print(f"    Min thermal margin:     {startup_result.min_thermal_margin:.0f} K")
        print(f"    Status:                 {'SAFE' if startup_result.is_safe() else 'EXCEEDS LIMIT'}")
        print()

        tss = startup_result.time_to_steady_state
        if tss is not None:
            print(f"    Time to steady state:   {tss:.2f} s")
        else:
            print("    Time to steady state:   Not reached")

        print(f"    Peak heat flux:         {startup_result.heat_flux.max()/1e6:.1f} MW/m²")
        print()

        # Generate plots
        ctx.log("Generating startup transient plots...")

        fig_startup = plot_thermal_transient(
            startup_result.time,
            startup_result.wall_temp_inner,
            startup_result.wall_temp_outer,
            startup_result.heat_flux,
            startup_result.max_material_temp,
            title="Engine Startup Thermal Transient (Throat)",
        )
        fig_startup.savefig(ctx.path("startup_transient.png"), dpi=150, bbox_inches="tight")

        fig_margin = plot_thermal_margin(
            startup_result.time,
            startup_result.thermal_margin,
            title="Thermal Margin During Startup",
        )
        fig_margin.savefig(ctx.path("startup_margin.png"), dpi=150, bbox_inches="tight")

        # =====================================================================
        # Study 2: Engine Shutdown Analysis
        # =====================================================================

        print("┌" + "─" * 68 + "┐")
        print("│ STUDY 2: ENGINE SHUTDOWN ANALYSIS".ljust(69) + "│")
        print("└" + "─" * 68 + "┘")
        print()

        print("  Simulating thermal soak-back after engine cutoff...")
        print("  - Heat flux decays exponentially over 0.5 seconds")
        print("  - Coolant continues flowing for 10 seconds")
        print()

        shutdown_result = simulate_shutdown(
            inputs, performance, geometry,
            wall_thickness=wall_thickness,
            wall_material="grcop84",
            coolant_temp=coolant_temp,
            shutdown_time=0.5,
            cooling_time=10.0,
            dt=0.001,
            location="throat",
            coolant_continues=True,
        )

        print("  Results:")
        print(f"    Initial wall temp:      {shutdown_result.wall_temp_inner[0]:.0f} K")
        print(f"    Final wall temp:        {shutdown_result.wall_temp_inner[-1]:.0f} K")
        print(f"    Peak temp (soak-back):  {shutdown_result.peak_wall_temp:.0f} K")
        print(f"    Cooling rate:           {(shutdown_result.wall_temp_inner[0] - shutdown_result.wall_temp_inner[-1]) / shutdown_result.duration:.1f} K/s")
        print()

        ctx.log("Generating shutdown plots...")
        fig_shutdown = plot_thermal_transient(
            shutdown_result.time,
            shutdown_result.wall_temp_inner,
            shutdown_result.wall_temp_outer,
            shutdown_result.heat_flux,
            shutdown_result.max_material_temp,
            title="Engine Shutdown Thermal Response",
        )
        fig_shutdown.savefig(ctx.path("shutdown_transient.png"), dpi=150, bbox_inches="tight")

        # =====================================================================
        # Study 3: Duty Cycle Analysis
        # =====================================================================

        print("┌" + "─" * 68 + "┐")
        print("│ STUDY 3: DUTY CYCLE ANALYSIS".ljust(69) + "│")
        print("└" + "─" * 68 + "┘")
        print()

        print("  Simulating pulsed operation (e.g., landing engine)...")
        print("  - 3-second burns followed by 5-second coasts")
        print("  - 4 complete cycles")
        print()

        duty_result = simulate_duty_cycle(
            inputs, performance, geometry,
            wall_thickness=wall_thickness,
            burn_time=3.0,
            coast_time=5.0,
            n_cycles=4,
            wall_material="grcop84",
            coolant_temp=coolant_temp,
            dt=0.001,
            location="throat",
        )

        print("  Results:")
        print(f"    Total simulation time:  {duty_result.duration:.1f} s")
        print(f"    Peak wall temp:         {duty_result.peak_wall_temp:.0f} K")
        print(f"    Min thermal margin:     {duty_result.min_thermal_margin:.0f} K")
        print(f"    Status:                 {'SAFE' if duty_result.is_safe() else 'EXCEEDS LIMIT'}")
        print()

        # Check for thermal ratcheting (increasing peak temp each cycle)
        cycle_time = 8.0
        cycle_peaks = []
        for i in range(4):
            start_idx = int(i * cycle_time / (duty_result.time[1] - duty_result.time[0]))
            end_idx = int((i + 1) * cycle_time / (duty_result.time[1] - duty_result.time[0]))
            cycle_peak = duty_result.wall_temp_inner[start_idx:end_idx].max()
            cycle_peaks.append(cycle_peak)
            print(f"    Cycle {i+1} peak temp:     {cycle_peak:.0f} K")

        ratcheting = cycle_peaks[-1] - cycle_peaks[0]
        print()
        print(f"    Thermal ratcheting:     {ratcheting:+.1f} K over {len(cycle_peaks)} cycles")
        print()

        ctx.log("Generating duty cycle plots...")
        fig_duty = plot_duty_cycle_thermal(
            duty_result.time,
            duty_result.wall_temp_inner,
            duty_result.heat_flux,
            duty_result.max_material_temp,
            burn_time=3.0,
            coast_time=5.0,
            n_cycles=4,
            title="Duty Cycle Thermal Analysis (4 × 3s burns)",
        )
        fig_duty.savefig(ctx.path("duty_cycle.png"), dpi=150, bbox_inches="tight")

        # =====================================================================
        # Study 4: Material Comparison
        # =====================================================================

        print("┌" + "─" * 68 + "┐")
        print("│ STUDY 4: LINER MATERIAL COMPARISON".ljust(69) + "│")
        print("└" + "─" * 68 + "┘")
        print()

        print("  Comparing candidate liner materials for thermal performance...")
        print()

        materials_to_test = ["copper", "copper_zirconium", "grcop84", "inconel718"]
        material_results = []

        for mat in materials_to_test:
            props = get_material_properties(mat)
            try:
                result = simulate_startup(
                    inputs, performance, geometry,
                    wall_thickness=wall_thickness,
                    wall_material=mat,
                    coolant_temp=coolant_temp,
                    startup_time=2.0,
                    steady_time=1.0,
                    location="throat",
                )

                material_results.append({
                    "material": mat,
                    "name": props["name"],
                    "k_W_mK": props["k"],
                    "max_service_temp_K": props["max_temp"],
                    "peak_wall_temp_K": result.peak_wall_temp,
                    "min_margin_K": result.min_thermal_margin,
                    "is_safe": result.is_safe(),
                    "time_to_steady_s": result.time_to_steady_state,
                })
            except Exception as e:
                print(f"  Warning: {mat} simulation failed: {e}")

        # Print comparison table
        print(f"  {'Material':<20} {'k (W/m·K)':<12} {'T_max (K)':<12} {'T_peak (K)':<12} {'Margin (K)':<12} {'Safe?'}")
        print("  " + "-" * 80)

        for r in material_results:
            safe_str = "Yes" if r["is_safe"] else "NO"
            print(f"  {r['name']:<20} {r['k_W_mK']:<12.0f} {r['max_service_temp_K']:<12.0f} {r['peak_wall_temp_K']:<12.0f} {r['min_margin_K']:<12.0f} {safe_str}")

        print()

        # Save material comparison data
        df = pl.DataFrame(material_results)
        df.write_csv(ctx.path("material_comparison.csv"))

        # =====================================================================
        # Save Results Summary
        # =====================================================================

        print("┌" + "─" * 68 + "┐")
        print("│ SAVING RESULTS".ljust(69) + "│")
        print("└" + "─" * 68 + "┘")
        print()

        # Save time series data
        startup_df = pl.DataFrame({
            "time_s": startup_result.time,
            "T_inner_K": startup_result.wall_temp_inner,
            "T_outer_K": startup_result.wall_temp_outer,
            "q_W_m2": startup_result.heat_flux,
            "margin_K": startup_result.thermal_margin,
        })
        startup_df.write_csv(ctx.path("startup_data.csv"))

        duty_df = pl.DataFrame({
            "time_s": duty_result.time,
            "T_inner_K": duty_result.wall_temp_inner,
            "T_outer_K": duty_result.wall_temp_outer,
            "q_W_m2": duty_result.heat_flux,
            "margin_K": duty_result.thermal_margin,
        })
        duty_df.write_csv(ctx.path("duty_cycle_data.csv"))

        # Save summary
        ctx.save_summary({
            "engine": {
                "name": inputs.name,
                "thrust_kN": inputs.thrust.to("kN").value,
                "chamber_pressure_MPa": inputs.chamber_pressure.to("MPa").value,
            },
            "wall": {
                "thickness_mm": wall_thickness.to("m").value * 1000,
                "material": "grcop84",
                "coolant_temp_K": coolant_temp.to("K").value,
            },
            "startup": {
                "peak_temp_K": startup_result.peak_wall_temp,
                "min_margin_K": startup_result.min_thermal_margin,
                "is_safe": startup_result.is_safe(),
            },
            "duty_cycle": {
                "peak_temp_K": duty_result.peak_wall_temp,
                "min_margin_K": duty_result.min_thermal_margin,
                "thermal_ratcheting_K": ratcheting,
                "is_safe": duty_result.is_safe(),
            },
        })

        ctx.log("All results saved!")
        print()
        print(f"  Output directory: {ctx.output_dir}")

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "Analysis Complete!".center(68) + "║")
    print("╚" + "═" * 68 + "╝")


if __name__ == "__main__":
    main()

