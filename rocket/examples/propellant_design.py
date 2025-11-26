#!/usr/bin/env python
"""Propellant-based engine design example for Rocket.

This example demonstrates the simplified workflow where you specify
propellants and the library automatically determines combustion properties.

No need to manually look up Tc, gamma, or molecular weight!
"""

from rocket import OutputContext, design_engine, plot_engine_dashboard
from rocket.engine import EngineInputs
from rocket.nozzle import full_chamber_contour, generate_nozzle_from_geometry
from rocket.units import kilonewtons, megapascals


def main() -> None:
    """Run the propellant-based design example."""
    print("=" * 70)
    print("Rocket - Propellant-Based Design")
    print("=" * 70)
    print()
    print("Using NASA CEA (via RocketCEA) for thermochemistry calculations")
    print()

    # Store all engine designs for comparison
    designs: list[tuple[str, EngineInputs, any, any]] = []

    # =========================================================================
    # Design a LOX/RP-1 Engine (like Merlin)
    # =========================================================================

    print("-" * 70)
    print("Design 1: LOX/RP-1 Engine (Kerolox)")
    print("-" * 70)
    print()

    # Just specify propellants, thrust, and pressure - that's it!
    lox_rp1 = EngineInputs.from_propellants(
        oxidizer="LOX",
        fuel="RP1",
        thrust=kilonewtons(100),  # 100 kN
        chamber_pressure=megapascals(7),  # 7 MPa (~1000 psi)
        mixture_ratio=2.7,  # Typical for LOX/RP-1
        name="Kerolox-100",
    )

    print(f"Engine: {lox_rp1.name}")
    print("  Propellants: LOX / RP-1")
    print(f"  Mixture Ratio: {lox_rp1.mixture_ratio}")
    print(f"  Chamber Temp: {lox_rp1.chamber_temp.to('K').value:.0f} K (auto-calculated!)")
    print(f"  Gamma: {lox_rp1.gamma:.3f} (auto-calculated!)")
    print(f"  Molecular Weight: {lox_rp1.molecular_weight:.1f} kg/kmol (auto-calculated!)")
    print()

    perf1, geom1 = design_engine(lox_rp1)
    print("Performance:")
    print(f"  Isp (SL): {perf1.isp.value:.1f} s")
    print(f"  Isp (Vac): {perf1.isp_vac.value:.1f} s")
    print(f"  Thrust Coeff: {perf1.thrust_coeff:.3f}")
    print(f"  Mass Flow: {perf1.mdot.value:.2f} kg/s")
    print()
    print("Geometry:")
    print(f"  Throat Diameter: {geom1.throat_diameter.to('m').value * 100:.1f} cm")
    print(f"  Exit Diameter: {geom1.exit_diameter.to('m').value * 100:.1f} cm")
    print(f"  Expansion Ratio: {geom1.expansion_ratio:.1f}")
    print()

    designs.append(("LOX/RP-1", lox_rp1, perf1, geom1))

    # =========================================================================
    # Design a LOX/Methane Engine (like Raptor)
    # =========================================================================

    print("-" * 70)
    print("Design 2: LOX/Methane Engine (Methalox)")
    print("-" * 70)
    print()

    lox_ch4 = EngineInputs.from_propellants(
        oxidizer="LOX",
        fuel="CH4",
        thrust=kilonewtons(200),
        chamber_pressure=megapascals(10),  # Higher pressure
        mixture_ratio=3.2,
        name="Methalox-200",
    )

    print(f"Engine: {lox_ch4.name}")
    print(f"  Chamber Temp: {lox_ch4.chamber_temp.to('K').value:.0f} K")
    print(f"  Gamma: {lox_ch4.gamma:.3f}")
    print()

    perf2, geom2 = design_engine(lox_ch4)
    print("Performance:")
    print(f"  Isp (SL): {perf2.isp.value:.1f} s")
    print(f"  Isp (Vac): {perf2.isp_vac.value:.1f} s")
    print()

    designs.append(("LOX/CH4", lox_ch4, perf2, geom2))

    # =========================================================================
    # Design a LOX/LH2 Engine (like RS-25/SSME)
    # =========================================================================

    print("-" * 70)
    print("Design 3: LOX/LH2 Engine (Hydrolox)")
    print("-" * 70)
    print()

    lox_lh2 = EngineInputs.from_propellants(
        oxidizer="LOX",
        fuel="LH2",
        thrust=kilonewtons(50),  # Smaller for demo
        chamber_pressure=megapascals(15),  # High pressure like SSME
        mixture_ratio=6.0,  # Typical for LOX/LH2
        name="Hydrolox-50",
    )

    print(f"Engine: {lox_lh2.name}")
    print(f"  Chamber Temp: {lox_lh2.chamber_temp.to('K').value:.0f} K")
    print(f"  Molecular Weight: {lox_lh2.molecular_weight:.1f} kg/kmol (low = high Isp!)")
    print()

    perf3, geom3 = design_engine(lox_lh2)
    print("Performance:")
    print(f"  Isp (SL): {perf3.isp.value:.1f} s")
    print(f"  Isp (Vac): {perf3.isp_vac.value:.1f} s  <- Highest!")
    print()

    designs.append(("LOX/LH2", lox_lh2, perf3, geom3))

    # =========================================================================
    # Comparison Summary
    # =========================================================================

    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Engine':<20} {'Isp(SL)':<10} {'Isp(Vac)':<10} {'MW':<8} {'Tc (K)':<10}")
    print("-" * 70)

    for name, inputs, perf, _ in designs:
        print(
            f"{name:<20} "
            f"{perf.isp.value:<10.1f} "
            f"{perf.isp_vac.value:<10.1f} "
            f"{inputs.molecular_weight:<8.1f} "
            f"{inputs.chamber_temp.to('K').value:<10.0f}"
        )

    print()

    # =========================================================================
    # Generate Dashboards for All Engines
    # =========================================================================

    print("=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    print()

    with OutputContext("propellant_comparison", include_timestamp=True) as ctx:
        # Save comparison summary
        ctx.save_summary({
            "designs": [
                {
                    "name": name,
                    "propellants": f"{inputs.name}",
                    "isp_sl": perf.isp.value,
                    "isp_vac": perf.isp_vac.value,
                    "molecular_weight": inputs.molecular_weight,
                    "chamber_temp_K": inputs.chamber_temp.to("K").value,
                    "gamma": inputs.gamma,
                    "thrust_kN": inputs.thrust.to("kN").value,
                    "chamber_pressure_MPa": inputs.chamber_pressure.to("MPa").value,
                }
                for name, inputs, perf, _ in designs
            ]
        }, "comparison_summary.json")

        for name, inputs, perf, geom in designs:
            ctx.log(f"Generating dashboard for {inputs.name}...")
            nozzle = generate_nozzle_from_geometry(geom)
            contour = full_chamber_contour(inputs, geom, nozzle)
            fig = plot_engine_dashboard(inputs, perf, geom, contour)

            safe_name = inputs.name.lower().replace("-", "_").replace(" ", "_")
            fig.savefig(ctx.path(f"{safe_name}_dashboard.png"), dpi=150, bbox_inches="tight")

        print()
        print(f"  All outputs saved to: {ctx.output_dir}")

    print()
    print("=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
