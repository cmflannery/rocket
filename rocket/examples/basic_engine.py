#!/usr/bin/env python
"""Basic engine design example for Rocket.

This example demonstrates the complete workflow for designing a small
liquid rocket engine:

1. Define engine inputs (thrust, pressures, temperatures, etc.)
2. Compute performance metrics (Isp, c*, Cf, mass flow rates)
3. Compute geometry (throat, chamber, exit dimensions)
4. Generate nozzle contour
5. Visualize the design
6. Export contour for CAD

All outputs are organized into a timestamped directory structure.
"""

from rocket import OutputContext
from rocket.engine import (
    EngineInputs,
    compute_geometry,
    compute_performance,
    format_geometry_summary,
    format_performance_summary,
)
from rocket.nozzle import (
    full_chamber_contour,
    generate_nozzle_from_geometry,
)
from rocket.plotting import (
    plot_engine_cross_section,
    plot_engine_dashboard,
    plot_nozzle_contour,
    plot_performance_vs_altitude,
)
from rocket.units import kelvin, megapascals, meters, newtons, pascals


def main() -> None:
    """Run the basic engine design example."""
    print("=" * 70)
    print("Rocket - Basic Engine Design Example")
    print("=" * 70)
    print()

    # =========================================================================
    # Step 1: Define Engine Inputs
    # =========================================================================
    #
    # We're designing a small pressure-fed engine with the following specs:
    # - ~5 kN (1100 lbf) sea-level thrust
    # - LOX/Ethanol propellants (assumed combustion properties)
    # - Pressure-fed, so moderate chamber pressure
    #
    # The thermochemical properties (Tc, gamma, MW) would normally come from
    # NASA CEA or similar tool. Here we use representative values for
    # LOX/Ethanol at O/F = 1.3

    print("Step 1: Defining engine inputs...")
    print()

    inputs = EngineInputs(
        name="Student Engine Mk1",
        # Performance targets
        thrust=newtons(5000),  # 5 kN sea-level thrust
        # Chamber conditions
        chamber_pressure=megapascals(2.0),  # 2 MPa (~290 psi)
        chamber_temp=kelvin(3200),  # Flame temperature from CEA
        exit_pressure=pascals(101325),  # Expanded to sea level
        # Propellant properties (from CEA for LOX/Ethanol)
        molecular_weight=21.5,  # kg/kmol
        gamma=1.22,  # Cp/Cv
        mixture_ratio=1.3,  # O/F mass ratio
        # Chamber geometry parameters
        lstar=meters(1.0),  # Characteristic length (typical for biprop)
        contraction_ratio=4.0,  # Ac/At
        contraction_angle=45.0,  # degrees
        bell_fraction=0.8,  # 80% bell nozzle
    )

    print(f"  Engine Name:        {inputs.name}")
    print(f"  Design Thrust:      {inputs.thrust}")
    print(f"  Chamber Pressure:   {inputs.chamber_pressure}")
    print(f"  Chamber Temp:       {inputs.chamber_temp}")
    print(f"  Exit Pressure:      {inputs.exit_pressure}")
    print(f"  Molecular Weight:   {inputs.molecular_weight} kg/kmol")
    print(f"  Gamma:              {inputs.gamma}")
    print(f"  Mixture Ratio:      {inputs.mixture_ratio}")
    print()

    # =========================================================================
    # Step 2: Compute Performance
    # =========================================================================

    print("Step 2: Computing performance...")
    print()

    performance = compute_performance(inputs)

    print(f"  Specific Impulse (SL):  {performance.isp.value:.1f} s")
    print(f"  Specific Impulse (Vac): {performance.isp_vac.value:.1f} s")
    print(f"  Characteristic Velocity: {performance.cstar.value:.0f} m/s")
    print(f"  Thrust Coefficient (SL): {performance.thrust_coeff:.3f}")
    print(f"  Exit Mach Number:       {performance.exit_mach:.2f}")
    print(f"  Expansion Ratio:        {performance.expansion_ratio:.1f}")
    print()
    print(f"  Total Mass Flow:        {performance.mdot.value:.3f} kg/s")
    print(f"  Oxidizer Flow:          {performance.mdot_ox.value:.3f} kg/s")
    print(f"  Fuel Flow:              {performance.mdot_fuel.value:.3f} kg/s")
    print()

    # =========================================================================
    # Step 3: Compute Geometry
    # =========================================================================

    print("Step 3: Computing geometry...")
    print()

    geometry = compute_geometry(inputs, performance)

    # Convert to more convenient units for display
    Dt_mm = geometry.throat_diameter.to("m").value * 1000
    De_mm = geometry.exit_diameter.to("m").value * 1000
    Dc_mm = geometry.chamber_diameter.to("m").value * 1000
    Lc_mm = geometry.chamber_length.to("m").value * 1000
    Ln_mm = geometry.nozzle_length.to("m").value * 1000

    print(f"  Throat Diameter:    {Dt_mm:.1f} mm")
    print(f"  Exit Diameter:      {De_mm:.1f} mm")
    print(f"  Chamber Diameter:   {Dc_mm:.1f} mm")
    print(f"  Chamber Length:     {Lc_mm:.1f} mm")
    print(f"  Nozzle Length:      {Ln_mm:.1f} mm")
    print(f"  Expansion Ratio:    {geometry.expansion_ratio:.1f}")
    print(f"  Contraction Ratio:  {geometry.contraction_ratio:.1f}")
    print()

    # =========================================================================
    # Step 4: Generate Nozzle Contour
    # =========================================================================

    print("Step 4: Generating nozzle contour...")
    print()

    # Generate just the divergent nozzle section
    nozzle_contour = generate_nozzle_from_geometry(
        geometry, bell_fraction=inputs.bell_fraction, num_points=100
    )

    print(f"  Contour Type:       {nozzle_contour.contour_type}")
    print(f"  Number of Points:   {len(nozzle_contour.x)}")
    print(f"  Nozzle Length:      {nozzle_contour.length * 1000:.1f} mm")
    print()

    # Generate full chamber contour (chamber + convergent + divergent)
    full_contour = full_chamber_contour(
        inputs, geometry, nozzle_contour, num_chamber_points=50, num_convergent_points=30
    )

    print(f"  Full Contour Type:  {full_contour.contour_type}")
    print(f"  Total Points:       {len(full_contour.x)}")
    print(f"  Total Length:       {full_contour.length * 1000:.1f} mm")
    print()

    # =========================================================================
    # Step 5-7: Save All Outputs to Organized Directory
    # =========================================================================

    print("Step 5-7: Saving outputs...")
    print()

    # Use OutputContext to organize all outputs
    with OutputContext("student_engine_mk1", include_timestamp=True) as ctx:
        # Add metadata about this run
        ctx.add_metadata("engine_name", inputs.name)
        ctx.add_metadata("thrust_N", inputs.thrust.to("N").value)
        ctx.add_metadata("chamber_pressure_MPa", inputs.chamber_pressure.to("MPa").value)

        # Export contours to CSV (automatically goes to data/)
        ctx.log("Exporting nozzle contours...")
        nozzle_contour.to_csv(ctx.path("nozzle_contour.csv"))
        full_contour.to_csv(ctx.path("full_chamber_contour.csv"))

        # Save text summaries (automatically goes to reports/)
        ctx.log("Saving performance summaries...")
        ctx.save_text(format_performance_summary(inputs, performance), "performance_summary.txt")
        ctx.save_text(format_geometry_summary(inputs, geometry), "geometry_summary.txt")

        # Save design summary as JSON
        ctx.save_summary({
            "engine_name": inputs.name,
            "performance": {
                "isp_sl_s": performance.isp.value,
                "isp_vac_s": performance.isp_vac.value,
                "cstar_m_s": performance.cstar.value,
                "thrust_coeff_sl": performance.thrust_coeff,
                "thrust_coeff_vac": performance.thrust_coeff_vac,
                "exit_mach": performance.exit_mach,
                "mdot_kg_s": performance.mdot.value,
                "mdot_ox_kg_s": performance.mdot_ox.value,
                "mdot_fuel_kg_s": performance.mdot_fuel.value,
            },
            "geometry": {
                "throat_diameter_mm": Dt_mm,
                "exit_diameter_mm": De_mm,
                "chamber_diameter_mm": Dc_mm,
                "chamber_length_mm": Lc_mm,
                "nozzle_length_mm": Ln_mm,
                "expansion_ratio": geometry.expansion_ratio,
                "contraction_ratio": geometry.contraction_ratio,
            },
            "inputs": {
                "thrust_N": inputs.thrust.to("N").value,
                "chamber_pressure_Pa": inputs.chamber_pressure.to("Pa").value,
                "chamber_temp_K": inputs.chamber_temp.to("K").value,
                "molecular_weight": inputs.molecular_weight,
                "gamma": inputs.gamma,
                "mixture_ratio": inputs.mixture_ratio,
            },
        })

        # Create visualizations (automatically goes to plots/)
        ctx.log("Generating visualizations...")

        fig1 = plot_engine_cross_section(
            geometry, full_contour, inputs, show_dimensions=True,
            title=f"{inputs.name} Cross-Section"
        )
        fig1.savefig(ctx.path("engine_cross_section.png"), dpi=150, bbox_inches="tight")

        fig2 = plot_nozzle_contour(nozzle_contour, title=f"{inputs.name} Nozzle Contour")
        fig2.savefig(ctx.path("nozzle_contour.png"), dpi=150, bbox_inches="tight")

        fig3 = plot_performance_vs_altitude(inputs, performance, geometry, max_altitude_km=80)
        fig3.savefig(ctx.path("altitude_performance.png"), dpi=150, bbox_inches="tight")

        fig4 = plot_engine_dashboard(inputs, performance, geometry, full_contour)
        fig4.savefig(ctx.path("engine_dashboard.png"), dpi=150, bbox_inches="tight")

        ctx.log("All outputs saved!")
        print()
        print(f"  Output directory: {ctx.output_dir}")

    print()
    print("=" * 70)
    print("Design complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
