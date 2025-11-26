#!/usr/bin/env python
"""Parametric trade study example for Rocket.

This example demonstrates how to run systematic trade studies to explore
the design space and make informed engineering decisions:

1. Chamber pressure sweeps (most impactful parameter)
2. Multi-parameter grid studies
3. Constraint-based filtering
4. Polars DataFrame export
5. Mixture ratio studies (requires CEA recalculation)

These tools let you answer questions like:
- "How does Isp change with chamber pressure?"
- "Which designs satisfy my throat size requirement?"
- "What's the tradeoff between performance and size?"
"""

from rocket import (
    EngineInputs,
    OutputContext,
    ParametricStudy,
    Range,
    design_engine,
)
from rocket.units import kilonewtons, megapascals


def main() -> None:
    """Run the parametric trade study example."""
    print("=" * 70)
    print("Rocket - Parametric Trade Study Example")
    print("=" * 70)
    print()

    # =========================================================================
    # Baseline Engine Design
    # =========================================================================

    print("Baseline Engine: LOX/CH4 Methalox")
    print("-" * 70)

    baseline = EngineInputs.from_propellants(
        oxidizer="LOX",
        fuel="CH4",
        thrust=kilonewtons(200),
        chamber_pressure=megapascals(10),
        mixture_ratio=3.2,
        name="Methalox-Baseline",
    )

    perf, geom = design_engine(baseline)
    print(f"  Isp (vac): {perf.isp_vac.value:.1f} s")
    print(f"  Thrust:    {baseline.thrust.to('kN').value:.0f} kN")
    print()

    # =========================================================================
    # Study 1: Mixture Ratio Trade (with CEA recalculation)
    # =========================================================================

    print("=" * 70)
    print("Study 1: Mixture Ratio Sweep (with CEA thermochemistry)")
    print("=" * 70)
    print()
    print("Question: What mixture ratio maximizes Isp for LOX/CH4?")
    print()
    print("Note: Each point recalculates combustion properties via NASA CEA")
    print()

    mixture_ratios = [2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
    mr_results = []

    print(f"  {'MR':<8} {'Tc (K)':<10} {'Isp (vac)':<12} {'Isp (SL)':<12}")
    print("  " + "-" * 42)

    for mr in mixture_ratios:
        inputs = EngineInputs.from_propellants(
            oxidizer="LOX",
            fuel="CH4",
            thrust=kilonewtons(200),
            chamber_pressure=megapascals(10),
            mixture_ratio=mr,
        )
        perf, geom = design_engine(inputs)
        mr_results.append({
            "mr": mr,
            "Tc": inputs.chamber_temp.to("K").value,
            "isp_vac": perf.isp_vac.value,
            "isp_sl": perf.isp.value,
        })
        print(f"  {mr:<8.1f} {inputs.chamber_temp.to('K').value:<10.0f} {perf.isp_vac.value:<12.1f} {perf.isp.value:<12.1f}")

    # Find optimal MR
    best = max(mr_results, key=lambda x: x["isp_vac"])
    print()
    print(f"  → Optimal MR for max Isp: {best['mr']:.1f} (Isp = {best['isp_vac']:.1f} s)")
    print(f"    At this MR: Tc = {best['Tc']:.0f} K")
    print()

    # =========================================================================
    # Study 2: Chamber Pressure Sweep with Range
    # =========================================================================

    print("=" * 70)
    print("Study 2: Chamber Pressure Trade")
    print("=" * 70)
    print()
    print("Question: How does performance scale with chamber pressure?")
    print()

    # Using Range for continuous sweeps with units
    pc_study = ParametricStudy(
        compute=design_engine,
        base=baseline,
        vary={"chamber_pressure": Range(5, 25, n=9, unit="MPa")},
    )

    pc_results = pc_study.run(progress=True)

    print()
    print("Results:")
    print(f"  {'Pc (MPa)':<10} {'Isp (vac)':<12} {'Throat (cm)':<12} {'c* (m/s)':<10}")
    print("  " + "-" * 44)

    df = pc_results.to_dataframe()
    for row in df.iter_rows(named=True):
        throat_cm = row["throat_diameter"] * 100
        # chamber_pressure is already in MPa (unit specified in Range)
        print(f"  {row['chamber_pressure']:<10.0f} {row['isp_vac']:<12.1f} {throat_cm:<12.1f} {row['cstar']:<10.0f}")

    # Compute actual insights from data
    print()
    pc_low = df["chamber_pressure"].min()
    pc_high = df["chamber_pressure"].max()
    isp_at_low = df.filter(df["chamber_pressure"] == pc_low)["isp_vac"][0]
    isp_at_high = df.filter(df["chamber_pressure"] == pc_high)["isp_vac"][0]
    dt_at_low = df.filter(df["chamber_pressure"] == pc_low)["throat_diameter"][0] * 100
    dt_at_high = df.filter(df["chamber_pressure"] == pc_high)["throat_diameter"][0] * 100

    isp_change = isp_at_high - isp_at_low
    dt_change = dt_at_high - dt_at_low

    print(f"  From {pc_low:.0f} to {pc_high:.0f} MPa:")
    print(f"    Isp changed by {isp_change:+.1f} s ({100*isp_change/isp_at_low:+.1f}%)")
    print(f"    Throat diameter changed by {dt_change:+.1f} cm ({100*dt_change/dt_at_low:+.1f}%)")
    print()

    # =========================================================================
    # Study 3: Multi-Parameter Grid with Constraints
    # =========================================================================

    print("=" * 70)
    print("Study 3: Multi-Parameter Design Space Exploration")
    print("=" * 70)
    print()
    print("Sweeping: Chamber Pressure (5-20 MPa) × Contraction Ratio (3-6)")
    print("Constraint: Throat diameter > 8 cm (manufacturability)")
    print()

    def throat_constraint(result: tuple) -> bool:
        """Filter out designs with throat diameter < 8 cm."""
        _, geometry = result
        return geometry.throat_diameter.to("m").value * 100 > 8.0

    grid_study = ParametricStudy(
        compute=design_engine,
        base=baseline,
        vary={
            "chamber_pressure": Range(5, 20, n=6, unit="MPa"),
            "contraction_ratio": [3.0, 4.0, 5.0, 6.0],
        },
        constraints=[throat_constraint],
    )

    grid_results = grid_study.run(progress=True)

    print()
    n_total = len(grid_results.inputs)
    n_feasible = grid_results.constraints_passed.sum()
    print(f"  Total designs evaluated: {n_total}")
    print(f"  Feasible designs: {n_feasible} ({100*n_feasible/n_total:.0f}%)")
    print()

    # Export to Polars DataFrame for further analysis
    df = grid_results.to_dataframe()

    # Filter to feasible designs and show best by Isp
    feasible_df = df.filter(df["feasible"])
    best_designs = feasible_df.sort("isp_vac", descending=True).head(5)

    print("Top 5 Feasible Designs by Vacuum Isp:")
    print(f"  {'Pc (MPa)':<10} {'CR':<8} {'Isp (vac)':<12} {'Dt (cm)':<10}")
    print("  " + "-" * 40)

    for row in best_designs.iter_rows(named=True):
        # chamber_pressure is already in MPa (unit specified in Range)
        dt_cm = row["throat_diameter"] * 100
        print(f"  {row['chamber_pressure']:<10.0f} {row['contraction_ratio']:<8.1f} {row['isp_vac']:<12.1f} {dt_cm:<10.1f}")

    print()

    # =========================================================================
    # Save All Results
    # =========================================================================

    print("=" * 70)
    print("Saving Results")
    print("=" * 70)
    print()

    with OutputContext("trade_study_results", include_timestamp=True) as ctx:
        # Export DataFrames to CSV
        ctx.log("Exporting chamber pressure study...")
        pc_results.to_csv(ctx.path("chamber_pressure_sweep.csv"))

        ctx.log("Exporting grid study...")
        grid_results.to_csv(ctx.path("grid_study.csv"))

        # Save summary
        ctx.save_summary({
            "studies": {
                "mixture_ratio_sweep": {
                    "parameter": "mixture_ratio",
                    "range": [2.4, 4.0],
                    "optimal_mr": best["mr"],
                    "optimal_isp_vac": best["isp_vac"],
                    "note": "Each point recalculates CEA thermochemistry",
                },
                "chamber_pressure_sweep": {
                    "parameter": "chamber_pressure",
                    "range_MPa": [5, 25],
                    "n_points": 9,
                },
                "grid_study": {
                    "parameters": ["chamber_pressure", "contraction_ratio"],
                    "total_designs": n_total,
                    "feasible_designs": int(n_feasible),
                    "constraint": "throat_diameter > 8 cm",
                },
            }
        })

        print()
        print(f"  All results saved to: {ctx.output_dir}")

    print()
    print("=" * 70)
    print("Trade Study Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

