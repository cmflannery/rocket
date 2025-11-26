#!/usr/bin/env python
"""Multi-objective optimization example for Rocket.

This example demonstrates how to find Pareto-optimal engine designs
that balance competing objectives:

1. Maximize Isp (specific impulse)
2. Minimize engine mass (via throat size as proxy)
3. Satisfy thermal constraints

Real engine design involves tradeoffs - you can't maximize everything.
Pareto fronts show the best achievable combinations.
"""

from openrocketengine import (
    EngineInputs,
    MultiObjectiveOptimizer,
    OutputContext,
    Range,
    design_engine,
)
from openrocketengine.units import kilonewtons, megapascals


def main() -> None:
    """Run the multi-objective optimization example."""
    print("=" * 70)
    print("Rocket - Multi-Objective Optimization Example")
    print("=" * 70)
    print()

    # =========================================================================
    # Define the Optimization Problem
    # =========================================================================

    print("Problem: Design a LOX/CH4 engine balancing Isp vs. compactness")
    print("-" * 70)
    print()
    print("Objectives:")
    print("  1. Maximize vacuum Isp (performance)")
    print("  2. Minimize throat diameter (smaller = lighter, cheaper)")
    print()
    print("Design variables:")
    print("  - Chamber pressure: 5-25 MPa")
    print("  - Mixture ratio: 2.5-4.0")
    print()
    print("Constraints:")
    print("  - Expansion ratio < 80 (practical nozzle size)")
    print("  - Throat diameter > 3 cm (manufacturability)")
    print()

    # Baseline design
    baseline = EngineInputs.from_propellants(
        oxidizer="LOX",
        fuel="CH4",
        thrust=kilonewtons(100),
        chamber_pressure=megapascals(10),
        mixture_ratio=3.2,
        name="Methalox-Baseline",
    )

    # Define constraints
    def expansion_constraint(result: tuple) -> bool:
        """Expansion ratio must be < 80."""
        _, geometry = result
        return geometry.expansion_ratio < 80

    def throat_constraint(result: tuple) -> bool:
        """Throat diameter must be > 3 cm for manufacturability."""
        _, geometry = result
        return geometry.throat_diameter.to("m").value * 100 > 3.0

    # =========================================================================
    # Run Multi-Objective Optimization
    # =========================================================================

    print("Running optimization...")
    print()

    optimizer = MultiObjectiveOptimizer(
        compute=design_engine,
        base=baseline,
        objectives=["isp_vac", "throat_diameter"],
        maximize=[True, False],  # Max Isp, Min throat (smaller = better)
        vary={
            "chamber_pressure": Range(5, 25, n=15, unit="MPa"),
            "mixture_ratio": Range(2.5, 4.0, n=10),
        },
        constraints=[expansion_constraint, throat_constraint],
    )

    pareto_results = optimizer.run(progress=True)

    print()
    print(f"Total designs evaluated: {pareto_results.n_total}")
    print(f"Feasible designs: {pareto_results.n_feasible}")
    print(f"Pareto-optimal designs: {pareto_results.n_pareto}")
    print()

    # =========================================================================
    # Analyze Pareto Front
    # =========================================================================

    print("=" * 70)
    print("PARETO-OPTIMAL DESIGNS")
    print("=" * 70)
    print()
    print("These designs represent the best tradeoffs between Isp and size:")
    print()
    print(f"  {'#':<4} {'Pc (MPa)':<10} {'MR':<8} {'Isp (vac)':<12} {'Throat (cm)':<12} {'ε':<8}")
    print("  " + "-" * 54)

    pareto_df = pareto_results.pareto_front()

    for i, row in enumerate(pareto_df.iter_rows(named=True)):
        pc_mpa = row["chamber_pressure"] / 1e6
        dt_cm = row["throat_diameter"] * 100
        print(f"  {i+1:<4} {pc_mpa:<10.0f} {row['mixture_ratio']:<8.1f} {row['isp_vac']:<12.1f} {dt_cm:<12.2f} {row['expansion_ratio']:<8.1f}")

    print()

    # =========================================================================
    # Interpret Results
    # =========================================================================

    print("=" * 70)
    print("DESIGN RECOMMENDATIONS")
    print("=" * 70)
    print()

    # Best Isp design
    best_isp_idx = pareto_df["isp_vac"].arg_max()
    best_isp_row = pareto_df.row(best_isp_idx, named=True)

    print("For MAXIMUM PERFORMANCE (highest Isp):")
    print(f"  Chamber pressure: {best_isp_row['chamber_pressure']/1e6:.0f} MPa")
    print(f"  Mixture ratio:    {best_isp_row['mixture_ratio']:.1f}")
    print(f"  Isp (vac):        {best_isp_row['isp_vac']:.1f} s")
    print(f"  Throat diameter:  {best_isp_row['throat_diameter']*100:.2f} cm")
    print()

    # Smallest design
    best_size_idx = pareto_df["throat_diameter"].arg_min()
    best_size_row = pareto_df.row(best_size_idx, named=True)

    print("For MINIMUM SIZE (smallest throat):")
    print(f"  Chamber pressure: {best_size_row['chamber_pressure']/1e6:.0f} MPa")
    print(f"  Mixture ratio:    {best_size_row['mixture_ratio']:.1f}")
    print(f"  Isp (vac):        {best_size_row['isp_vac']:.1f} s")
    print(f"  Throat diameter:  {best_size_row['throat_diameter']*100:.2f} cm")
    print()

    # Balanced design (middle of Pareto front)
    mid_idx = len(pareto_df) // 2
    mid_row = pareto_df.row(mid_idx, named=True)

    print("For BALANCED DESIGN (middle of Pareto front):")
    print(f"  Chamber pressure: {mid_row['chamber_pressure']/1e6:.0f} MPa")
    print(f"  Mixture ratio:    {mid_row['mixture_ratio']:.1f}")
    print(f"  Isp (vac):        {mid_row['isp_vac']:.1f} s")
    print(f"  Throat diameter:  {mid_row['throat_diameter']*100:.2f} cm")
    print()

    # =========================================================================
    # Tradeoff Analysis
    # =========================================================================

    print("=" * 70)
    print("TRADEOFF ANALYSIS")
    print("=" * 70)
    print()

    isp_range = pareto_df["isp_vac"].max() - pareto_df["isp_vac"].min()
    dt_range = (pareto_df["throat_diameter"].max() - pareto_df["throat_diameter"].min()) * 100

    print(f"  Isp range on Pareto front:    {pareto_df['isp_vac'].min():.1f} - {pareto_df['isp_vac'].max():.1f} s (Δ = {isp_range:.1f} s)")
    print(f"  Throat range on Pareto front: {pareto_df['throat_diameter'].min()*100:.2f} - {pareto_df['throat_diameter'].max()*100:.2f} cm (Δ = {dt_range:.2f} cm)")
    print()
    print("  Interpretation:")
    print(f"    - You can gain up to {isp_range:.1f} s of Isp...")
    print(f"    - ...at the cost of {dt_range:.1f} cm larger throat")
    print(f"    - Marginal tradeoff: {isp_range/dt_range:.1f} s of Isp per cm of throat")
    print()

    # =========================================================================
    # Save Results
    # =========================================================================

    print("=" * 70)
    print("Saving Results")
    print("=" * 70)
    print()

    with OutputContext("optimization_results", include_timestamp=True) as ctx:
        # Export all results
        ctx.log("Exporting full design space...")
        pareto_results.all_results.to_csv(ctx.path("all_designs.csv"))

        ctx.log("Exporting Pareto front...")
        pareto_df.write_csv(ctx.path("pareto_front.csv"))

        # Save summary
        ctx.save_summary({
            "problem": {
                "objectives": ["isp_vac (maximize)", "throat_diameter (minimize)"],
                "design_variables": {
                    "chamber_pressure_MPa": [5, 25],
                    "mixture_ratio": [2.5, 4.0],
                },
                "constraints": [
                    "expansion_ratio < 80",
                    "throat_diameter > 3 cm",
                ],
            },
            "results": {
                "total_designs": pareto_results.n_total,
                "feasible_designs": pareto_results.n_feasible,
                "pareto_optimal": pareto_results.n_pareto,
            },
            "recommendations": {
                "max_performance": {
                    "chamber_pressure_MPa": best_isp_row["chamber_pressure"] / 1e6,
                    "mixture_ratio": best_isp_row["mixture_ratio"],
                    "isp_vac_s": best_isp_row["isp_vac"],
                },
                "min_size": {
                    "chamber_pressure_MPa": best_size_row["chamber_pressure"] / 1e6,
                    "mixture_ratio": best_size_row["mixture_ratio"],
                    "throat_diameter_cm": best_size_row["throat_diameter"] * 100,
                },
                "balanced": {
                    "chamber_pressure_MPa": mid_row["chamber_pressure"] / 1e6,
                    "mixture_ratio": mid_row["mixture_ratio"],
                    "isp_vac_s": mid_row["isp_vac"],
                    "throat_diameter_cm": mid_row["throat_diameter"] * 100,
                },
            },
        })

        print()
        print(f"  All results saved to: {ctx.output_dir}")

    print()
    print("=" * 70)
    print("Optimization Complete!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()

