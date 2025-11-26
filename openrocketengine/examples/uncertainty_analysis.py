#!/usr/bin/env python
"""Uncertainty analysis example for Rocket.

This example demonstrates Monte Carlo uncertainty quantification to understand
how input uncertainties propagate through engine design calculations:

1. Define probability distributions for uncertain inputs
2. Run Monte Carlo sampling
3. Analyze output statistics and confidence intervals
4. Identify which inputs drive the most uncertainty

This answers questions like:
- "If my mixture ratio varies ±5%, how much does Isp vary?"
- "What's the 95% confidence interval on my thrust coefficient?"
- "Which input uncertainty should I focus on reducing?"
"""

from openrocketengine import (
    EngineInputs,
    Normal,
    OutputContext,
    UncertaintyAnalysis,
    Uniform,
    design_engine,
)
from openrocketengine.units import kilonewtons, megapascals


def main() -> None:
    """Run the uncertainty analysis example."""
    print("=" * 70)
    print("Rocket - Uncertainty Analysis Example")
    print("=" * 70)
    print()

    # =========================================================================
    # Define Nominal Design Point
    # =========================================================================

    print("Nominal Engine Design: LOX/RP-1 Kerolox")
    print("-" * 70)

    nominal = EngineInputs.from_propellants(
        oxidizer="LOX",
        fuel="RP1",
        thrust=kilonewtons(100),
        chamber_pressure=megapascals(7),
        mixture_ratio=2.7,
        name="Kerolox-100",
    )

    perf, geom = design_engine(nominal)
    print(f"  Nominal Isp (vac):     {perf.isp_vac.value:.1f} s")
    print(f"  Nominal Isp (SL):      {perf.isp.value:.1f} s")
    print(f"  Nominal Thrust Coeff:  {perf.thrust_coeff:.3f}")
    print(f"  Nominal Throat Dia:    {geom.throat_diameter.to('m').value*100:.2f} cm")
    print()

    # =========================================================================
    # Study 1: Single Source of Uncertainty (Mixture Ratio)
    # =========================================================================

    print("=" * 70)
    print("Study 1: Mixture Ratio Uncertainty")
    print("=" * 70)
    print()
    print("The mixture ratio is controlled by the propellant valves.")
    print("Assume MR = 2.7 ± 0.1 (Normal distribution, σ = 0.1)")
    print()

    mr_uncertainty = UncertaintyAnalysis(
        compute=design_engine,
        base=nominal,
        distributions={
            "mixture_ratio": Normal(mean=2.7, std=0.1),
        },
        seed=42,  # For reproducibility
    )

    mr_results = mr_uncertainty.run(n_samples=1000, progress=True)

    print()
    print("Monte Carlo Results (N=1000):")
    print()

    # Get statistics for key metrics
    stats = mr_results.statistics()

    print(f"  {'Metric':<20} {'Mean':<12} {'Std Dev':<12} {'95% CI':<20}")
    print("  " + "-" * 64)

    for metric in ["isp_vac", "isp", "thrust_coeff"]:
        mean = stats[metric]["mean"]
        std = stats[metric]["std"]
        ci_low, ci_high = mr_results.confidence_interval(metric, 0.95)
        print(f"  {metric:<20} {mean:<12.2f} {std:<12.4f} [{ci_low:.2f}, {ci_high:.2f}]")

    print()
    print("  Interpretation:")
    print(f"    - MR uncertainty of σ=0.1 causes Isp variation of σ≈{stats['isp_vac']['std']:.2f} s")
    print(f"    - 95% of the time, Isp(vac) will be in [{mr_results.confidence_interval('isp_vac', 0.95)[0]:.1f}, {mr_results.confidence_interval('isp_vac', 0.95)[1]:.1f}] s")
    print()

    # =========================================================================
    # Study 2: Multiple Sources of Uncertainty
    # =========================================================================

    print("=" * 70)
    print("Study 2: Multiple Uncertainty Sources")
    print("=" * 70)
    print()
    print("Real engines have uncertainty in multiple inputs:")
    print("  - Chamber pressure: Pc = 7 MPa ± 0.3 MPa (Normal)")
    print("  - Mixture ratio:    MR = 2.7 ± 0.1 (Normal)")
    print("  - Gamma:            γ = 1.24 ± 0.02 (Normal, combustion model uncertainty)")
    print()

    multi_uncertainty = UncertaintyAnalysis(
        compute=design_engine,
        base=nominal,
        distributions={
            "chamber_pressure": Normal(mean=7, std=0.3, unit="MPa"),
            "mixture_ratio": Normal(mean=2.7, std=0.1),
            "gamma": Normal(mean=nominal.gamma, std=0.02),
        },
        seed=42,
    )

    multi_results = multi_uncertainty.run(n_samples=2000, progress=True)

    print()
    print("Monte Carlo Results (N=2000):")
    print()

    stats = multi_results.statistics()

    print(f"  {'Metric':<20} {'Mean':<12} {'Std Dev':<12} {'Range (99%)':<20}")
    print("  " + "-" * 64)

    for metric in ["isp_vac", "isp", "thrust_coeff", "cstar", "expansion_ratio"]:
        mean = stats[metric]["mean"]
        std = stats[metric]["std"]
        ci_low, ci_high = multi_results.confidence_interval(metric, 0.99)
        print(f"  {metric:<20} {mean:<12.2f} {std:<12.4f} [{ci_low:.2f}, {ci_high:.2f}]")

    print()

    # =========================================================================
    # Study 3: Uniform Distribution (Manufacturing Tolerances)
    # =========================================================================

    print("=" * 70)
    print("Study 3: Manufacturing Tolerance Analysis")
    print("=" * 70)
    print()
    print("Manufacturing tolerances are often uniformly distributed.")
    print("  - Contraction ratio: CR = 4.0 ± 0.2 (Uniform)")
    print("  - L* (characteristic length): L* = 1.0 ± 0.05 m (Uniform)")
    print()

    mfg_uncertainty = UncertaintyAnalysis(
        compute=design_engine,
        base=nominal,
        distributions={
            "contraction_ratio": Uniform(low=3.8, high=4.2),
            "lstar": Uniform(low=0.95, high=1.05, unit="m"),
        },
        seed=42,
    )

    mfg_results = mfg_uncertainty.run(n_samples=1000, progress=True)

    print()
    print("Impact of Manufacturing Tolerances:")
    print()

    stats = mfg_results.statistics()

    # These parameters mainly affect geometry, not performance
    for metric in ["chamber_diameter", "chamber_length"]:
        mean = stats[metric]["mean"]
        std = stats[metric]["std"]
        cv = 100 * std / mean  # Coefficient of variation
        print(f"  {metric:<20}: mean={mean*100:.2f} cm, CV={cv:.2f}%")

    print()
    print("  Note: Geometric tolerances have minimal impact on performance")
    print("        but affect fit/interface dimensions.")
    print()

    # =========================================================================
    # Export Results
    # =========================================================================

    print("=" * 70)
    print("Saving Results")
    print("=" * 70)
    print()

    with OutputContext("uncertainty_analysis", include_timestamp=True) as ctx:
        # Export raw Monte Carlo samples
        ctx.log("Exporting MR uncertainty samples...")
        mr_results.to_csv(ctx.path("mr_uncertainty_samples.csv"))

        ctx.log("Exporting multi-source uncertainty samples...")
        multi_results.to_csv(ctx.path("multi_uncertainty_samples.csv"))

        ctx.log("Exporting manufacturing tolerance samples...")
        mfg_results.to_csv(ctx.path("mfg_tolerance_samples.csv"))

        # Save statistical summary
        ctx.save_summary({
            "mr_uncertainty": {
                "inputs": {"mixture_ratio": {"distribution": "Normal", "mean": 2.7, "std": 0.1}},
                "n_samples": 1000,
                "isp_vac": {
                    "mean": float(mr_results.statistics()["isp_vac"]["mean"]),
                    "std": float(mr_results.statistics()["isp_vac"]["std"]),
                    "ci_95": list(mr_results.confidence_interval("isp_vac", 0.95)),
                },
            },
            "multi_uncertainty": {
                "inputs": {
                    "chamber_pressure": {"distribution": "Normal", "mean_MPa": 7, "std_MPa": 0.3},
                    "mixture_ratio": {"distribution": "Normal", "mean": 2.7, "std": 0.1},
                    "gamma": {"distribution": "Normal", "mean": nominal.gamma, "std": 0.02},
                },
                "n_samples": 2000,
            },
        })

        print()
        print(f"  All results saved to: {ctx.output_dir}")

    print()
    print("=" * 70)
    print("Uncertainty Analysis Complete!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()

