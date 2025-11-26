"""High-level study APIs for common engineering analyses.

These classes encapsulate complete workflows so users write minimal code.
The library handles analysis, visualization, and reporting.

Example:
    >>> from rocket.studies import CycleTradeStudy
    >>> 
    >>> study = CycleTradeStudy(
    ...     thrust_kn=100,
    ...     delta_v_m_s=3500,
    ...     payload_kg=5000,
    ...     propellants=("LOX", "CH4"),
    ... )
    >>> study.run()
    >>> study.save("outputs/my_study")  # Generates everything
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from beartype import beartype

from rocket import EngineInputs, design_engine
from rocket.analysis import ParametricStudy, Range
from rocket.cycles import GasGeneratorCycle, PressureFedCycle, StagedCombustionCycle
from rocket.units import kelvin, kilonewtons, megapascals


# =============================================================================
# Utility Functions
# =============================================================================


def _rocket_equation(isp_s: float, delta_v_m_s: float, payload_kg: float,
                     dry_mass_fraction: float = 0.08) -> dict:
    """Tsiolkovsky rocket equation solver."""
    g0 = 9.80665
    ve = isp_s * g0
    mass_ratio = np.exp(delta_v_m_s / ve)
    
    f = dry_mass_fraction
    mp = payload_kg * (mass_ratio - 1) / (1 + f - mass_ratio * f)
    
    if mp < 0:
        return {"feasible": False}
    
    return {
        "feasible": True,
        "propellant_kg": mp,
        "dry_kg": f * mp,
        "gross_kg": payload_kg + (1 + f) * mp,
    }


def _normalize(values: list, invert: bool = False) -> list:
    """Normalize values to 0-1 range."""
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return [0.5] * len(values)
    norm = [(v - min_v) / (max_v - min_v) for v in values]
    return [1 - n for n in norm] if invert else norm


# =============================================================================
# Cycle Trade Study
# =============================================================================


@beartype
@dataclass
class CycleTradeStudy:
    """Compare engine cycles for a given mission.
    
    Analyzes pressure-fed, gas generator, and staged combustion cycles,
    evaluates mission impact, and recommends the best option.
    
    Example:
        >>> study = CycleTradeStudy(
        ...     thrust_kn=100,
        ...     delta_v_m_s=3500,
        ...     payload_kg=5000,
        ... )
        >>> study.run()
        >>> print(study.recommendation)
        >>> study.save("outputs/cycle_study")
    """
    
    # Mission requirements
    thrust_kn: float | int
    delta_v_m_s: float | int
    payload_kg: float | int
    propellants: tuple[str, str] = ("LOX", "CH4")
    
    # Weighting for decision matrix
    weight_performance: float = 0.30
    weight_cost: float = 0.25
    weight_schedule: float = 0.20
    weight_complexity: float = 0.15
    weight_trl: float = 0.10
    
    # Results (populated by run())
    cycles: list[dict] = field(default_factory=list)
    recommendation: str = ""
    _has_run: bool = field(default=False, repr=False)
    
    def run(self) -> "CycleTradeStudy":
        """Run the cycle comparison analysis."""
        ox, fuel = self.propellants
        
        # Analyze each cycle
        cycle_configs = [
            ("Pressure-Fed", 2.5, PressureFedCycle(), 2, 15, 50, 9),
            ("Gas Generator", 10.0, GasGeneratorCycle(
                turbine_inlet_temp=kelvin(900),
                pump_efficiency_ox=0.70,
                pump_efficiency_fuel=0.70,
            ), 4, 80, 500, 9),
            ("Staged Combustion", 20.0, StagedCombustionCycle(
                preburner_temp=kelvin(750),
                pump_efficiency_ox=0.75,
                pump_efficiency_fuel=0.75,
            ), 6, 200, 1500, 7),
        ]
        
        self.cycles = []
        for name, pc_mpa, cycle_obj, dev_years, dev_cost, parts, trl in cycle_configs:
            inputs = EngineInputs.from_propellants(
                oxidizer=ox, fuel=fuel,
                thrust=kilonewtons(self.thrust_kn),
                chamber_pressure=megapascals(pc_mpa),
            )
            perf, geom = design_engine(inputs)
            result = cycle_obj.analyze(inputs, perf, geom)
            
            # Mission impact
            mission = _rocket_equation(
                result.net_isp.to("s").value,
                self.delta_v_m_s,
                self.payload_kg,
            )
            
            pump_power = 0
            if hasattr(result, 'pump_power_ox') and result.pump_power_ox:
                pump_power = (result.pump_power_ox.to("kW").value + 
                             result.pump_power_fuel.to("kW").value)
            
            self.cycles.append({
                "name": name,
                "isp_vac": result.net_isp.to("s").value,
                "propellant_kg": mission.get("propellant_kg", float("inf")),
                "pump_power_kw": pump_power,
                "dev_years": dev_years,
                "dev_cost_M": dev_cost,
                "parts": parts,
                "trl": trl,
                "feasible": mission.get("feasible", False),
            })
        
        # Score cycles
        scores = {
            "performance": _normalize([c["propellant_kg"] for c in self.cycles], invert=True),
            "cost": _normalize([c["dev_cost_M"] for c in self.cycles], invert=True),
            "schedule": _normalize([c["dev_years"] for c in self.cycles], invert=True),
            "complexity": _normalize([c["parts"] for c in self.cycles], invert=True),
            "trl": _normalize([c["trl"] for c in self.cycles]),
        }
        
        for i, c in enumerate(self.cycles):
            c["score"] = (
                self.weight_performance * scores["performance"][i] +
                self.weight_cost * scores["cost"][i] +
                self.weight_schedule * scores["schedule"][i] +
                self.weight_complexity * scores["complexity"][i] +
                self.weight_trl * scores["trl"][i]
            )
        
        # Determine recommendation
        best = max(self.cycles, key=lambda x: x["score"])
        self.recommendation = best["name"]
        self._has_run = True
        
        return self
    
    def to_dataframe(self) -> pl.DataFrame:
        """Get results as a Polars DataFrame."""
        if not self._has_run:
            self.run()
        return pl.DataFrame(self.cycles)
    
    def save(self, output_dir: str | Path) -> Path:
        """Save complete analysis with plots and data."""
        if not self._has_run:
            self.run()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate visualization
        self._generate_plots(output_dir)
        
        # Save data
        self.to_dataframe().write_csv(output_dir / "cycle_comparison.csv")
        
        # Save summary
        import json
        summary = {
            "mission": {
                "thrust_kn": self.thrust_kn,
                "delta_v_m_s": self.delta_v_m_s,
                "payload_kg": self.payload_kg,
                "propellants": self.propellants,
            },
            "recommendation": self.recommendation,
            "cycles": self.cycles,
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        return output_dir
    
    def _generate_plots(self, output_dir: Path) -> None:
        """Generate analysis plots."""
        fig = plt.figure(figsize=(14, 10), facecolor='#1a1a2e')
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
        
        colors = ['#16c79a', '#f4a261', '#e94560']
        names = [c["name"] for c in self.cycles]
        
        # 1. Propellant comparison
        ax1 = fig.add_subplot(gs[0, 0], facecolor='#16213e')
        prop = [c["propellant_kg"]/1000 for c in self.cycles]
        bars = ax1.bar(names, prop, color=colors, edgecolor='white')
        ax1.set_ylabel('Propellant [tonnes]', color='white')
        ax1.set_title(f'Propellant for {self.delta_v_m_s} m/s ΔV', color='white', fontweight='bold')
        self._style_axis(ax1)
        for bar, val in zip(bars, prop):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.1f}t', ha='center', color='white')
        
        # 2. Cost vs Schedule
        ax2 = fig.add_subplot(gs[0, 1], facecolor='#16213e')
        for i, c in enumerate(self.cycles):
            ax2.scatter(c["dev_years"], c["dev_cost_M"], s=300, c=colors[i], 
                       edgecolors='white', linewidth=2)
            ax2.annotate(c["name"], (c["dev_years"], c["dev_cost_M"]),
                        xytext=(8, 8), textcoords='offset points', color='white')
        ax2.set_xlabel('Development Time [years]', color='white')
        ax2.set_ylabel('Development Cost [$M]', color='white')
        ax2.set_title('Cost vs Schedule', color='white', fontweight='bold')
        self._style_axis(ax2)
        
        # 3. Decision scores
        ax3 = fig.add_subplot(gs[1, 0], facecolor='#16213e')
        scores = [c["score"] for c in self.cycles]
        bars = ax3.barh(names, scores, color=colors, edgecolor='white')
        ax3.set_xlabel('Overall Score', color='white')
        ax3.set_title('Decision Matrix Score', color='white', fontweight='bold')
        ax3.set_xlim(0, 1)
        self._style_axis(ax3)
        
        # Highlight winner
        best_idx = scores.index(max(scores))
        bars[best_idx].set_edgecolor('#16c79a')
        bars[best_idx].set_linewidth(3)
        
        # 4. Recommendation
        ax4 = fig.add_subplot(gs[1, 1], facecolor='#16213e')
        ax4.axis('off')
        
        best = self.cycles[best_idx]
        runner = sorted(self.cycles, key=lambda x: x["score"], reverse=True)[1]
        
        text = f"""RECOMMENDATION: {best['name'].upper()}

Performance:
  Isp: {best['isp_vac']:.0f} s
  Propellant: {best['propellant_kg']/1000:.1f} tonnes

Development:
  Cost: ${best['dev_cost_M']}M
  Time: {best['dev_years']} years

vs {runner['name']}:
  {(runner['propellant_kg'] - best['propellant_kg'])/1000:+.1f}t propellant
  ${runner['dev_cost_M'] - best['dev_cost_M']:+.0f}M cost
  {runner['dev_years'] - best['dev_years']:+d} years"""
        
        ax4.text(0.1, 0.9, text, transform=ax4.transAxes, fontsize=11,
                color='white', fontfamily='monospace', verticalalignment='top')
        for spine in ax4.spines.values():
            spine.set_color('#16c79a')
            spine.set_linewidth(2)
            spine.set_visible(True)
        
        fig.suptitle(f'Cycle Selection: {self.thrust_kn} kN, {self.payload_kg} kg payload',
                    color='white', fontsize=14, fontweight='bold')
        
        fig.savefig(output_dir / "cycle_selection.png", dpi=150, 
                   bbox_inches='tight', facecolor='#1a1a2e')
        plt.close(fig)
    
    def _style_axis(self, ax):
        ax.tick_params(colors='white')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('white')


# =============================================================================
# Engine Trade Study
# =============================================================================


@beartype
@dataclass  
class EngineTrade:
    """Parametric engine design study.
    
    Sweeps chamber pressure and mixture ratio to find optimal operating point.
    
    Example:
        >>> study = EngineTrade(
        ...     thrust_kn=100,
        ...     propellants=("LOX", "CH4"),
        ... )
        >>> study.run()
        >>> study.save("outputs/engine_trade")
    """
    
    thrust_kn: float | int
    propellants: tuple[str, str] = ("LOX", "CH4")
    pc_range_mpa: tuple[float | int, float | int] = (5, 25)
    mr_range: tuple[float, float] | None = None  # Auto from propellants
    
    # Results
    results: pl.DataFrame | None = None
    best_design: dict = field(default_factory=dict)
    _has_run: bool = field(default=False, repr=False)
    
    def run(self) -> "EngineTrade":
        """Run parametric sweep."""
        ox, fuel = self.propellants
        
        base = EngineInputs.from_propellants(
            oxidizer=ox, fuel=fuel,
            thrust=kilonewtons(self.thrust_kn),
            chamber_pressure=megapascals(10),
        )
        
        # Determine MR range
        if self.mr_range is None:
            base_mr = base.mixture_ratio
            self.mr_range = (base_mr * 0.8, base_mr * 1.2)
        
        study = ParametricStudy(
            compute=design_engine,
            base=base,
            vary={
                "chamber_pressure": Range(self.pc_range_mpa[0], self.pc_range_mpa[1], 
                                         n=10, unit="MPa"),
                "mixture_ratio": Range(self.mr_range[0], self.mr_range[1], n=8),
            },
        )
        
        study_results = study.run()
        self.results = study_results.to_dataframe()
        
        # Find best Isp design
        best_idx = self.results["isp_vac"].arg_max()
        best_row = self.results.row(best_idx, named=True)
        self.best_design = {
            "chamber_pressure_mpa": best_row["chamber_pressure"],
            "mixture_ratio": best_row["mixture_ratio"],
            "isp_vac": best_row["isp_vac"],
            "isp_sl": best_row["isp"],
            "throat_diameter_m": best_row["throat_diameter"],
        }
        
        self._has_run = True
        return self
    
    def save(self, output_dir: str | Path) -> Path:
        """Save analysis with plots."""
        if not self._has_run:
            self.run()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        self.results.write_csv(output_dir / "trade_data.csv")
        
        # Generate contour plot
        self._generate_plots(output_dir)
        
        return output_dir
    
    def _generate_plots(self, output_dir: Path) -> None:
        """Generate trade study plots."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#1a1a2e')
        
        pc = self.results["chamber_pressure"].to_numpy()
        mr = self.results["mixture_ratio"].to_numpy()
        isp = self.results["isp_vac"].to_numpy()
        
        # Reshape for contour (if grid)
        n_pc = len(set(pc))
        n_mr = len(set(mr))
        
        if len(pc) == n_pc * n_mr:
            PC = pc.reshape(n_pc, n_mr)
            MR = mr.reshape(n_pc, n_mr)
            ISP = isp.reshape(n_pc, n_mr)
            
            ax1 = axes[0]
            ax1.set_facecolor('#16213e')
            contour = ax1.contourf(PC, MR, ISP, levels=15, cmap='plasma')
            ax1.scatter([self.best_design["chamber_pressure_mpa"]], 
                       [self.best_design["mixture_ratio"]], 
                       s=200, c='white', marker='*', zorder=5)
            plt.colorbar(contour, ax=ax1, label='Isp (vac) [s]')
            ax1.set_xlabel('Chamber Pressure [MPa]', color='white')
            ax1.set_ylabel('Mixture Ratio', color='white')
            ax1.set_title('Isp Trade Space', color='white', fontweight='bold')
            ax1.tick_params(colors='white')
        
        # Isp vs Pc at optimal MR
        ax2 = axes[1]
        ax2.set_facecolor('#16213e')
        optimal_mr = self.best_design["mixture_ratio"]
        mask = np.abs(mr - optimal_mr) < 0.1
        ax2.plot(pc[mask], isp[mask], 'o-', color='#e94560', linewidth=2, markersize=8)
        ax2.axhline(self.best_design["isp_vac"], color='#16c79a', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Chamber Pressure [MPa]', color='white')
        ax2.set_ylabel('Isp (vac) [s]', color='white')
        ax2.set_title(f'Isp vs Pressure (MR≈{optimal_mr:.1f})', color='white', fontweight='bold')
        ax2.tick_params(colors='white')
        for spine in ['top', 'right']:
            ax2.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax2.spines[spine].set_color('white')
        
        fig.suptitle(f'{self.thrust_kn} kN {self.propellants[0]}/{self.propellants[1]} Engine Trade',
                    color='white', fontsize=14, fontweight='bold')
        
        fig.savefig(output_dir / "engine_trade.png", dpi=150, 
                   bbox_inches='tight', facecolor='#1a1a2e')
        plt.close(fig)


# =============================================================================
# Quick Design
# =============================================================================


@beartype
def quick_design(
    thrust_kn: float | int,
    propellants: tuple[str, str] = ("LOX", "CH4"),
    chamber_pressure_mpa: float | int = 10.0,
    output_dir: str | Path | None = None,
) -> dict:
    """Design an engine with one function call.
    
    Returns key metrics and optionally saves plots.
    
    Example:
        >>> result = quick_design(100, ("LOX", "CH4"))
        >>> print(f"Isp: {result['isp_vac']:.0f} s")
    """
    inputs = EngineInputs.from_propellants(
        oxidizer=propellants[0],
        fuel=propellants[1],
        thrust=kilonewtons(thrust_kn),
        chamber_pressure=megapascals(chamber_pressure_mpa),
    )
    
    perf, geom = design_engine(inputs)
    
    result = {
        "isp_vac": perf.isp_vac.value,
        "isp_sl": perf.isp.value,
        "cstar": perf.cstar.value,
        "thrust_coeff": perf.thrust_coeff,
        "throat_diameter_cm": geom.throat_diameter.to("m").value * 100,
        "exit_diameter_cm": geom.exit_diameter.to("m").value * 100,
        "mdot_kg_s": perf.mdot.value,
        "expansion_ratio": geom.expansion_ratio,
    }
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Simple dashboard
        from rocket.plotting import plot_engine_dashboard
        from rocket.nozzle import full_chamber_contour, generate_nozzle_from_geometry
        
        nozzle = generate_nozzle_from_geometry(geom)
        contour = full_chamber_contour(inputs, geom, nozzle)
        
        fig = plot_engine_dashboard(inputs, perf, geom, contour)
        fig.savefig(output_dir / "engine_design.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return result

