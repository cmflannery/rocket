#!/usr/bin/env python
"""Project workflow example with rich visualizations.

This example demonstrates design management with comprehensive plots:
1. Design comparison dashboard
2. Trade study visualization
3. Project summary report

All outputs saved as publication-ready figures.
"""

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from rocket import EngineInputs, design_engine
from rocket.analysis import ParametricStudy, Range
from rocket.project import Project
from rocket.units import megapascals, newtons


def plot_design_comparison(designs: list[dict], figsize: tuple = (14, 10)) -> plt.Figure:
    """Create a dashboard comparing multiple engine designs."""
    fig = plt.figure(figsize=figsize, facecolor='#1a1a2e')
    
    # Color palette
    colors = ['#e94560', '#0f3460', '#16c79a', '#f4a261']
    
    # Extract data
    names = [d['name'] for d in designs]
    isp_vac = [d['isp_vac'] for d in designs]
    isp_sl = [d['isp_sl'] for d in designs]
    thrust = [d['thrust_kN'] for d in designs]
    pc = [d['pc_MPa'] for d in designs]
    throat_dia = [d['throat_cm'] for d in designs]
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Isp comparison (bar chart)
    ax1 = fig.add_subplot(gs[0, 0], facecolor='#16213e')
    x = np.arange(len(names))
    width = 0.35
    bars1 = ax1.bar(x - width/2, isp_vac, width, label='Vacuum', color=colors[0], edgecolor='white', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, isp_sl, width, label='Sea Level', color=colors[1], edgecolor='white', linewidth=0.5)
    ax1.set_ylabel('Specific Impulse [s]', color='white', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace('_', '\n') for n in names], color='white', fontsize=8)
    ax1.legend(facecolor='#16213e', edgecolor='white', labelcolor='white', fontsize=8)
    ax1.set_title('Performance Comparison', color='white', fontsize=12, fontweight='bold')
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, max(isp_vac) * 1.15)
    
    # Add value labels
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, 
                f'{bar.get_height():.0f}', ha='center', va='bottom', color='white', fontsize=8)
    
    # 2. Chamber pressure vs Throat size (scatter)
    ax2 = fig.add_subplot(gs[0, 1], facecolor='#16213e')
    scatter = ax2.scatter(pc, throat_dia, c=isp_vac, s=[t*2 for t in thrust], 
                          cmap='plasma', alpha=0.8, edgecolors='white', linewidth=1)
    for i, name in enumerate(names):
        ax2.annotate(name.split('_')[0], (pc[i], throat_dia[i]), 
                    xytext=(5, 5), textcoords='offset points', color='white', fontsize=8)
    ax2.set_xlabel('Chamber Pressure [MPa]', color='white', fontsize=10)
    ax2.set_ylabel('Throat Diameter [cm]', color='white', fontsize=10)
    ax2.set_title('Size vs Pressure Trade', color='white', fontsize=12, fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
    cbar.set_label('Isp (vac) [s]', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # 3. Radar chart
    ax3 = fig.add_subplot(gs[0, 2], polar=True, facecolor='#16213e')
    categories = ['Isp\n(vac)', 'Thrust', 'Chamber\nPressure', 'Compactness', 'Efficiency']
    
    # Normalize metrics to 0-1
    def normalize(vals, invert=False):
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return [0.5] * len(vals)
        norm = [(v - min_v) / (max_v - min_v) for v in vals]
        return [1 - n for n in norm] if invert else norm
    
    compactness = [1/t for t in throat_dia]  # Smaller is more compact
    efficiency = [isp_vac[i] / pc[i] for i in range(len(names))]  # Isp per MPa
    
    metrics = np.array([
        normalize(isp_vac),
        normalize(thrust),
        normalize(pc),
        normalize(compactness),
        normalize(efficiency),
    ]).T
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    for i, (name, data) in enumerate(zip(names, metrics)):
        values = data.tolist()
        values += values[:1]
        ax3.plot(angles, values, 'o-', linewidth=2, label=name.split('_')[0], color=colors[i % len(colors)])
        ax3.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
    
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, color='white', fontsize=8)
    ax3.set_ylim(0, 1)
    ax3.set_yticklabels([])
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1), facecolor='#16213e', 
               edgecolor='white', labelcolor='white', fontsize=8)
    ax3.set_title('Design Attributes', color='white', fontsize=12, fontweight='bold', pad=20)
    ax3.tick_params(colors='white')
    ax3.spines['polar'].set_color('white')
    
    # 4. Key metrics table
    ax4 = fig.add_subplot(gs[1, :], facecolor='#16213e')
    ax4.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Design', 'Thrust\n[kN]', 'Pc\n[MPa]', 'Isp (vac)\n[s]', 'Isp (SL)\n[s]', 
               'Throat\n[cm]', 'Mass Flow\n[kg/s]', 'Exit\n[cm]']
    
    for d in designs:
        table_data.append([
            d['name'].replace('_', ' '),
            f"{d['thrust_kN']:.0f}",
            f"{d['pc_MPa']:.1f}",
            f"{d['isp_vac']:.1f}",
            f"{d['isp_sl']:.1f}",
            f"{d['throat_cm']:.2f}",
            f"{d['mdot']:.2f}",
            f"{d['exit_cm']:.2f}",
        ])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=['#0f3460'] * len(headers),
        cellColours=[['#1a1a2e'] * len(headers) for _ in designs],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style table
    for (row, col), cell in table.get_celld().items():
        cell.set_text_props(color='white')
        cell.set_edgecolor('white')
        if row == 0:
            cell.set_text_props(fontweight='bold')
    
    ax4.set_title('Design Summary', color='white', fontsize=14, fontweight='bold', y=0.95)
    
    fig.suptitle('Engine Design Comparison Dashboard', color='white', fontsize=16, fontweight='bold', y=0.98)
    
    return fig


def plot_trade_study(df: pl.DataFrame, x_col: str, y_cols: list[str], 
                     x_label: str, title: str, figsize: tuple = (12, 8)) -> plt.Figure:
    """Create a trade study visualization."""
    fig, axes = plt.subplots(len(y_cols), 1, figsize=figsize, sharex=True, facecolor='#1a1a2e')
    if len(y_cols) == 1:
        axes = [axes]
    
    colors = ['#e94560', '#16c79a', '#f4a261', '#0f3460']
    x = df[x_col].to_numpy()
    
    for i, (ax, y_col) in enumerate(zip(axes, y_cols)):
        ax.set_facecolor('#16213e')
        y = df[y_col].to_numpy()
        
        # Plot with gradient fill
        ax.fill_between(x, y.min() * 0.95, y, alpha=0.3, color=colors[i % len(colors)])
        ax.plot(x, y, 'o-', color=colors[i % len(colors)], linewidth=2, markersize=6)
        
        # Mark best point
        best_idx = np.argmax(y) if 'isp' in y_col.lower() else np.argmin(y)
        ax.scatter([x[best_idx]], [y[best_idx]], s=150, color='#f4a261', 
                   zorder=5, edgecolors='white', linewidth=2, marker='*')
        ax.annotate(f'Best: {y[best_idx]:.1f}', (x[best_idx], y[best_idx]),
                   xytext=(10, 10), textcoords='offset points', color='white', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='#0f3460', edgecolor='white'))
        
        # Formatting
        y_label = y_col.replace('_', ' ').title()
        if 'isp' in y_col.lower():
            y_label = y_col.replace('isp', 'Isp').replace('_', ' ')
            y_label += ' [s]'
        elif 'diameter' in y_col.lower():
            y_label += ' [m]'
        
        ax.set_ylabel(y_label, color='white', fontsize=11)
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    axes[-1].set_xlabel(x_label, color='white', fontsize=11)
    fig.suptitle(title, color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_project_summary(project: Project, designs_data: list[dict], 
                         figsize: tuple = (16, 10)) -> plt.Figure:
    """Create a comprehensive project summary visualization."""
    fig = plt.figure(figsize=figsize, facecolor='#1a1a2e')
    
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3, 
                          height_ratios=[0.3, 1, 1])
    
    # Header
    ax_header = fig.add_subplot(gs[0, :], facecolor='#1a1a2e')
    ax_header.axis('off')
    
    header_text = f"""
    {project.name}
    {project.description}
    
    Created: {project.created_at.strftime('%Y-%m-%d')}  |  Author: {project.author}  |  Designs: {len(project.list_designs())}
    """
    ax_header.text(0.5, 0.5, header_text, transform=ax_header.transAxes,
                   fontsize=12, color='white', ha='center', va='center',
                   fontfamily='monospace')
    
    # Design cards
    for i, design in enumerate(designs_data[:4]):
        ax = fig.add_subplot(gs[1, i], facecolor='#16213e')
        ax.axis('off')
        
        # Card content
        card_text = f"""
{design['name'].replace('_', ' ').upper()}

Thrust:    {design['thrust_kN']:.0f} kN
Pc:        {design['pc_MPa']:.1f} MPa
Isp (vac): {design['isp_vac']:.1f} s
Isp (SL):  {design['isp_sl']:.1f} s
Throat:    {design['throat_cm']:.2f} cm
"""
        ax.text(0.5, 0.5, card_text, transform=ax.transAxes,
                fontsize=10, color='white', ha='center', va='center',
                fontfamily='monospace', linespacing=1.5)
        
        # Card border
        for spine in ax.spines.values():
            spine.set_color('#e94560')
            spine.set_linewidth(2)
            spine.set_visible(True)
    
    # Performance comparison (bottom row)
    ax_perf = fig.add_subplot(gs[2, :2], facecolor='#16213e')
    
    names = [d['name'].split('_')[0] for d in designs_data]
    isp_vac = [d['isp_vac'] for d in designs_data]
    isp_sl = [d['isp_sl'] for d in designs_data]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax_perf.barh(x - width/2, isp_vac, width, label='Vacuum Isp', 
                         color='#e94560', edgecolor='white')
    bars2 = ax_perf.barh(x + width/2, isp_sl, width, label='Sea Level Isp', 
                         color='#0f3460', edgecolor='white')
    
    ax_perf.set_yticks(x)
    ax_perf.set_yticklabels(names, color='white', fontsize=10)
    ax_perf.set_xlabel('Specific Impulse [s]', color='white', fontsize=11)
    ax_perf.legend(facecolor='#16213e', edgecolor='white', labelcolor='white')
    ax_perf.tick_params(colors='white')
    ax_perf.spines['bottom'].set_color('white')
    ax_perf.spines['left'].set_color('white')
    ax_perf.spines['top'].set_visible(False)
    ax_perf.spines['right'].set_visible(False)
    ax_perf.set_title('Performance Comparison', color='white', fontsize=12, fontweight='bold')
    
    # Geometry comparison
    ax_geom = fig.add_subplot(gs[2, 2:], facecolor='#16213e')
    
    throat = [d['throat_cm'] for d in designs_data]
    exit_d = [d['exit_cm'] for d in designs_data]
    
    ax_geom.scatter(throat, exit_d, s=[d['thrust_kN']*3 for d in designs_data],
                    c=[d['pc_MPa'] for d in designs_data], cmap='plasma',
                    alpha=0.8, edgecolors='white', linewidth=2)
    
    for i, name in enumerate(names):
        ax_geom.annotate(name, (throat[i], exit_d[i]), 
                        xytext=(8, 8), textcoords='offset points', 
                        color='white', fontsize=9)
    
    ax_geom.set_xlabel('Throat Diameter [cm]', color='white', fontsize=11)
    ax_geom.set_ylabel('Exit Diameter [cm]', color='white', fontsize=11)
    ax_geom.tick_params(colors='white')
    ax_geom.spines['bottom'].set_color('white')
    ax_geom.spines['left'].set_color('white')
    ax_geom.spines['top'].set_visible(False)
    ax_geom.spines['right'].set_visible(False)
    ax_geom.set_title('Geometry Trade Space\n(size = thrust, color = Pc)', 
                      color='white', fontsize=12, fontweight='bold')
    
    return fig


def main() -> None:
    """Run the project workflow with visualizations."""
    # Setup - all outputs go in the project directory
    project_path = Path("./outputs/methalox_upper_stage_study")
    
    if project_path.exists():
        shutil.rmtree(project_path)
    
    # Create project
    project = Project(
        path=project_path,
        name="Methalox Upper Stage Study",
        description="Trade study for a 100 kN class methalox upper stage engine",
        author="Rocket Team",
    )
    
    # Define and analyze designs
    design_specs = [
        ("baseline", 100_000, 8.0, "Conservative baseline"),
        ("high_perf", 100_000, 15.0, "High pressure for max Isp"),
        ("lightweight", 80_000, 6.0, "Lower thrust for mass savings"),
        ("balanced", 100_000, 10.0, "Balanced performance"),
    ]
    
    designs_data = []
    
    for name, thrust, pc, desc in design_specs:
        inputs = EngineInputs.from_propellants(
            oxidizer="LOX", fuel="CH4",
            thrust=newtons(thrust),
            chamber_pressure=megapascals(pc),
            name=name,
        )
        
        perf, geom = design_engine(inputs)
        
        # Save to project
        project.save_design(inputs, f"{name}_v1", description=desc)
        
        # Collect data for plotting
        designs_data.append({
            'name': f"{name}_v1",
            'thrust_kN': thrust / 1000,
            'pc_MPa': pc,
            'isp_vac': perf.isp_vac.value,
            'isp_sl': perf.isp.value,
            'throat_cm': geom.throat_diameter.to('m').value * 100,
            'exit_cm': geom.exit_diameter.to('m').value * 100,
            'mdot': perf.mdot.value,
        })
    
    # Generate design comparison dashboard
    plots_path = project_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)
    
    fig1 = plot_design_comparison(designs_data)
    fig1.savefig(plots_path / "design_comparison.png", dpi=150, 
                 bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig1)
    
    # Run trade study on baseline
    baseline = project.load_design("baseline_v1", EngineInputs)
    
    study = ParametricStudy(
        compute=design_engine,
        base=baseline,
        vary={"chamber_pressure": Range(5, 20, n=16, unit="MPa")},
    )
    results = study.run()
    df = results.to_dataframe()
    
    # Save results
    project.save_results(
        "baseline_v1", "pc_trade", "parametric",
        {"best_isp": float(df['isp_vac'].max())},
        data=df
    )
    
    # Generate trade study plot
    fig2 = plot_trade_study(
        df, 'chamber_pressure', ['isp_vac', 'isp', 'throat_diameter'],
        'Chamber Pressure [MPa]',
        'Chamber Pressure Trade Study - Baseline Design'
    )
    fig2.savefig(plots_path / "trade_study.png", dpi=150, 
                 bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig2)
    
    # Generate project summary
    fig3 = plot_project_summary(project, designs_data)
    fig3.savefig(plots_path / "project_summary.png", dpi=150, 
                 bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig3)
    
    # Minimal console output - just show where to find everything
    print(f"\n  Project saved to: {project_path.absolute()}")
    print()


if __name__ == "__main__":
    main()
