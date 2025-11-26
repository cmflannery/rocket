"""Visualization module for Rocket.

Provides plotting functions for:
- Engine cross-section views
- Performance curves (Isp vs altitude, thrust vs altitude)
- Nozzle contour visualization
- Trade study plots
- Cycle comparison charts

All plots use matplotlib with a consistent, professional style.
"""

import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from numpy.typing import NDArray

from rocket.engine import (
    EngineGeometry,
    EngineInputs,
    EnginePerformance,
    isp_at_altitude,
    thrust_at_altitude,
)
from rocket.nozzle import NozzleContour
from rocket.units import pascals

# =============================================================================
# Plot Style Configuration
# =============================================================================

# Professional color palette
COLORS = {
    "primary": "#2E86AB",  # Steel blue
    "secondary": "#A23B72",  # Berry
    "accent": "#F18F01",  # Orange
    "chamber": "#454545",  # Dark gray for chamber walls
    "fill": "#E8E8E8",  # Light gray for fill
    "grid": "#CCCCCC",  # Grid lines
    "text": "#333333",  # Text color
}

# Default figure size
DEFAULT_FIGSIZE = (12, 6)


def _setup_style() -> None:
    """Configure matplotlib style for consistent appearance."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.linewidth": 1.2,
            "axes.edgecolor": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "grid.alpha": 0.5,
            "grid.linewidth": 0.8,
        }
    )


# =============================================================================
# Engine Cross-Section Plot
# =============================================================================


@beartype
def plot_engine_cross_section(
    geometry: EngineGeometry,
    contour: NozzleContour,
    inputs: EngineInputs | None = None,
    show_dimensions: bool = True,
    show_centerline: bool = True,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    title: str | None = None,
) -> Figure:
    """Plot a 2D cross-section of the engine chamber and nozzle.

    Creates a symmetric cross-section view showing:
    - Chamber wall profile (top and bottom halves)
    - Throat location
    - Key dimensions (optional)
    - Centerline (optional)

    Args:
        geometry: Computed engine geometry
        contour: Nozzle contour (can be just nozzle or full chamber)
        inputs: Engine inputs (for title and additional info)
        show_dimensions: Whether to annotate key dimensions
        show_centerline: Whether to show the centerline
        figsize: Figure size (width, height) in inches
        title: Optional custom title

    Returns:
        matplotlib Figure object
    """
    _setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Get contour data
    x = contour.x
    y = contour.y

    # Create symmetric contour (top and bottom)
    x_full = np.concatenate([x, x[::-1]])
    y_full = np.concatenate([y, -y[::-1]])

    # Create filled polygon for chamber wall
    vertices = np.column_stack([x_full, y_full])
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices) - 2) + [MplPath.CLOSEPOLY]
    path = MplPath(vertices, codes)
    patch = PathPatch(
        path,
        facecolor=COLORS["fill"],
        edgecolor=COLORS["chamber"],
        linewidth=2,
    )
    ax.add_patch(patch)

    # Plot contour lines explicitly for clarity
    ax.plot(x, y, color=COLORS["chamber"], linewidth=2, label="Chamber wall")
    ax.plot(x, -y, color=COLORS["chamber"], linewidth=2)

    # Centerline
    if show_centerline:
        ax.axhline(y=0, color=COLORS["primary"], linestyle="--", linewidth=1, alpha=0.7)
        ax.text(
            x[-1] * 0.98,
            0,
            "CL",
            ha="right",
            va="bottom",
            fontsize=9,
            color=COLORS["primary"],
        )

    # Mark throat location (minimum radius point)
    throat_idx = np.argmin(y)
    throat_x = x[throat_idx]
    throat_y = y[throat_idx]

    ax.axvline(x=throat_x, color=COLORS["secondary"], linestyle=":", linewidth=1.5, alpha=0.7)
    ax.plot(throat_x, throat_y, "o", color=COLORS["secondary"], markersize=6)
    ax.plot(throat_x, -throat_y, "o", color=COLORS["secondary"], markersize=6)

    # Dimension annotations
    if show_dimensions:
        _add_dimension_annotations(ax, geometry, contour, x, y)

    # Set axis properties
    ax.set_aspect("equal")
    ax.set_xlabel("Axial Position (m)")
    ax.set_ylabel("Radial Position (m)")

    # Add margin
    x_margin = (x[-1] - x[0]) * 0.1
    y_max = max(y) * 1.3
    ax.set_xlim(x[0] - x_margin, x[-1] + x_margin)
    ax.set_ylim(-y_max, y_max)

    # Grid
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Title
    if title:
        ax.set_title(title)
    elif inputs and inputs.name:
        ax.set_title(f"Engine Cross-Section: {inputs.name}")
    else:
        ax.set_title("Engine Cross-Section")

    fig.tight_layout()
    return fig


def _add_dimension_annotations(
    ax: plt.Axes,
    geometry: EngineGeometry,
    contour: NozzleContour,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> None:
    """Add dimension annotations to the cross-section plot."""
    # Throat diameter
    throat_idx = np.argmin(y)
    throat_x = x[throat_idx]
    throat_r = y[throat_idx]

    # Exit diameter
    exit_r = y[-1]
    exit_x = x[-1]

    # Annotation style
    arrowprops = dict(arrowstyle="<->", color=COLORS["accent"], lw=1.5)
    text_offset = 0.02 * (x[-1] - x[0])

    # Throat diameter annotation
    Dt_mm = geometry.throat_diameter.to("m").value * 1000
    ax.annotate(
        "",
        xy=(throat_x, throat_r),
        xytext=(throat_x, -throat_r),
        arrowprops=arrowprops,
    )
    ax.text(
        throat_x + text_offset,
        0,
        f"Dt={Dt_mm:.1f}mm",
        fontsize=9,
        va="center",
        color=COLORS["accent"],
    )

    # Exit diameter annotation
    De_mm = geometry.exit_diameter.to("m").value * 1000
    ax.annotate(
        "",
        xy=(exit_x, exit_r),
        xytext=(exit_x, -exit_r),
        arrowprops=arrowprops,
    )
    ax.text(
        exit_x - text_offset,
        exit_r * 0.5,
        f"De={De_mm:.1f}mm",
        fontsize=9,
        va="center",
        ha="right",
        color=COLORS["accent"],
    )

    # Expansion ratio annotation
    eps = geometry.expansion_ratio
    ax.text(
        exit_x - text_offset,
        -exit_r * 0.5,
        f"ε={eps:.1f}",
        fontsize=9,
        va="center",
        ha="right",
        color=COLORS["text"],
    )


# =============================================================================
# Nozzle Contour Plot
# =============================================================================


@beartype
def plot_nozzle_contour(
    contour: NozzleContour,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
    units: str = "mm",
) -> Figure:
    """Plot a nozzle contour profile.

    Shows just the nozzle contour (single line, not symmetric view).
    Useful for verifying contour generation and CAD export.

    Args:
        contour: Nozzle contour to plot
        figsize: Figure size
        title: Optional title
        units: Display units ("m" or "mm")

    Returns:
        matplotlib Figure
    """
    _setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    if units == "mm":
        x = contour.x * 1000
        y = contour.y * 1000
        xlabel = "Axial Position (mm)"
        ylabel = "Radius (mm)"
    else:
        x = contour.x
        y = contour.y
        xlabel = "Axial Position (m)"
        ylabel = "Radius (m)"

    ax.plot(x, y, color=COLORS["primary"], linewidth=2, label="Contour")
    ax.fill_between(x, 0, y, alpha=0.2, color=COLORS["primary"])

    # Mark throat
    throat_idx = np.argmin(y)
    ax.axvline(x=x[throat_idx], color=COLORS["secondary"], linestyle=":", alpha=0.7)
    ax.plot(x[throat_idx], y[throat_idx], "o", color=COLORS["secondary"], markersize=8)
    ax.text(
        x[throat_idx],
        y[throat_idx] * 1.1,
        "Throat",
        ha="center",
        fontsize=10,
        color=COLORS["secondary"],
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"Nozzle Contour ({contour.contour_type})")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig


# =============================================================================
# Performance vs Altitude
# =============================================================================


@beartype
def plot_performance_vs_altitude(
    inputs: EngineInputs,
    performance: EnginePerformance,
    geometry: EngineGeometry,
    max_altitude_km: float | int = 100.0,
    num_points: int = 100,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> Figure:
    """Plot thrust and Isp vs altitude.

    Shows how engine performance changes with altitude due to
    decreasing ambient pressure.

    Args:
        inputs: Engine inputs
        performance: Computed performance
        geometry: Computed geometry
        max_altitude_km: Maximum altitude to plot (km)
        num_points: Number of altitude points
        figsize: Figure size

    Returns:
        matplotlib Figure with two subplots
    """
    _setup_style()

    # Generate altitude array
    altitudes_km = np.linspace(0, max_altitude_km, num_points)

    # Simple exponential atmosphere model
    # P = P0 * exp(-h/H) where H ≈ 8.5 km
    P0 = 101325  # Pa
    H = 8500  # m
    pressures_Pa = P0 * np.exp(-altitudes_km * 1000 / H)

    # Calculate thrust and Isp at each altitude
    thrust_vals = np.zeros(num_points)
    isp_vals = np.zeros(num_points)

    for i, pa in enumerate(pressures_Pa):
        pa_qty = pascals(pa)
        thrust_vals[i] = thrust_at_altitude(inputs, performance, geometry, pa_qty).to("kN").value
        isp_vals[i] = isp_at_altitude(inputs, performance, pa_qty).value

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Thrust plot
    ax1.plot(altitudes_km, thrust_vals, color=COLORS["primary"], linewidth=2)
    ax1.axhline(
        y=inputs.thrust.to("kN").value,
        color=COLORS["secondary"],
        linestyle="--",
        alpha=0.7,
        label="Design thrust (SL)",
    )
    ax1.set_xlabel("Altitude (km)")
    ax1.set_ylabel("Thrust (kN)")
    ax1.set_title("Thrust vs Altitude")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, max_altitude_km)

    # Isp plot
    ax2.plot(altitudes_km, isp_vals, color=COLORS["accent"], linewidth=2)
    ax2.axhline(
        y=performance.isp.value,
        color=COLORS["secondary"],
        linestyle="--",
        alpha=0.7,
        label="Isp (SL)",
    )
    ax2.axhline(
        y=performance.isp_vac.value,
        color=COLORS["primary"],
        linestyle=":",
        alpha=0.7,
        label="Isp (Vac)",
    )
    ax2.set_xlabel("Altitude (km)")
    ax2.set_ylabel("Specific Impulse (s)")
    ax2.set_title("Isp vs Altitude")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, max_altitude_km)

    # Overall title
    name = inputs.name or "Engine"
    fig.suptitle(f"Altitude Performance: {name}", fontsize=14, y=1.02)

    fig.tight_layout()
    return fig


# =============================================================================
# Trade Study Plots
# =============================================================================


@beartype
def plot_isp_vs_expansion_ratio(
    gamma: float | int = 1.2,
    pc_pe_range: tuple[float, float] = (10, 200),
    num_points: int = 100,
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """Plot theoretical Isp vs expansion ratio for different pressure ratios.

    Useful for understanding nozzle design trade-offs.

    Args:
        gamma: Ratio of specific heats
        pc_pe_range: Range of chamber-to-exit pressure ratios
        num_points: Number of points
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    from rocket.isentropic import (
        area_ratio_from_mach,
        mach_from_pressure_ratio,
        thrust_coefficient,
    )

    _setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Different pressure ratios to plot
    pc_pe_values = [20, 50, 100, 150, 200]

    for pc_pe in pc_pe_values:
        # Calculate exit Mach
        Me = mach_from_pressure_ratio(pc_pe, gamma)
        eps = area_ratio_from_mach(Me, gamma)

        # Calculate Cf for perfectly expanded nozzle (pa = pe)
        pe_pc = 1.0 / pc_pe
        Cf = thrust_coefficient(gamma, pe_pc, pe_pc, eps)

        # Plot point
        ax.scatter([eps], [Cf], s=100, zorder=5)
        ax.annotate(
            f"pc/pe={pc_pe}",
            xy=(eps, Cf),
            xytext=(10, 5),
            textcoords="offset points",
            fontsize=9,
        )

    # Generate curve for range of expansion ratios
    eps_range = np.linspace(2, 100, 200)
    Cf_optimal = np.zeros_like(eps_range)

    for i, eps in enumerate(eps_range):
        # Find pressure ratio that gives this expansion ratio
        Me = 2.0  # Initial guess
        for _ in range(50):
            eps_calc = area_ratio_from_mach(Me, gamma)
            if abs(eps_calc - eps) < 0.01:
                break
            Me += (eps - eps_calc) * 0.1

        # Cf for this expansion ratio (optimally expanded)
        pc_pe = (1 + (gamma - 1) / 2 * Me**2) ** (gamma / (gamma - 1))
        pe_pc = 1.0 / pc_pe
        Cf_optimal[i] = thrust_coefficient(gamma, pe_pc, pe_pc, eps)

    ax.plot(eps_range, Cf_optimal, color=COLORS["primary"], linewidth=2, label="Optimal Cf")

    ax.set_xlabel("Expansion Ratio (Ae/At)")
    ax.set_ylabel("Thrust Coefficient (Cf)")
    ax.set_title(f"Thrust Coefficient vs Expansion Ratio (γ={gamma})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig


# =============================================================================
# Summary Dashboard
# =============================================================================


@beartype
def plot_engine_dashboard(
    inputs: EngineInputs,
    performance: EnginePerformance,
    geometry: EngineGeometry,
    contour: NozzleContour,
    figsize: tuple[float, float] = (16, 10),
) -> Figure:
    """Create a comprehensive dashboard with engine summary.

    Includes:
    - Engine cross-section
    - Performance vs altitude
    - Key parameters table

    Args:
        inputs: Engine inputs
        performance: Computed performance
        geometry: Computed geometry
        contour: Nozzle contour
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    _setup_style()

    fig = plt.figure(figsize=figsize)

    # Create grid layout
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], width_ratios=[1.5, 1, 1])

    # Cross-section (spans two columns)
    ax_cross = fig.add_subplot(gs[0, :2])

    # Get contour data
    x = contour.x
    y = contour.y

    # Plot symmetric contour
    ax_cross.fill_between(x, y, -y, color=COLORS["fill"], alpha=0.8)
    ax_cross.plot(x, y, color=COLORS["chamber"], linewidth=2)
    ax_cross.plot(x, -y, color=COLORS["chamber"], linewidth=2)
    ax_cross.axhline(y=0, color=COLORS["primary"], linestyle="--", linewidth=1, alpha=0.5)

    # Mark throat
    throat_idx = np.argmin(y)
    ax_cross.axvline(x=x[throat_idx], color=COLORS["secondary"], linestyle=":", alpha=0.7)

    ax_cross.set_aspect("equal")
    ax_cross.set_xlabel("Axial Position (m)")
    ax_cross.set_ylabel("Radial Position (m)")
    ax_cross.set_title("Engine Cross-Section")
    ax_cross.grid(True, alpha=0.3)

    # Parameters table (right side of top row)
    ax_params = fig.add_subplot(gs[0, 2])
    ax_params.axis("off")

    # Create parameter text
    name = inputs.name or "Unnamed Engine"
    params_text = f"""
    {name}
    ─────────────────────
    PERFORMANCE
    Thrust (SL): {inputs.thrust.to('kN').value:.2f} kN
    Isp (SL):    {performance.isp.value:.1f} s
    Isp (Vac):   {performance.isp_vac.value:.1f} s
    C*:          {performance.cstar.value:.0f} m/s

    MASS FLOW
    Total:       {performance.mdot.value:.3f} kg/s
    O/F Ratio:   {inputs.mixture_ratio:.2f}

    GEOMETRY
    Dt:          {geometry.throat_diameter.to('m').value*1000:.1f} mm
    De:          {geometry.exit_diameter.to('m').value*1000:.1f} mm
    ε (Ae/At):   {geometry.expansion_ratio:.1f}

    CONDITIONS
    Pc:          {inputs.chamber_pressure.to('MPa').value:.2f} MPa
    Tc:          {inputs.chamber_temp.to('K').value:.0f} K
    γ:           {inputs.gamma:.3f}
    """

    ax_params.text(
        0.1,
        0.95,
        params_text,
        transform=ax_params.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor=COLORS["grid"]),
    )

    # Altitude performance plots (bottom row)
    altitudes_km = np.linspace(0, 80, 50)
    P0 = 101325
    H = 8500
    pressures_Pa = P0 * np.exp(-altitudes_km * 1000 / H)

    thrust_vals = np.zeros(len(altitudes_km))
    isp_vals = np.zeros(len(altitudes_km))

    for i, pa in enumerate(pressures_Pa):
        pa_qty = pascals(pa)
        thrust_vals[i] = thrust_at_altitude(inputs, performance, geometry, pa_qty).to("kN").value
        isp_vals[i] = isp_at_altitude(inputs, performance, pa_qty).value

    # Thrust vs altitude
    ax_thrust = fig.add_subplot(gs[1, 0])
    ax_thrust.plot(altitudes_km, thrust_vals, color=COLORS["primary"], linewidth=2)
    ax_thrust.set_xlabel("Altitude (km)")
    ax_thrust.set_ylabel("Thrust (kN)")
    ax_thrust.set_title("Thrust vs Altitude")
    ax_thrust.grid(True, alpha=0.3)

    # Isp vs altitude
    ax_isp = fig.add_subplot(gs[1, 1])
    ax_isp.plot(altitudes_km, isp_vals, color=COLORS["accent"], linewidth=2)
    ax_isp.axhline(y=performance.isp_vac.value, color=COLORS["secondary"], linestyle="--", alpha=0.7)
    ax_isp.set_xlabel("Altitude (km)")
    ax_isp.set_ylabel("Isp (s)")
    ax_isp.set_title("Specific Impulse vs Altitude")
    ax_isp.grid(True, alpha=0.3)

    # Nozzle contour detail
    ax_nozzle = fig.add_subplot(gs[1, 2])
    x_mm = contour.x * 1000
    y_mm = contour.y * 1000
    ax_nozzle.plot(x_mm, y_mm, color=COLORS["primary"], linewidth=2)
    ax_nozzle.fill_between(x_mm, 0, y_mm, alpha=0.2, color=COLORS["primary"])
    ax_nozzle.set_xlabel("x (mm)")
    ax_nozzle.set_ylabel("r (mm)")
    ax_nozzle.set_title(f"Nozzle Contour ({contour.contour_type})")
    ax_nozzle.grid(True, alpha=0.3)
    ax_nozzle.set_ylim(bottom=0)

    fig.suptitle(f"Engine Design Summary: {name}", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


# =============================================================================
# Mass Breakdown Plot
# =============================================================================


@beartype
def plot_mass_breakdown(
    masses: dict[str, float | int],
    title: str = "Mass Breakdown",
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    """Create a mass breakdown pie chart with bar chart.

    Args:
        masses: Dictionary mapping component names to masses (kg)
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    _setup_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    labels = list(masses.keys())
    values = list(masses.values())
    total = sum(values)

    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    # Pie chart
    ax1.pie(
        values,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%\n({pct*total/100:.0f} kg)",
        colors=colors,
        startangle=90,
        pctdistance=0.75,
    )
    ax1.set_title("Distribution")

    # Bar chart
    bars = ax2.barh(labels, values, color=colors)
    ax2.set_xlabel("Mass (kg)")
    ax2.set_title("Component Masses")

    # Add value labels on bars
    for bar, val in zip(bars, values, strict=True):
        ax2.text(
            val + total * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f} kg",
            va="center",
            fontsize=9,
        )

    ax2.set_xlim(0, max(values) * 1.2)

    fig.suptitle(f"{title} (Total: {total:,.0f} kg)", fontsize=14, fontweight="bold")
    fig.tight_layout()

    return fig


# =============================================================================
# Cycle Comparison Plots
# =============================================================================


@beartype
def plot_cycle_comparison_bars(
    cycle_data: list[dict],
    metrics: list[str] | None = None,
    figsize: tuple[float, float] = (14, 8),
    title: str = "Engine Cycle Comparison",
) -> Figure:
    """Create multi-metric bar chart comparing engine cycles.

    Args:
        cycle_data: List of dicts with keys 'name' and metric values.
            Example: [
                {'name': 'Pressure-Fed', 'net_isp': 320, 'efficiency': 1.0, ...},
                {'name': 'Gas Generator', 'net_isp': 310, 'efficiency': 0.95, ...},
            ]
        metrics: List of metrics to plot. Defaults to common cycle metrics.
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib Figure
    """
    _setup_style()

    if metrics is None:
        metrics = ["net_isp", "efficiency", "tank_pressure_MPa", "pump_power_kW"]

    # Filter to only metrics that exist in the data
    available_metrics = []
    for m in metrics:
        if all(m in d for d in cycle_data):
            available_metrics.append(m)

    if not available_metrics:
        raise ValueError("No valid metrics found in cycle_data")

    n_cycles = len(cycle_data)
    n_metrics = len(available_metrics)

    # Metric display names and units
    metric_info = {
        "net_isp": ("Net Isp", "s"),
        "efficiency": ("Cycle Efficiency", "%"),
        "tank_pressure_MPa": ("Tank Pressure", "MPa"),
        "pump_power_kW": ("Pump Power", "kW"),
        "turbine_power_kW": ("Turbine Power", "kW"),
    }

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    cycle_names = [d["name"] for d in cycle_data]
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], "#48A9A6", "#4B3F72"]

    for ax, metric in zip(axes, available_metrics, strict=True):
        values = [d.get(metric, 0) for d in cycle_data]

        # Scale efficiency to percentage
        if metric == "efficiency":
            values = [v * 100 for v in values]

        bars = ax.bar(cycle_names, values, color=colors[:n_cycles], edgecolor="white", linewidth=1.5)

        # Add value labels on bars
        for bar, val in zip(bars, values, strict=True):
            height = bar.get_height()
            ax.annotate(
                f"{val:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # Axis formatting
        display_name, unit = metric_info.get(metric, (metric, ""))
        ax.set_ylabel(f"{display_name} ({unit})" if unit else display_name)
        ax.set_title(display_name, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    return fig


@beartype
def plot_cycle_radar(
    cycle_data: list[dict],
    metrics: list[str] | None = None,
    figsize: tuple[float, float] = (10, 10),
    title: str = "Cycle Comparison Radar",
) -> Figure:
    """Create radar/spider chart comparing cycles across normalized dimensions.

    All metrics are normalized to 0-1 scale for comparison.

    Args:
        cycle_data: List of dicts with 'name' and metric values
        metrics: Metrics to include in radar. Defaults to standard set.
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib Figure
    """
    _setup_style()

    if metrics is None:
        metrics = ["net_isp", "efficiency", "simplicity", "tank_mass_factor"]

    # Filter to available metrics
    available_metrics = []
    for m in metrics:
        if all(m in d for d in cycle_data):
            available_metrics.append(m)

    if len(available_metrics) < 3:
        raise ValueError("Need at least 3 metrics for radar chart")

    n_metrics = len(available_metrics)

    # Metric display names
    metric_names = {
        "net_isp": "Performance\n(Isp)",
        "efficiency": "Efficiency",
        "simplicity": "Simplicity",
        "tank_mass_factor": "Low Tank\nPressure",
        "reliability": "Reliability",
        "trl": "Maturity\n(TRL)",
    }

    # Normalize all metrics to 0-1 scale
    normalized_data = []
    for d in cycle_data:
        norm_d = {"name": d["name"]}
        for m in available_metrics:
            values = [cd[m] for cd in cycle_data]
            min_val, max_val = min(values), max(values)
            if max_val > min_val:
                norm_d[m] = (d[m] - min_val) / (max_val - min_val)
            else:
                norm_d[m] = 1.0
        normalized_data.append(norm_d)

    # Setup radar chart
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], "#48A9A6", "#4B3F72"]

    for i, d in enumerate(normalized_data):
        values = [d[m] for m in available_metrics]
        values += values[:1]  # Complete the loop

        ax.plot(angles, values, "o-", linewidth=2, color=colors[i % len(colors)], label=d["name"])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

    # Set labels
    labels = [metric_names.get(m, m) for m in available_metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)

    # Radial limits
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, alpha=0.7)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", y=1.08)

    fig.tight_layout()

    return fig


@beartype
def plot_cycle_tradeoff(
    cycle_data: list[dict],
    x_metric: str = "net_isp",
    y_metric: str = "efficiency",
    size_metric: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    title: str = "Cycle Trade Space",
) -> Figure:
    """Create scatter plot showing cycle trade-offs.

    Plot cycles on 2D trade space with optional bubble size for third dimension.

    Args:
        cycle_data: List of dicts with 'name' and metric values
        x_metric: Metric for x-axis
        y_metric: Metric for y-axis
        size_metric: Optional metric for bubble size
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib Figure
    """
    _setup_style()

    # Metric display info
    metric_info = {
        "net_isp": ("Net Isp", "s"),
        "efficiency": ("Cycle Efficiency", ""),
        "tank_pressure_MPa": ("Tank Pressure", "MPa"),
        "pump_power_kW": ("Pump Power", "kW"),
        "simplicity": ("Simplicity Score", ""),
        "complexity": ("Complexity Score", ""),
    }

    fig, ax = plt.subplots(figsize=figsize)

    x_vals = [d[x_metric] for d in cycle_data]
    y_vals = [d[y_metric] for d in cycle_data]

    # Scale efficiency to percentage for display
    if y_metric == "efficiency":
        y_vals = [v * 100 for v in y_vals]

    colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], "#48A9A6", "#4B3F72"]

    if size_metric and all(size_metric in d for d in cycle_data):
        sizes = [d[size_metric] for d in cycle_data]
        # Normalize sizes for display
        max_size = max(sizes) if max(sizes) > 0 else 1
        normalized_sizes = [500 + 1500 * (s / max_size) for s in sizes]
    else:
        normalized_sizes = [800] * len(cycle_data)

    for i, (x, y, s, d) in enumerate(zip(x_vals, y_vals, normalized_sizes, cycle_data, strict=True)):
        ax.scatter(
            x, y, s=s, c=colors[i % len(colors)], alpha=0.7, edgecolors="white", linewidth=2, zorder=5
        )
        # Label
        ax.annotate(
            d["name"],
            xy=(x, y),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5),
        )

    # Axis formatting
    x_name, x_unit = metric_info.get(x_metric, (x_metric, ""))
    y_name, y_unit = metric_info.get(y_metric, (y_metric, ""))

    ax.set_xlabel(f"{x_name} ({x_unit})" if x_unit else x_name, fontsize=12)
    ax.set_ylabel(f"{y_name} ({y_unit})" if y_unit else y_name, fontsize=12)

    # Add quadrant annotations
    x_mid = (max(x_vals) + min(x_vals)) / 2
    y_mid = (max(y_vals) + min(y_vals)) / 2

    ax.axvline(x=x_mid, color="gray", linestyle="--", alpha=0.3)
    ax.axhline(y=y_mid, color="gray", linestyle="--", alpha=0.3)

    # Expand limits slightly
    x_range = max(x_vals) - min(x_vals)
    y_range = max(y_vals) - min(y_vals)
    ax.set_xlim(min(x_vals) - 0.1 * x_range, max(x_vals) + 0.15 * x_range)
    ax.set_ylim(min(y_vals) - 0.1 * y_range, max(y_vals) + 0.15 * y_range)

    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add size legend if applicable
    if size_metric and all(size_metric in d for d in cycle_data):
        size_name = metric_info.get(size_metric, (size_metric, ""))[0]
        ax.text(
            0.02,
            0.02,
            f"Bubble size: {size_name}",
            transform=ax.transAxes,
            fontsize=9,
            alpha=0.7,
        )

    fig.tight_layout()

    return fig


# =============================================================================
# Transient Thermal Plots
# =============================================================================


@beartype
def plot_thermal_transient(
    time: np.ndarray,
    wall_temp_inner: np.ndarray,
    wall_temp_outer: np.ndarray | None = None,
    heat_flux: np.ndarray | None = None,
    max_temp: float | None = None,
    title: str = "Thermal Transient Analysis",
    figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """Plot thermal transient simulation results.

    Creates a multi-panel plot showing:
    - Wall temperature vs time
    - Heat flux vs time (optional)
    - Material limit comparison

    Args:
        time: Time array [s]
        wall_temp_inner: Inner wall temperature [K]
        wall_temp_outer: Outer wall temperature [K] (optional)
        heat_flux: Heat flux array [W/m²] (optional)
        max_temp: Material temperature limit [K] (optional)
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    n_plots = 1 + (heat_flux is not None)

    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    if n_plots == 1:
        axes = [axes]

    # Color scheme
    inner_color = "#e63946"  # Red for hot side
    outer_color = "#457b9d"  # Blue for cold side
    limit_color = "#f4a261"  # Orange for limits

    # Temperature plot
    ax_temp = axes[0]
    ax_temp.plot(time, wall_temp_inner, color=inner_color, linewidth=2, label="Inner wall (gas side)")
    if wall_temp_outer is not None:
        ax_temp.plot(time, wall_temp_outer, color=outer_color, linewidth=2, label="Outer wall (coolant side)")

    if max_temp is not None:
        ax_temp.axhline(y=max_temp, color=limit_color, linestyle="--", linewidth=2, label=f"Material limit ({max_temp:.0f} K)")
        ax_temp.fill_between(time, max_temp, max(wall_temp_inner.max(), max_temp) * 1.1,
                             color=limit_color, alpha=0.2)

    ax_temp.set_ylabel("Temperature [K]", fontsize=11)
    ax_temp.legend(loc="upper right")
    ax_temp.grid(True, alpha=0.3)
    ax_temp.set_title(title, fontsize=14, fontweight="bold")

    # Heat flux plot
    if heat_flux is not None:
        ax_q = axes[1]
        q_mw = heat_flux / 1e6  # Convert to MW/m²
        ax_q.fill_between(time, 0, q_mw, color="#2a9d8f", alpha=0.4)
        ax_q.plot(time, q_mw, color="#2a9d8f", linewidth=2)
        ax_q.set_ylabel("Heat Flux [MW/m²]", fontsize=11)
        ax_q.set_xlabel("Time [s]", fontsize=11)
        ax_q.grid(True, alpha=0.3)
    else:
        axes[-1].set_xlabel("Time [s]", fontsize=11)

    fig.tight_layout()
    return fig


@beartype
def plot_thermal_margin(
    time: np.ndarray,
    thermal_margin: np.ndarray,
    title: str = "Thermal Margin Analysis",
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """Plot thermal margin over time.

    Shows how close the wall temperature gets to the material limit.
    Positive margin = safe, negative = exceeding limit.

    Args:
        time: Time array [s]
        thermal_margin: Temperature margin [K] (T_limit - T_wall)
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Color by safety
    safe_mask = thermal_margin >= 0

    ax.fill_between(time, 0, thermal_margin, where=safe_mask, color="#2a9d8f", alpha=0.3, label="Safe")
    ax.fill_between(time, 0, thermal_margin, where=~safe_mask, color="#e63946", alpha=0.3, label="Exceeds limit")
    ax.plot(time, thermal_margin, color="#264653", linewidth=2)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

    # Warning zones
    ax.axhline(y=50, color="#f4a261", linestyle="--", alpha=0.7, label="50 K margin")
    ax.axhline(y=100, color="#e9c46a", linestyle="--", alpha=0.7, label="100 K margin")

    ax.set_xlabel("Time [s]", fontsize=11)
    ax.set_ylabel("Thermal Margin [K]", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add min margin annotation
    min_margin = np.min(thermal_margin)
    min_idx = np.argmin(thermal_margin)
    ax.annotate(
        f"Min: {min_margin:.0f} K",
        xy=(time[min_idx], min_margin),
        xytext=(time[min_idx] + 0.5, min_margin + 30),
        arrowprops=dict(arrowstyle="->", color="#264653"),
        fontsize=10,
    )

    fig.tight_layout()
    return fig


@beartype
def plot_duty_cycle_thermal(
    time: np.ndarray,
    wall_temp: np.ndarray,
    heat_flux: np.ndarray,
    max_temp: float,
    burn_time: float,
    coast_time: float,
    n_cycles: int,
    title: str = "Duty Cycle Thermal Analysis",
    figsize: tuple[float, float] = (14, 10),
) -> Figure:
    """Plot thermal behavior over multiple burn/coast cycles.

    Args:
        time: Time array [s]
        wall_temp: Wall temperature [K]
        heat_flux: Heat flux [W/m²]
        max_temp: Material temperature limit [K]
        burn_time: Duration of each burn [s]
        coast_time: Duration of coast between burns [s]
        n_cycles: Number of cycles
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    cycle_time = burn_time + coast_time

    # Temperature plot
    ax_temp = axes[0]
    ax_temp.plot(time, wall_temp, color="#e63946", linewidth=1.5)
    ax_temp.axhline(y=max_temp, color="#f4a261", linestyle="--", linewidth=2, label=f"Limit: {max_temp:.0f} K")

    # Shade burn periods
    for i in range(n_cycles):
        start = i * cycle_time
        end = start + burn_time
        ax_temp.axvspan(start, end, color="#ffb4a2", alpha=0.3)

    ax_temp.set_ylabel("Wall Temp [K]", fontsize=11)
    ax_temp.legend(loc="upper right")
    ax_temp.grid(True, alpha=0.3)
    ax_temp.set_title(title, fontsize=14, fontweight="bold")

    # Heat flux plot
    ax_q = axes[1]
    q_mw = heat_flux / 1e6
    ax_q.fill_between(time, 0, q_mw, color="#2a9d8f", alpha=0.4)
    ax_q.plot(time, q_mw, color="#2a9d8f", linewidth=1)
    ax_q.set_ylabel("Heat Flux [MW/m²]", fontsize=11)
    ax_q.grid(True, alpha=0.3)

    # Thermal margin plot
    ax_margin = axes[2]
    margin = max_temp - wall_temp
    safe_mask = margin >= 0
    ax_margin.fill_between(time, 0, margin, where=safe_mask, color="#2a9d8f", alpha=0.3)
    ax_margin.fill_between(time, 0, margin, where=~safe_mask, color="#e63946", alpha=0.3)
    ax_margin.plot(time, margin, color="#264653", linewidth=1.5)
    ax_margin.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax_margin.set_ylabel("Margin [K]", fontsize=11)
    ax_margin.set_xlabel("Time [s]", fontsize=11)
    ax_margin.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
