"""Visualization module for Rocket.

Provides plotting functions for:
- Engine cross-section views
- Performance curves (Isp vs altitude, thrust vs altitude)
- Nozzle contour visualization
- Trade study plots

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


@beartype
def plot_mass_breakdown(
    masses: dict[str, float | int],
    title: str = "Vehicle Mass Breakdown",
) -> Figure:
    """Plot a mass breakdown pie chart and bar chart.

    Args:
        masses: Dictionary of component names to masses in kg
        title: Plot title

    Returns:
        matplotlib Figure
    """
    _setup_style()
    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))

    # Sort by mass
    sorted_items = sorted(masses.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    total = sum(values)

    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    # Pie chart
    wedges, texts, autotexts = ax_pie.pie(
        values, labels=None, autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        colors=colors, startangle=90, counterclock=False,
        wedgeprops=dict(linewidth=2, edgecolor="white")
    )
    ax_pie.set_title("Mass Distribution", fontsize=12, fontweight="bold")

    # Legend for pie chart
    legend_labels = [f"{l}: {v:,.0f} kg" for l, v in zip(labels, values)]
    ax_pie.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5))

    # Bar chart
    y_pos = np.arange(len(labels))
    bars = ax_bar.barh(y_pos, values, color=colors, edgecolor="black", linewidth=1)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels)
    ax_bar.set_xlabel("Mass (kg)")
    ax_bar.set_title("Component Masses", fontsize=12, fontweight="bold")
    ax_bar.grid(True, alpha=0.3, axis="x")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax_bar.text(width + total * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:,.0f} kg", va="center", fontsize=9)

    ax_bar.set_xlim(0, max(values) * 1.15)

    # Total mass annotation
    fig.suptitle(f"{title}\nTotal: {total:,.0f} kg", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    return fig

