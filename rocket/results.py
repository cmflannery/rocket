"""Results visualization and Pareto front analysis for Rocket.

This module provides plotting utilities for parametric study and uncertainty
analysis results. Integrates with the analysis module for seamless visualization.

Example:
    >>> from rocket.analysis import ParametricStudy, Range
    >>> from rocket.results import plot_1d, plot_2d_contour, plot_pareto
    >>>
    >>> results = study.run()
    >>> fig = plot_1d(results, "chamber_pressure", "isp_vac")
    >>> fig = plot_pareto(results, "isp_vac", "thrust_to_weight", maximize=[True, True])
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype
from matplotlib.figure import Figure
from numpy.typing import NDArray

from rocket.analysis import StudyResults, UncertaintyResults

# =============================================================================
# Plot Style Configuration
# =============================================================================

COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "feasible": "#2E86AB",
    "infeasible": "#CCCCCC",
    "pareto": "#E63946",
    "grid": "#CCCCCC",
    "text": "#333333",
}


def _setup_style() -> None:
    """Configure matplotlib style for consistent appearance."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.linewidth": 1.2,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "grid.alpha": 0.5,
    })


# =============================================================================
# 1D Parameter Sweep Plots
# =============================================================================


@beartype
def plot_1d(
    results: StudyResults,
    x_param: str,
    y_metric: str,
    show_infeasible: bool = True,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> Figure:
    """Plot a 1D parameter sweep.

    Args:
        results: StudyResults from ParametricStudy
        x_param: Parameter name for x-axis
        y_metric: Metric name for y-axis
        show_infeasible: Whether to show infeasible points (grayed out)
        figsize: Figure size (width, height)
        title: Optional plot title
        xlabel: Optional x-axis label
        ylabel: Optional y-axis label

    Returns:
        matplotlib Figure
    """
    _setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    x = results.get_parameter(x_param)
    y = results.get_metric(y_metric)

    if results.constraints_passed is not None and show_infeasible:
        # Plot infeasible points first (gray)
        infeasible = ~results.constraints_passed
        if np.any(infeasible):
            ax.scatter(
                x[infeasible], y[infeasible],
                c=COLORS["infeasible"], s=50, alpha=0.5,
                label="Infeasible", zorder=1,
            )

        # Plot feasible points
        feasible = results.constraints_passed
        if np.any(feasible):
            ax.scatter(
                x[feasible], y[feasible],
                c=COLORS["primary"], s=80,
                label="Feasible", zorder=2,
            )
            ax.plot(
                x[feasible], y[feasible],
                c=COLORS["primary"], alpha=0.5, linewidth=1.5, zorder=1,
            )
    else:
        ax.scatter(x, y, c=COLORS["primary"], s=80)
        ax.plot(x, y, c=COLORS["primary"], alpha=0.5, linewidth=1.5)

    # Mark optimum
    if results.constraints_passed is not None:
        feasible_mask = results.constraints_passed
        if np.any(feasible_mask):
            best_idx = np.argmax(y * feasible_mask.astype(float) - (~feasible_mask) * 1e10)
            ax.scatter(
                [x[best_idx]], [y[best_idx]],
                c=COLORS["accent"], s=150, marker="*",
                label=f"Best: {y[best_idx]:.3g}", zorder=3,
            )

    ax.set_xlabel(xlabel or x_param)
    ax.set_ylabel(ylabel or y_metric)
    ax.set_title(title or f"{y_metric} vs {x_param}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig


@beartype
def plot_1d_multi(
    results: StudyResults,
    x_param: str,
    y_metrics: list[str],
    figsize: tuple[float, float] = (12, 6),
    title: str | None = None,
    xlabel: str | None = None,
    normalize: bool = False,
) -> Figure:
    """Plot multiple metrics vs a single parameter.

    Args:
        results: StudyResults from ParametricStudy
        x_param: Parameter name for x-axis
        y_metrics: List of metric names to plot
        figsize: Figure size
        title: Optional title
        xlabel: Optional x-axis label
        normalize: If True, normalize all metrics to [0, 1]

    Returns:
        matplotlib Figure
    """
    _setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    x = results.get_parameter(x_param)
    colors = plt.cm.tab10(np.linspace(0, 1, len(y_metrics)))

    for i, metric in enumerate(y_metrics):
        y = results.get_metric(metric)
        if normalize:
            y = (y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y) + 1e-10)

        ax.plot(x, y, color=colors[i], linewidth=2, label=metric, marker="o", markersize=4)

    ax.set_xlabel(xlabel or x_param)
    ax.set_ylabel("Normalized Value" if normalize else "Metric Value")
    ax.set_title(title or f"Metrics vs {x_param}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    return fig


# =============================================================================
# 2D Contour Plots
# =============================================================================


@beartype
def plot_2d_contour(
    results: StudyResults,
    x_param: str,
    y_param: str,
    z_metric: str,
    figsize: tuple[float, float] = (10, 8),
    title: str | None = None,
    levels: int = 20,
    show_points: bool = True,
    show_infeasible: bool = True,
) -> Figure:
    """Plot a 2D contour for two-parameter sweeps.

    Args:
        results: StudyResults from ParametricStudy (must be 2D grid)
        x_param: Parameter for x-axis
        y_param: Parameter for y-axis
        z_metric: Metric for contour values
        figsize: Figure size
        title: Optional title
        levels: Number of contour levels
        show_points: Show scatter points at evaluated locations
        show_infeasible: Mark infeasible regions

    Returns:
        matplotlib Figure
    """
    _setup_style()

    x = results.get_parameter(x_param)
    y = results.get_parameter(y_param)
    z = results.get_metric(z_metric)

    # Get unique values to determine grid shape
    x_unique = np.unique(x)
    y_unique = np.unique(y)

    if len(x_unique) * len(y_unique) != len(x):
        raise ValueError(
            f"Data is not a complete 2D grid. Got {len(x)} points, "
            f"expected {len(x_unique)} x {len(y_unique)} = {len(x_unique) * len(y_unique)}"
        )

    # Reshape to 2D grid
    nx, ny = len(x_unique), len(y_unique)
    X = x.reshape(ny, nx)
    Y = y.reshape(ny, nx)
    Z = z.reshape(ny, nx)

    fig, ax = plt.subplots(figsize=figsize)

    # Contour plot
    contour = ax.contourf(X, Y, Z, levels=levels, cmap="viridis")
    ax.contour(X, Y, Z, levels=levels, colors="white", alpha=0.3, linewidths=0.5)

    # Colorbar
    _ = fig.colorbar(contour, ax=ax, label=z_metric)

    # Show evaluated points
    if show_points:
        ax.scatter(x, y, c="white", s=10, alpha=0.5, edgecolors="black", linewidths=0.5)

    # Mark infeasible regions
    if show_infeasible and results.constraints_passed is not None:
        infeasible = ~results.constraints_passed
        if np.any(infeasible):
            ax.scatter(
                x[infeasible], y[infeasible],
                c="red", s=30, marker="x", alpha=0.7,
                label="Infeasible",
            )
            ax.legend()

    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(title or f"{z_metric}")

    fig.tight_layout()
    return fig


# =============================================================================
# Pareto Front Analysis
# =============================================================================


def _compute_pareto_front(
    objectives: NDArray[np.float64],
    maximize: Sequence[bool],
) -> NDArray[np.bool_]:
    """Compute Pareto-optimal points.

    Args:
        objectives: Array of shape (n_points, n_objectives)
        maximize: List of booleans indicating whether to maximize each objective

    Returns:
        Boolean array indicating Pareto-optimal points
    """
    n_points = objectives.shape[0]
    is_pareto = np.ones(n_points, dtype=bool)

    # Flip signs for maximization objectives
    obj = objectives.copy()
    for i, is_max in enumerate(maximize):
        if is_max:
            obj[:, i] = -obj[:, i]

    for i in range(n_points):
        if not is_pareto[i]:
            continue

        # Check if any other point dominates this one
        for j in range(n_points):
            if i == j or not is_pareto[j]:
                continue

            # j dominates i if j is <= in all objectives and < in at least one
            dominates = np.all(obj[j] <= obj[i]) and np.any(obj[j] < obj[i])
            if dominates:
                is_pareto[i] = False
                break

    return is_pareto


@beartype
def plot_pareto(
    results: StudyResults,
    x_metric: str,
    y_metric: str,
    maximize: tuple[bool, bool] = (True, True),
    feasible_only: bool = True,
    figsize: tuple[float, float] = (10, 8),
    title: str | None = None,
    show_dominated: bool = True,
) -> Figure:
    """Plot Pareto front for two objectives.

    Args:
        results: StudyResults from ParametricStudy
        x_metric: First objective (x-axis)
        y_metric: Second objective (y-axis)
        maximize: Tuple of booleans for each objective (True = maximize)
        feasible_only: Only consider feasible points
        figsize: Figure size
        title: Optional title
        show_dominated: Show dominated points in gray

    Returns:
        matplotlib Figure
    """
    _setup_style()

    # Get metrics
    x = results.get_metric(x_metric)
    y = results.get_metric(y_metric)

    # Filter to feasible if requested
    if feasible_only and results.constraints_passed is not None:
        mask = results.constraints_passed
    else:
        mask = np.ones(len(x), dtype=bool)

    # Remove NaN values
    valid = mask & ~np.isnan(x) & ~np.isnan(y)
    x_valid = x[valid]
    y_valid = y[valid]

    if len(x_valid) == 0:
        raise ValueError("No valid data points for Pareto analysis")

    # Compute Pareto front
    objectives = np.column_stack([x_valid, y_valid])
    is_pareto = _compute_pareto_front(objectives, maximize)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot dominated points
    if show_dominated:
        dominated = ~is_pareto
        ax.scatter(
            x_valid[dominated], y_valid[dominated],
            c=COLORS["infeasible"], s=50, alpha=0.5,
            label="Dominated",
        )

    # Plot Pareto front points
    ax.scatter(
        x_valid[is_pareto], y_valid[is_pareto],
        c=COLORS["pareto"], s=100,
        label=f"Pareto Front ({np.sum(is_pareto)} points)",
        zorder=3,
    )

    # Connect Pareto points with line (sorted)
    pareto_x = x_valid[is_pareto]
    pareto_y = y_valid[is_pareto]
    sort_idx = np.argsort(pareto_x)
    ax.plot(
        pareto_x[sort_idx], pareto_y[sort_idx],
        c=COLORS["pareto"], linewidth=2, alpha=0.7,
        linestyle="--",
    )

    # Add direction arrows
    x_dir = "→" if maximize[0] else "←"
    y_dir = "↑" if maximize[1] else "↓"
    ax.annotate(
        f"Better {x_dir}",
        xy=(0.95, 0.02), xycoords="axes fraction",
        ha="right", fontsize=10, color=COLORS["text"],
    )
    ax.annotate(
        f"Better {y_dir}",
        xy=(0.02, 0.95), xycoords="axes fraction",
        ha="left", fontsize=10, color=COLORS["text"], rotation=90,
    )

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(title or f"Pareto Front: {x_metric} vs {y_metric}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig


@beartype
def get_pareto_points(
    results: StudyResults,
    metrics: list[str],
    maximize: list[bool],
    feasible_only: bool = True,
) -> tuple[list[int], NDArray[np.float64]]:
    """Get indices and values of Pareto-optimal points.

    Args:
        results: StudyResults from ParametricStudy
        metrics: List of objective metric names
        maximize: List of booleans for each metric
        feasible_only: Only consider feasible points

    Returns:
        Tuple of (indices in original results, objective values array)
    """
    # Get metrics
    obj_arrays = [results.get_metric(m) for m in metrics]
    objectives = np.column_stack(obj_arrays)

    # Filter to feasible
    if feasible_only and results.constraints_passed is not None:
        mask = results.constraints_passed
    else:
        mask = np.ones(objectives.shape[0], dtype=bool)

    # Remove NaN
    valid = mask & ~np.any(np.isnan(objectives), axis=1)
    valid_indices = np.where(valid)[0]
    objectives_valid = objectives[valid]

    # Compute Pareto front
    is_pareto = _compute_pareto_front(objectives_valid, maximize)

    pareto_indices = valid_indices[is_pareto].tolist()
    pareto_values = objectives_valid[is_pareto]

    return pareto_indices, pareto_values


# =============================================================================
# Uncertainty Visualization
# =============================================================================


@beartype
def plot_histogram(
    results: UncertaintyResults,
    metric: str,
    bins: int = 50,
    show_ci: bool = True,
    ci_level: float = 0.95,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
    feasible_only: bool = False,
) -> Figure:
    """Plot histogram of a metric from uncertainty analysis.

    Args:
        results: UncertaintyResults from UncertaintyAnalysis
        metric: Metric name to plot
        bins: Number of histogram bins
        show_ci: Show confidence interval lines
        ci_level: Confidence level for CI lines
        figsize: Figure size
        title: Optional title
        feasible_only: Only include feasible samples

    Returns:
        matplotlib Figure
    """
    _setup_style()

    values = results.metrics[metric]
    if feasible_only:
        values = values[results.constraints_passed]

    # Remove NaN
    values = values[~np.isnan(values)]

    fig, ax = plt.subplots(figsize=figsize)

    # Histogram
    ax.hist(values, bins=bins, color=COLORS["primary"], alpha=0.7, edgecolor="white")

    # Statistics
    mean = np.mean(values)
    std = np.std(values)

    ax.axvline(mean, color=COLORS["accent"], linewidth=2, label=f"Mean: {mean:.4g}")

    if show_ci:
        ci = results.confidence_interval(metric, ci_level, feasible_only)
        ax.axvline(ci[0], color=COLORS["secondary"], linewidth=1.5, linestyle="--",
                   label=f"{ci_level*100:.0f}% CI: [{ci[0]:.4g}, {ci[1]:.4g}]")
        ax.axvline(ci[1], color=COLORS["secondary"], linewidth=1.5, linestyle="--")

    ax.set_xlabel(metric)
    ax.set_ylabel("Frequency")
    ax.set_title(title or f"Distribution of {metric}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add text box with statistics
    stats_text = f"μ = {mean:.4g}\nσ = {std:.4g}\nn = {len(values)}"
    ax.text(
        0.95, 0.95, stats_text,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    return fig


@beartype
def plot_correlation(
    results: UncertaintyResults,
    x_param: str,
    y_metric: str,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
) -> Figure:
    """Plot correlation between input parameter and output metric.

    Args:
        results: UncertaintyResults from UncertaintyAnalysis
        x_param: Input parameter name (from samples)
        y_metric: Output metric name
        figsize: Figure size
        title: Optional title

    Returns:
        matplotlib Figure
    """
    _setup_style()

    if x_param not in results.samples:
        raise ValueError(f"Parameter '{x_param}' not in samples. Available: {list(results.samples.keys())}")

    x = results.samples[x_param]
    y = results.metrics[y_metric]

    # Remove NaN pairs
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(x, y, c=COLORS["primary"], alpha=0.3, s=20)

    # Compute correlation
    corr = np.corrcoef(x, y)[0, 1]

    # Add trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), c=COLORS["accent"], linewidth=2,
            label=f"Trend (r = {corr:.3f})")

    ax.set_xlabel(x_param)
    ax.set_ylabel(y_metric)
    ax.set_title(title or f"Correlation: {x_param} vs {y_metric}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


@beartype
def plot_tornado(
    results: UncertaintyResults,
    metric: str,
    parameters: list[str] | None = None,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
) -> Figure:
    """Plot tornado chart showing sensitivity of metric to input parameters.

    Shows which parameters have the strongest influence on the metric.

    Args:
        results: UncertaintyResults from UncertaintyAnalysis
        metric: Metric to analyze
        parameters: List of parameters to include (None = all)
        figsize: Figure size
        title: Optional title

    Returns:
        matplotlib Figure
    """
    _setup_style()

    if parameters is None:
        parameters = list(results.samples.keys())

    y_values = results.metrics[metric]
    correlations: dict[str, float] = {}

    for param in parameters:
        x_values = results.samples[param]
        valid = ~(np.isnan(x_values) | np.isnan(y_values))
        if np.sum(valid) > 2:
            corr = np.corrcoef(x_values[valid], y_values[valid])[0, 1]
            correlations[param] = corr

    # Sort by absolute correlation
    sorted_params = sorted(correlations.keys(), key=lambda p: abs(correlations[p]))
    sorted_corrs = [correlations[p] for p in sorted_params]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(sorted_params))
    colors = [COLORS["primary"] if c >= 0 else COLORS["secondary"] for c in sorted_corrs]

    ax.barh(y_pos, sorted_corrs, color=colors, alpha=0.8, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_params)
    ax.set_xlabel("Correlation Coefficient")
    ax.set_title(title or f"Sensitivity of {metric}")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlim(-1, 1)
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    return fig

