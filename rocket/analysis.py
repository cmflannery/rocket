"""Parametric analysis and uncertainty quantification for Rocket.

This module provides general-purpose tools for trade studies, sensitivity
analysis, and uncertainty quantification. The design is introspection-based
to avoid brittleness when dataclass fields change.

Key Design Principles:
- Works with ANY frozen dataclass + computation function
- Uses dataclass introspection to validate parameters (no hardcoding)
- Automatically discovers output metrics from return types
- Unit-aware parameter ranges

Example:
    >>> from rocket import EngineInputs, design_engine
    >>> from rocket.analysis import ParametricStudy, Range
    >>>
    >>> study = ParametricStudy(
    ...     compute=design_engine,
    ...     base=inputs,
    ...     vary={"chamber_pressure": Range(5, 15, n=11, unit="MPa")},
    ... )
    >>> results = study.run()
    >>> results.plot("chamber_pressure", "isp_vac")
"""

import dataclasses
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
import polars as pl
from beartype import beartype
from numpy.typing import NDArray
from tqdm import tqdm

from rocket.units import CONVERSIONS, Quantity

# Type variables for generic analysis
T_Input = TypeVar("T_Input")  # Input dataclass type
T_Output = TypeVar("T_Output")  # Output type (can be dataclass or tuple)


# =============================================================================
# Parameter Range Specifications
# =============================================================================


@beartype
@dataclass(frozen=True, slots=True)
class Range:
    """Specification for a parameter range in a parametric study.

    Supports both dimensionless parameters and Quantity fields with units.

    Examples:
        >>> Range(5, 15, n=11, unit="MPa")  # 5-15 MPa in 11 steps
        >>> Range(2.0, 3.5, n=5)  # Dimensionless parameter
        >>> Range(values=[2.5, 2.7, 3.0, 3.2])  # Explicit values
    """

    start: float | int | None = None
    stop: float | int | None = None
    n: int = 10
    unit: str | None = None
    values: Sequence[float | int] | None = None

    def __post_init__(self) -> None:
        """Validate range specification."""
        if self.values is not None:
            if self.start is not None or self.stop is not None:
                raise ValueError("Cannot specify both values and start/stop")
        else:
            if self.start is None or self.stop is None:
                raise ValueError("Must specify either values or start/stop")

    def generate(self) -> NDArray[np.float64]:
        """Generate array of parameter values."""
        if self.values is not None:
            return np.array(self.values, dtype=np.float64)
        return np.linspace(self.start, self.stop, self.n)

    def to_quantities(self, dimension: str) -> list[Quantity]:
        """Convert range values to Quantity objects.

        Args:
            dimension: The dimension of the quantity (e.g., "pressure")

        Returns:
            List of Quantity objects
        """
        values = self.generate()
        if self.unit is None:
            raise ValueError(f"Unit required to convert to Quantity for dimension {dimension}")

        return [Quantity(float(v), self.unit, dimension) for v in values]


@beartype
@dataclass(frozen=True, slots=True)
class Distribution:
    """Base class for probability distributions in uncertainty analysis."""

    pass


@beartype
@dataclass(frozen=True, slots=True)
class Normal(Distribution):
    """Normal (Gaussian) distribution.

    Args:
        mean: Distribution mean
        std: Standard deviation
        unit: Optional unit for Quantity fields
    """

    mean: float | int
    std: float | int
    unit: str | None = None

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """Generate n samples from the distribution."""
        return rng.normal(self.mean, self.std, n)


@beartype
@dataclass(frozen=True, slots=True)
class Uniform(Distribution):
    """Uniform distribution.

    Args:
        low: Lower bound
        high: Upper bound
        unit: Optional unit for Quantity fields
    """

    low: float | int
    high: float | int
    unit: str | None = None

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """Generate n samples from the distribution."""
        return rng.uniform(self.low, self.high, n)


@beartype
@dataclass(frozen=True, slots=True)
class Triangular(Distribution):
    """Triangular distribution.

    Args:
        low: Lower bound
        mode: Most likely value
        high: Upper bound
        unit: Optional unit for Quantity fields
    """

    low: float | int
    mode: float | int
    high: float | int
    unit: str | None = None

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """Generate n samples from the distribution."""
        return rng.triangular(self.low, self.mode, self.high, n)


@beartype
@dataclass(frozen=True, slots=True)
class LogNormal(Distribution):
    """Log-normal distribution.

    Args:
        mean: Mean of the underlying normal distribution
        sigma: Standard deviation of the underlying normal distribution
        unit: Optional unit for Quantity fields
    """

    mean: float | int
    sigma: float | int
    unit: str | None = None

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """Generate n samples from the distribution."""
        return rng.lognormal(self.mean, self.sigma, n)


# =============================================================================
# Introspection Utilities
# =============================================================================


def _get_dataclass_fields(obj: Any) -> dict[str, dataclasses.Field]:
    """Get all fields from a dataclass, including nested ones."""
    if not is_dataclass(obj):
        raise TypeError(f"Expected dataclass, got {type(obj).__name__}")
    return {f.name: f for f in fields(obj)}


def _get_field_info(base: Any, field_name: str) -> tuple[Any, str | None]:
    """Get the current value and dimension (if Quantity) of a field.

    Args:
        base: The base dataclass instance
        field_name: Name of the field to inspect

    Returns:
        Tuple of (current_value, dimension_or_none)
    """
    if not hasattr(base, field_name):
        raise ValueError(f"Field '{field_name}' not found in {type(base).__name__}")

    value = getattr(base, field_name)

    if isinstance(value, Quantity):
        return value, value.dimension
    return value, None


def _create_modified_input(
    base: T_Input,
    field_name: str,
    value: float | Quantity,
    original_dimension: str | None,
) -> T_Input:
    """Create a modified copy of the input with one field changed.

    Handles both Quantity and plain numeric fields.
    """
    current_value = getattr(base, field_name)

    if isinstance(current_value, Quantity):
        # Field is a Quantity - ensure we create a proper Quantity
        if isinstance(value, Quantity):
            new_value = value
        else:
            # Value is numeric, need unit from original
            new_value = Quantity(float(value), current_value.unit, current_value.dimension)
    else:
        # Field is a plain numeric type
        new_value = value

    return replace(base, **{field_name: new_value})


def _extract_metrics(result: Any, prefix: str = "") -> dict[str, float]:
    """Recursively extract all numeric values from a result.

    Handles dataclasses, tuples, and nested structures.
    Returns flat dict with keys for nested values.

    For tuples of dataclasses (common pattern like (Performance, Geometry)),
    fields are extracted without prefixes to keep names clean.
    """
    metrics: dict[str, float] = {}

    if isinstance(result, tuple):
        # Handle tuple of results (e.g., (performance, geometry))
        # Extract fields directly without prefixes for cleaner column names
        for item in result:
            metrics.update(_extract_metrics(item, prefix))

    elif is_dataclass(result) and not isinstance(result, type):
        # Handle dataclass
        for field in fields(result):
            field_value = getattr(result, field.name)
            field_key = f"{prefix}{field.name}" if prefix else field.name

            if isinstance(field_value, Quantity):
                metrics[field_key] = float(field_value.value)
            elif isinstance(field_value, (int, float)):
                metrics[field_key] = float(field_value)
            elif is_dataclass(field_value):
                metrics.update(_extract_metrics(field_value, f"{field_key}."))

    return metrics


# =============================================================================
# Study Results
# =============================================================================


@beartype
@dataclass
class StudyResults:
    """Results from a parametric study or uncertainty analysis.

    Contains all input combinations, computed outputs, and extracted metrics.
    Provides methods for plotting, filtering, and export.

    Attributes:
        inputs: List of input parameter combinations
        outputs: List of computed results
        metrics: Dict mapping metric names to arrays of values
        parameters: Dict mapping parameter names to arrays of values
        constraints_passed: Boolean array indicating which runs passed constraints
    """

    inputs: list[Any]
    outputs: list[Any]
    metrics: dict[str, NDArray[np.float64]]
    parameters: dict[str, NDArray[np.float64]]
    constraints_passed: NDArray[np.bool_] | None = None

    @property
    def n_runs(self) -> int:
        """Number of runs in the study."""
        return len(self.inputs)

    @property
    def n_feasible(self) -> int:
        """Number of runs that passed all constraints."""
        if self.constraints_passed is None:
            return self.n_runs
        return int(np.sum(self.constraints_passed))

    def get_metric(self, name: str, feasible_only: bool = False) -> NDArray[np.float64]:
        """Get values for a specific metric.

        Args:
            name: Metric name (e.g., "isp", "throat_diameter")
            feasible_only: If True, only return values where constraints passed

        Returns:
            Array of metric values
        """
        if name not in self.metrics:
            available = list(self.metrics.keys())
            raise ValueError(f"Unknown metric '{name}'. Available: {available}")

        values = self.metrics[name]
        if feasible_only and self.constraints_passed is not None:
            return values[self.constraints_passed]
        return values

    def get_parameter(self, name: str, feasible_only: bool = False) -> NDArray[np.float64]:
        """Get values for a specific input parameter.

        Args:
            name: Parameter name (e.g., "chamber_pressure")
            feasible_only: If True, only return values where constraints passed

        Returns:
            Array of parameter values
        """
        if name not in self.parameters:
            available = list(self.parameters.keys())
            raise ValueError(f"Unknown parameter '{name}'. Available: {available}")

        values = self.parameters[name]
        if feasible_only and self.constraints_passed is not None:
            return values[self.constraints_passed]
        return values

    def get_best(
        self,
        metric: str,
        maximize: bool = True,
        feasible_only: bool = True,
    ) -> tuple[Any, Any, float]:
        """Get the best run according to a metric.

        Args:
            metric: Metric to optimize
            maximize: If True, find maximum; if False, find minimum
            feasible_only: Only consider runs that passed constraints

        Returns:
            Tuple of (best_input, best_output, best_metric_value)
        """
        values = self.get_metric(metric, feasible_only=False)
        mask = self.constraints_passed if feasible_only and self.constraints_passed is not None else np.ones(len(values), dtype=bool)

        if not np.any(mask):
            raise ValueError("No feasible solutions found")

        masked_values = np.where(mask, values, -np.inf if maximize else np.inf)
        best_idx = int(np.argmax(masked_values) if maximize else np.argmin(masked_values))

        return self.inputs[best_idx], self.outputs[best_idx], float(values[best_idx])

    def to_dataframe(self) -> pl.DataFrame:
        """Export results to a Polars DataFrame.

        Returns:
            Polars DataFrame with parameters and metrics
        """
        data = {**self.parameters, **self.metrics}
        if self.constraints_passed is not None:
            data["feasible"] = self.constraints_passed
        return pl.DataFrame(data)

    def to_csv(self, path: str | Path) -> None:
        """Export results to CSV file.

        Args:
            path: Output file path
        """
        df = self.to_dataframe()
        df.write_csv(path)

    def list_metrics(self) -> list[str]:
        """List all available metric names."""
        return list(self.metrics.keys())

    def list_parameters(self) -> list[str]:
        """List all varied parameter names."""
        return list(self.parameters.keys())


# =============================================================================
# Parametric Study
# =============================================================================


@beartype
class ParametricStudy(Generic[T_Input, T_Output]):
    """General-purpose parametric study framework.

    Runs a computation over a grid of parameter variations, automatically
    discovering valid parameters through dataclass introspection.

    This design is non-brittle:
    - Adding new fields to input dataclasses automatically makes them available
    - No hardcoded parameter names
    - Works with any frozen dataclass + computation function

    Example:
        >>> study = ParametricStudy(
        ...     compute=design_engine,
        ...     base=inputs,
        ...     vary={
        ...         "chamber_pressure": Range(5, 15, n=11, unit="MPa"),
        ...         "mixture_ratio": Range(2.5, 3.5, n=5),
        ...     },
        ... )
        >>> results = study.run()
    """

    def __init__(
        self,
        compute: Callable[[T_Input], T_Output],
        base: T_Input,
        vary: dict[str, Range | Sequence[Any]],
        constraints: list[Callable[[T_Output], bool]] | None = None,
    ) -> None:
        """Initialize parametric study.

        Args:
            compute: Function that takes input and returns output
            base: Base input dataclass with default values
            vary: Dict mapping field names to Range specifications or plain sequences.
                  Use Range for unit-aware sweeps, or a plain list for discrete values.
            constraints: Optional list of constraint functions.
                         Each takes output and returns True if feasible.
        """
        self.compute = compute
        self.base = base
        self.vary = vary
        self.constraints = constraints or []

        # Validate that all varied parameters exist in the base dataclass
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate that all varied parameters exist and have compatible types."""
        valid_fields = _get_dataclass_fields(self.base)

        for param_name, param_spec in self.vary.items():
            if param_name not in valid_fields:
                raise ValueError(
                    f"Parameter '{param_name}' not found in {type(self.base).__name__}. "
                    f"Valid fields: {list(valid_fields.keys())}"
                )

            # Check unit compatibility for Quantity fields (only if Range is used)
            current_value = getattr(self.base, param_name)
            if isinstance(current_value, Quantity) and isinstance(param_spec, Range):
                if param_spec.unit is None:
                    raise ValueError(
                        f"Parameter '{param_name}' is a Quantity, but no unit specified in Range. "
                        f"Current unit: {current_value.unit}"
                    )
                # Verify unit is valid and has compatible dimension
                if param_spec.unit not in CONVERSIONS:
                    raise ValueError(f"Unknown unit '{param_spec.unit}' for parameter '{param_name}'")

                range_dim = CONVERSIONS[param_spec.unit][1]
                if range_dim != current_value.dimension:
                    raise ValueError(
                        f"Unit '{param_spec.unit}' has dimension '{range_dim}', "
                        f"but field '{param_name}' has dimension '{current_value.dimension}'"
                    )

    def _generate_grid(self) -> list[dict[str, float | Quantity]]:
        """Generate all parameter combinations."""
        # Generate values for each parameter
        param_values: dict[str, list[Any]] = {}

        for param_name, param_spec in self.vary.items():
            current_value = getattr(self.base, param_name)

            if isinstance(param_spec, Range):
                # Range specification
                if isinstance(current_value, Quantity):
                    # Generate Quantity values
                    param_values[param_name] = param_spec.to_quantities(current_value.dimension)
                else:
                    # Generate plain numeric values
                    param_values[param_name] = list(param_spec.generate())
            else:
                # Plain sequence - use values as-is
                param_values[param_name] = list(param_spec)

        # Generate all combinations
        keys = list(param_values.keys())
        value_lists = [param_values[k] for k in keys]
        combinations = list(itertools.product(*value_lists))

        return [dict(zip(keys, combo, strict=True)) for combo in combinations]

    def run(self, progress: bool = False) -> StudyResults:
        """Run the parametric study.

        Args:
            progress: If True, print progress (requires tqdm for fancy progress bar)

        Returns:
            StudyResults containing all inputs, outputs, and extracted metrics
        """
        grid = self._generate_grid()
        n_total = len(grid)

        inputs_list: list[T_Input] = []
        outputs_list: list[T_Output] = []
        all_metrics: list[dict[str, float]] = []
        all_params: list[dict[str, float]] = []
        constraints_passed: list[bool] = []

        # Create iterator with optional progress bar
        iterator: Any = grid
        if progress:
            iterator = tqdm(grid, desc="Running study", total=n_total)

        for i, param_combo in enumerate(iterator):
            # Create modified input
            modified_input = self.base
            for param_name, param_value in param_combo.items():
                modified_input = _create_modified_input(
                    modified_input,
                    param_name,
                    param_value,
                    current_value.dimension if isinstance((current_value := getattr(self.base, param_name)), Quantity) else None,
                )

            # Run computation
            try:
                output = self.compute(modified_input)
                success = True
            except Exception as e:
                # Store None for failed runs
                output = None  # type: ignore
                success = False
                if progress and not hasattr(iterator, 'set_postfix'):
                    print(f"  Run {i+1}/{n_total} failed: {e}")

            inputs_list.append(modified_input)
            outputs_list.append(output)

            # Extract metrics
            if success and output is not None:
                metrics = _extract_metrics(output)
                all_metrics.append(metrics)

                # Check constraints
                passed = all(constraint(output) for constraint in self.constraints)
            else:
                all_metrics.append({})
                passed = False

            constraints_passed.append(passed)

            # Extract parameter values (numeric form for plotting)
            param_dict: dict[str, float] = {}
            for param_name, param_value in param_combo.items():
                if isinstance(param_value, Quantity):
                    param_dict[param_name] = float(param_value.value)
                else:
                    param_dict[param_name] = float(param_value)
            all_params.append(param_dict)

        # Consolidate metrics into arrays
        if all_metrics and all_metrics[0]:
            metric_names = set()
            for m in all_metrics:
                metric_names.update(m.keys())

            metrics_arrays = {
                name: np.array([m.get(name, np.nan) for m in all_metrics])
                for name in metric_names
            }
        else:
            metrics_arrays = {}

        # Consolidate parameters into arrays
        if all_params:
            param_names = list(all_params[0].keys())
            params_arrays = {
                name: np.array([p[name] for p in all_params])
                for name in param_names
            }
        else:
            params_arrays = {}

        return StudyResults(
            inputs=inputs_list,
            outputs=outputs_list,
            metrics=metrics_arrays,
            parameters=params_arrays,
            constraints_passed=np.array(constraints_passed),
        )


# =============================================================================
# Uncertainty Analysis
# =============================================================================


@beartype
class UncertaintyAnalysis(Generic[T_Input, T_Output]):
    """Monte Carlo uncertainty quantification.

    Samples input parameters from specified distributions and propagates
    uncertainty through the computation.

    Example:
        >>> analysis = UncertaintyAnalysis(
        ...     compute=design_engine,
        ...     base=inputs,
        ...     distributions={
        ...         "gamma": Normal(1.22, 0.02),
        ...         "chamber_temp": Normal(3200, 50, unit="K"),
        ...     },
        ... )
        >>> results = analysis.run(n_samples=1000)
        >>> print(f"Isp = {results.mean('isp'):.1f} ± {results.std('isp'):.1f} s")
    """

    def __init__(
        self,
        compute: Callable[[T_Input], T_Output],
        base: T_Input,
        distributions: dict[str, Distribution],
        constraints: list[Callable[[T_Output], bool]] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize uncertainty analysis.

        Args:
            compute: Function that takes input and returns output
            base: Base input dataclass with nominal values
            distributions: Dict mapping field names to Distribution specifications
            constraints: Optional constraint functions
            seed: Random seed for reproducibility
        """
        self.compute = compute
        self.base = base
        self.distributions = distributions
        self.constraints = constraints or []
        self.rng = np.random.default_rng(seed)

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate that all uncertain parameters exist."""
        valid_fields = _get_dataclass_fields(self.base)

        for param_name, dist in self.distributions.items():
            if param_name not in valid_fields:
                raise ValueError(
                    f"Parameter '{param_name}' not found in {type(self.base).__name__}. "
                    f"Valid fields: {list(valid_fields.keys())}"
                )

            # Check unit compatibility for Quantity fields
            current_value = getattr(self.base, param_name)
            if isinstance(current_value, Quantity) and (not hasattr(dist, 'unit') or dist.unit is None):  # type: ignore
                raise ValueError(
                    f"Parameter '{param_name}' is a Quantity, but no unit specified in Distribution"
                )

    def run(self, n_samples: int = 1000, progress: bool = False) -> "UncertaintyResults":
        """Run Monte Carlo uncertainty analysis.

        Args:
            n_samples: Number of Monte Carlo samples
            progress: If True, show progress indicator

        Returns:
            UncertaintyResults with statistics and samples
        """
        # Generate all samples upfront
        samples: dict[str, NDArray[np.float64]] = {}
        for param_name, dist in self.distributions.items():
            samples[param_name] = dist.sample(n_samples, self.rng)

        inputs_list: list[T_Input] = []
        outputs_list: list[T_Output] = []
        all_metrics: list[dict[str, float]] = []
        constraints_passed: list[bool] = []

        iterator: Any = range(n_samples)
        if progress:
            iterator = tqdm(range(n_samples), desc="Sampling")

        for i in iterator:
            # Create modified input with sampled values
            modified_input = self.base

            for param_name, dist in self.distributions.items():
                sampled_value = samples[param_name][i]
                current_value = getattr(self.base, param_name)

                if isinstance(current_value, Quantity):
                    # Create Quantity with sampled value
                    unit = dist.unit  # type: ignore
                    new_value = Quantity(float(sampled_value), unit, current_value.dimension)
                else:
                    new_value = float(sampled_value)

                modified_input = replace(modified_input, **{param_name: new_value})

            # Run computation
            try:
                output = self.compute(modified_input)
                success = True
            except Exception:
                output = None  # type: ignore
                success = False

            inputs_list.append(modified_input)
            outputs_list.append(output)

            if success and output is not None:
                metrics = _extract_metrics(output)
                all_metrics.append(metrics)
                passed = all(constraint(output) for constraint in self.constraints)
            else:
                all_metrics.append({})
                passed = False

            constraints_passed.append(passed)

        # Consolidate metrics
        if all_metrics and all_metrics[0]:
            metric_names = set()
            for m in all_metrics:
                metric_names.update(m.keys())

            metrics_arrays = {
                name: np.array([m.get(name, np.nan) for m in all_metrics])
                for name in metric_names
            }
        else:
            metrics_arrays = {}

        return UncertaintyResults(
            inputs=inputs_list,
            outputs=outputs_list,
            metrics=metrics_arrays,
            samples=samples,
            constraints_passed=np.array(constraints_passed),
            n_samples=n_samples,
        )


@beartype
@dataclass
class UncertaintyResults:
    """Results from uncertainty analysis.

    Provides statistical summaries and access to all samples.
    """

    inputs: list[Any]
    outputs: list[Any]
    metrics: dict[str, NDArray[np.float64]]
    samples: dict[str, NDArray[np.float64]]
    constraints_passed: NDArray[np.bool_]
    n_samples: int

    def mean(self, metric: str, feasible_only: bool = False) -> float:
        """Get mean value of a metric."""
        values = self._get_values(metric, feasible_only)
        return float(np.nanmean(values))

    def std(self, metric: str, feasible_only: bool = False) -> float:
        """Get standard deviation of a metric."""
        values = self._get_values(metric, feasible_only)
        return float(np.nanstd(values))

    def percentile(
        self, metric: str, p: float | Sequence[float], feasible_only: bool = False
    ) -> float | NDArray[np.float64]:
        """Get percentile(s) of a metric.

        Args:
            metric: Metric name
            p: Percentile(s) to compute (0-100)
            feasible_only: Only use feasible samples

        Returns:
            Percentile value(s)
        """
        values = self._get_values(metric, feasible_only)
        result = np.nanpercentile(values, p)
        if isinstance(p, (int, float)):
            return float(result)
        return result

    def confidence_interval(
        self, metric: str, confidence: float = 0.95, feasible_only: bool = False
    ) -> tuple[float, float]:
        """Get confidence interval for a metric.

        Args:
            metric: Metric name
            confidence: Confidence level (0-1), default 0.95 for 95% CI
            feasible_only: Only use feasible samples

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        alpha = 1 - confidence
        lower_p = alpha / 2 * 100
        upper_p = (1 - alpha / 2) * 100

        values = self._get_values(metric, feasible_only)
        return (
            float(np.nanpercentile(values, lower_p)),
            float(np.nanpercentile(values, upper_p)),
        )

    def probability_of_success(self) -> float:
        """Get fraction of samples that passed all constraints."""
        return float(np.mean(self.constraints_passed))

    def _get_values(self, metric: str, feasible_only: bool) -> NDArray[np.float64]:
        """Get metric values, optionally filtered to feasible only."""
        if metric not in self.metrics:
            available = list(self.metrics.keys())
            raise ValueError(f"Unknown metric '{metric}'. Available: {available}")

        values = self.metrics[metric]
        if feasible_only:
            values = values[self.constraints_passed]
        return values

    def summary(self, metrics: list[str] | None = None) -> str:
        """Generate a text summary of uncertainty results.

        Args:
            metrics: List of metrics to summarize. If None, summarizes all.

        Returns:
            Formatted string summary
        """
        if metrics is None:
            metrics = [m for m in self.metrics if not m.endswith("_si")]

        lines = [
            "Uncertainty Analysis Results",
            "=" * 50,
            f"Samples: {self.n_samples}",
            f"Feasible: {np.sum(self.constraints_passed)} ({self.probability_of_success()*100:.1f}%)",
            "",
            f"{'Metric':<25} {'Mean':>12} {'Std':>12} {'95% CI':>20}",
            "-" * 70,
        ]

        for metric in metrics:
            if metric in self.metrics:
                mean = self.mean(metric)
                std = self.std(metric)
                ci = self.confidence_interval(metric)
                lines.append(
                    f"{metric:<25} {mean:>12.4g} {std:>12.4g} [{ci[0]:.4g}, {ci[1]:.4g}]"
                )

        return "\n".join(lines)

    def to_dataframe(self) -> pl.DataFrame:
        """Export results to Polars DataFrame."""
        data = {**self.samples, **self.metrics, "feasible": self.constraints_passed}
        return pl.DataFrame(data)

    def to_csv(self, path: str | Path) -> None:
        """Export results to CSV file.

        Args:
            path: Output file path
        """
        df = self.to_dataframe()
        df.write_csv(path)


# =============================================================================
# Multi-Objective Optimization
# =============================================================================


@beartype
def compute_pareto_front(
    objectives: NDArray[np.float64],
    maximize: Sequence[bool],
) -> NDArray[np.bool_]:
    """Identify Pareto-optimal points in a set of objectives.

    A point is Pareto-optimal if no other point dominates it (i.e., no
    other point is better in all objectives simultaneously).

    Args:
        objectives: Array of shape (n_points, n_objectives)
        maximize: List of booleans indicating whether to maximize each objective

    Returns:
        Boolean array indicating which points are Pareto-optimal
    """
    n_points = objectives.shape[0]
    is_pareto = np.ones(n_points, dtype=bool)

    # Flip signs for maximization (we'll minimize internally)
    obj = objectives.copy()
    for i, is_max in enumerate(maximize):
        if is_max:
            obj[:, i] = -obj[:, i]

    for i in range(n_points):
        if not is_pareto[i]:
            continue

        for j in range(n_points):
            if i == j or not is_pareto[j]:
                continue

            # j dominates i if j is <= in all objectives and < in at least one
            if np.all(obj[j] <= obj[i]) and np.any(obj[j] < obj[i]):
                is_pareto[i] = False
                break

    return is_pareto


@beartype
class MultiObjectiveOptimizer(Generic[T_Input, T_Output]):
    """Multi-objective optimizer for finding Pareto-optimal designs.

    Uses a combination of grid search and local refinement to find
    designs on the Pareto frontier.

    Example:
        >>> optimizer = MultiObjectiveOptimizer(
        ...     compute=design_engine,
        ...     base=inputs,
        ...     objectives=["isp_vac", "thrust_to_weight"],
        ...     maximize=[True, True],
        ...     vary={"chamber_pressure": Range(5, 20, n=10, unit="MPa")},
        ... )
        >>> pareto_results = optimizer.run()
    """

    def __init__(
        self,
        compute: Callable[[T_Input], T_Output],
        base: T_Input,
        objectives: list[str],
        maximize: list[bool],
        vary: dict[str, Range | Sequence[Any]],
        constraints: list[Callable[[T_Output], bool]] | None = None,
    ) -> None:
        """Initialize multi-objective optimizer.

        Args:
            compute: Function that takes input and returns output
            base: Base input dataclass
            objectives: List of metric names to optimize
            maximize: List of booleans for each objective (True = maximize)
            vary: Dict mapping field names to Range specifications or plain sequences
            constraints: Optional constraint functions
        """
        self.compute = compute
        self.base = base
        self.objectives = objectives
        self.maximize = maximize
        self.vary = vary
        self.constraints = constraints or []

        if len(objectives) != len(maximize):
            raise ValueError("objectives and maximize must have same length")

    def run(self, progress: bool = False) -> "ParetoResults":
        """Run the multi-objective optimization.

        First performs a parametric sweep, then identifies Pareto-optimal points.

        Args:
            progress: If True, show progress indicator

        Returns:
            ParetoResults with Pareto-optimal designs
        """
        # Run parametric study
        study = ParametricStudy(
            compute=self.compute,
            base=self.base,
            vary=self.vary,
            constraints=self.constraints,
        )
        results = study.run(progress=progress)

        # Extract objectives
        obj_arrays = []
        for obj_name in self.objectives:
            if obj_name not in results.metrics:
                raise ValueError(f"Objective '{obj_name}' not found in results")
            obj_arrays.append(results.get_metric(obj_name))

        objectives_matrix = np.column_stack(obj_arrays)

        # Filter to feasible points
        if results.constraints_passed is not None:
            feasible_mask = results.constraints_passed
        else:
            feasible_mask = np.ones(len(objectives_matrix), dtype=bool)

        # Remove NaN values
        valid_mask = feasible_mask & ~np.any(np.isnan(objectives_matrix), axis=1)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return ParetoResults(
                all_results=results,
                pareto_indices=[],
                pareto_inputs=[],
                pareto_outputs=[],
                pareto_objectives=np.array([]).reshape(0, len(self.objectives)),
                objective_names=self.objectives,
                maximize=self.maximize,
            )

        # Compute Pareto front on valid points only
        valid_objectives = objectives_matrix[valid_mask]
        is_pareto = compute_pareto_front(valid_objectives, self.maximize)

        # Map back to original indices
        pareto_indices = valid_indices[is_pareto].tolist()
        pareto_inputs = [results.inputs[i] for i in pareto_indices]
        pareto_outputs = [results.outputs[i] for i in pareto_indices]
        pareto_objectives = valid_objectives[is_pareto]

        return ParetoResults(
            all_results=results,
            pareto_indices=pareto_indices,
            pareto_inputs=pareto_inputs,
            pareto_outputs=pareto_outputs,
            pareto_objectives=pareto_objectives,
            objective_names=self.objectives,
            maximize=self.maximize,
        )


@beartype
@dataclass
class ParetoResults:
    """Results from multi-objective optimization.

    Contains the Pareto-optimal designs and full study results.
    """

    all_results: StudyResults
    pareto_indices: list[int]
    pareto_inputs: list[Any]
    pareto_outputs: list[Any]
    pareto_objectives: NDArray[np.float64]
    objective_names: list[str]
    maximize: list[bool]

    @property
    def n_pareto(self) -> int:
        """Number of Pareto-optimal points."""
        return len(self.pareto_indices)

    def get_best(self, objective: str) -> tuple[Any, Any, float]:
        """Get the best design for a specific objective.

        Args:
            objective: Name of objective to optimize

        Returns:
            Tuple of (input, output, objective_value)
        """
        if objective not in self.objective_names:
            raise ValueError(f"Unknown objective: {objective}")

        obj_idx = self.objective_names.index(objective)
        values = self.pareto_objectives[:, obj_idx]

        if self.maximize[obj_idx]:
            best_idx = int(np.argmax(values))
        else:
            best_idx = int(np.argmin(values))

        return (
            self.pareto_inputs[best_idx],
            self.pareto_outputs[best_idx],
            float(values[best_idx]),
        )

    def get_compromise(self, weights: list[float] | None = None) -> tuple[Any, Any]:
        """Get a compromise solution from the Pareto front.

        Uses weighted sum of normalized objectives.

        Args:
            weights: Weights for each objective (default: equal weights)

        Returns:
            Tuple of (input, output) for the compromise solution
        """
        if weights is None:
            weights = [1.0 / len(self.objectives) for _ in self.objective_names]

        if len(weights) != len(self.objective_names):
            raise ValueError("weights must have same length as objectives")

        # Normalize objectives to [0, 1]
        obj_norm = self.pareto_objectives.copy()
        for i in range(obj_norm.shape[1]):
            col = obj_norm[:, i]
            col_min, col_max = np.min(col), np.max(col)
            if col_max > col_min:
                obj_norm[:, i] = (col - col_min) / (col_max - col_min)
            else:
                obj_norm[:, i] = 0.5

            # Flip if minimizing
            if not self.maximize[i]:
                obj_norm[:, i] = 1 - obj_norm[:, i]

        # Weighted sum
        scores = np.sum(obj_norm * np.array(weights), axis=1)
        best_idx = int(np.argmax(scores))

        return self.pareto_inputs[best_idx], self.pareto_outputs[best_idx]

    def summary(self) -> str:
        """Generate text summary of Pareto results."""
        lines = [
            "Multi-Objective Optimization Results",
            "=" * 50,
            f"Total designs evaluated: {self.all_results.n_runs}",
            f"Feasible designs: {self.all_results.n_feasible}",
            f"Pareto-optimal designs: {self.n_pareto}",
            "",
            "Objectives:",
        ]

        for i, (name, is_max) in enumerate(zip(self.objective_names, self.maximize, strict=True)):
            direction = "maximize" if is_max else "minimize"
            if self.n_pareto > 0:
                values = self.pareto_objectives[:, i]
                lines.append(
                    f"  {name} ({direction}): "
                    f"range [{np.min(values):.4g}, {np.max(values):.4g}]"
                )
            else:
                lines.append(f"  {name} ({direction}): no feasible points")

        return "\n".join(lines)

