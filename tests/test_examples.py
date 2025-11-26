"""Smoke tests for all example scripts.

These tests verify that examples run without errors.
They don't verify correctness of results, just that the code executes.
"""

import subprocess
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent / "rocket" / "examples"


def run_example(example_name: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run an example script and return the result."""
    script_path = EXAMPLES_DIR / f"{example_name}.py"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=Path(__file__).parent.parent,  # Run from project root
    )

    return result


class TestExamplesSmoke:
    """Smoke tests that verify examples run without crashing."""

    def test_basic_engine_runs(self) -> None:
        """Test that basic_engine.py runs without errors."""
        result = run_example("basic_engine")
        assert result.returncode == 0, f"basic_engine failed:\n{result.stderr}"

    def test_cycle_comparison_runs(self) -> None:
        """Test that cycle_comparison.py runs without errors."""
        result = run_example("cycle_comparison")
        assert result.returncode == 0, f"cycle_comparison failed:\n{result.stderr}"

    def test_optimization_runs(self) -> None:
        """Test that optimization.py runs without errors."""
        result = run_example("optimization")
        assert result.returncode == 0, f"optimization failed:\n{result.stderr}"

    def test_propellant_design_runs(self) -> None:
        """Test that propellant_design.py runs without errors."""
        result = run_example("propellant_design")
        assert result.returncode == 0, f"propellant_design failed:\n{result.stderr}"

    def test_thermal_analysis_runs(self) -> None:
        """Test that thermal_analysis.py runs without errors."""
        result = run_example("thermal_analysis")
        assert result.returncode == 0, f"thermal_analysis failed:\n{result.stderr}"

    def test_trade_study_runs(self) -> None:
        """Test that trade_study.py runs without errors."""
        result = run_example("trade_study")
        assert result.returncode == 0, f"trade_study failed:\n{result.stderr}"

    def test_uncertainty_analysis_runs(self) -> None:
        """Test that uncertainty_analysis.py runs without errors."""
        result = run_example("uncertainty_analysis")
        assert result.returncode == 0, f"uncertainty_analysis failed:\n{result.stderr}"

    def test_vehicle_sizing_runs(self) -> None:
        """Test that vehicle_sizing.py runs without errors."""
        result = run_example("vehicle_sizing")
        assert result.returncode == 0, f"vehicle_sizing failed:\n{result.stderr}"

    def test_thermal_transient_runs(self) -> None:
        """Test that thermal_transient.py runs without errors."""
        result = run_example("thermal_transient", timeout=120)  # Longer timeout for compute
        assert result.returncode == 0, f"thermal_transient failed:\n{result.stderr}"


class TestExamplesOutput:
    """Tests that verify examples produce expected output."""

    def test_basic_engine_produces_output(self) -> None:
        """Test that basic_engine produces expected console output."""
        result = run_example("basic_engine")
        assert "Rocket" in result.stdout
        assert "Specific Impulse" in result.stdout

    def test_cycle_comparison_produces_plots(self) -> None:
        """Test that cycle_comparison mentions generating visualizations."""
        result = run_example("cycle_comparison")
        assert "visualization" in result.stdout.lower() or "plot" in result.stdout.lower() or ".png" in result.stdout

    def test_optimization_reports_pareto(self) -> None:
        """Test that optimization reports Pareto-optimal designs."""
        result = run_example("optimization")
        assert "Pareto" in result.stdout or "pareto" in result.stdout

