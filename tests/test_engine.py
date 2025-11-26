"""Tests for the engine module."""

import pytest

from rocket.engine import (
    EngineGeometry,
    EngineInputs,
    EnginePerformance,
    compute_geometry,
    compute_performance,
    design_engine,
    format_geometry_summary,
    format_performance_summary,
    isp_at_altitude,
    thrust_at_altitude,
)
from rocket.units import (
    kelvin,
    megapascals,
    meters,
    newtons,
    pascals,
)


class TestEngineInputsValidation:
    """Test EngineInputs validation."""

    def test_valid_inputs(self) -> None:
        """Test that valid inputs create an EngineInputs object."""
        inputs = EngineInputs(
            thrust=newtons(5000),
            chamber_pressure=megapascals(2.0),
            chamber_temp=kelvin(3200),
            exit_pressure=pascals(101325),
            molecular_weight=22.0,
            gamma=1.2,
            lstar=meters(1.0),
            mixture_ratio=2.0,
        )
        assert inputs.thrust.value == 5000
        assert inputs.gamma == 1.2

    def test_invalid_thrust_dimension(self) -> None:
        """Test that non-force thrust raises error."""
        with pytest.raises(ValueError, match="thrust must be force"):
            EngineInputs(
                thrust=meters(5000),  # Wrong dimension!
                chamber_pressure=megapascals(2.0),
                chamber_temp=kelvin(3200),
                exit_pressure=pascals(101325),
                molecular_weight=22.0,
                gamma=1.2,
                lstar=meters(1.0),
                mixture_ratio=2.0,
            )

    def test_invalid_pressure_dimension(self) -> None:
        """Test that non-pressure chamber_pressure raises error."""
        with pytest.raises(ValueError, match="chamber_pressure must be pressure"):
            EngineInputs(
                thrust=newtons(5000),
                chamber_pressure=kelvin(2.0),  # Wrong dimension!
                chamber_temp=kelvin(3200),
                exit_pressure=pascals(101325),
                molecular_weight=22.0,
                gamma=1.2,
                lstar=meters(1.0),
                mixture_ratio=2.0,
            )

    def test_invalid_gamma(self) -> None:
        """Test that gamma <= 1 raises error."""
        with pytest.raises(ValueError, match="gamma must be > 1"):
            EngineInputs(
                thrust=newtons(5000),
                chamber_pressure=megapascals(2.0),
                chamber_temp=kelvin(3200),
                exit_pressure=pascals(101325),
                molecular_weight=22.0,
                gamma=0.9,  # Invalid!
                lstar=meters(1.0),
                mixture_ratio=2.0,
            )

    def test_invalid_molecular_weight(self) -> None:
        """Test that negative MW raises error."""
        with pytest.raises(ValueError, match="molecular_weight must be > 0"):
            EngineInputs(
                thrust=newtons(5000),
                chamber_pressure=megapascals(2.0),
                chamber_temp=kelvin(3200),
                exit_pressure=pascals(101325),
                molecular_weight=-10.0,  # Invalid!
                gamma=1.2,
                lstar=meters(1.0),
                mixture_ratio=2.0,
            )

    def test_invalid_mixture_ratio(self) -> None:
        """Test that negative MR raises error."""
        with pytest.raises(ValueError, match="mixture_ratio must be > 0"):
            EngineInputs(
                thrust=newtons(5000),
                chamber_pressure=megapascals(2.0),
                chamber_temp=kelvin(3200),
                exit_pressure=pascals(101325),
                molecular_weight=22.0,
                gamma=1.2,
                lstar=meters(1.0),
                mixture_ratio=-1.0,  # Invalid!
            )

    def test_invalid_contraction_ratio(self) -> None:
        """Test that contraction ratio < 1 raises error."""
        with pytest.raises(ValueError, match="contraction_ratio must be >= 1"):
            EngineInputs(
                thrust=newtons(5000),
                chamber_pressure=megapascals(2.0),
                chamber_temp=kelvin(3200),
                exit_pressure=pascals(101325),
                molecular_weight=22.0,
                gamma=1.2,
                lstar=meters(1.0),
                mixture_ratio=2.0,
                contraction_ratio=0.5,  # Invalid!
            )

    def test_effective_ambient_pressure_default(self) -> None:
        """Test that ambient_pressure defaults to exit_pressure."""
        inputs = EngineInputs(
            thrust=newtons(5000),
            chamber_pressure=megapascals(2.0),
            chamber_temp=kelvin(3200),
            exit_pressure=pascals(101325),
            molecular_weight=22.0,
            gamma=1.2,
            lstar=meters(1.0),
            mixture_ratio=2.0,
        )
        assert inputs.effective_ambient_pressure == inputs.exit_pressure

    def test_effective_ambient_pressure_explicit(self) -> None:
        """Test explicit ambient pressure."""
        inputs = EngineInputs(
            thrust=newtons(5000),
            chamber_pressure=megapascals(2.0),
            chamber_temp=kelvin(3200),
            exit_pressure=pascals(50000),
            ambient_pressure=pascals(101325),
            molecular_weight=22.0,
            gamma=1.2,
            lstar=meters(1.0),
            mixture_ratio=2.0,
        )
        assert inputs.effective_ambient_pressure.value == 101325


class TestComputePerformance:
    """Test performance calculations."""

    @pytest.fixture
    def basic_inputs(self) -> EngineInputs:
        """Create basic engine inputs for testing."""
        return EngineInputs(
            thrust=newtons(5000),
            chamber_pressure=megapascals(2.0),
            chamber_temp=kelvin(3200),
            exit_pressure=pascals(101325),
            molecular_weight=22.0,
            gamma=1.2,
            lstar=meters(1.0),
            mixture_ratio=2.0,
        )

    def test_performance_returns_correct_type(self, basic_inputs: EngineInputs) -> None:
        """Test that compute_performance returns EnginePerformance."""
        perf = compute_performance(basic_inputs)
        assert isinstance(perf, EnginePerformance)

    def test_isp_reasonable_range(self, basic_inputs: EngineInputs) -> None:
        """Test that Isp is in a reasonable range for bipropellant."""
        perf = compute_performance(basic_inputs)
        # Bipropellant Isp typically 250-350s at sea level
        assert 200 < perf.isp.value < 400

    def test_isp_vac_greater_than_sl(self, basic_inputs: EngineInputs) -> None:
        """Test that vacuum Isp is greater than sea level Isp."""
        perf = compute_performance(basic_inputs)
        assert perf.isp_vac.value > perf.isp.value

    def test_cstar_reasonable(self, basic_inputs: EngineInputs) -> None:
        """Test that c* is in reasonable range."""
        perf = compute_performance(basic_inputs)
        # c* typically 1500-2500 m/s
        assert 1000 < perf.cstar.value < 3000

    def test_mdot_positive(self, basic_inputs: EngineInputs) -> None:
        """Test that mass flow rate is positive."""
        perf = compute_performance(basic_inputs)
        assert perf.mdot.value > 0

    def test_mdot_ox_plus_fuel_equals_total(self, basic_inputs: EngineInputs) -> None:
        """Test that mdot_ox + mdot_fuel = mdot."""
        perf = compute_performance(basic_inputs)
        total = perf.mdot_ox.value + perf.mdot_fuel.value
        assert total == pytest.approx(perf.mdot.value)

    def test_mixture_ratio_reflected_in_flow(self, basic_inputs: EngineInputs) -> None:
        """Test that O/F ratio is reflected in mass flows."""
        perf = compute_performance(basic_inputs)
        computed_mr = perf.mdot_ox.value / perf.mdot_fuel.value
        assert computed_mr == pytest.approx(basic_inputs.mixture_ratio, rel=1e-6)

    def test_expansion_ratio_positive(self, basic_inputs: EngineInputs) -> None:
        """Test that expansion ratio is > 1."""
        perf = compute_performance(basic_inputs)
        assert perf.expansion_ratio > 1

    def test_exit_mach_supersonic(self, basic_inputs: EngineInputs) -> None:
        """Test that exit Mach is supersonic."""
        perf = compute_performance(basic_inputs)
        assert perf.exit_mach > 1


class TestComputeGeometry:
    """Test geometry calculations."""

    @pytest.fixture
    def inputs_and_performance(self) -> tuple[EngineInputs, EnginePerformance]:
        """Create inputs and performance for geometry testing."""
        inputs = EngineInputs(
            thrust=newtons(5000),
            chamber_pressure=megapascals(2.0),
            chamber_temp=kelvin(3200),
            exit_pressure=pascals(101325),
            molecular_weight=22.0,
            gamma=1.2,
            lstar=meters(1.0),
            mixture_ratio=2.0,
        )
        perf = compute_performance(inputs)
        return inputs, perf

    def test_geometry_returns_correct_type(
        self, inputs_and_performance: tuple[EngineInputs, EnginePerformance]
    ) -> None:
        """Test that compute_geometry returns EngineGeometry."""
        inputs, perf = inputs_and_performance
        geom = compute_geometry(inputs, perf)
        assert isinstance(geom, EngineGeometry)

    def test_throat_area_positive(
        self, inputs_and_performance: tuple[EngineInputs, EnginePerformance]
    ) -> None:
        """Test that throat area is positive."""
        inputs, perf = inputs_and_performance
        geom = compute_geometry(inputs, perf)
        assert geom.throat_area.value > 0

    def test_exit_larger_than_throat(
        self, inputs_and_performance: tuple[EngineInputs, EnginePerformance]
    ) -> None:
        """Test that exit is larger than throat."""
        inputs, perf = inputs_and_performance
        geom = compute_geometry(inputs, perf)
        assert geom.exit_diameter.value > geom.throat_diameter.value

    def test_chamber_larger_than_throat(
        self, inputs_and_performance: tuple[EngineInputs, EnginePerformance]
    ) -> None:
        """Test that chamber is larger than throat."""
        inputs, perf = inputs_and_performance
        geom = compute_geometry(inputs, perf)
        assert geom.chamber_diameter.value > geom.throat_diameter.value

    def test_contraction_ratio_matches(
        self, inputs_and_performance: tuple[EngineInputs, EnginePerformance]
    ) -> None:
        """Test that contraction ratio matches input."""
        inputs, perf = inputs_and_performance
        geom = compute_geometry(inputs, perf)
        assert geom.contraction_ratio == pytest.approx(inputs.contraction_ratio)

    def test_expansion_ratio_matches(
        self, inputs_and_performance: tuple[EngineInputs, EnginePerformance]
    ) -> None:
        """Test that expansion ratio matches performance."""
        inputs, perf = inputs_and_performance
        geom = compute_geometry(inputs, perf)
        assert geom.expansion_ratio == pytest.approx(perf.expansion_ratio)

    def test_nozzle_length_positive(
        self, inputs_and_performance: tuple[EngineInputs, EnginePerformance]
    ) -> None:
        """Test that nozzle length is positive."""
        inputs, perf = inputs_and_performance
        geom = compute_geometry(inputs, perf)
        assert geom.nozzle_length.value > 0


class TestDesignEngine:
    """Test the convenience design_engine function."""

    def test_design_engine_returns_both(self) -> None:
        """Test that design_engine returns both perf and geometry."""
        inputs = EngineInputs(
            thrust=newtons(5000),
            chamber_pressure=megapascals(2.0),
            chamber_temp=kelvin(3200),
            exit_pressure=pascals(101325),
            molecular_weight=22.0,
            gamma=1.2,
            lstar=meters(1.0),
            mixture_ratio=2.0,
        )

        perf, geom = design_engine(inputs)

        assert isinstance(perf, EnginePerformance)
        assert isinstance(geom, EngineGeometry)


class TestAltitudeAnalysis:
    """Test altitude performance analysis."""

    @pytest.fixture
    def full_design(self) -> tuple[EngineInputs, EnginePerformance, EngineGeometry]:
        """Create complete engine design."""
        inputs = EngineInputs(
            thrust=newtons(5000),
            chamber_pressure=megapascals(2.0),
            chamber_temp=kelvin(3200),
            exit_pressure=pascals(101325),
            molecular_weight=22.0,
            gamma=1.2,
            lstar=meters(1.0),
            mixture_ratio=2.0,
        )
        perf, geom = design_engine(inputs)
        return inputs, perf, geom

    def test_thrust_increases_with_altitude(
        self, full_design: tuple[EngineInputs, EnginePerformance, EngineGeometry]
    ) -> None:
        """Test that thrust increases as ambient pressure decreases."""
        inputs, perf, geom = full_design

        thrust_sl = thrust_at_altitude(inputs, perf, geom, pascals(101325))
        thrust_high = thrust_at_altitude(inputs, perf, geom, pascals(10000))

        assert thrust_high.value > thrust_sl.value

    def test_isp_increases_with_altitude(
        self, full_design: tuple[EngineInputs, EnginePerformance, EngineGeometry]
    ) -> None:
        """Test that Isp increases as ambient pressure decreases."""
        inputs, perf, geom = full_design

        isp_sl = isp_at_altitude(inputs, perf, pascals(101325))
        isp_high = isp_at_altitude(inputs, perf, pascals(10000))

        assert isp_high.value > isp_sl.value


class TestSummaryFormatting:
    """Test summary string formatting."""

    def test_performance_summary_contains_key_values(self) -> None:
        """Test that performance summary contains expected values."""
        inputs = EngineInputs(
            name="Test Engine",
            thrust=newtons(5000),
            chamber_pressure=megapascals(2.0),
            chamber_temp=kelvin(3200),
            exit_pressure=pascals(101325),
            molecular_weight=22.0,
            gamma=1.2,
            lstar=meters(1.0),
            mixture_ratio=2.0,
        )
        perf = compute_performance(inputs)

        summary = format_performance_summary(inputs, perf)

        assert "Test Engine" in summary
        assert "Isp" in summary
        assert "Thrust" in summary

    def test_geometry_summary_contains_key_values(self) -> None:
        """Test that geometry summary contains expected values."""
        inputs = EngineInputs(
            name="Test Engine",
            thrust=newtons(5000),
            chamber_pressure=megapascals(2.0),
            chamber_temp=kelvin(3200),
            exit_pressure=pascals(101325),
            molecular_weight=22.0,
            gamma=1.2,
            lstar=meters(1.0),
            mixture_ratio=2.0,
        )
        perf, geom = design_engine(inputs)

        summary = format_geometry_summary(inputs, geom)
        summary_lower = summary.lower()

        assert "Test Engine" in summary
        assert "throat" in summary_lower
        assert "exit" in summary_lower
        assert "chamber" in summary_lower

