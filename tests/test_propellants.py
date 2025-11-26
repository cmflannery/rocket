"""Tests for the propellants module."""

import pytest

from openrocketengine.propellants import (
    CombustionProperties,
    get_combustion_properties,
    is_cea_available,
    list_database_propellants,
)


class TestCombustionProperties:
    """Test CombustionProperties dataclass."""

    def test_create_properties(self) -> None:
        """Test creating combustion properties."""
        props = CombustionProperties(
            chamber_temp_k=3500.0,
            molecular_weight=22.0,
            gamma=1.2,
            specific_heat_cp=2000.0,
            characteristic_velocity=1800.0,
            oxidizer="LOX",
            fuel="RP1",
            mixture_ratio=2.7,
            chamber_pressure_pa=7e6,
            source="test",
        )
        assert props.chamber_temp_k == 3500.0
        assert props.gamma == 1.2
        assert props.source == "test"


class TestDatabasePropellants:
    """Test propellant database functionality."""

    def test_list_database_propellants(self) -> None:
        """Test listing available propellants."""
        propellants = list_database_propellants()

        assert len(propellants) > 0
        assert ("LOX", "RP1") in propellants
        assert ("LOX", "LH2") in propellants
        assert ("LOX", "CH4") in propellants

    def test_get_lox_rp1_properties(self) -> None:
        """Test getting LOX/RP1 properties from database."""
        props = get_combustion_properties(
            oxidizer="LOX",
            fuel="RP1",
            mixture_ratio=2.7,
            chamber_pressure_pa=7e6,
            use_cea=False,  # Force database
        )

        # Verify reasonable values for LOX/RP1
        assert 3400 < props.chamber_temp_k < 3800
        assert 20 < props.molecular_weight < 26
        assert 1.1 < props.gamma < 1.25
        assert 1700 < props.characteristic_velocity < 1900
        assert props.source == "database"

    def test_get_lox_lh2_properties(self) -> None:
        """Test getting LOX/LH2 properties from database."""
        props = get_combustion_properties(
            oxidizer="LOX",
            fuel="LH2",
            mixture_ratio=6.0,
            chamber_pressure_pa=10e6,
            use_cea=False,
        )

        # LOX/LH2 has higher Isp, lower MW
        assert 3300 < props.chamber_temp_k < 3600
        assert 12 < props.molecular_weight < 18
        assert 1.1 < props.gamma < 1.2
        assert 2300 < props.characteristic_velocity < 2500

    def test_get_lox_ch4_properties(self) -> None:
        """Test getting LOX/CH4 properties from database."""
        props = get_combustion_properties(
            oxidizer="LOX",
            fuel="CH4",
            mixture_ratio=3.2,
            chamber_pressure_pa=7e6,
            use_cea=False,
        )

        assert 3400 < props.chamber_temp_k < 3700
        assert 18 < props.molecular_weight < 24
        assert props.source == "database"

    def test_get_n2o4_mmh_properties(self) -> None:
        """Test getting N2O4/MMH (hypergolic) properties."""
        props = get_combustion_properties(
            oxidizer="N2O4",
            fuel="MMH",
            mixture_ratio=2.0,
            chamber_pressure_pa=1e6,
            use_cea=False,
        )

        assert 3000 < props.chamber_temp_k < 3400
        assert 20 < props.molecular_weight < 25

    def test_interpolation(self) -> None:
        """Test that intermediate mixture ratios are interpolated."""
        props_low = get_combustion_properties(
            oxidizer="LOX", fuel="RP1", mixture_ratio=2.0, chamber_pressure_pa=7e6, use_cea=False
        )
        props_mid = get_combustion_properties(
            oxidizer="LOX", fuel="RP1", mixture_ratio=2.5, chamber_pressure_pa=7e6, use_cea=False
        )
        props_high = get_combustion_properties(
            oxidizer="LOX", fuel="RP1", mixture_ratio=3.0, chamber_pressure_pa=7e6, use_cea=False
        )

        # Mid should be between low and high (roughly)
        # Note: Tc may not be monotonic, but MW usually is
        assert props_low.molecular_weight < props_mid.molecular_weight < props_high.molecular_weight

    def test_name_normalization(self) -> None:
        """Test that propellant names are normalized."""
        # These should all work
        props1 = get_combustion_properties("LOX", "RP1", 2.7, 7e6, use_cea=False)
        props2 = get_combustion_properties("LO2", "RP-1", 2.7, 7e6, use_cea=False)
        props3 = get_combustion_properties("OXYGEN", "KEROSENE", 2.7, 7e6, use_cea=False)

        assert props1.chamber_temp_k == props2.chamber_temp_k == props3.chamber_temp_k

    def test_unknown_propellant_raises(self) -> None:
        """Test that unknown propellants raise ValueError."""
        with pytest.raises(ValueError, match="not in database"):
            get_combustion_properties(
                oxidizer="UNKNOWN_OX",
                fuel="UNKNOWN_FUEL",
                mixture_ratio=2.0,
                chamber_pressure_pa=7e6,
                use_cea=False,
            )


class TestCEAAvailability:
    """Test CEA availability checking."""

    def test_is_cea_available_returns_bool(self) -> None:
        """Test that is_cea_available returns boolean."""
        result = is_cea_available()
        assert isinstance(result, bool)


class TestEngineInputsFromPropellants:
    """Test EngineInputs.from_propellants() factory method."""

    def test_from_propellants_basic(self) -> None:
        """Test basic from_propellants usage."""
        from openrocketengine.engine import EngineInputs
        from openrocketengine.units import kilonewtons, megapascals

        inputs = EngineInputs.from_propellants(
            oxidizer="LOX",
            fuel="RP1",
            thrust=kilonewtons(100),
            chamber_pressure=megapascals(7),
            mixture_ratio=2.7,
            use_cea=False,
        )

        assert inputs.thrust.to("kN").value == pytest.approx(100)
        assert inputs.chamber_pressure.to("MPa").value == pytest.approx(7)
        assert inputs.mixture_ratio == pytest.approx(2.7)
        # Chamber temp should be set from propellant data
        assert 3400 < inputs.chamber_temp.to("K").value < 3800
        assert inputs.name == "LOX/RP1 Engine"

    def test_from_propellants_with_defaults(self) -> None:
        """Test from_propellants with default parameters."""
        from openrocketengine.engine import EngineInputs
        from openrocketengine.units import newtons, pascals

        inputs = EngineInputs.from_propellants(
            oxidizer="LOX",
            fuel="LH2",
            thrust=newtons(50000),
            chamber_pressure=pascals(5e6),
            mixture_ratio=6.0,
            use_cea=False,
        )

        # Check defaults were applied
        assert inputs.exit_pressure.to("Pa").value == pytest.approx(101325)
        assert inputs.lstar.to("m").value == pytest.approx(1.0)
        assert inputs.contraction_ratio == pytest.approx(4.0)
        assert inputs.bell_fraction == pytest.approx(0.8)

    def test_from_propellants_full_workflow(self) -> None:
        """Test complete workflow from propellants to geometry."""
        from openrocketengine.engine import EngineInputs, design_engine
        from openrocketengine.units import kilonewtons, megapascals

        inputs = EngineInputs.from_propellants(
            oxidizer="LOX",
            fuel="CH4",
            thrust=kilonewtons(50),
            chamber_pressure=megapascals(5),
            mixture_ratio=3.2,
            name="Methane Test Engine",
            use_cea=False,
        )

        # Should be able to compute performance and geometry
        performance, geometry = design_engine(inputs)

        # Verify reasonable results
        assert 280 < performance.isp.value < 360  # LOX/CH4 Isp range
        assert performance.mdot.value > 0
        assert geometry.throat_diameter.value > 0
        assert geometry.expansion_ratio > 1

    def test_from_propellants_custom_name(self) -> None:
        """Test custom engine name."""
        from openrocketengine.engine import EngineInputs
        from openrocketengine.units import newtons, megapascals

        inputs = EngineInputs.from_propellants(
            oxidizer="LOX",
            fuel="RP1",
            thrust=newtons(10000),
            chamber_pressure=megapascals(2),
            mixture_ratio=2.5,
            name="My Custom Engine",
            use_cea=False,
        )

        assert inputs.name == "My Custom Engine"

