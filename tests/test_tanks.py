"""Tests for the tanks module."""

import math

import pytest

from rocket.tanks import (
    TANK_MATERIALS,
    format_tank_summary,
    get_propellant_density,
    list_materials,
    list_propellants,
    size_propellant,
    size_tank,
)
from rocket.units import (
    kilograms,
    km_per_second,
    meters,
    meters_per_second,
    pascals,
)


class TestPropellantDatabase:
    """Test propellant density database."""

    def test_list_propellants(self) -> None:
        """Test listing available propellants."""
        propellants = list_propellants()

        assert len(propellants) > 0
        assert "LOX" in propellants
        assert "RP1" in propellants
        assert "LH2" in propellants
        assert "CH4" in propellants

    def test_get_lox_density(self) -> None:
        """Test getting LOX density."""
        rho = get_propellant_density("LOX")
        assert 1100 < rho < 1200  # ~1141 kg/m³

    def test_get_lh2_density(self) -> None:
        """Test getting LH2 density."""
        rho = get_propellant_density("LH2")
        assert 60 < rho < 80  # ~70.8 kg/m³

    def test_get_rp1_density(self) -> None:
        """Test getting RP-1 density."""
        rho = get_propellant_density("RP1")
        assert 800 < rho < 850  # ~810 kg/m³

    def test_name_normalization(self) -> None:
        """Test propellant name normalization."""
        # These should all work
        assert get_propellant_density("LOX") == get_propellant_density("LO2")
        assert get_propellant_density("RP1") == get_propellant_density("RP-1")
        assert get_propellant_density("CH4") == get_propellant_density("LCH4")

    def test_unknown_propellant_raises(self) -> None:
        """Test that unknown propellant raises ValueError."""
        with pytest.raises(ValueError, match="Unknown propellant"):
            get_propellant_density("UNKNOWN_PROP")


class TestMaterialDatabase:
    """Test tank material database."""

    def test_list_materials(self) -> None:
        """Test listing available materials."""
        materials = list_materials()

        assert len(materials) > 0
        assert "Al2219" in materials
        assert "SS301" in materials
        assert "CFRP" in materials

    def test_material_properties(self) -> None:
        """Test that materials have required properties."""
        for _name, props in TANK_MATERIALS.items():
            assert "density" in props
            assert "yield_strength" in props
            assert "ultimate_strength" in props
            assert props["density"] > 0
            assert props["yield_strength"] > 0
            assert props["ultimate_strength"] >= props["yield_strength"]


class TestSizePropellant:
    """Test propellant sizing calculations."""

    def test_basic_propellant_sizing(self) -> None:
        """Test basic propellant mass calculation."""
        prop = size_propellant(
            isp_s=300,
            delta_v=meters_per_second(3000),
            dry_mass=kilograms(500),
            mixture_ratio=2.7,
        )

        # Check all outputs are valid
        assert prop.total_propellant.value > 0
        assert prop.oxidizer_mass.value > 0
        assert prop.fuel_mass.value > 0
        assert prop.burn_time.value > 0
        assert prop.mass_ratio > 1.0

        # Check mass balance
        total = prop.oxidizer_mass.value + prop.fuel_mass.value
        assert total == pytest.approx(prop.total_propellant.value, rel=1e-10)

        # Check mixture ratio is preserved
        actual_mr = prop.oxidizer_mass.value / prop.fuel_mass.value
        assert actual_mr == pytest.approx(2.7, rel=1e-10)

    def test_rocket_equation_accuracy(self) -> None:
        """Test that rocket equation is correctly implemented."""
        # Known values: dv = Isp * g0 * ln(MR)
        # For dv = 3000 m/s, Isp = 300s, MR = exp(3000 / (300 * 9.80665)) = 2.78
        prop = size_propellant(
            isp_s=300,
            delta_v=meters_per_second(3000),
            dry_mass=kilograms(1000),
            mixture_ratio=1.0,  # No split for simple calculation
        )

        expected_mr = math.exp(3000 / (300 * 9.80665))
        assert prop.mass_ratio == pytest.approx(expected_mr, rel=1e-6)

        # Propellant mass = dry_mass * (MR - 1)
        expected_prop = 1000 * (expected_mr - 1)
        assert prop.total_propellant.value == pytest.approx(expected_prop, rel=1e-6)

    def test_high_delta_v(self) -> None:
        """Test with high delta-V (e.g., orbital)."""
        prop = size_propellant(
            isp_s=450,  # Hydrolox Isp
            delta_v=km_per_second(9.5),  # Orbital velocity
            dry_mass=kilograms(1000),
            mixture_ratio=6.0,
        )

        # Mass ratio should be high for orbital
        assert prop.mass_ratio > 5

    def test_with_mass_flow_rate(self) -> None:
        """Test burn time calculation with explicit mdot."""
        from rocket.units import kg_per_second

        prop = size_propellant(
            isp_s=300,
            delta_v=meters_per_second(3000),
            dry_mass=kilograms(500),
            mixture_ratio=2.7,
            mdot=kg_per_second(10),
        )

        # Burn time = propellant_mass / mdot
        expected_burn_time = prop.total_propellant.value / 10
        assert prop.burn_time.value == pytest.approx(expected_burn_time, rel=1e-6)

    def test_dimension_validation(self) -> None:
        """Test that dimension validation works."""
        with pytest.raises(ValueError, match="delta_v must be velocity"):
            size_propellant(
                isp_s=300,
                delta_v=kilograms(3000),  # Wrong dimension!
                dry_mass=kilograms(500),
            )

        with pytest.raises(ValueError, match="dry_mass must be mass"):
            size_propellant(
                isp_s=300,
                delta_v=meters_per_second(3000),
                dry_mass=meters(500),  # Wrong dimension!
            )


class TestSizeTank:
    """Test tank sizing calculations."""

    def test_basic_tank_sizing(self) -> None:
        """Test basic tank sizing."""
        tank = size_tank(
            propellant_mass=kilograms(10000),
            propellant="LOX",
            tank_pressure=pascals(500000),  # 5 bar
            material="Al2219",
        )

        # Check all outputs are valid
        assert tank.volume.value > 0
        assert tank.diameter.value > 0
        assert tank.barrel_length.value >= 0
        assert tank.dome_height.value > 0
        assert tank.total_length.value > 0
        assert tank.wall_thickness.value > 0
        assert tank.dry_mass.value > 0
        assert tank.propellant == "LOX"
        assert tank.material == "Al2219"

    def test_volume_calculation(self) -> None:
        """Test that tank volume is correct for propellant mass."""
        tank = size_tank(
            propellant_mass=kilograms(1141),  # 1 m³ of LOX
            propellant="LOX",
            tank_pressure=pascals(300000),
            ullage_fraction=0,  # No ullage for exact calculation
        )

        # Volume should be approximately 1 m³
        assert tank.volume.value == pytest.approx(1.0, rel=0.01)

    def test_fixed_diameter(self) -> None:
        """Test tank sizing with fixed diameter."""
        tank = size_tank(
            propellant_mass=kilograms(5000),
            propellant="RP1",
            tank_pressure=pascals(400000),
            diameter=meters(1.5),
        )

        assert tank.diameter.value == pytest.approx(1.5, rel=1e-6)

    def test_wall_thickness_increases_with_pressure(self) -> None:
        """Test that wall thickness increases with pressure."""
        tank_low = size_tank(
            propellant_mass=kilograms(5000),
            propellant="LOX",
            tank_pressure=pascals(200000),
            diameter=meters(1.0),
        )

        tank_high = size_tank(
            propellant_mass=kilograms(5000),
            propellant="LOX",
            tank_pressure=pascals(1000000),
            diameter=meters(1.0),
        )

        assert tank_high.wall_thickness.value > tank_low.wall_thickness.value

    def test_different_materials(self) -> None:
        """Test that different materials give different results."""
        tank_al = size_tank(
            propellant_mass=kilograms(5000),
            propellant="LOX",
            tank_pressure=pascals(500000),
            material="Al2219",
            diameter=meters(1.0),
        )

        tank_ss = size_tank(
            propellant_mass=kilograms(5000),
            propellant="LOX",
            tank_pressure=pascals(500000),
            material="SS301",
            diameter=meters(1.0),
        )

        # Steel is stronger so thinner wall, but denser so may be heavier
        assert tank_al.wall_thickness.value != tank_ss.wall_thickness.value

    def test_unknown_material_raises(self) -> None:
        """Test that unknown material raises ValueError."""
        with pytest.raises(ValueError, match="Unknown material"):
            size_tank(
                propellant_mass=kilograms(5000),
                propellant="LOX",
                tank_pressure=pascals(500000),
                material="UNOBTAINIUM",
            )

    def test_lh2_tank_larger_volume(self) -> None:
        """Test that LH2 tank is larger due to low density."""
        tank_lox = size_tank(
            propellant_mass=kilograms(1000),
            propellant="LOX",
            tank_pressure=pascals(300000),
        )

        tank_lh2 = size_tank(
            propellant_mass=kilograms(1000),
            propellant="LH2",
            tank_pressure=pascals(300000),
        )

        # LH2 density is ~16x lower than LOX
        assert tank_lh2.volume.value > tank_lox.volume.value * 10


class TestTankSummary:
    """Test tank summary formatting."""

    def test_format_tank_summary(self) -> None:
        """Test formatting tank geometry as string."""
        tank = size_tank(
            propellant_mass=kilograms(5000),
            propellant="LOX",
            tank_pressure=pascals(500000),
            material="Al2219",
        )

        summary = format_tank_summary(tank)

        assert "LOX" in summary
        assert "Al2219" in summary
        assert "Volume" in summary
        assert "Diameter" in summary
        assert "Wall thickness" in summary
        assert "Dry mass" in summary


class TestIntegration:
    """Integration tests combining propellant and tank sizing."""

    def test_full_vehicle_sizing_workflow(self) -> None:
        """Test complete workflow from delta-V to tanks."""
        # Step 1: Size propellant
        prop = size_propellant(
            isp_s=300,
            delta_v=km_per_second(3),
            dry_mass=kilograms(500),
            mixture_ratio=2.7,
        )

        # Step 2: Size oxidizer tank
        lox_tank = size_tank(
            propellant_mass=prop.oxidizer_mass,
            propellant="LOX",
            tank_pressure=pascals(400000),
        )

        # Step 3: Size fuel tank
        rp1_tank = size_tank(
            propellant_mass=prop.fuel_mass,
            propellant="RP1",
            tank_pressure=pascals(300000),
        )

        # Verify results are reasonable
        assert lox_tank.dry_mass.value > 0
        assert rp1_tank.dry_mass.value > 0

        # With MR=2.7, oxidizer mass is ~73% of total
        # LOX is denser than RP-1, but more mass, so LOX tank is larger
        assert lox_tank.volume.value > rp1_tank.volume.value

