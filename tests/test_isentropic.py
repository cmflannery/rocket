"""Tests for the isentropic flow equations module."""

import math

import numpy as np
import pytest

from openrocketengine.isentropic import (
    G0_SI,
    R_UNIVERSAL_SI,
    area_ratio_from_mach,
    bell_nozzle_length,
    chamber_volume,
    characteristic_velocity,
    conical_nozzle_length,
    diameter_from_area,
    exhaust_velocity,
    expansion_ratio_from_pressure_ratio,
    mach_from_area_ratio_subsonic,
    mach_from_area_ratio_supersonic,
    mach_from_pressure_ratio,
    mass_flow_rate,
    mass_flow_rate_from_throat,
    pressure_ratio_from_mach,
    specific_gas_constant,
    specific_impulse,
    temperature_ratio_from_mach,
    throat_area,
    throat_pressure,
    throat_temperature,
    thrust_coefficient,
    thrust_coefficient_sweep,
    thrust_coefficient_vacuum,
)


class TestConstants:
    """Test physical constants."""

    def test_g0_si(self) -> None:
        assert pytest.approx(9.80665) == G0_SI

    def test_r_universal(self) -> None:
        assert pytest.approx(8314.46, rel=1e-4) == R_UNIVERSAL_SI


class TestGasProperties:
    """Test gas property calculations."""

    def test_specific_gas_constant_air(self) -> None:
        """Test R for air (MW ≈ 28.97)."""
        R = specific_gas_constant(28.97)
        assert pytest.approx(287.0, rel=1e-2) == R

    def test_specific_gas_constant_h2o(self) -> None:
        """Test R for water vapor (MW ≈ 18)."""
        R = specific_gas_constant(18.0)
        assert pytest.approx(461.9, rel=1e-2) == R


class TestCharacteristicVelocity:
    """Test c* calculations."""

    def test_cstar_typical_rocket(self) -> None:
        """Test c* for typical rocket conditions."""
        gamma = 1.2
        R = 350.0  # J/(kg·K), typical for rocket exhaust
        Tc = 3000.0  # K

        cstar = characteristic_velocity(gamma, R, Tc)

        # c* should be in the range 1500-2500 m/s for most propellants
        assert 1500 < cstar < 2500

    def test_cstar_increases_with_temperature(self) -> None:
        """Test that c* increases with chamber temperature."""
        gamma = 1.2
        R = 350.0

        cstar_low = characteristic_velocity(gamma, R, 2500.0)
        cstar_high = characteristic_velocity(gamma, R, 3500.0)

        assert cstar_high > cstar_low

    def test_cstar_increases_with_R(self) -> None:
        """Test that c* increases with specific gas constant (lower MW)."""
        gamma = 1.2
        Tc = 3000.0

        cstar_low = characteristic_velocity(gamma, 300.0, Tc)
        cstar_high = characteristic_velocity(gamma, 400.0, Tc)

        assert cstar_high > cstar_low


class TestThrustCoefficient:
    """Test thrust coefficient calculations."""

    def test_cf_optimally_expanded(self) -> None:
        """Test Cf when pe = pa (optimally expanded)."""
        gamma = 1.2
        pe_pc = 0.01  # pe/pc
        pa_pc = 0.01  # pa/pc (optimally expanded)
        eps = 20.0

        Cf = thrust_coefficient(gamma, pe_pc, pa_pc, eps)

        # Cf should be ~1.5-1.8 for typical expansion
        assert 1.4 < Cf < 2.0

    def test_cf_overexpanded(self) -> None:
        """Test Cf when pe < pa (overexpanded)."""
        gamma = 1.2
        pe_pc = 0.005
        pa_pc = 0.01  # Higher ambient pressure
        eps = 30.0

        Cf = thrust_coefficient(gamma, pe_pc, pa_pc, eps)

        # Cf should be lower due to pressure drag
        assert Cf > 0  # Still positive thrust

    def test_cf_underexpanded(self) -> None:
        """Test Cf when pe > pa (underexpanded)."""
        gamma = 1.2
        pe_pc = 0.02
        pa_pc = 0.01  # Lower ambient pressure
        eps = 10.0

        Cf = thrust_coefficient(gamma, pe_pc, pa_pc, eps)

        # Cf should be positive
        assert Cf > 0

    def test_cf_vacuum(self) -> None:
        """Test vacuum thrust coefficient."""
        gamma = 1.2
        pe_pc = 0.01
        eps = 20.0

        Cf_vac = thrust_coefficient_vacuum(gamma, pe_pc, eps)
        Cf_sl = thrust_coefficient(gamma, pe_pc, 0.01, eps)

        # Vacuum Cf should be higher than sea level
        assert Cf_vac > Cf_sl


class TestSpecificImpulse:
    """Test specific impulse calculations."""

    def test_isp_typical_biprop(self) -> None:
        """Test Isp for typical bipropellant engine."""
        cstar = 1800.0  # m/s
        Cf = 1.6

        Isp = specific_impulse(cstar, Cf, G0_SI)

        # Should be in reasonable range for biprop
        assert 250 < Isp < 350

    def test_isp_increases_with_cstar(self) -> None:
        Cf = 1.6
        Isp_low = specific_impulse(1500.0, Cf, G0_SI)
        Isp_high = specific_impulse(2000.0, Cf, G0_SI)

        assert Isp_high > Isp_low

    def test_isp_increases_with_cf(self) -> None:
        cstar = 1800.0
        Isp_low = specific_impulse(cstar, 1.4, G0_SI)
        Isp_high = specific_impulse(cstar, 1.8, G0_SI)

        assert Isp_high > Isp_low


class TestExhaustVelocity:
    """Test exhaust velocity calculations."""

    def test_exhaust_velocity_typical(self) -> None:
        gamma = 1.2
        R = 350.0
        Tc = 3000.0
        pe_pc = 0.01

        ue = exhaust_velocity(gamma, R, Tc, pe_pc)

        # Should be 2000-3500 m/s for typical rocket
        assert 2000 < ue < 3500

    def test_exhaust_velocity_increases_with_expansion(self) -> None:
        """Test that ue increases with expansion (lower pe/pc)."""
        gamma = 1.2
        R = 350.0
        Tc = 3000.0

        ue_low_expansion = exhaust_velocity(gamma, R, Tc, 0.1)
        ue_high_expansion = exhaust_velocity(gamma, R, Tc, 0.01)

        assert ue_high_expansion > ue_low_expansion


class TestMassFlow:
    """Test mass flow rate calculations."""

    def test_mass_flow_from_thrust_isp(self) -> None:
        thrust = 50000.0  # N
        Isp = 300.0  # s

        mdot = mass_flow_rate(thrust, Isp, G0_SI)

        # mdot = F / (Isp * g0)
        expected = thrust / (Isp * G0_SI)
        assert mdot == pytest.approx(expected)

    def test_mass_flow_from_throat(self) -> None:
        """Test mass flow from throat conditions."""
        pc = 2e6  # Pa
        At = 0.01  # m^2
        gamma = 1.2
        R = 350.0
        Tc = 3000.0

        mdot = mass_flow_rate_from_throat(pc, At, gamma, R, Tc)

        assert mdot > 0

    def test_throat_area_inverse(self) -> None:
        """Test that throat_area inverts mass_flow calculation."""
        mdot = 10.0  # kg/s
        cstar = 1800.0  # m/s
        pc = 2e6  # Pa

        At = throat_area(mdot, cstar, pc)

        # At = mdot * cstar / pc
        expected = mdot * cstar / pc
        assert At == pytest.approx(expected)


class TestMachRelations:
    """Test Mach number relations."""

    def test_mach_from_pressure_ratio(self) -> None:
        """Test Mach number from pressure ratio."""
        gamma = 1.4
        pc_p = 10.0  # stagnation/static pressure ratio

        M = mach_from_pressure_ratio(pc_p, gamma)

        assert M > 1  # Supersonic for this pressure ratio

    def test_pressure_ratio_from_mach_sonic(self) -> None:
        """Test pressure ratio at sonic conditions (M=1)."""
        gamma = 1.4

        pc_p = pressure_ratio_from_mach(1.0, gamma)

        # For gamma=1.4, critical pressure ratio is ~1.893
        assert pc_p == pytest.approx(1.893, rel=1e-2)

    def test_pressure_ratio_mach_round_trip(self) -> None:
        """Test that pressure ratio and Mach conversions are consistent."""
        gamma = 1.2
        M_original = 2.5

        pc_p = pressure_ratio_from_mach(M_original, gamma)
        M_recovered = mach_from_pressure_ratio(pc_p, gamma)

        assert M_recovered == pytest.approx(M_original, rel=1e-6)

    def test_temperature_ratio_sonic(self) -> None:
        """Test temperature ratio at sonic conditions."""
        gamma = 1.4

        Tc_T = temperature_ratio_from_mach(1.0, gamma)

        # For gamma=1.4, critical temperature ratio is 1.2
        assert Tc_T == pytest.approx(1.2)

    def test_area_ratio_sonic(self) -> None:
        """Test area ratio at sonic conditions (should be 1)."""
        gamma = 1.4

        A_Astar = area_ratio_from_mach(1.0, gamma)

        assert A_Astar == pytest.approx(1.0)

    def test_area_ratio_supersonic(self) -> None:
        """Test area ratio for supersonic flow."""
        gamma = 1.4

        A_Astar = area_ratio_from_mach(2.0, gamma)

        assert A_Astar > 1.0

    def test_mach_from_area_ratio_supersonic(self) -> None:
        """Test supersonic Mach from area ratio."""
        gamma = 1.2
        eps = 10.0

        M = mach_from_area_ratio_supersonic(eps, gamma)

        assert M > 1.0

        # Verify by computing area ratio back
        eps_check = area_ratio_from_mach(M, gamma)
        assert eps_check == pytest.approx(eps, rel=1e-4)

    def test_mach_from_area_ratio_subsonic(self) -> None:
        """Test subsonic Mach from area ratio."""
        gamma = 1.2
        eps = 2.0

        M = mach_from_area_ratio_subsonic(eps, gamma)

        assert M < 1.0

        # Verify by computing area ratio back
        eps_check = area_ratio_from_mach(M, gamma)
        assert eps_check == pytest.approx(eps, rel=1e-4)


class TestThroatConditions:
    """Test throat (critical) conditions."""

    def test_throat_temperature(self) -> None:
        Tc = 3000.0
        gamma = 1.2

        Tt = throat_temperature(Tc, gamma)

        assert Tt < Tc
        # Tt = Tc / (1 + (gamma-1)/2) = Tc / 1.1 ≈ 2727 K
        assert Tt == pytest.approx(Tc / 1.1)

    def test_throat_pressure(self) -> None:
        pc = 2e6
        gamma = 1.2

        pt = throat_pressure(pc, gamma)

        assert pt < pc


class TestExpansionRatio:
    """Test expansion ratio calculations."""

    def test_expansion_ratio_from_pressure_ratio(self) -> None:
        pc_pe = 50.0
        gamma = 1.2

        eps = expansion_ratio_from_pressure_ratio(pc_pe, gamma)

        assert eps > 1.0


class TestGeometry:
    """Test geometric calculations."""

    def test_chamber_volume(self) -> None:
        lstar = 1.0  # m
        At = 0.01  # m^2

        Vc = chamber_volume(lstar, At)

        assert Vc == pytest.approx(0.01)

    def test_diameter_from_area(self) -> None:
        area = math.pi  # m^2 (gives diameter of 2)

        D = diameter_from_area(area)

        assert pytest.approx(2.0) == D

    def test_conical_nozzle_length(self) -> None:
        Rt = 0.05  # m
        Re = 0.15  # m
        half_angle = math.radians(15)

        Ln = conical_nozzle_length(Rt, Re, half_angle)

        expected = (Re - Rt) / math.tan(half_angle)
        assert Ln == pytest.approx(expected)

    def test_bell_nozzle_shorter_than_conical(self) -> None:
        """Test that 80% bell is shorter than conical."""
        Rt = 0.05
        Re = 0.15

        L_conical = conical_nozzle_length(Rt, Re, math.radians(15))
        L_bell = bell_nozzle_length(Rt, Re, bell_fraction=0.8)

        assert L_bell < L_conical
        assert L_bell == pytest.approx(0.8 * L_conical, rel=1e-4)


class TestVectorized:
    """Test vectorized functions."""

    def test_thrust_coefficient_sweep(self) -> None:
        gamma = 1.2
        pe_pc = 0.01
        pa_pc_array = np.array([0.02, 0.01, 0.005, 0.001, 0.0])
        eps = 20.0

        Cf_array = thrust_coefficient_sweep(gamma, pe_pc, pa_pc_array, eps)

        assert len(Cf_array) == len(pa_pc_array)
        # Cf should increase as ambient pressure decreases
        assert all(Cf_array[i] <= Cf_array[i + 1] for i in range(len(Cf_array) - 1))

