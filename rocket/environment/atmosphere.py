"""US Standard Atmosphere 1976 model.

Provides temperature, pressure, and density as functions of altitude
for Earth's atmosphere up to 86 km (geometric altitude).

The model divides the atmosphere into layers with different lapse rates:
- Troposphere (0-11 km): -6.5 K/km lapse rate
- Tropopause (11-20 km): isothermal at 216.65 K
- Stratosphere (20-32 km): +1.0 K/km
- Stratosphere (32-47 km): +2.8 K/km
- Stratopause (47-51 km): isothermal at 270.65 K
- Mesosphere (51-71 km): -2.8 K/km
- Mesosphere (71-86 km): -2.0 K/km

Reference: U.S. Standard Atmosphere, 1976 (NASA-TM-X-74335)

Example:
    >>> from rocket.environment import Atmosphere
    >>>
    >>> atm = Atmosphere()
    >>> result = atm.at_altitude(10000)  # 10 km
    >>> print(f"Density: {result.density:.4f} kg/m^3")
    >>> print(f"Temperature: {result.temperature:.1f} K")
    >>> print(f"Pressure: {result.pressure:.0f} Pa")
    >>> print(f"Speed of sound: {result.speed_of_sound:.1f} m/s")
"""

from dataclasses import dataclass

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

# =============================================================================
# Constants
# =============================================================================

# Sea level conditions
T0 = 288.15  # Temperature [K]
P0 = 101325.0  # Pressure [Pa]
RHO0 = 1.225  # Density [kg/m^3]

# Physical constants
R_AIR = 287.05287  # Specific gas constant for dry air [J/(kg·K)]
GAMMA_AIR = 1.4  # Ratio of specific heats for air
G0 = 9.80665  # Standard gravity [m/s^2]
M_AIR = 0.0289644  # Molar mass of dry air [kg/mol]
R_UNIVERSAL = 8.31447  # Universal gas constant [J/(mol·K)]

# Earth parameters
R_EARTH = 6356766.0  # Earth radius for geopotential altitude [m]


# Layer definitions: (base_altitude_km, base_temp_K, lapse_rate_K_per_km)
LAYERS = [
    (0.0, 288.15, -6.5),      # Troposphere
    (11.0, 216.65, 0.0),      # Tropopause
    (20.0, 216.65, 1.0),      # Stratosphere 1
    (32.0, 228.65, 2.8),      # Stratosphere 2
    (47.0, 270.65, 0.0),      # Stratopause
    (51.0, 270.65, -2.8),     # Mesosphere 1
    (71.0, 214.65, -2.0),     # Mesosphere 2
]


# =============================================================================
# Result Classes
# =============================================================================


@beartype
@dataclass(frozen=True)
class AtmosphereResult:
    """Atmospheric conditions at a given altitude.

    Attributes:
        altitude: Geometric altitude [m]
        temperature: Static temperature [K]
        pressure: Static pressure [Pa]
        density: Air density [kg/m^3]
        speed_of_sound: Speed of sound [m/s]
        dynamic_viscosity: Dynamic viscosity [Pa·s]
        mach_number: Mach number if velocity provided
    """
    altitude: float
    temperature: float
    pressure: float
    density: float
    speed_of_sound: float
    dynamic_viscosity: float
    mach_number: float | None = None

    @property
    def is_vacuum(self) -> bool:
        """Check if altitude is above atmosphere (< 1e-6 Pa)."""
        return self.pressure < 1e-6


# =============================================================================
# Atmosphere Model
# =============================================================================


@beartype
class Atmosphere:
    """US Standard Atmosphere 1976 model.

    Provides atmospheric properties as functions of altitude for
    Earth's atmosphere from sea level to 86 km.

    Example:
        >>> atm = Atmosphere()
        >>> rho = atm.density(10000)  # Density at 10 km
        >>> T = atm.temperature(30000)  # Temperature at 30 km
        >>> result = atm.at_altitude(50000)  # All properties at 50 km
    """

    def __init__(self) -> None:
        """Initialize atmosphere model."""
        # Precompute base pressures for each layer
        self._base_pressures = self._compute_base_pressures()

    def _compute_base_pressures(self) -> list[float]:
        """Compute pressure at the base of each layer."""
        pressures = [P0]  # Sea level

        for i in range(len(LAYERS) - 1):
            h0, T0_layer, lapse = LAYERS[i]
            h1, _, _ = LAYERS[i + 1]

            dh = (h1 - h0) * 1000  # Convert km to m

            if abs(lapse) < 1e-10:
                # Isothermal layer
                p = pressures[-1] * np.exp(-G0 * dh / (R_AIR * T0_layer))
            else:
                # Gradient layer
                lapse_m = lapse / 1000  # K/m
                T1 = T0_layer + lapse_m * dh
                p = pressures[-1] * (T1 / T0_layer) ** (-G0 / (R_AIR * lapse_m))

            pressures.append(p)

        return pressures

    @staticmethod
    def _geometric_to_geopotential(h_geometric: float) -> float:
        """Convert geometric altitude to geopotential altitude.

        Args:
            h_geometric: Geometric altitude [m]

        Returns:
            Geopotential altitude [m]
        """
        return R_EARTH * h_geometric / (R_EARTH + h_geometric)

    def _find_layer(self, h_km: float) -> int:
        """Find the atmospheric layer index for a given altitude."""
        for i in range(len(LAYERS) - 1, -1, -1):
            if h_km >= LAYERS[i][0]:
                return i
        return 0

    @beartype
    def temperature(self, altitude: float) -> float:
        """Get temperature at altitude.

        Args:
            altitude: Geometric altitude [m]

        Returns:
            Temperature [K]
        """
        if altitude < 0:
            return T0
        if altitude > 86000:
            return 186.87  # Approximate temperature above 86 km

        h_geo = self._geometric_to_geopotential(altitude)
        h_km = h_geo / 1000

        layer_idx = self._find_layer(h_km)
        h0, T0_layer, lapse = LAYERS[layer_idx]

        dh = (h_km - h0) * 1000  # m above layer base
        return T0_layer + (lapse / 1000) * dh

    @beartype
    def pressure(self, altitude: float) -> float:
        """Get pressure at altitude.

        Args:
            altitude: Geometric altitude [m]

        Returns:
            Pressure [Pa]
        """
        if altitude < 0:
            return P0
        if altitude > 86000:
            return 0.0  # Essentially vacuum

        h_geo = self._geometric_to_geopotential(altitude)
        h_km = h_geo / 1000

        layer_idx = self._find_layer(h_km)
        h0, T0_layer, lapse = LAYERS[layer_idx]
        p0 = self._base_pressures[layer_idx]

        dh = (h_km - h0) * 1000  # m above layer base

        if abs(lapse) < 1e-10:
            # Isothermal layer
            return p0 * np.exp(-G0 * dh / (R_AIR * T0_layer))
        else:
            # Gradient layer
            lapse_m = lapse / 1000  # K/m
            T = T0_layer + lapse_m * dh
            return p0 * (T / T0_layer) ** (-G0 / (R_AIR * lapse_m))

    @beartype
    def density(self, altitude: float) -> float:
        """Get density at altitude.

        Args:
            altitude: Geometric altitude [m]

        Returns:
            Density [kg/m^3]
        """
        p = self.pressure(altitude)
        T = self.temperature(altitude)
        return p / (R_AIR * T) if T > 0 else 0.0

    @beartype
    def speed_of_sound(self, altitude: float) -> float:
        """Get speed of sound at altitude.

        Args:
            altitude: Geometric altitude [m]

        Returns:
            Speed of sound [m/s]
        """
        T = self.temperature(altitude)
        return np.sqrt(GAMMA_AIR * R_AIR * T)

    @beartype
    def dynamic_viscosity(self, altitude: float) -> float:
        """Get dynamic viscosity using Sutherland's formula.

        Args:
            altitude: Geometric altitude [m]

        Returns:
            Dynamic viscosity [Pa·s]
        """
        T = self.temperature(altitude)
        # Sutherland's formula constants for air
        mu0 = 1.716e-5  # Reference viscosity [Pa·s]
        T0_ref = 273.15  # Reference temperature [K]
        S = 110.4  # Sutherland constant [K]

        return mu0 * (T / T0_ref) ** 1.5 * (T0_ref + S) / (T + S)

    @beartype
    def at_altitude(
        self,
        altitude: float,
        velocity: float | None = None,
    ) -> AtmosphereResult:
        """Get all atmospheric properties at altitude.

        Args:
            altitude: Geometric altitude [m]
            velocity: Optional velocity for Mach number calculation [m/s]

        Returns:
            AtmosphereResult with all properties
        """
        T = self.temperature(altitude)
        p = self.pressure(altitude)
        rho = self.density(altitude)
        a = self.speed_of_sound(altitude)
        mu = self.dynamic_viscosity(altitude)

        mach = velocity / a if velocity is not None else None

        return AtmosphereResult(
            altitude=altitude,
            temperature=T,
            pressure=p,
            density=rho,
            speed_of_sound=a,
            dynamic_viscosity=mu,
            mach_number=mach,
        )

    @beartype
    def dynamic_pressure(self, altitude: float, velocity: float) -> float:
        """Get dynamic pressure (q = 0.5 * rho * v^2).

        Args:
            altitude: Geometric altitude [m]
            velocity: Velocity [m/s]

        Returns:
            Dynamic pressure [Pa]
        """
        rho = self.density(altitude)
        return 0.5 * rho * velocity ** 2

    @beartype
    def profile(
        self,
        altitudes: NDArray[np.float64] | list[float],
    ) -> dict[str, NDArray[np.float64]]:
        """Get atmospheric properties over a range of altitudes.

        Args:
            altitudes: Array of altitudes [m]

        Returns:
            Dictionary with arrays of temperature, pressure, density, speed_of_sound
        """
        altitudes = np.asarray(altitudes)

        return {
            "altitude": altitudes,
            "temperature": np.array([self.temperature(h) for h in altitudes]),
            "pressure": np.array([self.pressure(h) for h in altitudes]),
            "density": np.array([self.density(h) for h in altitudes]),
            "speed_of_sound": np.array([self.speed_of_sound(h) for h in altitudes]),
        }


# =============================================================================
# Convenience Functions
# =============================================================================


# Singleton instance
_default_atmosphere = Atmosphere()


@beartype
def get_atmosphere() -> Atmosphere:
    """Get the default atmosphere model instance."""
    return _default_atmosphere


@beartype
def density_at_altitude(altitude: float) -> float:
    """Quick density lookup at altitude [m]."""
    return _default_atmosphere.density(altitude)


@beartype
def pressure_at_altitude(altitude: float) -> float:
    """Quick pressure lookup at altitude [m]."""
    return _default_atmosphere.pressure(altitude)


@beartype
def temperature_at_altitude(altitude: float) -> float:
    """Quick temperature lookup at altitude [m]."""
    return _default_atmosphere.temperature(altitude)

