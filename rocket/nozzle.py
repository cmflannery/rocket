"""Nozzle contour generation for rocket engines.

This module provides functions to generate nozzle contours for various
nozzle types including:
- Conical nozzles (simple 15° half-angle)
- Rao parabolic bell nozzles (optimized for performance)

The contours can be exported to CSV for CAD import.

References:
    - Rao, G.V.R., "Exhaust Nozzle Contour for Optimum Thrust",
      Jet Propulsion, Vol. 28, No. 6, 1958
    - Huzel & Huang, "Modern Engineering for Design of Liquid-Propellant
      Rocket Engines", Chapter 4
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

from rocket.engine import EngineGeometry, EngineInputs
from rocket.units import Quantity

# =============================================================================
# Nozzle Contour Data Structure
# =============================================================================


@beartype
@dataclass(frozen=True)
class NozzleContour:
    """Nozzle contour defined by axial and radial coordinates.

    The contour represents the inner wall of the nozzle from the chamber
    through the throat to the exit. Coordinates are in meters.

    Attributes:
        x: Axial positions [m], with x=0 at throat
        y: Radial positions [m] (radius, not diameter)
        contour_type: Type of contour ("rao_bell", "conical", etc.)
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    contour_type: str

    def __post_init__(self) -> None:
        """Validate contour data."""
        if len(self.x) != len(self.y):
            raise ValueError(
                f"x and y arrays must have same length, got {len(self.x)} and {len(self.y)}"
            )
        if len(self.x) < 2:
            raise ValueError("Contour must have at least 2 points")

    def to_csv(self, path: str | Path, include_header: bool = True) -> None:
        """Export contour to CSV file for CAD import.

        Args:
            path: Output file path
            include_header: Whether to include column headers
        """
        path = Path(path)
        with path.open("w") as f:
            if include_header:
                f.write("x_m,y_m,x_mm,y_mm\n")
            for xi, yi in zip(self.x, self.y, strict=True):
                f.write(f"{xi:.8f},{yi:.8f},{xi * 1000:.6f},{yi * 1000:.6f}\n")

    def to_arrays_mm(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return contour coordinates in millimeters.

        Returns:
            Tuple of (x_mm, y_mm) arrays
        """
        return self.x * 1000, self.y * 1000

    @property
    def length(self) -> float:
        """Total axial length of the contour [m]."""
        return float(self.x[-1] - self.x[0])

    @property
    def throat_radius(self) -> float:
        """Radius at throat (minimum y value) [m]."""
        return float(np.min(self.y))

    @property
    def exit_radius(self) -> float:
        """Radius at exit [m]."""
        return float(self.y[-1])

    @property
    def inlet_radius(self) -> float:
        """Radius at inlet [m]."""
        return float(self.y[0])


# =============================================================================
# Rao Bell Nozzle Contour
# =============================================================================


@beartype
def rao_bell_contour(
    throat_radius: Quantity,
    exit_radius: Quantity,
    expansion_ratio: float,
    bell_fraction: float = 0.8,
    num_points: int = 100,
) -> NozzleContour:
    """Generate a Rao parabolic bell nozzle contour.

    The Rao bell nozzle uses a parabolic approximation to the ideal
    thrust-optimized contour. It consists of:
    1. A circular arc leaving the throat (radius = 0.382 * Rt)
    2. A parabolic section to the exit

    The bell_fraction parameter specifies the length as a fraction of
    an equivalent 15° conical nozzle.

    Args:
        throat_radius: Throat radius [length]
        exit_radius: Exit radius [length]
        expansion_ratio: Area ratio Ae/At [-]
        bell_fraction: Length as fraction of 15° cone (typically 0.8)
        num_points: Number of points in the contour

    Returns:
        NozzleContour with the bell nozzle shape
    """
    # Convert to SI
    Rt = throat_radius.to("m").value
    Re = exit_radius.to("m").value

    # Rao parameters (from empirical correlations)
    # Initial angle leaving throat depends on expansion ratio
    # Final angle at exit also depends on expansion ratio
    # These are approximations from Rao's paper and Huzel & Huang

    # Throat circular arc radius
    Rn = 0.382 * Rt  # Radius of curvature leaving throat

    # Reference conical nozzle length (15° half-angle)
    Lc_15 = (Re - Rt) / math.tan(math.radians(15))

    # Bell nozzle length
    Ln = Lc_15 * bell_fraction

    # Initial and final angles (empirical fits from Rao curves)
    # These depend on expansion ratio and bell fraction
    theta_n = _rao_initial_angle(expansion_ratio, bell_fraction)
    theta_e = _rao_exit_angle(expansion_ratio, bell_fraction)

    # Convert to radians
    theta_n_rad = math.radians(theta_n)
    theta_e_rad = math.radians(theta_e)

    # Generate contour points

    # Point N: End of throat circular arc
    # x_N is relative to throat center
    x_N = Rn * math.sin(theta_n_rad)
    y_N = Rt + Rn * (1 - math.cos(theta_n_rad))

    # Point E: Exit
    x_E = Ln
    y_E = Re

    # Generate throat arc (from throat to point N)
    n_arc = num_points // 4
    theta_arc = np.linspace(0, theta_n_rad, n_arc)
    x_arc = Rn * np.sin(theta_arc)
    y_arc = Rt + Rn * (1 - np.cos(theta_arc))

    # Generate parabolic section (from N to E)
    # Using quadratic Bezier curve approximation
    n_parabola = num_points - n_arc

    # Control point for quadratic Bezier
    # The tangent at N has slope tan(theta_n)
    # The tangent at E has slope tan(theta_e)
    # Find intersection of these tangents

    m_N = math.tan(theta_n_rad)
    m_E = math.tan(theta_e_rad)

    # Line from N: y - y_N = m_N * (x - x_N)
    # Line from E: y - y_E = m_E * (x - x_E)
    # Solve for intersection (control point Q)

    if abs(m_N - m_E) > 1e-10:
        x_Q = (y_E - y_N + m_N * x_N - m_E * x_E) / (m_N - m_E)
        y_Q = y_N + m_N * (x_Q - x_N)
    else:
        # Parallel tangents (shouldn't happen for reasonable parameters)
        x_Q = (x_N + x_E) / 2
        y_Q = (y_N + y_E) / 2

    # Generate Bezier curve points
    t = np.linspace(0, 1, n_parabola)
    x_parabola = (1 - t) ** 2 * x_N + 2 * (1 - t) * t * x_Q + t**2 * x_E
    y_parabola = (1 - t) ** 2 * y_N + 2 * (1 - t) * t * y_Q + t**2 * y_E

    # Combine arc and parabola (skip first point of parabola to avoid duplicate)
    x = np.concatenate([x_arc, x_parabola[1:]])
    y = np.concatenate([y_arc, y_parabola[1:]])

    return NozzleContour(x=x, y=y, contour_type="rao_bell")


def _rao_initial_angle(expansion_ratio: float, bell_fraction: float) -> float:
    """Calculate initial expansion angle for Rao bell nozzle.

    Empirical correlation based on Rao's curves.

    Args:
        expansion_ratio: Area ratio Ae/At
        bell_fraction: Bell length fraction (0.6-1.0)

    Returns:
        Initial angle in degrees
    """
    # Empirical fit (approximation of Rao curves)
    # For 80% bell: ~30-35° for low eps, ~20-25° for high eps
    eps = expansion_ratio

    if bell_fraction <= 0.6:
        theta = 38 - 2.5 * math.log10(eps)
    elif bell_fraction <= 0.8:
        theta = 33 - 3.5 * math.log10(eps)
    else:
        theta = 28 - 4.0 * math.log10(eps)

    # Clamp to reasonable values
    return max(15.0, min(45.0, theta))


def _rao_exit_angle(expansion_ratio: float, bell_fraction: float) -> float:
    """Calculate exit angle for Rao bell nozzle.

    Empirical correlation based on Rao's curves.

    Args:
        expansion_ratio: Area ratio Ae/At
        bell_fraction: Bell length fraction (0.6-1.0)

    Returns:
        Exit angle in degrees
    """
    # Empirical fit
    # Exit angle is typically 6-12° for most practical nozzles
    eps = expansion_ratio

    if bell_fraction <= 0.6:
        theta = 14 - 1.5 * math.log10(eps)
    elif bell_fraction <= 0.8:
        theta = 11 - 2.0 * math.log10(eps)
    else:
        theta = 8 - 2.5 * math.log10(eps)

    # Clamp to reasonable values
    return max(4.0, min(15.0, theta))


# =============================================================================
# Conical Nozzle Contour
# =============================================================================


@beartype
def conical_contour(
    throat_radius: Quantity,
    exit_radius: Quantity,
    half_angle: float = 15.0,
    num_points: int = 100,
) -> NozzleContour:
    """Generate a conical nozzle contour.

    A simple conical nozzle with constant divergence angle.
    The standard half-angle is 15°.

    Args:
        throat_radius: Throat radius [length]
        exit_radius: Exit radius [length]
        half_angle: Nozzle half-angle in degrees (default 15°)
        num_points: Number of points in the contour

    Returns:
        NozzleContour with the conical shape
    """
    Rt = throat_radius.to("m").value
    Re = exit_radius.to("m").value

    # Nozzle length
    Ln = (Re - Rt) / math.tan(math.radians(half_angle))

    # Generate linear contour
    x = np.linspace(0, Ln, num_points)
    y = Rt + x * math.tan(math.radians(half_angle))

    return NozzleContour(x=x, y=y, contour_type="conical")


# =============================================================================
# Full Chamber Contour (Chamber + Convergent + Nozzle)
# =============================================================================


@beartype
def full_chamber_contour(
    inputs: EngineInputs,
    geometry: EngineGeometry,
    nozzle_contour: NozzleContour,
    num_chamber_points: int = 50,
    num_convergent_points: int = 30,
) -> NozzleContour:
    """Generate complete chamber contour including chamber and convergent section.

    Combines:
    1. Cylindrical chamber section
    2. Convergent section (circular arc transition + conical)
    3. Throat region with circular arc
    4. Divergent nozzle section

    Args:
        inputs: Engine inputs (for contraction angle)
        geometry: Computed geometry
        nozzle_contour: Pre-computed divergent nozzle contour
        num_chamber_points: Points for cylindrical section
        num_convergent_points: Points for convergent section

    Returns:
        Complete chamber contour from inlet to exit
    """
    # Extract dimensions
    Rc = geometry.chamber_diameter.to("m").value / 2
    Rt = geometry.throat_diameter.to("m").value / 2
    Lcyl = geometry.chamber_length.to("m").value
    contraction_angle = math.radians(inputs.contraction_angle)

    # Upstream radius of curvature (typically 1.5 * Rt for smooth transition)
    R1 = 1.5 * Rt

    # Convergent section geometry
    # The convergent has a circular arc transition from chamber to conical section

    # Calculate convergent section
    # Point where circular arc meets conical section
    theta_c = contraction_angle
    x_tan = R1 * math.sin(theta_c)  # axial distance from throat to tangent point
    y_tan = Rt + R1 * (1 - math.cos(theta_c))  # radius at tangent point

    # Length of conical section (from tangent point to chamber)
    L_cone = (Rc - y_tan) / math.tan(theta_c) if Rc > y_tan else 0

    # Total convergent length
    L_conv = x_tan + L_cone

    # Generate chamber section (negative x, before throat)
    x_chamber = np.linspace(-(Lcyl + L_conv), -L_conv, num_chamber_points)
    y_chamber = np.full_like(x_chamber, Rc)

    # Generate convergent conical section
    if L_cone > 0:
        n_cone = num_convergent_points // 2
        x_cone = np.linspace(-L_conv, -(x_tan), n_cone)
        y_cone = Rc - (x_cone + L_conv) * math.tan(theta_c)
    else:
        x_cone = np.array([])
        y_cone = np.array([])

    # Generate convergent circular arc (transition to throat)
    # Arc center is at (0, Rt + R1), tangent to throat at bottom
    # Arc goes from tangent point with cone (angle = theta_c) to throat (angle = 0)
    n_arc = num_convergent_points - len(x_cone)
    theta_range = np.linspace(theta_c, 0, n_arc)
    x_arc = -R1 * np.sin(theta_range)  # Negative (upstream of throat)
    y_arc = Rt + R1 * (1 - np.cos(theta_range))  # From y_tan down to Rt

    # Shift nozzle contour (it starts at x=0 at throat)
    x_nozzle = nozzle_contour.x
    y_nozzle = nozzle_contour.y

    # Combine all sections
    x_all = np.concatenate([x_chamber, x_cone, x_arc[:-1], x_nozzle])
    y_all = np.concatenate([y_chamber, y_cone, y_arc[:-1], y_nozzle])

    return NozzleContour(x=x_all, y=y_all, contour_type=f"full_{nozzle_contour.contour_type}")


# =============================================================================
# Convenience Functions
# =============================================================================


@beartype
def generate_nozzle_from_geometry(
    geometry: EngineGeometry,
    bell_fraction: float = 0.8,
    num_points: int = 100,
) -> NozzleContour:
    """Generate a Rao bell nozzle contour from engine geometry.

    Convenience function that extracts the necessary parameters from
    EngineGeometry.

    Args:
        geometry: Computed engine geometry
        bell_fraction: Bell length fraction (default 0.8)
        num_points: Number of contour points

    Returns:
        NozzleContour for the divergent section
    """
    return rao_bell_contour(
        throat_radius=geometry.throat_diameter / 2,
        exit_radius=geometry.exit_diameter / 2,
        expansion_ratio=geometry.expansion_ratio,
        bell_fraction=bell_fraction,
        num_points=num_points,
    )

