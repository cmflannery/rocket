"""Transient thermal analysis for rocket engine chambers.

This module simulates time-varying thermal behavior during:
- Engine startup (heat flux ramp-up, wall temperature rise)
- Steady-state operation (equilibrium conditions)
- Engine shutdown (cooling, thermal soak-back)

The model uses 1D radial heat conduction through the chamber wall
with time-varying boundary conditions.

Example:
    >>> from rocket.thermal.transient import simulate_startup, TransientThermalResult
    >>> from rocket import design_engine, EngineInputs
    >>>
    >>> inputs = EngineInputs.from_propellants("LOX", "CH4", ...)
    >>> perf, geom = design_engine(inputs)
    >>>
    >>> result = simulate_startup(
    ...     inputs, perf, geom,
    ...     wall_thickness=meters(0.003),
    ...     wall_material="copper",
    ...     coolant_temp=kelvin(100),
    ...     startup_time=2.0,
    ... )
    >>> print(f"Peak wall temp: {result.wall_temp_inner.max():.0f} K")
"""

import math
from dataclasses import dataclass

import numpy as np
from beartype import beartype
from numpy.typing import NDArray
from tqdm import tqdm

from rocket.engine import EngineGeometry, EngineInputs, EnginePerformance
from rocket.thermal.heat_flux import bartz_heat_flux
from rocket.thermal.materials import get_material_properties
from rocket.units import Quantity, kelvin, meters


# =============================================================================
# Transient Result Dataclass
# =============================================================================


@beartype
@dataclass(frozen=True)
class TransientThermalResult:
    """Results from a transient thermal simulation.

    All arrays are indexed by time step.

    Attributes:
        time: Time points [s]
        wall_temp_inner: Gas-side wall temperature [K]
        wall_temp_outer: Coolant-side wall temperature [K]
        heat_flux: Instantaneous heat flux at gas-side surface [W/m²]
        thermal_margin: Distance to material limit [K] (positive = safe)
        coolant_heat_absorbed: Cumulative heat absorbed by coolant [J]
        max_material_temp: Material temperature limit used [K]
        location: Location in engine (e.g., "throat")
    """

    time: NDArray[np.float64]
    wall_temp_inner: NDArray[np.float64]
    wall_temp_outer: NDArray[np.float64]
    heat_flux: NDArray[np.float64]
    thermal_margin: NDArray[np.float64]
    coolant_heat_absorbed: NDArray[np.float64]
    max_material_temp: float
    location: str

    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return len(self.time)

    @property
    def duration(self) -> float:
        """Total simulation duration [s]."""
        return float(self.time[-1] - self.time[0])

    @property
    def peak_wall_temp(self) -> float:
        """Peak inner wall temperature [K]."""
        return float(np.max(self.wall_temp_inner))

    @property
    def min_thermal_margin(self) -> float:
        """Minimum thermal margin [K] (most critical point)."""
        return float(np.min(self.thermal_margin))

    @property
    def time_to_steady_state(self) -> float | None:
        """Time to reach steady state [s], or None if not reached.

        Defined as when dT/dt < 1 K/s.
        """
        if len(self.wall_temp_inner) < 3:
            return None

        dt = self.time[1] - self.time[0]
        dT_dt = np.gradient(self.wall_temp_inner, dt)

        # Find first time dT/dt stays below threshold
        threshold = 1.0  # K/s
        steady_mask = np.abs(dT_dt) < threshold

        # Need sustained steady state (not just a momentary dip)
        for i in range(len(steady_mask) - 10):
            if np.all(steady_mask[i:i+10]):
                return float(self.time[i])

        return None

    def is_safe(self) -> bool:
        """Check if wall temperature stays within material limits."""
        return bool(np.all(self.thermal_margin > 0))

    def summary(self) -> str:
        """Generate text summary of transient results."""
        safe_status = "SAFE" if self.is_safe() else "EXCEEDS LIMIT"
        tss = self.time_to_steady_state

        lines = [
            "Transient Thermal Analysis Results",
            "=" * 50,
            f"Location: {self.location}",
            f"Duration: {self.duration:.2f} s",
            f"Time steps: {self.n_steps}",
            "",
            f"Peak inner wall temp: {self.peak_wall_temp:.0f} K",
            f"Material limit: {self.max_material_temp:.0f} K",
            f"Min thermal margin: {self.min_thermal_margin:.0f} K",
            f"Status: {safe_status}",
            "",
            f"Peak heat flux: {np.max(self.heat_flux)/1e6:.1f} MW/m²",
            f"Steady-state heat flux: {self.heat_flux[-1]/1e6:.1f} MW/m²",
        ]

        if tss is not None:
            lines.append(f"Time to steady state: {tss:.2f} s")
        else:
            lines.append("Time to steady state: Not reached")

        return "\n".join(lines)


# =============================================================================
# 1D Heat Conduction Solver
# =============================================================================


def _solve_1d_conduction(
    n_nodes: int,
    thickness: float,
    dt: float,
    n_steps: int,
    k: float | NDArray[np.float64],
    rho: float,
    cp: float | NDArray[np.float64],
    q_inner: NDArray[np.float64],
    T_outer: float | NDArray[np.float64],
    T_initial: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solve 1D transient heat conduction using explicit finite difference.

    Uses the Forward-Time Central-Space (FTCS) scheme.

    Args:
        n_nodes: Number of spatial nodes through wall thickness
        thickness: Wall thickness [m]
        dt: Time step [s]
        n_steps: Number of time steps
        k: Thermal conductivity [W/m·K] (scalar or temperature-dependent array)
        rho: Density [kg/m³]
        cp: Specific heat [J/kg·K] (scalar or temperature-dependent)
        q_inner: Heat flux at inner surface for each time step [W/m²]
        T_outer: Temperature at outer surface [K] (scalar or array)
        T_initial: Initial temperature throughout wall [K]

    Returns:
        Tuple of (T_inner_history, T_outer_history) arrays
    """
    dx = thickness / (n_nodes - 1)

    # Initialize temperature field
    T = np.full(n_nodes, T_initial, dtype=np.float64)

    # Storage for surface temperatures
    T_inner_history = np.zeros(n_steps, dtype=np.float64)
    T_outer_history = np.zeros(n_steps, dtype=np.float64)

    # Get thermal diffusivity
    if isinstance(k, np.ndarray):
        k_avg = np.mean(k)
    else:
        k_avg = k

    if isinstance(cp, np.ndarray):
        cp_avg = np.mean(cp)
    else:
        cp_avg = cp

    alpha = k_avg / (rho * cp_avg)

    # Check stability (Fo < 0.5 for explicit scheme)
    Fo = alpha * dt / dx**2
    if Fo >= 0.5:
        # Reduce time step or increase nodes
        raise ValueError(
            f"Explicit scheme unstable: Fo={Fo:.3f} >= 0.5. "
            f"Try smaller dt or more nodes."
        )

    for step in range(n_steps):
        # Get current boundary conditions
        q_bc = q_inner[step] if step < len(q_inner) else q_inner[-1]
        T_bc = T_outer[step] if isinstance(T_outer, np.ndarray) and step < len(T_outer) else T_outer

        # Inner boundary: heat flux condition
        # q = -k * dT/dx -> T[0] = T[1] + q*dx/k
        T_new = T.copy()

        # Interior nodes: explicit finite difference
        for i in range(1, n_nodes - 1):
            T_new[i] = T[i] + Fo * (T[i+1] - 2*T[i] + T[i-1])

        # Inner boundary: Neumann (heat flux)
        # Using ghost node: T[-1] = T[1] - 2*q*dx/k
        T_ghost = T[1] - 2 * q_bc * dx / k_avg
        T_new[0] = T[0] + Fo * (T[1] - 2*T[0] + T_ghost)

        # Outer boundary: Dirichlet (fixed temperature)
        T_new[-1] = T_bc

        # Update
        T = T_new

        # Store surface temperatures
        T_inner_history[step] = T[0]
        T_outer_history[step] = T[-1]

    return T_inner_history, T_outer_history


# =============================================================================
# Startup Simulation
# =============================================================================


@beartype
def simulate_startup(
    inputs: EngineInputs,
    performance: EnginePerformance,
    geometry: EngineGeometry,
    wall_thickness: Quantity,
    wall_material: str = "copper",
    coolant_temp: Quantity | None = None,
    max_material_temp: Quantity | None = None,
    startup_time: float = 2.0,
    steady_time: float = 3.0,
    dt: float = 0.001,
    n_nodes: int = 20,
    location: str = "throat",
    startup_profile: str = "ramp",
    progress: bool = False,
) -> TransientThermalResult:
    """Simulate thermal transient during engine startup.

    Models the wall temperature evolution from ambient to steady-state
    during engine ignition and initial operation.

    Args:
        inputs: Engine input parameters
        performance: Computed engine performance
        geometry: Computed engine geometry
        wall_thickness: Chamber wall thickness
        wall_material: Wall material name (e.g., "copper", "inconel718")
        coolant_temp: Coolant inlet temperature (default: 100 K for cryo)
        max_material_temp: Material temperature limit (default: from database)
        startup_time: Time for heat flux to ramp from 0 to steady [s]
        steady_time: Additional time to run at steady state [s]
        dt: Time step [s]
        n_nodes: Number of spatial nodes through wall
        location: Location to analyze ("throat", "chamber", "exit")
        startup_profile: Ramp profile ("ramp", "exponential", "step")
        progress: Show progress bar

    Returns:
        TransientThermalResult with time histories
    """
    # Get material properties
    mat = get_material_properties(wall_material)
    T_max = max_material_temp.to("K").value if max_material_temp else mat["max_temp"]

    # Default coolant temp for cryo propellants
    if coolant_temp is None:
        coolant_temp = kelvin(100.0)
    T_coolant = coolant_temp.to("K").value

    # Get steady-state heat flux at this location
    if location == "throat":
        local_diameter = geometry.throat_diameter
        local_mach = 1.0
    elif location == "chamber":
        local_diameter = geometry.chamber_diameter
        local_mach = 0.1
    elif location == "exit":
        local_diameter = geometry.exit_diameter
        local_mach = performance.exit_mach
    else:
        raise ValueError(f"Unknown location: {location}")

    q_steady = bartz_heat_flux(
        chamber_pressure=inputs.chamber_pressure,
        chamber_temp=inputs.chamber_temp,
        throat_diameter=geometry.throat_diameter,
        local_diameter=local_diameter,
        characteristic_velocity=performance.cstar,
        gamma=inputs.gamma,
        molecular_weight=inputs.molecular_weight,
        local_mach=local_mach,
        wall_temp=kelvin(T_coolant + 200),  # Estimate wall temp
    )
    q_ss = q_steady.to("W/m^2").value

    # Create time array
    total_time = startup_time + steady_time
    n_steps = int(total_time / dt) + 1
    time = np.linspace(0, total_time, n_steps)

    # Create heat flux profile
    q_profile = np.zeros(n_steps)
    for i, t in enumerate(time):
        if t < startup_time:
            # Ramp phase
            if startup_profile == "ramp":
                fraction = t / startup_time
            elif startup_profile == "exponential":
                tau = startup_time / 3
                fraction = 1 - math.exp(-t / tau)
            elif startup_profile == "step":
                fraction = 1.0 if t > 0.1 else 0.0
            else:
                raise ValueError(f"Unknown startup profile: {startup_profile}")

            q_profile[i] = q_ss * fraction
        else:
            # Steady state
            q_profile[i] = q_ss

    # Solve heat conduction
    thickness_m = wall_thickness.to("m").value
    T_initial = T_coolant + 50  # Start near coolant temp

    # Check stability and adjust if needed
    alpha = mat["k"] / (mat["rho"] * mat["cp"])
    dx = thickness_m / (n_nodes - 1)
    Fo = alpha * dt / dx**2

    actual_dt = dt
    if Fo >= 0.5:
        # Reduce time step for stability
        actual_dt = 0.4 * dx**2 / alpha
        n_steps = int(total_time / actual_dt) + 1
        time = np.linspace(0, total_time, n_steps)

        # Recalculate heat flux profile
        q_profile = np.zeros(n_steps)
        for i, t in enumerate(time):
            if t < startup_time:
                if startup_profile == "ramp":
                    fraction = t / startup_time
                elif startup_profile == "exponential":
                    tau = startup_time / 3
                    fraction = 1 - math.exp(-t / tau)
                else:
                    fraction = 1.0 if t > 0.1 else 0.0
                q_profile[i] = q_ss * fraction
            else:
                q_profile[i] = q_ss

    # Run simulation
    T_inner, T_outer = _solve_1d_conduction(
        n_nodes=n_nodes,
        thickness=thickness_m,
        dt=actual_dt,
        n_steps=n_steps,
        k=mat["k"],
        rho=mat["rho"],
        cp=mat["cp"],
        q_inner=q_profile,
        T_outer=T_coolant,
        T_initial=T_initial,
    )

    # Calculate thermal margin and cumulative heat
    thermal_margin = T_max - T_inner

    # Cumulative heat absorbed (integral of q over time)
    # Heat absorbed = q * A * dt, but we'll compute per unit area
    cumulative_heat = np.cumsum(q_profile) * actual_dt

    return TransientThermalResult(
        time=time,
        wall_temp_inner=T_inner,
        wall_temp_outer=T_outer,
        heat_flux=q_profile,
        thermal_margin=thermal_margin,
        coolant_heat_absorbed=cumulative_heat,
        max_material_temp=float(T_max),
        location=location,
    )


@beartype
def simulate_shutdown(
    inputs: EngineInputs,
    performance: EnginePerformance,
    geometry: EngineGeometry,
    wall_thickness: Quantity,
    wall_material: str = "copper",
    coolant_temp: Quantity | None = None,
    max_material_temp: Quantity | None = None,
    initial_wall_temp: Quantity | None = None,
    shutdown_time: float = 0.5,
    cooling_time: float = 10.0,
    dt: float = 0.001,
    n_nodes: int = 20,
    location: str = "throat",
    coolant_continues: bool = True,
    progress: bool = False,
) -> TransientThermalResult:
    """Simulate thermal transient during engine shutdown.

    Models the wall temperature evolution after engine cutoff,
    including thermal soak-back effects.

    Args:
        inputs: Engine input parameters
        performance: Computed engine performance
        geometry: Computed engine geometry
        wall_thickness: Chamber wall thickness
        wall_material: Wall material name
        coolant_temp: Coolant inlet temperature
        max_material_temp: Material temperature limit
        initial_wall_temp: Wall temp at shutdown (default: estimates steady-state)
        shutdown_time: Time for heat flux to decay to zero [s]
        cooling_time: Time to simulate after full shutdown [s]
        dt: Time step [s]
        n_nodes: Number of spatial nodes through wall
        location: Location to analyze
        coolant_continues: Whether coolant flow continues after shutdown
        progress: Show progress bar

    Returns:
        TransientThermalResult with time histories
    """
    # Get material properties
    mat = get_material_properties(wall_material)
    T_max = max_material_temp.to("K").value if max_material_temp else mat["max_temp"]

    if coolant_temp is None:
        coolant_temp = kelvin(100.0)
    T_coolant = coolant_temp.to("K").value

    # Estimate steady-state conditions if not provided
    if initial_wall_temp is None:
        # Run a short startup simulation to get steady state
        startup_result = simulate_startup(
            inputs, performance, geometry,
            wall_thickness=wall_thickness,
            wall_material=wall_material,
            coolant_temp=coolant_temp,
            startup_time=1.0,
            steady_time=2.0,
            dt=dt,
            n_nodes=n_nodes,
            location=location,
        )
        T_init = startup_result.wall_temp_inner[-1]
        q_steady = startup_result.heat_flux[-1]
    else:
        T_init = initial_wall_temp.to("K").value
        # Estimate steady heat flux
        if location == "throat":
            local_diameter = geometry.throat_diameter
            local_mach = 1.0
        elif location == "chamber":
            local_diameter = geometry.chamber_diameter
            local_mach = 0.1
        else:
            local_diameter = geometry.exit_diameter
            local_mach = performance.exit_mach

        q_result = bartz_heat_flux(
            chamber_pressure=inputs.chamber_pressure,
            chamber_temp=inputs.chamber_temp,
            throat_diameter=geometry.throat_diameter,
            local_diameter=local_diameter,
            characteristic_velocity=performance.cstar,
            gamma=inputs.gamma,
            molecular_weight=inputs.molecular_weight,
            local_mach=local_mach,
        )
        q_steady = q_result.to("W/m^2").value

    # Create time array
    total_time = shutdown_time + cooling_time
    n_steps = int(total_time / dt) + 1
    time = np.linspace(0, total_time, n_steps)

    # Create heat flux profile (exponential decay)
    q_profile = np.zeros(n_steps)
    tau = shutdown_time / 3  # Time constant

    for i, t in enumerate(time):
        if t < shutdown_time:
            q_profile[i] = q_steady * math.exp(-t / tau)
        else:
            q_profile[i] = 0.0

    # Coolant boundary condition
    if coolant_continues:
        T_outer = T_coolant
    else:
        # Coolant stops, outer surface becomes adiabatic
        # Simplified: assume it gradually warms up
        T_outer = np.linspace(T_coolant, T_init * 0.8, n_steps)

    # Solve heat conduction
    thickness_m = wall_thickness.to("m").value

    # Check stability
    alpha = mat["k"] / (mat["rho"] * mat["cp"])
    dx = thickness_m / (n_nodes - 1)
    Fo = alpha * dt / dx**2

    actual_dt = dt
    if Fo >= 0.5:
        actual_dt = 0.4 * dx**2 / alpha
        n_steps = int(total_time / actual_dt) + 1
        time = np.linspace(0, total_time, n_steps)

        q_profile = np.zeros(n_steps)
        for i, t in enumerate(time):
            if t < shutdown_time:
                q_profile[i] = q_steady * math.exp(-t / tau)
            else:
                q_profile[i] = 0.0

        if not coolant_continues:
            T_outer = np.linspace(T_coolant, T_init * 0.8, n_steps)

    T_inner, T_outer_hist = _solve_1d_conduction(
        n_nodes=n_nodes,
        thickness=thickness_m,
        dt=actual_dt,
        n_steps=n_steps,
        k=mat["k"],
        rho=mat["rho"],
        cp=mat["cp"],
        q_inner=q_profile,
        T_outer=T_outer if isinstance(T_outer, np.ndarray) else T_outer,
        T_initial=T_init,
    )

    thermal_margin = T_max - T_inner
    cumulative_heat = np.cumsum(q_profile) * actual_dt

    return TransientThermalResult(
        time=time,
        wall_temp_inner=T_inner,
        wall_temp_outer=T_outer_hist,
        heat_flux=q_profile,
        thermal_margin=thermal_margin,
        coolant_heat_absorbed=cumulative_heat,
        max_material_temp=float(T_max),
        location=location,
    )


@beartype
def simulate_duty_cycle(
    inputs: EngineInputs,
    performance: EnginePerformance,
    geometry: EngineGeometry,
    wall_thickness: Quantity,
    burn_time: float,
    coast_time: float,
    n_cycles: int = 1,
    wall_material: str = "copper",
    coolant_temp: Quantity | None = None,
    max_material_temp: Quantity | None = None,
    dt: float = 0.001,
    n_nodes: int = 20,
    location: str = "throat",
    progress: bool = False,
) -> TransientThermalResult:
    """Simulate thermal behavior over multiple burn/coast cycles.

    Useful for analyzing pulsed engines or restart scenarios.

    Args:
        inputs: Engine input parameters
        performance: Computed engine performance
        geometry: Computed engine geometry
        wall_thickness: Chamber wall thickness
        burn_time: Duration of each burn [s]
        coast_time: Duration of coast between burns [s]
        n_cycles: Number of burn/coast cycles
        wall_material: Wall material name
        coolant_temp: Coolant inlet temperature
        max_material_temp: Material temperature limit
        dt: Time step [s]
        n_nodes: Number of spatial nodes through wall
        location: Location to analyze
        progress: Show progress bar

    Returns:
        TransientThermalResult with full time history
    """
    mat = get_material_properties(wall_material)
    T_max = max_material_temp.to("K").value if max_material_temp else mat["max_temp"]

    if coolant_temp is None:
        coolant_temp = kelvin(100.0)
    T_coolant = coolant_temp.to("K").value

    # Get steady-state heat flux
    if location == "throat":
        local_diameter = geometry.throat_diameter
        local_mach = 1.0
    elif location == "chamber":
        local_diameter = geometry.chamber_diameter
        local_mach = 0.1
    else:
        local_diameter = geometry.exit_diameter
        local_mach = performance.exit_mach

    q_result = bartz_heat_flux(
        chamber_pressure=inputs.chamber_pressure,
        chamber_temp=inputs.chamber_temp,
        throat_diameter=geometry.throat_diameter,
        local_diameter=local_diameter,
        characteristic_velocity=performance.cstar,
        gamma=inputs.gamma,
        molecular_weight=inputs.molecular_weight,
        local_mach=local_mach,
    )
    q_steady = q_result.to("W/m^2").value

    # Create time array
    cycle_time = burn_time + coast_time
    total_time = cycle_time * n_cycles
    n_steps = int(total_time / dt) + 1
    time = np.linspace(0, total_time, n_steps)

    # Create heat flux profile
    q_profile = np.zeros(n_steps)
    startup_tau = 0.5  # Quick startup ramp

    for i, t in enumerate(time):
        cycle_phase = t % cycle_time

        if cycle_phase < burn_time:
            # Burn phase with quick startup ramp
            if cycle_phase < startup_tau:
                fraction = 1 - math.exp(-cycle_phase / (startup_tau / 3))
            else:
                fraction = 1.0
            q_profile[i] = q_steady * fraction
        else:
            # Coast phase - exponential decay
            coast_phase = cycle_phase - burn_time
            q_profile[i] = q_steady * math.exp(-coast_phase / (coast_time / 5))

    # Solve
    thickness_m = wall_thickness.to("m").value
    T_initial = T_coolant + 50

    alpha = mat["k"] / (mat["rho"] * mat["cp"])
    dx = thickness_m / (n_nodes - 1)
    Fo = alpha * dt / dx**2

    actual_dt = dt
    if Fo >= 0.5:
        actual_dt = 0.4 * dx**2 / alpha
        n_steps = int(total_time / actual_dt) + 1
        time = np.linspace(0, total_time, n_steps)

        q_profile = np.zeros(n_steps)
        for i, t in enumerate(time):
            cycle_phase = t % cycle_time
            if cycle_phase < burn_time:
                if cycle_phase < startup_tau:
                    fraction = 1 - math.exp(-cycle_phase / (startup_tau / 3))
                else:
                    fraction = 1.0
                q_profile[i] = q_steady * fraction
            else:
                coast_phase = cycle_phase - burn_time
                q_profile[i] = q_steady * math.exp(-coast_phase / (coast_time / 5))

    T_inner, T_outer = _solve_1d_conduction(
        n_nodes=n_nodes,
        thickness=thickness_m,
        dt=actual_dt,
        n_steps=n_steps,
        k=mat["k"],
        rho=mat["rho"],
        cp=mat["cp"],
        q_inner=q_profile,
        T_outer=T_coolant,
        T_initial=T_initial,
    )

    thermal_margin = T_max - T_inner
    cumulative_heat = np.cumsum(q_profile) * actual_dt

    return TransientThermalResult(
        time=time,
        wall_temp_inner=T_inner,
        wall_temp_outer=T_outer,
        heat_flux=q_profile,
        thermal_margin=thermal_margin,
        coolant_heat_absorbed=cumulative_heat,
        max_material_temp=float(T_max),
        location=location,
    )

