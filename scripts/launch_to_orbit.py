#!/usr/bin/env python
"""Example: Launch to orbit using the new simulation architecture.

This script demonstrates the clean separation between:
- Simulation infrastructure (rocket/) - the "plant" / truth model
- Flight software (flight/) - GNC algorithms

The simulation loop follows real flight software patterns:
1. Read state (sensors in reality, truth in sim)
2. Run guidance (compute desired trajectory)
3. Run control (compute actuator commands)
4. Step simulation (actuators move the vehicle)

Usage:
    uv run python scripts/launch_to_orbit.py
"""

import numpy as np

from flight.guidance import GravityTurnGuidance
from rocket.environment.gravity import GravityModel
from rocket.orbital import OMEGA_EARTH, compute_orbital_elements, eci_to_lla, launch_azimuth
from rocket.simulation import SimConfig, Simulator


def eci_to_ecef_velocity(position: np.ndarray, velocity_eci: np.ndarray) -> np.ndarray:
    """Convert ECI velocity to ECEF (velocity relative to rotating Earth).

    This removes the Earth rotation component, giving velocity relative to ground.
    More intuitive for launch visualization.
    """
    omega = np.array([0.0, 0.0, OMEGA_EARTH])
    v_rotation = np.cross(omega, position)
    return velocity_eci - v_rotation


def run_launch():
    """Run a launch simulation from Cape Canaveral."""
    print("=" * 60)
    print("LAUNCH TO ORBIT SIMULATION")
    print("=" * 60)

    # =========================================================================
    # Mission Setup
    # =========================================================================
    LAUNCH_LAT = 28.5       # Cape Canaveral [deg]
    LAUNCH_LON = -80.6      # [deg]
    TARGET_ALT = 200e3      # 200 km target altitude [m]
    TARGET_INC = 28.5       # Match launch latitude for efficiency [deg]

    # Vehicle parameters (Falcon 9-like first stage)
    VEHICLE_MASS = 500000.0  # Initial mass [kg]
    DRY_MASS = 25000.0       # Dry mass [kg]
    THRUST = 7600000.0       # Thrust [N] (7.6 MN - 9 Merlin engines)
    ISP = 282.0              # Sea-level specific impulse [s]

    # Compute mass flow rate
    g0 = 9.80665
    mdot = THRUST / (ISP * g0)

    # Compute delta-V budget
    dv = ISP * g0 * np.log(VEHICLE_MASS / DRY_MASS)
    print("\nMission Parameters:")
    print(f"  Launch site: {LAUNCH_LAT}°N, {LAUNCH_LON}°E")
    print(f"  Target altitude: {TARGET_ALT/1000:.0f} km")
    print(f"  Target inclination: {TARGET_INC}°")
    print("\nVehicle:")
    print(f"  Initial mass: {VEHICLE_MASS:.0f} kg")
    print(f"  Dry mass: {DRY_MASS:.0f} kg")
    print(f"  Thrust: {THRUST/1000:.0f} kN")
    print(f"  Isp: {ISP:.0f} s")
    print(f"  Delta-V budget: {dv:.0f} m/s")
    print(f"  Burn time: {(VEHICLE_MASS - DRY_MASS) / mdot:.0f} s")

    # =========================================================================
    # Initialize Simulation (the "plant")
    # =========================================================================
    print("\nInitializing simulation...")

    # Compute launch azimuth for target inclination
    az = launch_azimuth(np.radians(LAUNCH_LAT), np.radians(TARGET_INC))
    print(f"  Launch azimuth: {np.degrees(az):.1f}° from North")

    # Create simulator
    config = SimConfig(gravity_model=GravityModel.SPHERICAL, include_atmosphere=True)

    # Approximate inertia for a 70m tall, 3.7m diameter rocket (Falcon 9 scale)
    Ixx = VEHICLE_MASS * 70**2 / 12
    Izz = VEHICLE_MASS * 1.85**2 / 2
    inertia = np.diag([Ixx, Ixx, Izz])

    sim = Simulator.from_launch_pad(
        latitude=LAUNCH_LAT,
        longitude=LAUNCH_LON,
        heading=np.degrees(az),
        vehicle_mass=VEHICLE_MASS,
        inertia=inertia,
        config=config,
    )

    # Initial state check
    state = sim.get_state()
    coords = eci_to_lla(state.position)
    print(f"  Initial position: {coords.latitude_deg:.2f}°N, {coords.longitude_deg:.2f}°E")
    print(f"  Initial altitude: {coords.altitude:.0f} m")
    print(f"  Initial velocity: {np.linalg.norm(state.velocity):.0f} m/s (Earth rotation)")

    # =========================================================================
    # Initialize Flight Software (GNC)
    # =========================================================================
    print("\nInitializing GNC...")

    guidance = GravityTurnGuidance(
        target_altitude=TARGET_ALT,
        max_thrust=THRUST,
        vertical_rise_time=30.0,        # Longer vertical rise to build vertical velocity
        pitch_kick_duration=30.0,       # Gradual pitch over
        pitch_kick_angle=np.radians(25.0),  # Pitch over significantly
    )

    # =========================================================================
    # Simulation Loop
    # =========================================================================
    print("\nRunning simulation...")
    print("-" * 60)

    dt = 0.02  # 50 Hz
    t_max = 1800.0  # 30 minutes max (enough for ~1/3 orbit)
    record_interval = 10  # Record every 10 steps (5 Hz) for dashboard

    # Data collection for dashboard
    times = []
    positions = []
    velocities = []
    altitudes = []
    phases = []
    inclinations = []
    eccentricities = []

    step = 0
    last_print_time = -10.0
    target_reached = False
    target_reached_time = None

    while sim.time < t_max:
        # ---------------------------------------------------------------------
        # 1. Read State (Navigation)
        # ---------------------------------------------------------------------
        state = sim.get_state()
        env = sim.get_environment()

        # ---------------------------------------------------------------------
        # 2. Guidance - compute desired trajectory
        # ---------------------------------------------------------------------
        cmd = guidance.compute(state)

        # Check for engine cutoff (out of propellant or target reached)
        if state.mass <= DRY_MASS:
            cmd = cmd._replace(thrust=0.0)

        # ---------------------------------------------------------------------
        # 3. Control - set attitude directly (simplified for demo)
        # ---------------------------------------------------------------------
        # Instead of using gimbal control, directly set the target attitude
        # This simulates perfect attitude control for now
        if cmd.target_attitude is not None:
            sim.set_attitude(cmd.target_attitude)

        # ---------------------------------------------------------------------
        # 4. Step Simulation (apply commands to plant)
        # ---------------------------------------------------------------------
        mass_rate = -mdot if cmd.thrust > 0 else 0.0

        # Apply thrust along body X (no gimbal for this demo)
        sim.step(
            thrust=cmd.thrust,
            gimbal=None,
            mass_rate=mass_rate,
            dt=dt,
        )

        # ---------------------------------------------------------------------
        # Record Data for Dashboard
        # ---------------------------------------------------------------------
        if step % record_interval == 0:
            elements = compute_orbital_elements(state.position, state.velocity)
            # Convert velocity to ECEF (relative to ground) for intuitive display
            velocity_ecef = eci_to_ecef_velocity(state.position, state.velocity)
            times.append(state.time)
            positions.append(state.position.copy())
            velocities.append(velocity_ecef)  # ECEF velocity for display
            altitudes.append(env.altitude)
            phases.append(guidance.phase)
            inclinations.append(np.degrees(elements.inclination))
            eccentricities.append(elements.eccentricity)

        # ---------------------------------------------------------------------
        # Progress Output
        # ---------------------------------------------------------------------
        if sim.time - last_print_time >= 10.0:
            elements = compute_orbital_elements(state.position, state.velocity)
            v_ecef = eci_to_ecef_velocity(state.position, state.velocity)
            speed_ecef = np.linalg.norm(v_ecef)

            print(
                f"T+{sim.time:5.0f}s | "
                f"Alt: {env.altitude/1000:6.1f} km | "
                f"Speed: {speed_ecef:6.0f} m/s | "
                f"Phase: {guidance.phase_name:12s} | "
                f"Mass: {state.mass:6.0f} kg | "
                f"Apo: {elements.apogee_alt/1000:6.1f} km"
            )
            last_print_time = sim.time

        # Check termination conditions
        if guidance.is_complete(state) and not target_reached:
            print(f"\n✓ Target altitude reached at T+{sim.time:.1f}s")
            target_reached = True
            target_reached_time = state.time
            # Continue propagating to see orbit/trajectory

        if env.altitude < -100 and state.time > 60:  # Allow initial negative altitude
            print(f"\n✗ Impact at T+{sim.time:.1f}s")
            break

        # Stop after 10 minutes of coast past target
        if target_reached and state.time > target_reached_time + 600:
            print(f"\nCoast phase complete at T+{sim.time:.1f}s")
            break

        step += 1

    # =========================================================================
    # Results
    # =========================================================================
    print("-" * 60)
    print("\nFINAL STATE:")

    state = sim.get_state()
    elements = compute_orbital_elements(state.position, state.velocity)
    coords = eci_to_lla(state.position)

    print(f"  Time: {sim.time:.1f} s")
    print(f"  Position: {coords.latitude_deg:.2f}°N, {coords.longitude_deg:.2f}°E")
    print(f"  Altitude: {elements.apogee_alt/1000:.1f} km (apogee), {elements.perigee_alt/1000:.1f} km (perigee)")
    print(f"  Inclination: {np.degrees(elements.inclination):.1f}°")
    print(f"  Eccentricity: {elements.eccentricity:.4f}")
    print(f"  Speed: {state.speed:.0f} m/s")
    print(f"  Remaining mass: {state.mass:.0f} kg")

    # Check success
    if elements.perigee_alt > 100e3:
        print("\n✓ ORBIT ACHIEVED!")
    else:
        print(f"\n✗ Suborbital - perigee at {elements.perigee_alt/1000:.1f} km")
        print("\n  Note: The simple gravity turn guidance doesn't achieve circular orbit.")
        print("  This demonstrates the simulation infrastructure is working correctly.")
        print("  Achieving proper orbit insertion requires more sophisticated guidance")
        print("  (e.g., pitch programming, throttle modulation, or optimal control).")

    # Prepare trajectory data for visualization
    trajectory_data = {
        'times': np.array(times),
        'positions': np.array(positions),
        'velocities': np.array(velocities),
        'altitudes': np.array(altitudes),
        'phases': np.array(phases),
        'inclinations': np.array(inclinations),
        'eccentricities': np.array(eccentricities),
        'target_altitude': TARGET_ALT,
    }

    return sim, trajectory_data


if __name__ == "__main__":
    from rocket.orbital_plotting import plot_launch_dashboard, plot_orbital_dashboard

    sim, data = run_launch()

    # Phase names for the dashboard
    phase_names = {
        1: 'Vertical Rise',
        2: 'Pitch Kick',
        3: 'Gravity Turn',
        4: 'Insertion',
    }

    print("\n" + "=" * 60)
    print("GENERATING DASHBOARD")
    print("=" * 60)

    # Generate launch telemetry dashboard
    print("\nCreating launch telemetry dashboard...")
    fig_telemetry = plot_launch_dashboard(
        data,
        title="🚀 Launch to Orbit - Telemetry",
        phase_names=phase_names,
    )
    output_telemetry = "outputs/launch_telemetry.html"
    fig_telemetry.write_html(output_telemetry)
    print(f"  Saved to: {output_telemetry}")

    # Generate orbital dashboard
    print("\nCreating orbital dashboard...")
    fig_orbital = plot_orbital_dashboard(
        data,
        title="🚀 Launch to Orbit - Orbital View",
        target_altitude=data['target_altitude'],
    )
    output_orbital = "outputs/launch_orbital.html"
    fig_orbital.write_html(output_orbital)
    print(f"  Saved to: {output_orbital}")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  • {output_telemetry} (telemetry dashboard)")
    print(f"  • {output_orbital} (orbital view)")

    print("\nOpening dashboards in browser...")
    fig_telemetry.show()
    fig_orbital.show()

