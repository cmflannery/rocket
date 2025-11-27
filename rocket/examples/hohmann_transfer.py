#!/usr/bin/env python
"""Hohmann transfer simulation (Vacuum).

Tests the orbital mechanics and burn timing logic in isolation
by performing a transfer from LEO to a higher orbit.

Runs for 2 full orbits to show complete transfer maneuver.
"""

import numpy as np

from rocket.dynamics.rigid_body import DynamicsConfig, RigidBodyDynamics, rk4_step
from rocket.dynamics.state import State, dcm_to_quaternion
from rocket.environment.gravity import MU_EARTH, R_EARTH_EQ, GravityModel


def compute_orbital_elements(position, velocity):
    """Compute orbital elements from state vectors.

    Returns:
        apogee_alt: Apogee altitude above Earth surface [m]
        perigee_alt: Perigee altitude above Earth surface [m]
        semi_major: Semi-major axis [m]
        eccentricity: Orbital eccentricity
        time_to_apogee: Time to next apogee passage [s]
    """
    r_vec = position
    v_vec = velocity
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # Specific orbital energy
    energy = v**2 / 2 - MU_EARTH / r

    # Semi-major axis (negative for hyperbolic)
    if abs(energy) < 1e-10:
        return 1e9, r - R_EARTH_EQ, 1e9, 1.0, 0.0

    semi_major = -MU_EARTH / (2 * energy)

    # Specific angular momentum
    h_vec = np.cross(r_vec, v_vec)

    # Eccentricity
    ecc_vec = np.cross(v_vec, h_vec) / MU_EARTH - r_vec / r
    eccentricity = np.linalg.norm(ecc_vec)

    # Apogee and perigee
    if eccentricity >= 1.0:
        apogee_alt = 1e9
        perigee_alt = semi_major * (1 - eccentricity) - R_EARTH_EQ
        time_to_apogee = 0.0
    else:
        apogee_alt = semi_major * (1 + eccentricity) - R_EARTH_EQ
        perigee_alt = semi_major * (1 - eccentricity) - R_EARTH_EQ

        # Calculate time to apogee
        n = np.sqrt(MU_EARTH / semi_major**3)

        # Eccentric anomaly (E) using cos(E) = (1 - r/a) / e
        # Handle circular orbits where e ~ 0
        if eccentricity < 1e-6:
            E = 0.0
        else:
            cos_E = (1 - r / semi_major) / eccentricity
            cos_E = np.clip(cos_E, -1.0, 1.0)
            E = np.arccos(cos_E)

        # Determine sign of E based on radial velocity
        r_dot = np.dot(r_vec, v_vec)
        if r_dot < 0:
            E = 2 * np.pi - E

        # Mean anomaly
        M = E - eccentricity * np.sin(E)

        time_to_apogee = (np.pi - M) / n if np.pi >= M else (3 * np.pi - M) / n

    return apogee_alt, perigee_alt, semi_major, eccentricity, time_to_apogee


def run_simulation():
    print("=" * 70)
    print("HOHMANN TRANSFER SIMULATION (300km -> 1000km)")
    print("=" * 70)

    # Configuration
    ALT_INIT = 300e3   # Initial orbit altitude [m]
    TARGET_ALT = 1000e3  # Target orbit altitude [m]

    # 1. Setup Initial State (circular orbit at ALT_INIT)
    r_init = R_EARTH_EQ + ALT_INIT
    v_init = np.sqrt(MU_EARTH / r_init)  # Circular velocity

    # Position at equator, Velocity East (prograde)
    position = np.array([r_init, 0, 0])
    velocity = np.array([0, v_init, 0])

    # Orientation: Body X = Velocity direction, Body Z = Nadir
    body_x = velocity / np.linalg.norm(velocity)
    body_z = -position / np.linalg.norm(position)
    body_y = np.cross(body_z, body_x)

    dcm_body_to_inertial = np.column_stack([body_x, body_y, body_z])
    quaternion = dcm_to_quaternion(dcm_body_to_inertial.T)

    state = State(
        position=position,
        velocity=velocity,
        quaternion=quaternion,
        angular_velocity=np.zeros(3),
        mass=1000.0,  # 1 ton satellite
        time=0.0
    )

    # Dynamics (spherical gravity, no atmosphere)
    config = DynamicsConfig(gravity_model=GravityModel.SPHERICAL, include_atmosphere=False)
    dynamics = RigidBodyDynamics(inertia=np.eye(3)*1000, config=config)

    # Engine parameters
    thrust_force = 1000.0  # 1 kN thrust
    isp = 300.0
    mdot = thrust_force / (isp * 9.81)

    # Calculate orbital periods for timing
    # Initial orbit period
    T_init = 2 * np.pi * np.sqrt(r_init**3 / MU_EARTH)
    # Target orbit period
    r_target = R_EARTH_EQ + TARGET_ALT
    T_target = 2 * np.pi * np.sqrt(r_target**3 / MU_EARTH)
    # Transfer orbit period (semi-major axis = average of r_init and r_target)
    a_transfer = (r_init + r_target) / 2
    T_transfer = 2 * np.pi * np.sqrt(a_transfer**3 / MU_EARTH)

    print("\nOrbital Periods:")
    print(f"  Initial ({ALT_INIT/1000:.0f} km): {T_init/60:.1f} min")
    print(f"  Transfer: {T_transfer/60:.1f} min")
    print(f"  Target ({TARGET_ALT/1000:.0f} km): {T_target/60:.1f} min")

    # Run for ~2 target orbits to see full transfer
    t_max = 2.5 * T_target
    print(f"  Simulation duration: {t_max/60:.1f} min ({t_max/3600:.2f} hrs)")

    # Simulation parameters
    dt = 1.0  # 1 second timestep (fine for orbital dynamics)
    record_interval = 5  # Record every 5 seconds

    # Data storage
    times = []
    positions = []
    velocities = []
    altitudes = []
    eccentricities = []
    phases = []

    phase = 1  # 1: Burn 1 (raise apogee), 2: Coast, 3: Burn 2 (circularize), 4: Complete

    print(f"\nInitial Orbit: {ALT_INIT/1000:.1f} km circular")
    print(f"Target Orbit:  {TARGET_ALT/1000:.1f} km circular")
    print("\nStarting simulation...")

    step = 0
    while state.time < t_max:
        t = state.time
        r_mag = np.linalg.norm(state.position)
        alt = r_mag - R_EARTH_EQ

        apogee, perigee, sma, ecc, t_apogee = compute_orbital_elements(state.position, state.velocity)

        thrust_body = np.zeros(3)
        is_burning = False

        # Guidance Logic
        if phase == 1:
            # BURN 1: Raise apogee to target altitude
            if apogee < TARGET_ALT * 0.99:
                thrust_body = np.array([thrust_force, 0, 0])  # Prograde
                is_burning = True
            else:
                print(f"\n✓ Burn 1 Complete at T+{t:.1f}s ({t/60:.1f} min)")
                print(f"  Apogee raised to: {apogee/1000:.1f} km")
                print(f"  Transfer orbit: {perigee/1000:.1f} x {apogee/1000:.1f} km")
                print(f"  Eccentricity: {ecc:.4f}")
                phase = 2

        elif phase == 2:
            # COAST: Wait until approaching apogee
            # Calculate when to start circularization burn
            r_apogee = R_EARTH_EQ + apogee
            v_circ_target = np.sqrt(MU_EARTH / r_apogee)
            v_apogee_current = np.sqrt(MU_EARTH * (2/r_apogee - 1/sma))
            dv_needed = v_circ_target - v_apogee_current

            burn_time = dv_needed * state.mass / thrust_force

            # Start burn when time to apogee < half burn duration
            if t_apogee < burn_time / 2 + dt:
                print(f"\n✓ Starting Burn 2 at T+{t:.1f}s ({t/60:.1f} min)")
                print(f"  Time to apogee: {t_apogee:.1f} s")
                print(f"  Estimated burn duration: {burn_time:.1f} s")
                print(f"  Delta-V needed: {dv_needed:.1f} m/s")
                phase = 3

        elif phase == 3:
            # BURN 2: Circularize at apogee
            target_sma = R_EARTH_EQ + TARGET_ALT

            # Burn until semi-major axis matches target
            if sma < target_sma * 0.999:
                thrust_body = np.array([thrust_force, 0, 0])  # Prograde
                is_burning = True
            else:
                print(f"\n✓ Burn 2 Complete at T+{t:.1f}s ({t/60:.1f} min)")
                print(f"  Final orbit: {perigee/1000:.1f} x {apogee/1000:.1f} km")
                print(f"  Eccentricity: {ecc:.5f}")
                print(f"  Target was: {TARGET_ALT/1000:.0f} km circular")
                phase = 4

        elif phase == 4:
            # COMPLETE: Continue propagating to show stable orbit
            pass

        # Orientation Control: Perfect prograde pointing
        v_unit = state.velocity / np.linalg.norm(state.velocity)
        body_x = v_unit
        body_z = -state.position / np.linalg.norm(state.position)
        body_y = np.cross(body_z, body_x)
        body_y /= np.linalg.norm(body_y)
        body_z = np.cross(body_x, body_y)

        dcm = np.column_stack([body_x, body_y, body_z])
        state.quaternion = dcm_to_quaternion(dcm.T)

        # Integrate dynamics
        state_dot = dynamics.derivatives(
            state=state,
            thrust_body=thrust_body,
            moment_body=np.zeros(3),
            mass_rate=-mdot if is_burning else 0.0
        )
        state = rk4_step(state, dt, lambda s, sd=state_dot: sd)

        # Record data at intervals
        if step % record_interval == 0:
            times.append(t)
            positions.append(state.position.copy())
            velocities.append(state.velocity.copy())
            altitudes.append(alt)
            eccentricities.append(ecc)
            phases.append(phase)

        step += 1

    # Final summary
    print(f"\n{'='*70}")
    print("SIMULATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total simulation time: {state.time/60:.1f} min")
    print(f"Data points recorded: {len(times)}")

    final_apogee, final_perigee, _, final_ecc, _ = compute_orbital_elements(
        state.position, state.velocity
    )
    print("\nFinal Orbit:")
    print(f"  Perigee: {final_perigee/1000:.1f} km")
    print(f"  Apogee:  {final_apogee/1000:.1f} km")
    print(f"  Eccentricity: {final_ecc:.6f}")

    return {
        'times': np.array(times),
        'positions': np.array(positions),
        'velocities': np.array(velocities),
        'altitudes': np.array(altitudes),
        'eccentricities': np.array(eccentricities),
        'phases': np.array(phases),
        'initial_altitude': ALT_INIT,
        'target_altitude': TARGET_ALT,
    }


if __name__ == "__main__":
    from rocket.orbital_plotting import (
        plot_hohmann_animation,
        plot_orbital_dashboard,
        plot_orbital_transfer_dashboard,
    )

    data = run_simulation()

    # Generate all three visualizations
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)

    # 1. Animated 3D visualization
    print("\n1. Animated 3D visualization...")
    fig_3d = plot_hohmann_animation(
        data,
        title="🛰️ Hohmann Transfer: 300km → 1000km",
        target_altitude=data['target_altitude'],
        initial_altitude=data['initial_altitude']
    )
    output_3d = "outputs/hohmann_transfer_3d.html"
    fig_3d.write_html(output_3d)
    print(f"   Saved to: {output_3d}")

    # 2. Static orbital dashboard (general purpose)
    print("\n2. Static orbital dashboard...")
    fig_static = plot_orbital_dashboard(
        data,
        title="Hohmann Transfer - Orbital View",
        target_altitude=data['target_altitude']
    )
    output_static = "outputs/hohmann_transfer_static.html"
    fig_static.write_html(output_static)
    print(f"   Saved to: {output_static}")

    # 3. Telemetry dashboard with phase breakdown
    print("\n3. Telemetry dashboard...")
    fig_telemetry = plot_orbital_transfer_dashboard(
        data,
        title="Hohmann Transfer - Telemetry",
        target_altitude=data['target_altitude'],
        initial_altitude=data['initial_altitude']
    )
    output_telemetry = "outputs/hohmann_transfer_telemetry.html"
    fig_telemetry.write_html(output_telemetry)
    print(f"   Saved to: {output_telemetry}")

    print("\n" + "="*50)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*50)
    print("\nGenerated files:")
    print(f"  • {output_3d} (animated 3D)")
    print(f"  • {output_static} (static dashboard)")
    print(f"  • {output_telemetry} (telemetry plots)")

    print("\nOpening visualizations in browser...")
    fig_3d.show()
    fig_static.show()
