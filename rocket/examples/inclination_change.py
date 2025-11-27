#!/usr/bin/env python
"""Orbital inclination change simulation.

Demonstrates a plane change maneuver where a satellite changes
its orbital inclination. This is one of the most expensive
maneuvers in terms of delta-V.

The maneuver is performed at the ascending/descending node where
the velocity vector is parallel to the equatorial plane.
"""

import numpy as np

from rocket.dynamics.rigid_body import DynamicsConfig, RigidBodyDynamics, rk4_step
from rocket.dynamics.state import State, dcm_to_quaternion
from rocket.environment.gravity import MU_EARTH, R_EARTH_EQ, GravityModel


def compute_orbital_elements(position, velocity):
    """Compute orbital elements from state vectors."""
    r_vec = position
    v_vec = velocity
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # Specific orbital energy
    energy = v**2 / 2 - MU_EARTH / r
    if abs(energy) < 1e-10:
        return 1e9, r - R_EARTH_EQ, 1e9, 1.0, 0.0, 0.0

    semi_major = -MU_EARTH / (2 * energy)

    # Angular momentum vector
    h_vec = np.cross(r_vec, v_vec)
    h_mag = np.linalg.norm(h_vec)

    # Inclination (angle between h and z-axis)
    inclination = np.degrees(np.arccos(h_vec[2] / h_mag))

    # Eccentricity
    ecc_vec = np.cross(v_vec, h_vec) / MU_EARTH - r_vec / r
    eccentricity = np.linalg.norm(ecc_vec)

    # Apogee and perigee
    if eccentricity >= 1.0:
        apogee_alt = 1e9
        perigee_alt = semi_major * (1 - eccentricity) - R_EARTH_EQ
    else:
        apogee_alt = semi_major * (1 + eccentricity) - R_EARTH_EQ
        perigee_alt = semi_major * (1 - eccentricity) - R_EARTH_EQ

    return apogee_alt, perigee_alt, semi_major, eccentricity, inclination, h_mag


def run_simulation():
    print("=" * 70)
    print("INCLINATION CHANGE SIMULATION")
    print("=" * 70)

    # Configuration
    ORBIT_ALT = 400e3  # 400 km circular orbit (ISS-like)
    INITIAL_INC = 28.5  # degrees (Cape Canaveral launch)
    TARGET_INC = 51.6   # degrees (ISS inclination)

    # Calculate delta-V required for plane change
    # dV = 2 * V * sin(delta_i / 2)
    r_orbit = R_EARTH_EQ + ORBIT_ALT
    v_orbit = np.sqrt(MU_EARTH / r_orbit)
    delta_inc = np.radians(TARGET_INC - INITIAL_INC)
    dv_required = 2 * v_orbit * np.sin(abs(delta_inc) / 2)

    print("\nManeuver Parameters:")
    print(f"  Orbit Altitude: {ORBIT_ALT/1000:.0f} km")
    print(f"  Orbital Velocity: {v_orbit:.0f} m/s")
    print(f"  Initial Inclination: {INITIAL_INC:.1f}°")
    print(f"  Target Inclination: {TARGET_INC:.1f}°")
    print(f"  Inclination Change: {TARGET_INC - INITIAL_INC:.1f}°")
    print(f"  Required Delta-V: {dv_required:.0f} m/s")

    # Initial state: Circular orbit at INITIAL_INC inclination
    # Position at ascending node (where orbit crosses equator going north)
    inc_rad = np.radians(INITIAL_INC)

    # At ascending node: position is in equatorial plane, velocity has Z component
    position = np.array([r_orbit, 0, 0])

    # Velocity perpendicular to position, tilted by inclination
    # v = v_orbit * (cos(i) * y_hat + sin(i) * z_hat)
    velocity = np.array([0, v_orbit * np.cos(inc_rad), v_orbit * np.sin(inc_rad)])

    # Orientation: Body X along velocity
    body_x = velocity / np.linalg.norm(velocity)
    body_z = -position / np.linalg.norm(position)
    body_y = np.cross(body_z, body_x)
    body_y /= np.linalg.norm(body_y)
    body_z = np.cross(body_x, body_y)

    dcm_body_to_inertial = np.column_stack([body_x, body_y, body_z])
    quaternion = dcm_to_quaternion(dcm_body_to_inertial.T)

    state = State(
        position=position,
        velocity=velocity,
        quaternion=quaternion,
        angular_velocity=np.zeros(3),
        mass=5000.0,  # 5 ton spacecraft
        time=0.0
    )

    # Verify initial inclination
    _, _, _, _, init_inc, _ = compute_orbital_elements(state.position, state.velocity)
    print(f"\n  Computed Initial Inclination: {init_inc:.2f}°")

    # Dynamics
    config = DynamicsConfig(gravity_model=GravityModel.SPHERICAL, include_atmosphere=False)
    dynamics = RigidBodyDynamics(inertia=np.eye(3)*5000, config=config)

    # Engine parameters (high-thrust for impulsive maneuver)
    thrust_force = 50000.0  # 50 kN
    isp = 320.0
    mdot = thrust_force / (isp * 9.81)

    # Orbital period
    T_orbit = 2 * np.pi * np.sqrt(r_orbit**3 / MU_EARTH)
    print(f"  Orbital Period: {T_orbit/60:.1f} min")

    # Simulation time: 2 orbits
    t_max = 2.0 * T_orbit
    dt = 1.0
    record_interval = 5

    # Data storage
    times = []
    positions = []
    velocities = []
    altitudes = []
    inclinations = []
    eccentricities = []
    phases = []

    # Phase: 1=Pre-burn, 2=Burn, 3=Post-burn
    phase = 1
    burn_started = False

    # Burn at the ascending node (when Z crosses zero going positive)
    # We'll wait until we're back at the ascending node after half an orbit

    print("\nStarting simulation...")
    print("  Will burn at ascending node (Z=0, dZ/dt > 0)")

    step = 0
    prev_z = state.position[2]

    while state.time < t_max:
        t = state.time
        r_mag = np.linalg.norm(state.position)
        alt = r_mag - R_EARTH_EQ

        apogee, perigee, sma, ecc, inc, h_mag = compute_orbital_elements(
            state.position, state.velocity
        )

        thrust_body = np.zeros(3)
        is_burning = False

        # Detect ascending node crossing (Z goes from negative to positive)
        current_z = state.position[2]
        z_velocity = state.velocity[2]

        # Guidance Logic
        if phase == 1:
            # Wait for ascending node
            # Ascending node: Z ~ 0 and dZ/dt > 0
            if prev_z < 0 and current_z >= 0 and not burn_started and t > T_orbit * 0.4:
                print(f"\n✓ Ascending Node Detected at T+{t:.1f}s ({t/60:.1f} min)")
                print(f"  Position Z: {current_z:.0f} m")
                print(f"  Velocity Z: {z_velocity:.0f} m/s")
                print(f"  Current Inclination: {inc:.2f}°")
                phase = 2
                burn_started = True

        elif phase == 2:
            # Perform inclination change burn
            # For a pure plane change at ascending node:
            # - Thrust must be perpendicular to velocity (to avoid changing speed)
            # - Thrust should be in the direction that rotates the orbital plane

            if inc < TARGET_INC - 0.1:
                # Velocity unit vector
                v_unit = state.velocity / np.linalg.norm(state.velocity)

                # At the ascending node, the velocity is in the Y-Z plane
                # To increase inclination, we need to rotate the velocity vector
                # toward higher Z. This means thrusting in the +Z direction
                # (perpendicular to the current velocity)

                # The out-of-plane direction at this point
                # Cross product of r and v gives the angular momentum direction
                # We want to thrust perpendicular to v, in the plane containing v and Z-axis

                # Simple approach: thrust purely in +Z direction (works at ascending node)
                # This is not perfectly perpendicular to v, but close enough
                # For a more accurate simulation, we'd compute the exact perpendicular

                # Better: thrust perpendicular to v, toward +Z
                # Project Z onto the plane perpendicular to v
                z_axis = np.array([0, 0, 1])
                z_perp = z_axis - np.dot(z_axis, v_unit) * v_unit
                if np.linalg.norm(z_perp) > 0.01:
                    thrust_dir_inertial = z_perp / np.linalg.norm(z_perp)
                else:
                    thrust_dir_inertial = z_axis

                # Transform to body frame
                dcm_inertial_to_body = state.dcm_inertial_to_body
                thrust_dir_body = dcm_inertial_to_body @ thrust_dir_inertial

                # Apply thrust
                thrust_body = thrust_force * thrust_dir_body
                is_burning = True
            else:
                print(f"\n✓ Burn Complete at T+{t:.1f}s ({t/60:.1f} min)")
                print(f"  Final Inclination: {inc:.2f}°")
                print(f"  Target was: {TARGET_INC:.1f}°")
                print(f"  Burn Duration: {t - (step - sum(1 for p in phases if p == 2)) * dt:.1f}s")
                phase = 3

        elif phase == 3:
            # Coast in new orbit
            pass

        prev_z = current_z

        # Orientation Control: Prograde pointing (for visualization)
        v_unit = state.velocity / np.linalg.norm(state.velocity)
        body_x = v_unit
        body_z = -state.position / np.linalg.norm(state.position)
        body_y = np.cross(body_z, body_x)
        if np.linalg.norm(body_y) > 0.01:
            body_y /= np.linalg.norm(body_y)
            body_z = np.cross(body_x, body_y)
            dcm = np.column_stack([body_x, body_y, body_z])
            state.quaternion = dcm_to_quaternion(dcm.T)

        # Integrate
        state_dot = dynamics.derivatives(
            state=state,
            thrust_body=thrust_body,
            moment_body=np.zeros(3),
            mass_rate=-mdot if is_burning else 0.0
        )
        state = rk4_step(state, dt, lambda s, sd=state_dot: sd)

        # Record data
        if step % record_interval == 0:
            times.append(t)
            positions.append(state.position.copy())
            velocities.append(state.velocity.copy())
            altitudes.append(alt)
            inclinations.append(inc)
            eccentricities.append(ecc)
            phases.append(phase)

        step += 1

    # Final summary
    print(f"\n{'='*70}")
    print("SIMULATION COMPLETE")
    print(f"{'='*70}")

    final_apogee, final_perigee, _, final_ecc, final_inc, _ = compute_orbital_elements(
        state.position, state.velocity
    )
    print("\nFinal Orbit:")
    print(f"  Perigee: {final_perigee/1000:.1f} km")
    print(f"  Apogee:  {final_apogee/1000:.1f} km")
    print(f"  Inclination: {final_inc:.2f}°")
    print(f"  Eccentricity: {final_ecc:.6f}")

    inc_change = final_inc - INITIAL_INC
    print(f"\nInclination Change: {inc_change:+.2f}°")

    return {
        'times': np.array(times),
        'positions': np.array(positions),
        'velocities': np.array(velocities),
        'altitudes': np.array(altitudes),
        'inclinations': np.array(inclinations),
        'eccentricities': np.array(eccentricities),
        'phases': np.array(phases),
        'initial_inclination': INITIAL_INC,
        'target_inclination': TARGET_INC,
        'orbit_altitude': ORBIT_ALT,
    }


if __name__ == "__main__":
    from rocket.orbital_plotting import plot_orbit_animation, plot_orbital_dashboard

    data = run_simulation()

    # Generate visualizations
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)

    # 1. Animated 3D visualization with reference orbits
    print("\n1. Animated 3D visualization...")

    # Create reference orbits at initial and target inclinations
    reference_orbits = [
        {
            'altitude': data['orbit_altitude'],
            'inclination': data['initial_inclination'],
            'color': 'rgba(0,255,255,0.4)',
            'name': f"Initial ({data['initial_inclination']:.1f}° inc)"
        },
        {
            'altitude': data['orbit_altitude'],
            'inclination': data['target_inclination'],
            'color': 'rgba(255,100,100,0.4)',
            'name': f"Target ({data['target_inclination']:.1f}° inc)"
        }
    ]

    # Custom phase names for inclination change
    phase_names = {
        1: 'Coast to Node',
        2: 'Plane Change Burn',
        3: 'Final Orbit'
    }

    fig_3d = plot_orbit_animation(
        data,
        title=f"🛰️ Inclination Change: {data['initial_inclination']:.1f}° → {data['target_inclination']:.1f}°",
        reference_orbits=reference_orbits,
        show_earth_axes=True,
        phase_names=phase_names
    )
    output_3d = "outputs/inclination_change_3d.html"
    fig_3d.write_html(output_3d)
    print(f"   Saved to: {output_3d}")

    # 2. Static dashboard
    print("\n2. Static orbital dashboard...")
    fig_static = plot_orbital_dashboard(
        data,
        title="Inclination Change Maneuver"
    )
    output_static = "outputs/inclination_change_static.html"
    fig_static.write_html(output_static)
    print(f"   Saved to: {output_static}")

    print("\n" + "="*50)
    print("VISUALIZATIONS COMPLETE")
    print("="*50)
    print("\nGenerated files:")
    print(f"  • {output_3d} (animated 3D)")
    print(f"  • {output_static} (static dashboard)")

    print("\nOpening 3D visualization...")
    fig_3d.show()

