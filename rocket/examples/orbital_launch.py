#!/usr/bin/env python
"""Orbital launch simulation: Cape Canaveral to ISS orbit.

Simulates a two-stage rocket launch from Cape Canaveral (28.5°N)
to the International Space Station orbit (51.6° inclination, 400 km).

Uses Earth-Centered Inertial (ECI) coordinates with:
- Spherical gravity model for proper orbital mechanics
- Realistic atmosphere model
- Two-stage vehicle with staging
- Gravity turn trajectory

Target orbit: 400 km circular, 51.6° inclination
"""

import numpy as np

from rocket import EngineInputs, design_engine
from rocket.dynamics.rigid_body import DynamicsConfig, RigidBodyDynamics, rk4_step
from rocket.dynamics.state import State, dcm_to_quaternion
from rocket.environment.atmosphere import Atmosphere
from rocket.environment.gravity import MU_EARTH, R_EARTH_EQ, GravityModel, orbital_velocity
from rocket.propulsion.throttle_model import ThrottleModel
from rocket.units import kilonewtons, megapascals
from rocket.vehicle.aerodynamics import SimpleAero


# =============================================================================
# Helper Functions
# =============================================================================

def eci_to_lla(position):
    """Convert ECI position to latitude, longitude, altitude."""
    x, y, z = position
    r = np.linalg.norm(position)
    lat = np.arcsin(z / r) if r > 0 else 0
    lon = np.arctan2(y, x)
    alt = r - R_EARTH_EQ
    return np.degrees(lat), np.degrees(lon), alt


def compute_orbital_elements(position, velocity):
    """Compute orbital elements from state vectors.

    Returns:
        apogee_alt, perigee_alt, semi_major, eccentricity, inclination, time_to_apogee
    """
    r_vec = position
    v_vec = velocity
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # Specific orbital energy
    energy = v**2 / 2 - MU_EARTH / r
    if abs(energy) < 1e-10:
        return 1e9, r - R_EARTH_EQ, 1e9, 1.0, 0.0, 0.0

    semi_major = -MU_EARTH / (2 * energy)

    # Angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h_mag = np.linalg.norm(h_vec)

    # Inclination
    inclination = np.degrees(np.arccos(np.clip(h_vec[2] / h_mag, -1, 1)))

    # Eccentricity
    ecc_vec = np.cross(v_vec, h_vec) / MU_EARTH - r_vec / r
    eccentricity = np.linalg.norm(ecc_vec)

    if eccentricity >= 1.0:
        return 1e9, semi_major * (1 - eccentricity) - R_EARTH_EQ, semi_major, eccentricity, inclination, 0.0

    apogee_alt = semi_major * (1 + eccentricity) - R_EARTH_EQ
    perigee_alt = semi_major * (1 - eccentricity) - R_EARTH_EQ

    # Time to apogee
    n = np.sqrt(MU_EARTH / semi_major**3)
    if eccentricity < 1e-6:
        time_to_apogee = 0.0
    else:
        cos_E = np.clip((1 - r / semi_major) / eccentricity, -1.0, 1.0)
        E = np.arccos(cos_E)
        if np.dot(r_vec, v_vec) < 0:
            E = 2 * np.pi - E
        M = E - eccentricity * np.sin(E)
        time_to_apogee = (np.pi - M) / n if np.pi >= M else (3 * np.pi - M) / n

    return apogee_alt, perigee_alt, semi_major, eccentricity, inclination, time_to_apogee


# =============================================================================
# Main Simulation
# =============================================================================

def run_orbital_launch():
    """Run orbital launch simulation from Cape Canaveral to ISS orbit."""
    print("=" * 70)
    print("ORBITAL LAUNCH SIMULATION")
    print("Cape Canaveral → ISS Orbit (400 km, 51.6°)")
    print("=" * 70)

    # =========================================================================
    # Mission Parameters
    # =========================================================================
    TARGET_ALT = 400e3       # 400 km (ISS altitude)
    TARGET_INC = 51.6        # ISS inclination (degrees)
    LAUNCH_LAT = 28.5        # Cape Canaveral latitude
    LAUNCH_LON = -80.6       # Cape Canaveral longitude

    # Launch azimuth for target inclination
    # For inclination > latitude, we can reach it by launching northeast or southeast
    # sin(azimuth) = cos(inclination) / cos(latitude) gives the azimuth from East
    # We want azimuth from North
    cos_inc = np.cos(np.radians(TARGET_INC))
    cos_lat = np.cos(np.radians(LAUNCH_LAT))
    sin_az = cos_inc / cos_lat
    # Azimuth from North: 90° - arcsin(sin_az) for ascending pass (northeast)
    launch_azimuth = 90 - np.degrees(np.arcsin(np.clip(sin_az, -1, 1)))
    # This gives ~44° for ISS from Cape Canaveral

    print(f"\nMission Parameters:")
    print(f"  Target Altitude: {TARGET_ALT/1000:.0f} km")
    print(f"  Target Inclination: {TARGET_INC:.1f}°")
    print(f"  Launch Site: {LAUNCH_LAT:.1f}°N, {LAUNCH_LON:.1f}°E")
    print(f"  Launch Azimuth: {launch_azimuth:.1f}° (from North)")

    # =========================================================================
    # Vehicle Configuration
    # =========================================================================
    print("\nVehicle Configuration:")

    # Stage 1 Engine (Merlin-class)
    s1_inputs = EngineInputs.from_propellants(
        oxidizer="LOX", fuel="RP1",
        thrust=kilonewtons(900),
        chamber_pressure=megapascals(9.7),
        mixture_ratio=2.36,
        name="Stage 1 Engine",
    )
    s1_perf, _ = design_engine(s1_inputs)

    # Stage 2 Engine (Vacuum optimized)
    s2_inputs = EngineInputs.from_propellants(
        oxidizer="LOX", fuel="RP1",
        thrust=kilonewtons(100),
        chamber_pressure=megapascals(6.0),
        mixture_ratio=2.36,
        name="Stage 2 Engine",
    )
    s2_perf, _ = design_engine(s2_inputs)

    # Mass budget
    s1_dry = 3000.0    # kg
    s1_prop = 28000.0  # kg
    s2_dry = 500.0     # kg
    s2_prop = 8000.0   # kg
    payload = 500.0    # kg

    total_mass = s1_dry + s1_prop + s2_dry + s2_prop + payload
    s2_mass = s2_dry + s2_prop + payload

    # Delta-V budget
    ve1 = s1_perf.isp_vac.value * 9.81
    ve2 = s2_perf.isp_vac.value * 9.81
    dv1 = ve1 * np.log(total_mass / (total_mass - s1_prop))
    dv2 = ve2 * np.log(s2_mass / (s2_mass - s2_prop))

    print(f"  Stage 1: {s1_prop:.0f} kg prop, Isp={s1_perf.isp_vac.value:.0f}s, ΔV={dv1:.0f} m/s")
    print(f"  Stage 2: {s2_prop:.0f} kg prop, Isp={s2_perf.isp_vac.value:.0f}s, ΔV={dv2:.0f} m/s")
    print(f"  Total: {total_mass:.0f} kg, ΔV={dv1+dv2:.0f} m/s")
    print(f"  T/W at liftoff: {900000 / (total_mass * 9.81):.2f}")

    v_orbital = orbital_velocity(TARGET_ALT)
    print(f"  Target orbital velocity: {v_orbital:.0f} m/s")

    # =========================================================================
    # Initial State (on launch pad)
    # =========================================================================
    print("\nInitializing on launch pad...")

    lat_rad = np.radians(LAUNCH_LAT)
    lon_rad = np.radians(LAUNCH_LON)
    azimuth_rad = np.radians(launch_azimuth)

    # Position on Earth's surface in ECI
    r = R_EARTH_EQ
    position = np.array([
        r * np.cos(lat_rad) * np.cos(lon_rad),
        r * np.cos(lat_rad) * np.sin(lon_rad),
        r * np.sin(lat_rad)
    ])

    # Velocity from Earth's rotation
    omega_earth = 7.2921159e-5  # rad/s
    velocity = np.cross(np.array([0, 0, omega_earth]), position)

    # Initial attitude: vertical (Body X = radial out)
    r_hat = position / np.linalg.norm(position)

    # Local East and North directions
    z_eci = np.array([0, 0, 1])
    east = np.cross(z_eci, r_hat)
    east = east / np.linalg.norm(east)
    north = np.cross(r_hat, east)

    # Heading direction (for yaw reference)
    heading_dir = np.cos(azimuth_rad) * north + np.sin(azimuth_rad) * east

    # Body frame: X=up, Y=heading, Z=right
    body_x = r_hat
    body_y = heading_dir
    body_z = np.cross(body_x, body_y)

    dcm_body_to_inertial = np.column_stack([body_x, body_y, body_z])
    quaternion = dcm_to_quaternion(dcm_body_to_inertial.T)

    state = State(
        position=position,
        velocity=velocity,
        quaternion=quaternion,
        angular_velocity=np.zeros(3),
        mass=total_mass,
        time=0.0,
        flat_earth=False,
    )

    # =========================================================================
    # Simulation Setup
    # =========================================================================
    # Vehicle parameters
    vehicle_diameter = 3.0  # m
    ref_area = np.pi * (vehicle_diameter / 2) ** 2

    Ixx = total_mass * 25**2 / 12  # Approximate for 25m tall rocket
    inertia = np.diag([Ixx, Ixx, total_mass * (vehicle_diameter/2)**2 / 2])

    config = DynamicsConfig(gravity_model=GravityModel.SPHERICAL)
    dynamics = RigidBodyDynamics(inertia=inertia, config=config)
    atmosphere = Atmosphere()
    aero = SimpleAero(Cd0=0.3, reference_area=ref_area)

    s1_throttle = ThrottleModel(engine=s1_perf, min_throttle=0.6)
    s2_throttle = ThrottleModel(engine=s2_perf, min_throttle=0.4)

    # Simulation parameters
    dt = 0.02  # 50 Hz
    t_max = 7200.0  # 2 hours max (enough for full orbit + circularization)

    # Data storage
    times = []
    positions = []
    velocities = []
    altitudes = []
    inclinations = []
    eccentricities = []
    phases = []  # 1=S1 burn, 2=coast, 3=S2 burn, 4=coast to apogee, 5=circ burn, 6=orbit

    # State tracking
    stage = 1
    s1_remaining = s1_prop
    s2_remaining = s2_prop
    phase = 1
    meco_done = False
    ses_done = False
    seco_done = False
    circ_started = False
    circ_done = False

    lat, lon, alt = eci_to_lla(state.position)
    print(f"  Position: {lat:.2f}°N, {lon:.2f}°E, alt={alt:.0f}m")
    print(f"  Initial velocity: {np.linalg.norm(velocity):.0f} m/s (Earth rotation)")

    # =========================================================================
    # Simulation Loop
    # =========================================================================
    print("\nRunning simulation...")
    print("-" * 70)

    step = 0
    record_interval = 5  # Record every 5 steps (0.1s)

    while state.time < t_max:
        t = state.time
        lat, lon, alt = eci_to_lla(state.position)
        speed = np.linalg.norm(state.velocity)

        # Orbital elements
        apogee, perigee, sma, ecc, inc, t_apogee = compute_orbital_elements(
            state.position, state.velocity
        )

        # Local frame
        r_hat = state.position / np.linalg.norm(state.position)
        v_radial = np.dot(state.velocity, r_hat)
        v_horiz_vec = state.velocity - v_radial * r_hat
        v_horiz = np.linalg.norm(v_horiz_vec)
        fpa = np.degrees(np.arctan2(v_radial, v_horiz))

        # =====================================================================
        # Guidance: Gravity Turn
        # =====================================================================
        # Pitch program (angle from vertical)
        if alt < 300:
            pitch_deg = 0.0  # Vertical
        elif alt < 1000:
            pitch_deg = 2.0 * (alt - 300) / 700  # Gentle kickoff
        elif alt < 10000:
            pitch_deg = 2.0 + 25.0 * (alt - 1000) / 9000  # Through max-Q
        elif alt < 50000:
            pitch_deg = 27.0 + 40.0 * (alt - 10000) / 40000  # Accelerate horizontal
        elif alt < 100000:
            pitch_deg = 67.0 + 18.0 * (alt - 50000) / 50000  # Near horizontal
        else:
            pitch_deg = 85.0 + 4.0 * min((alt - 100000) / 100000, 1.0)  # Hold ~89°

        # Heading: follow velocity direction above 10km (gravity turn)
        if alt < 10000:
            # Hold launch azimuth during initial ascent
            heading_vec = np.cos(azimuth_rad) * north + np.sin(azimuth_rad) * east
        else:
            # Prograde steering (align with horizontal velocity)
            if v_horiz > 10:
                heading_vec = v_horiz_vec / v_horiz
            else:
                heading_vec = np.cos(azimuth_rad) * north + np.sin(azimuth_rad) * east

        # Target thrust direction
        pitch_rad = np.radians(pitch_deg)
        target_dir = np.cos(pitch_rad) * r_hat + np.sin(pitch_rad) * heading_vec
        target_dir = target_dir / np.linalg.norm(target_dir)

        # =====================================================================
        # Attitude Control (simplified - instant pointing)
        # =====================================================================
        body_x = target_dir
        body_z_approx = -r_hat  # Belly toward Earth
        body_y = np.cross(body_z_approx, body_x)
        if np.linalg.norm(body_y) > 0.01:
            body_y = body_y / np.linalg.norm(body_y)
            body_z = np.cross(body_x, body_y)
            dcm = np.column_stack([body_x, body_y, body_z])
            state.quaternion = dcm_to_quaternion(dcm.T)

        # =====================================================================
        # Propulsion
        # =====================================================================
        thrust_body = np.zeros(3)
        mdot = 0.0

        if stage == 1:
            if s1_remaining > 0:
                thrust_mag, mdot = s1_throttle.at(1.0, alt)
                s1_remaining -= mdot * dt
                thrust_body = np.array([thrust_mag, 0, 0])
                phase = 1
            else:
                # MECO
                if not meco_done:
                    print(f"  T+{t:6.1f}s MECO: alt={alt/1000:.1f}km, v={speed:.0f}m/s, "
                          f"γ={fpa:.1f}°, inc={inc:.1f}°")
                    meco_done = True
                    phase = 2

                # Coast for 3 seconds then stage
                if t > state.time + 3.0 or (meco_done and s1_remaining <= -mdot * 3):
                    stage = 2
                    state = State(
                        position=state.position.copy(),
                        velocity=state.velocity.copy(),
                        quaternion=state.quaternion.copy(),
                        angular_velocity=np.zeros(3),
                        mass=s2_mass,
                        time=state.time,
                        flat_earth=False,
                    )
                    # Update inertia for smaller stage
                    s2_length = 8.0
                    Ixx_s2 = s2_mass * s2_length**2 / 12
                    dynamics = RigidBodyDynamics(
                        inertia=np.diag([Ixx_s2, Ixx_s2, s2_mass * 1.0**2 / 2]),
                        config=config
                    )
                    print(f"  T+{t:6.1f}s STAGING: S2 ignition")
                    ses_done = True
                    phase = 3

        elif stage == 2:
            if not circ_done:
                # Check if we should burn
                should_burn = False

                if apogee < TARGET_ALT * 0.98 and s2_remaining > 0 and not seco_done:
                    # Still raising apogee
                    should_burn = True
                    phase = 3
                elif apogee >= TARGET_ALT * 0.98 and not circ_started:
                    # Apogee reached, coast to circularization
                    if not seco_done:
                        print(f"  T+{t:6.1f}s SECO-1: Apogee={apogee/1000:.1f}km, "
                              f"Perigee={perigee/1000:.1f}km")
                        seco_done = True
                    phase = 4

                    # Calculate circularization burn timing
                    v_circ = np.sqrt(MU_EARTH / (R_EARTH_EQ + apogee))
                    v_at_apogee = np.sqrt(MU_EARTH * (2/(R_EARTH_EQ + apogee) - 1/sma))
                    dv_circ = v_circ - v_at_apogee
                    burn_duration = dv_circ * state.mass / (100000)  # Approx thrust

                    if t_apogee < burn_duration / 2 + dt and t_apogee > 0:
                        print(f"  T+{t:6.1f}s CIRC IGNITION: t_to_apogee={t_apogee:.1f}s, "
                              f"ΔV={dv_circ:.0f}m/s")
                        circ_started = True
                        phase = 5

                elif circ_started and not circ_done:
                    # Circularization burn
                    v_circ_here = np.sqrt(MU_EARTH / np.linalg.norm(state.position))
                    if v_horiz < v_circ_here * 0.998 and s2_remaining > 0:
                        should_burn = True
                        phase = 5
                    else:
                        print(f"  T+{t:6.1f}s SECO-2: Orbit={perigee/1000:.1f}x{apogee/1000:.1f}km, "
                              f"inc={inc:.1f}°, ecc={ecc:.5f}")
                        circ_done = True
                        phase = 6

                if should_burn and s2_remaining > 0:
                    thrust_mag, mdot = s2_throttle.at(1.0, alt)
                    s2_remaining -= mdot * dt
                    thrust_body = np.array([thrust_mag, 0, 0])

        # =====================================================================
        # Aerodynamics
        # =====================================================================
        aero_force = np.zeros(3)
        if alt < 100000:
            atm = atmosphere.at_altitude(alt, speed)
            if atm.density > 1e-9:
                omega_earth_vec = np.array([0, 0, omega_earth])
                v_atm = np.cross(omega_earth_vec, state.position)
                v_rel = state.velocity - v_atm
                v_body = state.dcm_inertial_to_body @ v_rel
                if np.linalg.norm(v_rel) > 5:
                    aero_force = aero.forces_body(v_body, atm.density, atm.speed_of_sound)

        # =====================================================================
        # Integration
        # =====================================================================
        state_dot = dynamics.derivatives(
            state=state,
            thrust_body=thrust_body,
            moment_body=np.zeros(3),
            mass_rate=-mdot,
            aero_force_body=aero_force,
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

        # Progress output
        if step % 2500 == 0 and step > 0:
            print(f"  T+{t:6.1f}s: alt={alt/1000:6.1f}km, v={speed:6.0f}m/s, "
                  f"γ={fpa:5.1f}°, inc={inc:5.1f}°, phase={phase}")

        # Termination conditions
        if alt < -1000:
            print(f"  T+{t:.1f}s IMPACT")
            break

        if circ_done and t > state.time + 60:
            # Propagate one more minute then stop
            break

        step += 1

    # =========================================================================
    # Results Summary
    # =========================================================================
    print("-" * 70)
    print("\nFINAL ORBIT:")

    final_apogee, final_perigee, _, final_ecc, final_inc, _ = compute_orbital_elements(
        state.position, state.velocity
    )
    print(f"  Perigee:      {final_perigee/1000:.1f} km")
    print(f"  Apogee:       {final_apogee/1000:.1f} km")
    print(f"  Inclination:  {final_inc:.2f}°")
    print(f"  Eccentricity: {final_ecc:.6f}")

    v_circ_target = orbital_velocity(TARGET_ALT)
    v_horiz_final = np.linalg.norm(state.velocity - np.dot(state.velocity, r_hat) * r_hat)

    if abs(final_inc - TARGET_INC) < 1.0 and final_perigee > 350e3 and final_ecc < 0.01:
        print("\n  ✓ MISSION SUCCESS - Orbit achieved!")
    else:
        print(f"\n  Target was: {TARGET_ALT/1000:.0f} km circular, {TARGET_INC:.1f}° inclination")

    return {
        'times': np.array(times),
        'positions': np.array(positions),
        'velocities': np.array(velocities),
        'altitudes': np.array(altitudes),
        'inclinations': np.array(inclinations),
        'eccentricities': np.array(eccentricities),
        'phases': np.array(phases),
        'target_altitude': TARGET_ALT,
        'target_inclination': TARGET_INC,
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    from rocket.orbital_plotting import plot_orbit_animation, plot_orbital_dashboard

    data = run_orbital_launch()

    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)

    # Reference orbits
    reference_orbits = [
        {
            'altitude': data['target_altitude'],
            'inclination': data['target_inclination'],
            'color': 'rgba(255,100,100,0.4)',
            'name': f"Target: {data['target_altitude']/1000:.0f}km, {data['target_inclination']:.1f}°"
        }
    ]

    # Phase names for launch
    phase_names = {
        1: 'Stage 1 Burn',
        2: 'Coast/Staging',
        3: 'Stage 2 Burn',
        4: 'Coast to Apogee',
        5: 'Circularization',
        6: 'Final Orbit'
    }

    # 3D Animation
    print("\n1. Animated 3D visualization...")
    fig_3d = plot_orbit_animation(
        data,
        title="🚀 Cape Canaveral → ISS Orbit",
        reference_orbits=reference_orbits,
        show_earth_axes=True,
        phase_names=phase_names
    )
    output_3d = "outputs/orbital_launch_3d.html"
    fig_3d.write_html(output_3d)
    print(f"   Saved to: {output_3d}")

    # Static dashboard
    print("\n2. Static orbital dashboard...")
    fig_static = plot_orbital_dashboard(
        data,
        title="Orbital Launch Telemetry",
        target_altitude=data['target_altitude']
    )
    output_static = "outputs/orbital_launch_static.html"
    fig_static.write_html(output_static)
    print(f"   Saved to: {output_static}")

    print("\n" + "="*50)
    print("COMPLETE")
    print("="*50)

    print("\nOpening 3D visualization...")
    fig_3d.show()
