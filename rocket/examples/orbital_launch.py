#!/usr/bin/env python
"""Orbital launch trajectory simulation with spherical Earth.

Uses proper Earth-Centered Inertial (ECI) coordinates where:
- Position is relative to Earth's center
- Gravity points radially inward (changes direction as rocket moves)
- Orbital mechanics work naturally

Target orbit: ~200 km LEO
Orbital velocity at 200km: ~7,784 m/s
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rocket import EngineInputs, design_engine
from rocket.dynamics.rigid_body import DynamicsConfig, RigidBodyDynamics, rk4_step
from rocket.dynamics.state import State
from rocket.environment.atmosphere import Atmosphere
from rocket.environment.gravity import R_EARTH_EQ, GravityModel, orbital_velocity
from rocket.propulsion.throttle_model import GimbalModel, ThrottleModel
from rocket.units import kilonewtons, megapascals
from rocket.vehicle.aerodynamics import SimpleAero


def eci_to_lla(position):
    """Convert ECI position to latitude, longitude, altitude."""
    x, y, z = position
    r = np.linalg.norm(position)

    lat = np.arcsin(z / r) if r > 0 else 0
    lon = np.arctan2(y, x)
    alt = r - R_EARTH_EQ

    return np.degrees(lat), np.degrees(lon), alt


def local_vertical_horizontal(position, velocity):
    """Get local vertical and horizontal velocity components.

    Returns:
        v_vertical: velocity component radially outward (m/s)
        v_horizontal: velocity component tangent to Earth (m/s)
    """
    r = np.linalg.norm(position)
    r_hat = position / r if r > 0 else np.array([0, 0, 1])

    v_radial = np.dot(velocity, r_hat)  # Positive = going up
    v_tangent = np.linalg.norm(velocity - v_radial * r_hat)

    return v_radial, v_tangent


def flight_path_angle(position, velocity):
    """Compute flight path angle (angle between velocity and local horizontal).

    Returns angle in degrees. 90° = straight up, 0° = horizontal, -90° = straight down.
    """
    v_radial, v_tangent = local_vertical_horizontal(position, velocity)
    speed = np.linalg.norm(velocity)

    if speed < 1:
        return 90.0

    return np.degrees(np.arctan2(v_radial, v_tangent))


def compute_orbital_elements(position, velocity):
    """Compute orbital elements from state vectors.

    Returns:
        apogee_alt: Apogee altitude above Earth surface [m]
        perigee_alt: Perigee altitude above Earth surface [m]
        semi_major: Semi-major axis [m]
        eccentricity: Orbital eccentricity
        time_to_apogee: Time to next apogee passage [s]
    """
    from rocket.environment.gravity import MU_EARTH, R_EARTH_EQ

    r_vec = position
    v_vec = velocity
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # Specific orbital energy
    energy = v**2 / 2 - MU_EARTH / r

    # Semi-major axis (negative for hyperbolic)
    if abs(energy) < 1e-10:
        # Parabolic - very high apogee
        return 1e9, r - R_EARTH_EQ, 1e9, 1.0, 0.0

    semi_major = -MU_EARTH / (2 * energy)

    # Specific angular momentum
    h_vec = np.cross(r_vec, v_vec)

    # Eccentricity
    ecc_vec = np.cross(v_vec, h_vec) / MU_EARTH - r_vec / r
    eccentricity = np.linalg.norm(ecc_vec)

    # Apogee and perigee
    if eccentricity >= 1.0:
        # Hyperbolic/parabolic - escaping
        apogee_alt = 1e9
        perigee_alt = semi_major * (1 - eccentricity) - R_EARTH_EQ
        time_to_apogee = 0.0
    else:
        apogee_alt = semi_major * (1 + eccentricity) - R_EARTH_EQ
        perigee_alt = semi_major * (1 - eccentricity) - R_EARTH_EQ

        # Calculate time to apogee
        # Mean motion
        n = np.sqrt(MU_EARTH / semi_major**3)

        # Eccentric anomaly (E)
        # cos(E) = (e + cos(nu)) / (1 + e*cos(nu)) is harder since we need nu.
        # Easier: r = a(1 - e cos E) -> cos E = (1 - r/a) / e
        cos_E = (1 - r / semi_major) / eccentricity
        cos_E = np.clip(cos_E, -1.0, 1.0)
        E = np.arccos(cos_E)

        # Determine sign of E based on radial velocity (r_dot)
        # r_dot = dot(r, v) / |r|
        r_dot = np.dot(r_vec, v_vec)
        if r_dot < 0:
            # Moving towards perigee (past apogee) -> E is in (pi, 2pi)
            E = 2 * np.pi - E

        # Mean anomaly
        M = E - eccentricity * np.sin(E)

        # Time to apogee (M = pi at apogee)
        # If M < pi (before apogee), dt = (pi - M) / n
        # If M > pi (after apogee), dt = (3pi - M) / n  (time to NEXT apogee)
        time_to_apogee = (np.pi - M) / n if np.pi >= M else (3 * np.pi - M) / n

    return apogee_alt, perigee_alt, semi_major, eccentricity, time_to_apogee


def run_orbital_simulation():
    """Run orbital launch simulation in ECI coordinates."""
    print("=" * 70)
    print("ORBITAL LAUNCH SIMULATION (Spherical Earth)")
    print("=" * 70)

    # =========================================================================
    # Vehicle Configuration
    # =========================================================================
    print("\n1. Configuring Vehicle...")

    # Stage 1 engine
    s1_inputs = EngineInputs.from_propellants(
        oxidizer="LOX", fuel="RP1",
        thrust=kilonewtons(800),  # 800 kN
        chamber_pressure=megapascals(8.0),
        mixture_ratio=2.3,
        name="Stage 1",
    )
    s1_perf, _ = design_engine(s1_inputs)

    # Stage 2 engine
    s2_inputs = EngineInputs.from_propellants(
        oxidizer="LOX", fuel="RP1",
        thrust=kilonewtons(80),  # 80 kN vacuum stage
        chamber_pressure=megapascals(5.0),
        mixture_ratio=2.3,
        name="Stage 2",
    )
    s2_perf, _ = design_engine(s2_inputs)

    # Mass budget (optimized for orbit)
    # Improved design with better mass ratios
    s1_dry = 2500.0   # kg (lighter structure)
    s1_prop = 24000.0 # kg (more propellant)
    s2_dry = 400.0    # kg (lighter upper stage)
    s2_prop = 7000.0  # kg (more propellant for circularization)
    payload = 300.0   # kg

    total_mass = s1_dry + s1_prop + s2_dry + s2_prop + payload
    s2_total = s2_dry + s2_prop + payload

    # Delta-V budget
    ve1 = s1_perf.isp_vac.value * 9.81
    ve2 = s2_perf.isp_vac.value * 9.81
    dv1 = ve1 * np.log(total_mass / (total_mass - s1_prop))
    dv2 = ve2 * np.log(s2_total / (s2_total - s2_prop))

    print(f"   Stage 1: {s1_prop:.0f} kg, Isp={s1_perf.isp_vac.value:.0f}s, ΔV={dv1:.0f} m/s")
    print(f"   Stage 2: {s2_prop:.0f} kg, Isp={s2_perf.isp_vac.value:.0f}s, ΔV={dv2:.0f} m/s")
    print(f"   Total mass: {total_mass:.0f} kg, Total ΔV: {dv1+dv2:.0f} m/s")
    print(f"   T/W at liftoff: {800000 / (total_mass * 9.81):.2f}")
    print(f"   Target orbital velocity: {orbital_velocity(200e3):.0f} m/s")

    # =========================================================================
    # Initialize State in ECI coordinates
    # =========================================================================
    print("\n2. Initializing on launch pad (Cape Canaveral)...")

    # Launch site parameters
    lat_deg = 28.5   # Cape Canaveral
    lon_deg = -80.6
    heading_deg = 90.0  # Due East

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    heading = np.radians(heading_deg)

    # Position on Earth's surface in ECI
    r = R_EARTH_EQ
    pos_x = r * np.cos(lat) * np.cos(lon)
    pos_y = r * np.cos(lat) * np.sin(lon)
    pos_z = r * np.sin(lat)
    position = np.array([pos_x, pos_y, pos_z])

    # Velocity from Earth's rotation
    omega_earth = 7.2921159e-5  # rad/s
    velocity = np.cross(np.array([0, 0, omega_earth]), position)

    # Attitude: Body +X pointing radially outward (up)
    # Body +Y pointing East, Body +Z pointing South
    r_hat = position / np.linalg.norm(position)  # Up (radial)

    # East direction (perpendicular to r_hat in equatorial plane)
    z_eci = np.array([0, 0, 1])
    east = np.cross(z_eci, r_hat)
    east = east / np.linalg.norm(east)  # Normalize

    # North direction
    north = np.cross(r_hat, east)

    # Create rotation matrix (body to inertial)
    # Body X = up (r_hat), Body Y = heading direction, Body Z completes right-hand
    heading_dir = np.cos(heading) * north + np.sin(heading) * east
    body_x = r_hat  # Forward (up for launch)
    body_y = heading_dir  # Right wing direction
    body_z = np.cross(body_x, body_y)  # Down

    # DCM from body to inertial (columns are body axes in inertial frame)
    dcm_body_to_inertial = np.column_stack([body_x, body_y, body_z])

    # Convert DCM to quaternion (inertial to body)
    dcm_inertial_to_body = dcm_body_to_inertial.T
    from rocket.dynamics.state import dcm_to_quaternion
    quaternion = dcm_to_quaternion(dcm_inertial_to_body)

    state = State(
        position=position,
        velocity=velocity,
        quaternion=quaternion,
        angular_velocity=np.array([0.0, 0.0, 0.0]),
        mass=total_mass,
        time=0.0,
        flat_earth=False,  # ECI coordinates
    )

    # Vehicle parameters
    vehicle_length = 30.0  # m
    vehicle_diameter = 2.0 # m
    ref_area = np.pi * (vehicle_diameter / 2) ** 2

    Ixx = total_mass * vehicle_length**2 / 12
    Iyy = Ixx
    Izz = total_mass * (vehicle_diameter / 2)**2 / 2

    # Use SPHERICAL gravity for proper orbital mechanics
    config = DynamicsConfig(gravity_model=GravityModel.SPHERICAL)
    dynamics = RigidBodyDynamics(inertia=np.diag([Ixx, Iyy, Izz]), config=config)
    atmosphere = Atmosphere()

    s1_throttle = ThrottleModel(engine=s1_perf, min_throttle=0.6)
    s2_throttle = ThrottleModel(engine=s2_perf, min_throttle=0.4)
    gimbal = GimbalModel(max_gimbal_angle=np.radians(5), gimbal_rate=np.radians(15))
    aero = SimpleAero(Cd0=0.3, reference_area=ref_area)

    lat, lon, alt = eci_to_lla(state.position)
    print(f"   Initial position: {lat:.2f}°N, {lon:.2f}°E, alt={alt:.0f}m")
    print(f"   Initial velocity: {state.speed:.0f} m/s (Earth rotation)")

    # =========================================================================
    # Run Simulation
    # =========================================================================
    print("\n3. Running simulation...")

    dt = 0.01  # Smaller timestep for stability
    t_max = 6000.0  # Enough time for one orbit (~90 min for LEO)

    # Storage for main vehicle
    times = [0.0]
    positions = [state.position.copy()]
    velocities = [state.velocity.copy()]
    masses = [state.mass]
    altitudes = [alt]
    speeds = [state.speed]
    v_verticals = [0.0]
    v_horizontals = [state.speed]
    flight_path_angles = [90.0]  # Start vertical
    thrusts = [0.0]  # Thrust magnitude
    accelerations = [0.0]  # Total acceleration magnitude
    stages = [1]  # Current stage number

    # Storage for stage 1 after separation (will track its ballistic trajectory)
    s1_times = []
    s1_positions = []
    s1_velocities = []
    s1_altitudes = []
    s1_state = None  # Will be set at staging

    stage = 1
    s1_remaining = s1_prop
    s2_remaining = s2_prop
    staging_time = None
    seco_time = None
    circularization_burn_time = None

    # ==========================================================================
    # HOHMANN TRANSFER GUIDANCE
    # ==========================================================================
    # Target orbit altitude (200 km LEO)
    TARGET_ORBIT_ALT = 200e3  # meters

    # Circular velocity at target altitude
    v_circular_target = orbital_velocity(TARGET_ORBIT_ALT)
    print(f"   Target orbit: {TARGET_ORBIT_ALT/1000:.0f} km, V_circular={v_circular_target:.0f} m/s")

    # Track burn phases:
    # Phase 1: Injection burn - burn until apogee reaches target altitude
    # Phase 2: Coast to apogee
    # Phase 3: Circularization burn - burn until v_horizontal = v_circular
    s2_burn_phase = 1
    circularization_complete = False

    # For orbit tracking
    start_lon = lon_deg
    crossed_start = False
    orbit_complete = False

    step = 0
    while state.time < t_max:
        t = state.time

        # Current state in local coordinates
        lat, lon, alt = eci_to_lla(state.position)
        speed = state.speed
        v_vertical, v_horizontal = local_vertical_horizontal(state.position, state.velocity)
        fpa = flight_path_angle(state.position, state.velocity)

        # Get body attitude relative to local vertical
        r = np.linalg.norm(state.position)
        r_hat = state.position / r  # Local vertical (up)

        # Current pitch relative to local horizon
        body_z = state.dcm_body_to_inertial[:, 2]  # Body Z axis in inertial

        # =====================================================================
        # Staging
        # =====================================================================
        if stage == 1 and s1_remaining <= 0:
            print(f"   MECO T+{t:.1f}s: alt={alt/1000:.1f}km, v={speed:.0f}m/s, v_h={v_horizontal:.0f}m/s, γ={fpa:.1f}°")
            staging_time = t
            stage = 2

            # Save stage 1 state for ballistic propagation
            s1_state = State(
                position=state.position.copy(),
                velocity=state.velocity.copy(),
                quaternion=state.quaternion.copy(),
                angular_velocity=np.zeros(3),  # Stage tumbles
                mass=s1_dry,  # Just the empty stage
                time=state.time,
                flat_earth=False,
            )
            s1_times.append(state.time)
            s1_positions.append(state.position.copy())
            s1_velocities.append(state.velocity.copy())
            s1_altitudes.append(alt)

            new_mass = s2_dry + s2_remaining + payload
            state = State(
                position=state.position.copy(),
                velocity=state.velocity.copy(),
                quaternion=state.quaternion.copy(),
                angular_velocity=state.angular_velocity.copy(),
                mass=new_mass,
                time=state.time,
                flat_earth=False,
            )

            # Update inertia
            s2_length = 8.0
            Ixx = new_mass * s2_length**2 / 12
            dynamics = RigidBodyDynamics(
                inertia=np.diag([Ixx, Ixx, new_mass * 0.6**2 / 2]),
                config=config
            )

            # Coast phase
            for _ in range(int(3.0 / dt)):
                state_dot = dynamics.derivatives(
                    state=state,
                    thrust_body=np.zeros(3),
                    moment_body=np.zeros(3),
                    mass_rate=0.0,
                )
                state = rk4_step(state, dt, lambda s, sd=state_dot: sd)

                lat_c, lon_c, alt_c = eci_to_lla(state.position)
                v_v, v_h = local_vertical_horizontal(state.position, state.velocity)

                times.append(state.time)
                positions.append(state.position.copy())
                velocities.append(state.velocity.copy())
                masses.append(state.mass)
                altitudes.append(alt_c)
                speeds.append(state.speed)
                v_verticals.append(v_v)
                v_horizontals.append(v_h)
                flight_path_angles.append(flight_path_angle(state.position, state.velocity))
                thrusts.append(0.0)
                accel = np.linalg.norm(state_dot.velocity_dot) / 9.81
                accelerations.append(accel)
                stages.append(1.5)  # Coast between stages

            print(f"   SES-2 T+{state.time:.1f}s")
            continue

        # =====================================================================
        # Guidance - Gravity Turn (prograde pointing after pitch kick)
        # =====================================================================
        # Real rockets use gravity turn: after initial pitch kick, thrust along velocity

        if alt < 10000:
            # Initial ascent: Hold launch azimuth
            z_eci = np.array([0.0, 0.0, 1.0])
            east = np.cross(z_eci, r_hat)
            if np.linalg.norm(east) > 0.1:
                east = east / np.linalg.norm(east)
            else:
                east = np.array([0.0, 1.0, 0.0])
            heading_vec = east
        else:
            # Gravity turn: Align yaw with velocity vector (projected on horizontal plane)
            # This prevents "fighting" the natural orbit inclination
            v_radial = np.dot(state.velocity, r_hat) * r_hat
            v_horiz = state.velocity - v_radial
            if np.linalg.norm(v_horiz) > 1.0:
                heading_vec = v_horiz / np.linalg.norm(v_horiz)
            else:
                # Fallback if velocity is vertical
                z_eci = np.array([0.0, 0.0, 1.0])
                east = np.cross(z_eci, r_hat)
                heading_vec = east / np.linalg.norm(east)

        # OPTIMIZED PITCH PROGRAM - Minimize gravity losses
        # More aggressive horizontal turn to reduce gravity losses
        # Goal: Get horizontal quickly while above atmosphere

        # "Standard" LEO Profile
        # Kick turn start: 500m
        # Gravity turn takes over.
        # We force the pitch profile here to ensure we get horizontal.

        if alt < 500:  # Vertical rise
            pitch_from_vert_deg = 0.0
        elif alt < 10000:  # 0.5-10 km: Pitch over to 45 deg by 10km? No, maybe 30.
            # Gentle turn through Max Q
            pitch_from_vert_deg = 30.0 * (alt - 500) / (10000 - 500)
        elif alt < 40000:  # 10-40 km: Aggressive turn to 70 deg
            pitch_from_vert_deg = 30.0 + 40.0 * (alt - 10000) / (40000 - 10000)
        elif alt < 80000:  # 40-80 km: Flatten to 85 deg
            pitch_from_vert_deg = 70.0 + 15.0 * (alt - 40000) / (80000 - 40000)
        else:  # > 80 km: Hold 88 deg (almost horizontal)
             # Pitch slightly up if vertical speed is negative to maintain altitude?
             # For now, just hold nearly horizontal
            pitch_from_vert_deg = 88.0

        # Convert to direction vector
        pitch_rad = np.radians(pitch_from_vert_deg)
        target_direction = np.cos(pitch_rad) * r_hat + np.sin(pitch_rad) * heading_vec
        target_direction = target_direction / np.linalg.norm(target_direction)

        # =====================================================================
        # Control - Point along target direction (no active feedback)
        # =====================================================================
        # For a gravity turn, we simply point along velocity after kick.
        # Instead of using gimbal feedback control (which can oscillate),
        # we'll directly set the attitude by updating quaternion to match target.
        # This is a "perfect attitude control" simplification.

        dcm = state.dcm_body_to_inertial
        body_x = dcm[:, 0]  # Current thrust direction

        # For stable simulation, apply zero gimbal - thrust along body axis
        gimbal_pitch = 0.0
        gimbal_yaw = 0.0

        # Instead, we'll update the quaternion to align body_x with target_direction
        # This simulates perfect attitude control without gimbal dynamics
        # Compute rotation from current body_x to target_direction
        axis = np.cross(body_x, target_direction)
        axis_norm = np.linalg.norm(axis)

        if axis_norm > 1e-6:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(body_x, target_direction), -1, 1))

            # Apply only a fraction of the correction per timestep (rate limiting)
            # Real rockets have limited gimbal authority
            max_rate = np.radians(2.0)  # 2 deg/s max attitude rate (conservative)
            angle = np.clip(angle, -max_rate * dt, max_rate * dt)

            # Small rotation quaternion
            dq = np.array([
                np.cos(angle / 2),
                axis[0] * np.sin(angle / 2),
                axis[1] * np.sin(angle / 2),
                axis[2] * np.sin(angle / 2)
            ])

            # Apply rotation: q_new = dq * q
            q = state.quaternion
            new_q = np.array([
                dq[0]*q[0] - dq[1]*q[1] - dq[2]*q[2] - dq[3]*q[3],
                dq[0]*q[1] + dq[1]*q[0] + dq[2]*q[3] - dq[3]*q[2],
                dq[0]*q[2] - dq[1]*q[3] + dq[2]*q[0] + dq[3]*q[1],
                dq[0]*q[3] + dq[1]*q[2] - dq[2]*q[1] + dq[3]*q[0]
            ])

            # Normalize and update
            new_q = new_q / np.linalg.norm(new_q)
            state = State(
                position=state.position,
                velocity=state.velocity,
                quaternion=new_q,
                angular_velocity=np.zeros(3),  # Clear any angular velocity
                mass=state.mass,
                time=state.time,
                flat_earth=False,
            )

        # =====================================================================
        # Propulsion - Hohmann Transfer (Two-burn orbital insertion)
        # =====================================================================
        # Compute current orbital elements for guidance
        apogee_alt, perigee_alt, semi_major, eccentricity, time_to_apogee = compute_orbital_elements(
            state.position, state.velocity
        )

        # Circular velocity at current altitude
        v_circular_here = orbital_velocity(alt)

        if stage == 1 and s1_remaining > 0:
            # Stage 1: Full thrust
            thrust_mag, mdot = s1_throttle.at(1.0, alt)
            s1_remaining -= mdot * dt
        elif stage == 2 and not circularization_complete:
            # Stage 2: Hohmann transfer guidance

            if s2_burn_phase == 1:
                # PHASE 1: Injection burn
                # Burn until predicted apogee reaches target altitude
                if apogee_alt < TARGET_ORBIT_ALT * 0.98 and s2_remaining > 0:
                    thrust_mag, mdot = s2_throttle.at(1.0, alt)
                    s2_remaining -= mdot * dt
                else:
                    # Injection burn complete - apogee at target
                    print(f"   Injection burn complete T+{t:.1f}s:")
                    print(f"     alt={alt/1000:.1f}km, v={speed:.0f}m/s, γ={fpa:.1f}°")
                    print(f"     Predicted apogee: {apogee_alt/1000:.1f} km")
                    print(f"     Predicted perigee: {perigee_alt/1000:.1f} km")
                    s2_burn_phase = 2
                    thrust_mag, mdot = 0.0, 0.0

            elif s2_burn_phase == 2:
                # PHASE 2: Coast to apogee
                thrust_mag, mdot = 0.0, 0.0

                # Calculate circularization parameters
                mu = 3.986004418e14
                r_apogee = R_EARTH_EQ + apogee_alt

                # Velocity for circular orbit at apogee distance
                v_circular_apogee = np.sqrt(mu / r_apogee)

                # Current velocity at apogee (vis-viva equation)
                v_apogee_current = np.sqrt(mu * (2/r_apogee - 1/semi_major))

                delta_v_needed = v_circular_apogee - v_apogee_current

                # Estimate burn time
                # F = m * a -> t = dv * m / F
                # Use current mass and max thrust (80 kN)
                s2_thrust_vac = 80000.0
                burn_duration = delta_v_needed * state.mass / s2_thrust_vac

                # Trigger burn centered on apogee
                # Start when time_to_apogee < burn_duration / 2
                if (time_to_apogee < (burn_duration / 2) + dt and time_to_apogee > 0):
                     print(f"   Ignition for circularization T+{t:.1f}s:")
                     print(f"     Time to apogee: {time_to_apogee:.1f} s")
                     print(f"     Estimated burn duration: {burn_duration:.1f} s")
                     print(f"     Target Delta-V: {delta_v_needed:.1f} m/s")
                     s2_burn_phase = 3
                     circularization_burn_time = t

            elif s2_burn_phase == 3:
                # PHASE 3: Circularization burn
                # Burn prograde until v_horizontal = v_circular at current altitude

                if v_horizontal < v_circular_here * 0.998 and s2_remaining > 0:
                    # Still need more velocity
                    thrust_mag, mdot = s2_throttle.at(1.0, alt)
                    s2_remaining -= mdot * dt
                else:
                    # Circularization complete!
                    thrust_mag, mdot = 0.0, 0.0
                    if not circularization_complete:
                        circularization_complete = True
                        print(f"   Circularization complete T+{t:.1f}s:")
                        print(f"     alt={alt/1000:.1f}km, v_h={v_horizontal:.0f}m/s")
                        print(f"     V_circular: {v_circular_here:.0f} m/s")
                        print(f"     Remaining propellant: {s2_remaining:.0f} kg")
                        print(f"     Eccentricity: {eccentricity:.4f}")
            else:
                thrust_mag, mdot = 0.0, 0.0
        else:
            thrust_mag, mdot = 0.0, 0.0

        if thrust_mag > 0:
            thrust_body = gimbal.thrust_vector(thrust_mag, gimbal_pitch, gimbal_yaw)
            thrust_moment = gimbal.moment(thrust_mag, gimbal_pitch, gimbal_yaw)
        else:
            thrust_body = np.zeros(3)
            thrust_moment = np.zeros(3)

        # =====================================================================
        # Aerodynamics (relative to rotating atmosphere)
        # =====================================================================
        aero_force = np.zeros(3)
        aero_moment = np.zeros(3)
        if alt < 100000:
            atm = atmosphere.at_altitude(alt, speed)
            if atm.density > 1e-9:
                # Compute velocity relative to atmosphere (rotating with Earth)
                omega_earth = np.array([0, 0, 7.2921159e-5])
                v_atmosphere = np.cross(omega_earth, state.position)  # Atmosphere velocity in ECI
                v_relative_inertial = state.velocity - v_atmosphere   # Relative to atmosphere
                v_body = state.dcm_inertial_to_body @ v_relative_inertial  # Transform to body

                v_rel_mag = np.linalg.norm(v_relative_inertial)
                if v_rel_mag > 5:  # Only compute aero if moving relative to air
                    aero_force = aero.forces_body(v_body, atm.density, atm.speed_of_sound)

        # =====================================================================
        # Integrate
        # =====================================================================
        total_moment = thrust_moment + aero_moment


        state_dot = dynamics.derivatives(
            state=state,
            thrust_body=thrust_body,
            moment_body=total_moment,
            mass_rate=-mdot,
            aero_force_body=aero_force,
        )
        state = rk4_step(state, dt, lambda s, sd=state_dot: sd)

        # Record
        lat, lon, alt = eci_to_lla(state.position)
        v_vertical, v_horizontal = local_vertical_horizontal(state.position, state.velocity)

        times.append(state.time)
        positions.append(state.position.copy())
        velocities.append(state.velocity.copy())
        masses.append(state.mass)
        altitudes.append(alt)
        speeds.append(state.speed)
        v_verticals.append(v_vertical)
        v_horizontals.append(v_horizontal)
        flight_path_angles.append(flight_path_angle(state.position, state.velocity))
        thrusts.append(thrust_mag)
        # Compute total acceleration (thrust + gravity + drag) / mass
        accel = np.linalg.norm(state_dot.velocity_dot) / 9.81  # in G's
        accelerations.append(accel)
        stages.append(stage)

        # Propagate stage 1 ballistically if it exists
        if s1_state is not None:
            # Simple ballistic propagation for stage 1
            s1_dynamics = RigidBodyDynamics(
                inertia=np.diag([1000.0, 1000.0, 500.0]),  # Simple tumbling stage
                config=config
            )
            s1_state_dot = s1_dynamics.derivatives(
                state=s1_state,
                thrust_body=np.zeros(3),
                moment_body=np.zeros(3),
                mass_rate=0.0,
                aero_force_body=np.zeros(3),  # Simplified - no detailed aero
            )
            s1_state = rk4_step(s1_state, dt, lambda s, sd=s1_state_dot: sd)

            # Record stage 1 data
            s1_lat, s1_lon, s1_alt = eci_to_lla(s1_state.position)
            s1_times.append(s1_state.time)
            s1_positions.append(s1_state.position.copy())
            s1_velocities.append(s1_state.velocity.copy())
            s1_altitudes.append(s1_alt)

            # Stop tracking if stage 1 impacts
            if s1_alt < -1000:
                print(f"   Stage 1 IMPACT at T+{s1_state.time:.1f}s")
                s1_state = None  # Stop propagating

        step += 1
        # Print progress - more frequent during powered flight, less during coast
        print_interval = 400 if stage < 3 else 6000  # ~1 min during coast
        if step % print_interval == 0:
            if stage < 3:
                # Check thrust alignment during powered flight
                dcm = state.dcm_body_to_inertial
                thrust_dir_eci = dcm @ (thrust_body / max(np.linalg.norm(thrust_body), 1))
                vel_dir = state.velocity / max(speed, 1)
                thrust_vel_dot = np.dot(thrust_dir_eci, vel_dir)
                print(f"   T+{t:6.1f}s: alt={alt/1000:6.1f}km, v={speed:6.0f}m/s, "
                      f"v_h={v_horizontal:5.0f}m/s, γ={fpa:5.1f}°, T·v={thrust_vel_dot:.2f}")
            else:
                # Coast phase - just show orbital parameters
                print(f"   T+{t:6.1f}s: alt={alt/1000:6.1f}km, v={speed:6.0f}m/s, "
                      f"v_h={v_horizontal:5.0f}m/s, γ={fpa:5.1f}°")

        # Termination
        if alt < -1000:
            print(f"   IMPACT at T+{t:.1f}s")
            break

        # Mark SECO but continue propagating
        if stage == 2 and s2_remaining <= 0 and s2_remaining > -mdot * dt * 2:
            print(f"   SECO T+{t:.1f}s: alt={alt/1000:.1f}km, v={speed:.0f}m/s, v_h={v_horizontal:.0f}m/s, γ={fpa:.1f}°")
            seco_time = t
            stage = 3  # Coast phase
            print("   Entering coast phase - propagating orbit...")

        # Track orbit completion (crossed starting longitude while in space)
        if stage == 3 and alt > 100000:  # Above 100 km
            # Check if we've crossed the starting longitude
            lon_diff = lon - start_lon
            if lon_diff > 180:
                lon_diff -= 360
            elif lon_diff < -180:
                lon_diff += 360

            if abs(lon_diff) < 5 and crossed_start and not orbit_complete:
                orbit_complete = True
                orbit_time = t - seco_time if seco_time else t
                print(f"   ORBIT COMPLETE at T+{t:.1f}s ({orbit_time/60:.1f} min period)")
                print(f"   Perigee: {np.min(altitudes[-int(60/dt):])/1000:.1f} km, "
                      f"Apogee: {np.max(altitudes[-int(60/dt):])/1000:.1f} km")
                break
            elif abs(lon_diff) > 30:
                crossed_start = True

        # Check for numerical issues
        state_arr = state.to_array()
        if np.any(np.isnan(state_arr)) or np.any(np.abs(state_arr) > 1e15):
            print(f"   ERROR: Numerical instability at T+{t:.1f}s")
            print(f"          omega={state.angular_velocity}, mass={state.mass}")
            break

    # =========================================================================
    # Results
    # =========================================================================
    print("\n4. Results:")
    print("-" * 50)

    data = {
        'times': np.array(times),
        'positions': np.array(positions),
        'velocities': np.array(velocities),
        'masses': np.array(masses),
        'altitudes': np.array(altitudes),
        'speeds': np.array(speeds),
        'v_verticals': np.array(v_verticals),
        'v_horizontals': np.array(v_horizontals),
        'flight_path_angles': np.array(flight_path_angles),
        'thrusts': np.array(thrusts),
        'accelerations': np.array(accelerations),
        'stages': np.array(stages),
        'staging_time': staging_time,
        'seco_time': seco_time,
        'circularization_burn_time': circularization_burn_time,
        'orbit_complete': orbit_complete,
        's1_dry': s1_dry,
        's2_dry': s2_dry,
        's1_prop': s1_prop,
        's2_prop': s2_prop,
        'payload': payload,
        # Stage 1 trajectory
        's1_times': np.array(s1_times) if len(s1_times) > 0 else np.array([]),
        's1_positions': np.array(s1_positions) if len(s1_positions) > 0 else np.array([]),
        's1_velocities': np.array(s1_velocities) if len(s1_velocities) > 0 else np.array([]),
        's1_altitudes': np.array(s1_altitudes) if len(s1_altitudes) > 0 else np.array([]),
    }

    max_alt = np.max(data['altitudes']) / 1000
    max_speed = np.max(data['speeds'])
    final_alt = data['altitudes'][-1] / 1000
    final_speed = data['speeds'][-1]
    final_v_h = data['v_horizontals'][-1]
    final_fpa = data['flight_path_angles'][-1]

    print(f"   Max altitude: {max_alt:.1f} km")
    print(f"   Max speed: {max_speed:.0f} m/s")
    print(f"   Final altitude: {final_alt:.1f} km")
    print(f"   Final total velocity: {final_speed:.0f} m/s")
    print(f"   Final horizontal velocity: {final_v_h:.0f} m/s")
    print(f"   Final flight path angle: {final_fpa:.1f}°")

    v_orbital = orbital_velocity(final_alt * 1000)
    print(f"   Required orbital velocity at {final_alt:.0f}km: {v_orbital:.0f} m/s")

    if final_v_h > v_orbital * 0.98 and abs(final_fpa) < 3:
        print("   STATUS: ✓ ORBIT ACHIEVED!")
    elif final_v_h > v_orbital * 0.90:
        print(f"   STATUS: Near-orbital ({100*final_v_h/v_orbital:.1f}% of orbital velocity)")
    else:
        print(f"   STATUS: Suborbital ({100*final_v_h/v_orbital:.1f}% of orbital velocity)")

    return data


def create_visualization(data):
    """Create comprehensive visualization of orbital trajectory."""
    times = data['times']
    positions = data['positions']
    altitudes = data['altitudes'] / 1000
    v_horizontals = data['v_horizontals']
    v_verticals = data['v_verticals']
    flight_path_angles = data['flight_path_angles']
    masses = data['masses']
    thrusts = data['thrusts']
    accelerations = data['accelerations']
    # Compute ground track (lat/lon)
    lats, lons = [], []
    for pos in positions:
        r = np.linalg.norm(pos)
        lats.append(np.degrees(np.arcsin(pos[2] / r)))
        lons.append(np.degrees(np.arctan2(pos[1], pos[0])))
    lats = np.array(lats)
    lons = np.array(lons)

    # Create figure with 3 rows x 2 cols for more diagnostic plots
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "scene", "rowspan": 2}, {"type": "xy"}],
            [None, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=(
            "3D Trajectory (ECI Frame)",
            "Altitude & Velocities",
            "Mass & Thrust",
            "Ground Track",
            "Acceleration & Flight Path Angle"
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.08,
    )

    # 3D trajectory in ECI
    # Scale positions for visualization
    pos_km = positions / 1000

    # Compute speeds for hover
    speeds_array = np.linalg.norm(data['velocities'], axis=1)

    # Create custom hover text with orbital parameters
    hover_text = []
    for i in range(len(times)):
        text = (
            f"<b>T+{times[i]:.1f}s</b><br>"
            f"Altitude: {altitudes[i]:.1f} km<br>"
            f"Speed: {speeds_array[i]:.0f} m/s<br>"
            f"V_horiz: {v_horizontals[i]:.0f} m/s<br>"
            f"V_vert: {v_verticals[i]:.0f} m/s<br>"
            f"Flight path: {flight_path_angles[i]:.1f}°<br>"
            f"Mass: {masses[i]/1000:.1f} tons"
        )
        hover_text.append(text)

    fig.add_trace(go.Scatter3d(
        x=pos_km[:, 0], y=pos_km[:, 1], z=pos_km[:, 2],
        mode='lines',
        line=dict(color=v_horizontals, colorscale='Plasma', width=6,
                  colorbar=dict(title="V_horiz<br>(m/s)", x=0.45, len=0.4, y=0.8)),
        name="Upper Stage",
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
    ), row=1, col=1)

    # Add Stage 1 ballistic trajectory if it exists
    s1_positions_data = data.get('s1_positions', np.array([]))
    if len(s1_positions_data) > 0:
        s1_pos_km = s1_positions_data / 1000
        s1_times_data = data.get('s1_times', np.array([]))
        s1_alts = data.get('s1_altitudes', np.array([])) / 1000
        s1_speeds = np.linalg.norm(data.get('s1_velocities', np.array([[0,0,0]])), axis=1)

        s1_hover = []
        for i in range(len(s1_times_data)):
            text = (
                f"<b>STAGE 1 - T+{s1_times_data[i]:.1f}s</b><br>"
                f"Altitude: {s1_alts[i]:.1f} km<br>"
                f"Speed: {s1_speeds[i]:.0f} m/s"
            )
            s1_hover.append(text)

        fig.add_trace(go.Scatter3d(
            x=s1_pos_km[:, 0], y=s1_pos_km[:, 1], z=s1_pos_km[:, 2],
            mode='lines',
            line=dict(color='yellow', width=4, dash='dash'),
            name="Stage 1 (ballistic)",
            text=s1_hover,
            hovertemplate="%{text}<extra></extra>",
        ), row=1, col=1)

    # Earth sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 30)
    r_earth_km = R_EARTH_EQ / 1000
    x_earth = r_earth_km * np.outer(np.cos(u), np.sin(v))
    y_earth = r_earth_km * np.outer(np.sin(u), np.sin(v))
    z_earth = r_earth_km * np.outer(np.ones(50), np.cos(v))

    fig.add_trace(go.Surface(
        x=x_earth, y=y_earth, z=z_earth,
        colorscale=[[0, 'rgb(30,60,120)'], [1, 'rgb(30,80,140)']],
        showscale=False,
        opacity=0.7,
        name="Earth",
    ), row=1, col=1)

    # Launch and final points
    fig.add_trace(go.Scatter3d(
        x=[pos_km[0, 0]], y=[pos_km[0, 1]], z=[pos_km[0, 2]],
        mode='markers', marker=dict(size=8, color='lime'),
        name="Launch",
        hovertemplate=(
            "<b>LAUNCH</b><br>"
            f"Alt: {altitudes[0]:.1f} km<br>"
            f"Speed: {speeds_array[0]:.0f} m/s<br>"
            f"Mass: {masses[0]/1000:.1f} tons<br>"
            "<extra></extra>"
        ),
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=[pos_km[-1, 0]], y=[pos_km[-1, 1]], z=[pos_km[-1, 2]],
        mode='markers', marker=dict(size=8, color='red'),
        name="End",
        hovertemplate=(
            f"<b>T+{times[-1]:.1f}s</b><br>"
            f"Alt: {altitudes[-1]:.1f} km<br>"
            f"Speed: {speeds_array[-1]:.0f} m/s<br>"
            f"V_horiz: {v_horizontals[-1]:.0f} m/s<br>"
            f"Flight path: {flight_path_angles[-1]:.1f}°<br>"
            f"Mass: {masses[-1]/1000:.1f} tons<br>"
            "<extra></extra>"
        ),
    ), row=1, col=1)

    # Plot 1 (row 1, col 2): Altitude and velocities
    fig.add_trace(go.Scatter(x=times, y=altitudes, name='Altitude (km)',
                             line=dict(color='cyan')), row=1, col=2)
    fig.add_trace(go.Scatter(x=times, y=v_horizontals, name='V_horizontal (m/s)',
                             line=dict(color='orange')), row=1, col=2)
    fig.add_trace(go.Scatter(x=times, y=v_verticals, name='V_vertical (m/s)',
                             line=dict(color='lime', dash='dot')), row=1, col=2)

    # Orbital velocity reference
    v_orb = orbital_velocity(altitudes[-1] * 1000)
    fig.add_trace(go.Scatter(x=[times[0], times[-1]], y=[v_orb, v_orb],
                             name=f'V_orbital ({v_orb:.0f} m/s)',
                             line=dict(dash='dash', color='red')), row=1, col=2)

    # Plot 2 (row 2, col 2): Mass and Thrust
    fig.add_trace(go.Scatter(x=times, y=masses/1000, name='Mass (tons)',
                             line=dict(color='purple')), row=2, col=2)
    fig.add_trace(go.Scatter(x=times, y=thrusts/1000, name='Thrust (kN)',
                             line=dict(color='red')), row=2, col=2)

    # Plot 3 (row 3, col 1): Ground track
    fig.add_trace(go.Scatter(x=lons, y=lats, mode='lines',
                             line=dict(color='orange', width=3),
                             name='Ground Track'), row=3, col=1)
    fig.add_trace(go.Scatter(x=[lons[0]], y=[lats[0]], mode='markers',
                             marker=dict(size=10, color='lime'),
                             name='Launch Site'), row=3, col=1)

    # Plot 4 (row 3, col 2): Acceleration and Flight path angle
    fig.add_trace(go.Scatter(x=times, y=accelerations, name='Acceleration (G)',
                             line=dict(color='cyan')), row=3, col=2)
    fig.add_trace(go.Scatter(x=times, y=flight_path_angles, name='Flight Path γ (°)',
                             line=dict(color='magenta')), row=3, col=2)
    fig.add_trace(go.Scatter(x=[times[0], times[-1]], y=[0, 0],
                             name='Horizontal', line=dict(dash='dash', color='gray')), row=3, col=2)

    # Update layout
    fig.update_scenes(
        xaxis_title="X (km)",
        yaxis_title="Y (km)",
        zaxis_title="Z (km)",
        aspectmode='data',
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.5)),
    )

    # Update axes labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Value", row=1, col=2)

    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Mass (tons) / Thrust (kN)", row=2, col=2)

    fig.update_xaxes(title_text="Longitude (°)", row=3, col=1)
    fig.update_yaxes(title_text="Latitude (°)", row=3, col=1)

    fig.update_xaxes(title_text="Time (s)", row=3, col=2)
    fig.update_yaxes(title_text="Acceleration (G) / Flight Path (°)", row=3, col=2)

    fig.update_layout(
        title=dict(text="🚀 Orbital Launch Simulation (Spherical Earth ECI)", font=dict(size=24), x=0.5),
        height=1400,  # Increased height for 3 rows
        showlegend=True,
        template='plotly_dark',
    )

    return fig


def main():
    data = run_orbital_simulation()

    print("\n5. Creating visualization...")
    fig = create_visualization(data)

    output_path = "outputs/orbital_launch.html"
    fig.write_html(output_path)
    print(f"   Saved to: {output_path}")

    print("\n   Opening in browser...")
    fig.show()

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
