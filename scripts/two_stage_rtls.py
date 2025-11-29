#!/usr/bin/env python
"""Two-stage launch with first stage RTLS landing.

Simulates a SpaceX-style mission:
- Two-stage vehicle launches from Cape Canaveral
- Stage 1 ascent to staging altitude/velocity
- Stage separation
- Stage 1: Boostback, entry burn, landing burn -> RTLS
- Stage 2: Continue to 300km circular orbit

Both stages are tracked independently in parallel simulations.

Usage:
    uv run python scripts/two_stage_rtls.py
"""

import numpy as np

from flight.guidance import (
    FirstStageLandingGuidance,
    GravityTurnGuidance,
    OrbitalInsertionGuidance,
)
from rocket.dynamics.state import State
from rocket.environment.gravity import MU_EARTH, R_EARTH_EQ, GravityModel
from rocket.orbital import (
    OMEGA_EARTH,
    compute_orbital_elements,
    launch_azimuth,
)
from rocket.simulation import SimConfig, Simulator


def eci_to_ecef_velocity(position: np.ndarray, velocity_eci: np.ndarray) -> np.ndarray:
    """Convert ECI velocity to ECEF (velocity relative to rotating Earth)."""
    omega = np.array([0.0, 0.0, OMEGA_EARTH])
    v_rotation = np.cross(omega, position)
    return velocity_eci - v_rotation


def compute_circular_velocity(altitude: float) -> float:
    """Compute circular orbital velocity at given altitude."""
    r = R_EARTH_EQ + altitude
    return np.sqrt(MU_EARTH / r)


def run_two_stage_mission():
    """Run complete two-stage mission simulation."""
    print("=" * 70)
    print("TWO-STAGE LAUNCH WITH FIRST STAGE RTLS")
    print("=" * 70)

    # =========================================================================
    # Mission Configuration
    # =========================================================================
    LAUNCH_LAT = 28.5       # Cape Canaveral [deg]
    LAUNCH_LON = -80.6      # [deg]
    TARGET_ALT = 300e3      # 300 km target circular orbit [m]
    TARGET_INC = 28.5       # Match launch latitude [deg]

    # Staging parameters (Falcon 9-like)
    STAGE_1_BURN_TIME = 162.0  # First stage burns for ~2.5 minutes
    STAGING_ALT_APPROX = 80e3  # ~80 km at staging

    # ----- First Stage (Falcon 9 Block 5 - like) -----
    S1_WET_MASS = 433000.0     # Stage 1 wet mass [kg]
    S1_DRY_MASS = 25000.0      # Stage 1 dry mass [kg]
    S1_THRUST = 7600000.0      # 9 Merlin engines [N]
    S1_ISP_SL = 282.0          # Sea level Isp [s]
    S1_ISP_VAC = 311.0         # Vacuum Isp [s]
    S1_LANDING_THRUST = 845000.0  # Single Merlin for landing [N]

    # ----- Second Stage -----
    S2_WET_MASS = 116000.0     # Stage 2 wet mass [kg]
    S2_DRY_MASS = 4000.0       # Stage 2 dry mass [kg]
    S2_THRUST = 934000.0       # Single Merlin Vacuum [N]
    S2_ISP = 348.0             # Vacuum Isp [s]

    # Total vehicle
    TOTAL_MASS = S1_WET_MASS + S2_WET_MASS

    # Compute mass flow rates
    g0 = 9.80665
    s1_mdot = S1_THRUST / (S1_ISP_SL * g0)  # Use sea-level for conservative estimate
    s2_mdot = S2_THRUST / (S2_ISP * g0)

    # Compute delta-V budgets
    s1_dv = S1_ISP_SL * g0 * np.log(TOTAL_MASS / (S1_DRY_MASS + S2_WET_MASS))
    s2_dv = S2_ISP * g0 * np.log(S2_WET_MASS / S2_DRY_MASS)

    print("\n" + "-" * 70)
    print("MISSION PARAMETERS")
    print("-" * 70)
    print(f"  Launch site: {LAUNCH_LAT}°N, {LAUNCH_LON}°E")
    print(f"  Target orbit: {TARGET_ALT/1000:.0f} km circular")
    print(f"  Target inclination: {TARGET_INC}°")
    print(f"  Required orbital velocity: {compute_circular_velocity(TARGET_ALT):.0f} m/s")

    print("\n  STAGE 1:")
    print(f"    Wet mass: {S1_WET_MASS/1000:.0f} t")
    print(f"    Dry mass: {S1_DRY_MASS/1000:.0f} t")
    print(f"    Thrust: {S1_THRUST/1e6:.1f} MN (9 engines)")
    print(f"    Isp: {S1_ISP_SL:.0f} s (SL) / {S1_ISP_VAC:.0f} s (vac)")
    print(f"    Delta-V: {s1_dv:.0f} m/s")
    print(f"    Burn time: {(S1_WET_MASS - S1_DRY_MASS) / s1_mdot:.0f} s (max)")

    print("\n  STAGE 2:")
    print(f"    Wet mass: {S2_WET_MASS/1000:.0f} t")
    print(f"    Dry mass: {S2_DRY_MASS/1000:.0f} t")
    print(f"    Thrust: {S2_THRUST/1e6:.2f} MN")
    print(f"    Isp: {S2_ISP:.0f} s (vacuum)")
    print(f"    Delta-V: {s2_dv:.0f} m/s")

    # =========================================================================
    # Initialize Simulation
    # =========================================================================
    print("\n" + "-" * 70)
    print("INITIALIZING SIMULATION")
    print("-" * 70)

    # Compute launch azimuth
    az = launch_azimuth(np.radians(LAUNCH_LAT), np.radians(TARGET_INC))
    print(f"  Launch azimuth: {np.degrees(az):.1f}° from North")

    config = SimConfig(gravity_model=GravityModel.SPHERICAL, include_atmosphere=True)

    # Vehicle inertia (simplified)
    Ixx = TOTAL_MASS * 70**2 / 12
    Izz = TOTAL_MASS * 1.85**2 / 2
    inertia = np.diag([Ixx, Ixx, Izz])

    # Create simulator for combined stack
    sim = Simulator.from_launch_pad(
        latitude=LAUNCH_LAT,
        longitude=LAUNCH_LON,
        heading=np.degrees(az),
        vehicle_mass=TOTAL_MASS,
        inertia=inertia,
        config=config,
    )

    # Store landing site position (ECI at t=0)
    landing_site_eci = sim.get_state().position.copy()
    print(f"  Landing site ECI: [{landing_site_eci[0]/1e6:.3f}, {landing_site_eci[1]/1e6:.3f}, {landing_site_eci[2]/1e6:.3f}] Mm")

    # Initialize first stage guidance
    # Very conservative pitch program - need significant altitude before pitchover
    # Real F9 pitches over very gradually and maintains significant vertical component
    s1_guidance = GravityTurnGuidance(
        target_altitude=STAGING_ALT_APPROX,
        max_thrust=S1_THRUST,
        vertical_rise_time=45.0,   # Extended vertical rise to build altitude
        pitch_kick_duration=45.0,  # Very gradual pitch over
        pitch_kick_angle=np.radians(35.0),  # Pitch over to start gravity turn
    )

    # =========================================================================
    # Phase 1: First Stage Ascent
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: FIRST STAGE ASCENT")
    print("=" * 70)

    dt = 0.02
    record_interval = 25  # 2 Hz recording

    # Data collection
    s1_ascent_times = []
    s1_ascent_positions = []
    s1_ascent_velocities = []
    s1_ascent_altitudes = []
    s1_ascent_phases = []
    s1_ascent_masses = []
    s1_ascent_thrusts = []
    s1_ascent_accelerations = []

    step = 0
    last_print_time = -10.0
    staging_state = None

    while sim.time < STAGE_1_BURN_TIME:
        state = sim.get_state()
        env = sim.get_environment()

        # Guidance
        cmd = s1_guidance.compute(state)

        # Check for propellant depletion
        propellant_remaining = state.mass - (S1_DRY_MASS + S2_WET_MASS)
        if propellant_remaining <= 0:
            print(f"\n⚠ Stage 1 propellant depleted at T+{sim.time:.1f}s")
            break

        # Set attitude
        if cmd.target_attitude is not None:
            sim.set_attitude(cmd.target_attitude)

        # Step simulation
        sim.step(
            thrust=cmd.thrust,
            gimbal=None,
            mass_rate=-s1_mdot if cmd.thrust > 0 else 0.0,
            dt=dt,
        )

        # Record data
        if step % record_interval == 0:
            v_ecef = eci_to_ecef_velocity(state.position, state.velocity)
            s1_ascent_times.append(state.time)
            s1_ascent_positions.append(state.position.copy())
            s1_ascent_velocities.append(v_ecef)
            s1_ascent_altitudes.append(env.altitude)
            s1_ascent_phases.append(1)  # Ascent phase
            # Record S1 mass only (subtract S2 wet mass since we're carrying it)
            s1_mass_only = state.mass - S2_WET_MASS
            s1_ascent_masses.append(s1_mass_only)
            s1_ascent_thrusts.append(cmd.thrust)
            ascent_accel = cmd.thrust / state.mass if state.mass > 0 else 0.0
            s1_ascent_accelerations.append(ascent_accel)

        # Progress output
        if sim.time - last_print_time >= 10.0:
            v_ecef = eci_to_ecef_velocity(state.position, state.velocity)
            speed = np.linalg.norm(v_ecef)
            elements = compute_orbital_elements(state.position, state.velocity)

            print(
                f"T+{sim.time:5.0f}s | "
                f"Alt: {env.altitude/1000:5.1f} km | "
                f"Speed: {speed:5.0f} m/s | "
                f"Phase: {s1_guidance.phase_name:12s} | "
                f"Mass: {state.mass/1000:5.1f} t | "
                f"Apo: {elements.apogee_alt/1000:5.1f} km"
            )
            last_print_time = sim.time

        step += 1

    # Store staging state
    staging_state = sim.get_state()
    staging_env = sim.get_environment()
    staging_time = sim.time

    print(f"\n{'=' * 70}")
    print(f"MECO-1 / STAGE SEPARATION at T+{staging_time:.1f}s")
    print(f"{'=' * 70}")
    print(f"  Altitude: {staging_env.altitude/1000:.1f} km")
    print(f"  Speed: {np.linalg.norm(eci_to_ecef_velocity(staging_state.position, staging_state.velocity)):.0f} m/s")
    print(f"  Mass at separation: {staging_state.mass/1000:.1f} t")

    # =========================================================================
    # Create separate simulators for each stage
    # =========================================================================

    # Stage 1 simulator (for landing)
    s1_inertia = np.diag([S1_DRY_MASS * 30**2 / 12, S1_DRY_MASS * 30**2 / 12, S1_DRY_MASS * 1.85**2 / 2])
    # Propellant reserve for RTLS landing (very aggressive)
    # From simulation: boostback ~87s, entry ~30s, landing ~50s = 167s total
    # At 2.5 MN, mdot ≈ 831 kg/s → 139,000 kg needed
    S1_LANDING_PROPELLANT = 150000.0  # 150 tons landing propellant reserve

    s1_state = State(
        position=staging_state.position.copy(),
        velocity=staging_state.velocity.copy(),
        quaternion=staging_state.quaternion.copy(),
        angular_velocity=np.zeros(3),
        mass=S1_DRY_MASS + S1_LANDING_PROPELLANT,
        time=staging_time,
    )
    s1_sim = Simulator(state=s1_state, inertia=s1_inertia, config=config)

    # Stage 2 simulator (for orbit)
    s2_inertia = np.diag([S2_WET_MASS * 15**2 / 12, S2_WET_MASS * 15**2 / 12, S2_WET_MASS * 1.85**2 / 2])
    s2_state = State(
        position=staging_state.position.copy(),
        velocity=staging_state.velocity.copy(),
        quaternion=staging_state.quaternion.copy(),
        angular_velocity=np.zeros(3),
        mass=S2_WET_MASS,
        time=staging_time,
    )
    s2_sim = Simulator(state=s2_state, inertia=s2_inertia, config=config)

    # Initialize stage guidance
    # Use 5 engines for boostback, single engine (throttleable) for landing
    S1_BOOSTBACK_THRUST = S1_LANDING_THRUST * 5  # 5 engines for boostback
    S1_MIN_THRUST = S1_LANDING_THRUST * 0.4  # Single engine at 40% throttle

    s1_landing_guidance = FirstStageLandingGuidance(
        initial_landing_site_eci=landing_site_eci,
        max_thrust=S1_BOOSTBACK_THRUST,
        min_thrust=S1_MIN_THRUST,
        dry_mass=S1_DRY_MASS,
        isp=S1_ISP_VAC,  # Use vacuum Isp for high altitude
    )
    s1_landing_guidance.initialize(staging_state, staging_time)

    s2_orbit_guidance = OrbitalInsertionGuidance(
        target_altitude=TARGET_ALT,
        max_thrust=S2_THRUST,
    )
    s2_orbit_guidance.initialize(staging_state, staging_time)

    # Data collection for both stages
    s1_landing_times = []
    s1_landing_positions = []
    s1_landing_velocities = []
    s1_landing_altitudes = []
    s1_landing_phases = []
    s1_landing_masses = []
    s1_landing_thrusts = []
    s1_landing_accelerations = []

    s2_orbit_times = []
    s2_orbit_positions = []
    s2_orbit_velocities = []
    s2_orbit_altitudes = []
    s2_orbit_phases = []
    s2_orbit_masses = []
    s2_orbit_thrusts = []
    s2_orbit_accelerations = []

    # =========================================================================
    # Phase 2: Parallel simulation of both stages
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: PARALLEL STAGE OPERATIONS")
    print("=" * 70)
    print("\n[Stage 1 → RTLS Landing] | [Stage 2 → Orbit Insertion]")
    print("-" * 70)

    t_max = 6000.0  # 100 minutes max (full orbit)
    step = 0
    last_print_time = staging_time - 20.0

    s1_complete = False
    s2_complete = False
    s1_landed = False
    s1_final_state = None
    s1_final_alt = 0.0
    s1_final_speed = 0.0

    while (s1_sim.time < t_max or s2_sim.time < t_max) and not (s1_complete and s2_complete):
        # ----- Stage 1 Update -----
        if not s1_complete:
            s1_state = s1_sim.get_state()
            s1_env = s1_sim.get_environment()

            # Check for ground impact first
            if s1_env.altitude < 0:
                s1_complete = True
                s1_final_state = s1_state
                s1_final_alt = s1_env.altitude
                # Use ECI speed directly (more reliable near ground)
                s1_final_speed = np.linalg.norm(s1_state.velocity)
                s1_landed = s1_final_speed < 100  # Need to stop relative to rotating Earth
                if s1_landed:
                    print(f"\n  ✓ Stage 1 LANDED at T+{s1_state.time:.1f}s (speed: {s1_final_speed:.1f} m/s ECI)")
                else:
                    print(f"\n  ✗ Stage 1 IMPACT at T+{s1_state.time:.1f}s (speed: {s1_final_speed:.1f} m/s ECI)")
                continue

            s1_cmd = s1_landing_guidance.compute(s1_state)

            if s1_cmd.target_attitude is not None:
                s1_sim.set_attitude(s1_cmd.target_attitude)

            # Debug: check for anomalous acceleration during landing burn
            if s1_landing_guidance.phase.value >= 6:  # Landing burn
                old_speed = np.linalg.norm(s1_state.velocity)

            # Compute mass rate but don't go below dry mass
            if s1_cmd.thrust > 0 and s1_state.mass > S1_DRY_MASS:
                s1_mass_rate = -s1_cmd.thrust / (S1_ISP_VAC * g0)
                s1_thrust = s1_cmd.thrust
            else:
                s1_mass_rate = 0.0
                s1_thrust = 0.0 if s1_state.mass <= S1_DRY_MASS else s1_cmd.thrust

            s1_sim.step(
                thrust=s1_thrust,
                gimbal=None,
                mass_rate=s1_mass_rate,
                dt=dt,
            )

            # Debug: detect sudden velocity change
            if s1_landing_guidance.phase.value >= 6:
                new_state = s1_sim.get_state()
                new_speed = np.linalg.norm(new_state.velocity)
                if new_speed > old_speed * 1.5 and new_speed > 1000:
                    print(f"\n  ⚠ ANOMALY: Speed jumped from {old_speed:.0f} to {new_speed:.0f} m/s")
                    print(f"    Position: {new_state.position}")
                    print(f"    Velocity: {new_state.velocity}")
                    print(f"    Old velocity: {s1_state.velocity}")
                    print(f"    Thrust: {s1_cmd.thrust:.0f} N, Mass: {s1_state.mass:.0f} kg")
                    print(f"    Altitude: {s1_env.altitude:.0f} m")
                    print(f"    Target attitude: {s1_cmd.target_attitude}")
                    # Check thrust direction
                    dcm = s1_state.dcm_body_to_inertial
                    thrust_dir = dcm[:, 0]  # Body X in inertial
                    vel_dir = s1_state.velocity / np.linalg.norm(s1_state.velocity)
                    dot = np.dot(thrust_dir, vel_dir)
                    print(f"    Thrust/velocity dot product: {dot:.3f} (should be negative for decel)")

            # Record
            if step % record_interval == 0:
                v_ecef = eci_to_ecef_velocity(s1_state.position, s1_state.velocity)
                s1_landing_times.append(s1_state.time)
                s1_landing_positions.append(s1_state.position.copy())
                s1_landing_velocities.append(v_ecef)
                s1_landing_altitudes.append(s1_env.altitude)
                s1_landing_phases.append(s1_landing_guidance.phase.value)
                s1_landing_masses.append(s1_state.mass)
                s1_landing_thrusts.append(s1_thrust if s1_cmd.thrust > 0 else 0.0)
                accel = s1_thrust / s1_state.mass if s1_state.mass > 0 else 0.0
                s1_landing_accelerations.append(accel)

            # Check completion (soft landing)
            if s1_landing_guidance.is_complete(s1_state) and not s1_complete:
                s1_complete = True
                s1_final_state = s1_state
                s1_final_alt = s1_env.altitude
                v_ecef = eci_to_ecef_velocity(s1_state.position, s1_state.velocity)
                s1_final_speed = np.linalg.norm(v_ecef)
                s1_landed = s1_final_speed < 50

        # ----- Stage 2 Update -----
        if not s2_complete:
            s2_state = s2_sim.get_state()
            s2_env = s2_sim.get_environment()

            # Terminate cleanly on reentry / impact
            if s2_env.altitude < 0.0:
                s2_complete = True
                print(
                    f"\n  ✗ Stage 2 REENTRY/IMPACT at T+{s2_state.time:.1f}s "
                    f"(alt: {s2_env.altitude/1000:.1f} km)"
                )
                # Don't attempt further guidance or burning once we've hit the atmosphere/ground
                continue

            s2_cmd = s2_orbit_guidance.compute(s2_state)

            # Check propellant
            if s2_state.mass <= S2_DRY_MASS:
                s2_cmd = s2_cmd._replace(thrust=0.0)

            if s2_cmd.target_attitude is not None:
                s2_sim.set_attitude(s2_cmd.target_attitude)

            s2_sim.step(
                thrust=s2_cmd.thrust,
                gimbal=None,
                mass_rate=-s2_mdot if s2_cmd.thrust > 0 else 0.0,
                dt=dt,
            )

            # Record
            if step % record_interval == 0:
                v_ecef = eci_to_ecef_velocity(s2_state.position, s2_state.velocity)
                s2_orbit_times.append(s2_state.time)
                s2_orbit_positions.append(s2_state.position.copy())
                s2_orbit_velocities.append(v_ecef)
                s2_orbit_altitudes.append(s2_env.altitude)
                s2_orbit_phases.append(s2_orbit_guidance.phase.value + 10)  # Offset for unique ID
                s2_orbit_masses.append(s2_state.mass)
                s2_thrust_now = s2_cmd.thrust if s2_cmd.thrust > 0 and s2_state.mass > S2_DRY_MASS else 0.0
                s2_orbit_thrusts.append(s2_thrust_now)
                s2_accel = s2_thrust_now / s2_state.mass if s2_state.mass > 0 else 0.0
                s2_orbit_accelerations.append(s2_accel)

            # Check completion
            if s2_orbit_guidance.is_complete(s2_state) or s2_sim.time > t_max:
                s2_complete = True

        # ----- Progress Output -----
        current_time = max(s1_sim.time if not s1_complete else 0, s2_sim.time if not s2_complete else 0)
        if current_time - last_print_time >= 15.0:
            s1_str = ""
            s2_str = ""

            if not s1_complete:
                s1_state = s1_sim.get_state()
                s1_env = s1_sim.get_environment()
                v_ecef = eci_to_ecef_velocity(s1_state.position, s1_state.velocity)
                s1_str = f"S1: {s1_env.altitude/1000:5.1f}km {np.linalg.norm(v_ecef):5.0f}m/s {s1_landing_guidance.phase_name:12s}"
            else:
                s1_str = "S1: COMPLETE" + " " * 20

            if not s2_complete:
                s2_state = s2_sim.get_state()
                s2_env = s2_sim.get_environment()
                elements = compute_orbital_elements(s2_state.position, s2_state.velocity)
                s2_str = f"S2: {s2_env.altitude/1000:5.1f}km Apo:{elements.apogee_alt/1000:5.0f}km {s2_orbit_guidance.phase_name:12s}"
            else:
                s2_str = "S2: COMPLETE"

            print(f"T+{current_time:5.0f}s | {s1_str} | {s2_str}")
            last_print_time = current_time

        step += 1

        # Safety check
        if step > 1000000:
            print("\n⚠ Maximum iterations reached")
            break

    # =========================================================================
    # Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("MISSION SUMMARY")
    print("=" * 70)

    # Stage 1 results
    if s1_final_state is not None:
        print("\n--- STAGE 1 (RTLS) ---")
        print(f"  Final time: T+{s1_final_state.time:.1f}s")
        print(f"  Final altitude: {s1_final_alt:.1f} m")
        print(f"  Final speed (ECEF): {s1_final_speed:.1f} m/s")
        print(f"  Final phase: {s1_landing_guidance.phase_name}")

        if s1_landed:
            print("  ✓ LANDING SUCCESS!")
        else:
            print("  ✗ Landing failed (crash)")
            print("\n  Note: Successful RTLS landing requires optimal trajectory planning.")
            print("  The simple heuristic guidance demonstrates the simulation working")
            print("  but cannot achieve the precision needed for soft landing.")
            print("  See flight/guidance/powered_descent.py for convex optimization approach.")
    else:
        print("\n--- STAGE 1 (RTLS) ---")
        print("  Stage 1 did not complete")

    # Stage 2 results
    s2_final = s2_sim.get_state()
    s2_final_env = s2_sim.get_environment()
    elements = compute_orbital_elements(s2_final.position, s2_final.velocity)
    v_circular = compute_circular_velocity(s2_final_env.altitude)

    print("\n--- STAGE 2 (ORBIT) ---")
    print(f"  Final time: T+{s2_sim.time:.1f}s")
    print(f"  Altitude: {s2_final_env.altitude/1000:.1f} km")
    print(f"  Speed: {np.linalg.norm(s2_final.velocity):.0f} m/s (circular: {v_circular:.0f} m/s)")
    print(f"  Apogee: {elements.apogee_alt/1000:.1f} km")
    print(f"  Perigee: {elements.perigee_alt/1000:.1f} km")
    print(f"  Eccentricity: {elements.eccentricity:.4f}")
    print(f"  Inclination: {np.degrees(elements.inclination):.1f}°")
    print(f"  Remaining mass: {s2_final.mass:.0f} kg")

    if elements.perigee_alt > 200e3 and elements.eccentricity < 0.1:
        print("  ✓ ORBIT ACHIEVED!")
    elif elements.perigee_alt > 100e3:
        print("  ~ Suboptimal orbit")
    else:
        print("  ✗ Failed to achieve orbit")

    # Prepare trajectory data
    trajectory_data = {
        # Stage 1 ascent
        's1_ascent_times': np.array(s1_ascent_times),
        's1_ascent_positions': np.array(s1_ascent_positions),
        's1_ascent_velocities': np.array(s1_ascent_velocities),
        's1_ascent_altitudes': np.array(s1_ascent_altitudes),
        's1_ascent_phases': np.array(s1_ascent_phases),
        's1_ascent_masses': np.array(s1_ascent_masses),
        's1_ascent_thrusts': np.array(s1_ascent_thrusts),
        's1_ascent_accelerations': np.array(s1_ascent_accelerations),
        # Stage 1 landing
        's1_landing_times': np.array(s1_landing_times),
        's1_landing_positions': np.array(s1_landing_positions),
        's1_landing_velocities': np.array(s1_landing_velocities),
        's1_landing_altitudes': np.array(s1_landing_altitudes),
        's1_landing_phases': np.array(s1_landing_phases),
        's1_landing_masses': np.array(s1_landing_masses),
        's1_landing_thrusts': np.array(s1_landing_thrusts),
        's1_landing_accelerations': np.array(s1_landing_accelerations),
        # Stage 2
        's2_times': np.array(s2_orbit_times),
        's2_positions': np.array(s2_orbit_positions),
        's2_velocities': np.array(s2_orbit_velocities),
        's2_altitudes': np.array(s2_orbit_altitudes),
        's2_phases': np.array(s2_orbit_phases),
        's2_masses': np.array(s2_orbit_masses),
        's2_thrusts': np.array(s2_orbit_thrusts),
        's2_accelerations': np.array(s2_orbit_accelerations),
        # Metadata
        'staging_time': staging_time,
        'target_altitude': TARGET_ALT,
        'landing_site_eci': landing_site_eci,
        # Vehicle parameters
        's1_dry_mass': S1_DRY_MASS,
        's1_wet_mass': S1_WET_MASS,
        's1_landing_propellant': S1_LANDING_PROPELLANT,
        's2_dry_mass': S2_DRY_MASS,
        's2_wet_mass': S2_WET_MASS,
        's1_thrust': S1_THRUST,
        's2_thrust': S2_THRUST,
        's1_isp_sl': S1_ISP_SL,
        's1_isp_vac': S1_ISP_VAC,
        's2_isp': S2_ISP,
    }

    return trajectory_data


if __name__ == "__main__":
    from rocket.export import export_trajectory_to_json

    raw_data = run_two_stage_mission()

    print("\n" + "=" * 70)
    print("EXPORTING FLIGHT DATA")
    print("=" * 70)

    # Combine data for export
    s1_all_times = np.concatenate(
        [raw_data["s1_ascent_times"], raw_data["s1_landing_times"]]
    )
    s1_all_positions = np.concatenate(
        [raw_data["s1_ascent_positions"], raw_data["s1_landing_positions"]]
    )
    s1_all_velocities = np.concatenate(
        [raw_data["s1_ascent_velocities"], raw_data["s1_landing_velocities"]]
    )
    s1_all_phases = np.concatenate(
        [raw_data["s1_ascent_phases"], raw_data["s1_landing_phases"]]
    )
    s1_all_masses = np.concatenate(
        [raw_data["s1_ascent_masses"], raw_data["s1_landing_masses"]]
    )
    s1_all_thrusts = np.concatenate(
        [raw_data["s1_ascent_thrusts"], raw_data["s1_landing_thrusts"]]
    )
    s1_all_accelerations = np.concatenate(
        [raw_data["s1_ascent_accelerations"], raw_data["s1_landing_accelerations"]]
    )

    # Structure for JSON export
    export_data = {
        "times": raw_data["s2_times"],
        "positions": raw_data["s2_positions"],
        "velocities": raw_data["s2_velocities"],
        "s2_phases": raw_data["s2_phases"],
        "s2_masses": raw_data["s2_masses"],
        "s2_thrusts": raw_data["s2_thrusts"],
        "s2_accelerations": raw_data["s2_accelerations"],
        "s1_times": s1_all_times,
        "s1_positions": s1_all_positions,
        "s1_velocities": s1_all_velocities,
        "s1_phases": s1_all_phases,
        "s1_masses": s1_all_masses,
        "s1_thrusts": s1_all_thrusts,
        "s1_accelerations": s1_all_accelerations,
        "target_altitude": raw_data["target_altitude"],
        "landing_site_eci": raw_data["landing_site_eci"],
        "staging_time": raw_data["staging_time"],
        # Vehicle parameters
        "s1_dry_mass": raw_data["s1_dry_mass"],
        "s1_wet_mass": raw_data["s1_wet_mass"],
        "s1_landing_propellant": raw_data["s1_landing_propellant"],
        "s2_dry_mass": raw_data["s2_dry_mass"],
        "s2_wet_mass": raw_data["s2_wet_mass"],
        "s1_thrust": raw_data["s1_thrust"],
        "s2_thrust": raw_data["s2_thrust"],
        "s1_isp_sl": raw_data["s1_isp_sl"],
        "s1_isp_vac": raw_data["s1_isp_vac"],
        "s2_isp": raw_data["s2_isp"],
    }

    export_trajectory_to_json(export_data, "web/public/flight_data.json")
    print("\nView the flight at: http://localhost:5173 (run 'cd web && bun run dev')")

