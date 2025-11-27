#!/usr/bin/env python
"""6DOF trajectory simulation example.

This example demonstrates the full GNC (Guidance, Navigation, Control) stack:
1. Design an engine
2. Create a vehicle model
3. Set up gravity turn guidance
4. Run 6DOF simulation
5. Analyze and visualize results

This simulates a simple sounding rocket flight to ~100 km altitude.
"""

from pathlib import Path

import numpy as np

from rocket import EngineInputs, design_engine
from rocket.gnc.guidance import GravityTurnGuidance
from rocket.simulation import Simulator
from rocket.units import kilonewtons, megapascals
from rocket.vehicle import MassProperties, Vehicle


def main() -> None:
    """Run the trajectory simulation example."""

    print("=" * 60)
    print("6DOF TRAJECTORY SIMULATION")
    print("=" * 60)

    # =========================================================================
    # 1. Design the engine
    # =========================================================================
    print("\n1. Designing engine...")

    inputs = EngineInputs.from_propellants(
        oxidizer="LOX",
        fuel="CH4",
        thrust=kilonewtons(50),
        chamber_pressure=megapascals(5.0),
        mixture_ratio=3.0,
        name="Sounding Rocket Engine",
    )

    performance, geometry = design_engine(inputs)

    print(f"   Isp (vac): {performance.isp_vac.value:.1f} s")
    print(f"   Thrust:    {inputs.thrust.to('N').value/1000:.1f} kN")
    print(f"   mdot:      {performance.mdot.value:.2f} kg/s")

    # =========================================================================
    # 2. Create vehicle model
    # =========================================================================
    print("\n2. Creating vehicle model...")

    # Dry mass properties (structure, engine, avionics)
    # For a 6m tall, 0.4m diameter rocket:
    # - Izz (roll) ≈ 1/2 * m * r² ≈ 4 kg·m²
    # - Ixx, Iyy (pitch/yaw) ≈ 1/12 * m * L² ≈ 600 kg·m²
    dry_mass = MassProperties.from_principal(
        mass=200.0,  # kg
        cg=[0.0, 0.0, 2.0],  # Center of gravity [m]
        Ixx=600.0, Iyy=600.0, Izz=4.0,  # Moments of inertia [kg*m^2]
    )

    propellant_mass = 800.0  # kg

    # Create vehicle
    vehicle = Vehicle(
        dry_mass=dry_mass,
        initial_propellant_mass=propellant_mass,
        propellant_cg=np.array([0.0, 0.0, 1.5]),  # Propellant CG [m]
        engine=performance,
        reference_area=np.pi * 0.2**2,  # 0.4m diameter
        reference_length=6.0,  # 6m tall
    )

    print(f"   Dry mass:       {vehicle.dry_mass.mass:.0f} kg")
    print(f"   Propellant:     {vehicle.initial_propellant_mass:.0f} kg")
    print(f"   Total mass:     {vehicle.total_mass:.0f} kg")
    print(f"   Mass ratio:     {vehicle.mass_ratio:.2f}")
    print(f"   Ideal delta-V:  {vehicle.delta_v(performance.isp_vac.value):.0f} m/s")
    print(f"   Burn time:      {vehicle.burn_time(performance.mdot.value):.1f} s")

    # =========================================================================
    # 3. Set up guidance
    # =========================================================================
    print("\n3. Setting up guidance...")

    guidance = GravityTurnGuidance(
        pitch_kick=np.radians(3.0),    # 3 degree kick
        pitch_kick_time=5.0,           # At T+5s
        pitch_kick_duration=5.0,       # Over 5 seconds
        target_altitude=100000.0,      # 100 km target
    )

    print(f"   Pitch kick:     {np.degrees(guidance.pitch_kick):.1f} deg")
    print(f"   Kick time:      {guidance.pitch_kick_time:.1f} s")
    print(f"   Target alt:     {guidance.target_altitude/1000:.0f} km")

    # =========================================================================
    # 4. Run simulation
    # =========================================================================
    print("\n4. Running simulation...")

    sim = Simulator(
        vehicle=vehicle,
        guidance=guidance,
        dt=0.01,  # 10ms time step (100 Hz) for numerical stability
    )

    result = sim.run(t_final=200.0)  # Max 200 seconds

    print(f"   Simulation complete: {len(result.states)} steps")

    # =========================================================================
    # 5. Analyze results
    # =========================================================================
    print("\n5. Results:")
    print("-" * 40)

    print(f"   Max altitude:    {result.max_altitude/1000:.1f} km")
    print(f"   Max speed:       {result.max_speed:.0f} m/s")
    print(f"   Max Q:           {result.max_dynamic_pressure/1000:.1f} kPa")

    burnout = result.burnout_time
    if burnout:
        print(f"   Burnout time:    {burnout:.1f} s")
        burnout_idx = int(burnout / sim.dt)
        if burnout_idx < len(result.states):
            print(f"   Burnout alt:     {result.altitude[burnout_idx]/1000:.1f} km")
            print(f"   Burnout speed:   {result.speed[burnout_idx]:.0f} m/s")

    # =========================================================================
    # 6. Save results
    # =========================================================================
    print("\n6. Saving results...")

    output_dir = Path("outputs/trajectory_sim")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save trajectory plot
    result.plot_trajectory(output_dir / "trajectory.png")
    print(f"   Plot saved: {output_dir}/trajectory.png")

    # Save data
    result.save_csv(output_dir / "trajectory_data.csv")
    print(f"   Data saved: {output_dir}/trajectory_data.csv")

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

