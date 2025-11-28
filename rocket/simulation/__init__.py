"""Simulation module for rocket vehicle flight simulation.

Provides the step-driven simulation interface where GNC code controls
the loop and the simulator maintains truth state.

Example:
    >>> from rocket.simulation import Simulator, SimConfig, ThrustCommand
    >>> 
    >>> # Create simulator on launch pad
    >>> sim = Simulator.from_launch_pad(latitude=28.5, vehicle_mass=40000)
    >>> 
    >>> # GNC loop
    >>> while sim.altitude < 100000:
    ...     state = sim.get_state()
    ...     env = sim.get_environment()
    ...     
    ...     thrust = ThrustCommand(magnitude=1e6, gimbal_pitch=0.01)
    ...     sim.step(thrust, mass_rate=-300, dt=0.02)
"""

from rocket.simulation.simulator import (
    EnvironmentData,
    SimConfig,
    SimulationResult,
    Simulator,
    ThrustCommand,
)

__all__ = [
    "EnvironmentData",
    "SimConfig",
    "SimulationResult",
    "Simulator",
    "ThrustCommand",
]
