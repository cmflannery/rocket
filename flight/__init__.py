"""Flight software package - GNC algorithms for rocket vehicles.

This package contains guidance, navigation, and control algorithms
that would run on the flight computer. These are developed and tested
against the simulation infrastructure in rocket/.

Architecture:
    The simulation (rocket/) provides the "plant" - accurate models of
    vehicle dynamics and environment. Flight software (flight/) provides
    the algorithms that command the vehicle.

    Simulation loop:
        state = sim.get_state()          # "Sensors" (truth for now)
        cmd = guidance.compute(state)    # Your algorithm
        sim.step(cmd, dt)                # Apply to plant

Subpackages:
    guidance: Trajectory and steering algorithms
    control: Attitude and thrust vector control

Example:
    >>> from flight.guidance import GravityTurnGuidance
    >>> from flight.control import AttitudeController
    >>> from rocket.simulation import Simulator
    >>>
    >>> sim = Simulator.from_launch_pad(...)
    >>> guidance = GravityTurnGuidance(target_altitude=200e3)
    >>> controller = AttitudeController()
    >>>
    >>> while sim.altitude < 200e3:
    ...     state = sim.get_state()
    ...     target_attitude = guidance.compute(state)
    ...     thrust_cmd = controller.compute(state, target_attitude)
    ...     sim.step(thrust_cmd, dt=0.02)
"""

from flight.control import AttitudeController
from flight.guidance import GravityTurnGuidance

__all__ = [
    "GravityTurnGuidance",
    "AttitudeController",
]


