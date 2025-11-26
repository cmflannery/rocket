"""Simulation module for rocket vehicle flight simulation.

Provides the main simulation harness that integrates vehicle,
propulsion, environment, and GNC models.

Example:
    >>> from rocket.simulation import Simulator, SimulationResult
    >>> from rocket.vehicle import Vehicle
    >>> from rocket.gnc import GravityTurnGuidance
    >>>
    >>> sim = Simulator(vehicle, guidance)
    >>> result = sim.run(t_final=300)
    >>> result.plot_trajectory()
"""

from rocket.simulation.simulator import (
    SimulationResult,
    Simulator,
)

__all__ = [
    "Simulator",
    "SimulationResult",
]

