"""Guidance algorithms for rocket vehicles.

Guidance computes the desired trajectory and attitude commands
based on current state and mission objectives.

Available algorithms:
    GravityTurnGuidance: Efficient ascent trajectory for launch vehicles
    FirstStageLandingGuidance: RTLS landing guidance for reusable first stages
    OrbitalInsertionGuidance: Second stage orbital insertion guidance
"""

from flight.guidance.first_stage_landing import (
    FirstStageLandingGuidance,
    FirstStageLandingPhase,
    LandingCommand,
)
from flight.guidance.gravity_turn import GravityTurnGuidance, GuidanceCommand
from flight.guidance.orbital_insertion import (
    InsertionCommand,
    OrbitalInsertionGuidance,
    OrbitalInsertionPhase,
)

__all__ = [
    "FirstStageLandingGuidance",
    "FirstStageLandingPhase",
    "GravityTurnGuidance",
    "GuidanceCommand",
    "InsertionCommand",
    "LandingCommand",
    "OrbitalInsertionGuidance",
    "OrbitalInsertionPhase",
]

