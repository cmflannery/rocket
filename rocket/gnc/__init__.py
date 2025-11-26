"""GNC (Guidance, Navigation, Control) module for rocket vehicles.

Provides guidance laws, navigation/estimation, and control algorithms
for rocket vehicle simulation and control system design.

Example:
    >>> from rocket.gnc.control import PIDController
    >>> from rocket.gnc.guidance import GravityTurnGuidance
    >>>
    >>> # Create attitude controller
    >>> pitch_ctrl = PIDController(kp=1.0, ki=0.1, kd=0.5)
    >>>
    >>> # Create guidance
    >>> guidance = GravityTurnGuidance(initial_pitch=90, pitch_rate=0.5)
"""

from rocket.gnc.control import (
    PIDController,
    TVCController,
)
from rocket.gnc.guidance import (
    GravityTurnGuidance,
)

__all__ = [
    # Control
    "PIDController",
    "TVCController",
    # Guidance
    "GravityTurnGuidance",
]

