"""Control algorithms for rocket vehicles.

Provides PID controllers and thrust vector control for attitude
and trajectory control.
"""

from rocket.gnc.control.pid import (
    PIDController,
    PIDGains,
)
from rocket.gnc.control.tvc import (
    TVCController,
)

__all__ = [
    "PIDController",
    "PIDGains",
    "TVCController",
]

