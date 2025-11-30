"""Control algorithms for rocket vehicles.

Control computes actuator commands to achieve desired attitude
and trajectory from guidance.

Available controllers:
    AttitudeController: Quaternion-based attitude control with TVC
"""

from flight.control.attitude import AttitudeController

__all__ = [
    "AttitudeController",
]


