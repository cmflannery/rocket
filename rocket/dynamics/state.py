"""6DOF state vector representation for rocket vehicle simulation.

The state vector contains:
- Position (3): [x, y, z] in Earth-Centered Inertial (ECI) frame
- Velocity (3): [vx, vy, vz] in ECI frame
- Quaternion (4): [q0, q1, q2, q3] attitude (scalar-first convention)
- Angular velocity (3): [p, q, r] body rates in body frame
- Mass (1): current vehicle mass

Total: 14 state variables

Coordinate frames:
- ECI: Earth-Centered Inertial (X toward vernal equinox, Z toward north pole)
- ECEF: Earth-Centered Earth-Fixed (rotates with Earth)
- NED: North-East-Down (local tangent plane)
- Body: Vehicle body frame (X forward, Y right, Z down)

Quaternion convention:
- Scalar-first: q = [q0, q1, q2, q3] where q0 is the scalar part
- Represents rotation from ECI to body frame
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

# =============================================================================
# Quaternion Utilities
# =============================================================================


@beartype
def normalize_quaternion(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize a quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


@beartype
def quaternion_multiply(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> NDArray[np.float64]:
    """Multiply two quaternions (Hamilton product).

    Args:
        q1: First quaternion [q0, q1, q2, q3]
        q2: Second quaternion [q0, q1, q2, q3]

    Returns:
        Product quaternion q1 * q2
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


@beartype
def quaternion_conjugate(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute quaternion conjugate (inverse for unit quaternions)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


@beartype
def quaternion_to_dcm(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert quaternion to Direction Cosine Matrix (DCM).

    Args:
        q: Quaternion [q0, q1, q2, q3] representing rotation from frame A to B

    Returns:
        3x3 DCM that transforms vectors from frame A to frame B
    """
    q = normalize_quaternion(q)
    q0, q1, q2, q3 = q

    return np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)],
    ])


@beartype
def dcm_to_quaternion(dcm: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert Direction Cosine Matrix to quaternion.

    Uses Shepperd's method for numerical stability.
    """
    trace = np.trace(dcm)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q0 = 0.25 / s
        q1 = (dcm[2, 1] - dcm[1, 2]) * s
        q2 = (dcm[0, 2] - dcm[2, 0]) * s
        q3 = (dcm[1, 0] - dcm[0, 1]) * s
    elif dcm[0, 0] > dcm[1, 1] and dcm[0, 0] > dcm[2, 2]:
        s = 2.0 * np.sqrt(1.0 + dcm[0, 0] - dcm[1, 1] - dcm[2, 2])
        q0 = (dcm[2, 1] - dcm[1, 2]) / s
        q1 = 0.25 * s
        q2 = (dcm[0, 1] + dcm[1, 0]) / s
        q3 = (dcm[0, 2] + dcm[2, 0]) / s
    elif dcm[1, 1] > dcm[2, 2]:
        s = 2.0 * np.sqrt(1.0 + dcm[1, 1] - dcm[0, 0] - dcm[2, 2])
        q0 = (dcm[0, 2] - dcm[2, 0]) / s
        q1 = (dcm[0, 1] + dcm[1, 0]) / s
        q2 = 0.25 * s
        q3 = (dcm[1, 2] + dcm[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + dcm[2, 2] - dcm[0, 0] - dcm[1, 1])
        q0 = (dcm[1, 0] - dcm[0, 1]) / s
        q1 = (dcm[0, 2] + dcm[2, 0]) / s
        q2 = (dcm[1, 2] + dcm[2, 1]) / s
        q3 = 0.25 * s

    q = np.array([q0, q1, q2, q3])
    return normalize_quaternion(q)


@beartype
def euler_to_quaternion(
    roll: float,
    pitch: float,
    yaw: float,
    sequence: Literal["ZYX", "XYZ"] = "ZYX",
) -> NDArray[np.float64]:
    """Convert Euler angles to quaternion.

    Args:
        roll: Roll angle (rotation about X) in radians
        pitch: Pitch angle (rotation about Y) in radians
        yaw: Yaw angle (rotation about Z) in radians
        sequence: Euler angle sequence (default ZYX = yaw-pitch-roll)

    Returns:
        Quaternion [q0, q1, q2, q3]
    """
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

    if sequence == "ZYX":
        q0 = cr * cp * cy + sr * sp * sy
        q1 = sr * cp * cy - cr * sp * sy
        q2 = cr * sp * cy + sr * cp * sy
        q3 = cr * cp * sy - sr * sp * cy
    else:  # XYZ
        q0 = cr * cp * cy - sr * sp * sy
        q1 = sr * cp * cy + cr * sp * sy
        q2 = cr * sp * cy - sr * cp * sy
        q3 = cr * cp * sy + sr * sp * cy

    return normalize_quaternion(np.array([q0, q1, q2, q3]))


@beartype
def quaternion_to_euler(q: NDArray[np.float64]) -> tuple[float, float, float]:
    """Convert quaternion to Euler angles (ZYX sequence).

    Args:
        q: Quaternion [q0, q1, q2, q3]

    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    q = normalize_quaternion(q)
    q0, q1, q2, q3 = q

    # Roll (rotation about X)
    sinr_cosp = 2 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1 - 2 * (q1**2 + q2**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (rotation about Y) - handle gimbal lock
    sinp = 2 * (q0 * q2 - q3 * q1)
    pitch = np.copysign(np.pi / 2, sinp) if abs(sinp) >= 1 else np.arcsin(sinp)

    # Yaw (rotation about Z)
    siny_cosp = 2 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1 - 2 * (q2**2 + q3**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


# =============================================================================
# State Classes
# =============================================================================


@beartype
@dataclass
class State:
    """6DOF state vector for rigid body dynamics.

    Attributes:
        position: [x, y, z] position in inertial frame [m]
        velocity: [vx, vy, vz] velocity in inertial frame [m/s]
        quaternion: [q0, q1, q2, q3] attitude quaternion (scalar-first)
        angular_velocity: [p, q, r] body angular rates [rad/s]
        mass: current vehicle mass [kg]
        time: simulation time [s]
        flat_earth: if True, z is altitude directly; if False, ECI coordinates
    """
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    quaternion: NDArray[np.float64]
    angular_velocity: NDArray[np.float64]
    mass: float
    time: float = 0.0
    flat_earth: bool = False

    def __post_init__(self) -> None:
        """Validate and normalize state."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        self.quaternion = normalize_quaternion(np.asarray(self.quaternion, dtype=np.float64))
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float64)

        if self.position.shape != (3,):
            raise ValueError(f"Position must be shape (3,), got {self.position.shape}")
        if self.velocity.shape != (3,):
            raise ValueError(f"Velocity must be shape (3,), got {self.velocity.shape}")
        if self.quaternion.shape != (4,):
            raise ValueError(f"Quaternion must be shape (4,), got {self.quaternion.shape}")
        if self.angular_velocity.shape != (3,):
            raise ValueError(f"Angular velocity must be shape (3,), got {self.angular_velocity.shape}")

    @classmethod
    def from_launch_pad(
        cls,
        latitude_deg: float = 28.5,  # Kennedy Space Center
        longitude_deg: float = -80.6,
        altitude_m: float = 0.0,
        heading_deg: float = 90.0,  # Due east
        mass_kg: float = 1000.0,
    ) -> "State":
        """Create initial state on launch pad.

        Args:
            latitude_deg: Launch site latitude [degrees]
            longitude_deg: Launch site longitude [degrees]
            altitude_m: Launch site altitude above sea level [m]
            heading_deg: Initial heading (0=north, 90=east) [degrees]
            mass_kg: Initial vehicle mass [kg]

        Returns:
            State at launch pad with vehicle pointing up
        """
        # Earth parameters
        R_earth = 6378137.0  # Equatorial radius [m]

        lat = np.radians(latitude_deg)
        lon = np.radians(longitude_deg)
        heading = np.radians(heading_deg)

        # Position in ECEF (simplified - ignoring Earth's oblateness)
        r = R_earth + altitude_m
        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)
        position = np.array([x, y, z])

        # Velocity = 0 relative to ground (but rotating with Earth)
        omega_earth = 7.2921159e-5  # Earth rotation rate [rad/s]
        velocity = np.cross(np.array([0, 0, omega_earth]), position)

        # Attitude: vehicle pointing "up" (radially outward)
        # with body X axis pointing in heading direction
        # Start with identity (body = inertial)
        # Rotate to align with local vertical
        pitch = lat  # Pitch up by latitude
        yaw = lon + np.pi/2  # Yaw to align with local frame
        roll = heading - np.pi/2  # Roll for heading

        quaternion = euler_to_quaternion(roll, pitch, yaw)

        # No initial rotation
        angular_velocity = np.array([0.0, 0.0, 0.0])

        return cls(
            position=position,
            velocity=velocity,
            quaternion=quaternion,
            angular_velocity=angular_velocity,
            mass=mass_kg,
            time=0.0,
        )

    @classmethod
    def from_flat_earth(
        cls,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,  # Altitude (positive up)
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        pitch_deg: float = 90.0,  # Pointing up
        yaw_deg: float = 0.0,
        roll_deg: float = 0.0,
        mass_kg: float = 1000.0,
    ) -> "State":
        """Create state in flat-Earth approximation (NED frame).

        Useful for short-range simulations where Earth curvature is negligible.

        Args:
            x, y, z: Position in local NED frame [m] (z positive UP)
            vx, vy, vz: Velocity in local NED frame [m/s]
            pitch_deg: Pitch angle [degrees] (90 = pointing up)
            yaw_deg: Yaw angle [degrees] (0 = north)
            roll_deg: Roll angle [degrees]
            mass_kg: Vehicle mass [kg]
        """
        position = np.array([x, y, z])
        velocity = np.array([vx, vy, vz])

        # Convert to radians and create quaternion
        roll = np.radians(roll_deg)
        pitch = np.radians(pitch_deg)
        yaw = np.radians(yaw_deg)
        quaternion = euler_to_quaternion(roll, pitch, yaw)

        angular_velocity = np.array([0.0, 0.0, 0.0])

        return cls(
            position=position,
            velocity=velocity,
            quaternion=quaternion,
            angular_velocity=angular_velocity,
            mass=mass_kg,
            time=0.0,
            flat_earth=True,
        )

    def to_array(self) -> NDArray[np.float64]:
        """Convert state to flat array for integration."""
        return np.concatenate([
            self.position,
            self.velocity,
            self.quaternion,
            self.angular_velocity,
            [self.mass],
        ])

    @classmethod
    def from_array(
        cls,
        arr: NDArray[np.float64],
        time: float = 0.0,
        flat_earth: bool = False,
    ) -> "State":
        """Create state from flat array."""
        return cls(
            position=arr[0:3],
            velocity=arr[3:6],
            quaternion=arr[6:10],
            angular_velocity=arr[10:13],
            mass=float(arr[13]),
            time=time,
            flat_earth=flat_earth,
        )

    def copy(self) -> "State":
        """Create a copy of this state."""
        return State(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quaternion=self.quaternion.copy(),
            angular_velocity=self.angular_velocity.copy(),
            mass=self.mass,
            time=self.time,
            flat_earth=self.flat_earth,
        )

    @property
    def dcm_body_to_inertial(self) -> NDArray[np.float64]:
        """Get DCM that transforms vectors from body to inertial frame."""
        return quaternion_to_dcm(quaternion_conjugate(self.quaternion))

    @property
    def dcm_inertial_to_body(self) -> NDArray[np.float64]:
        """Get DCM that transforms vectors from inertial to body frame."""
        return quaternion_to_dcm(self.quaternion)

    @property
    def euler_angles(self) -> tuple[float, float, float]:
        """Get Euler angles (roll, pitch, yaw) in radians."""
        return quaternion_to_euler(self.quaternion)

    @property
    def euler_angles_deg(self) -> tuple[float, float, float]:
        """Get Euler angles (roll, pitch, yaw) in degrees."""
        r, p, y = self.euler_angles
        return np.degrees(r), np.degrees(p), np.degrees(y)

    @property
    def altitude(self) -> float:
        """Get altitude above Earth's surface [m].

        For flat-earth states, returns z directly.
        For ECI states, returns distance from Earth center minus radius.
        """
        if self.flat_earth:
            return float(self.position[2])  # z is altitude
        else:
            R_earth = 6378137.0
            return float(np.linalg.norm(self.position) - R_earth)

    @property
    def speed(self) -> float:
        """Get speed magnitude [m/s]."""
        return float(np.linalg.norm(self.velocity))

    def velocity_body(self) -> NDArray[np.float64]:
        """Get velocity in body frame [m/s]."""
        return self.dcm_inertial_to_body @ self.velocity


@beartype
@dataclass
class StateDerivative:
    """Time derivative of the state vector.

    Attributes:
        position_dot: d(position)/dt = velocity [m/s]
        velocity_dot: d(velocity)/dt = acceleration [m/s^2]
        quaternion_dot: d(quaternion)/dt
        angular_velocity_dot: d(omega)/dt = angular acceleration [rad/s^2]
        mass_dot: d(mass)/dt = -mdot [kg/s] (negative for propellant consumption)
    """
    position_dot: NDArray[np.float64]
    velocity_dot: NDArray[np.float64]
    quaternion_dot: NDArray[np.float64]
    angular_velocity_dot: NDArray[np.float64]
    mass_dot: float

    def to_array(self) -> NDArray[np.float64]:
        """Convert to flat array for integration."""
        return np.concatenate([
            self.position_dot,
            self.velocity_dot,
            self.quaternion_dot,
            self.angular_velocity_dot,
            [self.mass_dot],
        ])

