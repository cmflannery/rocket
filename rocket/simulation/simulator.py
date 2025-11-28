"""Step-driven flight simulation for rocket vehicles.

Provides a clean simulation interface where external GNC code controls the loop.
The simulator maintains the "truth" state and propagates physics in response
to commanded inputs.

Architecture:
    GNC code owns the simulation loop and calls:
    - sim.get_state() -> current truth state
    - sim.get_environment() -> atmospheric conditions, gravity, etc.
    - sim.step(commands, dt) -> propagate physics

Example:
    >>> from rocket.simulation import Simulator, SimConfig
    >>> from rocket.vehicle import Vehicle
    >>> 
    >>> # Setup
    >>> vehicle = Vehicle(dry_mass=500, propellant_mass=4500, ...)
    >>> sim = Simulator(vehicle, SimConfig(gravity_model=GravityModel.SPHERICAL))
    >>> 
    >>> # GNC loop (this code would be in flight/)
    >>> while not mission_complete:
    ...     state = sim.get_state()
    ...     env = sim.get_environment()
    ...     
    ...     # Your guidance/control code here
    ...     thrust_cmd = guidance.compute(state, target)
    ...     gimbal_cmd = control.compute(state, thrust_cmd)
    ...     
    ...     sim.step(thrust=thrust_cmd, gimbal=gimbal_cmd, dt=0.02)
"""

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from beartype import beartype
from numba import njit
from numpy.typing import NDArray

from rocket.dynamics.state import (
    State,
    normalize_quaternion,
)
from rocket.environment.atmosphere import Atmosphere, AtmosphereResult
from rocket.environment.gravity import MU_EARTH, R_EARTH_EQ, Gravity, GravityModel

# =============================================================================
# Configuration
# =============================================================================


@beartype
@dataclass
class SimConfig:
    """Simulation configuration.
    
    Attributes:
        gravity_model: Gravity model fidelity
        include_atmosphere: Whether to model atmospheric drag
        flat_earth: Use flat-earth approximation (for short-range sims)
    """
    gravity_model: GravityModel = GravityModel.SPHERICAL
    include_atmosphere: bool = True
    flat_earth: bool = False


# =============================================================================
# Environment Data
# =============================================================================


class EnvironmentData(NamedTuple):
    """Environmental conditions at current state.
    
    Provided to GNC for sensor simulation and guidance decisions.
    """
    altitude: float              # Altitude above surface [m]
    gravity_magnitude: float     # Local gravity [m/s^2]
    gravity_vector: NDArray[np.float64]  # Gravity in inertial frame [m/s^2]
    atmosphere: AtmosphereResult | None  # Atmospheric conditions
    dynamic_pressure: float      # Dynamic pressure [Pa]
    mach_number: float          # Mach number [-]


# =============================================================================
# Command Inputs
# =============================================================================


@beartype
@dataclass
class ThrustCommand:
    """Thrust command from GNC.
    
    Attributes:
        magnitude: Thrust force magnitude [N]
        gimbal_pitch: Gimbal pitch angle [rad] (rotation about body Y)
        gimbal_yaw: Gimbal yaw angle [rad] (rotation about body Z)
    """
    magnitude: float = 0.0
    gimbal_pitch: float = 0.0
    gimbal_yaw: float = 0.0

    def to_body_force(self) -> NDArray[np.float64]:
        """Convert to force vector in body frame.
        
        Assumes thrust nominally along body +X axis.
        """
        # Small angle approximation for gimbal
        cp, sp = np.cos(self.gimbal_pitch), np.sin(self.gimbal_pitch)
        cy, sy = np.cos(self.gimbal_yaw), np.sin(self.gimbal_yaw)

        # Thrust direction in body frame
        fx = self.magnitude * cp * cy
        fy = self.magnitude * sp
        fz = -self.magnitude * cp * sy

        return np.array([fx, fy, fz])

    def to_moment(self, moment_arm: float = 2.0) -> NDArray[np.float64]:
        """Compute moment from thrust vectoring.
        
        Args:
            moment_arm: Distance from gimbal to CG [m]
        """
        # Moment = r x F (thrust offset creates torque)
        force = self.to_body_force()

        # Gimbal point is behind CG (negative X)
        r = np.array([-moment_arm, 0.0, 0.0])

        return np.cross(r, force)


# =============================================================================
# Numba-Optimized Integration
# =============================================================================


@njit(cache=True, fastmath=True)
def _quaternion_derivative(
    q0: float, q1: float, q2: float, q3: float,
    p: float, q: float, r: float,
) -> tuple[float, float, float, float]:
    """Numba-optimized quaternion kinematics."""
    return (
        0.5 * (-p*q1 - q*q2 - r*q3),
        0.5 * (p*q0 + r*q2 - q*q3),
        0.5 * (q*q0 - r*q1 + p*q3),
        0.5 * (r*q0 + q*q1 - p*q2),
    )


@njit(cache=True, fastmath=True)
def _euler_rotational_dynamics(
    px: float, py: float, pz: float,
    mx: float, my: float, mz: float,
    Ixx: float, Iyy: float, Izz: float,
) -> tuple[float, float, float]:
    """Numba-optimized Euler equations (diagonal inertia only)."""
    # Gyroscopic terms
    gx = (Iyy - Izz) * py * pz
    gy = (Izz - Ixx) * pz * px
    gz = (Ixx - Iyy) * px * py

    return (
        (mx - gx) / Ixx,
        (my - gy) / Iyy,
        (mz - gz) / Izz,
    )


@njit(cache=True, fastmath=True)
def _gravity_spherical(
    x: float, y: float, z: float,
    mu: float = MU_EARTH,
) -> tuple[float, float, float]:
    """Numba-optimized spherical gravity."""
    r_sq = x*x + y*y + z*z
    r = np.sqrt(r_sq)
    if r < 1e3:
        r = 1e3

    g_mag = mu / r_sq
    inv_r = 1.0 / r

    return (-g_mag * x * inv_r, -g_mag * y * inv_r, -g_mag * z * inv_r)


@njit(cache=True, fastmath=True)
def _rk4_derivatives(
    # State
    px: float, py: float, pz: float,
    vx: float, vy: float, vz: float,
    q0: float, q1: float, q2: float, q3: float,
    wx: float, wy: float, wz: float,
    mass: float,
    # Forces (body frame)
    fx_body: float, fy_body: float, fz_body: float,
    # Moments (body frame)
    mx: float, my: float, mz: float,
    # Mass rate
    mdot: float,
    # Inertia (diagonal)
    Ixx: float, Iyy: float, Izz: float,
    # Gravity
    gx: float, gy: float, gz: float,
) -> tuple[float, ...]:
    """Compute all state derivatives in one numba function."""
    # Position derivatives = velocity
    dpx, dpy, dpz = vx, vy, vz

    # Transform body force to inertial frame (quaternion rotation)
    # DCM from body to inertial using conjugate quaternion
    # This is the transpose of dcm_inertial_to_body
    q0c, q1c, q2c, q3c = q0, -q1, -q2, -q3

    # First row of DCM
    r00 = 1 - 2*(q2c*q2c + q3c*q3c)
    r01 = 2*(q1c*q2c - q0c*q3c)
    r02 = 2*(q1c*q3c + q0c*q2c)

    # Second row
    r10 = 2*(q1c*q2c + q0c*q3c)
    r11 = 1 - 2*(q1c*q1c + q3c*q3c)
    r12 = 2*(q2c*q3c - q0c*q1c)

    # Third row
    r20 = 2*(q1c*q3c - q0c*q2c)
    r21 = 2*(q2c*q3c + q0c*q1c)
    r22 = 1 - 2*(q1c*q1c + q2c*q2c)

    # Force in inertial = DCM @ force_body
    fx_inertial = r00*fx_body + r01*fy_body + r02*fz_body
    fy_inertial = r10*fx_body + r11*fy_body + r12*fz_body
    fz_inertial = r20*fx_body + r21*fy_body + r22*fz_body

    # Add gravity (already in inertial frame)
    ax = fx_inertial / mass + gx
    ay = fy_inertial / mass + gy
    az = fz_inertial / mass + gz

    # Quaternion derivatives
    dq0, dq1, dq2, dq3 = _quaternion_derivative(q0, q1, q2, q3, wx, wy, wz)

    # Angular velocity derivatives
    dwx, dwy, dwz = _euler_rotational_dynamics(wx, wy, wz, mx, my, mz, Ixx, Iyy, Izz)

    return (dpx, dpy, dpz, ax, ay, az, dq0, dq1, dq2, dq3, dwx, dwy, dwz, mdot)


@njit(cache=True, fastmath=True)
def _rk4_step_core(
    # Initial state
    px: float, py: float, pz: float,
    vx: float, vy: float, vz: float,
    q0: float, q1: float, q2: float, q3: float,
    wx: float, wy: float, wz: float,
    mass: float,
    time: float,
    # Forces (body frame)
    fx_body: float, fy_body: float, fz_body: float,
    # Moments (body frame)
    mx: float, my: float, mz: float,
    # Mass rate
    mdot: float,
    # Inertia (diagonal)
    Ixx: float, Iyy: float, Izz: float,
    # Gravity
    gx: float, gy: float, gz: float,
    # Time step
    dt: float,
) -> tuple[float, ...]:
    """Numba-optimized RK4 integration step."""

    # k1
    k1 = _rk4_derivatives(
        px, py, pz, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, mass,
        fx_body, fy_body, fz_body, mx, my, mz, mdot, Ixx, Iyy, Izz, gx, gy, gz
    )

    # k2 (at t + dt/2 with state + k1*dt/2)
    h = dt / 2
    px2 = px + k1[0]*h
    py2 = py + k1[1]*h
    pz2 = pz + k1[2]*h
    vx2 = vx + k1[3]*h
    vy2 = vy + k1[4]*h
    vz2 = vz + k1[5]*h
    q02 = q0 + k1[6]*h
    q12 = q1 + k1[7]*h
    q22 = q2 + k1[8]*h
    q32 = q3 + k1[9]*h
    wx2 = wx + k1[10]*h
    wy2 = wy + k1[11]*h
    wz2 = wz + k1[12]*h
    mass2 = mass + k1[13]*h

    # Normalize quaternion
    qnorm = np.sqrt(q02*q02 + q12*q12 + q22*q22 + q32*q32)
    if qnorm > 1e-10:
        q02 /= qnorm
        q12 /= qnorm
        q22 /= qnorm
        q32 /= qnorm

    # Recompute gravity at new position
    gx2, gy2, gz2 = _gravity_spherical(px2, py2, pz2)

    k2 = _rk4_derivatives(
        px2, py2, pz2, vx2, vy2, vz2, q02, q12, q22, q32, wx2, wy2, wz2, mass2,
        fx_body, fy_body, fz_body, mx, my, mz, mdot, Ixx, Iyy, Izz, gx2, gy2, gz2
    )

    # k3 (at t + dt/2 with state + k2*dt/2)
    px3 = px + k2[0]*h
    py3 = py + k2[1]*h
    pz3 = pz + k2[2]*h
    vx3 = vx + k2[3]*h
    vy3 = vy + k2[4]*h
    vz3 = vz + k2[5]*h
    q03 = q0 + k2[6]*h
    q13 = q1 + k2[7]*h
    q23 = q2 + k2[8]*h
    q33 = q3 + k2[9]*h
    wx3 = wx + k2[10]*h
    wy3 = wy + k2[11]*h
    wz3 = wz + k2[12]*h
    mass3 = mass + k2[13]*h

    qnorm = np.sqrt(q03*q03 + q13*q13 + q23*q23 + q33*q33)
    if qnorm > 1e-10:
        q03 /= qnorm
        q13 /= qnorm
        q23 /= qnorm
        q33 /= qnorm

    gx3, gy3, gz3 = _gravity_spherical(px3, py3, pz3)

    k3 = _rk4_derivatives(
        px3, py3, pz3, vx3, vy3, vz3, q03, q13, q23, q33, wx3, wy3, wz3, mass3,
        fx_body, fy_body, fz_body, mx, my, mz, mdot, Ixx, Iyy, Izz, gx3, gy3, gz3
    )

    # k4 (at t + dt with state + k3*dt)
    px4 = px + k3[0]*dt
    py4 = py + k3[1]*dt
    pz4 = pz + k3[2]*dt
    vx4 = vx + k3[3]*dt
    vy4 = vy + k3[4]*dt
    vz4 = vz + k3[5]*dt
    q04 = q0 + k3[6]*dt
    q14 = q1 + k3[7]*dt
    q24 = q2 + k3[8]*dt
    q34 = q3 + k3[9]*dt
    wx4 = wx + k3[10]*dt
    wy4 = wy + k3[11]*dt
    wz4 = wz + k3[12]*dt
    mass4 = mass + k3[13]*dt

    qnorm = np.sqrt(q04*q04 + q14*q14 + q24*q24 + q34*q34)
    if qnorm > 1e-10:
        q04 /= qnorm
        q14 /= qnorm
        q24 /= qnorm
        q34 /= qnorm

    gx4, gy4, gz4 = _gravity_spherical(px4, py4, pz4)

    k4 = _rk4_derivatives(
        px4, py4, pz4, vx4, vy4, vz4, q04, q14, q24, q34, wx4, wy4, wz4, mass4,
        fx_body, fy_body, fz_body, mx, my, mz, mdot, Ixx, Iyy, Izz, gx4, gy4, gz4
    )

    # Final update: y_new = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    c = dt / 6.0

    px_new = px + c * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    py_new = py + c * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    pz_new = pz + c * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    vx_new = vx + c * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
    vy_new = vy + c * (k1[4] + 2*k2[4] + 2*k3[4] + k4[4])
    vz_new = vz + c * (k1[5] + 2*k2[5] + 2*k3[5] + k4[5])
    q0_new = q0 + c * (k1[6] + 2*k2[6] + 2*k3[6] + k4[6])
    q1_new = q1 + c * (k1[7] + 2*k2[7] + 2*k3[7] + k4[7])
    q2_new = q2 + c * (k1[8] + 2*k2[8] + 2*k3[8] + k4[8])
    q3_new = q3 + c * (k1[9] + 2*k2[9] + 2*k3[9] + k4[9])
    wx_new = wx + c * (k1[10] + 2*k2[10] + 2*k3[10] + k4[10])
    wy_new = wy + c * (k1[11] + 2*k2[11] + 2*k3[11] + k4[11])
    wz_new = wz + c * (k1[12] + 2*k2[12] + 2*k3[12] + k4[12])
    mass_new = mass + c * (k1[13] + 2*k2[13] + 2*k3[13] + k4[13])

    # Final quaternion normalization
    qnorm = np.sqrt(q0_new*q0_new + q1_new*q1_new + q2_new*q2_new + q3_new*q3_new)
    if qnorm > 1e-10:
        q0_new /= qnorm
        q1_new /= qnorm
        q2_new /= qnorm
        q3_new /= qnorm

    return (
        px_new, py_new, pz_new,
        vx_new, vy_new, vz_new,
        q0_new, q1_new, q2_new, q3_new,
        wx_new, wy_new, wz_new,
        mass_new,
        time + dt,
    )


# =============================================================================
# Simulator
# =============================================================================


@beartype
@dataclass
class Simulator:
    """Step-driven flight simulator.
    
    Maintains truth state and propagates physics in response to commands.
    External GNC code controls the simulation loop.
    
    Example:
        >>> sim = Simulator.from_launch_pad(
        ...     latitude=28.5, longitude=-80.6,
        ...     vehicle_mass=40000.0,
        ...     inertia=np.diag([1e5, 1e5, 1e4]),
        ... )
        >>> 
        >>> for _ in range(10000):  # 200 seconds at 50 Hz
        ...     state = sim.get_state()
        ...     env = sim.get_environment()
        ...     
        ...     # Your GNC code here
        ...     thrust = compute_thrust(state, target)
        ...     
        ...     sim.step(thrust, dt=0.02)
    """
    state: State
    inertia: NDArray[np.float64]
    config: SimConfig = field(default_factory=SimConfig)

    # Internal
    _gravity: Gravity = field(init=False, repr=False)
    _atmosphere: Atmosphere = field(init=False, repr=False)
    _history: list[State] = field(default_factory=list, init=False, repr=False)
    _record_history: bool = field(default=True)

    def __post_init__(self) -> None:
        """Initialize environment models."""
        self._gravity = Gravity(model=self.config.gravity_model)
        self._atmosphere = Atmosphere()
        if self._record_history:
            self._history = [self.state.copy()]

    @classmethod
    def from_launch_pad(
        cls,
        latitude: float = 28.5,
        longitude: float = -80.6,
        altitude: float = 0.0,
        heading: float = 90.0,
        vehicle_mass: float = 10000.0,
        inertia: NDArray[np.float64] | None = None,
        config: SimConfig | None = None,
    ) -> "Simulator":
        """Create simulator initialized on launch pad.
        
        Args:
            latitude: Launch site latitude [degrees]
            longitude: Launch site longitude [degrees]
            altitude: Launch site altitude [m]
            heading: Initial heading (0=N, 90=E) [degrees]
            vehicle_mass: Initial vehicle mass [kg]
            inertia: 3x3 inertia tensor [kg*m^2]
            config: Simulation configuration
        """
        state = State.from_launch_pad(
            latitude_deg=latitude,
            longitude_deg=longitude,
            altitude_m=altitude,
            heading_deg=heading,
            mass_kg=vehicle_mass,
        )

        if inertia is None:
            # Default: slender rocket approximation
            inertia = np.diag([1e5, 1e5, 1e4])

        return cls(
            state=state,
            inertia=np.asarray(inertia, dtype=np.float64),
            config=config or SimConfig(),
        )

    @classmethod
    def from_orbit(
        cls,
        altitude: float,
        inclination: float = 0.0,
        raan: float = 0.0,
        true_anomaly: float = 0.0,
        vehicle_mass: float = 1000.0,
        inertia: NDArray[np.float64] | None = None,
        config: SimConfig | None = None,
    ) -> "Simulator":
        """Create simulator in circular orbit.
        
        Args:
            altitude: Orbital altitude [m]
            inclination: Orbital inclination [degrees]
            raan: Right ascension of ascending node [degrees]
            true_anomaly: Initial true anomaly [degrees]
            vehicle_mass: Vehicle mass [kg]
            inertia: 3x3 inertia tensor [kg*m^2]
            config: Simulation configuration
        """
        # Compute circular orbit state
        r = R_EARTH_EQ + altitude
        v = np.sqrt(MU_EARTH / r)

        inc = np.radians(inclination)
        omega = np.radians(raan)
        nu = np.radians(true_anomaly)

        # Position in orbital plane
        r_orbital = r * np.array([np.cos(nu), np.sin(nu), 0.0])
        v_orbital = v * np.array([-np.sin(nu), np.cos(nu), 0.0])

        # Rotation matrices
        R_omega = np.array([
            [np.cos(omega), -np.sin(omega), 0],
            [np.sin(omega), np.cos(omega), 0],
            [0, 0, 1],
        ])
        R_inc = np.array([
            [1, 0, 0],
            [0, np.cos(inc), -np.sin(inc)],
            [0, np.sin(inc), np.cos(inc)],
        ])

        # Transform to ECI
        R = R_omega @ R_inc
        position = R @ r_orbital
        velocity = R @ v_orbital

        # Attitude: prograde pointing
        from rocket.dynamics.state import dcm_to_quaternion

        body_x = velocity / np.linalg.norm(velocity)
        body_z = -position / np.linalg.norm(position)
        body_y = np.cross(body_z, body_x)
        body_y = body_y / np.linalg.norm(body_y)
        body_z = np.cross(body_x, body_y)

        dcm = np.column_stack([body_x, body_y, body_z])
        quaternion = dcm_to_quaternion(dcm.T)

        state = State(
            position=position,
            velocity=velocity,
            quaternion=quaternion,
            angular_velocity=np.zeros(3),
            mass=vehicle_mass,
            time=0.0,
            flat_earth=False,
        )

        if inertia is None:
            inertia = np.diag([1e3, 1e3, 1e2])

        return cls(
            state=state,
            inertia=np.asarray(inertia, dtype=np.float64),
            config=config or SimConfig(),
        )

    def get_state(self) -> State:
        """Get current truth state.
        
        Returns a copy to prevent external modification.
        """
        return self.state.copy()

    def get_environment(self) -> EnvironmentData:
        """Get environmental conditions at current state."""
        pos = self.state.position
        vel = self.state.velocity

        # Altitude
        if self.config.flat_earth:
            alt = pos[2]
        else:
            alt = np.linalg.norm(pos) - R_EARTH_EQ

        # Gravity
        g_vec = self._gravity.acceleration(pos)
        g_mag = np.linalg.norm(g_vec)

        # Atmosphere
        speed = np.linalg.norm(vel)
        if self.config.include_atmosphere and alt < 150000:
            atm = self._atmosphere.at_altitude(alt, speed)
            q_dyn = 0.5 * atm.density * speed * speed
            mach = speed / atm.speed_of_sound if atm.speed_of_sound > 0 else 0.0
        else:
            atm = None
            q_dyn = 0.0
            mach = 0.0

        return EnvironmentData(
            altitude=alt,
            gravity_magnitude=g_mag,
            gravity_vector=g_vec,
            atmosphere=atm,
            dynamic_pressure=q_dyn,
            mach_number=mach,
        )

    def step(
        self,
        thrust: ThrustCommand | float = 0.0,
        gimbal: tuple[float, float] | None = None,
        mass_rate: float = 0.0,
        aero_force: NDArray[np.float64] | None = None,
        aero_moment: NDArray[np.float64] | None = None,
        dt: float = 0.02,
    ) -> State:
        """Propagate physics by one time step.
        
        Args:
            thrust: ThrustCommand or thrust magnitude [N]
            gimbal: (pitch, yaw) gimbal angles [rad] if thrust is float
            mass_rate: Propellant mass flow rate (negative) [kg/s]
            aero_force: Aerodynamic force in body frame [N]
            aero_moment: Aerodynamic moment in body frame [N*m]
            dt: Time step [s]
            
        Returns:
            New state after integration
        """
        # Handle thrust input
        if isinstance(thrust, (int, float)):
            if gimbal is not None:
                thrust_cmd = ThrustCommand(
                    magnitude=float(thrust),
                    gimbal_pitch=gimbal[0],
                    gimbal_yaw=gimbal[1],
                )
            else:
                thrust_cmd = ThrustCommand(magnitude=float(thrust))
        else:
            thrust_cmd = thrust

        # Compute forces and moments in body frame
        force_body = thrust_cmd.to_body_force()
        moment_body = thrust_cmd.to_moment()

        if aero_force is not None:
            force_body = force_body + aero_force
        if aero_moment is not None:
            moment_body = moment_body + aero_moment

        # Get gravity
        g = self._gravity.acceleration(self.state.position)

        # Get diagonal inertia (assuming diagonal for numba)
        Ixx = self.inertia[0, 0]
        Iyy = self.inertia[1, 1]
        Izz = self.inertia[2, 2]

        # Transform force to body frame for gravity contribution
        # We apply gravity in the derivatives function

        # Call numba RK4
        result = _rk4_step_core(
            self.state.position[0], self.state.position[1], self.state.position[2],
            self.state.velocity[0], self.state.velocity[1], self.state.velocity[2],
            self.state.quaternion[0], self.state.quaternion[1],
            self.state.quaternion[2], self.state.quaternion[3],
            self.state.angular_velocity[0], self.state.angular_velocity[1],
            self.state.angular_velocity[2],
            self.state.mass,
            self.state.time,
            force_body[0], force_body[1], force_body[2],
            moment_body[0], moment_body[1], moment_body[2],
            mass_rate,
            Ixx, Iyy, Izz,
            g[0], g[1], g[2],
            dt,
        )

        # Update state
        self.state = State(
            position=np.array([result[0], result[1], result[2]]),
            velocity=np.array([result[3], result[4], result[5]]),
            quaternion=np.array([result[6], result[7], result[8], result[9]]),
            angular_velocity=np.array([result[10], result[11], result[12]]),
            mass=result[13],
            time=result[14],
            flat_earth=self.state.flat_earth,
        )

        if self._record_history:
            self._history.append(self.state.copy())

        return self.state

    def set_attitude(self, quaternion: NDArray[np.float64]) -> None:
        """Directly set vehicle attitude (for guidance testing)."""
        self.state.quaternion = normalize_quaternion(quaternion)

    def get_history(self) -> list[State]:
        """Get recorded state history."""
        return self._history.copy()

    def clear_history(self) -> None:
        """Clear recorded state history."""
        self._history = [self.state.copy()]

    @property
    def time(self) -> float:
        """Current simulation time [s]."""
        return self.state.time

    @property
    def altitude(self) -> float:
        """Current altitude [m]."""
        if self.config.flat_earth:
            return self.state.position[2]
        return np.linalg.norm(self.state.position) - R_EARTH_EQ


# =============================================================================
# Results and Analysis
# =============================================================================


@beartype
@dataclass
class SimulationResult:
    """Results from a completed simulation.
    
    Provides convenient access to trajectory data and analysis.
    """
    states: list[State]

    @property
    def time(self) -> NDArray[np.float64]:
        """Time array [s]."""
        return np.array([s.time for s in self.states])

    @property
    def position(self) -> NDArray[np.float64]:
        """Position history [m], shape (N, 3)."""
        return np.array([s.position for s in self.states])

    @property
    def velocity(self) -> NDArray[np.float64]:
        """Velocity history [m/s], shape (N, 3)."""
        return np.array([s.velocity for s in self.states])

    @property
    def altitude(self) -> NDArray[np.float64]:
        """Altitude history [m]."""
        return np.array([s.altitude for s in self.states])

    @property
    def speed(self) -> NDArray[np.float64]:
        """Speed history [m/s]."""
        return np.array([s.speed for s in self.states])

    @property
    def mass(self) -> NDArray[np.float64]:
        """Mass history [kg]."""
        return np.array([s.mass for s in self.states])

    @classmethod
    def from_simulator(cls, sim: Simulator) -> "SimulationResult":
        """Create result from simulator history."""
        return cls(states=sim.get_history())

    def to_dataframe(self):
        """Convert to Polars DataFrame."""
        import polars as pl

        return pl.DataFrame({
            "time": self.time,
            "altitude": self.altitude,
            "speed": self.speed,
            "mass": self.mass,
            "x": self.position[:, 0],
            "y": self.position[:, 1],
            "z": self.position[:, 2],
            "vx": self.velocity[:, 0],
            "vy": self.velocity[:, 1],
            "vz": self.velocity[:, 2],
        })
