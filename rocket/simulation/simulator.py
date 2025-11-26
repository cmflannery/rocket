"""Main simulation harness for rocket vehicle flight.

Integrates all subsystems (vehicle, propulsion, environment, GNC) into
a complete flight simulation.

Example:
    >>> from rocket import design_engine, EngineInputs
    >>> from rocket.simulation import Simulator
    >>> from rocket.vehicle import Vehicle, MassProperties
    >>> from rocket.gnc import GravityTurnGuidance
    >>>
    >>> # Design engine
    >>> inputs = EngineInputs.from_propellants("LOX", "CH4", ...)
    >>> perf, geom = design_engine(inputs)
    >>>
    >>> # Create vehicle
    >>> vehicle = Vehicle(dry_mass=..., propellant_mass=..., engine=perf)
    >>>
    >>> # Create guidance
    >>> guidance = GravityTurnGuidance()
    >>>
    >>> # Run simulation
    >>> sim = Simulator(vehicle, guidance)
    >>> result = sim.run(t_final=300)
    >>>
    >>> # Analyze results
    >>> print(f"Max altitude: {result.max_altitude/1000:.1f} km")
    >>> result.plot_trajectory()
"""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype
from numpy.typing import NDArray

from rocket.dynamics.rigid_body import DynamicsConfig, RigidBodyDynamics, rk4_step
from rocket.dynamics.state import State, StateDerivative
from rocket.environment.atmosphere import Atmosphere
from rocket.environment.gravity import GravityModel
from rocket.gnc.control.tvc import TVCController
from rocket.gnc.guidance.gravity_turn import GravityTurnGuidance
from rocket.propulsion.throttle_model import GimbalModel, ThrottleModel
from rocket.vehicle.aerodynamics import SimpleAero
from rocket.vehicle.mass import Vehicle

# =============================================================================
# Simulation Result
# =============================================================================


@beartype
@dataclass
class SimulationResult:
    """Results from a flight simulation.

    Contains the complete state history and derived quantities.

    Attributes:
        states: List of State objects at each time step
        time: Time array [s]
        position: Position history [m]
        velocity: Velocity history [m/s]
        altitude: Altitude history [m]
        speed: Speed history [m/s]
        mass: Mass history [kg]
        pitch: Pitch angle history [rad]
        throttle: Throttle history
        gimbal: Gimbal angle history (pitch, yaw) [rad]
    """
    states: list[State]
    throttle_history: list[float] = field(default_factory=list)
    gimbal_history: list[tuple[float, float]] = field(default_factory=list)

    @property
    def time(self) -> NDArray[np.float64]:
        """Time array [s]."""
        return np.array([s.time for s in self.states])

    @property
    def position(self) -> NDArray[np.float64]:
        """Position history [m]."""
        return np.array([s.position for s in self.states])

    @property
    def velocity(self) -> NDArray[np.float64]:
        """Velocity history [m/s]."""
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

    @property
    def pitch(self) -> NDArray[np.float64]:
        """Pitch angle history [rad]."""
        return np.array([s.euler_angles[1] for s in self.states])

    @property
    def max_altitude(self) -> float:
        """Maximum altitude achieved [m]."""
        return float(np.max(self.altitude))

    @property
    def max_speed(self) -> float:
        """Maximum speed achieved [m/s]."""
        return float(np.max(self.speed))

    @property
    def max_dynamic_pressure(self) -> float:
        """Maximum dynamic pressure (Max-Q) [Pa]."""
        atm = Atmosphere()
        q = np.array([
            atm.dynamic_pressure(alt, spd)
            for alt, spd in zip(self.altitude, self.speed, strict=False)
        ])
        return float(np.max(q))

    @property
    def burnout_time(self) -> float | None:
        """Time of engine burnout [s]."""
        mass = self.mass
        for i in range(1, len(mass)):
            if mass[i] == mass[i-1]:  # Mass stopped decreasing
                return self.time[i]
        return None

    def to_dataframe(self):
        """Convert to Polars DataFrame."""
        import polars as pl

        return pl.DataFrame({
            "time": self.time,
            "altitude": self.altitude,
            "speed": self.speed,
            "mass": self.mass,
            "pitch_deg": np.degrees(self.pitch),
            "x": self.position[:, 0],
            "y": self.position[:, 1],
            "z": self.position[:, 2],
            "vx": self.velocity[:, 0],
            "vy": self.velocity[:, 1],
            "vz": self.velocity[:, 2],
        })

    def save_csv(self, path: str | Path) -> None:
        """Save results to CSV file."""
        self.to_dataframe().write_csv(path)

    def plot_trajectory(self, save_path: str | Path | None = None) -> plt.Figure:
        """Generate trajectory plots.

        Args:
            save_path: Optional path to save figure

        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='#1a1a2e')

        color = '#16c79a'

        # 1. Altitude vs Time
        ax1 = axes[0, 0]
        ax1.set_facecolor('#16213e')
        ax1.plot(self.time, self.altitude / 1000, color=color, linewidth=2)
        ax1.set_xlabel('Time [s]', color='white')
        ax1.set_ylabel('Altitude [km]', color='white')
        ax1.set_title('Altitude', color='white', fontweight='bold')
        self._style_axis(ax1)

        # Mark max altitude
        max_idx = np.argmax(self.altitude)
        ax1.scatter([self.time[max_idx]], [self.altitude[max_idx]/1000],
                   color='#e94560', s=100, zorder=5)
        ax1.annotate(f'{self.max_altitude/1000:.1f} km',
                    (self.time[max_idx], self.altitude[max_idx]/1000),
                    xytext=(10, 10), textcoords='offset points', color='white')

        # 2. Speed vs Time
        ax2 = axes[0, 1]
        ax2.set_facecolor('#16213e')
        ax2.plot(self.time, self.speed, color=color, linewidth=2)
        ax2.set_xlabel('Time [s]', color='white')
        ax2.set_ylabel('Speed [m/s]', color='white')
        ax2.set_title('Speed', color='white', fontweight='bold')
        self._style_axis(ax2)

        # 3. Pitch vs Time
        ax3 = axes[1, 0]
        ax3.set_facecolor('#16213e')
        ax3.plot(self.time, np.degrees(self.pitch), color=color, linewidth=2)
        ax3.set_xlabel('Time [s]', color='white')
        ax3.set_ylabel('Pitch [deg]', color='white')
        ax3.set_title('Pitch Angle', color='white', fontweight='bold')
        ax3.set_ylim(-10, 100)
        self._style_axis(ax3)

        # 4. Mass vs Time
        ax4 = axes[1, 1]
        ax4.set_facecolor('#16213e')
        ax4.plot(self.time, self.mass, color=color, linewidth=2)
        ax4.set_xlabel('Time [s]', color='white')
        ax4.set_ylabel('Mass [kg]', color='white')
        ax4.set_title('Vehicle Mass', color='white', fontweight='bold')
        self._style_axis(ax4)

        fig.suptitle('Flight Trajectory', color='white', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')

        return fig

    def _style_axis(self, ax) -> None:
        """Apply consistent styling to axis."""
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='white')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('white')


# =============================================================================
# Simulator
# =============================================================================


@beartype
@dataclass
class Simulator:
    """Main flight simulation harness.

    Integrates vehicle dynamics with GNC to simulate complete flights.

    Example:
        >>> sim = Simulator(vehicle, guidance)
        >>> result = sim.run(t_final=300)
    """
    vehicle: Vehicle
    guidance: GravityTurnGuidance
    throttle_model: ThrottleModel | None = None
    gimbal_model: GimbalModel | None = None
    tvc_controller: TVCController | None = None
    aerodynamics: SimpleAero | None = None

    # Simulation parameters
    dt: float = 0.01  # Time step [s]

    def __post_init__(self) -> None:
        """Initialize subsystems."""
        # Create throttle model from engine if not provided
        if self.throttle_model is None and self.vehicle.engine is not None:
            self.throttle_model = ThrottleModel(
                engine=self.vehicle.engine,
                min_throttle=0.4,
            )

        # Create default gimbal model
        if self.gimbal_model is None:
            self.gimbal_model = GimbalModel(
                max_gimbal_angle=np.radians(6),
                gimbal_rate=np.radians(20),
            )

        # Create default TVC controller
        if self.tvc_controller is None:
            self.tvc_controller = TVCController()

        # Create default aerodynamics
        if self.aerodynamics is None:
            self.aerodynamics = SimpleAero(
                Cd0=0.3,
                reference_area=self.vehicle.reference_area,
            )

    @beartype
    def run(
        self,
        initial_state: State | None = None,
        t_final: float = 300.0,
        max_steps: int = 100000,
    ) -> SimulationResult:
        """Run the flight simulation.

        Args:
            initial_state: Initial state (defaults to launch pad)
            t_final: Maximum simulation time [s]
            max_steps: Maximum number of steps

        Returns:
            SimulationResult with full state history
        """
        # Initialize state
        if initial_state is None:
            initial_state = State.from_flat_earth(
                z=0.0,
                pitch_deg=90.0,  # Pointing up
                mass_kg=self.vehicle.total_mass,
            )

        # Get inertia from vehicle
        props = self.vehicle.current_properties

        # Configure dynamics model
        # Use constant gravity for flat-earth, spherical for ECI
        if initial_state.flat_earth:
            config = DynamicsConfig(gravity_model=GravityModel.CONSTANT)
        else:
            config = DynamicsConfig(gravity_model=GravityModel.SPHERICAL)

        # Create dynamics model
        dynamics = RigidBodyDynamics(
            inertia=props.inertia,
            config=config,
        )

        # Atmosphere for aero
        atmosphere = Atmosphere()

        # Initialize tracking
        states = [initial_state]
        throttle_history = []
        gimbal_history = []

        state = initial_state.copy()
        gimbal = (0.0, 0.0)

        n_steps = int(t_final / self.dt)
        n_steps = min(n_steps, max_steps)

        for _ in range(n_steps):
            # Get guidance commands
            throttle_cmd = self.guidance.throttle_command(state)
            attitude_cmd = self.guidance.attitude_command(state)

            # TVC control
            pitch_cmd, yaw_cmd = self.tvc_controller.update_attitude(
                state, attitude_cmd[1], attitude_cmd[2], self.dt
            )
            gimbal = (pitch_cmd, yaw_cmd)

            # Get thrust and mass flow
            thrust_mag = 0.0
            mdot = 0.0
            if self.throttle_model and throttle_cmd > 0 and state.mass > self.vehicle.dry_mass.mass:
                thrust_mag, mdot = self.throttle_model.at(throttle_cmd, state.altitude)

            # Compute thrust vector with gimbal
            if thrust_mag > 0 and self.gimbal_model:
                thrust_body = self.gimbal_model.thrust_vector(thrust_mag, gimbal[0], gimbal[1])
                thrust_moment = self.gimbal_model.moment(thrust_mag, gimbal[0], gimbal[1])
            else:
                thrust_body = np.array([0.0, 0.0, 0.0])
                thrust_moment = np.array([0.0, 0.0, 0.0])

            # Aerodynamics
            aero_force = np.array([0.0, 0.0, 0.0])
            if self.aerodynamics and state.altitude < 100000:
                atm = atmosphere.at_altitude(state.altitude, state.speed)
                if atm.density > 1e-6:
                    v_body = state.velocity_body()
                    aero_force = self.aerodynamics.forces_body(
                        v_body, atm.density, atm.speed_of_sound
                    )

            # Compute derivatives
            state_dot = dynamics.derivatives(
                state=state,
                thrust_body=thrust_body,
                moment_body=thrust_moment,
                mass_rate=-mdot,  # Negative because consuming propellant
                aero_force_body=aero_force,
            )

            # Integrate
            def deriv_fn(s: State, _state_dot: StateDerivative = state_dot) -> StateDerivative:
                return _state_dot  # Use computed derivatives

            state = rk4_step(state, self.dt, deriv_fn)

            # Update vehicle propellant
            self.vehicle.consume_propellant(mdot * self.dt)

            # Track
            states.append(state)
            throttle_history.append(throttle_cmd)
            gimbal_history.append(gimbal)

            # Check termination
            if self.guidance.is_complete(state):
                break
            if state.altitude < -100:  # Below ground
                break
            if state.mass <= self.vehicle.dry_mass.mass:
                # Burnout - continue coasting
                pass

        return SimulationResult(
            states=states,
            throttle_history=throttle_history,
            gimbal_history=gimbal_history,
        )

