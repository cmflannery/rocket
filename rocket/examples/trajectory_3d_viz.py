#!/usr/bin/env python
"""Interactive 3D trajectory visualization.

Provides a web-based dashboard for visualizing rocket trajectories in 3D
with real-time playback, telemetry display, and interactive controls.

Requires: pip install plotly dash
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rocket import EngineInputs, design_engine
from rocket.gnc.guidance import GravityTurnGuidance
from rocket.simulation import Simulator
from rocket.units import kilonewtons, megapascals
from rocket.vehicle import MassProperties, Vehicle


def run_simulation():
    """Run a trajectory simulation and return the results."""
    print("Designing engine...")
    inputs = EngineInputs.from_propellants(
        oxidizer="LOX",
        fuel="CH4",
        thrust=kilonewtons(50),
        chamber_pressure=megapascals(5.0),
        mixture_ratio=3.0,
        name="Sounding Rocket Engine",
    )
    performance, geometry = design_engine(inputs)

    print("Creating vehicle...")
    dry_mass = MassProperties.from_principal(
        mass=200.0,
        cg=[0.0, 0.0, 2.0],
        Ixx=600.0, Iyy=600.0, Izz=4.0,
    )
    vehicle = Vehicle(
        dry_mass=dry_mass,
        initial_propellant_mass=800.0,
        propellant_cg=np.array([0.0, 0.0, 1.5]),
        engine=performance,
        reference_area=np.pi * 0.2**2,
        reference_length=6.0,
    )

    print("Setting up guidance...")
    guidance = GravityTurnGuidance(
        pitch_kick=np.radians(3.0),
        pitch_kick_time=5.0,
        pitch_kick_duration=10.0,
        target_altitude=100000.0,
    )

    print("Running simulation...")
    sim = Simulator(vehicle=vehicle, guidance=guidance, dt=0.01)
    result = sim.run(t_final=80.0)

    return result


def create_3d_trajectory_figure(result, title="Rocket Trajectory"):
    """Create an interactive 3D trajectory visualization.

    Args:
        result: SimulationResult from the simulator
        title: Figure title

    Returns:
        Plotly Figure
    """
    # Extract data
    time = result.time
    x = result.position[:, 0] / 1000  # Convert to km
    y = result.position[:, 1] / 1000
    z = result.altitude / 1000  # Already altitude in m, convert to km
    speed = result.speed
    pitch = np.degrees(result.pitch)

    # Create figure with subplots - 3D takes up most of the space
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "scene", "rowspan": 2}, {"type": "xy"}],
            [None, {"type": "xy"}],
        ],
        subplot_titles=("", "Altitude & Speed", "Pitch Angle"),
        column_widths=[0.75, 0.25],  # 3D view gets 75% of width
        row_heights=[0.5, 0.5],
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
    )

    # 3D trajectory line with default colorscale
    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(
                color=speed,
                colorscale='Viridis',
                width=6,
                colorbar=dict(
                    title="Speed (m/s)",
                    x=0.65,
                    len=0.9,
                ),
            ),
            name="Trajectory",
            hovertemplate=(
                "X: %{x:.2f} km<br>"
                "Y: %{y:.2f} km<br>"
                "Alt: %{z:.2f} km<br>"
                "<extra></extra>"
            ),
        ),
        row=1, col=1,
    )

    # Start point
    fig.add_trace(
        go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode='markers',
            marker=dict(size=10, color='green', symbol='circle'),
            name="Launch",
            showlegend=True,
        ),
        row=1, col=1,
    )

    # Apogee point
    apogee_idx = np.argmax(z)
    fig.add_trace(
        go.Scatter3d(
            x=[x[apogee_idx]], y=[y[apogee_idx]], z=[z[apogee_idx]],
            mode='markers',
            marker=dict(size=12, color='red', symbol='diamond'),
            name=f"Apogee ({z[apogee_idx]:.1f} km)",
            showlegend=True,
        ),
        row=1, col=1,
    )

    # Altitude vs Time
    fig.add_trace(
        go.Scatter(
            x=time, y=z,
            mode='lines',
            name='Altitude (km)',
            line=dict(width=2),
        ),
        row=1, col=2,
    )

    # Speed vs Time (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=time, y=speed,
            mode='lines',
            name='Speed (m/s)',
            line=dict(width=2),
            yaxis='y2',
        ),
        row=1, col=2,
    )

    # Pitch vs Time
    fig.add_trace(
        go.Scatter(
            x=time, y=pitch,
            mode='lines',
            name='Pitch (°)',
            line=dict(width=2),
        ),
        row=2, col=2,
    )

    # Update 3D scene layout - clean white background
    fig.update_scenes(
        xaxis_title="X (km)",
        yaxis_title="Y (km)",
        zaxis_title="Altitude (km)",
        aspectmode='data',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=0.7),
        ),
    )

    # Update 2D plot layouts
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Altitude (km)", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Pitch (°)", row=2, col=2)

    # Create secondary y-axis for speed
    fig.update_layout(
        yaxis2=dict(
            title="Speed (m/s)",
            overlaying='y3',
            side='right',
        ),
    )

    # Overall layout - white background, large height for big 3D view
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=24),
            x=0.5,
        ),
        height=900,  # Taller for bigger 3D view
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            x=0.01,
            y=0.99,
        ),
    )

    return fig


def create_animated_trajectory(result, fps=30, speed_multiplier=5):
    """Create an animated trajectory visualization.

    Args:
        result: SimulationResult from the simulator
        fps: Frames per second for animation
        speed_multiplier: How much faster than real-time

    Returns:
        Plotly Figure with animation
    """
    time = result.time
    x = result.position[:, 0] / 1000
    y = result.position[:, 1] / 1000
    z = result.altitude / 1000
    speed = result.speed

    # Subsample for animation performance
    step = max(1, len(time) // (fps * int(time[-1] / speed_multiplier)))
    indices = range(0, len(time), step)

    # Create frames
    frames = []
    for i in indices:
        frame = go.Frame(
            data=[
                # Trajectory line up to current point
                go.Scatter3d(
                    x=x[:i+1], y=y[:i+1], z=z[:i+1],
                    mode='lines',
                    line=dict(color=speed[:i+1], colorscale='Viridis', width=6),
                    name="Trajectory",
                ),
                # Current position marker
                go.Scatter3d(
                    x=[x[i]], y=[y[i]], z=[z[i]],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='diamond'),
                    name="Rocket",
                ),
            ],
            name=str(i),
        )
        frames.append(frame)

    # Initial figure
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=[x[0]], y=[y[0]], z=[z[0]],
                mode='lines',
                line=dict(color=[speed[0]], colorscale='Viridis', width=6),
            ),
            go.Scatter3d(
                x=[x[0]], y=[y[0]], z=[z[0]],
                mode='markers',
                marker=dict(size=15, color='red', symbol='diamond'),
            ),
        ],
        frames=frames,
    )

    # Add play/pause buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0.1,
                x=0.1,
                xanchor="left",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, dict(
                            frame=dict(duration=1000/fps, redraw=True),
                            fromcurrent=True,
                            mode="immediate",
                        )],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                        )],
                    ),
                ],
            ),
        ],
        sliders=[
            dict(
                active=0,
                steps=[
                    dict(
                        args=[[str(i)], dict(
                            frame=dict(duration=0, redraw=True),
                            mode="immediate",
                        )],
                        label=f"{time[i]:.1f}s",
                        method="animate",
                    )
                    for i in indices
                ],
                currentvalue=dict(
                    font=dict(size=16),
                    prefix="Time: ",
                    suffix=" s",
                    visible=True,
                ),
                len=0.8,
                x=0.1,
                y=0,
            ),
        ],
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Altitude (km)",
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.7)),
        ),
        title=dict(
            text="Animated Rocket Trajectory",
            font=dict(size=24),
        ),
        height=900,  # Large for prominent 3D view
    )

    return fig


def main():
    """Run simulation and create visualizations."""
    print("=" * 60)
    print("3D TRAJECTORY VISUALIZATION")
    print("=" * 60)

    # Run simulation
    result = run_simulation()

    print(f"\nSimulation complete: {len(result.states)} time steps")
    print(f"Max altitude: {result.max_altitude/1000:.1f} km")
    print(f"Max speed: {result.max_speed:.0f} m/s")

    # Create static 3D visualization
    print("\nCreating 3D visualization...")
    fig = create_3d_trajectory_figure(result, "Sounding Rocket Trajectory")

    # Save to HTML (interactive)
    output_path = "outputs/trajectory_3d.html"
    fig.write_html(output_path)
    print(f"Saved interactive 3D visualization to: {output_path}")

    # Create animated version
    print("Creating animated visualization...")
    anim_fig = create_animated_trajectory(result, fps=30, speed_multiplier=5)
    anim_path = "outputs/trajectory_animated.html"
    anim_fig.write_html(anim_path)
    print(f"Saved animated visualization to: {anim_path}")

    # Show the static figure (opens in browser)
    print("\nOpening visualization in browser...")
    fig.show()

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print("\nTips:")
    print("- Drag to rotate the 3D view")
    print("- Scroll to zoom")
    print("- Double-click to reset view")
    print("- Hover for telemetry data")


if __name__ == "__main__":
    main()

