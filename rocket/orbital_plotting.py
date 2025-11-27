"""Orbital visualization module using Plotly.

Provides interactive 3D visualizations for:
- Orbital trajectories (ECI frame)
- Ground tracks
- Launch vehicle telemetry (altitude, velocity, thrust)
- Orbital elements
- Animated playback of orbital maneuvers
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rocket.environment.gravity import R_EARTH_EQ, orbital_velocity


def plot_orbital_dashboard(
    data: dict,
    title: str = "Orbital Simulation",
    target_altitude: float | None = None,
) -> go.Figure:
    """Create a comprehensive dashboard for orbital simulations.

    Args:
        data: Dictionary containing simulation data arrays:
            - times: Time [s]
            - positions: Position [x,y,z] [m]
            - velocities: Velocity [vx,vy,vz] [m/s]
            - altitudes: Altitude [m] (optional, computed if missing)
            - masses: Mass [kg] (optional)
            - thrusts: Thrust [N] (optional)
            - stages: Stage number (optional)
        title: Dashboard title
        target_altitude: Target orbital altitude [m] for reference lines

    Returns:
        Plotly Figure object
    """
    # Extract data
    times = data['times']
    positions = data['positions']
    velocities = data['velocities']

    # Derived data
    if 'altitudes' in data:
        altitudes = data['altitudes'] / 1000.0  # km
    else:
        altitudes = (np.linalg.norm(positions, axis=1) - R_EARTH_EQ) / 1000.0

    speeds = np.linalg.norm(velocities, axis=1)

    # Calculate horizontal/vertical velocity
    r_hat = positions / np.linalg.norm(positions, axis=1)[:, None]
    v_vertical = np.sum(velocities * r_hat, axis=1)
    v_horizontal = np.linalg.norm(velocities - v_vertical[:, None] * r_hat, axis=1)

    # Ground track
    lats = np.degrees(np.arcsin(positions[:, 2] / np.linalg.norm(positions, axis=1)))
    lons = np.degrees(np.arctan2(positions[:, 1], positions[:, 0]))

    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "scene", "rowspan": 2}, {"type": "xy"}],
            [None, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=(
            "3D Trajectory (ECI Frame)",
            "Altitude & Velocities",
            "Flight Path Angle",
            "Ground Track",
            "Orbital Elements (Eccentricity)"
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.10,
    )

    # 1. 3D Trajectory (Row 1-2, Col 1)
    pos_km = positions / 1000.0

    # Hover text
    hover_text = [
        f"T+{t:.1f}s<br>Alt: {a:.1f} km<br>Speed: {s:.0f} m/s"
        for t, a, s in zip(times, altitudes, speeds, strict=True)
    ]

    fig.add_trace(go.Scatter3d(
        x=pos_km[:, 0], y=pos_km[:, 1], z=pos_km[:, 2],
        mode='lines',
        line=dict(color=speeds, colorscale='Plasma', width=5,
                  colorbar=dict(title="Speed (m/s)", x=0.45, len=0.5, y=0.8)),
        name="Trajectory",
        text=hover_text,
        hovertemplate="%{text}<extra></extra>"
    ), row=1, col=1)

    # Earth Sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 30)
    r_earth_km = R_EARTH_EQ / 1000.0
    x_earth = r_earth_km * np.outer(np.cos(u), np.sin(v))
    y_earth = r_earth_km * np.outer(np.sin(u), np.sin(v))
    z_earth = r_earth_km * np.outer(np.ones(50), np.cos(v))

    fig.add_trace(go.Surface(
        x=x_earth, y=y_earth, z=z_earth,
        colorscale=[[0, 'rgb(30,60,120)'], [1, 'rgb(30,80,140)']],
        showscale=False,
        opacity=0.6,
        name="Earth"
    ), row=1, col=1)

    # Start/End markers
    fig.add_trace(go.Scatter3d(
        x=[pos_km[0, 0]], y=[pos_km[0, 1]], z=[pos_km[0, 2]],
        mode='markers', marker=dict(size=6, color='lime'),
        name="Start"
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=[pos_km[-1, 0]], y=[pos_km[-1, 1]], z=[pos_km[-1, 2]],
        mode='markers', marker=dict(size=6, color='red'),
        name="End"
    ), row=1, col=1)

    # Optional: Stage 1 / Secondary Trajectory
    if 's1_positions' in data and len(data['s1_positions']) > 0:
        s1_pos = data['s1_positions'] / 1000.0
        fig.add_trace(go.Scatter3d(
            x=s1_pos[:, 0], y=s1_pos[:, 1], z=s1_pos[:, 2],
            mode='lines',
            line=dict(color='yellow', width=4, dash='dash'),
            name="Stage 1 (Ballistic)"
        ), row=1, col=1)

    # 2. Altitude & Velocities (Row 1, Col 2)
    fig.add_trace(go.Scatter(x=times, y=altitudes, name='Altitude (km)',
                             line=dict(color='cyan')), row=1, col=2)
    fig.add_trace(go.Scatter(x=times, y=v_horizontal, name='V_horiz (m/s)',
                             line=dict(color='orange')), row=1, col=2)

    if target_altitude:
        v_orb = orbital_velocity(target_altitude)
        fig.add_trace(go.Scatter(
            x=[times[0], times[-1]], y=[v_orb, v_orb],
            name=f'V_orbit ({v_orb:.0f})',
            line=dict(dash='dash', color='red')
        ), row=1, col=2)

    # 3. Flight Path Angle (Row 2, Col 2)
    # Calculate FPA
    fpa = np.degrees(np.arctan2(v_vertical, v_horizontal))
    fig.add_trace(go.Scatter(
        x=times, y=fpa, name='Flight Path Angle',
        line=dict(color='magenta')
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=[times[0], times[-1]], y=[0, 0],
        showlegend=False, line=dict(dash='dot', color='gray')
    ), row=2, col=2)

    # 4. Ground Track (Row 3, Col 1)
    fig.add_trace(go.Scatter(
        x=lons, y=lats, mode='lines',
        line=dict(color='orange', width=2),
        name='Ground Track'
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=[lons[0]], y=[lats[0]], mode='markers',
        marker=dict(size=8, color='lime'), name='Start'
    ), row=3, col=1)

    # 5. Optional Data (Row 3, Col 2)
    # Try to plot eccentricity if available, or Thrust/Mass
    if 'eccentricities' in data:
        fig.add_trace(go.Scatter(
            x=times, y=data['eccentricities'], name='Eccentricity',
            line=dict(color='yellow')
        ), row=3, col=2)
    elif 'thrusts' in data:
        fig.add_trace(go.Scatter(
            x=times, y=data['thrusts']/1000, name='Thrust (kN)',
            line=dict(color='red')
        ), row=3, col=2)

    # Layout updates
    fig.update_layout(
        title=dict(text=title, font=dict(size=24), x=0.5),
        height=1200,
        template='plotly_dark',
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
            aspectmode='data'
        )
    )

    # Axis labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_xaxes(title_text="Longitude (deg)", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=2)

    fig.update_yaxes(title_text="Value", row=1, col=2)
    fig.update_yaxes(title_text="Angle (deg)", row=2, col=2)
    fig.update_yaxes(title_text="Latitude (deg)", row=3, col=1)

    return fig


def plot_orbit_animation(
    data: dict,
    title: str = "Orbital Maneuver Animation",
    reference_orbits: list[dict] | None = None,
    show_earth_axes: bool = True,
    phase_colors: dict | None = None,
    phase_names: dict | None = None,
) -> go.Figure:
    """Create an animated 3D visualization of orbital maneuvers.

    General-purpose orbital animation with:
    - Animated 3D trajectory with playback controls
    - Earth with coordinate axes (X=red, Y=green, Z=blue)
    - Optional reference orbit circles
    - Phase-colored trajectory segments
    - Apogee/Perigee markers
    - Moving satellite marker

    Args:
        data: Dictionary containing simulation data arrays:
            - times: Time [s]
            - positions: Position [x,y,z] [m]
            - velocities: Velocity [vx,vy,vz] [m/s]
            - altitudes: Altitude [m] (optional)
            - eccentricities: Eccentricity (optional)
            - phases: Maneuver phase numbers (optional)
        title: Dashboard title
        reference_orbits: List of reference orbit dicts with keys:
            - altitude: Orbit altitude [m]
            - inclination: Orbit inclination [deg] (default 0)
            - color: Line color (default auto)
            - name: Legend name (default auto)
        show_earth_axes: Whether to show Earth coordinate axes
        phase_colors: Dict mapping phase number to color
        phase_names: Dict mapping phase number to display name

    Returns:
        Plotly Figure object with animation
    """
    times = data['times']
    positions = data['positions']
    velocities = data['velocities']
    altitudes = data.get('altitudes', np.linalg.norm(positions, axis=1) - R_EARTH_EQ)
    eccentricities = data.get('eccentricities', np.zeros(len(times)))
    phases = data.get('phases', np.ones(len(times)))

    pos_km = positions / 1000.0
    speeds = np.linalg.norm(velocities, axis=1)

    # Find apogee and perigee points
    apogee_idx = np.argmax(altitudes)
    perigee_idx = np.argmin(altitudes)

    # Default phase colors and names if not provided
    if phase_colors is None:
        phase_colors = {1: 'red', 2: 'gray', 3: 'orange', 4: 'lime'}
    if phase_names is None:
        phase_names = {1: 'Burn 1', 2: 'Coast', 3: 'Burn 2', 4: 'Final Orbit'}

    # Create figure
    fig = go.Figure()

    # =========================================================================
    # Earth with coordinate axes
    # =========================================================================

    # Earth sphere
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)
    r_earth_km = R_EARTH_EQ / 1000.0
    x_earth = r_earth_km * np.outer(np.cos(u), np.sin(v))
    y_earth = r_earth_km * np.outer(np.sin(u), np.sin(v))
    z_earth = r_earth_km * np.outer(np.ones(60), np.cos(v))

    # Create Earth with texture-like coloring
    fig.add_trace(go.Surface(
        x=x_earth, y=y_earth, z=z_earth,
        colorscale=[
            [0, 'rgb(20,50,100)'],    # Deep blue (ocean)
            [0.3, 'rgb(30,80,140)'],  # Medium blue
            [0.5, 'rgb(40,100,160)'], # Light blue
            [0.7, 'rgb(30,80,140)'],  # Medium blue
            [1, 'rgb(20,50,100)']     # Deep blue
        ],
        showscale=False,
        opacity=0.85,
        name="Earth",
        hoverinfo='skip'
    ))

    # Earth coordinate axes
    if show_earth_axes:
        axis_length = r_earth_km * 1.8  # Extend beyond Earth surface

        # X-axis (Red) - points to vernal equinox
        fig.add_trace(go.Scatter3d(
            x=[0, axis_length], y=[0, 0], z=[0, 0],
            mode='lines+text',
            line=dict(color='red', width=4),
            text=['', 'X'],
            textposition='top center',
            textfont=dict(size=14, color='red'),
            name='X-axis (Vernal Equinox)',
            hoverinfo='name'
        ))

        # Y-axis (Green) - 90° East in equatorial plane
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, axis_length], z=[0, 0],
            mode='lines+text',
            line=dict(color='lime', width=4),
            text=['', 'Y'],
            textposition='top center',
            textfont=dict(size=14, color='lime'),
            name='Y-axis (90° East)',
            hoverinfo='name'
        ))

        # Z-axis (Blue) - North pole
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[0, axis_length],
            mode='lines+text',
            line=dict(color='cyan', width=4),
            text=['', 'Z (North)'],
            textposition='top center',
            textfont=dict(size=14, color='cyan'),
            name='Z-axis (North Pole)',
            hoverinfo='name'
        ))

        # Equatorial plane circle (faint reference)
        theta_eq = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter3d(
            x=r_earth_km * 1.01 * np.cos(theta_eq),
            y=r_earth_km * 1.01 * np.sin(theta_eq),
            z=np.zeros(100),
            mode='lines',
            line=dict(color='rgba(255,255,255,0.3)', width=1),
            name='Equator',
            hoverinfo='name'
        ))

    # =========================================================================
    # Reference orbits
    # =========================================================================
    if reference_orbits:
        default_colors = ['rgba(0,255,255,0.5)', 'rgba(255,100,100,0.5)',
                          'rgba(100,255,100,0.5)', 'rgba(255,255,100,0.5)']
        for idx, orbit in enumerate(reference_orbits):
            alt = orbit.get('altitude', 400e3)
            inc_deg = orbit.get('inclination', 0)
            color = orbit.get('color', default_colors[idx % len(default_colors)])
            name = orbit.get('name', f'Orbit @ {alt/1000:.0f}km, {inc_deg:.0f}°')

            r_orbit = (R_EARTH_EQ + alt) / 1000.0
            inc_rad = np.radians(inc_deg)
            theta = np.linspace(0, 2*np.pi, 100)

            # Orbit in equatorial plane, then rotate by inclination around X-axis
            x_orb = r_orbit * np.cos(theta)
            y_orb = r_orbit * np.sin(theta) * np.cos(inc_rad)
            z_orb = r_orbit * np.sin(theta) * np.sin(inc_rad)

            fig.add_trace(go.Scatter3d(
                x=x_orb, y=y_orb, z=z_orb,
                mode='lines',
                line=dict(color=color, width=2, dash='dot'),
                name=name,
                hoverinfo='name'
            ))

    # =========================================================================
    # Trajectory colored by phase
    # =========================================================================

    # Plot each phase segment separately for clear coloring
    current_phase = phases[0] if len(phases) > 0 else 1
    phase_start = 0

    for i in range(1, len(phases)):
        if phases[i] != current_phase or i == len(phases) - 1:
            # End of phase segment
            end_idx = i if i < len(phases) - 1 else i + 1
            segment_pos = pos_km[phase_start:end_idx+1]
            segment_times = times[phase_start:end_idx+1]
            segment_alts = altitudes[phase_start:end_idx+1] / 1000.0
            segment_speeds = speeds[phase_start:end_idx+1]

            hover_text = [
                f"<b>{phase_names.get(current_phase, 'Unknown')}</b><br>"
                f"T+{t:.1f}s<br>"
                f"Alt: {a:.1f} km<br>"
                f"Speed: {s:.0f} m/s"
                for t, a, s in zip(segment_times, segment_alts, segment_speeds, strict=True)
            ]

            fig.add_trace(go.Scatter3d(
                x=segment_pos[:, 0], y=segment_pos[:, 1], z=segment_pos[:, 2],
                mode='lines',
                line=dict(color=phase_colors.get(current_phase, 'white'), width=6),
                name=phase_names.get(current_phase, f'Phase {current_phase}'),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                showlegend=True
            ))

            phase_start = i
            current_phase = phases[i]

    # =========================================================================
    # Key markers
    # =========================================================================

    # Start marker
    fig.add_trace(go.Scatter3d(
        x=[pos_km[0, 0]], y=[pos_km[0, 1]], z=[pos_km[0, 2]],
        mode='markers+text',
        marker=dict(size=8, color='lime', symbol='diamond'),
        text=['START'],
        textposition='top center',
        textfont=dict(size=10, color='lime'),
        name='Start',
        hovertemplate=f"<b>START</b><br>Alt: {altitudes[0]/1000:.1f} km<br>Speed: {speeds[0]:.0f} m/s<extra></extra>"
    ))

    # Apogee marker
    fig.add_trace(go.Scatter3d(
        x=[pos_km[apogee_idx, 0]], y=[pos_km[apogee_idx, 1]], z=[pos_km[apogee_idx, 2]],
        mode='markers+text',
        marker=dict(size=10, color='yellow', symbol='circle'),
        text=['APOGEE'],
        textposition='top center',
        textfont=dict(size=10, color='yellow'),
        name=f'Apogee ({altitudes[apogee_idx]/1000:.1f} km)',
        hovertemplate=f"<b>APOGEE</b><br>Alt: {altitudes[apogee_idx]/1000:.1f} km<br>Speed: {speeds[apogee_idx]:.0f} m/s<br>T+{times[apogee_idx]:.1f}s<extra></extra>"
    ))

    # Perigee marker (only if different from start)
    if perigee_idx > 10:  # Not at start
        fig.add_trace(go.Scatter3d(
            x=[pos_km[perigee_idx, 0]], y=[pos_km[perigee_idx, 1]], z=[pos_km[perigee_idx, 2]],
            mode='markers+text',
            marker=dict(size=10, color='cyan', symbol='circle'),
            text=['PERIGEE'],
            textposition='bottom center',
            textfont=dict(size=10, color='cyan'),
            name=f'Perigee ({altitudes[perigee_idx]/1000:.1f} km)',
            hovertemplate=f"<b>PERIGEE</b><br>Alt: {altitudes[perigee_idx]/1000:.1f} km<br>Speed: {speeds[perigee_idx]:.0f} m/s<br>T+{times[perigee_idx]:.1f}s<extra></extra>"
        ))

    # End marker
    fig.add_trace(go.Scatter3d(
        x=[pos_km[-1, 0]], y=[pos_km[-1, 1]], z=[pos_km[-1, 2]],
        mode='markers+text',
        marker=dict(size=8, color='red', symbol='diamond'),
        text=['END'],
        textposition='bottom center',
        textfont=dict(size=10, color='red'),
        name='End',
        hovertemplate=f"<b>END</b><br>Alt: {altitudes[-1]/1000:.1f} km<br>Speed: {speeds[-1]:.0f} m/s<extra></extra>"
    ))

    # =========================================================================
    # Animation: Moving satellite marker
    # =========================================================================

    # Downsample for smooth animation
    n_frames = min(200, len(times))
    frame_indices = np.linspace(0, len(times)-1, n_frames, dtype=int)

    # Create frames
    frames = []
    for i, idx in enumerate(frame_indices):
        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=[pos_km[idx, 0]], y=[pos_km[idx, 1]], z=[pos_km[idx, 2]],
                    mode='markers',
                    marker=dict(size=12, color='white', symbol='circle',
                               line=dict(color='black', width=2)),
                    name='Satellite'
                )
            ],
            name=str(i),
            traces=[len(fig.data)]  # Index of the satellite trace
        )
        frames.append(frame)

    fig.frames = frames

    # Add initial satellite position (will be animated)
    fig.add_trace(go.Scatter3d(
        x=[pos_km[0, 0]], y=[pos_km[0, 1]], z=[pos_km[0, 2]],
        mode='markers',
        marker=dict(size=12, color='white', symbol='circle',
                   line=dict(color='black', width=2)),
        name='Satellite',
        showlegend=False
    ))

    # =========================================================================
    # Layout with animation controls
    # =========================================================================

    # Calculate axis range to fit everything
    max_range = np.max(np.abs(pos_km)) * 1.1

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=24, color='white'),
            x=0.5
        ),
        scene=dict(
            xaxis=dict(title="X (km)", range=[-max_range, max_range], backgroundcolor='rgb(10,10,30)'),
            yaxis=dict(title="Y (km)", range=[-max_range, max_range], backgroundcolor='rgb(10,10,30)'),
            zaxis=dict(title="Z (km)", range=[-max_range, max_range], backgroundcolor='rgb(10,10,30)'),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8),
                up=dict(x=0, y=0, z=1)
            ),
            bgcolor='rgb(5,5,20)'
        ),
        template='plotly_dark',
        height=900,
        width=1200,
        showlegend=True,
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(0,0,0,0.7)',
            bordercolor='gray',
            borderwidth=1
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                x=0.1,
                y=0,
                xanchor="right",
                yanchor="top",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 50, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0}
                        }]
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    )
                ]
            )
        ],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "prefix": "Time: T+",
                "suffix": "s",
                "visible": True,
                "xanchor": "center",
                "font": {"size": 14, "color": "white"}
            },
            "transition": {"duration": 0},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.05,
            "y": 0,
            "steps": [
                {
                    "args": [[str(i)], {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }],
                    "label": f"{times[idx]:.0f}",
                    "method": "animate"
                }
                for i, idx in enumerate(frame_indices)
            ]
        }]
    )

    # Add annotations for key info
    fig.add_annotation(
        text=f"<b>Orbit Summary</b><br>"
             f"Perigee: {altitudes[perigee_idx]/1000:.0f} km<br>"
             f"Apogee: {altitudes[apogee_idx]/1000:.0f} km<br>"
             f"Ecc: {eccentricities[-1]:.5f}",
        showarrow=False,
        x=0.98, y=0.98,
        xref="paper", yref="paper",
        xanchor="right", yanchor="top",
        font=dict(size=12, color='white'),
        bgcolor='rgba(0,0,0,0.7)',
        bordercolor='gray',
        borderwidth=1,
        borderpad=8
    )

    return fig


def plot_hohmann_animation(
    data: dict,
    title: str = "Hohmann Transfer Animation",
    target_altitude: float | None = None,
    initial_altitude: float | None = None,
) -> go.Figure:
    """Create an animated visualization of a Hohmann transfer.

    This is a convenience wrapper around plot_orbit_animation for Hohmann transfers.

    Args:
        data: Simulation data dictionary
        title: Dashboard title
        target_altitude: Target orbital altitude [m]
        initial_altitude: Initial orbital altitude [m]

    Returns:
        Plotly Figure object with animation
    """
    # Build reference orbits
    reference_orbits = []
    if initial_altitude:
        reference_orbits.append({
            'altitude': initial_altitude,
            'inclination': 0,
            'color': 'rgba(0,255,255,0.4)',
            'name': f'Initial Orbit ({initial_altitude/1000:.0f} km)'
        })
    if target_altitude:
        reference_orbits.append({
            'altitude': target_altitude,
            'inclination': 0,
            'color': 'rgba(255,100,100,0.4)',
            'name': f'Target Orbit ({target_altitude/1000:.0f} km)'
        })

    # Default Hohmann phase names
    phase_names = {
        1: 'Burn 1 (Raise Apogee)',
        2: 'Coast to Apogee',
        3: 'Burn 2 (Circularize)',
        4: 'Final Orbit'
    }

    return plot_orbit_animation(
        data,
        title=title,
        reference_orbits=reference_orbits if reference_orbits else None,
        show_earth_axes=True,
        phase_names=phase_names
    )


def plot_orbital_transfer_dashboard(
    data: dict,
    title: str = "Orbital Transfer",
    target_altitude: float | None = None,
    initial_altitude: float | None = None,
) -> go.Figure:
    """Create a comprehensive dashboard for orbital transfers with 2D plots.

    Includes:
    - Altitude vs Time with burn phases highlighted
    - Eccentricity vs Time
    - Velocity vs Time
    - Apogee/Perigee vs Time

    Args:
        data: Simulation data dictionary
        title: Dashboard title
        target_altitude: Target altitude [m]
        initial_altitude: Initial altitude [m]

    Returns:
        Plotly Figure
    """
    times = data['times']
    altitudes = data.get('altitudes', np.zeros(len(times))) / 1000.0  # km
    eccentricities = data.get('eccentricities', np.zeros(len(times)))
    phases = data.get('phases', np.ones(len(times)))
    velocities = data['velocities']

    speeds = np.linalg.norm(velocities, axis=1)

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Altitude Profile",
            "Orbital Eccentricity",
            "Velocity",
            "Mission Phases"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    # Phase colors
    phase_colors = {1: 'red', 2: 'gray', 3: 'orange', 4: 'lime'}
    phase_names = {1: 'Burn 1', 2: 'Coast', 3: 'Burn 2', 4: 'Complete'}

    # 1. Altitude Profile with burn highlighting
    fig.add_trace(go.Scatter(
        x=times, y=altitudes,
        mode='lines',
        line=dict(color='cyan', width=2),
        name='Altitude'
    ), row=1, col=1)

    # Add target altitude reference
    if target_altitude:
        fig.add_hline(
            y=target_altitude/1000, line_dash="dash", line_color="red",
            annotation_text=f"Target: {target_altitude/1000:.0f} km",
            row=1, col=1
        )

    # Shade burn regions
    for phase_num, color in [(1, 'rgba(255,0,0,0.2)'), (3, 'rgba(255,165,0,0.2)')]:
        phase_mask = np.array(phases) == phase_num
        if np.any(phase_mask):
            # Find contiguous regions
            changes = np.diff(phase_mask.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1

            if phase_mask[0]:
                starts = np.insert(starts, 0, 0)
            if phase_mask[-1]:
                ends = np.append(ends, len(phases))

            for start, end in zip(starts, ends, strict=True):
                fig.add_vrect(
                    x0=times[start], x1=times[min(end, len(times)-1)],
                    fillcolor=color, layer="below", line_width=0,
                    row=1, col=1
                )

    # 2. Eccentricity
    fig.add_trace(go.Scatter(
        x=times, y=eccentricities,
        mode='lines',
        line=dict(color='yellow', width=2),
        name='Eccentricity'
    ), row=1, col=2)

    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=2)

    # 3. Velocity
    fig.add_trace(go.Scatter(
        x=times, y=speeds,
        mode='lines',
        line=dict(color='orange', width=2),
        name='Speed'
    ), row=2, col=1)

    # Orbital velocity references
    if initial_altitude:
        v_init = orbital_velocity(initial_altitude)
        fig.add_hline(
            y=v_init, line_dash="dot", line_color="cyan",
            annotation_text=f"V_circ @ {initial_altitude/1000:.0f}km",
            row=2, col=1
        )
    if target_altitude:
        v_target = orbital_velocity(target_altitude)
        fig.add_hline(
            y=v_target, line_dash="dot", line_color="red",
            annotation_text=f"V_circ @ {target_altitude/1000:.0f}km",
            row=2, col=1
        )

    # 4. Phase timeline (bar chart style)
    phase_durations = []
    current_phase = phases[0]
    phase_start_time = times[0]

    for i in range(1, len(phases)):
        if phases[i] != current_phase:
            phase_durations.append({
                'phase': current_phase,
                'start': phase_start_time,
                'end': times[i],
                'duration': times[i] - phase_start_time
            })
            current_phase = phases[i]
            phase_start_time = times[i]

    # Add final phase
    phase_durations.append({
        'phase': current_phase,
        'start': phase_start_time,
        'end': times[-1],
        'duration': times[-1] - phase_start_time
    })

    for pd in phase_durations:
        fig.add_trace(go.Bar(
            x=[pd['duration']],
            y=[phase_names.get(pd['phase'], f"Phase {pd['phase']}")],
            orientation='h',
            marker_color=phase_colors.get(pd['phase'], 'gray'),
            name=phase_names.get(pd['phase'], f"Phase {pd['phase']}"),
            text=[f"{pd['duration']:.1f}s"],
            textposition='inside',
            showlegend=False
        ), row=2, col=2)

    # Layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=20), x=0.5),
        template='plotly_dark',
        height=700,
        showlegend=True,
        barmode='stack'
    )

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="Duration (s)", row=2, col=2)

    fig.update_yaxes(title_text="Altitude (km)", row=1, col=1)
    fig.update_yaxes(title_text="Eccentricity", row=1, col=2)
    fig.update_yaxes(title_text="Speed (m/s)", row=2, col=1)
    fig.update_yaxes(title_text="Phase", row=2, col=2)

    return fig
