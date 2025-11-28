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

from rocket.environment.gravity import MU_EARTH, R_EARTH_EQ, orbital_velocity


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
    booster_data: dict | None = None,
) -> go.Figure:
    """Create an animated 3D visualization of orbital maneuvers.

    General-purpose orbital animation with:
    - Animated 3D trajectory with playback controls
    - Earth with coordinate axes (X=red, Y=green, Z=blue)
    - Optional reference orbit circles
    - Phase-colored trajectory segments
    - Apogee/Perigee markers
    - Moving satellite marker
    - Optional booster/first stage trajectory

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
            - raan: Right Ascension of Ascending Node [deg] (default 0)
            - color: Line color (default auto)
            - name: Legend name (default auto)
        show_earth_axes: Whether to show Earth coordinate axes
        phase_colors: Dict mapping phase number to color
        phase_names: Dict mapping phase number to display name
        booster_data: Optional dict with first stage trajectory:
            - positions: Position [x,y,z] [m]
            - times: Time [s]
            - altitudes: Altitude [m]

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
            raan_deg = orbit.get('raan', 0)  # Right Ascension of Ascending Node
            color = orbit.get('color', default_colors[idx % len(default_colors)])
            name = orbit.get('name', f'Orbit @ {alt/1000:.0f}km, {inc_deg:.0f}°')

            r_orbit = (R_EARTH_EQ + alt) / 1000.0
            inc_rad = np.radians(inc_deg)
            raan_rad = np.radians(raan_deg)
            theta = np.linspace(0, 2*np.pi, 100)

            # Orbit in equatorial plane
            x_eq = r_orbit * np.cos(theta)
            y_eq = r_orbit * np.sin(theta)
            z_eq = np.zeros(100)

            # Rotate by inclination around X-axis (ascending node direction)
            x_inc = x_eq
            y_inc = y_eq * np.cos(inc_rad)
            z_inc = y_eq * np.sin(inc_rad)

            # Rotate by RAAN around Z-axis
            x_orb = x_inc * np.cos(raan_rad) - y_inc * np.sin(raan_rad)
            y_orb = x_inc * np.sin(raan_rad) + y_inc * np.cos(raan_rad)
            z_orb = z_inc

            fig.add_trace(go.Scatter3d(
                x=x_orb, y=y_orb, z=z_orb,
                mode='lines',
                line=dict(color=color, width=2, dash='dot'),
                name=name,
                hoverinfo='name'
            ))

    # =========================================================================
    # Booster trajectory (if provided)
    # =========================================================================
    if booster_data is not None and booster_data.get('positions') is not None:
        s1_pos = booster_data['positions']
        s1_times = booster_data.get('times', np.arange(len(s1_pos)))
        s1_alts = booster_data.get('altitudes', np.linalg.norm(s1_pos, axis=1) - R_EARTH_EQ)

        # Filter to valid (above ground) positions
        valid_mask = s1_alts > 0
        if np.any(valid_mask):
            s1_pos_valid = s1_pos[valid_mask] / 1000.0
            s1_times_valid = s1_times[valid_mask]
            s1_alts_valid = s1_alts[valid_mask] / 1000.0

            hover_text = [
                f"<b>Stage 1 (Booster)</b><br>"
                f"T+{t:.1f}s<br>"
                f"Alt: {a:.1f} km"
                for t, a in zip(s1_times_valid, s1_alts_valid, strict=True)
            ]

            fig.add_trace(go.Scatter3d(
                x=s1_pos_valid[:, 0], y=s1_pos_valid[:, 1], z=s1_pos_valid[:, 2],
                mode='lines',
                line=dict(color='rgba(255,150,50,0.7)', width=4, dash='dash'),
                name='Stage 1 (Booster)',
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                showlegend=True
            ))

            # Add booster impact/end marker
            fig.add_trace(go.Scatter3d(
                x=[s1_pos_valid[-1, 0]], y=[s1_pos_valid[-1, 1]], z=[s1_pos_valid[-1, 2]],
                mode='markers+text',
                marker=dict(size=8, color='orange', symbol='x'),
                text=['S1 END'],
                textposition='bottom center',
                textfont=dict(size=10, color='orange'),
                name=f'S1 End ({s1_alts_valid[-1]:.0f} km)',
                hovertemplate=f"<b>Stage 1 End</b><br>Alt: {s1_alts_valid[-1]:.1f} km<br>T+{s1_times_valid[-1]:.0f}s<extra></extra>"
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
    # Animation: SpaceX-style with combined stack before staging, split after
    # =========================================================================

    # Compute additional telemetry for display
    # Inclinations (if not provided)
    if 'inclinations' in data:
        inclinations = data['inclinations']
    else:
        inclinations = np.zeros(len(times))
        for i in range(len(times)):
            h_vec = np.cross(positions[i], velocities[i])
            h_mag = np.linalg.norm(h_vec)
            if h_mag > 0:
                inclinations[i] = np.degrees(np.arccos(np.clip(h_vec[2] / h_mag, -1, 1)))

    # Prepare booster data for animation if available
    has_booster = (booster_data is not None and
                   booster_data.get('positions') is not None and
                   len(booster_data['positions']) > 0)

    # Determine staging time from the data
    # Stage 2 data starts at staging, Stage 1 continues through landing
    staging_time = times[0] if len(times) > 0 else 0  # Default to start of S2 data

    if has_booster:
        s1_pos_anim = booster_data['positions'] / 1000.0  # km
        s1_times_anim = booster_data.get('times', np.arange(len(s1_pos_anim)))
        s1_alts_anim = booster_data.get('altitudes', np.linalg.norm(booster_data['positions'], axis=1) - R_EARTH_EQ)

        # Find staging time: when S1 times overlap with S2 times
        # The staging time is approximately when both trajectories start having data
        if len(s1_times_anim) > 0 and len(times) > 0:
            # Staging is approximately when S2 data starts
            staging_time = times[0]

    # Create unified timeline covering both stages
    if has_booster and len(s1_times_anim) > 0:
        all_times = np.unique(np.concatenate([s1_times_anim, times]))
        all_times = np.sort(all_times)
    else:
        all_times = times

    # Downsample for smooth animation
    n_frames = min(400, len(all_times))
    frame_time_indices = np.linspace(0, len(all_times)-1, n_frames, dtype=int)
    frame_times = all_times[frame_time_indices]

    # Track indices for animated traces - we use 2 traces: "Vehicle 1" and "Vehicle 2"
    # Before staging: Vehicle 1 = combined stack (visible), Vehicle 2 = hidden
    # After staging: Vehicle 1 = Stage 2 (to orbit), Vehicle 2 = Stage 1 (RTLS)
    vehicle1_trace_idx = len(fig.data)
    vehicle2_trace_idx = vehicle1_trace_idx + 1

    # Create frames - SpaceX style: combined before staging, split after
    frames = []

    for i, t in enumerate(frame_times):
        # Determine if we're before or after staging
        before_staging = t < staging_time

        # Get Stage 1 (booster) position at this time - available for entire mission
        s1_pos_now = None
        s1_alt_km = 0
        s1_in_flight = False
        s1_speed = 0
        if has_booster and len(s1_times_anim) > 0:
            s1_idx = np.searchsorted(s1_times_anim, t)
            s1_idx = min(max(0, s1_idx), len(s1_pos_anim) - 1)
            s1_pos_now = s1_pos_anim[s1_idx]
            s1_alt_km = s1_alts_anim[s1_idx] / 1000 if s1_idx < len(s1_alts_anim) else 0
            s1_in_flight = s1_alt_km > 0 and t <= s1_times_anim[-1]
            # Compute S1 speed from velocity if available
            if 'velocities' in booster_data and s1_idx < len(booster_data['velocities']):
                s1_speed = np.linalg.norm(booster_data['velocities'][s1_idx])

        # Get Stage 2 position at this time (only valid after staging)
        s2_pos_now = None
        s2_alt_km = 0
        s2_spd = 0
        s2_apo_km = 0
        s2_peri_km = 0
        if t >= staging_time and len(times) > 0:
            s2_idx = np.searchsorted(times, t)
            s2_idx = min(max(0, s2_idx), len(times) - 1)
            s2_pos_now = pos_km[s2_idx]
            s2_alt_km = altitudes[s2_idx] / 1000
            s2_spd = speeds[s2_idx]

            # Calculate orbital elements for S2
            r = np.linalg.norm(positions[s2_idx])
            v = speeds[s2_idx]
            energy = v**2 / 2 - MU_EARTH / r
            if abs(energy) > 1e-10:
                sma = -MU_EARTH / (2 * energy)
                h_vec = np.cross(positions[s2_idx], velocities[s2_idx])
                ecc_vec = np.cross(velocities[s2_idx], h_vec) / MU_EARTH - positions[s2_idx] / r
                ecc_now = np.linalg.norm(ecc_vec)
                if ecc_now < 1.0:
                    s2_apo_km = (sma * (1 + ecc_now) - R_EARTH_EQ) / 1000
                    s2_peri_km = (sma * (1 - ecc_now) - R_EARTH_EQ) / 1000
                else:
                    s2_apo_km = 9999
                    s2_peri_km = (sma * (1 - ecc_now) - R_EARTH_EQ) / 1000

        # Build frame data - ALWAYS update both traces in same order
        # Vehicle 1: Combined stack (before staging) or Stage 2 (after staging)
        # Vehicle 2: Hidden (before staging) or Stage 1 (after staging)

        if before_staging:
            # BEFORE STAGING: Combined stack at S1 position
            if s1_pos_now is not None:
                v1_marker = go.Scatter3d(
                    x=[s1_pos_now[0]], y=[s1_pos_now[1]], z=[s1_pos_now[2]],
                    mode='markers',
                    marker=dict(size=16, color='white', symbol='circle',
                               line=dict(color='lime', width=3)),
                    name='🚀 Combined Stack'
                )
                # Vehicle 2 hidden - same position, invisible
                v2_marker = go.Scatter3d(
                    x=[s1_pos_now[0]], y=[s1_pos_now[1]], z=[s1_pos_now[2]],
                    mode='markers',
                    marker=dict(size=0, opacity=0),
                    name='Stage 1'
                )
            else:
                # Fallback - shouldn't happen
                v1_marker = go.Scatter3d(
                    x=[0], y=[0], z=[R_EARTH_EQ/1000],
                    mode='markers', marker=dict(size=0, opacity=0), name='V1'
                )
                v2_marker = go.Scatter3d(
                    x=[0], y=[0], z=[R_EARTH_EQ/1000],
                    mode='markers', marker=dict(size=0, opacity=0), name='V2'
                )

            # Telemetry for combined stack
            telemetry_text = (
                f"<b>T+{t:.0f}s</b><br>"
                f"━━━━━━━━━━━━━━<br>"
                f"<b>🚀 ASCENT</b><br>"
                f"Alt: {s1_alt_km:.1f} km<br>"
                f"Speed: {s1_speed:.0f} m/s<br>"
            )

        else:
            # AFTER STAGING: Two separate vehicles
            # Vehicle 1 = Stage 2 (to orbit)
            if s2_pos_now is not None:
                v1_marker = go.Scatter3d(
                    x=[s2_pos_now[0]], y=[s2_pos_now[1]], z=[s2_pos_now[2]],
                    mode='markers',
                    marker=dict(size=14, color='white', symbol='circle',
                               line=dict(color='cyan', width=2)),
                    name='Stage 2'
                )
            else:
                # S2 data not available yet - use S1 position
                pos = s1_pos_now if s1_pos_now is not None else [0, 0, R_EARTH_EQ/1000]
                v1_marker = go.Scatter3d(
                    x=[pos[0]], y=[pos[1]], z=[pos[2]],
                    mode='markers', marker=dict(size=0, opacity=0), name='S2'
                )

            # Vehicle 2 = Stage 1 (RTLS)
            if s1_in_flight and s1_pos_now is not None:
                v2_marker = go.Scatter3d(
                    x=[s1_pos_now[0]], y=[s1_pos_now[1]], z=[s1_pos_now[2]],
                    mode='markers',
                    marker=dict(size=12, color='orange', symbol='diamond',
                               line=dict(color='red', width=2)),
                    name='Stage 1'
                )
            elif s1_pos_now is not None:
                # Booster landed - show X marker at last known position
                v2_marker = go.Scatter3d(
                    x=[s1_pos_now[0]], y=[s1_pos_now[1]], z=[s1_pos_now[2]],
                    mode='markers',
                    marker=dict(size=10, color='gray', symbol='x'),
                    name='S1 Landed'
                )
            else:
                # Fallback
                last_pos = s1_pos_anim[-1] if has_booster and len(s1_pos_anim) > 0 else [0, 0, R_EARTH_EQ/1000]
                v2_marker = go.Scatter3d(
                    x=[last_pos[0]], y=[last_pos[1]], z=[last_pos[2]],
                    mode='markers', marker=dict(size=10, color='gray', symbol='x'),
                    name='S1'
                )

            # Telemetry for both stages
            status = "🔶" if s1_in_flight else "✕"
            telemetry_text = (
                f"<b>T+{t:.0f}s</b><br>"
                f"━━━━━━━━━━━━━━<br>"
                f"<b>STAGE 2</b> ○<br>"
                f"Alt: {s2_alt_km:.1f} km<br>"
                f"Speed: {s2_spd:.0f} m/s<br>"
                f"Apo: {s2_apo_km:.0f} km<br>"
                f"━━━━━━━━━━━━━━<br>"
                f"<b>STAGE 1</b> {status}<br>"
                f"Alt: {s1_alt_km:.1f} km<br>"
                f"Speed: {s1_speed:.0f} m/s<br>"
            )

        # Always update both traces in order
        frame_data = [v1_marker, v2_marker]
        trace_indices = [vehicle1_trace_idx, vehicle2_trace_idx]

        frame = go.Frame(
            data=frame_data,
            name=str(i),
            traces=trace_indices,
            layout=go.Layout(
                annotations=[
                    dict(
                        text=telemetry_text,
                        showarrow=False,
                        x=0.98, y=0.98,
                        xref="paper", yref="paper",
                        xanchor="right", yanchor="top",
                        font=dict(size=11, color='white', family='monospace'),
                        bgcolor='rgba(0,0,0,0.8)',
                        bordercolor='cyan',
                        borderwidth=1,
                        borderpad=10,
                        align='left'
                    )
                ]
            )
        )
        frames.append(frame)

    fig.frames = frames

    # Add initial markers - 2 traces that will be animated
    # Vehicle 1: Combined stack initially (will become Stage 2 after staging)
    initial_pos = s1_pos_anim[0] if has_booster else pos_km[0]
    fig.add_trace(go.Scatter3d(
        x=[initial_pos[0]], y=[initial_pos[1]], z=[initial_pos[2]],
        mode='markers',
        marker=dict(size=16, color='white', symbol='circle',
                   line=dict(color='lime', width=3)),
        name='🚀 Vehicle',
        showlegend=True
    ))

    # Vehicle 2: Hidden initially (will become Stage 1 after staging)
    fig.add_trace(go.Scatter3d(
        x=[initial_pos[0]], y=[initial_pos[1]], z=[initial_pos[2]],
        mode='markers',
        marker=dict(size=0, opacity=0),
        name='Stage 1',
        showlegend=True
    ))

    # Create slider labels from full timeline (not just S2)
    slider_times = frame_times  # Use the full timeline for slider

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
        annotations=[
            # Initial telemetry display
            dict(
                text=(
                    f"<b>T+{times[0]:.0f}s</b><br>"
                    f"━━━━━━━━━━━━<br>"
                    f"<b>Alt:</b> {altitudes[0]/1000:.1f} km<br>"
                    f"<b>Speed:</b> {speeds[0]:.0f} m/s<br>"
                    f"<b>Inc:</b> {inclinations[0] if len(inclinations) > 0 else 0:.1f}°<br>"
                    f"━━━━━━━━━━━━<br>"
                    f"<b>Apo:</b> {altitudes[apogee_idx]/1000:.0f} km<br>"
                    f"<b>Peri:</b> {altitudes[perigee_idx]/1000:.0f} km<br>"
                    f"<b>Ecc:</b> {eccentricities[0]:.4f}<br>"
                    f"━━━━━━━━━━━━<br>"
                    f"<b>{phase_names.get(phases[0], 'Phase 1')}</b>"
                ),
                showarrow=False,
                x=0.98, y=0.98,
                xref="paper", yref="paper",
                xanchor="right", yanchor="top",
                font=dict(size=11, color='white', family='monospace'),
                bgcolor='rgba(0,0,0,0.8)',
                bordercolor='cyan',
                borderwidth=1,
                borderpad=10,
                align='left'
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
                    "label": f"{t:.0f}",
                    "method": "animate"
                }
                for i, t in enumerate(slider_times)
            ]
        }]
    )

    # Add orbit summary annotation (bottom-left to avoid overlap with telemetry)
    fig.add_annotation(
        text=f"<b>Orbit Summary</b><br>"
             f"Perigee: {altitudes[perigee_idx]/1000:.0f} km<br>"
             f"Apogee: {altitudes[apogee_idx]/1000:.0f} km<br>"
             f"Ecc: {eccentricities[-1]:.5f}",
        showarrow=False,
        x=0.02, y=0.02,
        xref="paper", yref="paper",
        xanchor="left", yanchor="bottom",
        font=dict(size=12, color='white'),
        bgcolor='rgba(0,0,0,0.7)',
        bordercolor='gray',
        borderwidth=1,
        borderpad=8
    )

    return fig


def plot_launch_animation(
    data: dict,
    title: str = "Launch Close-Up",
    max_altitude_km: float = 150.0,
    phase_names: dict | None = None,
) -> go.Figure:
    """Create a zoomed-in animation of the launch and gravity turn phase.

    Shows the trajectory from liftoff through the initial ascent in local
    East-North-Up (ENU) coordinates - as viewed from the ground. This removes
    Earth rotation effects and shows the trajectory as an observer would see it.

    Args:
        data: Simulation data dictionary with positions, velocities, times, phases
        title: Plot title
        max_altitude_km: Maximum altitude to show (km), defaults to 150 km
        phase_names: Dict mapping phase numbers to names

    Returns:
        Plotly Figure with zoomed launch animation
    """
    times = data['times']
    positions = data['positions']
    velocities = data['velocities']
    altitudes = data.get('altitudes', np.linalg.norm(positions, axis=1) - R_EARTH_EQ)
    phases = data.get('phases', np.ones(len(times)))

    if phase_names is None:
        phase_names = {1: 'Stage 1', 2: 'Coast', 3: 'Stage 2', 4: 'Coast', 5: 'Circ', 6: 'Orbit'}

    # Filter to launch phase only (up to max_altitude_km)
    max_alt_m = max_altitude_km * 1000.0
    launch_mask = altitudes <= max_alt_m
    if not np.any(launch_mask):
        launch_mask = np.ones(len(times), dtype=bool)
        launch_mask[100:] = False  # Just first 100 points

    launch_times = times[launch_mask]
    launch_positions = positions[launch_mask]
    launch_velocities = velocities[launch_mask]
    launch_altitudes = altitudes[launch_mask]
    launch_phases = phases[launch_mask]

    # Get launch site location
    launch_pos_eci = launch_positions[0]
    r_launch = np.linalg.norm(launch_pos_eci)
    lat0 = np.arcsin(launch_pos_eci[2] / r_launch)
    lon0 = np.arctan2(launch_pos_eci[1], launch_pos_eci[0])

    # Earth rotation rate
    OMEGA_EARTH = 7.2921159e-5  # rad/s

    # ENU unit vectors at launch site (fixed in ECEF)
    sin_lat = np.sin(lat0)
    cos_lat = np.cos(lat0)
    sin_lon = np.sin(lon0)
    cos_lon = np.cos(lon0)

    e_east = np.array([-sin_lon, cos_lon, 0])
    e_north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    e_up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])

    # Convert ECI positions to local ENU (accounting for Earth rotation)
    enu_positions = []
    for i in range(len(launch_times)):
        t = launch_times[i]
        pos_eci = launch_positions[i]

        # Rotate ECI to ECEF
        theta = OMEGA_EARTH * t
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        pos_ecef = np.array([
            cos_t * pos_eci[0] + sin_t * pos_eci[1],
            -sin_t * pos_eci[0] + cos_t * pos_eci[1],
            pos_eci[2]
        ])

        # Launch site in ECEF (fixed)
        launch_ecef = launch_pos_eci  # At t=0, ECI = ECEF

        # Relative position in ECEF
        rel_pos = pos_ecef - launch_ecef

        # Project to ENU
        east = np.dot(rel_pos, e_east)
        north = np.dot(rel_pos, e_north)
        up = np.dot(rel_pos, e_up)

        enu_positions.append([east, north, up])

    enu_positions = np.array(enu_positions) / 1000.0  # Convert to km

    # Create figure
    fig = go.Figure()

    # Add ground plane (Earth surface in local ENU)
    # Create a simple flat ground for this zoomed view
    ground_size = max(50, max_altitude_km * 0.5)  # km
    x_ground = np.linspace(-ground_size, ground_size * 2, 30)
    y_ground = np.linspace(-ground_size, ground_size * 2, 30)
    x_grid, y_grid = np.meshgrid(x_ground, y_ground)
    z_grid = np.zeros_like(x_grid)

    fig.add_trace(go.Surface(
        x=x_grid, y=y_grid, z=z_grid,
        colorscale=[[0, 'rgb(20,60,20)'], [1, 'rgb(40,100,40)']],
        showscale=False,
        opacity=0.8,
        name='Ground',
        hoverinfo='skip'
    ))

    # Add atmosphere boundary (100 km)
    if max_altitude_km > 100:
        # Karman line marker
        theta_atm = np.linspace(0, 2*np.pi, 50)
        r_atm = 100  # km
        x_atm = r_atm * np.cos(theta_atm)
        y_atm = r_atm * np.sin(theta_atm)
        fig.add_trace(go.Scatter3d(
            x=x_atm, y=y_atm, z=np.full_like(x_atm, 100),
            mode='lines',
            line=dict(color='cyan', width=2, dash='dash'),
            name='Karman Line (100 km)',
            hoverinfo='skip'
        ))

    # Phase colors
    phase_colors = {
        1: 'red', 2: 'yellow', 3: 'orange',
        4: 'gray', 5: 'purple', 6: 'lime'
    }

    # Add trajectory colored by phase
    speeds = np.linalg.norm(launch_velocities, axis=1)
    current_phase = launch_phases[0]
    phase_start = 0

    for i in range(1, len(launch_phases)):
        if launch_phases[i] != current_phase or i == len(launch_phases) - 1:
            end_idx = i if i < len(launch_phases) - 1 else i + 1
            segment_pos = enu_positions[phase_start:end_idx]
            segment_times = launch_times[phase_start:end_idx]
            segment_alts = launch_altitudes[phase_start:end_idx] / 1000.0
            segment_speeds = speeds[phase_start:end_idx]

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
                line=dict(color=phase_colors.get(current_phase, 'white'), width=8),
                name=phase_names.get(current_phase, f'Phase {current_phase}'),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                showlegend=True
            ))

            phase_start = i
            current_phase = launch_phases[i]

    # Add launch pad marker
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+text',
        marker=dict(size=10, color='lime', symbol='diamond'),
        text=['LAUNCH'],
        textposition='top center',
        textfont=dict(size=12, color='lime'),
        name='Launch Pad',
        hovertemplate="<b>LAUNCH PAD</b><br>Cape Canaveral<extra></extra>"
    ))

    # Animation frames
    n_frames = min(150, len(launch_times))
    frame_indices = np.linspace(0, len(launch_times)-1, n_frames, dtype=int)

    frames = []
    satellite_trace_idx = len(fig.data)

    for i, idx in enumerate(frame_indices):
        t = launch_times[idx]
        alt_km = launch_altitudes[idx] / 1000.0
        spd = speeds[idx]
        phase_num = launch_phases[idx]
        phase_str = phase_names.get(phase_num, f'Phase {phase_num}')

        # Flight path angle (in ECEF/ground frame)
        # Compute velocity relative to ground
        v_vec = launch_velocities[idx]
        r_vec = launch_positions[idx]
        r_hat = r_vec / np.linalg.norm(r_vec)
        omega_earth_vec = np.array([0, 0, OMEGA_EARTH])
        v_earth_rot = np.cross(omega_earth_vec, r_vec)
        v_rel = v_vec - v_earth_rot
        v_rad = np.dot(v_rel, r_hat)
        v_horiz = np.linalg.norm(v_rel - v_rad * r_hat)
        fpa = np.degrees(np.arctan2(v_rad, v_horiz))

        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=[enu_positions[idx, 0]],
                    y=[enu_positions[idx, 1]],
                    z=[enu_positions[idx, 2]],
                    mode='markers',
                    marker=dict(size=12, color='white', symbol='circle',
                               line=dict(color='black', width=2)),
                    name='Rocket'
                )
            ],
            name=str(i),
            traces=[satellite_trace_idx],
            layout=go.Layout(
                annotations=[
                    dict(
                        text=(
                            f"<b>T+{t:.1f}s</b><br>"
                            f"━━━━━━━━━━━━<br>"
                            f"<b>Alt:</b> {alt_km:.2f} km<br>"
                            f"<b>Speed:</b> {spd:.0f} m/s<br>"
                            f"<b>γ:</b> {fpa:.1f}°<br>"
                            f"━━━━━━━━━━━━<br>"
                            f"<b>{phase_str}</b>"
                        ),
                        showarrow=False,
                        x=0.98, y=0.98,
                        xref="paper", yref="paper",
                        xanchor="right", yanchor="top",
                        font=dict(size=12, color='white', family='monospace'),
                        bgcolor='rgba(0,0,0,0.8)',
                        bordercolor='orange',
                        borderwidth=1,
                        borderpad=10,
                        align='left'
                    )
                ]
            )
        )
        frames.append(frame)

    fig.frames = frames

    # Add initial rocket position
    fig.add_trace(go.Scatter3d(
        x=[enu_positions[0, 0]], y=[enu_positions[0, 1]], z=[enu_positions[0, 2]],
        mode='markers',
        marker=dict(size=12, color='white', symbol='circle',
                   line=dict(color='black', width=2)),
        name='Rocket',
        showlegend=False
    ))

    # Calculate view range based on trajectory extent
    max_east = np.max(np.abs(enu_positions[:, 0]))
    max_north = np.max(np.abs(enu_positions[:, 1]))
    max_alt_actual = np.max(enu_positions[:, 2])
    view_range = max(max_east, max_north, max_alt_actual) * 1.2

    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='white'), x=0.5),
        scene=dict(
            xaxis=dict(
                title="East (km)",
                range=[-view_range * 0.2, view_range],
                backgroundcolor='rgb(10,10,30)'
            ),
            yaxis=dict(
                title="North (km)",
                range=[-view_range * 0.2, view_range],
                backgroundcolor='rgb(10,10,30)'
            ),
            zaxis=dict(
                title="Altitude (km)",
                range=[-5, view_range],
                backgroundcolor='rgb(10,10,30)'
            ),
            aspectmode='data',
            camera=dict(
                eye=dict(x=-1.5, y=-1.5, z=0.8),
                up=dict(x=0, y=0, z=1)
            ),
            bgcolor='rgb(5,5,20)'
        ),
        template='plotly_dark',
        height=800,
        width=1000,
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
                x=0.1, y=0,
                xanchor="right", yanchor="top",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 80, "redraw": True},
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
        annotations=[
            dict(
                text=(
                    f"<b>T+{launch_times[0]:.1f}s</b><br>"
                    f"━━━━━━━━━━━━<br>"
                    f"<b>Alt:</b> {launch_altitudes[0]/1000:.2f} km<br>"
                    f"<b>Speed:</b> {speeds[0]:.0f} m/s<br>"
                    f"<b>γ:</b> 90.0°<br>"
                    f"━━━━━━━━━━━━<br>"
                    f"<b>{phase_names.get(launch_phases[0], 'Phase 1')}</b>"
                ),
                showarrow=False,
                x=0.98, y=0.98,
                xref="paper", yref="paper",
                xanchor="right", yanchor="top",
                font=dict(size=12, color='white', family='monospace'),
                bgcolor='rgba(0,0,0,0.8)',
                bordercolor='orange',
                borderwidth=1,
                borderpad=10,
                align='left'
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
                    "label": f"{launch_times[idx]:.0f}",
                    "method": "animate"
                }
                for i, idx in enumerate(frame_indices)
            ]
        }]
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


def plot_launch_dashboard(
    data: dict,
    title: str = "Launch Telemetry",
    phase_names: dict | None = None,
) -> go.Figure:
    """Create a comprehensive launch telemetry dashboard.

    Shows all key parameters for a rocket launch:
    - Altitude vs Time
    - Velocity (total, horizontal, vertical) vs Time
    - Flight path angle vs Time
    - Orbital elements (apogee, perigee, inclination) vs Time
    - Acceleration vs Time
    - Dynamic pressure (if in atmosphere)

    Args:
        data: Simulation data dictionary with:
            - times, positions, velocities, altitudes
            - inclinations, eccentricities (optional)
            - phases (optional)
        title: Dashboard title
        phase_names: Dict mapping phase numbers to names

    Returns:
        Plotly Figure
    """
    times = data['times']
    positions = data['positions']
    velocities = data['velocities']
    altitudes = data.get('altitudes', np.linalg.norm(positions, axis=1) - R_EARTH_EQ)
    phases = data.get('phases', np.ones(len(times)))

    # Default phase names
    if phase_names is None:
        phase_names = {1: 'Burn 1', 2: 'Coast', 3: 'Burn 2', 4: 'Coast', 5: 'Circ', 6: 'Orbit'}

    # Compute derived quantities
    speeds = np.linalg.norm(velocities, axis=1)

    # Vertical and horizontal velocity
    v_vertical = np.zeros(len(times))
    v_horizontal = np.zeros(len(times))
    flight_path_angles = np.zeros(len(times))

    for i in range(len(times)):
        r = np.linalg.norm(positions[i])
        if r > 0:
            r_hat = positions[i] / r
            v_rad = np.dot(velocities[i], r_hat)
            v_horiz_vec = velocities[i] - v_rad * r_hat
            v_vertical[i] = v_rad
            v_horizontal[i] = np.linalg.norm(v_horiz_vec)
            if v_horizontal[i] > 1:
                flight_path_angles[i] = np.degrees(np.arctan2(v_rad, v_horizontal[i]))

    # Orbital elements over time
    apogees = np.zeros(len(times))
    perigees = np.zeros(len(times))
    inclinations = data.get('inclinations', np.zeros(len(times)))
    eccentricities = data.get('eccentricities', np.zeros(len(times)))

    for i in range(len(times)):
        r = np.linalg.norm(positions[i])
        v = speeds[i]
        energy = v**2 / 2 - MU_EARTH / r
        if abs(energy) > 1e-10 and energy < 0:
            sma = -MU_EARTH / (2 * energy)
            h_vec = np.cross(positions[i], velocities[i])
            ecc_vec = np.cross(velocities[i], h_vec) / MU_EARTH - positions[i] / r
            ecc = np.linalg.norm(ecc_vec)
            if ecc < 1.0:
                apogees[i] = (sma * (1 + ecc) - R_EARTH_EQ) / 1000
                perigees[i] = (sma * (1 - ecc) - R_EARTH_EQ) / 1000
            else:
                apogees[i] = 10000
                perigees[i] = (sma * (1 - ecc) - R_EARTH_EQ) / 1000
        else:
            apogees[i] = altitudes[i] / 1000
            perigees[i] = altitudes[i] / 1000

    # Convert altitudes to km
    alt_km = altitudes / 1000.0

    # Create subplots - 3 rows x 2 cols
    # First row spans both columns with secondary y-axis for altitude
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"secondary_y": True, "colspan": 2}, None],
            [{}, {}],
            [{}, {}],
        ],
        subplot_titles=(
            "Altitude & Velocity",
            None,  # No title for merged cell
            "Flight Path Angle",
            "Orbital Elements",
            "Apogee & Perigee",
            "Inclination & Eccentricity"
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
    )

    # Phase colors for shading
    phase_colors_fill = {
        1: 'rgba(255,0,0,0.15)',
        2: 'rgba(128,128,128,0.15)',
        3: 'rgba(255,165,0,0.15)',
        4: 'rgba(100,100,255,0.15)',
        5: 'rgba(255,255,0,0.15)',
        6: 'rgba(0,255,0,0.15)'
    }

    # Helper to add phase shading to a subplot
    def add_phase_shading(row, col):
        current_phase = phases[0]
        phase_start = 0
        for i in range(1, len(phases)):
            if phases[i] != current_phase or i == len(phases) - 1:
                end_i = i if i < len(phases) - 1 else i
                color = phase_colors_fill.get(current_phase, 'rgba(128,128,128,0.1)')
                fig.add_vrect(
                    x0=times[phase_start], x1=times[end_i],
                    fillcolor=color, layer="below", line_width=0,
                    row=row, col=col
                )
                phase_start = i
                current_phase = phases[i]

    # 1. Combined Altitude & Velocity (Row 1, spans both columns)
    add_phase_shading(1, 1)

    # Velocity traces on primary y-axis (left)
    fig.add_trace(go.Scatter(
        x=times, y=speeds,
        mode='lines',
        line=dict(color='white', width=2),
        name='Total Speed',
        hovertemplate='T+%{x:.0f}s<br>Speed: %{y:.0f} m/s<extra></extra>'
    ), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=times, y=v_horizontal,
        mode='lines',
        line=dict(color='orange', width=2),
        name='V Horizontal',
        hovertemplate='T+%{x:.0f}s<br>V_h: %{y:.0f} m/s<extra></extra>'
    ), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=times, y=v_vertical,
        mode='lines',
        line=dict(color='lime', width=2, dash='dot'),
        name='V Vertical',
        hovertemplate='T+%{x:.0f}s<br>V_v: %{y:.0f} m/s<extra></extra>'
    ), row=1, col=1, secondary_y=False)

    # Altitude on secondary y-axis (right)
    fig.add_trace(go.Scatter(
        x=times, y=alt_km,
        mode='lines',
        line=dict(color='cyan', width=3),
        name='Altitude',
        hovertemplate='T+%{x:.0f}s<br>Alt: %{y:.1f} km<extra></extra>'
    ), row=1, col=1, secondary_y=True)

    # Add orbital velocity reference
    if 'target_altitude' in data:
        v_orb = orbital_velocity(data['target_altitude'])
        fig.add_hline(y=v_orb, line_dash="dash", line_color="red",
                      annotation_text=f"V_orb: {v_orb:.0f}", row=1, col=1, secondary_y=False)

    # 3. Flight Path Angle (Row 2, Col 1)
    add_phase_shading(2, 1)
    fig.add_trace(go.Scatter(
        x=times, y=flight_path_angles,
        mode='lines',
        line=dict(color='magenta', width=2),
        name='Flight Path γ',
        hovertemplate='T+%{x:.0f}s<br>γ: %{y:.1f}°<extra></extra>'
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=90, line_dash="dot", line_color="gray", row=2, col=1)

    # 4. Orbital Elements Text (Row 2, Col 2) - Show current phase
    add_phase_shading(2, 2)
    # Create phase indicator trace
    phase_values = [phase_names.get(p, f'P{p}') for p in phases]
    unique_phases = list(dict.fromkeys(phase_values))  # Preserve order

    for i, phase_name in enumerate(unique_phases):
        mask = [1 if pv == phase_name else None for pv in phase_values]
        fig.add_trace(go.Scatter(
            x=times, y=mask,
            mode='lines',
            fill='tozeroy',
            line=dict(width=0),
            fillcolor=list(phase_colors_fill.values())[i % len(phase_colors_fill)],
            name=phase_name,
            showlegend=True,
            hovertemplate=f'{phase_name}<extra></extra>'
        ), row=2, col=2)

    # 5. Apogee & Perigee (Row 3, Col 1)
    add_phase_shading(3, 1)
    fig.add_trace(go.Scatter(
        x=times, y=apogees,
        mode='lines',
        line=dict(color='red', width=2),
        name='Apogee',
        hovertemplate='T+%{x:.0f}s<br>Apo: %{y:.0f} km<extra></extra>'
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=times, y=perigees,
        mode='lines',
        line=dict(color='cyan', width=2),
        name='Perigee',
        hovertemplate='T+%{x:.0f}s<br>Peri: %{y:.0f} km<extra></extra>'
    ), row=3, col=1)

    if 'target_altitude' in data:
        fig.add_hline(y=data['target_altitude']/1000, line_dash="dash",
                      line_color="yellow", annotation_text="Target", row=3, col=1)

    # 6. Inclination & Eccentricity (Row 3, Col 2)
    add_phase_shading(3, 2)
    fig.add_trace(go.Scatter(
        x=times, y=inclinations,
        mode='lines',
        line=dict(color='yellow', width=2),
        name='Inclination',
        hovertemplate='T+%{x:.0f}s<br>Inc: %{y:.1f}°<extra></extra>'
    ), row=3, col=2)

    # Add eccentricity on secondary y-axis (scaled)
    ecc_scaled = eccentricities * 100  # Scale for visibility
    fig.add_trace(go.Scatter(
        x=times, y=ecc_scaled,
        mode='lines',
        line=dict(color='purple', width=2, dash='dot'),
        name='Ecc × 100',
        hovertemplate='T+%{x:.0f}s<br>Ecc: %{y:.2f}%<extra></extra>'
    ), row=3, col=2)

    if 'target_inclination' in data:
        fig.add_hline(y=data['target_inclination'], line_dash="dash",
                      line_color="red", annotation_text=f"Target: {data['target_inclination']:.1f}°",
                      row=3, col=2)

    # Layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=24), x=0.5),
        template='plotly_dark',
        height=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    # Axis labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=2)

    # Primary y-axis (left) - Velocity
    fig.update_yaxes(title_text="Velocity (m/s)", row=1, col=1, secondary_y=False)
    # Secondary y-axis (right) - Altitude
    fig.update_yaxes(title_text="Altitude (km)", row=1, col=1, secondary_y=True,
                     title_font=dict(color='cyan'), tickfont=dict(color='cyan'))

    fig.update_yaxes(title_text="Angle (°)", row=2, col=1)
    fig.update_yaxes(title_text="Phase", row=2, col=2)
    fig.update_yaxes(title_text="Altitude (km)", row=3, col=1)
    fig.update_yaxes(title_text="Inclination (°) / Ecc×100", row=3, col=2)

    return fig


def plot_two_stage_dashboard(
    data: dict,
    title: str = "Two-Stage Mission (RTLS)",
) -> go.Figure:
    """Simple two-stage dashboard with animated stage markers.

    This is a lightweight, opinionated visualization focused on:
    - One clean 3D view (Earth + trajectories)
    - Two moving dots (Stage 1 booster + Stage 2 upper stage)
    - Minimal but useful telemetry annotation

    Expected ``data`` format (matches ``two_stage_rtls.py`` plot_data):
        - times: Stage 2 times [s]
        - positions: Stage 2 positions [N, 3] [m]
        - velocities: Stage 2 velocities [N, 3] [m/s]
        - altitudes: Stage 2 altitudes [N] [m]
        - s1_times: Stage 1 times [M] [s]
        - s1_positions: Stage 1 positions [M, 3] [m]
        - s1_velocities: Stage 1 velocities [M, 3] [m/s]
        - s1_altitudes: Stage 1 altitudes [M] [m]
    """
    # Stage 2 (upper stage) data
    s2_times = np.asarray(data["times"])
    s2_positions = np.asarray(data["positions"])
    s2_velocities = np.asarray(data["velocities"])
    s2_altitudes = np.asarray(data["altitudes"])

    # Stage 1 (booster) data
    s1_times = np.asarray(data["s1_times"])
    s1_positions = np.asarray(data["s1_positions"])
    s1_velocities = np.asarray(data["s1_velocities"])
    s1_altitudes = np.asarray(data["s1_altitudes"])

    # Optional phase arrays (for highlighting burns)
    s1_phases = np.asarray(data["s1_phases"]) if "s1_phases" in data else None
    s2_phases = np.asarray(data["s2_phases"]) if "s2_phases" in data else None

    # Convert to km for plotting
    s2_pos_km = s2_positions / 1000.0
    s1_pos_km = s1_positions / 1000.0

    s2_alt_km = s2_altitudes / 1000.0
    s1_alt_km = s1_altitudes / 1000.0

    s2_speed = np.linalg.norm(s2_velocities, axis=1)
    s1_speed = np.linalg.norm(s1_velocities, axis=1)

    # Phase name maps (keep local to avoid coupling to guidance enums)
    s1_phase_names = {
        1: "Coast",
        2: "Boostback",
        3: "Entry Coast",
        4: "Entry Burn",
        5: "Descent",
        6: "Landing Burn",
        7: "Landed",
    }
    s2_phase_names = {
        11: "S2 Gravity Turn",
        12: "S2 Gravity Turn",
        13: "S2 Gravity Turn",
        14: "Coast to Apo",
        15: "Circularize",
        16: "Orbit",
    }
    s1_burn_phases = {2, 4, 6}  # highlight boostback & burn phases

    # Global time base covering both stages
    all_times = np.concatenate([s1_times, s2_times])
    all_times = np.unique(np.sort(all_times))

    # Downsample for animation smoothness vs size
    n_frames = min(350, len(all_times))
    frame_indices = np.linspace(0, len(all_times) - 1, n_frames, dtype=int)
    frame_times = all_times[frame_indices]

    # Base figure with Earth + full trajectories
    fig = go.Figure()

    # Earth sphere (minimal style)
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 25)
    r_earth_km = R_EARTH_EQ / 1000.0
    x_earth = r_earth_km * np.outer(np.cos(u), np.sin(v))
    y_earth = r_earth_km * np.outer(np.sin(u), np.sin(v))
    z_earth = r_earth_km * np.outer(np.ones_like(u), np.cos(v))

    fig.add_trace(
        go.Surface(
            x=x_earth,
            y=y_earth,
            z=z_earth,
            colorscale=[[0, "rgb(20,40,90)"], [1, "rgb(30,80,140)"]],
            showscale=False,
            opacity=0.85,
            name="Earth",
            hoverinfo="skip",
        )
    )

    # Static trajectories for context
    fig.add_trace(
        go.Scatter3d(
            x=s2_pos_km[:, 0],
            y=s2_pos_km[:, 1],
            z=s2_pos_km[:, 2],
            mode="lines",
            line=dict(color="cyan", width=4),
            name="Stage 2 Trajectory",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=s1_pos_km[:, 0],
            y=s1_pos_km[:, 1],
            z=s1_pos_km[:, 2],
            mode="lines",
            line=dict(color="orange", width=3, dash="dash"),
            name="Stage 1 Trajectory",
            hoverinfo="skip",
        )
    )

    # Indices of animated marker traces (added after static traces)
    s2_marker_trace_idx = len(fig.data)
    s1_marker_trace_idx = s2_marker_trace_idx + 1

    # Initial marker positions (first available samples)
    s2_init_idx = 0
    s1_init_idx = 0

    fig.add_trace(
        go.Scatter3d(
            x=[s2_pos_km[s2_init_idx, 0]],
            y=[s2_pos_km[s2_init_idx, 1]],
            z=[s2_pos_km[s2_init_idx, 2]],
            mode="markers",
            marker=dict(size=10, color="cyan", symbol="circle", line=dict(color="white", width=2)),
            name="Stage 2",
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[s1_pos_km[s1_init_idx, 0]],
            y=[s1_pos_km[s1_init_idx, 1]],
            z=[s1_pos_km[s1_init_idx, 2]],
            mode="markers+text",
            marker=dict(size=10, color="orange", symbol="diamond", line=dict(color="white", width=1)),
            text=[""],
            textposition="top center",
            textfont=dict(size=12, color="yellow", family="monospace"),
            name="Stage 1",
            showlegend=True,
        )
    )

    # Helper: sample index at time t from a monotonic time array
    def _sample_idx(t_array: np.ndarray, t: float) -> int:
        if t_array.size == 0:
            return 0
        idx = int(np.searchsorted(t_array, t))
        idx = max(0, min(idx, len(t_array) - 1))
        return idx

    # Build animation frames
    frames: list[go.Frame] = []

    for i, t in enumerate(frame_times):
        s1_idx = _sample_idx(s1_times, t)
        s2_idx = _sample_idx(s2_times, t)

        s1_pos_now = s1_pos_km[s1_idx]
        s2_pos_now = s2_pos_km[s2_idx]

        s1_alt_now = s1_alt_km[s1_idx]
        s2_alt_now = s2_alt_km[s2_idx]
        s1_spd_now = s1_speed[s1_idx]
        s2_spd_now = s2_speed[s2_idx]

        s1_phase_now = int(s1_phases[s1_idx]) if s1_phases is not None and s1_phases.size else 0
        s2_phase_now = int(s2_phases[s2_idx]) if s2_phases is not None and s2_phases.size else 0

        s1_phase_label = s1_phase_names.get(s1_phase_now, str(s1_phase_now))
        s2_phase_label = s2_phase_names.get(s2_phase_now, str(s2_phase_now))

        # Emphasize Stage 1 when burning (boostback / entry / landing burns)
        s1_is_burn = s1_phase_now in s1_burn_phases
        s1_marker_color = "red" if s1_is_burn else "gray"
        s1_marker_size = 15 if s1_is_burn else 9
        s1_marker_symbol = "cross" if s1_is_burn else "diamond"
        s1_text = ["🔥 BURN"] if s1_is_burn else [""]
        
        phase_icon = "🔥 " if s1_is_burn else ""

        telemetry_text = (
            f"<b>T+{t:.1f}s</b><br>"
            f"━━━━━━━━━━━━━━<br>"
            f"<b>Stage 2</b><br>"
            f"Alt: {s2_alt_now:.1f} km<br>"
            f"Speed: {s2_spd_now:.0f} m/s<br>"
            f"Phase: {s2_phase_label}<br>"
            f"━━━━━━━━━━━━━━<br>"
            f"<b>Stage 1</b><br>"
            f"Alt: {s1_alt_now:.1f} km<br>"
            f"Speed: {s1_spd_now:.0f} m/s<br>"
            f"Phase: {phase_icon}{s1_phase_label}"
        )

        frame = go.Frame(
            name=str(i),
            data=[
                go.Scatter3d(
                    x=[s2_pos_now[0]],
                    y=[s2_pos_now[1]],
                    z=[s2_pos_now[2]],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color="cyan",
                        symbol="circle",
                        line=dict(color="white", width=2),
                    ),
                    name="Stage 2",
                ),
                go.Scatter3d(
                    x=[s1_pos_now[0]],
                    y=[s1_pos_now[1]],
                    z=[s1_pos_now[2]],
                    mode="markers+text",
                    marker=dict(
                        size=s1_marker_size,
                        color=s1_marker_color,
                        symbol=s1_marker_symbol,
                        line=dict(color="white", width=1),
                    ),
                    text=s1_text,
                    textposition="top center",
                    textfont=dict(size=12, color="yellow", family="monospace"),
                    name="Stage 1",
                ),
            ],
            traces=[s2_marker_trace_idx, s1_marker_trace_idx],
            layout=go.Layout(
                annotations=[
                    dict(
                        text=telemetry_text,
                        showarrow=False,
                        x=0.98,
                        y=0.98,
                        xref="paper",
                        yref="paper",
                        xanchor="right",
                        yanchor="top",
                        font=dict(size=11, color="white", family="monospace"),
                        bgcolor="rgba(0,0,0,0.8)",
                        bordercolor="cyan",
                        borderwidth=1,
                        borderpad=8,
                        align="left",
                    )
                ]
            ),
        )
        frames.append(frame)

    fig.frames = frames

    # Scene layout
    all_pos_km = np.vstack([s1_pos_km, s2_pos_km])
    max_range = float(np.max(np.abs(all_pos_km))) * 1.1

    fig.update_layout(
        title=dict(text=title, font=dict(size=22, color="white"), x=0.5),
        scene=dict(
            xaxis=dict(title="X (km)", range=[-max_range, max_range], backgroundcolor="rgb(10,10,30)"),
            yaxis=dict(title="Y (km)", range=[-max_range, max_range], backgroundcolor="rgb(10,10,30)"),
            zaxis=dict(title="Z (km)", range=[-max_range, max_range], backgroundcolor="rgb(10,10,30)"),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9), up=dict(x=0, y=0, z=1)),
            bgcolor="rgb(5,5,20)",
        ),
        template="plotly_dark",
        height=850,
        width=1150,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor="gray",
            borderwidth=1,
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                x=0.05,
                y=0,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 60, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
        annotations=[
            dict(
                text=(
                    f"<b>T+{frame_times[0]:.1f}s</b><br>"
                    f"━━━━━━━━━━━━━━<br>"
                    f"<b>Stage 2</b><br>"
                    f"Alt: {s2_alt_km[0]:.1f} km<br>"
                    f"Speed: {s2_speed[0]:.0f} m/s<br>"
                    f"━━━━━━━━━━━━━━<br>"
                    f"<b>Stage 1</b><br>"
                    f"Alt: {s1_alt_km[0]:.1f} km<br>"
                    f"Speed: {s1_speed[0]:.0f} m/s"
                ),
                showarrow=False,
                x=0.98,
                y=0.98,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="top",
                font=dict(size=11, color="white", family="monospace"),
                bgcolor="rgba(0,0,0,0.8)",
                bordercolor="cyan",
                borderwidth=1,
                borderpad=8,
                align="left",
            )
        ],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "prefix": "Time: T+",
                    "suffix": " s",
                    "visible": True,
                    "xanchor": "center",
                    "font": {"size": 14, "color": "white"},
                },
                "transition": {"duration": 0},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.05,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [str(i)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": f"{t:.0f}",
                        "method": "animate",
                    }
                    for i, t in enumerate(frame_times)
                ],
            }
        ],
    )

    return fig


def plot_rtls_ground_track(
    data: dict,
    title: str = "Two-Stage RTLS – Ground Frame",
) -> go.Figure:
    """Ground-fixed 3D view of first-stage RTLS relative to the pad.

    This converts ECI positions to a local East-North-Up (ENU) frame
    centered at the launch site. In this frame, the pad is fixed at
    (0, 0, 0) and both the booster and upper stage trajectories are
    shown relative to that pad.

    Expected ``data`` keys:
        - times: Stage 2 times [s]
        - positions: Stage 2 ECI positions [N, 3] [m]
        - s1_times: Stage 1 times [M] [s]
        - s1_positions: Stage 1 ECI positions [M, 3] [m]
        - launch_site_eci: Launch site position in ECI at t=0 [3] [m]
    """
    # Constants for Earth rotation
    OMEGA_EARTH = 7.2921159e-5  # rad/s

    # Extract arrays
    s2_times = np.asarray(data["times"])
    s2_positions = np.asarray(data["positions"])

    s1_times = np.asarray(data["s1_times"])
    s1_positions = np.asarray(data["s1_positions"])

    launch_site_eci = np.asarray(data["launch_site_eci"])

    # Derive launch site geodetic latitude/longitude from ECI at t=0
    r_launch = np.linalg.norm(launch_site_eci)
    if r_launch == 0:
        raise ValueError("launch_site_eci must be non-zero")

    lat0 = np.arcsin(launch_site_eci[2] / r_launch)
    lon0 = np.arctan2(launch_site_eci[1], launch_site_eci[0])

    sin_lat = np.sin(lat0)
    cos_lat = np.cos(lat0)
    sin_lon = np.sin(lon0)
    cos_lon = np.cos(lon0)

    # ENU basis vectors at launch site (fixed in ECEF)
    e_east = np.array([-sin_lon, cos_lon, 0.0])
    e_north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    e_up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])

    def _eci_to_enu(times: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Convert ECI trajectory to local ENU (km) at launch site."""
        enu_list: list[np.ndarray] = []
        for t, pos_eci in zip(times, positions, strict=True):
            # Rotate ECI → ECEF
            theta = OMEGA_EARTH * t
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            pos_ecef = np.array(
                [
                    cos_t * pos_eci[0] + sin_t * pos_eci[1],
                    -sin_t * pos_eci[0] + cos_t * pos_eci[1],
                    pos_eci[2],
                ]
            )

            # Launch site ECEF at t=0 (assume ECI≈ECEF initially)
            launch_ecef = launch_site_eci

            # Relative position in ECEF
            rel = pos_ecef - launch_ecef

            # Project onto ENU basis
            east = float(np.dot(rel, e_east))
            north = float(np.dot(rel, e_north))
            up = float(np.dot(rel, e_up))

            enu_list.append(np.array([east, north, up], dtype=float))

        return np.asarray(enu_list) / 1000.0  # km

    # Compute ENU coordinates for both stages
    s2_enu = _eci_to_enu(s2_times, s2_positions)
    s1_enu = _eci_to_enu(s1_times, s1_positions)

    # Figure setup
    fig = go.Figure()

    # Ground plane
    max_extent = float(
        max(
            np.max(np.abs(s2_enu[:, :2])) if s2_enu.size else 0.0,
            np.max(np.abs(s1_enu[:, :2])) if s1_enu.size else 0.0,
            50.0,
        )
    )
    ground_size = max_extent * 1.2

    x_ground = np.linspace(-ground_size, ground_size, 30)
    y_ground = np.linspace(-ground_size, ground_size, 30)
    x_grid, y_grid = np.meshgrid(x_ground, y_ground)
    z_grid = np.zeros_like(x_grid)

    fig.add_trace(
        go.Surface(
            x=x_grid,
            y=y_grid,
            z=z_grid,
            colorscale=[[0, "rgb(20,60,20)"], [1, "rgb(40,100,40)"]],
            showscale=False,
            opacity=0.85,
            name="Ground",
            hoverinfo="skip",
        )
    )

    # Launch pad marker at origin
    fig.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode="markers+text",
            marker=dict(size=9, color="lime", symbol="diamond"),
            text=["PAD"],
            textposition="top center",
            textfont=dict(size=12, color="lime"),
            name="Launch / Landing Pad",
            hovertemplate="<b>Pad</b><br>East: 0 km<br>North: 0 km<extra></extra>",
        )
    )

    # Stage 2 trajectory (upper stage)
    if s2_enu.size:
        fig.add_trace(
            go.Scatter3d(
                x=s2_enu[:, 0],
                y=s2_enu[:, 1],
                z=s2_enu[:, 2],
                mode="lines",
                line=dict(color="cyan", width=4),
                name="Stage 2",
                hoverinfo="skip",
            )
        )

    # Stage 1 trajectory (booster: ascent + RTLS)
    if s1_enu.size:
        fig.add_trace(
            go.Scatter3d(
                x=s1_enu[:, 0],
                y=s1_enu[:, 1],
                z=s1_enu[:, 2],
                mode="lines",
                line=dict(color="orange", width=4),
                name="Stage 1 (Booster)",
                hoverinfo="skip",
            )
        )

    # Axis ranges
    all_enu = np.vstack([s1_enu, s2_enu]) if s1_enu.size and s2_enu.size else (
        s1_enu if s1_enu.size else s2_enu
    )
    if all_enu.size:
        max_range = float(np.max(np.abs(all_enu))) * 1.2
    else:
        max_range = 100.0

    fig.update_layout(
        title=dict(text=title, font=dict(size=22, color="white"), x=0.5),
        scene=dict(
            xaxis=dict(
                title="East (km)",
                range=[-max_range, max_range],
                backgroundcolor="rgb(10,10,30)",
            ),
            yaxis=dict(
                title="North (km)",
                range=[-max_range, max_range],
                backgroundcolor="rgb(10,10,30)",
            ),
            zaxis=dict(
                title="Altitude (km)",
                range=[-5.0, max_range * 0.6],
                backgroundcolor="rgb(10,10,30)",
            ),
            aspectmode="data",
            camera=dict(eye=dict(x=-1.6, y=-1.6, z=0.9), up=dict(x=0, y=0, z=1)),
            bgcolor="rgb(5,5,20)",
        ),
        template="plotly_dark",
        height=800,
        width=1100,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="gray",
            borderwidth=1,
        ),
    )

    return fig
