"""Data export utilities for rocket simulation."""

import json
from pathlib import Path

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for NumPy arrays."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def export_trajectory_to_json(data: dict, filepath: str | Path):
    """Export trajectory data to a compact JSON file for visualization.
    
    Args:
        data: Dictionary containing simulation arrays (times, positions, etc.)
        filepath: Path to save the JSON file
    """
    # Structure data for the frontend
    # We want to separate stages for cleaner parsing

    output = {
        "metadata": {
            "target_altitude": data.get("target_altitude", 0),
            "landing_site": data.get("landing_site_eci", [0,0,0]),
            "staging_time": data.get("staging_time", 0),
            # Vehicle parameters for telemetry display
            "s1_dry_mass": data.get("s1_dry_mass", 25000),
            "s1_wet_mass": data.get("s1_wet_mass", 433000),
            "s1_landing_propellant": data.get("s1_landing_propellant", 150000),
            "s2_dry_mass": data.get("s2_dry_mass", 4000),
            "s2_wet_mass": data.get("s2_wet_mass", 116000),
            "s1_thrust": data.get("s1_thrust", 7600000),
            "s2_thrust": data.get("s2_thrust", 934000),
            "s1_isp_sl": data.get("s1_isp_sl", 282),
            "s1_isp_vac": data.get("s1_isp_vac", 311),
            "s2_isp": data.get("s2_isp", 348),
            # O/F ratio for propellant breakdown (RP-1/LOX typical: 2.56)
            "of_ratio": data.get("of_ratio", 2.56),
        },
        "stages": [
            {
                "id": "stage1",
                "name": "Stage 1",
                "times": data.get("s1_times", []),
                "positions": data.get("s1_positions", []),
                "velocities": data.get("s1_velocities", []),
                "phases": data.get("s1_phases", []),
                "masses": data.get("s1_masses", []),
                "thrusts": data.get("s1_thrusts", []),
                "accelerations": data.get("s1_accelerations", []),
            },
            {
                "id": "stage2",
                "name": "Stage 2",
                "times": data.get("times", []),
                "positions": data.get("positions", []),
                "velocities": data.get("velocities", []),
                "phases": data.get("s2_phases", []),
                "masses": data.get("s2_masses", []),
                "thrusts": data.get("s2_thrusts", []),
                "accelerations": data.get("s2_accelerations", []),
            }
        ]
    }

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(output, f, cls=NumpyEncoder)

    print(f"Exported flight data to {path}")

