"""Data export utilities for rocket simulation."""

import json
import numpy as np
from pathlib import Path
from typing import Any

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
            "staging_time": data.get("staging_time", 0)
        },
        "stages": [
            {
                "id": "stage1",
                "name": "Stage 1 (Booster)",
                "times": data.get("s1_times", []),
                "positions": data.get("s1_positions", []),
                "velocities": data.get("s1_velocities", []),
                "phases": data.get("s1_phases", []),
                # Add attitude if available, otherwise use velocity direction in frontend
            },
            {
                "id": "stage2",
                "name": "Stage 2 (Orbiter)",
                "times": data.get("times", []), # S2 times
                "positions": data.get("positions", []), # S2 positions
                "velocities": data.get("velocities", []), # S2 velocities
                "phases": data.get("s2_phases", []),
            }
        ]
    }
    
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(output, f, cls=NumpyEncoder)
    
    print(f"Exported flight data to {path}")

