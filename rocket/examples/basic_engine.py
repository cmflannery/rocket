#!/usr/bin/env python
"""Design an engine in one line.

Shows the simplest possible workflow for engine design.
"""

from rocket.studies import quick_design

# Design a 50 kN LOX/CH4 engine
result = quick_design(
    thrust_kn=50,
    propellants=("LOX", "CH4"),
    chamber_pressure_mpa=8,
    output_dir="outputs/my_engine",
)

print(f"""
Engine Design Complete
======================
Isp (vacuum):    {result['isp_vac']:.0f} s
Isp (sea level): {result['isp_sl']:.0f} s
Throat:          {result['throat_diameter_cm']:.1f} cm
Exit:            {result['exit_diameter_cm']:.1f} cm
Mass flow:       {result['mdot_kg_s']:.2f} kg/s

Output saved to: outputs/my_engine/
""")
