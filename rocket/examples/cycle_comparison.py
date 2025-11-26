#!/usr/bin/env python
"""Which engine cycle should I use?

This tool analyzes your mission requirements and recommends the best cycle.
"""

from rocket.studies import CycleTradeStudy

# Define your mission
study = CycleTradeStudy(
    thrust_kn=100,           # Required thrust
    delta_v_m_s=3500,        # Mission delta-V
    payload_kg=5000,         # Payload to deliver
    propellants=("LOX", "CH4"),
)

# Run analysis and save results
study.run()
study.save("outputs/cycle_selection")

print(f"\nRecommendation: {study.recommendation}")
print(f"Results saved to: outputs/cycle_selection/")
