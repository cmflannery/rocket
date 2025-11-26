#!/usr/bin/env python
"""Find the optimal chamber pressure and mixture ratio.

Sweeps the design space and identifies the best operating point.
"""

from rocket.studies import EngineTrade

# Run trade study
study = EngineTrade(
    thrust_kn=100,
    propellants=("LOX", "CH4"),
    pc_range_mpa=(5, 25),
)

study.run()
study.save("outputs/engine_trade")

best = study.best_design
print(f"""
Trade Study Complete
====================
Optimal design:
  Chamber pressure: {best['chamber_pressure_mpa']:.0f} MPa
  Mixture ratio:    {best['mixture_ratio']:.2f}
  Isp (vacuum):     {best['isp_vac']:.0f} s

Full results: outputs/engine_trade/
""")
