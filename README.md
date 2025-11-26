# OpenRocketEngine

![Python package](https://github.com/cmflannery/openrocketengine/workflows/Python%20package/badge.svg)

Tools for liquid rocket engine design and analysis.

## Installation

Requires a Fortran compiler for RocketCEA (NASA CEA thermochemistry).

```bash
# macOS
brew install gcc

# Linux (Debian/Ubuntu)
sudo apt-get install gfortran

# Then install
pip install openrocketengine
```

## Quick Start

```python
from openrocketengine import EngineInputs, design_engine, plot_engine_dashboard
from openrocketengine.units import kilonewtons, megapascals

# Design from propellant selection (thermochemistry auto-calculated)
inputs = EngineInputs.from_propellants(
    oxidizer="LOX",
    fuel="RP1",
    thrust=kilonewtons(100),
    chamber_pressure=megapascals(7),
    mixture_ratio=2.7,
)

# Compute performance and geometry
performance, geometry = design_engine(inputs)

print(f"Isp (sea level): {performance.isp}")
print(f"Isp (vacuum): {performance.isp_vac}")
print(f"Throat diameter: {geometry.throat_diameter}")

# Visualize
plot_engine_dashboard(inputs, performance, geometry)
```

## Features

- **Type-safe**: Runtime type checking with beartype
- **Units handling**: Built-in `Quantity` class prevents unit errors
- **Fast**: Numba-accelerated isentropic flow calculations
- **Visualization**: Engine cross-sections, performance curves, dashboards
- **NASA CEA**: Accurate thermochemistry via RocketCEA
- **Nozzle contours**: Rao bell and conical nozzle generation with CSV export

## References

1. [GATech: Bell Nozzles](http://soliton.ae.gatech.edu/people/jseitzma/classes/ae6450/bell_nozzle.pdf)
2. [Design of Liquid Propellant Rocket Engines](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19710019929.pdf)
3. [Liquid Propellant Rocket Combustion Instability, NASA SP-194](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19720026079.pdf)
