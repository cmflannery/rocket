#!/usr/bin/env python
"""Validate the model against real rocket engines.

Compares computed Isp to published data for Merlin, Raptor, RS-25, etc.
"""

from rocket.validation import validation_report

# Run validation against all reference engines
report = validation_report(
    tolerance_pct=5.0,
    save_path="outputs/validation/validation_results.png"
)

print(report)

