"""Pure-Python row loop used as the GIL control."""

from __future__ import annotations


def python_sum(records):
    total = 0.0
    kept = 0
    for dist, fare, dur in records:
        if dur >= 60.0 and dist > 0.0 and 2.5 <= fare <= 500.0:
            total += fare
            kept += 1
    return total, kept
