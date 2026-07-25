"""Sims–Flanagan mismatch and analytic gradient.

Units: one consistent length/time/mass system; controls are dimensionless.
Expected: seven mismatch values and a 7 × 13 control/time Jacobian.
Runtime: constant four-segment work, normally below 1 ms with a release wheel.
Features: default wheel; no external data or runtime.
"""

import numpy as np

import pykep_rust as pk


leg = pk.SimsFlanaganLeg(
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    2.0,
    [
        [0.1, -0.2, 0.05],
        [0.3, 0.1, -0.15],
        [-0.25, 0.2, 0.1],
        [0.05, -0.1, 0.2],
    ],
    [0.2, 1.1, 0.1, -0.9, 0.15, -0.05],
    1.7,
    1.3,
    0.04,
    3.0,
    1.0,
)
mismatch = leg.mismatch_constraints()
_, _, controls_and_time = leg.mismatch_jacobian()
assert len(mismatch) == 7
assert np.asarray(controls_and_time).shape == (7, 13)
print(f"mismatch [dr,dv,dm]: {mismatch}")
print("Jacobian shapes: departure 7×7, arrival 7×7, controls/time 7×13")
