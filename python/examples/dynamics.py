"""CR3BP and zero-order-hold low-thrust propagation.

Units: normalized CR3BP/Kepler units; state order is `[r,v]` plus mass.
Expected: finite six- and seven-component final states at time 0.1.
Runtime: two short adaptive solves, normally below 1 ms with a release wheel.
Features: default wheel; no external data or runtime.
"""

import math

import pykep_rust as pk


cr3bp = pk.propagate_cr3bp(
    [0.8, -0.2, 0.1, 0.03, -0.04, 0.02],
    0.1,
    pk.CR3BP_MU_EARTH_MOON,
)
zoh = pk.propagate_zoh_kepler(
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.2],
    [0.0, 0.05, 0.1],
    [[0.01, 1.0, 0.0, 0.0], [0.01, 0.0, 1.0, 0.0]],
    0.02,
)
assert len(cr3bp) == 6 and all(math.isfinite(value) for value in cr3bp)
assert len(zoh) == 7 and all(math.isfinite(value) for value in zoh)
print(f"CR3BP final state: {cr3bp}")
print(f"ZOH Kepler final state: {zoh}")
