"""Element conversion and two-body propagation.

Units: SI metres, seconds, radians, and `mu` in m³/s².
Expected: one finite Cartesian state propagated by 60 seconds.
Runtime: constant work, normally below 1 ms with a release wheel.
Features: default wheel; no external data or runtime.
"""

import math

import pykep_rust as pk


state = pk.classical_to_cartesian(
    [7.0e6, 0.01, 0.4, 1.0, 0.5, 0.2], pk.MU_EARTH
)
future = pk.propagate_lagrangian(state, 60.0, pk.MU_EARTH)
recovered = pk.cartesian_to_classical(future, pk.MU_EARTH)
assert all(math.isfinite(value) for value in future)
assert len(recovered) == 6
print(f"initial Cartesian state [m,m/s]: {state}")
print(f"state after 60 s [m,m/s]: {future}")
