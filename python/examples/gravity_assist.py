"""Gravity-assist feasibility and outgoing velocity.

Units: SI velocity, distance, and gravitational parameter; beta is radians.
Expected: finite constraints, positive powered delta-v, finite output velocity.
Runtime: constant work, normally below 1 ms with a release wheel.
Features: default wheel; no external data or runtime.
"""

import math

import pykep_rust as pk


incoming = [7_200.0, -4_567.765_5, 1_234.423_3]
outgoing = [7_100.0, 220.123, -144.432]
earth_mu = pk.MU_EARTH
periapsis = 7.0e6
constraints = pk.flyby_constraints(incoming, outgoing, earth_mu, periapsis)
delta_v = pk.flyby_delta_v(incoming, outgoing, earth_mu, periapsis)
mapped = pk.flyby_outgoing_velocity(
    incoming, [10_000.0, 20_000.0, -1_000.0], periapsis, 0.2, earth_mu
)
assert all(math.isfinite(value) for value in [*constraints, *mapped])
assert delta_v > 0.0
print(f"constraints [equality, inequality]: {constraints}")
print(f"minimum powered delta-v [m/s]: {delta_v:.6f}")
print(f"unpowered outgoing velocity [m/s]: {mapped}")
