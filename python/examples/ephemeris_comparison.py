"""JPL low-precision and VSOP2013 ephemeris comparison.

Units: MJD2000 days, metres, and metres/second.
Expected: two finite states; their frames intentionally differ.
Runtime: first VSOP evaluation is normally below 1 ms with a release wheel.
Features: default wheel with embedded VSOP2013 coefficients.
"""

import math

import numpy as np

import pykep_rust as pk


jpl = pk.Planet.jpl_low_precision("earth")
vsop = pk.Planet.vsop2013("earth_moon")
jpl_state = jpl.state(0.5)
vsop_state = vsop.state(0.5)
jpl_distance = float(np.linalg.norm(jpl_state[:3]))
vsop_distance = float(np.linalg.norm(vsop_state[:3]))
assert math.isfinite(jpl_distance)
assert math.isfinite(vsop_distance)
print(f"JPL distance [m]: {jpl_distance:.3f}")
print(f"VSOP2013 distance [m]: {vsop_distance:.3f}")
print("JPL is J2000 ecliptic; VSOP2013 is ICRF.")
