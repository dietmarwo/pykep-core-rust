"""Epoch and anomaly conversion.

Units: MJD2000 days and radians.
Expected: arrival is 180 days later; anomaly round trip is 0.7 radians.
Runtime: constant work, normally below 1 ms with a release wheel.
Features: default wheel; no external data or runtime.
"""

import pykep_rust as pk


departure = pk.Epoch.from_iso("2030-01-15T12:30:00")
arrival = departure.add_days(180.0)
eccentric = pk.mean_to_eccentric_anomaly(0.7, 0.8)
round_trip = pk.eccentric_to_mean_anomaly(eccentric, 0.8)
assert arrival.seconds_since(departure) == 180.0 * pk.DAY_TO_SECONDS
assert abs(round_trip - 0.7) < 1e-14
print(f"departure: {departure}; arrival: {arrival}")
print(f"elliptic round trip [rad]: {round_trip:.16f}")
