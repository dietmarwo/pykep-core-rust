"""A guided tour of the astrodynamics introduction, from orbits to the gym.

Each section matches a section of docs/astrodynamics-introduction.md.
Units: SI (m, m/s, s, kg) unless a name says otherwise; epochs are MJD2000.
Expected: finite values; the printed numbers are quoted in the chapter.
Runtime: well below one second with a release wheel.
Features: default wheel with the trajectory gym; no external data.
"""

import json
import math
from pathlib import Path

import numpy as np

import pykep_rust as pk
from pykep_rust.trajopt import gym

AU = pk.ASTRONOMICAL_UNIT
DAY = pk.DAY_TO_SECONDS

# 1. An orbit around the Sun --------------------------------------------------
# Classical elements [a, e, i, RAAN, argument of periapsis, true anomaly].
elements = [AU, 0.0167, 0.0, 0.0, math.radians(102.9), 0.0]
state = pk.classical_to_cartesian(elements, pk.MU_SUN)
period = 2.0 * math.pi * math.sqrt(AU**3 / pk.MU_SUN)
speed = math.hypot(*state[3:])
print(f"1. period {period / DAY:.2f} days, speed at periapsis {speed / 1e3:.2f} km/s")
half = pk.propagate_lagrangian(state, period / 2.0, pk.MU_SUN)
radius = math.hypot(*half[:3])
assert abs(radius - AU * (1.0 + 0.0167)) < 1e3  # apoapsis after half a period
print(f"   half a period later: r = {radius / AU:.4f} AU (apoapsis)")

# 2. Planets move: ephemerides ----------------------------------------------
epoch = pk.Epoch.from_iso("2033-01-01")
earth = pk.Planet.jpl_low_precision("earth")
mars = pk.Planet.jpl_low_precision("mars")
r_earth = earth.state(epoch.mjd2000)
r_mars = mars.state(epoch.mjd2000)
separation = math.dist(r_earth[:3], r_mars[:3])
print(f"2. on {epoch.to_iso()[:10]} Earth-Mars distance {separation / AU:.3f} AU")

# 3. The rocket equation ----------------------------------------------------
delta_v = 5_000.0
for engine, isp in [("chemical", 320.0), ("electric", 3_000.0)]:
    remaining = math.exp(-delta_v / (isp * pk.STANDARD_GRAVITY))
    print(f"3. {engine:8s} Isp {isp:6.0f} s keeps {remaining:6.1%} of the mass")

# 4. A Hohmann transfer: the textbook two-impulse orbit change ---------------
total, duration, impulses = pk.hohmann(AU, 1.524 * AU, pk.MU_SUN)
print(
    f"4. Hohmann Earth->Mars orbit: {total / 1e3:.2f} km/s in "
    f"{duration / DAY:.0f} days, impulses {[round(v) for v in impulses]} m/s"
)

# 5. Lambert's problem and a small porkchop scan ---------------------------
departures = epoch.mjd2000 + np.arange(0.0, 361.0, 6.0)
flight_days = np.arange(120.0, 361.0, 10.0)
grid = np.array([(t0, tof) for t0 in departures for tof in flight_days])
start = earth.states(grid[:, 0])
end = mars.states(grid[:, 0] + grid[:, 1])
problems = pk.lambert_problem_batch(
    np.ascontiguousarray(start[:, :3]),
    np.ascontiguousarray(end[:, :3]),
    grid[:, 1] * DAY,
    pk.MU_SUN,
    maximum_revolutions=0,
)
departure_vinf = np.array(
    [np.linalg.norm(np.subtract(p.solutions[0].departure_velocity, s[3:]))
     for p, s in zip(problems, start)]
)
arrival_vinf = np.array(
    [np.linalg.norm(np.subtract(p.solutions[0].arrival_velocity, s[3:]))
     for p, s in zip(problems, end)]
)
best = int(np.argmin(departure_vinf + arrival_vinf))
launch = epoch.add_days(grid[best, 0] - epoch.mjd2000)
print(
    f"5. best of {len(grid)} Lambert arcs: depart {launch.to_iso()[:10]}, "
    f"{grid[best, 1]:.0f} days, v_inf {departure_vinf[best] / 1e3:.2f} + "
    f"{arrival_vinf[best] / 1e3:.2f} km/s"
)

# 6. A gravity assist at Venus ---------------------------------------------
venus = pk.Planet.jpl_low_precision("venus")
venus_state = venus.state(epoch.mjd2000)
v_planet = venus_state[3:]
v_in = [v + d for v, d in zip(v_planet, [4_000.0, -3_000.0, 500.0])]
v_out = pk.flyby_outgoing_velocity(v_in, v_planet, 1.1 * venus.radius, 0.5, venus.body_mu)
rel_in = np.subtract(v_in, v_planet)
rel_out = np.subtract(v_out, v_planet)
turn = math.degrees(math.acos(np.dot(rel_in, rel_out) / np.linalg.norm(rel_in) ** 2))
gain = np.linalg.norm(v_out) - np.linalg.norm(v_in)
assert abs(np.linalg.norm(rel_out) - np.linalg.norm(rel_in)) < 1e-6
print(f"6. Venus flyby turns v_inf by {turn:.1f} deg, heliocentric speed {gain:+.0f} m/s")

# 7. A multi-leg mission from the gym ---------------------------------------
cassini = gym.Problem("gym.cassini1")
print(f"7. {cassini.get_name()}: {cassini.metadata()['decisions']}")
# The best decision found by the fcmaes examples in the chapter.
x_cassini = [
    -772.4793517902046, 136.48521715578022, 449.40073064324264,
    57.37453338756138, 948.5084493971979, 4434.572372085129,
]
trajectory = cassini.decode(x_cassini)["trajectory"]
print(
    f"   {len(trajectory['arcs'])} Lambert legs, total delta-v "
    f"{cassini.fitness(x_cassini)[0]:.0f} m/s over "
    f"{trajectory['total_days'] / pk.JULIAN_YEAR_DAYS:.1f} years"
)
print(f"   delta-v per encounter [m/s]: {[round(v) for v in trajectory['delta_v_mps']]}")
# The Venus-Venus leg lasts almost exactly two Venus years: nearly a full
# revolution, where the Lambert arc is extremely sensitive to the flight time.
nudged = list(x_cassini)
nudged[2] += 1e-4
print(f"   one Venus-Venus leg 9 s longer: {cassini.fitness(nudged)[0]:.0f} m/s")

# 8. Low thrust: a Sims-Flanagan leg --------------------------------------
# Leave Earth on the best Lambert arc of section 5, but let the launcher give
# only 2.5 km/s of the required departure excess and coast (zero throttle).
t_launch, tof = grid[best]
r0 = earth.state(t_launch)
r1 = mars.state(t_launch + tof)
lambert = problems[best].solutions[0]
excess = np.subtract(lambert.departure_velocity, r0[3:])
departure = [*r0[:3], *np.add(r0[3:], excess / np.linalg.norm(excess) * 2_500.0)]
segments = 10
leg = pk.SimsFlanaganLeg(
    departure, 1_500.0, [[0.0, 0.0, 0.0]] * segments, r1, 1_500.0,
    tof * DAY, 0.5, 3_000.0 * pk.STANDARD_GRAVITY, pk.MU_SUN,
)
miss = leg.mismatch_constraints()
print(
    f"8. coasting Sims-Flanagan leg misses Mars by {math.hypot(*miss[:3]) / 1e9:.1f} "
    f"million km and {math.hypot(*miss[3:6]) / 1e3:.2f} km/s; "
    f"{len(leg.throttle_constraints())} throttle constraints"
)

# 9. Low thrust: a feasible TOPS solution ----------------------------------
root = Path(__file__).resolve().parents[2]
record = json.loads(
    (root / "crates/pykep-trajopt/data/solutions/twobody.P4.json").read_text("utf-8")
)
tops = gym.Problem(record["id"], nseg=record["options"]["nseg"])
result = tops.evaluate(record["x"])
worst = max(abs(value) for value in result["equalities"])
print(
    f"9. {record['id']} with {tops.metadata()['settings']['segments']} segments: "
    f"final mass {-result['objectives'][0]:.3f} of the initial mass, "
    f"largest mismatch {worst:.1e}"
)
assert worst < 1e-8
