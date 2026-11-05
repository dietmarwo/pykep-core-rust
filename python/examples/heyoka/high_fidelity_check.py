#!/usr/bin/env python3
"""Search a route with pykep-rust and fcmaes, then check it with heyoka.

Stage 1, pykep-rust and fcmaes-rust: BiteOpt restarts optimize the gym's
Cassini1 mission. The model is patched conics: Lambert arcs between
planet centers, instantaneous flybys and JPL low-precision ephemerides.

Stage 2, heyoka: every leg is propagated again in a higher-fidelity model.
The planets follow VSOP2013, evaluated inside the equations of motion, and
the spacecraft feels the Sun plus every other planet of the route. The
leg's own departure and arrival planets are left out, as in patched conics.
A closest-approach event reports where the spacecraft really passes the
target planet. First-order variational equations give the Jacobian for a
damped Newton retargeting of the departure velocity and arrival time.

Requires: python -m pip install pykep-rust fcmaes-rust heyoka
Runtime: a few seconds for the search, plus a few seconds of JIT
compilation per leg on the first run (heyoka caches compiled code on disk).
"""

import argparse
import json
from pathlib import Path

import numpy as np

import fcmaes_rust as fcmaes
import heyoka as hy
import pykep_rust as pk
from pykep_rust.trajopt import gym

AU = pk.ASTRONOMICAL_UNIT
DAY = pk.DAY_TO_SECONDS
AU_PER_DAY = AU / DAY  # m/s in one AU/day
SEQUENCE = ["earth", "venus", "venus", "earth", "jupiter", "saturn"]
VSOP_INDEX = {"venus": 2, "earth": 3, "jupiter": 5, "saturn": 6}  # Earth-Moon barycenter
# The best Cassini1 decision found by the examples of the astrodynamics introduction.
STORED_DECISION = [
    -772.4793517902046, 136.48521715578022, 449.40073064324264,
    57.37453338756138, 948.5084493971979, 4434.572372085129,
]


def search(restarts: int, evaluations: int) -> list[float]:
    """Stage 1: global search in the patched-conic model."""
    problem = gym.Problem("gym.cassini1")
    lower, upper = map(np.asarray, problem.get_bounds())

    def delta_v(x):
        try:
            return problem.fitness(x)[0]
        except (ArithmeticError, RuntimeError, ValueError):
            return 1e10

    best_x, best_value = None, np.inf
    for seed in range(restarts):
        x, value, _, _, _ = fcmaes.optimize_bite(
            delta_v, [], lower, upper, seed=seed, max_evaluations=evaluations, M=6)
        if value < best_value:
            best_x, best_value = x, value
    print(f"stage 1: best patched-conic delta-v {best_value:.1f} m/s "
          f"from {restarts} BiteOpt restarts")
    return [float(v) for v in best_x]


class HighFidelityModel:
    """Heliocentric ecliptic J2000 dynamics in AU and days, time in MJD2000."""

    def __init__(self, threshold: float):
        self.mu = hy.model.get_vsop2013_mus()  # Sun first, AU^3/day^2
        # VSOP2013 time: Julian millennia from JD 2451545.0 (MJD2000 0.5), TDB.
        millennia = (hy.time - 0.5) / 365250.0
        self.planets = {name: hy.model.vsop2013_cartesian(index, millennia, threshold)
                        for name, index in VSOP_INDEX.items()}
        self.names = list(self.planets)
        dummy = hy.make_vars("unused")
        self._ephemeris = hy.cfunc(
            [c for name in self.names for c in self.planets[name]], [dummy],
            compact_mode=True)
        self.r = list(hy.make_vars("x", "y", "z"))
        self.v = list(hy.make_vars("vx", "vy", "vz"))

    def state(self, name: str, epoch: float) -> np.ndarray:
        """VSOP2013 state [AU, AU/day] of a planet at an MJD2000 epoch."""
        values = self._ephemeris(np.zeros(1), time=epoch)
        k = self.names.index(name)
        return np.array(values[6 * k:6 * k + 6])

    def leg(self, departure: str, arrival: str):
        """Integrators for one leg: nominal with a closest-approach event, and variational."""
        r, v = self.r, self.v
        radius3 = hy.sum([q * q for q in r]) ** 1.5
        acc = [-self.mu[0] * q / radius3 for q in r]
        perturbers = [n for n in self.names if n not in (departure, arrival)]
        for k, name in enumerate(perturbers):
            s = self.planets[name]
            d = [s[i] - r[i] for i in range(3)]
            d3 = hy.sum([q * q for q in d]) ** 1.5
            s3 = hy.sum([s[i] * s[i] for i in range(3)]) ** 1.5
            # Direct pull minus the indirect term of the accelerated heliocentric frame;
            # hy.par[k] switches the perturber on (1) or off (0) without recompiling.
            acc = [acc[i] + hy.par[k] * self.mu[VSOP_INDEX[name]] * (d[i] / d3 - s[i] / s3)
                   for i in range(3)]
        system = [(r[i], v[i]) for i in range(3)] + [(v[i], acc[i]) for i in range(3)]
        target = self.planets[arrival]
        approach = hy.sum([(r[i] - target[i]) * (v[i] - target[3 + i]) for i in range(3)])
        hits = []
        nominal = hy.taylor_adaptive(
            system, [1.0, 0, 0, 0, 0.017, 0], pars=[1.0] * len(perturbers),
            compact_mode=True,
            nt_events=[hy.nt_event(approach, lambda ta, t, d: hits.append(t),
                                   direction=hy.event_direction.positive)])
        variational = hy.taylor_adaptive(
            hy.var_ode_sys(system, hy.var_args.vars, order=1),
            [1.0, 0, 0, 0, 0.017, 0], pars=[1.0] * len(perturbers), compact_mode=True)
        return nominal, variational, hits, len(perturbers)


def check_leg(model, departure, arrival, t0, t1, patched_start, *, newton_steps=20):
    nominal, variational, hits, n_perturbers = model.leg(departure, arrival)
    start = model.state(departure, t0)
    jpl = np.array(pk.Planet.jpl_low_precision(departure).state(t0))
    # Keep the patched-conic hyperbolic excess, relative to the VSOP2013 planet.
    v_patched = start[3:] + (np.asarray(patched_start[3:]) - jpl[3:]) / AU_PER_DAY

    def miss(velocity, t_final, perturbed=True):
        variational.pars[:] = [1.0 if perturbed else 0.0] * n_perturbers
        variational.time = t0
        variational.state[:6] = np.concatenate([start[:3], velocity])
        variational.state[6:42] = np.eye(6).ravel()
        variational.propagate_until(t_final)
        state = variational.state
        planet = model.state(arrival, t_final)
        jac_rv = state[6:42].reshape(6, 6)[:3, 3:]
        return state[:3] - planet[:3], jac_rv, state[3:6] - planet[3:]

    ephemeris_only = np.linalg.norm(miss(v_patched, t1, perturbed=False)[0])
    error, jacobian, _ = miss(v_patched, t1)
    first_miss = np.linalg.norm(error)
    condition = np.linalg.cond(jacobian)

    # Where does the uncorrected trajectory really pass the target?
    hits.clear()
    nominal.pars[:] = [1.0] * n_perturbers
    nominal.time = t0
    nominal.state[:] = np.concatenate([start[:3], v_patched])
    nominal.propagate_until(t1 + 30.0)
    closest = None
    for t in hits:
        nominal.update_d_output(t)
        distance = np.linalg.norm(nominal.d_output[:3] - model.state(arrival, t)[:3])
        if closest is None or distance < closest[1]:
            closest = (t, distance)

    # Damped Newton on departure velocity and arrival time. One day of arrival
    # shift is weighted like 1 m/s of correction.
    velocity, t_final, weight = v_patched.copy(), t1, AU_PER_DAY
    for _ in range(newton_steps):
        error, jacobian, relative = miss(velocity, t_final)
        if np.linalg.norm(error) * AU < 1e3:
            break
        matrix = np.hstack([jacobian, relative[:, None] * weight])
        step = np.linalg.lstsq(matrix, -error, rcond=None)[0]
        scale = 1.0
        while scale > 1e-4:
            trial_v, trial_t = velocity + scale * step[:3], t_final + scale * step[3] * weight
            if np.linalg.norm(miss(trial_v, trial_t)[0]) < np.linalg.norm(error):
                break
            scale /= 2
        velocity, t_final = trial_v, trial_t
    final_miss = np.linalg.norm(miss(velocity, t_final)[0])
    return {
        "leg": f"{departure}-{arrival}",
        "days": t1 - t0,
        "miss_vsop2013_km": ephemeris_only * AU / 1e3,
        "miss_high_fidelity_km": first_miss * AU / 1e3,
        "closest_approach_km": None if closest is None else closest[1] * AU / 1e3,
        "closest_approach_shift_hours": None if closest is None else (closest[0] - t1) * 24,
        "jacobian_condition": condition,
        "correction_mps": np.linalg.norm(velocity - v_patched) * AU_PER_DAY,
        "arrival_shift_hours": (t_final - t1) * 24,
        "retargeted_miss_km": final_miss * AU / 1e3,
        "converged": bool(final_miss * AU < 1e3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stored", action="store_true",
                        help="skip the search and check the stored best decision")
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--evaluations", type=int, default=50_000)
    parser.add_argument("--vsop-threshold", type=float, default=1e-6,
                        help="VSOP2013 truncation; smaller is more accurate and compiles slower")
    parser.add_argument("--output", type=Path, help="write the results as JSON")
    args = parser.parse_args()

    decision = STORED_DECISION if args.stored else search(args.restarts, args.evaluations)
    problem = gym.Problem("gym.cassini1")
    patched = problem.fitness(decision)[0]
    trajectory = problem.decode(decision)["trajectory"]
    epochs = trajectory["epochs_mjd2000"]

    print(f"stage 2: heyoka {hy.__version__}, VSOP2013 threshold {args.vsop_threshold:g}")
    model = HighFidelityModel(args.vsop_threshold)
    rows = [check_leg(model, SEQUENCE[k], SEQUENCE[k + 1], epochs[k], epochs[k + 1], arc["start"])
            for k, arc in enumerate(trajectory["arcs"])]

    print(f"{'leg':16} {'days':>7} {'VSOP km':>12} {'+planets km':>12} {'closest km':>11} "
          f"{'cond':>9} {'fix m/s':>8} {'shift h':>8} {'left km':>10}")
    for row in rows:
        print(f"{row['leg']:16} {row['days']:7.1f} {row['miss_vsop2013_km']:12,.0f} "
              f"{row['miss_high_fidelity_km']:12,.0f} "
              f"{row['closest_approach_km'] or float('nan'):11,.0f} "
              f"{row['jacobian_condition']:9.1e} {row['correction_mps']:8.2f} "
              f"{row['arrival_shift_hours']:8.2f} {row['retargeted_miss_km']:10,.3f}"
              f"{'' if row['converged'] else '  not retargeted'}")
    fixed = [r for r in rows if r["converged"]]
    print(f"patched conics: {patched:.1f} m/s; retargeting {len(fixed)} of {len(rows)} legs "
          f"in the high-fidelity model adds {sum(r['correction_mps'] for r in fixed):.1f} m/s")
    for row in rows:
        if not row["converged"]:
            print(f"{row['leg']}: Jacobian condition {row['jacobian_condition']:.1e}; this "
                  "near-resonant leg needs the previous flyby retargeted, not a departure burn")
    if args.output:
        args.output.write_text(json.dumps({"decision": decision, "patched_mps": patched,
                                           "legs": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
