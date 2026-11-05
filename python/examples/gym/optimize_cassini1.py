"""Optimize the gym's Cassini1 mission with fcmaes BiteOpt restarts.

Requires: ``python -m pip install fcmaes-rust`` next to the pykep-rust wheel.
Units: the objective is total delta-v in m/s; decisions are days (MJD2000
departure epoch and leg durations).
Runtime: about half a second per restart with a release wheel.
"""

import argparse

import numpy as np

import fcmaes_rust as fcmaes
from pykep_rust.trajopt import gym

PENALTY = 1e10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--evaluations", type=int, default=50_000)
    args = parser.parse_args()

    problem = gym.Problem("gym.cassini1")
    lower, upper = map(np.asarray, problem.get_bounds())

    def delta_v(x):
        # A rejected geometry (for example an overflowing flyby) is a bad
        # point, not a reason to stop the search.
        try:
            return problem.fitness(x)[0]
        except (ArithmeticError, RuntimeError, ValueError):
            return PENALTY

    best_x, best_value = None, np.inf
    for seed in range(args.restarts):
        x, value, evaluations, _, _ = fcmaes.optimize_bite(
            delta_v, [], lower, upper, seed=seed,
            max_evaluations=args.evaluations, M=6,
        )
        print(f"restart {seed}: {value:8.1f} m/s after {evaluations} evaluations")
        if value < best_value:
            best_x, best_value = x, value

    trajectory = problem.decode(list(best_x))["trajectory"]
    print(f"best {best_value:.1f} m/s, flight {trajectory['total_days']:.0f} days")
    for name, value in zip(problem.metadata()["decisions"], best_x):
        print(f"  {name:28s} {value:12.6f}")


if __name__ == "__main__":
    main()
