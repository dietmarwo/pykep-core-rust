"""Single- and multi-revolution Lambert transfer.

Units: normalized length, time, and gravitational parameter.
Expected: seven ordered branches, including the zero-revolution solution.
Runtime: constant bounded solve, normally below 1 ms with a release wheel.
Features: default wheel; no external data or runtime.
"""

import pykep_rust as pk


problem = pk.LambertProblem(
    [1.0, 0.0, 0.0], [0.2, 1.1, 0.3], 20.0, 1.0, False, 4
)
assert len(problem.solutions) == 7
assert problem.solutions[0].path == "zero"
print(f"{len(problem.solutions)} ordered solutions")
for solution in problem.solutions:
    print(
        f"{solution.revolutions} rev {solution.path}: "
        f"departure {solution.departure_velocity}"
    )
