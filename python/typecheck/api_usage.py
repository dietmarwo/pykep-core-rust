"""Representative statically checked uses of the public Python API."""

from __future__ import annotations

from typing import assert_type

import numpy as np
import numpy.typing as npt

import pykep_rust as pk
from pykep_rust.trajopt import SimsFlanaganChain, gym


epoch = pk.Epoch(0.0, "mjd2000")
assert_type(epoch.mjd, float)
assert_type(epoch.add_seconds(10.0), pk.Epoch)

elements = [7.0e6, 0.01, 0.4, 1.0, 0.5, 0.2]
state = pk.classical_to_cartesian(elements, pk.MU_EARTH)
assert_type(state, list[float])
assert_type(pk.propagate_lagrangian(state, 60.0, pk.MU_EARTH), list[float])

element_batch: npt.NDArray[np.float64] = np.asarray(
    [elements, elements], dtype=np.float64
)
state_batch = pk.classical_to_cartesian_batch(element_batch, pk.MU_EARTH)
assert_type(state_batch, npt.NDArray[np.float64])

earth = pk.Planet.jpl_low_precision("earth")
assert_type(earth.state(epoch.mjd2000), list[float])
assert_type(earth.elements(epoch.mjd2000, "classical_true"), list[float])

transfer = pk.LambertProblem(
    [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 2.0, 1.0
)
assert_type(transfer.solutions, list[pk.LambertSolution])

leg = pk.ZohLeg(
    pk.ZohModel.Kepler,
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    [[0.0, 1.0, 0.0, 0.0]],
    [0.9, 0.1, 0.0, -0.1, 0.9, 0.0, 1.0],
    [0.0, 0.1],
    [0.0],
)
assert_type(leg.mismatch_constraints(), list[float])

tops = gym.tops_twobody("P4", nseg=16, time_encoding="softmax")
assert_type(tops.get_bounds(), tuple[list[float], list[float]])
assert_type(tops.gradient(tops.get_bounds()[0]), list[float])

chain = SimsFlanaganChain(
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.96, 0.28, 0.01, -0.28, 0.96, 0.0],
    1.0, 3, 4, 0.3, 0.12, 1.5, 1.0,
)
assert_type(chain.dimension, int)
assert_type(chain.elements, list[list[int]])
decision = chain.lift([[0.04, 0.22, -0.01]] * 12, 0.98, [0.0, 0.02, 0.001])
assert_type(decision, list[float])
assert_type(chain.evaluate(decision), tuple[float, list[float]])
assert_type(chain.jacobian_values(decision), list[float])
