"""Installed wheel checks for the Sims–Flanagan multiple-shooting chain."""

import numpy as np
import pytest

from pykep_rust import _pykep_rust

pytestmark = pytest.mark.skipif(
    not hasattr(_pykep_rust, "SimsFlanaganChain"), reason="wheel built without trajopt"
)

DEPARTURE = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
ARRIVAL = [0.96, 0.28, 0.01, -0.28, 0.96, 0.0]
# time of flight, maximum thrust, exhaust velocity, mu
SETTINGS = (0.3, 0.12, 1.5, 1.0)


def chain(arcs, width=4, cut=0.5):
    from pykep_rust.trajopt import SimsFlanaganChain

    return SimsFlanaganChain(DEPARTURE, ARRIVAL, 1.0, arcs, width, *SETTINGS, cut=cut)


def lifted(c, arcs, width=4):
    throttle = np.tile([0.04, 0.22, -0.01], (arcs * width, 1))
    return np.asarray(c.lift(throttle, 0.98, [0.0, 0.02, 0.001]))


def test_layout_rows_and_elements():
    arcs, width = 3, 4
    c = chain(arcs, width)
    # excess, per arc 3*width controls, 7 states per interior node, final mass
    assert c.dimension == 3 + arcs * 3 * width + 7 * (arcs - 1) + 1
    indptr, indices = c.jacobian_indptr, c.jacobian_indices
    assert len(indptr) - 1 == arcs * (7 + width)
    assert indptr[-1] == len(indices)
    rows = [r for group in c.elements for r in group]
    assert rows == list(range(arcs * (7 + width)))
    x = lifted(c, arcs, width)
    objective, values = c.evaluate(x)
    assert objective == -x[-1]
    assert len(values) == arcs * (7 + width)
    assert len(c.jacobian_values(x)) == len(indices)
    with pytest.raises(ValueError):
        c.evaluate(x[:-1])


@pytest.mark.parametrize("cut", [0.0, 0.5, 1.0])
def test_jacobian_matches_central_differences(cut):
    arcs = 3
    c = chain(arcs, cut=cut)
    x = lifted(c, arcs)
    indptr, indices = c.jacobian_indptr, c.jacobian_indices
    analytic = np.zeros((len(indptr) - 1, c.dimension))
    values = c.jacobian_values(x)
    for i in range(len(indptr) - 1):
        for k in range(indptr[i], indptr[i + 1]):
            analytic[i, indices[k]] = values[k]
    for j in range(c.dimension):
        h = 1e-6 * max(1.0, abs(x[j]))
        plus, minus = x.copy(), x.copy()
        plus[j] += h
        minus[j] -= h
        column = (np.asarray(c.evaluate(plus)[1]) - np.asarray(c.evaluate(minus)[1])) / (2 * h)
        np.testing.assert_allclose(analytic[:, j], column, rtol=1e-5, atol=1e-7,
                                   err_msg=f"column {j}")


def test_lift_leaves_the_mismatch_only_at_the_global_cut():
    arcs = 4
    c = chain(arcs)
    _, values = c.evaluate(lifted(c, arcs))
    matching = np.asarray(values).reshape(arcs, 11)[:, :7]
    # The global mismatch stays in arc 2, whose left node comes from the
    # forward half; every other arc matches after the lift.
    for arc in range(arcs):
        if arc != arcs // 2:
            assert np.max(np.abs(matching[arc])) < 1e-12, arc
