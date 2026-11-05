"""Installed wheel checks for gym capabilities and dimensions."""

import copy
import numpy as np
import pytest

from pykep_rust import _pykep_rust
from pykep_rust.trajopt import gym

pytestmark = pytest.mark.skipif(
    not hasattr(_pykep_rust, "_gym_ids"), reason="wheel built without trajopt"
)


def midpoint(problem):
    low, high = np.asarray(problem.get_bounds())
    return (low+high)*.5


def test_catalog_and_protocol():
    names = gym.ids()
    assert len(names) == len(set(names)) == 82
    for name in names:
        encodings = ("uniform", "softmax") if name.startswith("tops.") else ("uniform",)
        for encoding in encodings:
            p = gym.Problem(name, time_encoding=encoding)
            bounds = np.array(p.get_bounds())
            assert bounds.shape[0] == 2
            assert np.isfinite(bounds).all() and np.all(bounds[0] <= bounds[1])
    for name in names:
        if name.startswith("gym."):
            p = gym.Problem(name)
            x = midpoint(p)
            f = p.fitness(x)
            assert np.isfinite(f).all() and len(f) == p.get_nobj()
            assert p.get_nec() == p.get_nic() == 0 and not p.has_gradient()
            assert copy.deepcopy(p).fitness(x) == f


@pytest.mark.parametrize("moving", [False, True])
@pytest.mark.parametrize("encoding", ["uniform", "softmax"])
def test_tops_complete_gradient_and_ordered_batch(moving, encoding):
    constructor = gym.tops_twobody_mb if moving else gym.tops_twobody
    p = constructor("P4", nseg=4, time_encoding=encoding)
    x = midpoint(p)
    for i, name in enumerate(p.metadata()["decisions"]):
        if name.startswith("direction_") and name.endswith("_x"):
            x[i:i+3] = [.7, .4, -.1]
    y = np.asarray(p.fitness(x))
    J = np.asarray(p.gradient(x)).reshape(len(y), len(x))
    assert np.isfinite(J).all()
    d = np.linspace(-.2, .2, len(x))
    h = 1e-5
    expected = (np.asarray(p.fitness(x+h*d)) - p.fitness(x-h*d))/(2*h)
    np.testing.assert_allclose(J @ d, expected, atol=1e-7, rtol=2e-6)
    array = np.vstack([x, x+h*d, x-h*d])[::-1]
    result = p.fitness_batch(array, workers=2)
    np.testing.assert_array_equal(result, [p.fitness(row) for row in array])
    with pytest.raises(ValueError):
        p.fitness_batch(array[:, :-1])
    with pytest.raises(TypeError):
        p.fitness_batch(array.astype(np.float32))
    assert len(p.decode(x)["grid"]) == 5


def test_sail_objective_and_launcher():
    p = gym.tops_ss("P0", nseg=6)
    x = midpoint(p)
    assert p.get_nec() == 6 and p.get_nic() == 0
    assert p.get_bounds()[0][-1] == p.get_bounds()[1][-1]
    assert p.fitness(x)[0] == x[-1]
    assert gym.launcher_mass("falcon9", -1) == .1
    assert np.isfinite(gym.launcher_mass("ariane5", 2.7, -13.))


def test_embedded_dataset_bytes_match_provenance():
    """The checkout must keep the pinned TOPS and launcher bytes on every platform."""
    import hashlib
    import json
    from pathlib import Path

    data = Path(__file__).resolve().parents[2] / "crates" / "pykep-trajopt" / "data"
    if not data.is_dir():
        pytest.skip("source checkout not available")
    inventory = json.loads((data / "provenance.json").read_text(encoding="utf-8"))
    for name, expected in inventory["sha256"].items():
        candidates = [data / name, data / "tops" / name]
        path = next(p for p in candidates if p.is_file())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
