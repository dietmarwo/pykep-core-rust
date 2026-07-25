"""Batch evaluation that avoids per-item Python calls.

Units: normalized two-body state, time, and gravitational parameter.
Expected: an independent 4096 × 6 array with finite propagated states.
Runtime: one native O(N) call; about 0.1 µs/item in the Phase 16 benchmark.
Features: default wheel and NumPy.
"""

import numpy as np

import pykep_rust as pk


count = 4_096
states = np.tile(
    np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),
    (count, 1),
)
times = np.linspace(0.0, 1.0, count, dtype=np.float64)
propagated = pk.propagate_lagrangian_batch(states, times, 1.0)
assert propagated.shape == (count, 6)
assert np.isfinite(propagated).all()
assert not np.shares_memory(propagated, states)
print(f"propagated {count} states in one native batch: {propagated.shape}")
