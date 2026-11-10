# Python API contract

Install the native wheel and import the collision-safe package:

```python
import numpy as np
import pykep_rust as pk

epoch = pk.Epoch.from_iso("2030-01")
state = pk.classical_to_cartesian(
    [7.0e6, 0.01, 0.4, 1.0, 0.5, 0.2], pk.MU_EARTH
)
future = pk.propagate_lagrangian(state, 60.0, pk.MU_EARTH)

batch = np.asarray([state, future], dtype=np.float64)
times = np.asarray([60.0, -60.0], dtype=np.float64)
propagated = pk.propagate_lagrangian_batch(batch, times, pk.MU_EARTH)
assert propagated.shape == (2, 6)

parallel = pk.propagate_lagrangian_batch(
    batch, times, pk.MU_EARTH, workers=4
)
assert np.array_equal(parallel, propagated)
```

## Trajectory gym

Release wheels from 0.1.5, built with the `trajopt` feature, include
`pykep_rust.trajopt.gym`. The problem protocol exposes raw fitness,
flattened TOPS gradients, metadata, decoding and ordered float64 batches.
See [gym constructors and examples](trajectory-gym.md) and
[Python timing boundaries](matched-benchmarks.md). From 0.1.6 the same
wheels export `pykep_rust.trajopt.SimsFlanaganChain` for multiple shooting
([low-thrust legs](low-thrust-legs.md#multiple-shooting-chain)).

## Units, shapes, and defaults

- Angles are radians. Epoch and ephemeris arguments are MJD2000 days.
  Propagation durations are seconds when the supplied gravitational parameter
  uses SI units; normalized inputs produce normalized output.
- Cartesian states are `[x, y, z, vx, vy, vz]`. Classical elements are
  `[a, e, i, Ω, ω, ν]`. Modified equinoctial elements are
  `[p, f, g, h, k, L]`.
- Scalar vector inputs accept finite Python sequences. Element and propagation
  batches require two-dimensional `float64` arrays with shape `N × 6`;
  one-dimensional epoch/time batches require `float64` arrays. Strided and
  read-only arrays are accepted and outputs are newly owned.
- Every parallel batch accepts `workers=0` for the shared pool, `workers=1`
  for a serial native loop, or an exact positive worker count. Results and the
  first reported error retain input order.
- Jacobians are row-major, output-by-input. State-transition matrices are
  `6 × 6`. Sims–Flanagan and ZOH leg matrix shapes are documented with their
  transcription in the low-thrust guides.
- Adaptive propagators default to relative and absolute tolerance `1e-12`,
  no maximum step, and an implementation limit of 100,000 accepted/rejected
  steps where exposed. ZOH legs default to `cut=0.5`.

## Errors and ownership

NaN, infinity, invalid dimensions, non-positive physical parameters, and
invalid time grids are rejected before numerical work. Invalid input uses
`ValueError` (or `TypeError` when a NumPy dtype/rank cannot satisfy the typed
buffer). Numerical failures derive from `PykepError`:
`ConvergenceError`, `SingularGeometryError`, `IntegrationError`, and
`UnsupportedCapabilityError`.

Objects copy constructor inputs and can outlive those Python sequences.
`Planet` and immutable leg objects can be reused from multiple Python threads.
Batch conversion, anomaly, Stumpff, ephemeris, mission, propagation, dynamics,
and leg workloads release the GIL while executing the native loop. Python
wrappers validate and convert data for the Rust libraries.

The nominal Kepler, CR3BP, BCP, Pontryagin, and ZOH propagators accept the
keyword-only `method="taylor"` or `method="dop853"` in their scalar and batch
forms. Kepler, CR3BP, and BCP default to Taylor. Pontryagin and ZOH schedule
calls default to DOP853. `ZohLeg` mismatch, history and Jacobian calls,
including their batch forms, accept the selector and default to DOP853.

See [batch-processing.md](batch-processing.md) for the complete batch matrix,
array shapes, worker-pool contract, SpOC 4 motivation, and guidance for
avoiding nested parallelism.

The complete callable surface and defaults are statically described by the
shipped `py.typed` marker, `_pykep_rust.pyi`, `trajopt/__init__.pyi` and
`trajopt/gym.pyi`. See
[python-migration.md](python-migration.md) for the upstream name and behavior
mapping, [conventions.md](conventions.md) for numerical conventions, and the
model-specific dynamics and leg guides for parameter layouts.
