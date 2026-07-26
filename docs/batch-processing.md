# Ordered parallel batch processing

`pykep-core` and `pykep_rust` deliberately extend the scalar-oriented pykep
surface with ordered parallel batches. This is not an upstream compatibility
claim. Experience optimizing large Lambert-transfer populations during the
SpOC 4 competition, captured in the
[`dietmarwo/pykep-lambert`](https://github.com/dietmarwo/pykep-lambert)
comparison project, motivated moving the reusable pattern into this library.

The extension addresses two independent costs:

1. one Python-to-native transition replaces thousands of scalar calls; and
2. independent native evaluations can use several CPU cores.

Every batch returns results in input order. A successful row is numerically
the same scalar computation; batching does not change the physical model,
units, frame, tolerances, Lambert branch order, or error criteria.

## Worker contract

APIs with a `workers` argument use one shared contract:

| `workers` | Execution |
|---:|---|
| `0` | Rayon's process-wide shared pool; this is the Python default |
| `1` | Serial native loop |
| `N > 1` | A cached pool containing exactly `N` worker threads |

Explicit pools are cached by worker count, so repeated optimizer generations
do not pay thread-startup cost. Explicit counts above 1024 are rejected.
Python releases the GIL around native work.

Parallel evaluation collects row results before returning an error. This
keeps both successful output and the reported first error deterministic in
input order, even if a later failing row finishes first on another worker.
An empty batch returns an empty result with the documented output shape.

Do not add another thread pool around a parallel batch. In an already
parallel optimizer or task scheduler, pass `workers=1` at the inner level to
avoid oversubscription. Cheap arithmetic often benefits from one native batch
because it removes Python-call overhead but may not benefit from multiple
threads. Measure representative batch sizes.

## Rust interfaces

The following astrodynamics operations have named Rust batch APIs:

- `propagate_lagrangian_batch`, `propagate_universal_batch`, and
  `propagate_keplerian_batch`;
- `propagate_lagrangian_with_stm_batch` and
  `propagate_lagrangian_grid_parallel`;
- `solve_lambert_batch`, using one `LambertRequest` per problem;
- `Ephemeris::states_parallel`, `accelerations_parallel`,
  `periods_parallel`, and `elements_parallel`;
- all anomaly-conversion functions with a `_batch` suffix;
- `dot_batch`, `norm_batch`, `normalize_batch`, `cross_batch`, and
  `skew_batch`;
- `evaluate_zoh_mismatch_batch_parallel`.

Other immutable scalar Rust operations can use the same public
`pykep_core::batch::try_map` executor. This avoids duplicating dozens of
near-identical request structures while preserving typed scalar functions as
the only numerical implementation:

```rust
use pykep_core::astro::transfers::hohmann;
use pykep_core::batch::try_map;

let radii = [(1.0, 2.0), (1.2, 2.4)];
let transfers = try_map(&radii, 2, |&(r1, r2)| hohmann(r1, r2, 1.0))?;
assert_eq!(transfers.len(), radii.len());
# Ok::<(), pykep_core::PykepError>(())
```

`try_map` is appropriate only for independent rows. Algorithms that carry
state from one time to the next remain scalar/sequential unless their public
contract explicitly defines independent initial-value problems.

## Python interfaces

Fixed-size numeric batches use `float64` NumPy arrays. Ragged schedules,
encoding vectors, and batches of immutable leg objects use Python sequences.
The shipped `_pykep_rust.pyi` is the authoritative signature reference.

### Foundations and representations

| Family | Batch functions |
|---|---|
| Stumpff | `stumpff_c_batch`, `stumpff_s_batch` |
| Anomalies | `_batch` counterpart for every elliptic and hyperbolic anomaly conversion |
| Three-vectors | `dot_batch`, `norm_batch`, `normalize_batch`, `cross_batch`, `skew_batch` |
| Elements | `_batch` counterpart for all six conversions |
| Element derivatives | `cartesian_to_modified_equinoctial_jacobian_batch`, `modified_equinoctial_to_cartesian_jacobian_batch` |

Vector input has shape `N × 3`; element/state input has shape `N × 6`.
Vector outputs have shape `N` or `N × 3`, skew matrices `N × 3 × 3`,
element outputs `N × 6`, and element Jacobians `N × 6 × 6`.

### Propagation, Lambert, and ephemerides

| Family | Batch functions |
|---|---|
| Two-body | `propagate_lagrangian_batch`, `propagate_universal_batch`, `propagate_keplerian_batch` |
| Two-body STM/grid | `propagate_lagrangian_with_stm_batch`, `propagate_lagrangian_grid(..., workers=...)` |
| Adaptive evaluated dynamics | `propagate_kepler_dynamics_batch`, `propagate_cr3bp_batch`, `propagate_bcp_batch` and all three `_with_stm_batch` variants |
| Lambert | `lambert_problem_batch` |
| Ephemerides | `Planet.states`, `Planet.acceleration_batch`, `Planet.period_batch`, `Planet.elements_batch` |

Propagation batches pair row `states[i]` with `times[i]` or
`final_times[i]`. STM batches return `(states[N,6], stms[N,6,6])`.
`lambert_problem_batch` pairs `initial_positions[N,3]`,
`final_positions[N,3]`, and `times[N]`, while sharing `mu`, direction, and
maximum-revolution configuration. It returns `N` normal `LambertProblem`
objects; each object retains the scalar zero/left/right branch ordering.

```python
states = np.tile(
    np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), (32_768, 1)
)
times = np.linspace(0.0, 1.0, len(states))
propagated = pk.propagate_lagrangian_batch(
    states, times, 1.0, workers=8
)
assert propagated.shape == states.shape
```

### Mission utilities

| Family | Batch functions |
|---|---|
| Circular transfers | `hohmann_batch`, `bielliptic_batch` |
| Time encodings | `alpha_to_direct_batch`, `direct_to_alpha_batch`, `eta_to_direct_batch`, `direct_to_eta_batch` |
| Flybys | `flyby_constraints_batch`, `flyby_constraints_jacobian_batch`, `flyby_delta_v_batch`, `flyby_outgoing_velocity_batch` |
| Mass estimates | `mima_batch`, `mima2_batch` |

Hohmann and bi-elliptic batches return `(delta_v[N], time[N],
impulses[N,K])`. Flyby constraint output is `N × 2`, its Jacobian is
`N × 2 × 6`, and outgoing velocity is `N × 3`. MIMA variants return
`(mass[N], acceleration[N])`.

### Controlled dynamics and legs

| Family | Batch functions |
|---|---|
| Evaluated RHS/invariants | `kepler_rhs_batch`, `cr3bp_rhs_batch`, `bcp_rhs_batch`, `cr3bp_effective_potential_batch`, `cr3bp_jacobi_constant_batch` |
| ZOH RHS | all four `zoh_*_rhs_batch` functions |
| ZOH schedules | all four `propagate_zoh_*_batch` functions |
| Pontryagin | Cartesian/equinoctial RHS, control, Hamiltonian, and propagation `_batch` functions |
| Sims–Flanagan objects | class batch methods for mismatch/throttle constraints and available Jacobians |
| Generic ZOH-leg objects | `mismatch_constraints_batch`, `mismatch_jacobian_batch`, `state_history_batch` |

ZOH schedule batches accept one state row, boundary vector, and control matrix
per independent schedule. Shared physical constants and integrator settings
remain scalar arguments. Pontryagin batches share one optimality mode and
parameter vector.

## Where the primitives are useful

These batches support existing computations without adding new physical
models:

- Lambert departure/arrival/time-of-flight sweeps;
- pork-chop or transfer-table calculations assembled from ephemeris and
  Lambert batches;
- optimizer populations for CMA-ES, differential evolution, PGPE, MODE, or
  similar methods;
- Monte Carlo evaluation of independent uncertain initial conditions;
- state/STM ensembles for covariance or sensitivity workflows;
- independent flyby candidates, MIMA estimates, and low-thrust legs.

Closest-approach, eclipse, access, conjunction, and other event physics are
not implemented merely because the underlying state batches make such
external calculations easier. Use an appropriate validated model for those
quantities.

## Validation

Rust tests compare named batches with scalar functions for serial, global,
and explicit worker pools and verify deterministic error order. Python tests
exercise every batch family, compare values with scalar calls, validate
shapes, and verify the runtime exports against the typed stub. The companion
`pykep-lambert` project remains the end-to-end performance comparison for the
SpOC 4–motivated propagation/Lambert workload.
