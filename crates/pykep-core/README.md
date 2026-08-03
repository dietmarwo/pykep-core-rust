# pykep-core

`pykep-core` is an independent native Rust implementation of numerical
algorithms from pykep version 3. Version 0.1.4 includes:

- **Numerical foundations:** physical constants, Julian-date arithmetic,
  microsecond-resolution epochs, Stumpff functions, Kepler equations, anomaly
  conversions, and allocation-free small-vector/matrix operations.
- **Elements and propagation:** Cartesian, classical, and modified
  equinoctial conversions; analytic 6 by 6 Jacobians; Lagrange and universal
  variable propagation; and Lagrangian/Reynolds state-transition matrices.
- **Mission design:** impulsive transfers, time encodings, flybys,
  single/multi-revolution Lambert branches, and MIMA mass approximations.
- **Ephemerides:** a Keplerian provider, JPL low-precision states for Mercury
  through Neptune over 1800–2050, and feature-gated VSOP2013 states for Mercury
  through Pluto at coefficient thresholds down to `1e-9`.
- **Dynamics and integration:** DOP853 and fixed-system Taylor facades;
  evaluated Kepler, CR3BP, bicircular, ZOH, and Pontryagin models; dense output,
  terminal DOP853 events, and first-order sensitivities.
- **Low-thrust transcriptions:** fixed/variable-duration Sims–Flanagan legs and
  generic continuous-thrust ZOH legs.
- **Ordered batches:** deterministic worker semantics across propagation,
  Lambert, anomaly, vector, ephemeris, and other fallible scalar operations.

Taylor is the default for nominal propagation of the built-in TA-derived
models. DOP853 supports arbitrary user dynamics and direct variational solves.

The ZOH models cover normalized Kepler, CR3BP, modified equinoctial, and ideal
solar-sail dynamics. They define deterministic switch ownership, backward
propagation, and segment-local sensitivities. Rust ZOH legs can select Taylor
or DOP853 for nominal mismatch and history evaluation; compatibility methods
and Jacobians retain DOP853.

The full Pontryagin and ZOH model Jacobians use fixed-size centered differences,
although canonical Pontryagin costate rates use forward-mode differentiation.
Integrator tolerance therefore does not guarantee derivative accuracy. Pinned
ZOH-leg derivative validation uses scaled tolerances up to `3e-5`; consult the
module-level sensitivity documentation before tightly converged optimization.

The crate has no C or C++ runtime dependency.

## Example

```rust
use pykep_core::math::linalg::cross;
use pykep_core::math::stumpff::stumpff_c;
use pykep_core::astro::anomalies::mean_to_eccentric_anomaly;
use pykep_core::astro::elements::{ClassicalElements, classical_to_cartesian};
use pykep_core::time::epoch::Epoch;
use pykep_core::astro::propagation::propagate_lagrangian;
use pykep_core::dynamics::Cr3bpDynamics;
use pykep_core::integration::IntegratorOptions;

assert_eq!(Epoch::from_iso("2000-01")?.mjd2000(), 0.0);
assert_eq!(cross(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0])?, [0.0, 0.0, 1.0]);
assert!((stumpff_c(1e-12)? - 0.5).abs() < 1e-13);
assert!(mean_to_eccentric_anomaly(0.1, 0.5)?.is_finite());
let elements = ClassicalElements::new(7.0e6, 0.01, 0.4, 1.0, 0.5, 0.2);
assert!(classical_to_cartesian(elements, 3.986_004_418e14)?[0].is_finite());
let state = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
assert!(propagate_lagrangian(&state, 0.5, 1.0)?[1] > 0.0);
let rotating = [0.8, -0.2, 0.1, 0.03, -0.04, 0.02];
let propagated = Cr3bpDynamics.propagate(
    0.0, rotating, 0.5, 0.012150585609624, IntegratorOptions::default()
)?;
assert!(propagated.state.iter().all(|value| value.is_finite()));

let states = [state; 64];
let times = [0.25; 64];
let batch = pykep_core::astro::propagation::propagate_lagrangian_batch(
    &states, &times, 1.0, 4
)?;
assert_eq!(batch.len(), states.len());
# Ok::<(), pykep_core::PykepError>(())
```

The crate is packaged independently from the `pykep-rust` Python wheel. The
numerical crate has no PyO3, NumPy, C, or C++ dependency; the internal
`publish = false` `pykep-py` workspace crate supplies conversion and exception
plumbing for the separately published Python distribution only.
