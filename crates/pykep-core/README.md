# pykep-core

`pykep-core` is an independent native Rust implementation of numerical
algorithms from pykep version 3. The current phase provides physical constants,
Julian-date arithmetic, microsecond-resolution epochs, stable Stumpff
functions, Kepler-equation residuals, anomaly conversions, and allocation-free
small-vector/matrix operations. Cartesian, classical Keplerian, and modified
equinoctial conversions include analytic 6 by 6 Jacobians. Two-body
propagation supports Lagrange coefficients and universal variables, with
analytic Lagrangian and Reynolds state-transition matrices.
Mission-design utilities include impulsive transfers, time encodings, flybys,
single/multi-revolution Lambert branches, and MIMA mass approximations.

The crate has no C or C++ runtime dependency. Later orbital algorithms remain
planned and are not represented by the current status probe.

## Example

```rust
use pykep_core::math::linalg::cross;
use pykep_core::math::stumpff::stumpff_c;
use pykep_core::astro::anomalies::mean_to_eccentric_anomaly;
use pykep_core::astro::elements::{ClassicalElements, classical_to_cartesian};
use pykep_core::time::epoch::Epoch;
use pykep_core::astro::propagation::propagate_lagrangian;

assert_eq!(Epoch::from_iso("2000-01")?.mjd2000(), 0.0);
assert_eq!(cross(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0])?, [0.0, 0.0, 1.0]);
assert!((stumpff_c(1e-12)? - 0.5).abs() < 1e-13);
assert!(mean_to_eccentric_anomaly(0.1, 0.5)?.is_finite());
let elements = ClassicalElements::new(7.0e6, 0.01, 0.4, 1.0, 0.5, 0.2);
assert!(classical_to_cartesian(elements, 3.986_004_418e14)?[0].is_finite());
let state = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
assert!(propagate_lagrangian(&state, 0.5, 1.0)?[1] > 0.0);
# Ok::<(), pykep_core::PykepError>(())
```

The crate is deliberately marked `publish = false` until a useful,
cross-module orbital-mechanics milestone is complete.
