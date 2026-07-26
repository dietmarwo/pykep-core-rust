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
The object-safe ephemeris interface and analytic Keplerian provider support
thread-safe scalar and ordered batch evaluation. The JPL low-precision
provider supplies approximate heliocentric states for Mercury through Neptune
over 1800–2050. The default `vsop2013` feature embeds a pure-Rust analytical
evaluator for Mercury through Pluto at coefficient thresholds down to `1e-9`.
An adaptive pure-Rust DOP853 facade now defines evaluated model, parameter,
dense-output, terminal-event, and first-order sensitivity contracts for the
remaining dynamics phases. Stateless evaluated Kepler, CR3BP, and bicircular
models provide direct right-hand sides, adaptive propagation, analytic
Jacobians and STMs; CR3BP also provides its effective potential and Jacobi
constant.
Validated zero-order-hold schedules drive normalized Kepler, CR3BP,
modified-equinoctial, and ideal solar-sail models with deterministic switch
ownership, backward propagation, and segment-local sensitivities.
Cartesian and modified-equinoctial Pontryagin models provide mass- and
time-optimal state/costate dynamics, evaluated controls and Hamiltonians, and
first-order sensitivity Jacobians. Their full model Jacobians, and those of
the four ZOH dynamics models, use fixed-size centered differences; the
integrator tolerance is not a derivative-accuracy guarantee. Canonical
Pontryagin costate rates themselves use forward-mode differentiation.
Fixed- and variable-duration Sims–Flanagan legs provide validated
forward/backward transcription, throttle constraints, and analytic fixed-leg
mismatch gradients.
The generic ZOH leg applies the same cut transcription to continuous
piecewise-constant controls for all four built-in ZOH dynamics models and
returns endpoint, control, and time-grid sensitivities. Pinned ZOH-leg
derivative validation uses scaled tolerances up to `3e-5`; see the
module-level sensitivity documentation before using those derivatives in a
tightly converged optimizer.

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
# Ok::<(), pykep_core::PykepError>(())
```

The crate is packaged independently from the `pykep-rust` Python wheel. The
numerical crate has no PyO3, NumPy, C, or C++ dependency; the unpublished
`pykep-py` workspace crate supplies conversion and exception plumbing only.
