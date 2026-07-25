# ADR 0004: adaptive integration backend

- Status: accepted
- Date: 2026-07-25

## Context

The upstream `ta` modules return heyoka expression graphs and cached
Taylor-adaptive integrators. Rust cannot preserve that implementation API
without a C++/LLVM runtime. The remaining physical models and legs instead
need an evaluated-model interface with:

- six-, seven-, fourteen-, and augmented state dimensions up to 126;
- constant parameters that can change exactly between ZOH segments;
- scalar relative and absolute tolerances down to the upstream test regimes;
- initial/maximum step and accepted/rejected-step limits;
- forward and backward propagation;
- deterministic exact-final-time stops;
- dense evaluation and terminal zero-crossing events;
- state and parameter Jacobians plus arbitrary first-order sensitivity seeds;
- no heap allocation per accepted step on nominal fixed-size hot paths.

Kepler and CR3BP were used as representative smooth and close-approach
systems. ZOH switching is treated as a known discontinuity: a leg ends one
solve exactly at a switch and starts the next with new parameters. It does not
ask a continuous interpolant to cross a discontinuity.

## Candidates

The research snapshot was taken on 2026-07-25.

[`differential-equations` 0.6.1](https://docs.rs/differential-equations/0.6.1/differential_equations/)
provides pure-Rust DOP853, seventh-order dense interpolation, backward solves,
terminal root finding, scalar/component tolerances, and explicit step and
rejection limits. It accepts stack-allocated arrays and user-defined state
types. Its Apache-2.0 license is compatible, and a no-default-feature build
passes Rust 1.88 even though the crate does not declare an MSRV.
Its `simba` dependency currently brings `paste` 1.0.15. RustSec advisory
RUSTSEC-2024-0436 marks that macro crate unmaintained but reports no
vulnerability and no safe upgrade. The exact advisory is narrowly allowlisted
in `deny.toml` and remains visible in `cargo audit` output.

[`ode_solvers` 0.6.2](https://docs.rs/ode_solvers/0.6.2/ode_solvers/) provides
pure-Rust DOP853 and dense output under Apache-2.0. The same-input spike was
faster, but the implementation allocates vectors and intermediate states
inside each solve/step. It has a post-step stop callback rather than
interpolated root location and no first-order sensitivity interface.

[`diffsol` 0.16.1](https://docs.rs/diffsol/0.16.1/diffsol/) has the richest
event, reset, dense-output, forward, and adjoint sensitivity system. It was
screened out before the timing final because its implicit-solver and matrix
abstractions are disproportionate for these small non-stiff explicit systems.
Its optional DSL/JIT path also solves a problem this port explicitly avoids.

A local DOP853 implementation would give maximum workspace control but would
make pykep responsible for a second adaptive-solver implementation, dense
coefficients, root location, step controller, and long-term maintenance.
C++ FFI was excluded by the release requirements.

## Decision

Use `differential-equations` 0.6.1 DOP853 behind the pykep-owned
`integration` facade. No backend type appears in a public signature.

`DynamicsModel<N, P>` evaluates into caller-owned arrays and validates the
physical domain. `DifferentiableDynamicsModel<N, P>` adds row-major state and
parameter Jacobians. `InitialValueProblem` owns time, state, and constant
parameters. `SensitivityProblem` supplies arbitrary `dstate/dseed` and
`dparameters/dseed` matrices, integrating

```text
dS/dt = (df/dstate) S + (df/dparameters) (dparameters/dseed).
```

This directly represents STMs, parameter variations, and the state/control
columns needed by ZOH legs. The facade provides final-state, dense-sampling,
terminal-event, and sensitivity propagations with stable pykep errors and
work counters.

The ordinary and variational final-state paths use fixed-size state storage
and an output callback that does not retain internal steps. Dense and event
APIs allocate returned samples intentionally; the event adapter currently
retains accepted steps because the selected crate's event wrapper owns its
output strategy. Events are not on the remaining leg hot path.

## Measurements

All values are orientation measurements on the same AMD Ryzen 9 9950X host.
CPU frequency was not pinned.

The Git-tracked candidate tool used 30 samples of 200 complete Kepler
propagations with `rtol = atol = 1e-12`:

| Candidate | Mean | Median | Range |
|---|---:|---:|---:|
| pykep facade / `differential-equations` | 15.431 µs | 15.228 µs | 14.882–18.109 µs |
| `ode_solvers` | 7.369 µs | 7.333 µs | 7.299–7.901 µs |

The maintained workspace Criterion profile measured the selected nominal
path at 11.865 µs (95% estimate 11.849–11.884 µs) and the state plus 6 × 6
STM path at 85.663 µs (85.260–86.102 µs).

The matching C++/heyoka harness measured 3.722 µs mean nominal steady-state
propagation and 84.440 µs mean for the STM. Cloning the cached C++ nominal
integrator once cost 0.467 ms; the steady-state result excludes that clone.

## Validation and risks

- A 100-orbit eccentric Kepler solve kept absolute specific-energy drift below
  `2e-10` at `1e-13` tolerances.
- Kepler final states and the STM match the independent analytic propagator;
  parameter columns match central finite differences.
- A CR3BP close approach preserves its Jacobi constant and reverses within
  scale-aware tolerances.
- Rejected steps, exhausted step limits, backward integration, dense samples,
  terminal roots, non-finite values, and bit-for-bit repeatability are tested.
- DOP853 is not a Taylor method. Equal numeric tolerance values do not imply
  equal internal steps or bitwise parity with heyoka.
- Nominal integration is about three times slower than warmed C++/heyoka in
  this case. The variational path is approximately equal. Phase 18 will
  profile before changing the solver.
- Dense interpolation error depends on internal step length as well as local
  solve tolerance. High-accuracy dense/event callers should set
  `maximum_step`; ZOH switches always use exact final-time solves.
- Close singularities may exhaust rejections or step size. Errors retain the
  model and integration context instead of returning a plausible state.

## Consequences

The backend supports every remaining `ta` and generic ZOH-leg requirement
without C++, LLVM, a JIT, global mutable caches, or a hidden thread pool.
Physical Kepler, CR3BP, BCP, ZOH, and Pontryagin models remain Phase 11–13
work; completing this gate does not claim those APIs are implemented.

The dependency is deliberately hidden so it can be upgraded or replaced
without changing model, problem, result, or Python contracts.
The transitive `paste` maintenance advisory must be reconsidered on every
backend upgrade; the allowlist must not be broadened to a vulnerability.
