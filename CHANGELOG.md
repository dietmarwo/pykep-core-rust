# Changelog

All notable changes to this project will be documented in this file. The
format follows Keep a Changelog, and releases will use semantic versioning
once the public API is ready to stabilize.

## [Unreleased]

### Added

- Independent Rust workspace and private PyO3 extension scaffold.
- Pinned pykep/kep3 3.0.1 provenance, full MPL-2.0 license, and deterministic
  C++ foundation oracle data.
- Cross-platform formatting, lint, test, documentation, coverage, packaging,
  dependency-license, and vulnerability CI gates.
- Physical constants, Julian-date conversions, numerically stable Stumpff
  functions, Kepler-equation residuals/derivatives, and fixed small linear
  algebra in Rust and Python.
- Explicit numerical error taxonomy, fixed-shape type decision, golden C++
  parity tests, independent derivative/geometry tests, and foundation
  benchmarks.
- Immutable microsecond-resolution epochs with MJD2000/MJD/JD and cropped ISO
  conversion, checked duration arithmetic, rich Python comparison, and
  explicit time-scale limitations.
- Elliptic and hyperbolic anomaly conversions, bounded solvers, scalar and
  batch Python APIs, seeded C++ parity vectors, round-trip tests, and
  benchmarks.
- Cartesian, classical Keplerian, and prograde/retrograde modified
  equinoctial conversions with named Rust element values, analytic
  output-by-input Jacobians, NumPy batch APIs, and explicit singularity
  handling.
- Forward/backward two-body propagation using Lagrange coefficients and
  universal variables, analytic Lagrangian and Reynolds STMs, time-grid and
  GIL-releasing NumPy batch APIs, golden parity vectors, invariant and
  finite-difference validation, and Criterion benchmarks.
- Hohmann and bi-elliptic transfers, reversible alpha/eta encodings, flyby
  constraints and analytic Jacobian, unpowered flyby mapping, deterministic
  single/multi-revolution Lambert branches, and both MIMA approximations in
  Rust and Python.
- Object-safe, thread-safe ephemeris providers with explicit optional
  capabilities and metadata, derived period/elements behavior, a Keplerian
  provider, shared Python `Planet`, and GIL-releasing epoch batches.
- JPL low-precision heliocentric ephemerides for Mercury through Neptune,
  traceable coefficient data, explicit 1800–2050 validity checks, all element
  representations, configurable safe radii, and scalar/batch Python access.
- Feature-gated pure-Rust VSOP2013 ephemerides for all nine upstream bodies,
  embedded MPL-2.0 coefficient data down to `1e-9`, direct ICRF state
  evaluation without JIT or network access, and explicit Python availability
  and threshold reporting.
- A pykep-owned adaptive DOP853 integration facade with fixed-size evaluated
  model contracts, exact final-time and backward propagation, dense output,
  terminal events, step diagnostics, and allocation-free first-order
  sensitivity states, plus a reproducible candidate decision benchmark.
- Stateless evaluated Kepler, CR3BP, and bicircular dynamics with explicit
  rotating-frame conventions, CR3BP effective potential and Jacobi constant,
  analytic state/parameter Jacobians, adaptive propagation and STMs, Python
  wrappers, C++ Taylor-reference trajectories, and Criterion benchmarks.
- A validated zero-order-hold control schedule plus Kepler, CR3BP,
  modified-equinoctial, and ideal solar-sail models with deterministic
  switches, backward propagation, fixed-width segment sensitivities, Python
  wrappers, Taylor-reference variations, and scaling benchmarks.
- Cartesian and modified-equinoctial Pontryagin state/costate dynamics for
  mass- and time-optimal control, including controls, Hamiltonians,
  propagation, sensitivities, Python APIs, parity vectors, and benchmarks.
- Fixed- and variable-duration Sims–Flanagan legs with immutable validation,
  mismatch/throttle constraints, analytic gradients, Python classes, and
  C++/independent derivative validation.
- A generic ZOH leg over all four controlled dynamics families, with
  endpoint/control/time-grid sensitivities, state histories, batch evaluation,
  contextual integration failures, and Python bindings.
- A complete typed Python surface audit, including all-export Pytest coverage,
  adversarial NumPy/lifetime/thread tests, explicit GIL-releasing batches,
  migration matrix, clean-wheel CI, Mypy example, and wrapper benchmarks.
- Paired deterministic Rust/Python examples for every major public module,
  plus quick starts and a batch-throughput example.
- Release-candidate packaging, performance regression tooling, bounded fuzz
  targets, memory/cache profiling procedure, and local empty-project smoke
  tests for the crate and wheel.
- A Reynolds-STM overflow fuzz target, reusable Markdown link checker, and
  executable Rustdoc examples on every major module landing page.

### Changed

- Enabled independent packaging of `pykep-core`; the PyO3 implementation crate
  remains unpublished and the Python distribution remains a separate wheel.
- ZOH state histories use one dense-output solve per segment. Backward
  histories use an equivalent increasing-time coordinate to avoid the
  selected backend's decreasing-time interpolation boundary defect.
- Python stub audits now compare runtime parameter names, order, defaults,
  return-annotation presence, and documented class members.

### Fixed

- Reynolds state-transition matrices now return a typed overflow error instead
  of panicking when finite inputs overflow intermediate products.
- `alpha_to_direct` rejects both zero and one with the documented invalid-input
  error class.
- Extended signed-year ISO strings emitted by `Epoch` now parse back to the
  same epoch.
- Lambert solution properties now expose runtime docstrings, and anomaly
  wrapper parameter names match the shipped stub.

### Known release blockers

- Permanent repository ownership, private security contact, trusted publisher
  configuration, external API review, registry publication, registry-download
  smoke tests, and the release tag require release-owner action.
