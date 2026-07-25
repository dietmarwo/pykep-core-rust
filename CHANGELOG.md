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

Pontryagin and low-thrust leg APIs are not yet implemented.
