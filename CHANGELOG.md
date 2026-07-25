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

JPL/VSOP ephemerides, dynamics, and low-thrust APIs are not yet implemented.
