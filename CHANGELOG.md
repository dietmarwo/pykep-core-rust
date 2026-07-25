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

Propagation, Lambert, ephemerides, dynamics, and low-thrust APIs are not yet
implemented.
