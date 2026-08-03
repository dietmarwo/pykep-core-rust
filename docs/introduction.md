# Native astrodynamics in Rust

`pykep-rust` is an independent native Rust port of the numerical C++ library
in pykep version 3 (`kep3`). The reusable `pykep-core` crate contains the
astrodynamics implementation without a C or C++ runtime dependency. The
optional `pykep-rust` Python distribution exposes the same implementation
through PyO3.

This book connects the narrative material for both interfaces:

- [examples and quick starts](examples.md) provide runnable Rust and Python
  entry points;
- [numerical conventions](conventions.md) define units, epochs, frames, array
  layouts, tolerances, and error behavior;
- the dynamics, ephemeris, propagation, and low-thrust guides explain
  algorithm-specific contracts;
- [validation](validation.md) records current parity and independent checks;
  the [stabilization evidence](stabilization.md) preserves the historical
  pre-0.1.0 coverage, performance, and release-candidate snapshot.

Exact Rust types, methods, and compiled examples are in the generated
[`pykep-core` API reference](https://docs.rs/pykep-core). Python users should
start with the [Python API contract](python-api.md) and
[migration matrix](python-migration.md).

This project is not an official ESA release. Its pinned upstream source and
MPL-2.0 adaptation policy are recorded in the
[GitHub repository](https://github.com/dietmarwo/pykep-core-rust).
