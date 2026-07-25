# Documentation

This directory will contain the Rust-facing documentation for pykep-rust.
Documentation will be added alongside implemented functionality rather than
copied from the Python wrapper.

Planned documents include:

- conventions for units, epochs, anomalies, coordinate systems, and array
  layouts;
- numerical behavior and error handling;
- Lambert and two-body propagation guides;
- planet and ephemeris provider interfaces;
- low-thrust leg and dynamics guides;
- Rust and Python API quick starts;
- performance methodology and C++ parity reports;
- migration notes for users familiar with `kep3` or `pykep`.

Current implementation coverage is tracked in [source-map.md](source-map.md).
Development and validation commands are in
[development.md](development.md) and [validation.md](validation.md).
