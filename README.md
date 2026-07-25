# pykep-rust

This repository contains an independent native Rust port of the numerical C++
library in pykep version 3 (`kep3`), together with a thin Python API built with
PyO3.

The numerical foundations, microsecond-resolution epochs, anomaly
conversions, and Cartesian/classical/modified-equinoctial element conversions
are implemented. Orbit propagation, Lambert solutions, ephemerides, dynamics,
and low-thrust legs remain planned. Do not infer full pykep parity from the
current API.

## Layout

- `crates/pykep-core`: C/C++-free Rust numerical library.
- `crates/pykep-py`: thin PyO3 extension over `pykep-core`.
- `python/pykep_rust`: Python package and future type stubs.
- `examples`: runnable Rust examples, added as capabilities become available.
- `docs`: Rust-specific design and usage documentation.

The pinned upstream source and adaptation policy are recorded in
[UPSTREAM_NOTICE.md](UPSTREAM_NOTICE.md). The complete port checklist is in
[docs/source-map.md](docs/source-map.md), and the evidence policy is in
[docs/validation.md](docs/validation.md).

## Intended properties

- Native Rust algorithms with no C or C++ runtime dependency.
- One numerical implementation in `pykep-core`; Python wrappers contain no
  duplicate astrodynamics logic.
- Numerical parity checked against the current `kep3` C++ tests and generated
  reference vectors.
- Explicit units, shapes, branch ordering, error behavior, and tolerances.
- Rust and Python documentation and tests developed with each module.

## Current smoke checks

From this directory:

```bash
cargo test --workspace
cargo run -p pykep-examples --bin port-status
cargo run -p pykep-examples --bin foundations
cargo run -p pykep-examples --bin epoch-anomalies
cargo run -p pykep-examples --bin elements
```

Once Maturin is installed, the Python module can be built in a virtual
environment with:

```bash
python -m pip install -e .
python -c "import pykep_rust as pk; print(pk.Epoch.from_iso('2030-01'))"
```

## Licensing

The upstream source and this adaptation are licensed under the Mozilla Public
License 2.0. See [LICENSE](LICENSE) and
[UPSTREAM_NOTICE.md](UPSTREAM_NOTICE.md).
