# Rust examples

All examples are deterministic, perform no network or file I/O, and run
against `pykep-core` directly:

```bash
cargo run --release -p pykep-examples --bin epoch-anomalies
cargo run --release -p pykep-examples --bin elements
cargo run --release -p pykep-examples --bin propagation
cargo run --release -p pykep-examples --bin lambert
cargo run --release -p pykep-examples --bin ephemeris-comparison
cargo run --release -p pykep-examples --bin gravity-assist
cargo run --release -p pykep-examples --bin low-thrust-legs
cargo run --release -p pykep-examples --bin dynamics
```

Every source file states its units, expected result, runtime expectation, and
feature set. `ephemeris-comparison` uses the default `vsop2013` feature; the
other numerical examples work without optional data. `foundations`,
`jpl-low-precision`, `keplerian-ephemeris`, and `port-status` are smaller
diagnostic examples retained from earlier phases.

Equivalent installed-wheel scripts are under `python/examples`. See
[`docs/examples.md`](../docs/examples.md) for the complete example matrix and
quick starts.
