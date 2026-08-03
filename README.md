# pykep-rust

![Pure Rust astrodynamics core](https://img.shields.io/badge/astrodynamics%20core-100%25%20Rust-brightgreen)
![No C++ backend](https://img.shields.io/badge/C%2B%2B%20backend-none-brightgreen)
[![crates.io](https://img.shields.io/crates/v/pykep-core.svg?cacheSeconds=300)](https://crates.io/crates/pykep-core)
[![docs.rs](https://docs.rs/pykep-core/badge.svg)](https://docs.rs/pykep-core)
[![PyPI](https://img.shields.io/pypi/v/pykep_rust.svg?cacheSeconds=300)](https://pypi.org/project/pykep-rust/)
[![mdBook guide](https://img.shields.io/badge/guide-mdBook-blue)](https://dietmarwo.github.io/pykep-core-rust/)

This repository contains an independent native Rust port of the numerical C++
library in pykep version 3 (`kep3`), together with a thin Python API built with
PyO3.

The [narrative documentation](https://dietmarwo.github.io/pykep-core-rust/)
is configured to publish from `main`. crates.io releases generate the Rust API
reference on [docs.rs](https://docs.rs/pykep-core).
For a measurement-backed choice between the original Python/C++ fcmaes/pykep
stack and the native Rust crates, see
[PERFORMANCE_GUIDE.md](PERFORMANCE_GUIDE.md).

The numerical foundations, epochs, anomalies, element conversions, two-body
propagation, state-transition matrices, Lambert solutions, impulsive
transfers, flybys, encodings, MIMA approximations, the planet interface, and
the Keplerian, JPL low-precision, and feature-gated VSOP2013 ephemerides are
implemented. The pure-Rust adaptive integration backend now drives evaluated
Kepler, CR3BP, bicircular, four zero-order-hold dynamics families, and
Cartesian/equinoctial Pontryagin models with first-order sensitivities. An
adaptive Taylor backend is the default for nominal propagation of the
TA-derived built-in dynamics, matching upstream pykep's algorithm family;
the general-purpose DOP853 backend remains available for arbitrary models,
events, and direct variational solves.
Validated fixed- and variable-duration Sims–Flanagan legs provide mismatch
and throttle constraints, with analytic fixed-leg gradients. The generic ZOH
leg supports the four built-in controlled dynamics families, complete
first-order mismatch sensitivities, histories, and batches. Built-in Rust ZOH
legs can explicitly select Taylor or DOP853 for nominal mismatch and history
evaluation; compatibility methods, Jacobians, and the Python surface retain
DOP853. Do not infer full pykep parity from the current API.

## Layout

- `crates/pykep-core`: C/C++-free Rust numerical library.
- `crates/pykep-py`: thin PyO3 extension over `pykep-core`.
- `python/pykep_rust`: typed Python package and extension stub.
- `examples`: runnable deterministic Rust examples.
- `docs`: Rust-specific design and usage documentation.
- `PERFORMANCE_GUIDE.md`: warmed-kernel measurements, pure-Rust rationale,
  and original-versus-Rust stack selection guidance.
- `ai-context.md`: operational model-selection, convention, implementation,
  and validation guidance for AI-assisted user problems.
- `docs/add-ode-system.md`: definition of done for implementing, validating,
  benchmarking, documenting, and exposing another dynamics family.
- `SECURITY.md`: supported-release and private numerical-integrity reporting
  policy.
- `tools/release-benchmark`: fixed-protocol release regression benchmark.
- `tools/taylor-benchmark`: matched-accuracy DOP853/Taylor benchmark.
- `tools/lambert-optimization-benchmark`: native KTTSP Lambert objective and
  `fcmaes-core` optimization benchmark.

## End-to-end mission tutorial

The external
[GTOC1 “Save the Earth” Rust tutorial](https://github.com/dietmarwo/fcmaes-rust/tree/main/tutorials/gtoc1)
combines `pykep-core` ephemerides, Kepler propagation, multi-revolution
Lambert arcs, gravity assists, and Sims–Flanagan low-thrust legs with
`fcmaes-core` optimization. It develops a complete fixed-sequence trajectory,
then explains multi-fidelity planet-order search, numerical validation, and
the ephemeris limitations of the resulting model score.

The pinned upstream source and adaptation policy are recorded in
[UPSTREAM_NOTICE.md](UPSTREAM_NOTICE.md). The complete port checklist is in
[docs/source-map.md](docs/source-map.md), and the evidence policy is in
[docs/validation.md](docs/validation.md). Ephemeris frames, validity, and
accuracy are documented in [docs/ephemerides.md](docs/ephemerides.md).
Dynamics frames, parameters, errors, and reference tolerances are documented
in [docs/dynamics.md](docs/dynamics.md).
Taylor selection, supported models, measured crossover, and current
sensitivity limitations are documented in
[docs/taylor-integration.md](docs/taylor-integration.md).
Piecewise-constant controls, switching boundaries, and ZOH sensitivities are
documented in [docs/zero-order-hold.md](docs/zero-order-hold.md).
Indirect-control state, costate, control, and parameter conventions are
documented in [docs/pontryagin.md](docs/pontryagin.md).
Sims–Flanagan endpoint, cut, impulse, constraint, and Jacobian conventions are
documented in [docs/low-thrust-legs.md](docs/low-thrust-legs.md).
Generic continuous-thrust ZOH leg models, grids, sensitivities, and integration
settings are documented in [docs/zoh-leg.md](docs/zoh-leg.md).
The SpOC 4–motivated ordered parallel batch extension, worker semantics,
complete Python batch matrix, and nested-parallelism guidance are documented
in [docs/batch-processing.md](docs/batch-processing.md).
The typed Python contract and upstream migration matrix are documented in
[docs/python-api.md](docs/python-api.md) and
[docs/python-migration.md](docs/python-migration.md).
Rust and Python quick starts plus the complete runnable matrix are in
[docs/examples.md](docs/examples.md).
The historical pre-0.1.0 release-candidate performance distributions,
profiling, dynamic analysis, and since-resolved publication blockers are in
[docs/stabilization.md](docs/stabilization.md). The current publication and
clean-consumer procedure is in [RELEASE.md](RELEASE.md).

## Intended properties

- Native Rust algorithms with no C or C++ runtime dependency.
- One numerical implementation in `pykep-core`; Python wrappers contain no
  duplicate astrodynamics logic.
- Numerical parity checked against the pinned `kep3` C++ tests and generated
  reference vectors.
- Explicit units, shapes, branch ordering, error behavior, and tolerances.
- Ordered serial/parallel batches that preserve scalar numerical semantics.
- Rust and Python documentation and tests developed with each module.

## Current smoke checks

From this directory:

```bash
cargo test --workspace
cargo run -p pykep-examples --bin port-status
cargo run -p pykep-examples --bin foundations
cargo run -p pykep-examples --bin epoch-anomalies
cargo run -p pykep-examples --bin elements
cargo run -p pykep-examples --bin propagation
cargo run -p pykep-examples --bin lambert
cargo run -p pykep-examples --bin jpl-low-precision
cargo run -p pykep-examples --bin ephemeris-comparison
cargo run -p pykep-examples --bin gravity-assist
cargo run -p pykep-examples --bin low-thrust-legs
cargo run -p pykep-examples --bin dynamics
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
