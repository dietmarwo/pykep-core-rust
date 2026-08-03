# Performance methodology

Benchmarks use release mode and Criterion. They contain no file or console I/O
inside the measured loop. Raw Criterion state is build output under `target/`
and is not committed.

An orientation run on 2026-07-25 used Rust 1.97.1 on an AMD Ryzen 9 9950X
under Linux 6.8.0-136. One Criterion process reported:

| Foundation workload | Median estimate |
|---|---:|
| `stumpff_c(1e-12)` | 1.709 ns |
| `stumpff_s(-4)` | 9.437 ns |
| `jd_to_mjd2000` | 0.240 ns |
| three-vector cross product | 4.794 ns |
| elliptic mean → eccentric, `e = 0.999` | 89.51 ns |
| hyperbolic mean → anomaly, `e = 1.5` | 118.3 ns |
| 64 elliptic conversions, `e = 0.9` | 5.857 µs |
| classical → Cartesian | 40.25 ns |
| Cartesian → modified equinoctial | 29.33 ns |
| Cartesian → equinoctial Jacobian | 151.7 ns |
| 64 classical → Cartesian conversions | 2.575 µs |

The Phase 5 orientation run used identical scalar inputs in Rust and the
pinned C++ oracle, with both built in release mode on the same machine:

| Propagation workload | Rust Criterion median | C++ elapsed average |
|---|---:|---:|
| Lagrange, elliptic | 146.24 ns | 149.263 ns |
| Lagrange, hyperbolic | 567.50 ns | 450.166 ns |
| Universal variables, elliptic | 250.05 ns | 260.613 ns |
| Lagrange propagation + STM | 216.68 ns | 444.289 ns |
| 1,024 Lagrange calls | 113.78 µs (9.00 million/s) | — |

## Mission-design kernels

Phase 6 measured each operation separately in the same Rust orientation run:

| Workload | Rust median |
|---|---:|
| Hohmann transfer | 6.258 ns |
| Flyby constraints | 10.923 ns |
| Flyby delta-v | 34.567 ns |
| Zero-revolution Lambert problem | 235.76 ns |
| Seven-solution multi-revolution Lambert problem | 1.263 µs |

## Ephemerides

Scalar and ordered-batch measurements were kept separate:

| Phase | Provider and workload | Result |
|---|---|---:|
| 7 | Keplerian, one epoch | 80.299 ns |
| 7 | Keplerian, 256 ordered epochs | 22.575 µs |
| 8 | JPL low-precision Earth, one epoch | 95.198 ns |
| 8 | JPL low-precision Earth, 256 ordered epochs | 31.611 µs |
| 9 | VSOP2013 default-threshold initialization | 11.615 µs |
| 9 | VSOP2013 default-threshold scalar state | 339.98 ns |
| 9 | VSOP2013 default-threshold 256-state batch | 89.554 µs |
| 9 | VSOP2013 `1e-9` scalar state | 37.157 µs |

The matching C++/heyoka VSOP2013 harness measured first-time JIT costs of
63.96 ms at `1e-5` and 552.53 ms at `1e-9`. Warm scalar calls took 166 ns and
5.908 µs, respectively. Enabling VSOP2013 increased the Rust release benchmark
executable from 2.6 MiB to 7.0 MiB. ADR 0003 explains the data and cache
decision.

## Integration and dynamics

### DOP853 selection

The Phase 10 decision harness produced these orientation measurements:

| Workload | Selected Rust facade | Warmed C++/heyoka |
|---|---:|---:|
| Representative nominal Kepler solve | 11.865 µs | 3.722 µs |
| State plus 6 by 6 STM | 85.663 µs | 84.440 µs |

Cloning the cached C++ nominal integrator once cost 0.467 ms. In a same-profile
candidate test, the selected facade took 15.431 µs and `ode_solvers` took
7.369 µs. The latter lacked the required root and sensitivity facilities and
allocated per step, so nominal timing alone did not decide the dependency.
ADR 0004 records the configuration, ranges, and risks.

### Taylor integration

The fixed-system Taylor backend uses a matched-accuracy protocol because
coefficient sweeps are not comparable to DOP853 right-hand-side calls. For one
eccentric nondimensional revolution, Taylor's speed relative to DOP853 was:

| Tolerance | Taylor/DOP853 speed |
|---|---:|
| `1e-9` | 0.87× |
| `1e-12` | 1.18× |
| `1e-14` | 1.61× |
| Machine epsilon | 2.16× |

Taylor produced smaller final-state errors at every point. This is deliberately
a narrow conclusion: Taylor is a high-accuracy option, not a low-accuracy
replacement. The committed CSV also contains the 100- and 1,000-revolution
rows; see [High-accuracy Taylor integration](taylor-integration.md).

All eleven built-in Taylor models now use incremental coefficient evaluators.
Across the representative eight-model migration benchmark, replacing repeated
full-series evaluation reduced warmed end-to-end propagation time by 6.45× to
28.97×. The linked Taylor guide gives per-model results and validation limits.

### Evaluated models and controlled dynamics

| Phase | Workload | Rust | Warmed C++/heyoka |
|---|---|---:|---:|
| 11 | Kepler RHS | 10.828 ns | — |
| 11 | CR3BP RHS | 32.833 ns | — |
| 11 | BCP RHS | 63.386 ns | — |
| 11 | CR3BP propagation | 7.808 µs | 2.885 µs |
| 11 | CR3BP state plus STM | 45.368 µs | 95.017 µs |
| 12 | ZOH Kepler RHS | 8.928 ns | — |
| 12 | 32-segment alternating-control schedule | 36.089 µs | 10.268 µs |
| 13 | Cartesian mass-optimal RHS | 52.384 ns | — |
| 13 | Cartesian mass-optimal propagation | 142.52 µs | 11.716 µs |

The Phase 11 comparisons use the same initial state, final time, parameter,
and `1e-12` tolerance.

The Phase 12 Rust schedule deliberately starts an independent DOP853 solve at
each switch. Its timing includes all 32 restarts and confirms that integration
does not perform a segment-count-dependent control search.

The Phase 13 propagation covers 1.2345 normalized time units. Rust uses a
`0.01` maximum step to meet the recorded oracle tolerance, whereas the C++
Taylor solve has no equivalent limit. Treat this as an identified performance
target, not as a like-for-like algorithm comparison.

## Low-thrust legs

| Phase | Workload | Rust | Warmed C++/heyoka |
|---|---|---:|---:|
| 14 | Five-segment Sims–Flanagan mismatch | 1.002 µs | 1.037 µs |
| 14 | Complete analytic mismatch Jacobian | 4.241 µs | 12.748 µs |
| 15 | 20-segment normalized Kepler ZOH mismatch | 23.67 µs | 6.865 µs |
| 15 | Complete endpoint/control/time-grid Jacobian | 493.55 µs | 182.17 µs |

The Phase 14 comparison uses the same endpoints, masses, controls, duration,
propulsion parameters, gravity parameter, and `cut = 0.6`. Construction and
validation are outside both timings.

The Phase 15 comparison uses the same states, chronological controls, time
grid, constants, cut, and `1e-12` tolerance, with no maximum step. Rust
integrates every segment independently and uses fixed-size numerical dynamics
Jacobians. Both operations therefore remain visible optimization targets.

## Python batch throughput

The Phase 16 release-wheel harness used 20,000 items and the median of nine
samples:

| Workload | Python scalar loop | NumPy batch | Batch improvement |
|---|---:|---:|---:|
| `stumpff_c` | 38.9 ns/item | 14.6 ns/item | 2.67× |
| Lagrange propagation | 0.72 µs/item | 0.09 µs/item | 7.82× |

These measurements include Python input/output conversion; they are not
Rust-core timings. They show why throughput-sensitive Python code should use
the explicit batch APIs.

## Interpreting the results

These are not cross-language speed claims. CPU frequency was not fixed and
the run is not a substitute for distributions collected under controlled
affinity and power settings. The Julian arithmetic result is small enough that
compiler optimization and timer resolution dominate its interpretation.
The C++ column is an elapsed average from five million calls rather than a
Criterion distribution; it is included as the required same-input orientation
baseline, not as a statistically controlled language comparison.

## Reproducing the measurements

Run the maintained harness with:

```bash
cargo bench -p pykep-core --bench foundation
cargo bench -p pykep-core --bench elements
cargo bench -p pykep-core --bench propagation
cargo bench -p pykep-core --bench mission
cargo bench -p pykep-core --bench integration
cargo bench -p pykep-core --bench dynamics
cargo bench -p pykep-core --bench legs
python python/benchmarks/wrapper_overhead.py
cargo run --release -p pykep-lambert-optimization-benchmark
cargo run --release -p pykep-taylor-benchmark
```

Each benchmark group has a distinct scope:

| Group | Workloads |
|---|---|
| Foundation | Arithmetic kernels plus scalar elliptic/hyperbolic anomaly solvers and a 64-value loop |
| Elements | Scalar classical/equinoctial conversions, analytic Jacobians, and a 64-state loop |
| Propagation | Elliptic/hyperbolic Lagrange coefficients, universal variables, analytic STMs, and a 1,024-state loop |
| Mission | Transfers, flyby constraints/delta-v, Lambert branches, and scalar/batch ephemerides |
| Integration | Nominal six-state DOP853 and Taylor propagation plus STM paths |
| Dynamics | Evaluated right-hand sides, CR3BP nominal/variational propagation, ZOH schedules, and Pontryagin propagation |
| Legs | Sims–Flanagan mismatch/analytic gradients and generic ZOH mismatch/sensitivities |

The foundation anomaly loop keeps branch-heavy iterative work separate from
arithmetic kernels. Scalar propagation and STM APIs use fixed-size arrays and
perform no heap allocation. Ephemeris measurements distinguish initialization,
scalar, high-precision, and batch paths. The integration final-state callback
does not retain internal steps.

## Standalone Lambert optimization

The standalone Lambert optimization benchmark ports the fixed `easy.kttsp`
leg from `pykep-lambert`. It measures deterministic objective throughput and
then applies the native `fcmaes-core` CMA-ES and BiteOpt implementations to
wait time and time of flight. Its source revision, physical constants,
decision bounds, penalty, optimizer budget, and seed are printed with every
run; see `tools/lambert-optimization-benchmark/README.md` for the complete
protocol.

## Cross-language comparison policy

C++ comparisons are added only when both sides execute identical input data,
validation policy, branch families, tolerances, and output work. Initialization
and batch throughput are reported separately from warm scalar latency.

## Release stabilization evidence

Phase 18 adds a protocol-matched 100-sample Rust/C++ distribution, bootstrap
median confidence intervals, CI regression limits, allocation/cache/
vectorization profiles, and five-point Python batch scaling. Full results and
environment limitations are in [stabilization.md](stabilization.md).
