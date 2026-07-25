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

Phase 6 Rust medians from the same orientation run were 6.258 ns for Hohmann,
10.923 ns for flyby constraints, 34.567 ns for flyby delta-v, 235.76 ns for a
zero-revolution Lambert problem, and 1.263 µs for the seven-solution
multi-revolution case.
Keplerian ephemeris evaluation measured 80.299 ns for one scalar epoch and
22.575 µs for an ordered 256-epoch batch in the Phase 7 orientation run.
JPL low-precision Earth evaluation measured 95.198 ns for one scalar epoch
and 31.611 µs for an ordered 256-epoch batch in the Phase 8 orientation run.
The Phase 9 VSOP2013 spike measured 11.615 µs default-threshold
initialization, 339.98 ns per default scalar state, 89.554 µs per ordered
256-state batch, and 37.157 µs per `1e-9` scalar state. The matching C++/heyoka
orientation harness measured 63.96 ms and 552.53 ms first-time JIT
initialization at `1e-5` and `1e-9`, then 166 ns and 5.908 µs per scalar call.
The feature increased the release benchmark executable from 2.6 MiB to
7.0 MiB. See ADR 0003 for the data/cache decision and caveats.
The Phase 10 integration decision harness measured the selected DOP853 facade
at 11.865 µs for the representative nominal Kepler solve and 85.663 µs for
the state plus 6 by 6 STM. The equivalent warmed C++/heyoka solves measured
3.722 µs and 84.440 µs; cloning the cached C++ nominal integrator once cost
0.467 ms. A same-profile candidate tool measured 15.431 µs for the selected
facade and 7.369 µs for `ode_solvers`, whose missing root/sensitivity
facilities and per-step allocations outweighed the nominal timing advantage.
See ADR 0004 for configuration, ranges, and risks.

These are not cross-language speed claims. CPU frequency was not fixed and
the run is not a substitute for distributions collected under controlled
affinity and power settings. The Julian arithmetic result is small enough that
compiler optimization and timer resolution dominate its interpretation.
The C++ column is an elapsed average from five million calls rather than a
Criterion distribution; it is included as the required same-input orientation
baseline, not as a statistically controlled language comparison.

Run the maintained harness with:

```bash
cargo bench -p pykep-core --bench foundation
cargo bench -p pykep-core --bench elements
cargo bench -p pykep-core --bench propagation
cargo bench -p pykep-core --bench mission
cargo bench -p pykep-core --bench integration
```

The same harness includes scalar elliptic/hyperbolic anomaly solvers and a
64-value batch-equivalent loop. This keeps branch-heavy iterative work
separate from the foundation arithmetic measurements.
Element benchmarks separate scalar classical/equinoctial conversion, analytic
Jacobian evaluation, and a 64-state batch-equivalent loop.
Propagation benchmarks separate elliptic and hyperbolic Lagrange coefficients,
universal variables, analytic STM evaluation, and a 1,024-state throughput
loop. Scalar propagation and STM APIs operate entirely on fixed-size arrays
and perform no heap allocation.
Mission benchmarks cover basic transfers, flyby constraints/delta-v, and both
zero- and multi-revolution Lambert solution construction. The same harness
measures scalar and 256-epoch Keplerian and JPL low-precision ephemeris
evaluation separately, plus VSOP2013 initialization, scalar, high-precision,
and batch paths.
Integration benchmarks separate nominal six-state DOP853 propagation from an
augmented state plus STM solve. The final-state output callback does not retain
internal steps.

C++ comparisons are added only when both sides execute identical input data,
validation policy, branch families, tolerances, and output work. Initialization
and batch throughput are reported separately from warm scalar latency.
