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

These are not cross-language speed claims. CPU frequency was not fixed and
the run is not a substitute for distributions collected under controlled
affinity and power settings. The Julian arithmetic result is small enough that
compiler optimization and timer resolution dominate its interpretation.

Run the maintained harness with:

```bash
cargo bench -p pykep-core --bench foundation
cargo bench -p pykep-core --bench elements
```

The same harness includes scalar elliptic/hyperbolic anomaly solvers and a
64-value batch-equivalent loop. This keeps branch-heavy iterative work
separate from the foundation arithmetic measurements.
Element benchmarks separate scalar classical/equinoctial conversion, analytic
Jacobian evaluation, and a 64-state batch-equivalent loop.

C++ comparisons are added only when both sides execute identical input data,
validation policy, branch families, tolerances, and output work. Initialization
and batch throughput are reported separately from warm scalar latency.
