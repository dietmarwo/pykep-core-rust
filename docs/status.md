# Implementation status

Evidence-backed status as of 2026-07-25:

| Module | Rust core | Python API | Golden parity | Independent tests | Benchmarked | Docs |
|---|---|---|---|---|---|---|
| Foundations | implemented | implemented | 3.0.1 | series, derivatives, geometry | Criterion harness | complete |
| Epoch/anomalies | implemented | implemented | 3.0.1 | round trips, calendar boundaries | Criterion harness | complete |
| Elements | implemented | implemented | 3.0.1 | 2,000 round trips, finite differences | Criterion harness | complete |
| Propagation/STM | implemented | implemented | 3.0.1 | invariants, reversal, finite differences, composition | Criterion harness | complete |
| Lambert/transfers/flyby/MIMA | implemented | implemented | 3.0.1 | round trips, endpoint reconstruction, finite differences | Criterion harness | complete |
| Planet/Keplerian ephemeris | implemented | implemented | 3.0.1 | period, element round trips, thread stress | Criterion harness | complete |
| JPL/VSOP ephemerides | planned | planned | planned | planned | planned | planned |
| Dynamics | planned | planned | planned | planned | planned | planned |
| Low-thrust legs | planned | planned | planned | planned | planned | planned |

“Implemented” means the public contract is documented, validation is explicit,
the committed C++ golden data passes except for documented numerical
improvements, independent properties pass, and Rust/Python tests call the same
core implementation. It does not imply that later modules exist.
