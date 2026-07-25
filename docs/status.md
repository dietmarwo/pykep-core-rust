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
| JPL low-precision ephemerides | implemented | implemented | 3.0.1 | names, window boundaries, ordered batches | Criterion harness | complete |
| VSOP2013 ephemerides | implemented (`>=1e-9` feature) | implemented | 3.0.1/heyoka 7.10.0 | expanded epoch grid, threshold selection, feature-off build | Criterion + C++ harness | complete |
| Adaptive integration backend | implemented | intentionally deferred to model APIs | analytic/C++ orientation | drift, reversal, rejection, dense/event, sensitivities | Criterion + candidate + C++ harness | complete |
| Kepler/CR3BP/BCP dynamics | implemented | implemented | 3.0.1/heyoka 7.10.0 | equilibria, invariants, finite differences, singularities | Criterion + C++ harness | complete |
| ZOH dynamics | implemented | implemented | 3.0.1/heyoka 7.10.0 | switches, reversal, zero control, sensitivity activation | Criterion + C++ harness | complete |
| Pontryagin dynamics | implemented | implemented | 3.0.1/heyoka 7.10.0 | Hamiltonians, coordinate transform, finite differences, singular primer | Criterion + C++ harness | complete |
| Sims–Flanagan legs | implemented | implemented | 3.0.1 | cuts, odd/even and one-segment cases, central differences, validation | Criterion + C++ harness | complete |
| Generic ZOH leg | implemented | implemented | 3.0.1/heyoka 7.10.0 | four models, cuts, central differences, contextual failures | Criterion + C++ harness | complete |

“Implemented” means the public contract is documented, validation is explicit,
the committed C++ golden data passes except for documented numerical
improvements, independent properties pass, and Rust/Python tests call the same
core implementation. It does not imply that later modules exist.
