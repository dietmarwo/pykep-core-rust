# Implementation status

Evidence-backed status as of 2026-08-03:

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
| Adaptive integration backends | DOP853 general; Taylor for 11 built-ins | DOP853 model APIs | analytic/C++/heyoka 7.10.1 | drift, reversal, dense, events (DOP853), seeded sensitivities, closed-form series | Criterion + fixed Taylor protocol | complete |
| Kepler/CR3BP/BCP dynamics | implemented | implemented | 3.0.1/heyoka 7.10.0 | equilibria, invariants, finite differences, singularities | Criterion + C++ harness | complete |
| ZOH dynamics | implemented | implemented | 3.0.1/heyoka 7.10.0 | switches, reversal, zero control, sensitivity activation | Criterion + C++ harness | complete |
| Pontryagin dynamics | implemented | implemented | 3.0.1/heyoka 7.10.0 | Hamiltonians, coordinate transform, finite differences, singular primer | Criterion + C++ harness | complete |
| Sims–Flanagan legs | implemented | implemented | 3.0.1 | cuts, odd/even and one-segment cases, central differences, validation | Criterion + C++ harness | complete |
| Generic ZOH leg | implemented | implemented | 3.0.1/heyoka 7.10.0 | four models, cuts, central differences, contextual failures | Criterion + C++ harness | complete |
| Ordered parallel batches | shared executor plus named core batches | listed numerical families | same scalar entry points | scalar parity, shapes, ordering, worker modes, error order | companion Lambert benchmark | complete |
| Python API audit | same native core | complete typed surface | same core entry points | exports, adversarial buffers, ownership, threads, clean wheels | wrapper/batch harness | complete |

“Implemented” means the public contract is documented, validation is explicit,
the committed C++ golden data passes except for documented numerical
improvements, independent properties pass, and Rust/Python tests call the same
core implementation. It does not imply that later modules exist.

The Python wheel uses the collision-safe `pykep_rust` import, ships a complete
stub and `py.typed`, and has no C++ runtime dependency. Clean-wheel CI covers
CPython 3.11–3.13 on Linux, macOS, and Windows. The upstream migration matrix
records renames, deliberate contract changes, deferrals, and unsupported
ecosystem modules.

The runnable example matrix covers every major public module in Rust and
through the installed Python extension. Each example states units, expected
behavior, runtime orientation, and required features; CI compiles all Rust
examples and executes all Python scripts.

The synchronized 0.1.4 release is published as `pykep-core` on crates.io and
`pykep-rust` on PyPI. The internal `pykep-py` implementation crate remains
`publish = false` by design. Tag-gated trusted-publishing workflows, clean
crate/wheel/source-distribution consumption, docs.rs, and GitHub Pages are part
of the release process. Performance regression, Miri, fuzz, Valgrind,
dependency, MSRV, cross-platform wheel, and API-documentation checks remain
maintained quality gates rather than claims of formal verification.

Version 0.1.4 includes incremental Taylor coefficient evaluation for all
eleven built-in models and Rust-only method selection for nominal ZOH-leg
mismatch/history calculations. Existing no-suffix ZOH-leg methods, Jacobians,
and the Python ZOH-leg surface retain their DOP853 behavior. See the
[performance decision guide](https://github.com/dietmarwo/pykep-core-rust/blob/main/PERFORMANCE_GUIDE.md)
for the measured boundary between the ahead-of-time Rust backend and warmed
upstream heyoka.
