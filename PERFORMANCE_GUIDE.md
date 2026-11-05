# Choosing between the original and Rust fcmaes/pykep stacks

The 0.1.5 comparison uses common inputs, PyKEP 3.1.0 and achieved
accuracy gates. Warmed heyoka retains an advantage on several scalar
Taylor models and on nominal ZOH mismatch. Rust leads on selected
equinoctial and full-leg derivative workloads. Use the row for the required model, accuracy and timing boundary.
[Current matched measurements](docs/matched-benchmarks.md)

The tables on this page record the released 0.1.4 baseline. The intermediate
[Taylor performance follow-up](docs/taylor-performance-followup.md) compares
the earlier development rounds with that baseline.

## The two stacks

| | Original stack | Rust stack |
|---|---|---|
| Optimizer | Python/C++ `fcmaes` | `fcmaes-core`, optionally through `fcmaes-rust` |
| Astrodynamics | Python/C++ `pykep`/`kep3` | `pykep-core`, optionally through `pykep-rust` |
| Taylor integration | heyoka symbolic decomposition and LLVM JIT | Ahead-of-time Rust plus fixed-system coefficient kernels |
| Native dependencies | C++, heyoka, LLVM, and their transitive build/runtime requirements | Rust only for the numerical cores |
| Python in a native hot path | Normally crosses a Python/native boundary unless the complete objective is native | None when the objective uses the Rust crates directly |
| First use | JIT and cache construction for heyoka systems | No JIT setup |
| Best steady state | Reuse one compiled heyoka integrator per worker | Reuse native objective data and parallelize without Python |

## Historical measurement boundary

The Taylor comparison was run on 2026-07-28 on one logical core of an AMD
Ryzen 9 9950X under Linux 6.8, using release builds, Rust 1.97.1, upstream
`kep3` 3.0.1, and heyoka 7.10.0. Each reported time is a median from the
fixed benchmark protocol.

The original-pykep measurements call C++/heyoka directly. They do not include
Python, NumPy conversion, optimizer callbacks, or process communication.
The warmed result resets and reuses one already JIT-compiled integrator, which
is the best steady-state way to use heyoka. The Rust result is a stateless
ahead-of-time-compiled propagation.

Both implementations receive the same numeric tolerance and their final
states are checked against each other. Their adaptive controllers interpret
tolerance differently, however, so equal tolerance is not a rigorous
matched-achieved-error experiment. Treat these numbers as measured guidance
for the stated fixtures, not portable performance guarantees.

The optimized Rust timings below are the coefficient-engine changes shipped
in `pykep-core` 0.1.4. The complete reproducible harness and raw data are in
the development source tree described by
[`docs/taylor-integration.md`](docs/taylor-integration.md).

## Warmed Taylor propagation in 0.1.4

The five systems measured at `1e-12` in the 0.1.4 release all favored
original PyKEP on that host.

| System | `pykep-core` 0.1.4 | Warmed `kep3`/heyoka | Original-pykep advantage |
|---|---:|---:|---:|
| Kepler | 2.205 µs | 0.780 µs | 2.83× |
| CR3BP | 21.515 µs | 6.582 µs | 3.27× |
| Bicircular problem | 16.257 µs | 3.977 µs | 4.09× |
| ZOH Kepler | 1.195 µs | 0.327 µs | 3.66× |
| Cartesian mass-optimal | 48.302 µs | 10.663 µs | 4.53× |

The 0.1.4 Rust release evaluated most coefficients through a fixed
expression tape. Heyoka compiled a concrete model to a specialized
straight-line kernel. The development Rust tree generates selected
fixed-model coefficient kernels ahead of time.

### Low-thrust tolerance detail

The two directly compared low-thrust fixtures remain in the same range at
`1e-14`:

| System | Tolerance | `pykep-core` 0.1.4 | Warmed `kep3`/heyoka | Original-pykep advantage |
|---|---:|---:|---:|---:|
| ZOH Kepler | `1e-12` | 1.195 µs | 0.327 µs | 3.66× |
| ZOH Kepler | `1e-14` | 1.510 µs | 0.470 µs | 3.21× |
| Cartesian mass-optimal | `1e-12` | 48.302 µs | 10.663 µs | 4.53× |
| Cartesian mass-optimal | `1e-14` | 70.396 µs | 15.598 µs | 4.51× |

For the GTOC1-relevant ZOH fixture at `1e-12`, this corresponds to roughly
0.84 million Rust propagations per second and 3.06 million warmed-heyoka
propagations per second on the benchmark core.

### JIT setup is intentionally excluded above

The warmed comparison excludes the following first cache-miss costs:

| System | heyoka setup at `1e-12` | heyoka setup at `1e-14` |
|---|---:|---:|
| Kepler | 49.7 ms | 72.3 ms |
| CR3BP | 104.9 ms | 150.4 ms |
| Bicircular problem | 191.4 ms | 262.5 ms |
| ZOH Kepler | 86.9 ms | 119.3 ms |
| Cartesian mass-optimal | 450.4 ms | 741.5 ms |

The setup cost matters for short command-line programs, server workers that
start frequently, or applications that construct many system/tolerance
combinations. It becomes negligible in a long optimization campaign that
reuses one compiled integrator per worker.

## Coverage beyond the 0.1.4 comparison

The [current matched measurements](docs/matched-benchmarks.md) cover all
eleven scalar models, including both equinoctial Pontryagin objectives.
The current and historical measurements use different fixtures and hosts.

## Analytical and transcription kernels

The warmed-heyoka advantage does not automatically extend to conventional
numerical algorithms:

| Operation | Rust | Original C++ | Interpretation |
|---|---:|---:|---|
| Elliptic Lagrange propagation | 146.24 ns | 149.263 ns | Essentially equal |
| Hyperbolic Lagrange propagation | 567.50 ns | 450.166 ns | Original 1.26× faster |
| Universal-variable elliptic propagation | 250.05 ns | 260.613 ns | Rust slightly faster |
| Lagrange propagation plus STM | 216.68 ns | 444.289 ns | Rust about 2.05× faster |
| Five-segment Sims–Flanagan mismatch | 1.002 µs | 1.037 µs | Essentially equal |
| Five-segment Sims–Flanagan analytic Jacobian | 4.241 µs | 12.748 µs | Rust about 3.00× faster |

These were same-input release-mode orientation measurements on the same
development host. Criterion medians were used for Rust and elapsed averages
for the pinned C++ oracle, so small differences should not be overinterpreted.

The established 20-segment ZOH-leg comparison uses the compatibility-default
DOP853 Rust path and the upstream heyoka leg:

| Operation | Rust | Original pykep | Original-pykep advantage |
|---|---:|---:|---:|
| ZOH mismatch | 23.67 µs | 6.865 µs | 3.45× |
| Complete endpoint/control/time-grid Jacobian | 493.55 µs | 182.17 µs | 2.71× |

The 0.1.5 benchmark runs the shared 20-, 40- and 80-segment PyKEP
workloads with native Taylor mismatch and full Jacobian calls. At the
tighter reference setting the complete Jacobian favored Rust by
1.93–2.22×, while nominal mismatch favored heyoka by 1.23–1.54×. Its report
records the nominal and variational tolerances for both arms.
[Current leg comparisons](docs/matched-benchmarks.md)

## Optimizer evidence is separate

The astrodynamics tables do not compare the optimizer implementations.
Optimizer quality must be compared with the same objective, bounds, evaluation
budget, worker count, and stopping rule.

The existing native GTOP study gives useful compatibility evidence rather
than a universal speed ranking. Under its reported campaigns,
`fcmaes-rust` coordinated DE-to-CMA retry reached the difficult Tandem target
in 85 of 100 experiments; the historical original Python/C++ fcmaes table
reports a similar 81 of 100. The execution models are not identical, so this
shows comparable search capability, not a four-percentage-point performance
win. See the
[fcmaes-rust optimizer comparison](https://github.com/dietmarwo/fcmaes-rust/tree/main/benchmarks/optimizer-comparison)
for the budgets and raw results.

Python objective calls also matter independently of optimizer quality.
`fcmaes-rust` releases the GIL while its optimizer runs, but must reacquire it
for every Python objective callback. A native Rust objective avoids that
boundary and can be evaluated directly by retry workers. Conversely, an
original-pykep objective implemented and retained entirely in C++ can also
avoid most Python overhead.

## Why `pykep-core` remains pure Rust

License compatibility is not the blocker. Upstream pykep, heyoka, and
`pykep-core` use the Mozilla Public License 2.0. Correct notices and provenance
would be required, but heyoka can legally be used.

The decision is instead about the distributed product:

1. **One Cargo-native numerical core.** `pykep-core` builds without a C++
   compiler, C++ ABI, Boost, LLVM development package, or foreign build
   system.
2. **Reliable crates and wheels.** Linux, Windows, and macOS packages do not
   need to locate, link, or bundle a compatible LLVM/heyoka installation.
3. **No JIT lifecycle.** There is no runtime compiler, tolerance-keyed JIT
   cache, executable-memory policy, or per-worker compilation phase.
4. **Predictable provenance and testing.** The Rust implementation is tested
   directly against pinned upstream fixtures while released crates contain no
   upstream binary.
5. **Native composition.** Astrodynamics and optimization share Rust data
   structures and error handling, with no Python or C++ transition inside a
   native objective.
6. **Fixed scope.** `pykep-core` supports eleven known dynamics systems;
   heyoka's ability to JIT arbitrary symbolic ODEs is valuable generality but
   is not required for this closed set.
7. **Safety and maintenance.** Rust ownership and type checking remove many
   memory and FFI failure modes. Avoiding a C++ boundary also avoids ABI and
   exception-translation problems.

There was no maintained native heyoka Rust crate when this architecture was
chosen. Calling the original implementation would therefore require a C++
wrapper and a substantial native dependency stack.

### Could Rust reproduce heyoka's LLVM approach?

Yes, technically. A Rust implementation could construct the symbolic
expression graph, decompose and simplify it, emit LLVM intermediate
representation, and invoke LLVM's ORC JIT through the stable C API. Rust
bindings such as `llvm-sys` or a higher-level wrapper can expose those entry
points. The generated propagation kernel could then have the same essential
specialization opportunity as heyoka's generated C++ path.

That would be a Rust-hosted LLVM JIT, not a pure-Rust deployment. Users and
wheel builders would still need a compatible native LLVM library. Reaching
heyoka-like capability would also require much more than emitting arithmetic:

- a symbolic expression system with canonicalization and common-subexpression
  elimination;
- Taylor decomposition and efficient coefficient code generation;
- LLVM IR generation, optimization, linking, and ORC JIT lifecycle handling;
- code and cache specialization by model, numeric type, tolerance, and target
  CPU;
- adaptive-order and step-size control plus dense output;
- state, parameter, variational, event, batch, and SIMD interfaces;
- thread-safe caches, diagnostics, serialization, and cross-platform tests.

Implementing that stack is possible, but it would be a substantially larger
project than the fixed-system Taylor backend in `pykep-core`. Binding the
existing C++ heyoka library would reduce algorithm-development work but retain
the C++ ABI and native packaging costs.

The development Rust backend already generates straight-line
coefficients for selected fixed-model paths. It also shares nominal
coefficients across practical seed widths. The complete-leg comparisons
measure the resulting effect on segment derivatives, assembly and
call overhead. [Current matched measurements](docs/matched-benchmarks.md)

## Decision guide

Prefer original `fcmaes` plus `pykep` when:

- an existing application already depends on the original Python APIs;
- strict behavioral compatibility with upstream pykep is more important than
  a C/C++-free deployment;
- the measured warmed nominal kernel for the required dynamics leads at
  the application's accuracy, and each worker can reuse its integrator;
- arbitrary symbolic ODE construction or other heyoka features are required;
- the native C++/LLVM dependency stack is acceptable.

Prefer `fcmaes-core` plus `pykep-core` when:

- the complete objective can be written in Rust;
- retry, population, or batch evaluations must run without Python callbacks
  or the GIL;
- simple Cargo dependency management and cross-platform binary deployment are
  priorities;
- startup latency, deterministic packaging, or short-lived workers matter;
- analytical propagation, Lambert, flyby, ephemeris, and optimization kernels
  dominate rather than warmed Taylor integration alone;
- one memory-safe native implementation should serve both Rust and Python.

Consider a staged or hybrid workflow when:

- a cheap Rust Lambert/flyby model must screen millions of planet-order or
  timing candidates;
- only a small promoted set requires expensive low-thrust optimization;
- the matched leg or objective results favor one backend for that final stage;
- Rust DOP853 or an independent tool should validate the finalists.

## GTOC1 recommendation

For the current GTOC1 tutorial, the evidence supports a split decision:

- the native Rust stack remains attractive for planet-order experiments,
  cheap Lambert/flyby scoring, coordinated retry, and parallel objective
  evaluation;
- use the warmed nominal and complete-Jacobian rows for the actual
  low-thrust workload when selecting a backend;
- the complete GTOC1 campaign has compatibility checks. The benchmark
  report gives optimizer comparisons for the separate gym and SF workloads;
- every claimed trajectory still needs independent propagation and constraint
  validation, regardless of which stack optimized it.

Before choosing a stack for a production campaign, benchmark the exact
objective with the same trajectory representation, tolerance, segment count,
evaluation budget, worker count, CPU affinity, and validation criteria. Report
JIT setup and warmed throughput separately.
