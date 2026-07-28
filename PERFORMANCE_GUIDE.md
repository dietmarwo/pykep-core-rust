# Choosing between the original and Rust fcmaes/pykep stacks

The original Python/C++ stack and the native Rust stack make different
trade-offs. There is no universal winner:

- original `pykep`/`kep3` uses heyoka's LLVM-generated Taylor kernels and is
  currently faster for warmed propagation of the built-in Taylor systems;
- `pykep-core` and `fcmaes-core` provide a C/C++-free numerical and optimizer
  path with simple Cargo deployment, no runtime JIT, and no Python callback in
  a native objective;
- analytical astrodynamics kernels are generally much closer, and some Rust
  implementations are faster;
- complete optimization throughput depends on objective cost, callback
  overhead, parallelization, integrator reuse, and the number of evaluations.

This guide records the available measurements and explains when each stack is
the better engineering choice. It does not claim complete API parity or an
end-to-end GTOC1 speed result.

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

## Measurement boundary

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

## Warmed Taylor propagation

At `1e-12`, the five systems with direct current-Rust versus warmed-heyoka
measurements show a consistent original-pykep advantage:

| System | `pykep-core` 0.1.4 | Warmed `kep3`/heyoka | Original-pykep advantage |
|---|---:|---:|---:|
| Kepler | 2.205 µs | 0.780 µs | 2.83× |
| CR3BP | 21.515 µs | 6.582 µs | 3.27× |
| Bicircular problem | 16.257 µs | 3.977 µs | 4.09× |
| ZOH Kepler | 1.195 µs | 0.327 µs | 3.66× |
| Cartesian mass-optimal | 48.302 µs | 10.663 µs | 4.53× |

This is not a general “C++ beats Rust” result. Both compilers ultimately use
LLVM. Heyoka constructs the concrete symbolic system and JIT-compiles a
model-specific straight-line coefficient kernel. Most Rust models currently
execute a compact fixed expression tape: the graph is simplified once, but
runtime coefficient evaluation still performs indexed loads, convolution
loops, and operation dispatch. Rust's ahead-of-time LLVM pass cannot see the
runtime tape as one fixed expression graph.

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

## Taylor systems still lacking a direct comparison

All eleven Rust models have incremental coefficient evaluators, but the live
upstream harness has not yet been extended to six of them. Their current Rust
`1e-12` warmed times are:

| System | `pykep-core` 0.1.4 |
|---|---:|
| ZOH CR3BP | 6.791 µs |
| ZOH equinoctial | 6.979 µs |
| ZOH solar sail | 12.876 µs |
| Cartesian time-optimal | 4.285 µs |
| Equinoctial mass-optimal | 377.991 µs |
| Equinoctial time-optimal | 333.640 µs |

These rows must not be turned into an original-versus-Rust ratio until the
same fixture is run through warmed upstream heyoka. The especially large
equinoctial Pontryagin systems are the most important missing comparison.

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

`pykep-core` 0.1.4 additionally allows the nominal ZOH mismatch and history
paths to select Taylor explicitly. A current full-leg Taylor-versus-upstream
measurement has not yet been recorded.

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

The intended long-term performance answer is ahead-of-time generation of
straight-line Rust coefficient kernels. Such kernels would let rustc/LLVM see
the complete fixed expression graph, removing the runtime tape dispatch while
preserving the pure-Rust deployment model.

## Decision guide

Prefer original `fcmaes` plus `pykep` when:

- an existing application already depends on the original Python APIs;
- strict behavioral compatibility with upstream pykep is more important than
  a C/C++-free deployment;
- warmed Taylor or ZOH propagation dominates a long-running optimization;
- one heyoka integrator can be created once and reused per worker;
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
- original pykep/heyoka can perform that final warmed low-thrust stage;
- Rust DOP853 or an independent tool should validate the finalists.

## GTOC1 recommendation

For the current GTOC1 tutorial, the evidence supports a split decision:

- the native Rust stack remains attractive for planet-order experiments,
  cheap Lambert/flyby scoring, coordinated retry, and parallel objective
  evaluation;
- a warmed and reused original-pykep ZOH integrator should currently be faster
  for the isolated low-thrust propagation stage;
- the complete Rust campaign may recover some of that kernel disadvantage by
  avoiding Python callbacks and foreign-function transitions, but this has
  not been demonstrated by a controlled end-to-end comparison;
- every claimed trajectory still needs independent propagation and constraint
  validation, regardless of which stack optimized it.

Before choosing a stack for a production campaign, benchmark the exact
objective with the same trajectory representation, tolerance, segment count,
evaluation budget, worker count, CPU affinity, and validation criteria. Report
JIT setup and warmed throughput separately.

