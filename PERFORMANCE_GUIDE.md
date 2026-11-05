# Choosing between the original and Rust fcmaes/pykep stacks

This guide uses the 0.1.5 shared-input comparison with PyKEP 3.1.0. Warmed
heyoka is faster on nine of the eleven scalar Taylor fixtures and on
nominal ZOH-leg mismatch. Rust is faster on both equinoctial Pontryagin
fixtures, on complete ZOH-leg Jacobians, on most analytical propagation and
STM kernels and on the Sims–Flanagan Jacobian. Use the row for the required
model, accuracy and timing boundary.
[Full 0.1.5 report](docs/matched-benchmarks.md)

## The two stacks

| | Original stack | Rust stack |
|---|---|---|
| Optimizer | Python/C++ `fcmaes` | `fcmaes-core`, optionally through `fcmaes-rust` |
| Astrodynamics | Python/C++ `pykep`/`kep3` | `pykep-core`, optionally through `pykep-rust` |
| Taylor integration | heyoka symbolic decomposition and LLVM JIT | Ahead-of-time Rust with generated fixed-model coefficient kernels |
| Native dependencies | C++, heyoka, LLVM, and their transitive build/runtime requirements | Rust only for the numerical cores |
| Python in a native hot path | Normally crosses a Python/native boundary unless the complete objective is native | None when the objective uses the Rust crates directly |
| First use | JIT and cache construction for heyoka systems | No JIT setup |
| Best steady state | Reuse one compiled heyoka integrator per worker | Reuse native objective data and parallelize without Python |

## Measurement boundary

The 0.1.5 study ran on an Intel Xeon Platinum 8573C under Linux, with each
process pinned to one logical CPU. Rust used Rust 1.88, release thin LTO and
the portable x86-64 target with SIMD compiled in. The reference used PyKEP
3.1.0 at `c01e19de`, heyoka 7.10.0 (ABI 42) and a Clang 20.1.2 driver
built with `-O3 -march=x86-64`.

Both arms receive the same initial states, parameters and final times.
Both request `1e-12` and must pass a `1e-11` scaled endpoint-error gate
against a tight DOP853 reference, so the rows compare achieved accuracy,
not only equal tolerance settings. Ten paired rounds alternate the order of
the arms. A ratio is the median of the paired Rust/heyoka times. The
interval resamples whole pairs.

The heyoka arm resets and reuses one already JIT-compiled integrator, the
best steady-state way to use heyoka. The Rust arm is a stateless
ahead-of-time-compiled propagation. Neither timing includes Python, NumPy
conversion, optimizer callbacks or process communication. Treat the numbers
as measured guidance for the stated fixtures, not portable performance
guarantees.

## Warmed Taylor propagation

| Model | Rust µs | Warmed heyoka µs | Faster side | 95% paired interval of Rust / heyoka |
|---|---:|---:|---|---:|
| Kepler | 4.6 | 2.02 | heyoka 2.26× | 1.73–3.31 |
| CR3BP | 10.8 | 4.59 | heyoka 2.57× | 1.98–3.24 |
| Bicircular problem | 7.01 | 4.22 | heyoka 1.94× | 1.31–2.22 |
| ZOH Kepler | 5.11 | 2.92 | heyoka 1.77× | 1.25–2.11 |
| ZOH CR3BP | 11.2 | 6.54 | heyoka 1.67× | 1.50–1.84 |
| ZOH equinoctial | 15 | 6.73 | heyoka 2.19× | 1.69–3.12 |
| ZOH solar sail | 21.3 | 9.09 | heyoka 2.49× | 2.00–2.85 |
| Cartesian mass-optimal | 8.88 | 4.02 | heyoka 2.28× | 1.91–2.54 |
| Cartesian time-optimal | 8.24 | 3.72 | heyoka 2.27× | 1.71–2.39 |
| Equinoctial mass-optimal | 536 | 673 | Rust 1.28× | 0.69–0.87 |
| Equinoctial time-optimal | 516 | 623 | Rust 1.20× | 0.75–0.91 |

Heyoka compiles each concrete model to a specialized straight-line kernel.
Rust 0.1.5 generates fixed-model coefficient kernels ahead of time for
selected Taylor paths and evaluates the rest through incremental
recurrences. On the ZOH Kepler fixture the medians correspond to about
0.20 million Rust propagations and 0.34 million warmed-heyoka propagations
per second on one core.

### JIT setup is excluded above

The warmed rows exclude heyoka's first construction of each integrator,
which includes JIT compilation. The driver measured it once per process for
each fixture at `1e-12`. The table gives the median of the ten rounds.

| Model | heyoka setup |
|---|---:|
| Kepler | 167 ms |
| CR3BP | 340 ms |
| Bicircular problem | 630 ms |
| ZOH Kepler | 234 ms |
| ZOH CR3BP | 448 ms |
| ZOH equinoctial | 1.14 s |
| ZOH solar sail | 831 ms |
| Cartesian mass-optimal | 1.49 s |
| Cartesian time-optimal | 1.26 s |
| Equinoctial mass-optimal | 1.21 s |
| Equinoctial time-optimal | 1.39 s |

Rust has no setup phase. The setup cost matters for short command-line
programs, server workers that start frequently, or applications that
construct many system/tolerance combinations. It becomes negligible in a
long optimization campaign that reuses one compiled integrator per worker.

## Analytical and transcription kernels

The seven upstream benchmark programs run with the upstream case counts
and shared inputs. A ratio above 1 means Rust is faster. Each band is one
group of cases in the upstream program. These are orientation results from
one ABBA block, so small differences are sensitive to host load.

| Workload | Cases per row | Upstream / Rust time by case band | Reading |
|---|---:|---|---|
| Anomaly conversion | 1,000,000 | 1.21× / 1.25× / 0.69× | Mixed |
| Anomaly round trip | 100,000 | 0.48× / 1.15× / 1.47× | Mixed |
| Lagrange state | 1,000,000 | 1.31× / 2.94× / 1.15× / 2.28× | Rust faster |
| Lagrange round trip | 100,000 | 1.54× / 1.23× / 1.57× | Rust faster |
| State and 6 × 6 STM | 100,000 | 2.47× / 2.49× / 1.58× | Rust faster |
| Lambert, 0 and ≤20 revolutions | 50,000 | 0.84× / 0.79× | Upstream faster |
| Sims–Flanagan mismatch | 1 | 1.33× | Rust faster |
| Sims–Flanagan complete Jacobian | 1,000 | 4.05× / 4.47× / 3.29× / 2.53× | Rust faster |
| Sims–Flanagan numerical UDP gradient | 1,000 | 1.05× / 1.14× / 1.12× / 1.17× | About equal |

The warmed-heyoka advantage in the Taylor table therefore does not extend
to these conventional numerical kernels.

## Complete ZOH legs

Each segment count uses 200 shared cases and cached legs. Rust uses the
Taylor backend, nominal integration at `1e-14` and both variational
tolerances at `1e-13`. The complete Jacobian includes both endpoint-state
blocks, every segment control and all grid-node derivatives.

| Segments | Operation | Rust µs | Upstream µs | Faster side |
|---:|---|---:|---:|---|
| 20 | mismatch | 36.9 | 23.9 | upstream 1.54× |
| 20 | complete Jacobian | 272 | 556 | Rust 2.05× |
| 40 | mismatch | 67.2 | 51.2 | upstream 1.31× |
| 40 | complete Jacobian | 525 | 1,160 | Rust 2.22× |
| 80 | mismatch | 126 | 102 | upstream 1.23× |
| 80 | complete Jacobian | 1,100 | 2,130 | Rust 1.93× |

`ZohLeg` calls without a method selector keep the DOP853 backend for
compatibility. Select Taylor explicitly, with
`ZohLeg::mismatch_jacobian_with_method(IntegrationMethod::Taylor)` in Rust or
`method="taylor"` in Python, to get the rows above.

## Python gym evaluation

Four TOPS cases with 16 segments and two classical problems ran through the
Python interfaces of both packages. Each cell is the median of seven samples
in one process.

| Problem | Rust fitness µs | PyKEP fitness µs | Rust gradient µs | PyKEP gradient µs |
|---|---:|---:|---:|---:|
| `tops.twobody.P4` | 27.2 | 61.2 | 301 | 670 |
| `tops.mee.P8` | 5,030 | 2,900 | 38,000 | 82,800 |
| `tops.cr3bp.P0` | 434 | 817 | 5,490 | 12,100 |
| `tops.sail.P4` | 1,540 | 1,010 | 6,910 | 11,000 |
| `gym.eve_mga1dsm` | 6.11 | 66.7 | — | — |
| `gym.juice_mo` | 22.6 | 530 | — | — |

Rust gradients were faster on all four TOPS cases. PyKEP fitness was faster
on the equinoctial and solar-sail cases. The classical impulsive problems
were 11–23× faster in Rust.

## Optimizer evidence is separate

The astrodynamics tables do not compare the optimizer implementations.
Optimizer quality must be compared with the same objective, bounds,
evaluation budget, worker count, and stopping rule.

For Sims–Flanagan, PyGMO NLopt SLSQP with the same 100 starting points per
segment count reached the same number of feasible solutions with the Rust
and PyKEP evaluators, 83–85 of 100 for 5, 10 and 15 segments. The evaluator
choice therefore changes time per call, not the solutions found.

The native GTOP study gives compatibility evidence for the optimizers rather
than a universal speed ranking. Under its reported campaigns,
`fcmaes-rust` coordinated DE-to-CMA retry reached the difficult Tandem
target in 85 of 100 experiments, with a 40.2 s mean wall time on 32 native
retry threads. The historical original Python/C++ fcmaes table reports 81 of
100 and 166.9 s with 32 Python processes. The execution models are not
identical, so this shows comparable search capability, not a
four-percentage-point performance win. See the
[fcmaes-rust optimizer comparison](https://github.com/dietmarwo/fcmaes-rust/tree/main/benchmarks/optimizer-comparison)
for the budgets and raw results.

Python objective calls also matter independently of optimizer quality.
`fcmaes-rust` releases the GIL while its optimizer runs, but must reacquire
it for every Python objective callback. A native Rust objective avoids that
boundary and can be evaluated directly by retry workers. Conversely, an
original-pykep objective implemented and retained entirely in C++ can also
avoid most Python overhead.

## Why `pykep-core` remains pure Rust

License compatibility is not the blocker. Upstream pykep, heyoka, and
`pykep-core` use the Mozilla Public License 2.0. Correct notices and
provenance would be required, but heyoka can legally be used.

The decision is instead about the distributed product:

1. **One Cargo-native numerical core.** `pykep-core` builds without a C++
   compiler, C++ ABI, Boost, LLVM development package, or foreign build
   system.
2. **Reliable crates and wheels.** Linux, Windows, and macOS packages do not
   need to locate, link, or bundle a compatible LLVM/heyoka installation.
3. **No JIT lifecycle.** There is no runtime compiler, tolerance-keyed JIT
   cache, executable-memory policy, or per-worker compilation phase.
4. **Predictable provenance and testing.** The Rust implementation is tested
   directly against pinned upstream fixtures while released crates contain
   no upstream binary.
5. **Native composition.** Astrodynamics and optimization share Rust data
   structures and error handling, with no Python or C++ transition inside a
   native objective.
6. **Fixed scope.** The Taylor backend covers eleven built-in dynamics
   systems. Adding another is a source change described in
   [Adding an ODE system](docs/add-ode-system.md). Heyoka's ability to JIT
   arbitrary symbolic ODEs is valuable generality but is not required for
   this closed set.
7. **Safety and maintenance.** Rust ownership and type checking remove many
   memory and FFI failure modes. Avoiding a C++ boundary also avoids ABI and
   exception-translation problems.

There was no maintained native heyoka Rust crate when this architecture was
chosen. Calling the original implementation would therefore require a C++
wrapper and a substantial native dependency stack. Where a study needs
heyoka features such as arbitrary equations of motion, the
[heyoka high-fidelity check](python/examples/heyoka/README.md) shows how to
call heyoka.py next to `pykep-rust` without making it a dependency.

### Could Rust reproduce heyoka's LLVM approach?

Yes, technically. A Rust implementation could construct the symbolic
expression graph, decompose and simplify it, emit LLVM intermediate
representation, and invoke LLVM's ORC JIT through the stable C API. Rust
bindings such as `llvm-sys` or a higher-level wrapper can expose those entry
points. The generated propagation kernel could then have the same essential
specialization opportunity as heyoka's generated C++ path.

That would be a Rust-hosted LLVM JIT, not a pure-Rust deployment. Users and
wheel builders would still need a compatible native LLVM library. Reaching
heyoka-like capability would also require much more than emitting
arithmetic:

- a symbolic expression system with canonicalization and
  common-subexpression elimination;
- Taylor decomposition and efficient coefficient code generation;
- LLVM IR generation, optimization, linking, and ORC JIT lifecycle handling;
- code and cache specialization by model, numeric type, tolerance, and
  target CPU;
- adaptive-order and step-size control plus dense output;
- state, parameter, variational, event, batch, and SIMD interfaces;
- thread-safe caches, diagnostics, serialization, and cross-platform tests.

Implementing that stack is possible, but it would be a substantially larger
project than the fixed-system Taylor backend in `pykep-core`. Binding the
existing C++ heyoka library would reduce algorithm-development work but
retain the C++ ABI and native packaging costs.

Release 0.1.5 already generates straight-line Rust coefficient kernels
ahead of time for selected fixed-model paths and shares nominal
coefficients across practical seed widths. CI checks that the generated
kernels match their generator. The complete-leg rows above measure the
resulting effect on segment derivatives, assembly and call overhead.

## Decision guide

Prefer original `fcmaes` plus `pykep` when:

- an existing application already depends on the original Python APIs;
- strict behavioral compatibility with upstream pykep is more important than
  a C/C++-free deployment;
- warmed nominal Taylor propagation of one of the nine heyoka-leading models
  dominates, and each worker can reuse its compiled integrator;
- arbitrary symbolic ODE construction or other heyoka features are required;
- the native C++/LLVM dependency stack is acceptable.

Prefer `fcmaes-core` plus `pykep-core` when:

- the complete objective can be written in Rust;
- retry, population, or batch evaluations must run without Python callbacks
  or the GIL;
- simple Cargo dependency management and cross-platform binary deployment
  are priorities;
- startup latency, deterministic packaging, or short-lived workers matter;
- analytical propagation, STMs, Sims–Flanagan Jacobians, flyby, ephemeris,
  and optimization kernels dominate rather than warmed Taylor integration
  alone;
- gradient-based low-thrust optimization needs complete ZOH-leg Jacobians
  or equinoctial Pontryagin propagation;
- one memory-safe native implementation should serve both Rust and Python.

Consider a staged or hybrid workflow when:

- a cheap Rust Lambert/flyby model must screen millions of planet-order or
  timing candidates;
- only a small promoted set requires expensive low-thrust optimization;
- the matched leg or objective results favor one backend for that final
  stage;
- Rust DOP853, heyoka or another independent tool should validate the
  finalists.

## GTOC1 recommendation

For the current GTOC1 tutorial, the evidence supports a split decision:

- the native Rust stack remains attractive for planet-order experiments,
  cheap Lambert/flyby scoring, coordinated retry, and parallel objective
  evaluation;
- for low-thrust refinement, select the Taylor backend for ZOH legs: Rust
  computes the complete Jacobian about twice as fast, while warmed heyoka
  computes the nominal mismatch 1.2–1.5× faster;
- the complete GTOC1 campaign has compatibility checks, and the benchmark
  report gives optimizer comparisons for the separate gym and SF workloads;
- every claimed trajectory still needs independent propagation and
  constraint validation, regardless of which stack optimized it.

Before choosing a stack for a production campaign, benchmark the exact
objective with the same trajectory representation, tolerance, segment count,
evaluation budget, worker count, CPU affinity, and validation criteria.
Report JIT setup and warmed throughput separately.

## Earlier measurements

The 0.1.4 version of this guide measured five fixtures on an AMD Ryzen 9
9950X with upstream `kep3` 3.0.1 and heyoka 7.10.0 and found warmed heyoka
2.83–4.53× faster. The
fixtures, host and protocol differ from 0.1.5, so the rows cannot be compared
one to one. The [Taylor performance follow-up](docs/taylor-performance-followup.md)
records the implementation rounds between the two releases, and
[Performance methodology](docs/performance.md) keeps the per-phase
orientation measurements.
