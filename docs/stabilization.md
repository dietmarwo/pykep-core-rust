# Stabilization and release-candidate evidence

This document records the Phase 18 checks that complement numerical parity
and invariant testing. It does not claim that registry publication,
cross-platform hosted CI, external API review, or a release tag has occurred.

## Matched Rust/C++ distribution

The representative five-segment Sims–Flanagan case uses byte-for-byte
equivalent inputs and output work in the Rust release tool and an external C++
oracle harness. Both ran sequentially on CPU 0 with one thread, a three-second
warmup per workload, 100 samples, 10,000 mismatch calls/sample, and 1,000
gradient calls/sample. Rust used 1.97.1 with the workspace release profile
(`opt-level=3`, thin LTO, one codegen unit); C++ used GCC 14.3.0 and CMake
Release. The host is an AMD Ryzen 9 9950X under Linux 6.8.0-136.

The host governor reported `powersave`; CPU frequency was not fixed. Hardware
performance counters were unavailable because `perf_event_paranoid=4`.
Accordingly, these are same-session orientation distributions, not universal
language-speed claims. The deterministic analysis script reports a
percentile-bootstrap 95% interval for the median:

| Implementation | Workload | N | Mean | Median | P05–P95 | Median 95% CI |
|---|---|---:|---:|---:|---:|---:|
| Rust | Sims–Flanagan mismatch | 100 | 1,014.60 ns | 1,013.85 ns | 1,012.53–1,019.03 ns | 1,013.67–1,014.24 ns |
| Rust | Sims–Flanagan gradient | 100 | 4,287.93 ns | 4,253.52 ns | 4,234.28–4,334.24 ns | 4,247.88–4,261.24 ns |
| C++ | Sims–Flanagan mismatch | 100 | 1,051.25 ns | 1,046.24 ns | 1,044.23–1,082.56 ns | 1,045.86–1,046.90 ns |
| C++ | Sims–Flanagan gradient | 100 | 13,720.82 ns | 12,831.45 ns | 12,767.07–20,305.14 ns | 12,816.10–12,845.25 ns |

Reproduce the Rust samples and table with:

```bash
cargo run --release -p pykep-release-benchmark > rust.csv
python tools/analyze_benchmark.py Rust=rust.csv C++=cpp.csv
```

The C++ oracle remains outside the public repository by policy.

## Regression policy

CI runs an 11-sample/100-ms-warmup smoke protocol. Median limits are 10 µs for
mismatch and 45 µs for the gradient, about 9.9× and 10.6× the local baselines.
They intentionally catch order-of-magnitude regressions while leaving room
for shared-runner scheduling and CPU differences. They are not used for
cross-language assertions or small optimization claims.

## Allocation, cache, vectorization, and scaling

Valgrind 3.22.0 found zero memory errors and zero definitely/indirectly/
possibly lost bytes; one 544-byte runtime block remained reachable at exit.
DHAT quick-protocol aggregates were:

| Workload | Allocated bytes | Blocks | Peak live |
|---|---:|---:|---:|
| mismatch only | 547,760 | 13,609 | 1,938 bytes / 9 blocks |
| gradient only | 24,615,408 | 116,945 | 6,898 bytes / 33 blocks |

The gradient returns dynamic output matrices and is the clear allocation
target. Cachegrind over both quick workloads recorded 335,788,370
instructions, 133,876,521 data references, 3,775 L1 data misses, 5,590
last-level misses, and a 2.2% branch-mispredict rate. LLVM loop-vectorization
remarks and emitted IR showed no packed-double vector loop in this workload.
No algorithm was changed: the active parity and invariant suite is green, and
the profile points to API output storage rather than numerical formulas.

Release-wheel Lagrange batch scaling (median of 11) was:

| Batch | ns/item | items/s |
|---:|---:|---:|
| 1 | 590.00 | 1.69 million |
| 16 | 138.12 | 7.24 million |
| 256 | 97.00 | 10.31 million |
| 4,096 | 99.45 | 10.06 million |
| 65,536 | 93.51 | 10.69 million |

## Dynamic analysis and fuzzing

- Miri nightly passed selected fixed-layout, derivative-consistency,
  error-formatting, and epoch-parser tests. Native tests remain authoritative
  for exact floating-point references because Miri software floating point can
  differ by a few ULPs.
- Valgrind Memcheck passed the release benchmark with zero errors.
- Ten-second libFuzzer/AddressSanitizer campaigns completed without a crash:
  13,016,052 epoch-parser inputs, 10,115,059 element-conversion inputs, and
  4,588,894 Lambert inputs.
- RustSec and cargo-deny pass. `paste 1.0.15` remains an allowed unmaintained
  transitive dependency under `RUSTSEC-2024-0436`, documented in ADR 0004.

The fuzz harness is excluded from released artifacts. Its native compiler and
sanitizer are development tools, not core build/runtime dependencies.

## Public API documentation

The Rust core denies missing documentation for public items, and rustdoc builds
with warnings denied. The Python audit inventories all 145 exported names;
each is represented in the checked type stub and directly exercised by the
integration suite. Public module and object docstrings are checked alongside
that inventory. The migration matrix records every unavailable upstream name
with a technical reason and tracking identifier. External review is still
required before declaring these public names frozen.

## Local artifact consumption

`cargo package` produced and verified an 80-file, 2.3 MiB compressed
`pykep-core-0.1.0.crate`. Its normal/build graph contains only Rust crates and
no `cc`/CMake/native-library dependency. The extracted archive—not the
workspace source—compiled and ran from a fresh Cargo binary project.

Maturin produced a 2.9 MiB CPython 3.12 manylinux wheel and a 2.4 MiB sdist.
The wheel and the wheel built from the sdist both imported from separate fresh
virtual environments. The packaged stub and `py.typed` marker were present.
`ldd` reported only `libgcc_s`, `libm`, `libc`, and the ELF loader; there was
no C++/kep3/Boost/heyoka runtime. This is the local Linux result. The hosted
matrix is configured to build and consume wheels for CPython 3.11–3.13 on
Linux, macOS, and Windows, but is not represented here as an executed remote
run.

## External blockers

Public names are documented but not declared permanently frozen: external API
review is still required. Permanent repository/owner/security metadata,
trusted registry publishers, registry uploads, downloads of published
artifacts, and the release tag all require release-owner authority. See
`RELEASE.md` for the exact boundary.
