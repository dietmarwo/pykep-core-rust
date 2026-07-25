# ADR 0003: embedded thresholded VSOP2013 evaluator

- Status: accepted
- Date: 2026-07-25

## Context

The upstream pykep provider asks heyoka to construct VSOP2013 expressions and
JIT-compile a function for each `(body, threshold)` pair. That is not suitable
for a C/C++-free Rust crate, reproducible offline builds, or bounded global
state.

The authoritative IMCCE solution contains 2,607,947 terms. In heyoka's packed
layout, retaining every term would require about 100 MiB before compression.
The pykep tests evaluate at thresholds down to `1e-9`; 112,270 terms survive
that floor and occupy 4.3 MiB in the committed encoding. The ordinary pykep
default is `1e-5`, for which only 1,858 terms survive across all bodies.

The coefficient source in heyoka 7.10.0 is MPL-2.0-covered code generated from
the IMCCE VSOP2013 release. The public data notice pins the exact heyoka commit,
generator, retained threshold, and binary hash.

## Decision

The default `vsop2013` Cargo feature embeds the original integer multipliers
and binary64 coefficients down to `1e-9`. A provider validates and decodes only
its selected planet, applies the requested threshold once, and owns the
result through `Arc`. Evaluation is a direct Rust series evaluator followed by
the published equinoctial-to-Cartesian transformation and ICRF rotation.

Thresholds below `1e-9` return an explicit validation error. Disabling default
features removes the coefficient asset; construction then returns an explicit
unsupported-capability error. Availability and the threshold floor are
queryable in Rust and Python.

There is no process-global provider cache. Default-threshold initialization
measured about 11.6 µs, so a cache would add synchronization and lifetime
complexity without justifying itself. Cloning an existing provider shares its
immutable decoded data.

No source checkout, generator, network access, C++, LLVM, or heyoka is needed
to build or run the public crate.

## Measurements

An orientation run on an AMD Ryzen 9 9950X measured:

| Workload | Rust | C++/heyoka JIT |
|---|---:|---:|
| default `1e-5` initialization | 11.6 µs | 64.0 ms |
| default scalar state | 340 ns | 166 ns |
| Rust default 256-state batch | 89.6 µs | no batch API |
| `1e-9` initialization | not separately isolated | 553 ms |
| `1e-9` scalar state | 37.2 µs | 5.91 µs |
| release benchmark executable size increase | about 4.4 MiB | requires shared heyoka/LLVM runtime |

The Rust and C++ timing harnesses use the same Earth–Moon provider and epoch.
These are orientation results without fixed CPU frequency, not release
performance guarantees.

Across 54 golden states at `1e-9`, the largest absolute difference from the
JIT-compiled upstream path was 0.185 m in position and
`6.6e-8 m/s` in velocity. The difference is consistent with floating-point
summation and trigonometric evaluation order.

## Consequences

- Startup, builds, and runtime dependencies are bounded and reproducible.
- Keplerian and JPL low-precision providers remain available without the
  feature.
- The full sub-`1e-9` theory is intentionally not embedded because its data
  and evaluation cost are disproportionate to the upstream-tested public use.
- The direct evaluator is slower than optimized JIT code at `1e-9`; Phase 18
  may profile argument precomputation, SIMD-friendly term grouping, and batch
  evaluation while golden tests remain active.
