# Upstream source and provenance

This project is an independent Rust adaptation of numerical algorithms from
the European Space Agency's `pykep`/`kep3` project. It is not an official ESA
release and does not imply endorsement by ESA or the upstream authors.

## Pinned baseline

- Upstream repository: <https://github.com/esa/pykep>
- Upstream version: `3.0.1`
- Upstream commit: `53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e`
- Commit date: 2026-07-07
- Local snapshot verification: the `src/`, `include/kep3/`, `test/`, and
  `benchmark/` trees used for the port match that commit byte-for-byte, apart
  from the generated `include/kep3/config.hpp` and notebook files excluded
  from the comparison.

The upstream source and adaptations are distributed under the Mozilla Public
License 2.0. See [LICENSE](LICENSE). Rust files translated or substantially
adapted from upstream retain an SPDX header and identify the relevant upstream
source file in a provenance comment.

Golden numerical data committed under
`crates/pykep-core/tests/data/` is generated from the pinned C++ source by
development-only tools kept outside this standalone public tree. The released
Rust crates do not link to or call the C++ implementation.

## Names and ownership

The working names are `pykep-core`, `pykep-rust`, and `pykep_rust`. They remain
provisional until the repository owner and permanent public repository URL are
chosen. Cargo metadata deliberately omits an upstream ESA repository URL.

## Deliberate differences

- Rust errors replace C++ exceptions and non-finite sentinel results at public
  validation boundaries.
- Rust APIs expose evaluated dynamics and propagators instead of heyoka
  expression graphs.
- Boost archive bytes, C++ ABI compatibility, RTTI, and `tanuki` type erasure
  are not ported.
- Python bindings are a thin facade over the same Rust numerical
  implementation.

Substantial algorithm-specific deviations are documented beside the affected
module and in the architecture decision records.

In particular, the hyperbolic anomaly and Lagrange propagators use wider
convergence safeguards, Lambert iteration exhaustion is an error instead of
an unchecked last iterate, direct Cartesian-to-equinoctial conversion uses
`|h|² / mu`, and time-optimal Pontryagin fixes the upstream implicit
`lambda0` to one. The Python migration matrix records the user-visible
Pontryagin parameter difference.
