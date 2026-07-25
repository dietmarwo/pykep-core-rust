# Security policy

Numerical correctness defects are security-relevant when they can silently
return plausible but invalid mission-analysis results. Reports should include
the smallest reproducible input, platform, crate/Python version, and observed
residual or invariant failure.

The permanent repository and private reporting address have not yet been
assigned, so this release candidate must not be published until that release
owner completes the contact metadata. Do not include secrets or sensitive
mission data in a public issue. Once hosted, use the repository's private
security-advisory channel until a dedicated address is listed here.

Dependency advisories are checked with RustSec. Dependency sources, duplicate
versions, and licenses are checked with `cargo-deny`. The native core forbids
unsafe Rust, and release validation includes Miri where supported, Valgrind,
bounded fuzz campaigns, golden parity, and independent numerical properties.
