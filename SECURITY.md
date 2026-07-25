# Security policy

This project is pre-release and currently provides no numerical API. Security
and correctness reports should be sent privately to the repository owner once
the permanent repository and contact channel are assigned. Until then, do not
include secrets or sensitive mission data in a public issue.

Supported releases and a permanent reporting address will be listed here
before the first publication. Dependency advisories are checked in CI with
RustSec, and dependency sources and licenses are checked with `cargo-deny`.

Numerical correctness defects are treated as security-relevant when they can
silently return plausible but invalid mission-analysis results. Reports should
include the smallest reproducible input, platform, crate/Python version, and
observed residual or invariant failure.
