# Security policy

## Supported versions

Security and numerical-correctness fixes are made for the latest published
`pykep-core` crate and `pykep-rust` Python package. Check whether a suspected
defect reproduces on the latest release when practical, but report a credible
issue even if it was first observed on an older version. Older releases are
not normally patched separately during the pre-1.0 development series.

## What is security-relevant?

For an astrodynamics library, security includes numerical integrity. A defect
is security-relevant when it can silently return a plausible but invalid
mission-analysis result, especially when ordinary API use provides no warning.
Examples include:

- propagation, Lambert, anomaly, or low-thrust results that violate their
  documented endpoint residuals or physical invariants;
- platform-dependent results that cross a documented feasibility boundary;
- memory-safety, denial-of-service, parser, Python-boundary, or dependency
  vulnerabilities; and
- malformed inputs that bypass validation and produce trusted-looking output.

Documented approximation limits, expected non-convergence, and differences
between supported ephemeris models are not vulnerabilities by themselves.
They may still be reported as ordinary correctness or documentation issues.

## Reporting a vulnerability

Report vulnerabilities privately through the repository's
[GitHub security-advisory form](https://github.com/dietmarwo/pykep-core-rust/security/advisories/new).
Do not put exploit details, credentials, private keys, proprietary trajectories,
or sensitive mission data in a public issue.

If private reporting is temporarily unavailable, open a public issue containing
only a request for a private contact channel. Do not include the vulnerability
details or sensitive reproducer in that issue.

A useful report contains:

- the smallest non-sensitive input that reproduces the problem;
- the affected Rust API or Python function;
- the crate or Python-package version and relevant feature flags;
- the Rust/Python version, operating system, architecture, and CPU;
- the observed result and the expected result;
- the residual, invariant failure, or trusted independent reference; and
- the potential impact and whether the issue is already public.

Maintainers aim to acknowledge a private report within seven days and provide
an initial assessment within fourteen days. These are best-effort targets for
a volunteer-maintained project, not a service-level guarantee. Reporter and
maintainer should coordinate disclosure until a fix or advisory is available.
Credit will be included when requested.

## Project safeguards

Dependency advisories are checked with RustSec. Dependency sources, duplicate
versions, and licenses are checked with `cargo-deny`. The native core forbids
unsafe Rust. This restriction applies to project code; dependencies may contain
audited unsafe implementations.

Release validation includes Miri where supported, Valgrind, bounded fuzz
campaigns, golden parity tests, and independent numerical-property tests. These
checks reduce risk but do not constitute formal verification or guarantee that
every mission-analysis result is correct. Users remain responsible for
independent validation appropriate to the consequence of their application.
