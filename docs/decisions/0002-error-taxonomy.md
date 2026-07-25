# ADR 0002: Numerical error taxonomy

- Status: accepted
- Date: 2026-07-25

## Context

The C++ implementation mixes exceptions, NaN sentinels, and unchecked
floating-point behavior. A mission-analysis library must not turn failed
validation or convergence into a plausible output.

## Decision

All fallible public numerical operations return `pykep_core::Result<T>` with a
non-exhaustive `PykepError`. Stable categories distinguish:

- invalid values and non-finite values;
- singular geometry;
- convergence failure;
- dimension mismatch;
- unsupported capabilities;
- floating-point overflow;
- integration failure.

Input validation happens at public boundaries. Private, already-validated
helpers may avoid repeated checks in iterative hot paths. Panics are reserved
for violated internal invariants, not caller input.

The Python layer maps invalid values and dimensions to `ValueError`, numerical
overflow to `OverflowError`, and exposes package exceptions for convergence,
singular geometry, unsupported capabilities, and integration failures.

## Consequences

This intentionally differs from upstream functions that return NaN for an
invalid anomaly domain or accidentally return finite limit values for NaN
inputs. Error strings retain useful context, but callers should match the
category rather than punctuation.
