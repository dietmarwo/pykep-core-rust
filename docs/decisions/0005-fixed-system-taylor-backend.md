# ADR 0005: Add Taylor as the built-in nominal backend

- Status: accepted
- Date: 2026-07-28
- Supersedes: ADR 0004's default for built-in nominal dynamics; DOP853 remains
  the general backend

## Context

ADR 0004 selected DOP853 because it supplied final state, dense output,
events, and direct variational equations behind a small pure-Rust dependency.
The same evidence showed a nominal high-accuracy performance gap relative to
the upstream warmed heyoka implementation.

The eleven pykep dynamics types come from only nine fixed equation families.
They do not require arbitrary symbolic expressions, LLVM, events in Taylor
mode, heyoka batch mode, or arbitrary precision. A disposable eccentric
Kepler prototype passed a predeclared matched-error gate at tight tolerances.

## Decision

Add a private fixed-capacity Taylor-series engine inside `pykep-core`.
Implement coefficient kernels only for the built-in models. Publish
`Taylor`, `IntegrationMethod`, and `AdaptiveIntegrator`, but keep the
coefficient trait private until a real third-party system requires it.

Make Taylor the default of `AdaptiveIntegrator` and of the nominal
Kepler/CR3BP/BCP convenience methods. This follows upstream pykep, where these
TA-derived equation families are exposed as heyoka Taylor-adaptive
integrators. Keep direct `Dop853` calls, generic user dynamics, event
location, and the existing direct-variational sensitivity convenience methods
on DOP853. Add explicit `*_with_method` variants for built-in dynamics and ZOH
schedules.

Use tolerance-selected order 8–24 and component-scaled step selection from
the final two coefficients. Preserve an optimized Kepler recurrence where
full-series generic evaluation would erase the measured crossover.

For the first release, dense grids clip Taylor steps at sample times and
seeded sensitivities use centered complete propagations. Document their cost
instead of claiming the not-yet-implemented polynomial interpolation and
direct Taylor variational optimizations.

## Consequences

- No C/C++ runtime, LLVM, JIT latency, or new runtime dependency is added.
- At `1e-14` and machine epsilon the representative Kepler solve is faster
  and more accurate than DOP853; at `1e-9` it is slower.
- Built-in nominal dynamics default to the algorithm family used by upstream
  pykep; callers can still request DOP853 explicitly.
- User-defined `DynamicsModel` implementations continue to use DOP853.
- Wide STMs and event-driven problems should continue to use DOP853.
- Equation duplication between evaluated and coefficient kernels is accepted
  as localized technical debt, protected by upstream, DOP853, and
  official-heyoka cross-validation.

The raw measurements and validation boundaries are recorded in
[High-accuracy Taylor integration](../taylor-integration.md).
