# Documentation

The documentation describes the implemented native Rust core and its thin
Python interface:

- [examples.md](examples.md): Rust/Python quick starts and runnable examples;
- [conventions.md](conventions.md): units, epochs, frames, array layouts, and
  numerical behavior;
- [dynamics.md](dynamics.md), [pontryagin.md](pontryagin.md),
  [zero-order-hold.md](zero-order-hold.md), [ephemerides.md](ephemerides.md),
  [low-thrust-legs.md](low-thrust-legs.md), and [zoh-leg.md](zoh-leg.md):
  module guides;
- [python-api.md](python-api.md): Python units, arrays, defaults, errors,
  ownership, GIL, and typing contract;
- [python-migration.md](python-migration.md): upstream C++/Python names,
  deliberate differences, deferrals, and unsupported ecosystem modules;
- `decisions/`: architecture and dependency decisions;
- [performance.md](performance.md): benchmark methodology and results;
- [stabilization.md](stabilization.md): matched distributions, profiling,
  regression limits, Miri/fuzz/Valgrind evidence, and release blockers;
- [status.md](status.md), [source-map.md](source-map.md), and
  [validation.md](validation.md): limitations, provenance, and validation
  evidence;
- [development.md](development.md): contributor commands and repository
  policy.

Documentation is updated only for implemented behavior. Explicitly deferred or
unavailable functionality is labelled as such in the migration and status
documents.
