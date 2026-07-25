# Evaluated dynamics

Phase 11 provides three stateless six-state models:

- `KeplerDynamics` for inertial two-body Cartesian motion;
- `Cr3bpDynamics` for the circular restricted three-body problem;
- `BcpDynamics` for the time-dependent bicircular problem.

All states are ordered `[x, y, z, vx, vy, vz]`. Kepler inputs use one
consistent dimensional unit system: if position and time are SI, `mu` is in
`m³/s²`. CR3BP and BCP use the upstream nondimensional rotating-frame
convention. Their primaries are fixed at `(-mu, 0, 0)` and
`(1 - mu, 0, 0)`. The BCP Sun is at
`rho_sun [cos(omega_sun t), sin(omega_sun t), 0]`.

`mu` is the secondary-to-total mass ratio and accepts the mathematical range
`[0, 1]`; the customary ordering has `mu <= 0.5`. BCP parameters are
`[mu, mu_sun, rho_sun, omega_sun]`. `mu_sun` must be non-negative and
`rho_sun` positive. The constants module supplies the upstream Earth–Moon–Sun
defaults.

## Evaluation and propagation

Each model exposes `evaluate`, `propagate`, and `propagate_with_stm`. The
propagators use the Phase 10 DOP853 facade. State-transition matrices are
row-major, with output state components in rows and initial state components
in columns. Analytic state and parameter Jacobians implement the generic
sensitivity contract.

For comparisons with the pinned C++ Taylor-adaptive implementation, the test
profile uses relative and absolute tolerances of `2e-13` and a maximum step
of `0.01` nondimensional time units. Across the committed sample grid this
supports state tolerances of `3e-11` for Kepler/BCP and `2e-9` for the longer,
close-approach CR3BP trajectory. These are validation tolerances, not a
guarantee for arbitrary trajectories; callers must choose tolerances based on
their scale, duration, and invariant drift requirements.

CR3BP also exposes the positive effective potential

```text
U = (x² + y²)/2 + (1-mu)/r1 + mu/r2
```

and the Jacobi constant `C = 2 U - |v|²`.

## Python boundary

Python exposes `kepler_rhs`, `cr3bp_rhs`, `bcp_rhs`, the CR3BP potential and
Jacobi functions, and model-specific propagation functions. The propagation
calls release the GIL and accept `initial_time`, scalar tolerances, and an
optional maximum step. Variational variants return `(state, stm)` without
exposing heyoka or any internal expression graph.

Exact collisions with the active model bodies are
`SingularGeometryError`. Invalid mass or distance parameters are
`ValueError`. A failure to finish an otherwise valid propagation is
`IntegrationError`; tests keep those model and integrator failure categories
separate.

Unlike upstream, no compiled-expression cache is needed: the evaluated model
types are zero-sized and integration configuration is local to each call.
