# Pontryagin low-thrust dynamics

Phase 13 provides evaluated Cartesian and modified-equinoctial canonical
dynamics for indirect low-thrust optimization. The implementation is native
Rust and uses the common integration and sensitivity contracts; it does not
expose or depend on a symbolic-expression runtime. The four built-in types can
be passed to the default Taylor `AdaptiveIntegrator`, to `Taylor` directly, or
to `Dop853` when events or direct variational propagation are needed.

## Orders and units

The Cartesian augmented state is

```text
[x, y, z, vx, vy, vz, m, lx, ly, lz, lvx, lvy, lvz, lm]
```

The modified-equinoctial augmented state is

```text
[p, f, g, h, k, L, m, lp, lf, lg, lh, lk, lL, lm]
```

The equinoctial convention is prograde and thrust direction is expressed in
the radial, transverse, normal (RTN) frame. Cartesian thrust direction is
inertial. `p`, position, velocity, mass, time, `mu`, thrust, and exhaust
velocity must form one consistent unit system. Angles are radians. Costate
units follow from the caller's chosen normalization.

Mass-optimal models use the parameter order

```text
[mu, maximum_thrust, exhaust_velocity, barrier, lambda0]
```

`barrier` is the positive logarithmic regularization coefficient. Time-optimal
models use

```text
[mu, maximum_thrust, exhaust_velocity]
```

and have full throttle with the upstream implicit `lambda0 = 1`. Maximum
thrust may be zero, which is useful for canonical coordinate checks; all other
physical scale parameters and `lambda0` must be positive.

## Control and Hamiltonian

The public control evaluators return `OptimalControl`, containing throttle,
direction, and the switching function `rho`. Mass optimality uses the
regularized upstream control

```text
u = 2 eps / (rho + 2 eps + sqrt(rho² + 4 eps²))
```

evaluated with an algebraically equivalent branch that avoids cancellation.
Time optimality returns `u = 1`. The Hamiltonian functions evaluate the
minimized autonomous Hamiltonian using that control.

The minimizing direction is undefined when the relevant primer norm is
exactly zero. Rust reports `PykepError::SingularGeometry`, and Python reports
`SingularGeometryError`; no arbitrary direction or NaN vector is returned.
Cartesian position at the origin, non-positive mass or equinoctial
semilatus rectum, and a zero equinoctial radial denominator are also explicit
errors.

## Models and sensitivities

The four zero-sized model types are:

- `CartesianMassOptimal`
- `CartesianTimeOptimal`
- `EquinoctialMassOptimal`
- `EquinoctialTimeOptimal`

Each implements `DynamicsModel<14, P>` and
`DifferentiableDynamicsModel<14, P>`. Canonical costate rates use forward-mode
derivatives of the minimized Hamiltonian. Full state and parameter Jacobians
use centered fixed-size differences and are stored as
`jacobian[output][input]`. This path performs no heap allocation in the model
right-hand side.

In the Taylor recurrence the minimizing control evolves with the complete
state/costate series. When forming canonical costate rates, the control is
held fixed with respect to the physical-state derivative, as required by the
envelope theorem. Zero primer norm remains an explicit non-analytic error.

Python requires the native `Optimality.Mass` or `Optimality.Time` enum.
Unvalidated strings are not accepted. The `pontryagin_*_rhs`, control,
Hamiltonian, and propagation functions use the same `pykep-core`
implementation.

## Validation

The committed C++ oracle covers both coordinates and both objectives, their
eight upstream variational arguments (seven initial costates and `lambda0` in
mass mode), and the upstream dimensional 100-day trajectory. Independent
checks cover Hamiltonian conservation, propagated central differences,
Jacobian orientation, zero-primer errors, normalized controls, and canonical
Hamiltonian agreement after transforming costates between coordinates with
the analytic element Jacobian.

At `2e-12` DOP853 scalar tolerances and a `0.01` maximum step for normalized
cases, nominal trajectories agree with the pinned C++/heyoka oracle within a
scaled `3e-10`. The dimensional case uses a 12-hour maximum step and a scaled
`3e-9`. First-order variations use a scaled `2e-4` tolerance because the
generic Jacobian contract is numerical.
