# Zero-order-hold dynamics

Phase 12 provides a common `ControlSchedule<C>` and four evaluated models:

- `ZohKeplerDynamics`: `[x,y,z,vx,vy,vz,mass]`, normalized `mu = 1`;
- `ZohCr3bpDynamics`: the same seven-state layout in the CR3BP synodic frame;
- `ZohEquinoctialDynamics`: `[p,f,g,h,k,L,mass]`, normalized `mu = 1`;
- `ZohSolarSailDynamics`: six Cartesian states with cone/clock attitude.

The first three control rows are `[thrust, i1, i2, i3]`. The direction is
Cartesian for Kepler/CR3BP and radial-transverse-normal for equinoctial
dynamics. Kepler/equinoctial constants contain mass-flow coefficient `c`;
CR3BP constants are `[c, mu]`. The preserved upstream mass equation is
`dm/dt = -c thrust exp(-1 / (mass 1e16))`.

Solar-sail rows are `[alpha, beta]` in radians. Its constant `c` scales ideal
sail acceleration as `c cos(alpha)^2 / r²`.

## Schedule and switch contract

A schedule has `S + 1` strictly increasing finite boundaries and exactly `S`
finite control rows. Construction performs all dimension, finiteness, and
monotonicity checks.

For lookup, segment `i` owns `[t_i, t_(i+1))`; the final boundary belongs to
the final segment. Forward propagation ends each integration exactly at a
switch and starts the next segment with its new control. Backward propagation
uses the interval to the left of each encountered switch. This directional
rule avoids evaluating a segment across a discontinuity.

Each segment is integrated exactly once. The active control is copied into a
fixed-size parameter array before the solve, so no allocation or schedule
scan occurs in an RHS call. `control_at` uses binary search for occasional
external lookup.

The established `propagate_schedule*` functions continue to use DOP853.
The corresponding `*_with_method` functions accept
`IntegrationMethod::Taylor` for all four built-in models. Each segment still
ends exactly at its switching boundary; backend selection does not change
switch ownership.

## Sensitivities

`ZohSensitivitySeeds` carries an arbitrary compile-time seed width through the
schedule in one augmented solve per segment. It contains:

- initial state seeds;
- one control seed matrix per segment;
- constant-parameter seeds.

State sensitivities are continuous at switches. A control column for a future
segment remains exactly zero until that segment becomes active. Runtime is
linear in the segment count for a fixed seed width; the DOP853 implementation
does not repropagate every prior segment for every control.

Taylor's current seeded-sensitivity implementation uses centered complete
propagations and therefore costs `2W + 1` nominal solves per segment. Prefer
DOP853 for wide ZOH Jacobians until direct Taylor variational recurrences are
implemented.

Model Jacobians use fixed-size, allocation-free central differentiation of
the evaluated source equations. They are checked against C++ Taylor
variations and end-to-end schedule finite differences. The longest CR3BP
reference uses a `2e-5` relative/absolute variation comparison; nominal state
parity remains `3e-10`. This looser derivative tolerance is explicit and will
be revisited if later leg gradients require analytic Jacobians.

## Python API

Python exposes all four RHS functions and `propagate_zoh_*` functions.
Boundaries and controls are ordinary nested sequences. Propagation releases
the GIL and accepts `backward`, scalar tolerances, and `maximum_step`.
Malformed row widths and grids are rejected before native integration.
