# Sims–Flanagan low-thrust legs

The [matched benchmarks](matched-benchmarks.md) cover direct leg calls,
complete optimizer-facing gradients and PyGMO SLSQP convergence.

`pykep-core::leg` provides the fixed-duration `SimsFlanaganLeg` and the
variable-duration `SimsFlanaganAlphaLeg`. Both are immutable after validated
construction, use native Rust two-body propagation, and have no C or C++
runtime dependency.

## Units and ordering

All inputs must use one self-consistent unit system:

- endpoint state is `[x, y, z, vx, vy, vz]`;
- endpoint mass and maximum thrust use compatible mass and force units;
- time of flight, segment durations, and exhaust velocity use compatible time
  units;
- `mu` has length cubed per time squared;
- throttle vectors are dimensionless and appear in chronological order.

An endpoint is represented by `SpacecraftEndpoint`, which rejects non-finite
state components, zero radius, and non-positive mass.
`SimsFlanaganSettings` rejects negative time of flight or maximum thrust,
non-positive exhaust velocity or `mu`, non-finite values, and cuts outside
`[0, 1]`. At least one three-component throttle is required.

## Transcription and cut

For a leg with `N` segments, the number propagated forward from departure is

```text
floor(N * cut)
```

and the remaining segments are propagated backward from arrival. A cut of
zero therefore starts at the departure endpoint and propagates every segment
backward; a cut of one propagates every segment forward.

Each segment applies its finite impulse at the segment midpoint. The
fixed-duration leg uses duration `time_of_flight / N` for every segment.
The velocity increment has magnitude
`maximum_thrust * duration * |throttle| / mass`; the mass update follows the
Tsiolkovsky exponential using `exhaust_velocity`. Backward propagation
reverses both the impulse and mass update.

The alpha leg accepts direct non-negative segment durations. As in the
upstream class, their sum is not required to equal `time_of_flight`.
`SimsFlanaganAlphaLeg::from_time_weights` is the explicit alternative that
requires positive weights and normalizes them to the configured time of
flight.

## Constraints and derivatives

`mismatch_constraints()` returns seven values in this order:

```text
[forward_rx - backward_rx,
 forward_ry - backward_ry,
 forward_rz - backward_rz,
 forward_vx - backward_vx,
 forward_vy - backward_vy,
 forward_vz - backward_vz,
 forward_mass - backward_mass]
```

`throttle_constraints()` returns `N` values,
`dot(throttle_i, throttle_i) - 1`. A zero value is the unit-throttle limit;
negative values are inside it.

The fixed leg exposes an analytic `mismatch_jacobian()` with output-by-input
rows:

| Group | Shape | Column order |
|---|---:|---|
| departure | `7 × 7` | `[x,y,z,vx,vy,vz,mass]` |
| arrival | `7 × 7` | `[x,y,z,vx,vy,vz,mass]` |
| controls and time | `7 × (3N + 1)` | `[u0x,u0y,u0z,...,u(N-1)z,time_of_flight]` |

`throttle_jacobian()` has shape `N × 3N` in the same flattened control order.
At zero throttle the mass-direction derivative uses the defined zero
subgradient, so the returned matrix remains finite.

The upstream alpha class does not expose an analytic mismatch gradient, and
neither does this port. Its exact throttle Jacobian remains available.

## Python

`pykep_rust.SimsFlanaganLeg` and
`pykep_rust.SimsFlanaganAlphaLeg` expose the same immutable evaluations.
Python states and controls are converted once at construction; all
astrodynamics calculations call `pykep-core`. The fixed-leg
`mismatch_jacobian()` returns a tuple of the three row matrices described
above.

Construction and evaluation raise `ValueError` for invalid values or shapes
and `RuntimeError` for propagation failures.

## Multiple-shooting chain

`pykep_trajopt::SimsFlanaganChain` adds explicit node states between uniform
arcs of one fixed-duration leg. Each arc returns seven matching constraints
and the squared norms of its controls. The total duration, impulse model and
global forward/backward cut match the single leg. The backward mass update
is retained on the far side of the cut. The Jacobian is analytic and
available as CSR, and the arc row groups can be passed to a partitioned
SQP solver.

Python wheels from 0.1.6 export
`pykep_rust.trajopt.SimsFlanaganChain`. The `lift` method converts an array of
throttles, a terminal mass and a departure velocity excess to a node-ordered
decision vector. Read `jacobian_indptr`, `jacobian_indices` and `elements`
once, then call `evaluate(x)` and `jacobian_values(x)` at each point. The
objective is negative terminal mass, so the last gradient entry is -1.
The [SQP5 tutorial](https://github.com/dietmarwo/SQP-craft/tree/main/tutorials/low-thrust-pykep)
shows the full Earth–Mars example and the independent DOP853 replay.
