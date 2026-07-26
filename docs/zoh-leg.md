# Generic zero-order-hold leg

`pykep_core::leg::ZohLeg` transcribes a transfer between two fixed endpoint
states as non-uniform segments with continuous, piecewise-constant controls.
It is generic over any Rust model implementing `ZeroOrderHoldModel`; aliases
cover every built-in ZOH dynamics family:

| Alias | State | Control | Constants |
|---|---|---|---|
| `ZohKeplerLeg` | `[x,y,z,vx,vy,vz,m]` | `[thrust,ix,iy,iz]` | `[c]` |
| `ZohCr3bpLeg` | `[x,y,z,vx,vy,vz,m]` | `[thrust,ix,iy,iz]` | `[c,mu]` |
| `ZohEquinoctialLeg` | `[p,f,g,h,k,L,m]` | `[thrust,ir,it,in]` | `[c]` |
| `ZohSolarSailLeg` | `[x,y,z,vx,vy,vz]` | `[cone,clock]` | `[c]` |

The units and frame conventions of each row are those of its model in
[zero-order-hold.md](zero-order-hold.md). `c` is the normalized mass-flow
coefficient for the three low-thrust models and the normalized lightness
coefficient for the solar sail.

## Grid, controls, and cut

For `S` segments, construction requires exactly `S + 1` finite,
strictly-increasing time-grid nodes and `S` finite control vectors. Controls
are stored in chronological order and own the half-open interval
`[time_grid[i], time_grid[i+1])`; the final endpoint belongs to the final
segment.

The cut uses the same convention as the upstream leg:

```text
forward_segments = floor(S * cut)
backward_segments = S - forward_segments
```

The forward state starts at the initial endpoint and the backward state starts
at the final endpoint. The mismatch is their component-wise difference at the
cut. Cuts zero and one are supported without special placeholder states.

The leg is immutable after construction. It rejects invalid endpoint states,
model constants, dimensions, grids, cuts, tolerances, maximum steps, and
maximum-step magnitudes before evaluation.

## Sensitivity layout

`mismatch_jacobian()` returns four output-by-input matrices:

| Group | Shape | Column order |
|---|---:|---|
| initial state | `N × N` | model state order |
| final state | `N × N` | model state order |
| controls | `N × (C*S)` | segment 0 controls, then segment 1, and so on |
| time grid | `N × (S+1)` | every grid node, including both endpoints |

Each segment propagates a local state-transition and active-control matrix.
The leg composes those matrices from each side of the cut and includes the
dynamics jump at every switching time. Constant model parameters are fixed
leg configuration and therefore are not a returned derivative group.

The four built-in models compute their local RHS Jacobians with fixed-size
centered differences using a relative step of `3e-6`. Consequently,
integrator tolerances such as `1e-12` do not imply `1e-12` derivative
accuracy: the pinned end-to-end validation uses scaled tolerances up to
`3e-5`. See [zero-order-hold.md](zero-order-hold.md#sensitivities) for the
model-level contract and validation evidence.

## Integration and failures

`IntegratorOptions` applies independently to every segment, including
relative and absolute tolerances, optional initial and maximum step sizes,
maximum steps, and maximum rejections. A propagation failure is reported as
an `IntegrationFailure` containing the direction, chronological segment
index, and attempted time interval.

`state_history(samples_per_segment)` uses one DOP853 dense solve per segment
for uniformly spaced states including both endpoints. The selected backend's
decreasing-time dense interpolation can panic on a rounded step boundary, so
backward segments are evaluated through the equivalent increasing coordinate
`tau = segment_start - time` rather than handed to that backend path. At least
two samples are required. Backward histories list the final segment first,
matching propagation order. `evaluate_zoh_mismatch_batch()` evaluates
validated Rust legs in input order.

## Python

`pykep_rust.ZohLeg` accepts a `ZohModel` value: `Kepler`, `Cr3bp`,
`Equinoctial`, or `SolarSail`. The constructor validates dynamic state,
control, and constant dimensions for that selection. Mismatch, Jacobian, and
history evaluation release the Python GIL.

The ZOH-leg batch methods clone the small immutable leg descriptors, release
the GIL once, and return results in input order. `workers=0` uses the shared
pool, `workers=1` is serial, and a larger value selects a cached pool of that
exact size.
