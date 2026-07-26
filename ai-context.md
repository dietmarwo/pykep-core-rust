# AI context for solving astrodynamics problems with pykep-rust

This file is operational context for an AI that must design, implement, and
validate an astrodynamics calculation with this repository. It is not a claim
that one model, ephemeris, propagator, or transcription is universally best.
Choose the smallest physical model that captures the user's decision, state
every convention, and validate the result at the accuracy level the decision
requires.

This repository is an independent native Rust port of the numerical C++
library in pykep/kep3 3.0.1. `pykep-core` contains the astrodynamics
implementation and has no C or C++ runtime dependency. `pykep-py` is a thin,
unpublished PyO3 conversion and exception layer over the same core. It does
not duplicate the numerical algorithms.

The public distribution surfaces have deliberately distinct names:

```text
cargo add pykep-core                -> use pykep_core
python -m pip install pykep-rust    -> import pykep_rust
```

The current synchronized release is 0.1.1. Rust 1.88 is the tested minimum
toolchain. Published wheels target CPython 3.11 through 3.13 and need neither a
Rust toolchain nor a C/C++ compiler. Building the Python source distribution
does require Rust. The default Rust feature embeds the VSOP2013 coefficient
data; use `default-features = false` when only the smaller numerical core,
Keplerian ephemerides, and JPL low-precision ephemerides are needed.

Do not infer complete pykep ecosystem parity from the numerical surface.
SPICE kernels, TLE parsing, Python-defined ephemeris providers, plotting,
trajectory-optimization UDPs, gym helpers, arbitrary heyoka expression
graphs, and several legacy aliases are intentionally unavailable. Consult
[`docs/python-migration.md`](docs/python-migration.md) before promising
drop-in compatibility with `pykep` or `kep3`.

## Required workflow for the AI

Before choosing an API or writing code, create a problem card. Inspect the
user's code and data where possible. Ask only for facts that cannot be inferred
and would materially change the calculation.

| Question | Why it matters |
|---|---|
| What output is required: state, transfer velocities, delta-v, constraints, gradients, or a trajectory history? | Different APIs can solve related physical problems but return materially different contracts. |
| What bodies and central force model are involved? | Two-body, CR3BP, bicircular, ephemeris, and low-thrust models make different assumptions. |
| What is the origin and reference frame of every state? | Heliocentric ICRF, J2000 ecliptic, inertial Cartesian, and synodic rotating states cannot be mixed directly. |
| What epoch scale is supplied: MJD2000, MJD, JD, UTC, TT, or TDB? | The library performs arithmetic Julian-date conversion but does not convert physical time scales. |
| What units are used for length, time, mass, velocity, force, and `mu`? | Most APIs allow normalized units, but every quantity in one calculation must be consistent. |
| Which state or element representation is used? | Classical elements are singular for circular/equatorial cases; modified equinoctial elements have prograde/retrograde exclusions. |
| What fidelity and validity interval are required? | JPL low precision, VSOP2013, constructed Keplerian motion, and integrated dynamics have different error sources and validity ranges. |
| Is the trajectory impulsive, ballistic, continuously controlled, or piecewise controlled? | Lambert, flyby, Sims–Flanagan, ZOH, and Pontryagin models represent different control assumptions. |
| Are multiple Lambert revolutions or both transfer directions allowed? | Branch enumeration and direction directly affect feasible transfers and delta-v. |
| Are derivatives required, and with respect to which variables? | STMs, analytic element/leg Jacobians, and numerically differentiated dynamics have different layouts and accuracy. |
| What tolerances, maximum step, and failure policy are acceptable? | Integrator tolerances control local error estimates; they are not a guarantee on trajectory or derivative error. |
| Is this a scalar call, a large ordered batch, or a threaded application? | Python throughput generally requires explicit NumPy batches; native objects are reusable across threads. |
| What independent reference or invariant can validate the result? | A successful solve is not evidence that units, frame, branch, or physical model are correct. |

Do not proceed with an unstated frame, ambiguous epoch scale, or mixed unit
system. If the user supplies insufficient metadata, make the smallest explicit
assumption only when it cannot affect the decision; otherwise ask.

For every completed solution, report the problem card, selected physical
model, rejected alternatives, units, frame, epoch interpretation, state
ordering, non-default solver settings, branch choices, validation method, and
observed residuals or invariant errors. Code without these facts is not a
reproducible astrodynamics result.

## First choose the physical model

Use the simplest model justified by the required accuracy and dynamics.

1. Use algebraic anomaly, element, transfer, encoding, or flyby functions when
   the requested quantity is exactly that local calculation.
2. Use two-body Lagrangian or universal-variable propagation for point-mass
   central gravity with no perturbations.
3. Use `LambertProblem` for an impulsive two-point boundary-value transfer
   between known positions and a positive flight time.
4. Use a built-in ephemeris when body states, rather than integrated spacecraft
   motion, are required:
   - constructed `KeplerianEphemeris` for an ideal conic;
   - `JplLowPrecision` for approximate 1800–2050 planetary design;
   - `Vsop2013` for its documented heliocentric analytical theory.
5. Use evaluated Kepler dynamics when adaptive integration is required for a
   two-body state or when a common model/integrator interface matters.
6. Use CR3BP for normalized rotating-frame motion under two circular
   primaries. Use the bicircular problem only when its periodic third-body
   approximation is part of the intended model.
7. Use ZOH dynamics for a known piecewise-constant control schedule.
8. Use a Sims–Flanagan leg for a finite-impulse low-thrust transcription with
   mismatch and throttle constraints.
9. Use a generic ZOH leg for continuous piecewise-constant controls and
   endpoint/control/time-grid sensitivities.
10. Use Pontryagin dynamics only for an indirect mass- or time-optimal
    formulation with meaningful costates and normalization.

Do not use a higher-fidelity-looking model by name alone. A rotating
normalized CR3BP state is not a more accurate replacement for a heliocentric
inertial state. An ephemeris is not a spacecraft force model. A Lambert arc is
not a finite-thrust trajectory. An integrator with a small tolerance does not
repair missing physics.

When the requested model requires atmospheric drag, oblateness beyond the
available constants, solar radiation pressure outside the ideal sail model,
high-precision integrated planetary ephemerides, SPICE frames, maneuvers with
events not represented here, or another unsupported force, use an appropriate
external tool or implement and validate a new `DynamicsModel` in Rust. State
that expansion explicitly rather than approximating it silently.

## Fast capability-selection decision tree

1. If the user is converting time, anomalies, or orbital representations, use
   the corresponding foundation API and round-trip the result.
2. If one initial Cartesian state must move under point-mass gravity:
   - use `propagate_lagrangian` for the normal fast two-body path;
   - use `propagate_universal` when the exact parabolic limit or an independent
     universal-variable path matters;
   - use `propagate_lagrangian_with_stm` when the initial-state sensitivity is
     required;
   - do not use `propagate_keplerian` for circular or equatorial states,
     because its classical-element reference path is singular there.
3. If positions are known at two epochs but velocities are not, solve Lambert.
   Enumerate the requested direction and revolution families, then evaluate
   every returned branch against the actual departure and arrival velocities.
4. If body states are needed:
   - choose the provider from frame, interval, and accuracy requirements;
   - query provider metadata and optional capabilities rather than inventing
     missing values;
   - transform frames externally before combining providers with different
     frame contracts.
5. If an impulsive circular-orbit estimate is enough, use `hohmann` or
   `bielliptic`. If a gravity assist is involved, use the flyby API and enforce
   its feasibility constraints.
6. If continuous integration is needed:
   - inertial point-mass: `KeplerDynamics`;
   - normalized synodic two-primary motion: `Cr3bpDynamics`;
   - normalized bicircular motion: `BcpDynamics`;
   - piecewise controls: a ZOH model or leg.
7. If optimizing a low-thrust leg:
   - use fixed-duration Sims–Flanagan when midpoint impulses and its analytic
     mismatch Jacobian fit the transcription;
   - use the alpha leg for explicit non-uniform segment durations, noting that
     it has no mismatch Jacobian;
   - use `ZohLeg` for continuous ZOH dynamics and time-grid derivatives;
   - use Pontryagin models for indirect optimal control, not as a generic
     replacement for a direct transcription.
8. If many independent rows are evaluated from Python, find an explicit batch
   API before writing a Python loop.

## Choose the integration surface

Prefer `pykep-core` for Rust applications, optimization objectives, custom
dynamics, and high-throughput native loops. Fixed-size arrays make state and
Jacobian layouts explicit, Rust errors preserve numerical categories, and the
native core avoids Python conversion overhead.

Use `pykep_rust` when the surrounding application is Python or NumPy. The
package is a typed low-level numerical API, not a compatibility facade for all
upstream names. Consult the shipped `_pykep_rust.pyi` for authoritative Python
signatures and [`docs/python-api.md`](docs/python-api.md) for shape,
ownership, error, and default contracts.

Python batch APIs accept `float64` NumPy arrays with documented ranks, preserve
input order, return newly owned output, and release the GIL around native work.
They do not create an implicit thread pool. Strided and read-only input arrays
are accepted. A Python loop around cheap scalar native calls can spend more
time crossing the extension boundary than doing astrodynamics.

Native ephemeris providers and validated leg objects are immutable or
thread-safe and can be reused. Give each concurrent operation isolated output
and optimizer state. Do not add a broad mutex around numerical evaluation.

## Numerical conventions

Unless an API explicitly says otherwise:

- scalar arithmetic is binary64;
- Cartesian state is `[x, y, z, vx, vy, vz]`;
- classical elements are `[a, e, i, Ω, ω, ν]`;
- modified equinoctial elements are `[p, f, g, h, k, L]`;
- angles are radians;
- SI examples use metres, seconds, kilograms, and `mu` in `m³/s²`;
- normalized calculations are valid when every input uses the same normalized
  length, time, and mass system;
- ephemeris epochs are MJD2000 days;
- propagation durations are caller-consistent time units, seconds for SI
  states and SI `mu`;
- matrices are row-major and output-by-input;
- an STM is `∂state_final / ∂state_initial`;
- deterministic batches preserve row order;
- NaN and infinity are rejected.

Never attach SI labels to normalized output. Never pass an MJD2000 day count
as a propagation duration in seconds. Never infer the state frame from its six
numbers.

## Capability comparison

| Capability | Primary Rust surface | Python surface | Main caution |
|---|---|---|---|
| Epochs and Julian arithmetic | `time::epoch::Epoch`, `time::julian` | `Epoch`, `*_to_*` functions | Arithmetic dates do not imply UTC/TT/TDB conversion. |
| Elliptic/hyperbolic anomalies | `astro::anomalies` | descriptive scalar functions and selected batches | Respect elliptic versus hyperbolic eccentricity domains. |
| Cartesian/classical/MEE conversion | `astro::elements` | scalar and `N × 6` batch functions | Classical angles are singular for circular/equatorial states. |
| Two-body propagation | `astro::propagation` | scalar, grid, batch, and STM functions | Units must be consistent; the grid is relative to its first time. |
| Basic impulsive transfers | `astro::transfers` | `hohmann`, `bielliptic` | Assumes coplanar circular-orbit transfer geometry. |
| Time encodings and MIMA | `astro::encodings`, `astro::mima` | descriptive encoding functions, `mima`, `mima2` | Encodings have strict domains; MIMA is an approximation, not a propagated low-thrust solution. |
| Lambert | `astro::lambert::LambertProblem` | `LambertProblem` | Direction, branch, and endpoint-velocity reduction are caller decisions. |
| Flyby | `astro::flyby` | `flyby_*` functions | Enforce feasibility and periapsis/body conventions. |
| Ephemerides | `ephemeris` providers and `Ephemeris` | `Planet` constructors | Provider frames, bodies, validity, and accuracy differ. |
| Evaluated dynamics | `dynamics`, `integration` | RHS, propagation, and STM functions | CR3BP/BCP are normalized rotating-frame models. |
| ZOH schedules | `dynamics::zoh` | `propagate_zoh_*` | Switch ownership, row widths, and model constants are explicit. |
| Pontryagin dynamics | `dynamics::pontryagin` | `pontryagin_*`, `Optimality` | Costates and normalization belong to the user's indirect formulation. |
| Sims–Flanagan legs | `leg::SimsFlanagan*` | matching immutable classes | Constraint and Jacobian ordering must match the optimizer. |
| Generic ZOH legs | `leg::ZohLeg` and aliases | `ZohLeg`, `ZohModel` | Numerical model Jacobians limit derivative accuracy. |

## Epochs, calendars, and anomalies

`Epoch` stores a microsecond-granular offset and requires an explicit numeric
scale for numeric construction. Prefer MJD2000 or calendar/ISO construction
when microsecond input resolution matters; binary64 Julian dates have roughly
40 microseconds of spacing near J2000.

`Epoch` arithmetic with numeric days is checked. Use `add_seconds()` and
`seconds_since()` when seconds are the intended unit. Julian conversion
functions are arithmetic:

```text
JD 2451544.5 = MJD 51544.0 = MJD2000 0.0
```

They do not convert UTC, TAI, TT, or TDB and do not insert leap seconds.
Ephemeris documentation states the expected scale separately. If a user
supplies UTC observation time for a TDB theory, time-scale conversion is an
external preprocessing requirement.

For anomaly work, first classify the conic. Elliptic conversions require the
elliptic eccentricity domain; hyperbolic conversions require `e > 1`. Test
round trips modulo the appropriate angular convention, and test difficult
inputs near circular, parabolic, high-eccentricity, and large-mean-anomaly
limits when relevant.

## Elements, singularities, and Jacobians

Classical elements provide familiar geometry but their node and periapsis
angles are undefined for equatorial and circular states. The Rust API reports
these singularities rather than returning an arbitrary convention. Use
modified equinoctial elements for those cases:

- the prograde convention excludes inclination `π`;
- the retrograde convention excludes inclination zero.

Do not compare angular elements with a raw absolute difference across a
`2π` wrap. Validate conversions by reconstructing Cartesian state and compare
position and velocity with scale-aware tolerances.

Element Jacobians are `6 × 6`, with output components in rows and input
components in columns. When transforming gradients or costates, verify the
direction of the mapping and use a finite-difference check at a nonsingular
state. An integrator tolerance does not validate an element Jacobian.

## Two-body propagation and STMs

`propagate_lagrangian` and `propagate_universal` support forward and backward
time for elliptic and hyperbolic conics. Universal variables also cover the
exact parabolic limit. Both use caller-consistent units. Validate important
arcs with at least:

- specific orbital energy;
- angular momentum;
- forward/backward recovery;
- agreement between independent propagation paths away from singular limits.

`propagate_keplerian` converts through classical elements and advances mean
anomaly. It is a slower reference path and is undefined for circular or
equatorial states. Do not choose it merely because its name sounds more
general.

`propagate_lagrangian_grid` interprets all grid entries relative to the first
entry. The output at the first grid value is the input state, even when that
first value is not zero.

For linearized work, the analytic Lagrangian and Reynolds STMs are
output-by-input. Check a requested STM against scale-adjusted central
differences and composition over two sub-arcs. Large state-component scale
differences require component-aware perturbations.

## Lambert, impulsive transfers, and flybys

`LambertProblem` solves every feasible branch through a requested maximum
number of complete revolutions. Returned order is deterministic:

```text
zero revolution,
1 revolution left, 1 revolution right,
2 revolutions left, 2 revolutions right, ...
```

The requested maximum is a search limit, not a promise that every family is
feasible at the supplied time. Always inspect the returned solutions.

`clockwise = false` chooses prograde motion as viewed from positive `z`;
`clockwise = true` reverses that convention. Collinear endpoints are rejected
because this automatic direction rule is undefined. If the physical plane is
not the library's `xy` plane, transform coordinates or make the convention
explicit before solving.

A Lambert solution returns transfer departure and arrival velocities. It does
not know the endpoint bodies' velocities and therefore does not return mission
delta-v. For each branch compute, in a common frame,

```text
delta_v = |v_transfer_departure - v_body_departure|
        + |v_body_arrival - v_transfer_arrival|
```

Apply any capture, launch, parking-orbit, or powered-flyby model separately.
When direction is not predetermined, evaluate both clockwise values.

`hohmann` and `bielliptic` are ideal coplanar circular-orbit estimates, not
general Lambert replacements. Flyby constraints and Jacobians use explicit
incoming/outgoing excess velocities, body `mu`, and periapsis. Treat the
returned feasibility constraints as constraints, not as an optional
diagnostic, and keep all vectors in the same body-centered frame.

The alpha and eta time encodings are reversible parameterizations, not
dynamics:

- alpha decoding requires every value strictly inside `(0, 1)` and produces
  positive durations summing to a supplied positive total time;
- eta decoding accepts values in `[0, 1]` and allocates each duration from the
  remaining positive maximum-time budget.

Round-trip the chosen encoding and enforce its domain in any optimizer
boundary. `mima` and `mima2` return a maximum-initial-mass approximation and a
characteristic acceleration from endpoint impulse information. Use them for
screening or initialization under their documented assumptions, not as proof
that a low-thrust trajectory exists. Validate promising candidates with an
actual leg or dynamics model. The upstream `mima_from_hop` and
`mima2_from_hop` helpers are unavailable.

## Ephemeris selection

| Provider | Bodies | Epoch/frame contract | Intended use |
|---|---|---|---|
| `KeplerianEphemeris` | Caller-defined | MJD2000 epochs converted to elapsed seconds; state and `mu` use a caller-defined length unit with seconds as the time unit, in the caller's frame | Ideal conic propagation from a reference state or elements |
| `JplLowPrecision` | Mercury through Neptune; `earth` is the Earth–Moon barycentre | Open interval `-73048 < MJD2000 < 18263`; source coefficients use JDTDB; heliocentric mean ecliptic/equinox J2000; SI output | Approximate 1800–2050 mission design |
| `Vsop2013` | Mercury through Pluto; Earth–Moon barycentre is `earth_moon` | MJD2000 interpreted as TDB; heliocentric ICRF; SI output | Analytical planetary theory with documented fit/error limits |

JPL low precision and VSOP2013 do not return states in the same frame. Do not
subtract or compare their vectors component by component without a frame
transformation. Their Earth identifiers also represent the Earth–Moon
barycentre, not the geocentre.

The default VSOP2013 threshold is `1e-5`. The embedded data supports thresholds
down to `1e-9`; smaller values are rejected because those coefficients are not
shipped. The theory was fitted over 1890–2000. It has no artificial date
cutoff, but extrapolation accuracy degrades. JPL low precision enforces its
1800–2050 validity interval and is explicitly unsuitable for precision
navigation.

Use `Vsop2013::available()` or `Planet.vsop2013_available()` before depending
on the optional data feature. Query provider metadata and
`has_acceleration()` rather than treating missing values as zero. An
unsupported capability is a typed error.

## Evaluated dynamics and adaptive integration

The three six-state evaluated models are:

- `KeplerDynamics`: inertial Cartesian point-mass motion;
- `Cr3bpDynamics`: normalized CR3BP synodic-frame motion;
- `BcpDynamics`: normalized time-dependent bicircular motion.

CR3BP and BCP place their primaries at `(-mu, 0, 0)` and
`(1 - mu, 0, 0)`. The customary mass ordering has `mu <= 0.5`, although the
mathematical API accepts `[0, 1]`. BCP parameters are
`[mu, mu_sun, rho_sun, omega_sun]`.

Each model supports RHS evaluation, propagation, and propagation with an STM.
The default DOP853 options use relative and absolute tolerances `1e-12`, no
maximum step, a 100,000-step limit, and 100 consecutive rejections. These are
defaults, not universal settings. Tighten or bound the step based on duration,
scale, close approaches, switching behavior, and required invariant drift.

Validate Kepler propagation with energy/angular momentum. Validate CR3BP with
the Jacobi constant when the modeled trajectory avoids singular primaries.
For BCP, use reversal, convergence under tighter tolerances/steps, and an
independent reference because the model is time-dependent.

## Zero-order-hold and Pontryagin dynamics

ZOH schedules require `S + 1` strictly increasing boundaries and exactly `S`
control rows. Segment `i` owns `[t_i, t_(i+1))`; the final node belongs to the
last segment. Forward and backward propagation deliberately choose controls
on opposite sides of a switch so no integration step crosses a discontinuity.

The ZOH models use these state/control layouts:

| Model | State | Control | Constants |
|---|---|---|---|
| Kepler | `[x,y,z,vx,vy,vz,m]` | `[thrust,ix,iy,iz]` inertial | `[c]` |
| CR3BP | `[x,y,z,vx,vy,vz,m]` | `[thrust,ix,iy,iz]` synodic | `[c,mu]` |
| Equinoctial | `[p,f,g,h,k,L,m]` | `[thrust,ir,it,in]` RTN | `[c]` |
| Solar sail | `[x,y,z,vx,vy,vz]` | `[cone,clock]` | `[c]` |

The low-thrust ZOH equations preserve the upstream normalized mass-flow
contract. Do not substitute a physical exhaust-velocity interpretation
without deriving the required normalization.

Pontryagin models use 14-component state/costate vectors. Cartesian order is:

```text
[x,y,z,vx,vy,vz,m,lx,ly,lz,lvx,lvy,lvz,lm]
```

The modified-equinoctial form is:

```text
[p,f,g,h,k,L,m,lp,lf,lg,lh,lk,lL,lm]
```

Mass-optimal parameters are
`[mu, maximum_thrust, exhaust_velocity, barrier, lambda0]`; time-optimal
parameters are `[mu, maximum_thrust, exhaust_velocity]` with implicit
`lambda0 = 1`. A zero primer direction is singular and returns an error. Do
not invent a direction or suppress that failure.

Canonical costate rates use forward-mode differentiation, while complete
model state/parameter Jacobians use fixed-size centered differences. The
recorded first-order validation tolerance is much looser than nominal
trajectory tolerance. Validate gradients independently before using them as a
tightly converged optimization stopping criterion.

## Sims–Flanagan and generic ZOH legs

A fixed `SimsFlanaganLeg` has seven endpoint components per side:

```text
[x,y,z,vx,vy,vz,mass]
```

For `N` chronological throttle vectors, `floor(N * cut)` segments propagate
forward and the remainder propagate backward. Each segment applies a finite
impulse at its midpoint. Mismatch order is position, velocity, then mass:

```text
[forward_rx - backward_rx, ...,
 forward_vz - backward_vz,
 forward_mass - backward_mass]
```

Throttle constraint `i` is `dot(throttle_i, throttle_i) - 1`, so feasibility
uses `<= 0`. The fixed-leg mismatch Jacobian groups are:

| Group | Shape | Column order |
|---|---:|---|
| departure | `7 × 7` | departure state and mass |
| arrival | `7 × 7` | arrival state and mass |
| controls/time | `7 × (3N + 1)` | chronological flattened controls, then time of flight |

The alpha leg accepts direct non-negative segment durations. Their sum is not
automatically required to equal the configured flight time.
`from_time_weights` is the explicit normalized alternative. The alpha leg has
no mismatch Jacobian; do not fabricate one from the fixed leg.

`ZohLeg` represents continuous piecewise-constant control over a strictly
increasing time grid. Its mismatch is the component-wise difference between
forward and backward states at the cut. Its four Jacobian groups are initial
state, final state, flattened chronological controls, and all time-grid nodes.
The built-in model Jacobians use centered differences, and validated
end-to-end derivative tolerances reach `3e-5`. Solver tolerances of `1e-12`
must not be reported as derivative accuracy.

Leg objects validate and copy their configuration. Reuse them for repeated
constraint evaluation. Use the explicit ordered batch mismatch API for Python
throughput; it releases the GIL but does not create worker threads.

## Batch performance and concurrency

Use scalar APIs for clarity and small calculations. Use explicit batches for
large independent workloads:

- anomaly and Stumpff batches for long scalar arrays;
- `N × 6` element conversion batches;
- `N × 6` two-body propagation batches with one time per row;
- ordered ephemeris state batches;
- ordered ZOH-leg mismatch batches.

The release-wheel benchmark measured substantial Python-boundary savings from
batching, but those measurements are orientation, not a universal speed claim.
Benchmark the user's real shape in release mode.

Batch APIs preserve order and allocate output. They do not parallelize
internally. In Rust, parallelize at the application or optimizer layer when
each row is expensive enough. In Python, native batches release the GIL;
multiple Python threads are useful only after measuring the actual workload.
Avoid nested full-size thread pools.

Do not benchmark debug Rust builds, first-time construction mixed with warm
evaluation, or I/O inside the timed loop. Report initialization and steady
state separately for ephemerides and other cached data.

## Errors and failure policy

Rust numerical functions return `pykep_core::Result<T>`. Important error
categories include invalid input, convergence failure, singular geometry,
integration failure, unsupported capability, and non-finite output.

Python uses `ValueError` or `TypeError` for invalid values, shapes, and typed
NumPy buffer mismatches. Numerical failures derive from `PykepError`:

- `ConvergenceError`;
- `SingularGeometryError`;
- `IntegrationError`;
- `UnsupportedCapabilityError`.

Do not replace every error with NaN. A typed singularity or unsupported model
often means the formulation must change. In optimization code, convert an
expected invalid candidate to a documented finite penalty at the objective
boundary, while logging or counting failure categories. Unexpected errors
should fail the run.

## Validation and reporting procedure

Before declaring a result correct:

1. Unit-test units, state ordering, epoch scale, frame, and known reference
   values.
2. Check input/output finiteness and physical domains.
3. Round-trip anomaly and element conversions where applicable.
4. Check invariants, reversal, or endpoint reconstruction for propagation.
5. Check Lambert branch endpoints and independently compute mission delta-v.
6. Check ephemeris provider interval, frame, body identity, and expected
   accuracy.
7. Compare adaptive propagation at tighter tolerances and smaller maximum
   steps; report convergence, not only one solve.
8. Check every requested Jacobian against scale-aware central differences at
   representative nonsingular points.
9. Re-evaluate an optimizer's final trajectory independently from its
   objective wrapper and report raw constraints.
10. Compare with a trusted independent model or the pinned upstream behavior
    when the decision is high consequence.

Use release mode for timings and record:

- package versions and Cargo features;
- hardware and toolchain;
- model and all physical constants;
- units, frames, and epoch interpretation;
- solver tolerances and step limits;
- batch size and worker topology;
- initialization policy;
- sample count, distribution statistic, and variability.

This library is suitable for numerical astrodynamics and mission-design
experimentation under its documented models. It does not certify navigation,
flight safety, or mission feasibility. High-stakes results require independent
software, authoritative ephemerides/frames, uncertainty analysis, and domain
review.

## Minimal implementation patterns

### Rust two-body propagation

```rust
use pykep_core::astro::propagation::propagate_lagrangian;

fn main() -> pykep_core::Result<()> {
    // Normalized circular orbit: radius = speed = mu = 1.
    let initial = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let final_state =
        propagate_lagrangian(&initial, core::f64::consts::FRAC_PI_2, 1.0)?;
    assert!((final_state[1] - 1.0).abs() < 1e-12);
    Ok(())
}
```

### Rust Lambert branch enumeration

```rust
use pykep_core::astro::lambert::{LambertPath, LambertProblem};

fn main() -> pykep_core::Result<()> {
    let problem = LambertProblem::new(
        [1.0, 0.0, 0.0],
        [0.2, 1.1, 0.3],
        20.0,
        1.0,
        false,
        2,
    )?;
    assert_eq!(problem.solutions()[0].path, LambertPath::ZeroRevolution);
    for solution in problem.solutions() {
        println!(
            "{} rev {:?}: {:?} -> {:?}",
            solution.revolutions,
            solution.path,
            solution.departure_velocity,
            solution.arrival_velocity
        );
    }
    Ok(())
}
```

### Python ordered propagation batch

```python
import numpy as np
import pykep_rust as pk

states = np.tile([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], (4096, 1))
times = np.linspace(0.0, 1.0, len(states), dtype=np.float64)
result = pk.propagate_lagrangian_batch(states, times, 1.0)
assert result.shape == (4096, 6)
```

### Python ephemeris with explicit contract

```python
import pykep_rust as pk

# MJD2000 interpreted as TDB; heliocentric ICRF state in SI units.
earth_moon = pk.Planet.vsop2013("earth_moon", threshold=1e-5)
state = earth_moon.state(0.5)
assert len(state) == 6
```

Copy maintained patterns from [`docs/examples.md`](docs/examples.md) and the
paired source under `examples/src/bin/` and `python/examples/` instead of
rebuilding conventions from memory.

## Failure diagnosis

| Symptom | Likely cause | First action |
|---|---|---|
| State magnitude is wrong by orders of magnitude | Days passed as seconds, kilometres mixed with metres, or normalized output labelled SI | Write the complete unit system beside every input and recompute one dimensional scale. |
| Planet-to-planet vector looks implausible | Different origins/frames or Earth versus Earth–Moon barycentre confusion | Print provider names and frame contracts; transform both states to one frame. |
| Epoch is shifted by hours or seconds | MJD/MJD2000/JD confusion or missing UTC/TT/TDB conversion | Check the numeric epoch constructor and perform time-scale conversion externally. |
| Classical conversion fails near a valid orbit | Circular/equatorial angular singularity | Use modified equinoctial elements and the correct prograde/retrograde convention. |
| Lambert returns fewer branches than requested | Flight time cannot support every requested revolution family | Inspect returned `solutions`; increase time only if physically allowed. |
| Lambert fails for aligned endpoints | Automatic prograde/clockwise plane direction is undefined | Reformulate the geometry or provide an explicit coordinate-plane convention externally. |
| Transfer delta-v is unexpectedly zero or huge | Transfer velocity was not differenced from body velocity in the same frame | Recompute both endpoint impulses explicitly. |
| CR3BP trajectory diverges or approaches a singularity | Wrong normalization/frame, close primary encounter, or inadequate step control | Verify nondimensionalization and Jacobi constant; cap maximum step and test convergence. |
| Adaptive solve reaches its step limit | Tolerance too tight, singular approach, discontinuity crossed, or maximum step inappropriate | Inspect the typed integration failure and trajectory scale before increasing limits. |
| ZOH result changes at a switch | Boundary ownership or control order is wrong | Check the strictly increasing grid and chronological control rows. |
| Gradient disagrees with finite differences | Matrix orientation, perturbation scale, nonsmooth point, or numerical model Jacobian limit | Check output-by-input order and sweep central-difference step sizes. |
| Sims–Flanagan optimizer stays infeasible | Mismatch/throttle constraint order or sign is wrong | Print all seven mismatches and verify throttle feasibility is `u·u - 1 <= 0`. |
| Python scalar loop is slow | Extension-boundary overhead dominates cheap native work | Use the explicit NumPy batch API and time end-to-end throughput. |
| Optional ephemeris construction fails | VSOP2013 feature omitted or unsupported threshold/body requested | Query availability, minimum threshold, and supported-body lists. |

## Repository references

- [`README.md`](README.md): repository scope, layout, and smoke commands.
- [`crates/pykep-core/README.md`](crates/pykep-core/README.md): native crate
  overview and Rust landing-page example.
- [`python/README.md`](python/README.md): Python package landing page.
- [`docs/conventions.md`](docs/conventions.md): authoritative units, shapes,
  element conventions, branch ordering, and error behavior.
- [`docs/examples.md`](docs/examples.md): Rust/Python quick starts and complete
  runnable capability matrix.
- [`docs/python-api.md`](docs/python-api.md): Python units, buffers, ownership,
  concurrency, errors, and defaults.
- [`docs/python-migration.md`](docs/python-migration.md): upstream name map,
  intentional differences, and unsupported ecosystem areas.
- [`docs/ephemerides.md`](docs/ephemerides.md): JPL low-precision and VSOP2013
  frames, bodies, intervals, thresholds, and accuracy.
- [`docs/dynamics.md`](docs/dynamics.md): Kepler, CR3BP, BCP, STMs, parameters,
  and validation tolerances.
- [`docs/zero-order-hold.md`](docs/zero-order-hold.md): control layouts, switch
  ownership, schedule propagation, and sensitivities.
- [`docs/pontryagin.md`](docs/pontryagin.md): augmented state, parameters,
  controls, Hamiltonians, and derivative limitations.
- [`docs/low-thrust-legs.md`](docs/low-thrust-legs.md): Sims–Flanagan
  transcription, constraints, cuts, and Jacobian layouts.
- [`docs/zoh-leg.md`](docs/zoh-leg.md): generic continuous ZOH leg, histories,
  model aliases, and four sensitivity groups.
- [`docs/validation.md`](docs/validation.md): golden-oracle provenance,
  independent properties, tolerances, and Python surface evidence.
- [`docs/status.md`](docs/status.md): evidence-backed implementation matrix.
- [`docs/performance.md`](docs/performance.md): benchmark methodology and
  interpretation.
- [`docs/source-map.md`](docs/source-map.md): pinned upstream implementation
  coverage and explicit gaps.
- [`docs/development.md`](docs/development.md): formatting, lint, test,
  documentation, MSRV, coverage, fuzz, and benchmark commands.
- [`RELEASE.md`](RELEASE.md): registry artifacts and clean-consumer checks.
- Generated Rust API documentation: `cargo doc -p pykep-core --open`.
- Shipped Python signatures: `python/pykep_rust/_pykep_rust.pyi`.

When code and this guide disagree, treat the current public Rust API, Python
stub, tests, and model-specific documentation as authoritative. Update this
guide with the implementation and record the reason for any behavioral
change.
