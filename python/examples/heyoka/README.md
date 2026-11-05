# Search with pykep-rust, verify with heyoka

This optional example shows the usual division of labour in a trajectory
competition. A fast, simple model searches millions of candidates. A
slower, more faithful model then checks the few candidates that matter.

| Stage | Tool | Model |
|---|---|---|
| 1. Global search | `pykep_rust` gym and `fcmaes_rust` BiteOpt restarts | Patched conics: Lambert arcs between planet centers, instantaneous flybys, JPL low-precision ephemerides |
| 2. Verification | [heyoka](https://github.com/bluescarni/heyoka.py) Taylor integration | VSOP2013 planets evaluated inside the equations of motion, the Sun plus third-body perturbations from the other planets of the route, closest-approach events, variational equations |

heyoka is not a dependency of pykep-rust. It is only needed for this
example and is installed separately:

```bash
python -m pip install pykep-rust fcmaes-rust heyoka
python python/examples/heyoka/high_fidelity_check.py            # search, then check
python python/examples/heyoka/high_fidelity_check.py --stored   # check the stored best route
```

The first run JIT-compiles one integrator pair per leg, a few seconds each.
heyoka keeps compiled code in an on-disk cache, so later runs start faster.
The search takes about three seconds, and the whole run takes about half a
minute on two cores.

## What the check does

For each leg of the Cassini1 route (Earth, Venus, Venus, Earth, Jupiter,
Saturn), the script:

1. starts at the departure planet's VSOP2013 position. It keeps the
   patched-conic hyperbolic excess velocity, measured relative to the
   VSOP2013 planet;
2. propagates to the patched-conic arrival epoch, once with the Sun alone
   and once with the other planets switched on;
3. records the real closest approach to the target with a heyoka event, an
   exact root of `(r − r_planet)·(v − v_planet) = 0` found from the Taylor
   polynomials;
4. retargets the planet center with a damped Newton iteration on the
   departure velocity and the arrival time. The 3 × 3 Jacobian comes from
   heyoka's first-order variational equations.

The leg's own departure and arrival planets exert no gravity in this model,
as in patched conics. Modelling their pull would need the flyby hyperbolas
and B-plane targeting, which is beyond a short example.

## Result

With the stored decision, which needs 2570.0 m/s in patched conics:

```text
leg                 days      VSOP km  +planets km  closest km      cond  fix m/s  shift h    left km
earth-venus        136.5        4,442        4,589       4,008   1.0e+02     0.45     0.00      0.109
venus-venus        449.4       70,566      121,666      73,922   1.2e+06   361.88    -0.01 19,137.463  not retargeted
venus-earth         57.4        4,271        4,788         774   2.2e+00     0.12     0.08      0.056
earth-jupiter      948.5      265,596      236,931     129,563   2.2e+01     0.80    12.74      0.028
jupiter-saturn    4434.6   13,584,859   13,463,846  13,419,493   1.4e+01    10.27   -83.52      0.000
patched conics: 2570.0 m/s; retargeting 4 of 5 legs in the high-fidelity model adds 11.6 m/s
```

What it shows:

- **Ephemerides dominate the outer legs.** JPL's low-precision model places
  Jupiter and Saturn about a million and a few million kilometres from
  VSOP2013 here. That turns the Jupiter–Saturn leg into a miss of 13.6
  million km before any perturbation. Retargeting costs only about 10 m/s,
  plus an arrival 83 hours earlier, because a long leg is cheap to steer.
- **Third-body perturbations matter on the inner legs.** On the Venus–Venus
  leg they add 51,000 km to the miss. On the Earth–Jupiter leg they partly
  cancel the ephemeris error.
- **The near-resonant Venus–Venus leg cannot be fixed by a departure burn.**
  It lasts almost exactly two Venus years. The spacecraft returns to nearly
  the point it left, and the Jacobian has a condition number of 1.2 × 10⁶.
  After 362 m/s of correction it still misses by 19,000 km. A real redesign
  would retarget the previous flyby, or re-optimize the route in the
  high-fidelity model. This is the fragile leg the
  [astrodynamics introduction](../../../docs/astrodynamics-introduction.md#6-multi-leg-missions)
  points out. The verification stage exists to catch exactly this.

## Caveats

- VSOP2013 body 3 is the Earth–Moon barycenter, as is JPL's low-precision
  "earth". VSOP2013 time is TDB, and the script uses pykep's MJD2000 epochs
  directly; the difference of about a minute is irrelevant here.
- `--vsop-threshold` trades accuracy for compile time. The default `1e-6`
  truncates the series noticeably; `1e-8` is closer to the full theory and
  takes longer to compile.
- The retargeted legs are independent. Each uses the patched-conic departure
  epoch, so the corrected arrival times are not fed into the next leg.

## The heyoka features used

- **Expression system:** a custom force model with time-dependent VSOP2013
  positions inside the equations of motion.
- **Runtime parameters (`hy.par`):** switch the perturbers on and off
  without recompiling.
- **Non-terminal events with dense output (`update_d_output`):** the exact
  time and distance of closest approach.
- **Variational equations (`var_ode_sys`):** the departure-velocity
  Jacobian.
- **Compiled functions (`cfunc`):** evaluate the same VSOP2013 expressions
  for targets and starting states.

For a user-defined model, pykep-rust offers less. In Rust, its DOP853
backend integrates a custom `DynamicsModel` with user events. It can
evaluate an ephemeris inside the right-hand side, and gives first-order
sensitivities when you implement the model Jacobians by hand. It has no
Taylor integration for custom models and no automatically derived or
higher-order variational equations, and none of this is available from
Python. That is why the verification stage uses heyoka.
