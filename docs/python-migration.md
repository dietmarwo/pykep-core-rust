# Python migration from `kep3`

The native package is named `pykep_rust`, so it can be installed beside
`pykep`/`kep3` without import collisions. Phase 16 deliberately does not add a
compatibility facade: a partial set of legacy spellings would hide important
differences in ownership, dynamics construction, and error behavior. A
separate facade can be considered after the deferred ecosystem modules have
clear support decisions.

The comparison below is against the public Python surface of the pinned
upstream version recorded in `UPSTREAM_NOTICE.md`.

## Equivalent capabilities with descriptive names

| Upstream | `pykep_rust` | Notes |
|---|---|---|
| `AU`, `CAVENDISH`, `EARTH_VELOCITY`, `G0` | `ASTRONOMICAL_UNIT`, `CAVENDISH_CONSTANT`, `EARTH_ORBITAL_VELOCITY`, `STANDARD_GRAVITY` | Same pinned SI values |
| `RAD2DEG`, `DEG2RAD`, `DAY2SEC`, `SEC2DAY` | `RADIANS_TO_DEGREES`, `DEGREES_TO_RADIANS`, `DAY_TO_SECONDS`, `SECONDS_TO_DAY` | Same conversion factors |
| `m2e`, `e2m`, `m2f`, `f2m`, `e2f`, `f2e` | `mean_to_eccentric_anomaly`, `eccentric_to_mean_anomaly`, `mean_to_true_anomaly`, `true_to_mean_anomaly`, `eccentric_to_true_anomaly`, `true_to_eccentric_anomaly` | Scalar elliptic conversions |
| `n2h`, `h2n`, `n2f`, `f2n`, `h2f`, `f2h` | Gudermannian/hyperbolic functions with full names | Scalar hyperbolic conversions |
| `ic2par`, `par2ic`, `ic2mee`, `mee2ic`, `par2mee`, `mee2par` | Cartesian/classical/modified-equinoctial functions with full names | `N × 6` NumPy batches are explicit `_batch` APIs |
| Lagrangian and Taylor propagation families | `propagate_lagrangian`, `propagate_lagrangian_grid`, evaluated model propagators | Native Rust numerical implementation |
| `lambert_problem` | `LambertProblem`, `LambertSolution` | Deterministic branch objects |
| `hohmann`, `bielliptic`, `mima`, `mima2` | Same names | Return values are typed in the stub |
| `alpha2direct`, `direct2alpha`, `eta2direct`, `direct2eta` | `alpha_to_direct`, `direct_to_alpha`, `eta_to_direct`, `direct_to_eta` | Descriptive direction |
| `fb_con`, `fb_dv`, `fb_vout` | `flyby_constraints`, `flyby_delta_v`, `flyby_outgoing_velocity` | Jacobian has a separate named function |
| `leg.sims_flanagan`, `leg.sims_flanagan_alpha`, `leg.zoh` | `SimsFlanaganLeg`, `SimsFlanaganAlphaLeg`, `ZohLeg` | Same numerical cores, immutable validated construction |

## Intentionally different

| Area | `pykep_rust` contract |
|---|---|
| Epochs | `Epoch` is immutable, has microsecond resolution, and requires an explicit `mjd2000`, `mjd`, or `jd` numeric scale. Arithmetic day counts do not imply UTC, TT, or TDB. |
| Planets | `Planet` uses explicit static constructors and owns a thread-safe native provider instead of accepting arbitrary Python UDPLA objects. |
| Taylor dynamics | Python receives evaluated RHS and propagation functions. It does not expose or require heyoka expression graphs or integrator objects. |
| ZOH legs | `ZohModel` selects one of four built-in native dynamics. Constructor input is copied; validated legs do not expose mutation setters. |
| Errors | Invalid values/shapes raise `ValueError`; singular geometry, convergence, integration, and missing capabilities have typed `PykepError` subclasses. |
| Batches | Throughput-sensitive entry points explicitly accept `float64` NumPy arrays, preserve order, return owned arrays, and release the GIL during native work. |

## Deferred or unsupported

The following upstream areas are not part of the completed native numerical
core and are not emulated:

- SPICE kernels, TLE parsing, and Python-defined UDPLA providers
  (`PY-EXT-001`);
- plotting helpers, trajectory-optimization UDPs, gym utilities, and optional
  ecosystem integrations (`PY-ECOSYSTEM-001`);
- symbolic heyoka dynamics/integrator construction and arbitrary user-supplied
  expression graphs (`TA-SYMBOLIC-001`);
- `mima_from_hop` and `mima2_from_hop` (`MIMA-HOP-001`);
- deprecated compatibility aliases and nested namespace layouts.

Use `has_acceleration()` and the VSOP2013 availability/threshold queries when
code depends on an optional provider capability. Unsupported provider
operations raise `UnsupportedCapabilityError`; unsupported ecosystem modules
are absent rather than silently approximated.
