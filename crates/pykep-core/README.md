# pykep-core

`pykep-core` is an independent native Rust implementation of numerical
algorithms from pykep version 3. The current phase provides physical constants,
Julian-date arithmetic, stable Stumpff functions, Kepler-equation residuals,
and allocation-free small-vector/matrix operations.

The crate has no C or C++ runtime dependency. Later orbital algorithms remain
planned and are not represented by the current status probe.

## Example

```rust
use pykep_core::math::linalg::cross;
use pykep_core::math::stumpff::stumpff_c;
use pykep_core::time::julian::jd_to_mjd2000;

assert_eq!(jd_to_mjd2000(2_451_544.5)?, 0.0);
assert_eq!(cross(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0])?, [0.0, 0.0, 1.0]);
assert!((stumpff_c(1e-12)? - 0.5).abs() < 1e-13);
# Ok::<(), pykep_core::PykepError>(())
```

The crate is deliberately marked `publish = false` until a useful,
cross-module orbital-mechanics milestone is complete.
