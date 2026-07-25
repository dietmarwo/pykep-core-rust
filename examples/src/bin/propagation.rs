// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Propagate a normalized circular orbit and inspect its STM.
//!
//! Units: normalized radius, velocity, time, and `mu`.
//! Expected: quarter-orbit state `[0,1,0,-1,0,0]` and a 6 × 6 STM.
//! Runtime: constant work, normally below 1 ms in a release build.
//! Features: default `pykep-core`; no external data or runtime.

use pykep_core::astro::propagation::propagate_lagrangian_with_stm;

fn main() -> Result<(), pykep_core::PykepError> {
    // Dimensionless units: radius = speed = mu = 1, so one period is 2π.
    let initial = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let (quarter_orbit, stm) =
        propagate_lagrangian_with_stm(&initial, core::f64::consts::FRAC_PI_2, 1.0)?;
    println!("quarter-orbit state: {quarter_orbit:?}");
    println!("STM first row: {:?}", stm[0]);
    Ok(())
}
