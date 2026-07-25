// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Evaluate an approximate heliocentric Earth state at J2000.
//!
//! Units: MJD2000 days, metres, and metres/second.
//! Expected: one finite six-component J2000-ecliptic state.
//! Runtime: constant work, normally below 1 ms in a release build.
//! Features: works without optional `pykep-core` data features.

use pykep_core::ephemeris::{Ephemeris, JplLowPrecision};

fn main() -> pykep_core::Result<()> {
    let earth = JplLowPrecision::new("earth")?;
    let state = earth.state(0.0)?;
    println!("{} at MJD2000 0:", earth.name());
    println!("position [m] = {:?}", &state[..3]);
    println!("velocity [m/s] = {:?}", &state[3..]);
    Ok(())
}
