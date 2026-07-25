// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Compare approximate and analytical heliocentric distance at J2000.

use pykep_core::ephemeris::{Ephemeris, JplLowPrecision, Vsop2013};
use pykep_core::math::linalg::norm;

fn main() -> pykep_core::Result<()> {
    let jpl = JplLowPrecision::new("earth")?;
    let vsop = Vsop2013::new("earth_moon")?;
    let jpl_state = jpl.state(0.5)?;
    let vsop_state = vsop.state(0.5)?;
    let jpl_distance = norm(&[jpl_state[0], jpl_state[1], jpl_state[2]])?;
    let vsop_distance = norm(&[vsop_state[0], vsop_state[1], vsop_state[2]])?;
    println!("MJD2000 0.5 heliocentric distance [m]");
    println!("JPL low precision: {jpl_distance:.3}");
    println!("VSOP2013:         {vsop_distance:.3}");
    println!("The JPL state is J2000 ecliptic; the VSOP state is ICRF.");
    Ok(())
}
