//! Evaluate constants, Julian arithmetic, Stumpff functions, and vector math.
//!
//! Units: SI constants, MJD2000 days, and dimensionless numerical functions.
//! Expected: epoch zero, unit z normal, `C(0)=1/2`, and `S(0)=1/6`.
//! Runtime: constant work, normally below 1 ms in a release build.
//! Features: default `pykep-core`; no external data or runtime.

use pykep_core::Result;
use pykep_core::constants::ASTRONOMICAL_UNIT;
use pykep_core::math::linalg::cross;
use pykep_core::math::stumpff::{stumpff_c, stumpff_s};
use pykep_core::time::julian::jd_to_mjd2000;

fn main() -> Result<()> {
    let epoch = jd_to_mjd2000(2_451_544.5)?;
    let normal = cross(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0])?;
    println!("J2000 epoch in MJD2000 days: {epoch}");
    println!("x × y: {normal:?}");
    println!("C(0): {}", stumpff_c(0.0)?);
    println!("S(0): {}", stumpff_s(0.0)?);
    println!("astronomical unit: {ASTRONOMICAL_UNIT} m");
    Ok(())
}
