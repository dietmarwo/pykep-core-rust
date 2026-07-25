//! Convert classical elements, round-trip them, and evaluate a Jacobian.
//!
//! Units: SI metres, seconds, radians, and `mu` in m³/s².
//! Expected: a stable Cartesian/MEE round trip and a finite 6 × 6 Jacobian.
//! Runtime: constant work, normally below 1 ms in a release build.
//! Features: default `pykep-core`; no external data or runtime.

use pykep_core::Result;
use pykep_core::astro::elements::{
    ClassicalElements, cartesian_to_modified_equinoctial,
    cartesian_to_modified_equinoctial_jacobian, classical_to_cartesian,
    modified_equinoctial_to_cartesian,
};

fn main() -> Result<()> {
    let classical = ClassicalElements::new(7_000_000.0, 0.01, 0.4, 1.0, 0.5, 0.2);
    let state = classical_to_cartesian(classical, 3.986_004_418e14)?;
    let equinoctial = cartesian_to_modified_equinoctial(&state, 3.986_004_418e14, false)?;
    let reconstructed = modified_equinoctial_to_cartesian(equinoctial, 3.986_004_418e14, false)?;
    let jacobian = cartesian_to_modified_equinoctial_jacobian(&state, 3.986_004_418e14, false)?;

    println!("Cartesian state: {state:?}");
    println!("MEE [p,f,g,h,k,L]: {:?}", equinoctial.to_array());
    println!("Round trip: {reconstructed:?}");
    println!("Jacobian row 0: {:?}", jacobian[0]);
    Ok(())
}
