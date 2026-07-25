// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Evaluate gravity-assist feasibility and an unpowered outgoing velocity.
//!
//! Units: SI velocity, distance, and gravitational parameter; beta is radians.
//! Expected: two finite constraints, positive powered delta-v, finite output.
//! Runtime: constant work, normally below 1 ms in a release build.
//! Features: default `pykep-core`; no external data or runtime.

use pykep_core::astro::flyby::{flyby_constraints, flyby_delta_v, flyby_outgoing_velocity};

fn main() -> pykep_core::Result<()> {
    let incoming_excess = [7_200.0, -4_567.765_5, 1_234.423_3];
    let outgoing_excess = [7_100.0, 220.123, -144.432];
    let earth_mu = 3.986_004_418e14;
    let periapsis = 7.0e6;

    let constraints = flyby_constraints(&incoming_excess, &outgoing_excess, earth_mu, periapsis)?;
    let powered_delta_v = flyby_delta_v(&incoming_excess, &outgoing_excess, earth_mu, periapsis)?;
    let outgoing = flyby_outgoing_velocity(
        &incoming_excess,
        &[10_000.0, 20_000.0, -1_000.0],
        periapsis,
        0.2,
        earth_mu,
    )?;

    assert!(constraints.into_iter().all(f64::is_finite));
    assert!(powered_delta_v > 0.0);
    assert!(outgoing.into_iter().all(f64::is_finite));
    println!("constraints [equality, inequality]: {constraints:?}");
    println!("minimum powered delta-v [m/s]: {powered_delta_v:.6}");
    println!("unpowered outgoing velocity [m/s]: {outgoing:?}");
    Ok(())
}
