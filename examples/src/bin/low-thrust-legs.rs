// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Evaluate a Sims–Flanagan mismatch and its analytic Jacobian.
//!
//! Units: one consistent length/time/mass system; controls are dimensionless.
//! Expected: seven mismatch values and a 7 × 13 control/time Jacobian.
//! Runtime: constant four-segment work, normally below 1 ms in release mode.
//! Features: default `pykep-core`; no external data or runtime.

use pykep_core::leg::{SimsFlanaganLeg, SimsFlanaganSettings, SpacecraftEndpoint};

fn main() -> pykep_core::Result<()> {
    let leg = SimsFlanaganLeg::new(
        SpacecraftEndpoint::new([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2.0)?,
        vec![
            [0.1, -0.2, 0.05],
            [0.3, 0.1, -0.15],
            [-0.25, 0.2, 0.1],
            [0.05, -0.1, 0.2],
        ],
        SpacecraftEndpoint::new([0.2, 1.1, 0.1, -0.9, 0.15, -0.05], 1.7)?,
        SimsFlanaganSettings::new(1.3, 0.04, 3.0, 1.0, 0.5)?,
    )?;
    let mismatch = leg.mismatch_constraints()?;
    let jacobian = leg.mismatch_jacobian()?;

    assert_eq!(mismatch.len(), 7);
    assert_eq!(jacobian.controls_and_time.len(), 7);
    assert_eq!(jacobian.controls_and_time[0].len(), 13);
    println!("mismatch [dr,dv,dm]: {mismatch:?}");
    println!(
        "Jacobian shapes: departure 7×7, arrival 7×7, controls/time {}×{}",
        jacobian.controls_and_time.len(),
        jacobian.controls_and_time[0].len()
    );
    Ok(())
}
