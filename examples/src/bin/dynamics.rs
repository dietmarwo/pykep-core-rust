// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Propagate CR3BP and piecewise-constant low-thrust Kepler dynamics.
//!
//! Units: normalized CR3BP/Kepler units; state order is `[r,v]` plus mass.
//! Expected: finite six- and seven-component final states at requested times.
//! Runtime: two short adaptive solves, normally below 1 ms in release mode.
//! Features: default `pykep-core`; no external data or runtime.

use pykep_core::dynamics::Cr3bpDynamics;
use pykep_core::dynamics::zoh::{ControlSchedule, ZohKeplerDynamics, propagate_schedule};
use pykep_core::integration::IntegratorOptions;

fn main() -> pykep_core::Result<()> {
    let cr3bp_initial = [0.8, -0.2, 0.1, 0.03, -0.04, 0.02];
    let cr3bp = Cr3bpDynamics.propagate(
        0.0,
        cr3bp_initial,
        0.1,
        0.012_150_585_609_624_04,
        IntegratorOptions::default(),
    )?;

    let schedule = ControlSchedule::new(
        vec![0.0, 0.05, 0.1],
        vec![[0.01, 1.0, 0.0, 0.0], [0.01, 0.0, 1.0, 0.0]],
    )?;
    let zoh = propagate_schedule(
        &ZohKeplerDynamics,
        &schedule,
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.2],
        [0.02],
        IntegratorOptions::default(),
    )?;

    assert_eq!(cr3bp.time, 0.1);
    assert_eq!(zoh.time, 0.1);
    assert!(cr3bp.state.into_iter().all(f64::is_finite));
    assert!(zoh.state.into_iter().all(f64::is_finite));
    println!("CR3BP final state: {:?}", cr3bp.state);
    println!("ZOH Kepler final state: {:?}", zoh.state);
    Ok(())
}
