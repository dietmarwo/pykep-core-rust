// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Evaluate a constructed Keplerian ephemeris at MJD2000 epochs.

use pykep_core::astro::elements::ClassicalElements;
use pykep_core::ephemeris::{Ephemeris, KeplerianEphemeris};
use pykep_core::time::epoch::Epoch;

fn main() -> Result<(), pykep_core::PykepError> {
    let provider = KeplerianEphemeris::from_classical(
        Epoch::default(),
        ClassicalElements::new(3.0, 0.2, 0.4, 0.3, 0.2, 0.1),
        1.0,
        "example",
        None,
        None,
        None,
    )?;
    for (epoch, state) in [0.0, 0.001, 0.01]
        .into_iter()
        .zip(provider.states(&[0.0, 0.001, 0.01])?)
    {
        println!("MJD2000 {epoch}: {state:?}");
    }
    Ok(())
}
