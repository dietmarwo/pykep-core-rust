// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
//                          Advanced Concepts Team, European Space Agency (ESA)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Low-thrust trajectory leg models.
//!
//! ```
//! use pykep_core::leg::{
//!     SimsFlanaganLeg, SimsFlanaganSettings, SpacecraftEndpoint,
//! };
//!
//! let endpoint =
//!     SpacecraftEndpoint::new([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 1.0)?;
//! let settings = SimsFlanaganSettings::new(0.0, 0.0, 1.0, 1.0, 0.5)?;
//! let leg = SimsFlanaganLeg::new(endpoint, vec![[0.0; 3]], endpoint, settings)?;
//! assert_eq!(leg.mismatch_constraints()?, [0.0; 7]);
//! # Ok::<(), pykep_core::PykepError>(())
//! ```

mod sims_flanagan;
mod zoh;

pub use sims_flanagan::{
    SimsFlanaganAlphaLeg, SimsFlanaganLeg, SimsFlanaganMismatchJacobian, SimsFlanaganSettings,
    SpacecraftEndpoint,
};
pub use zoh::{
    ZohCr3bpLeg, ZohEquinoctialLeg, ZohKeplerLeg, ZohLeg, ZohLegHistory, ZohLegMismatchJacobian,
    ZohSolarSailLeg, evaluate_zoh_mismatch_batch,
};
