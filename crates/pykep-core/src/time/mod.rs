// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Epoch types and time-system conversions.
//!
//! ```
//! use pykep_core::time::epoch::Epoch;
//!
//! let epoch = Epoch::from_iso("2000-01-02T00:00:00")?;
//! assert_eq!(epoch.mjd2000(), 1.0);
//! assert_eq!(Epoch::from_iso(&epoch.to_string())?, epoch);
//! # Ok::<(), pykep_core::PykepError>(())
//! ```

/// Microsecond-resolution epochs and ISO calendar conversion.
pub mod epoch;
/// Scalar Julian date conversions.
pub mod julian;
