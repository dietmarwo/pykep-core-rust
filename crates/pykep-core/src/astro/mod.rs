// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Orbital-mechanics algorithms.
//!
//! ```
//! use pykep_core::astro::transfers::hohmann;
//!
//! let transfer = hohmann(7.0e6, 42.0e6, 3.986_004_418e14)?;
//! assert!(transfer.delta_v > 0.0);
//! assert_eq!(transfer.impulses.len(), 2);
//! # Ok::<(), pykep_core::PykepError>(())
//! ```

/// Elliptic and hyperbolic anomaly conversions.
pub mod anomalies;
/// Cartesian, classical, and modified-equinoctial element conversions.
pub mod elements;
/// Time-of-flight decision-vector encodings.
pub mod encodings;
/// Unpowered gravity-assist constraints and velocity mapping.
pub mod flyby;
/// Single- and multi-revolution Lambert boundary-value solver.
pub mod lambert;
/// Maximum-initial-mass approximations for low-thrust transfers.
pub mod mima;
/// Two-body propagation and state-transition matrices.
pub mod propagation;
/// Impulsive circular-orbit transfer approximations.
pub mod transfers;
