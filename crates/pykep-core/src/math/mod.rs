// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Numerical functions used by orbital algorithms.
//!
//! ```
//! use pykep_core::math::{linalg::cross, stumpff::stumpff_c};
//!
//! assert_eq!(cross(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0])?, [0.0, 0.0, 1.0]);
//! assert!((stumpff_c(0.0)? - 0.5).abs() < f64::EPSILON);
//! # Ok::<(), pykep_core::PykepError>(())
//! ```

/// Kepler-equation residuals and derivatives.
pub mod kepler_equations;
/// Fixed-size vector and matrix operations.
pub mod linalg;
/// Numerically stable Stumpff functions.
pub mod stumpff;
