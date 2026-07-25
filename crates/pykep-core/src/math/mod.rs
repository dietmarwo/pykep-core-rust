// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Numerical functions used by orbital algorithms.

/// Kepler-equation residuals and derivatives.
pub mod kepler_equations;
/// Fixed-size vector and matrix operations.
pub mod linalg;
/// Numerically stable Stumpff functions.
pub mod stumpff;
