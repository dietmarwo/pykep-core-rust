// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Common fixed-shape numerical values.

/// Three Cartesian components, with units specified by the consuming API.
pub type Vector3 = [f64; 3];

/// Cartesian state ordered as `[x, y, z, vx, vy, vz]`.
///
/// Unless documented otherwise, position is in metres and velocity is in
/// metres per second.
pub type CartesianState = [f64; 6];

/// Six orbital elements whose ordering is specified by the consuming API.
pub type Elements6 = [f64; 6];

/// Row-major 3 × 3 binary64 matrix.
pub type Matrix3 = [[f64; 3]; 3];

/// Row-major 6 × 6 binary64 matrix.
pub type Matrix6 = [[f64; 6]; 6];
