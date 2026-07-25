// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
//                         Advanced Concepts Team, European Space Agency (ESA)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
//
// Adapted from src/linalg.cpp and include/kep3/linalg.hpp at pykep commit
// 53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e.

//! Allocation-free operations for small fixed vectors and matrices.

use crate::error::{ensure_finite, ensure_finite_output};
use crate::{Matrix3, PykepError, Result, Vector3};

fn validate_vector(parameter: &'static str, vector: &Vector3) -> Result<()> {
    for &value in vector {
        ensure_finite(parameter, value)?;
    }
    Ok(())
}

/// Computes the Euclidean dot product of two three-vectors.
///
/// Input and output units follow ordinary dimensional multiplication.
///
/// # Errors
///
/// Returns an error for non-finite components or binary64 overflow.
pub fn dot(left: &Vector3, right: &Vector3) -> Result<f64> {
    validate_vector("left", left)?;
    validate_vector("right", right)?;
    ensure_finite_output(
        "dot",
        left[0].mul_add(right[0], left[1].mul_add(right[1], left[2] * right[2])),
    )
}

/// Computes the Euclidean norm of a three-vector.
///
/// The result has the same units as the vector components.
///
/// # Errors
///
/// Returns an error for non-finite components or binary64 overflow.
pub fn norm(vector: &Vector3) -> Result<f64> {
    validate_vector("vector", vector)?;
    ensure_finite_output("norm", vector[0].hypot(vector[1]).hypot(vector[2]))
}

/// Returns a normalized unit vector.
///
/// # Errors
///
/// Returns an error for non-finite input, a zero vector, or binary64
/// overflow.
pub fn normalize(vector: &Vector3) -> Result<Vector3> {
    let magnitude = norm(vector)?;
    if magnitude == 0.0 {
        return Err(PykepError::SingularGeometry {
            operation: "normalize",
        });
    }
    let result = [
        vector[0] / magnitude,
        vector[1] / magnitude,
        vector[2] / magnitude,
    ];
    validate_vector("normalized_vector", &result)?;
    Ok(result)
}

/// Computes the right-handed cross product `left × right`.
///
/// Input and output units follow ordinary dimensional multiplication.
///
/// # Errors
///
/// Returns an error for non-finite components or binary64 overflow.
pub fn cross(left: &Vector3, right: &Vector3) -> Result<Vector3> {
    validate_vector("left", left)?;
    validate_vector("right", right)?;
    let result = [
        left[1].mul_add(right[2], -(left[2] * right[1])),
        left[2].mul_add(right[0], -(left[0] * right[2])),
        left[0].mul_add(right[1], -(left[1] * right[0])),
    ];
    for &value in &result {
        ensure_finite_output("cross", value)?;
    }
    Ok(result)
}

/// Returns the row-major skew-symmetric matrix `[vector]×`.
///
/// Multiplying the result by another vector produces their cross product.
///
/// # Errors
///
/// Returns an error when a component is NaN or infinite.
pub fn skew(vector: &Vector3) -> Result<Matrix3> {
    validate_vector("vector", vector)?;
    Ok([
        [0.0, -vector[2], vector[1]],
        [vector[2], 0.0, -vector[0]],
        [-vector[1], vector[0], 0.0],
    ])
}

/// Returns an `N × N` row-major identity matrix.
#[must_use]
pub fn identity<const N: usize>() -> [[f64; N]; N] {
    let mut result = [[0.0; N]; N];
    for (index, row) in result.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    result
}

/// Multiplies fixed row-major `M × K` and `K × N` matrices.
///
/// # Errors
///
/// Returns an error for any non-finite input or output component.
pub fn matrix_multiply<const M: usize, const K: usize, const N: usize>(
    left: &[[f64; K]; M],
    right: &[[f64; N]; K],
) -> Result<[[f64; N]; M]> {
    let mut result = [[0.0; N]; M];
    for row in 0..M {
        for column in 0..N {
            let mut value = 0.0;
            for inner in 0..K {
                ensure_finite("left", left[row][inner])?;
                ensure_finite("right", right[inner][column])?;
                value = left[row][inner].mul_add(right[inner][column], value);
            }
            result[row][column] = ensure_finite_output("matrix_multiply", value)?;
        }
    }
    Ok(result)
}

/// Multiplies a fixed row-major `M × N` matrix by an `N`-vector.
///
/// # Errors
///
/// Returns an error for any non-finite input or output component.
pub fn matrix_vector_multiply<const M: usize, const N: usize>(
    matrix: &[[f64; N]; M],
    vector: &[f64; N],
) -> Result<[f64; M]> {
    let mut result = [0.0; M];
    for row in 0..M {
        let mut value = 0.0;
        for column in 0..N {
            ensure_finite("matrix", matrix[row][column])?;
            ensure_finite("vector", vector[column])?;
            value = matrix[row][column].mul_add(vector[column], value);
        }
        result[row] = ensure_finite_output("matrix_vector_multiply", value)?;
    }
    Ok(result)
}

/// Transposes a fixed row-major `M × N` matrix.
#[must_use]
pub fn transpose<const M: usize, const N: usize>(matrix: &[[f64; N]; M]) -> [[f64; M]; N] {
    let mut result = [[0.0; M]; N];
    for (row, values) in matrix.iter().enumerate() {
        for (column, &value) in values.iter().enumerate() {
            result[column][row] = value;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_operations_match_right_handed_geometry() {
        let x = [1.0, 0.0, 0.0];
        let y = [0.0, 1.0, 0.0];
        assert_eq!(dot(&x, &y), Ok(0.0));
        assert_eq!(norm(&[3.0, 4.0, 12.0]), Ok(13.0));
        assert_eq!(cross(&x, &y), Ok([0.0, 0.0, 1.0]));
        assert_eq!(
            matrix_vector_multiply(&skew(&x).unwrap(), &y),
            cross(&x, &y)
        );
    }

    #[test]
    fn normalization_is_stable_for_large_finite_vectors() {
        let value = normalize(&[1e308, 1e308, 0.0]).unwrap();
        let expected = core::f64::consts::FRAC_1_SQRT_2;
        assert!((value[0] - expected).abs() < 2e-16);
        assert!((value[1] - expected).abs() < 2e-16);
    }

    #[test]
    fn zero_and_non_finite_vectors_are_rejected() {
        assert!(matches!(
            normalize(&[0.0; 3]),
            Err(PykepError::SingularGeometry { .. })
        ));
        assert!(dot(&[f64::NAN, 0.0, 0.0], &[1.0; 3]).is_err());
    }

    #[test]
    fn fixed_matrix_operations_preserve_layout() {
        let left = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let right = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];
        assert_eq!(
            matrix_multiply(&left, &right),
            Ok([[58.0, 64.0], [139.0, 154.0]])
        );
        assert_eq!(transpose(&left), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
        assert_eq!(
            identity::<3>(),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );
    }
}
