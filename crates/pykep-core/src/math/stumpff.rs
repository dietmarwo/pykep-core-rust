// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
//                         Advanced Concepts Team, European Space Agency (ESA)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
//
// Adapted from include/kep3/core_astro/special_functions.hpp at pykep commit
// 53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e. The near-zero evaluation uses a
// stable series instead of the cancellation-prone upstream expression.

//! Stumpff functions used by universal-variable formulations.

use crate::Result;
use crate::error::{ensure_finite, ensure_finite_output};

const SERIES_THRESHOLD: f64 = 0.5;
const MAX_SERIES_TERMS: usize = 32;

fn series_c(x: f64) -> f64 {
    let mut sum = 0.5;
    let mut term = sum;
    for index in 0..MAX_SERIES_TERMS {
        let n = 2.0 * index as f64;
        term *= -x / ((n + 3.0) * (n + 4.0));
        sum += term;
        if term.abs() <= f64::EPSILON * sum.abs() {
            break;
        }
    }
    sum
}

fn series_s(x: f64) -> f64 {
    let mut sum = 1.0 / 6.0;
    let mut term = sum;
    for index in 0..MAX_SERIES_TERMS {
        let n = 2.0 * index as f64;
        term *= -x / ((n + 4.0) * (n + 5.0));
        sum += term;
        if term.abs() <= f64::EPSILON * sum.abs() {
            break;
        }
    }
    sum
}

/// Evaluates the Stumpff function
/// `C(x) = Σ (-x)^k / (2k + 2)!`.
///
/// A series is used for `|x| < 0.5`; trigonometric or hyperbolic identities
/// are used elsewhere. The function is dimensionless.
///
/// # Errors
///
/// Returns an error for NaN/infinite input or when the finite mathematical
/// result overflows binary64.
pub fn stumpff_c(x: f64) -> Result<f64> {
    ensure_finite("x", x)?;
    let value = if x.abs() < SERIES_THRESHOLD {
        series_c(x)
    } else if x > 0.0 {
        let root = x.sqrt();
        2.0 * (0.5 * root).sin().powi(2) / x
    } else {
        let minus_x = -x;
        let root = minus_x.sqrt();
        2.0 * (0.5 * root).sinh().powi(2) / minus_x
    };
    ensure_finite_output("stumpff_c", value)
}

/// Evaluates the Stumpff function
/// `S(x) = Σ (-x)^k / (2k + 3)!`.
///
/// A series is used for `|x| < 0.5`; trigonometric or hyperbolic identities
/// are used elsewhere. The function is dimensionless.
///
/// # Errors
///
/// Returns an error for NaN/infinite input or when the finite mathematical
/// result overflows binary64.
pub fn stumpff_s(x: f64) -> Result<f64> {
    ensure_finite("x", x)?;
    let value = if x.abs() < SERIES_THRESHOLD {
        series_s(x)
    } else if x > 0.0 {
        let root = x.sqrt();
        (root - root.sin()) / root.powi(3)
    } else {
        let root = (-x).sqrt();
        (root.sinh() - root) / root.powi(3)
    };
    ensure_finite_output("stumpff_s", value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PykepError;

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual:.17e} != {expected:.17e}"
        );
    }

    #[test]
    fn exact_limits_are_returned_at_zero() {
        assert_eq!(stumpff_c(0.0), Ok(0.5));
        assert_eq!(stumpff_c(-0.0), Ok(0.5));
        assert_eq!(stumpff_s(0.0), Ok(1.0 / 6.0));
    }

    #[test]
    fn near_zero_series_avoids_catastrophic_cancellation() {
        for x in [-1e-14, -1e-12, 1e-12, 1e-14] {
            close(stumpff_c(x).unwrap(), 0.5 - x / 24.0, 2e-16);
            close(stumpff_s(x).unwrap(), 1.0 / 6.0 - x / 120.0, 2e-16);
        }
    }

    #[test]
    fn ordinary_cpp_reference_values_match() {
        close(stumpff_c(1.0).unwrap(), 0.459_697_694_131_860_3, 2e-16);
        close(stumpff_s(1.0).unwrap(), 0.158_529_015_192_103_5, 2e-16);
        close(stumpff_c(-1.0).unwrap(), 0.543_080_634_815_243_7, 2e-16);
        close(stumpff_s(-1.0).unwrap(), 0.175_201_193_643_801_46, 2e-16);
    }

    #[test]
    fn non_finite_and_overflowing_inputs_are_errors() {
        assert_eq!(
            stumpff_c(f64::NAN),
            Err(PykepError::NonFiniteInput { parameter: "x" })
        );
        assert!(matches!(
            stumpff_s(-1e7),
            Err(PykepError::NumericalOverflow { .. })
        ));
    }
}
