// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
//
// Adapted from src/core_astro/encodings.cpp at pykep commit
// 53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e.

//! Reversible encodings for vectors of leg durations.

use crate::error::{ensure_finite, ensure_finite_output};
use crate::{PykepError, Result};

fn validate_slice(parameter: &'static str, values: &[f64], nonempty: bool) -> Result<()> {
    if nonempty && values.is_empty() {
        return Err(PykepError::InvalidInput {
            parameter,
            reason: "must be non-empty".into(),
        });
    }
    for &value in values {
        ensure_finite(parameter, value)?;
    }
    Ok(())
}

/// Decodes positive alpha variables into leg durations summing to `total_time`.
///
/// # Errors
///
/// Returns an error unless every alpha is strictly between zero and one and
/// the total time is finite and positive.
pub fn alpha_to_direct(alphas: &[f64], total_time: f64) -> Result<Vec<f64>> {
    validate_slice("alphas", alphas, true)?;
    ensure_finite("total_time", total_time)?;
    if total_time <= 0.0 || alphas.iter().any(|&alpha| !(0.0..1.0).contains(&alpha)) {
        return Err(PykepError::InvalidInput {
            parameter: "alphas",
            reason: "alphas must satisfy 0 < alpha < 1 and total_time must be positive".into(),
        });
    }
    let logarithms: Vec<_> = alphas.iter().map(|alpha| alpha.ln()).collect();
    let sum: f64 = logarithms.iter().sum();
    if sum == 0.0 {
        return Err(PykepError::SingularGeometry {
            operation: "alpha_to_direct",
        });
    }
    logarithms
        .into_iter()
        .map(|value| ensure_finite_output("alpha_to_direct", value * total_time / sum))
        .collect()
}

/// Encodes positive direct leg durations as alpha variables.
///
/// Returns the alpha vector and its total time.
///
/// # Errors
///
/// Returns an error for an empty vector or non-positive/non-finite duration.
pub fn direct_to_alpha(times: &[f64]) -> Result<(Vec<f64>, f64)> {
    validate_slice("times", times, true)?;
    if times.iter().any(|&time| time <= 0.0) {
        return Err(PykepError::InvalidInput {
            parameter: "times",
            reason: "all durations must be greater than zero".into(),
        });
    }
    let total: f64 = times.iter().sum();
    ensure_finite_output("direct_to_alpha", total)?;
    let alphas = times
        .iter()
        .map(|&time| ensure_finite_output("direct_to_alpha", (-time / total).exp()))
        .collect::<Result<Vec<_>>>()?;
    Ok((alphas, total))
}

/// Decodes eta fractions into leg durations bounded by `maximum_time`.
///
/// # Errors
///
/// Returns an error for empty input, non-finite values, non-positive maximum
/// time, or eta values outside `[0, 1]`.
pub fn eta_to_direct(etas: &[f64], maximum_time: f64) -> Result<Vec<f64>> {
    validate_slice("etas", etas, true)?;
    ensure_finite("maximum_time", maximum_time)?;
    if maximum_time <= 0.0 || etas.iter().any(|&eta| !(0.0..=1.0).contains(&eta)) {
        return Err(PykepError::InvalidInput {
            parameter: "etas",
            reason: "etas must be in [0, 1] and maximum_time must be positive".into(),
        });
    }
    let mut remaining = maximum_time;
    let mut times = Vec::with_capacity(etas.len());
    for &eta in etas {
        let time = remaining * eta;
        times.push(time);
        remaining -= time;
    }
    Ok(times)
}

/// Encodes direct durations as eta fractions of the remaining time.
///
/// # Errors
///
/// Returns an error unless input is non-empty, all durations are
/// non-negative, and their sum does not exceed the positive maximum time.
pub fn direct_to_eta(times: &[f64], maximum_time: f64) -> Result<Vec<f64>> {
    validate_slice("times", times, true)?;
    ensure_finite("maximum_time", maximum_time)?;
    if maximum_time <= 0.0
        || times.iter().any(|&time| time < 0.0)
        || times.iter().sum::<f64>() > maximum_time
    {
        return Err(PykepError::InvalidInput {
            parameter: "times",
            reason: "durations must be non-negative and sum to at most maximum_time".into(),
        });
    }
    let mut remaining = maximum_time;
    let mut etas = Vec::with_capacity(times.len());
    for &time in times {
        if remaining == 0.0 {
            if time == 0.0 {
                etas.push(0.0);
                continue;
            }
            return Err(PykepError::SingularGeometry {
                operation: "direct_to_eta",
            });
        }
        etas.push(time / remaining);
        remaining -= time;
    }
    Ok(etas)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_pairs_round_trip() {
        let direct = [0.1, 0.2, 0.3];
        let (alpha, total) = direct_to_alpha(&direct).unwrap();
        let decoded = alpha_to_direct(&alpha, total).unwrap();
        for (actual, expected) in decoded.iter().zip(direct) {
            assert!((actual - expected).abs() < 2e-16);
        }
        let eta = direct_to_eta(&direct, 1.0).unwrap();
        assert_eq!(eta_to_direct(&eta, 1.0).unwrap(), direct);
    }
}
