// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
//
// Adapted from src/core_astro/basic_transfers.cpp at pykep commit
// 53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e.

//! Hohmann and bi-elliptic transfers between coplanar circular orbits.

use crate::error::{ensure_finite_output, ensure_finite_values};
use crate::{PykepError, Result};

/// Summary of an impulsive transfer.
#[derive(Clone, Debug, PartialEq)]
pub struct Transfer<const N: usize> {
    /// Total characteristic velocity.
    pub delta_v: f64,
    /// Ballistic transfer duration.
    pub time: f64,
    /// Individual impulse magnitudes in chronological order.
    pub impulses: [f64; N],
}

fn validate(radii: &[f64], mu: f64) -> Result<()> {
    ensure_finite_values(&[("mu", mu)])?;
    if mu <= 0.0 {
        return Err(PykepError::InvalidInput {
            parameter: "mu",
            reason: "must be greater than zero".into(),
        });
    }
    for &radius in radii {
        ensure_finite_values(&[("radius", radius)])?;
        if radius <= 0.0 {
            return Err(PykepError::InvalidInput {
                parameter: "radius",
                reason: "must be greater than zero".into(),
            });
        }
    }
    Ok(())
}

/// Computes a two-impulse Hohmann transfer.
///
/// # Errors
///
/// Returns an error for non-finite values, non-positive radii or `mu`, or
/// binary64 overflow.
pub fn hohmann(r1: f64, r2: f64, mu: f64) -> Result<Transfer<2>> {
    validate(&[r1, r2], mu)?;
    let v1 = (mu / r1).sqrt();
    let v2 = (mu / r2).sqrt();
    let transfer_v1 = (mu / r1 * (2.0 * r2 / (r1 + r2))).sqrt();
    let transfer_v2 = (mu / r2 * (2.0 * r1 / (r1 + r2))).sqrt();
    let impulses = [(transfer_v1 - v1).abs(), (v2 - transfer_v2).abs()];
    let delta_v = impulses.iter().sum();
    let time = core::f64::consts::PI * ((r1 + r2).powi(3) / (8.0 * mu)).sqrt();
    ensure_finite_output("hohmann", delta_v)?;
    ensure_finite_output("hohmann", time)?;
    Ok(Transfer {
        delta_v,
        time,
        impulses,
    })
}

/// Computes a three-impulse bi-elliptic transfer through apoapsis `rb`.
///
/// # Errors
///
/// Returns an error for non-finite values, non-positive radii or `mu`, or
/// binary64 overflow.
pub fn bielliptic(r1: f64, r2: f64, rb: f64, mu: f64) -> Result<Transfer<3>> {
    validate(&[r1, r2, rb], mu)?;
    let v1 = (mu / r1).sqrt();
    let v2 = (mu / r2).sqrt();
    let transfer_v1 = (mu / r1 * (2.0 * rb / (r1 + rb))).sqrt();
    let apoapsis_v1 = (mu / rb * (2.0 * r1 / (r1 + rb))).sqrt();
    let apoapsis_v2 = (mu / rb * (2.0 * r2 / (rb + r2))).sqrt();
    let transfer_v2 = (mu / r2 * (2.0 * rb / (rb + r2))).sqrt();
    let impulses = [
        (transfer_v1 - v1).abs(),
        (apoapsis_v2 - apoapsis_v1).abs(),
        (v2 - transfer_v2).abs(),
    ];
    let delta_v = impulses.iter().sum();
    let time = core::f64::consts::PI
        * (((r1 + rb).powi(3) / (8.0 * mu)).sqrt() + ((rb + r2).powi(3) / (8.0 * mu)).sqrt());
    ensure_finite_output("bielliptic", delta_v)?;
    ensure_finite_output("bielliptic", time)?;
    Ok(Transfer {
        delta_v,
        time,
        impulses,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_reference_values_match() {
        let transfer = hohmann(1.0, 2.0, 1.0).unwrap();
        assert!((transfer.delta_v - 0.284_457_05).abs() < 1e-8);
        assert!((transfer.impulses[0] - 0.154_700_54).abs() < 1e-8);
        assert!((transfer.impulses[1] - 0.129_756_5).abs() < 2e-8);
        let bi = bielliptic(1.0, 2.0, 2.0, 1.0).unwrap();
        assert!((bi.delta_v - transfer.delta_v).abs() < 2e-15);
        assert_eq!(bi.impulses[2], 0.0);
    }
}
