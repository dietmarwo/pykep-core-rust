// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
//
// Adapted from src/planet.cpp at pykep commit
// 53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e.

//! Object-safe ephemeris interface and built-in providers.

mod jpl_lp;
mod keplerian;
mod vsop2013;

pub use jpl_lp::JplLowPrecision;
pub use keplerian::KeplerianEphemeris;
pub use vsop2013::{VSOP2013_MINIMUM_THRESHOLD, Vsop2013};

use crate::astro::anomalies::true_to_mean_anomaly;
use crate::astro::elements::{
    ClassicalElements, cartesian_to_classical, cartesian_to_modified_equinoctial,
};
use crate::error::ensure_finite;
use crate::math::linalg::norm;
use crate::{CartesianState, Elements6, PykepError, Result, Vector3};

/// Physical metadata exposed by built-in ephemeris providers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EphemerisMetadata {
    /// Central-body gravitational parameter.
    pub central_mu: Option<f64>,
    /// Provider-body gravitational parameter.
    pub body_mu: Option<f64>,
    /// Mean physical radius.
    pub radius: Option<f64>,
    /// Recommended safe encounter radius.
    pub safe_radius: Option<f64>,
}

/// Requested osculating element representation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ElementRepresentation {
    /// `[a,e,i,Omega,omega,true_anomaly]`.
    ClassicalTrue,
    /// `[a,e,i,Omega,omega,mean_anomaly]` for elliptic states.
    ClassicalMean,
    /// Prograde modified equinoctial `[p,f,g,h,k,L]`.
    ModifiedEquinoctial,
    /// Retrograde modified equinoctial `[p,f,g,h,k,L]`.
    ModifiedEquinoctialRetrograde,
}

/// Thread-safe object-safe source of Cartesian states at MJD2000 epochs.
pub trait Ephemeris: Send + Sync {
    /// Evaluate `[x,y,z,vx,vy,vz]` at an MJD2000 epoch.
    ///
    /// # Errors
    ///
    /// Returns a provider-specific validation or evaluation error.
    fn state(&self, epoch_mjd2000: f64) -> Result<CartesianState>;

    /// Stable human-readable provider/body name.
    fn name(&self) -> &str;

    /// Optional physical metadata.
    fn metadata(&self) -> EphemerisMetadata {
        EphemerisMetadata::default()
    }

    /// Optional Cartesian acceleration at an MJD2000 epoch.
    ///
    /// # Errors
    ///
    /// Returns an unsupported-capability or provider evaluation error.
    fn acceleration(&self, _epoch_mjd2000: f64) -> Result<Vector3> {
        Err(PykepError::UnsupportedCapability {
            provider: self.name().into(),
            capability: "acceleration",
        })
    }

    /// Evaluate an ordered epoch batch without creating an implicit thread
    /// pool.
    ///
    /// # Errors
    ///
    /// Returns the first provider evaluation error.
    fn states(&self, epochs_mjd2000: &[f64]) -> Result<Vec<CartesianState>> {
        epochs_mjd2000
            .iter()
            .map(|&epoch| self.state(epoch))
            .collect()
    }

    /// Derive the orbital period from osculating energy.
    ///
    /// Hyperbolic states return `None`.
    ///
    /// # Errors
    ///
    /// Returns an error when central-body metadata or state evaluation is
    /// unavailable or invalid.
    fn period(&self, epoch_mjd2000: f64) -> Result<Option<f64>> {
        ensure_finite("epoch_mjd2000", epoch_mjd2000)?;
        let mu = self
            .metadata()
            .central_mu
            .ok_or_else(|| PykepError::UnsupportedCapability {
                provider: self.name().into(),
                capability: "central_mu",
            })?;
        let state = self.state(epoch_mjd2000)?;
        let position = [state[0], state[1], state[2]];
        let velocity_squared = state[3] * state[3] + state[4] * state[4] + state[5] * state[5];
        let energy = velocity_squared / 2.0 - mu / norm(&position)?;
        if energy >= 0.0 {
            Ok(None)
        } else {
            let semi_major_axis = -mu / (2.0 * energy);
            Ok(Some(
                2.0 * core::f64::consts::PI * (semi_major_axis.powi(3) / mu).sqrt(),
            ))
        }
    }

    /// Derive osculating orbital elements from the Cartesian state.
    ///
    /// # Errors
    ///
    /// Returns an error for unavailable metadata, provider failure, singular
    /// classical geometry, or an incompatible representation.
    fn elements(
        &self,
        epoch_mjd2000: f64,
        representation: ElementRepresentation,
    ) -> Result<Elements6> {
        let state = self.state(epoch_mjd2000)?;
        let mu = self
            .metadata()
            .central_mu
            .ok_or_else(|| PykepError::UnsupportedCapability {
                provider: self.name().into(),
                capability: "central_mu",
            })?;
        match representation {
            ElementRepresentation::ClassicalTrue => {
                cartesian_to_classical(&state, mu).map(ClassicalElements::to_array)
            }
            ElementRepresentation::ClassicalMean => {
                let mut elements = cartesian_to_classical(&state, mu)?;
                if elements.semi_major_axis <= 0.0 {
                    return Err(PykepError::InvalidInput {
                        parameter: "representation",
                        reason: "mean anomaly elements require an elliptic orbit".into(),
                    });
                }
                elements.true_anomaly =
                    true_to_mean_anomaly(elements.true_anomaly, elements.eccentricity)?;
                Ok(elements.to_array())
            }
            ElementRepresentation::ModifiedEquinoctial => {
                cartesian_to_modified_equinoctial(&state, mu, false).map(|value| value.to_array())
            }
            ElementRepresentation::ModifiedEquinoctialRetrograde => {
                cartesian_to_modified_equinoctial(&state, mu, true).map(|value| value.to_array())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Minimal;

    impl Ephemeris for Minimal {
        fn state(&self, _: f64) -> Result<CartesianState> {
            Ok([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        }

        fn name(&self) -> &'static str {
            "minimal"
        }
    }

    #[test]
    fn optional_capabilities_are_explicit() {
        let minimal = Minimal;
        assert!(minimal.acceleration(0.0).is_err());
        assert!(minimal.period(0.0).is_err());
    }
}
