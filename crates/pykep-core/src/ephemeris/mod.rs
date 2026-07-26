// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
//
// Adapted from src/planet.cpp at pykep commit
// 53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e.

//! Object-safe ephemeris interface and built-in providers.
//!
//! ```
//! use pykep_core::ephemeris::{Ephemeris, KeplerianEphemeris};
//! use pykep_core::time::epoch::Epoch;
//!
//! let provider = KeplerianEphemeris::from_state(
//!     Epoch::new(),
//!     [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
//!     1.0,
//!     "unit orbit",
//!     None,
//!     None,
//!     None,
//! )?;
//! assert_eq!(provider.state(0.0)?, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
//! # Ok::<(), pykep_core::PykepError>(())
//! ```

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

    /// Evaluate an ordered epoch batch, optionally in parallel.
    ///
    /// Zero workers uses Rayon's shared global pool, one executes serially,
    /// and larger values use exactly that many cached worker threads. This is
    /// available to every thread-safe ephemeris provider.
    ///
    /// # Errors
    ///
    /// Returns an invalid worker count or the first provider evaluation error
    /// in input order.
    fn states_parallel(
        &self,
        epochs_mjd2000: &[f64],
        workers: usize,
    ) -> Result<Vec<CartesianState>> {
        crate::batch::try_map(epochs_mjd2000, workers, |epoch| self.state(*epoch))
    }

    /// Evaluate an ordered acceleration batch, optionally in parallel.
    ///
    /// Worker and ordering semantics match [`Ephemeris::states_parallel`].
    ///
    /// # Errors
    ///
    /// Returns an invalid worker count or the first unsupported-capability or
    /// provider evaluation error in input order.
    fn accelerations_parallel(
        &self,
        epochs_mjd2000: &[f64],
        workers: usize,
    ) -> Result<Vec<Vector3>> {
        crate::batch::try_map(epochs_mjd2000, workers, |epoch| self.acceleration(*epoch))
    }

    /// Derive an ordered orbital-period batch, optionally in parallel.
    ///
    /// Worker and ordering semantics match [`Ephemeris::states_parallel`].
    ///
    /// # Errors
    ///
    /// Returns an invalid worker count or the first metadata/provider error in
    /// input order.
    fn periods_parallel(&self, epochs_mjd2000: &[f64], workers: usize) -> Result<Vec<Option<f64>>> {
        crate::batch::try_map(epochs_mjd2000, workers, |epoch| self.period(*epoch))
    }

    /// Derive an ordered orbital-element batch, optionally in parallel.
    ///
    /// Worker and ordering semantics match [`Ephemeris::states_parallel`].
    ///
    /// # Errors
    ///
    /// Returns an invalid worker count or the first metadata, geometry, or
    /// provider error in input order.
    fn elements_parallel(
        &self,
        epochs_mjd2000: &[f64],
        representation: ElementRepresentation,
        workers: usize,
    ) -> Result<Vec<Elements6>> {
        crate::batch::try_map(epochs_mjd2000, workers, |epoch| {
            self.elements(*epoch, representation)
        })
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

    struct Minimal {
        state: CartesianState,
        central_mu: Option<f64>,
    }

    impl Ephemeris for Minimal {
        fn state(&self, _: f64) -> Result<CartesianState> {
            Ok(self.state)
        }

        fn name(&self) -> &'static str {
            "minimal"
        }

        fn metadata(&self) -> EphemerisMetadata {
            EphemerisMetadata {
                central_mu: self.central_mu,
                ..EphemerisMetadata::default()
            }
        }
    }

    #[test]
    fn optional_capabilities_are_explicit() {
        let minimal = Minimal {
            state: [1.0, 0.0, 0.0, 0.0, 0.8, 0.1],
            central_mu: None,
        };
        assert!(minimal.acceleration(0.0).is_err());
        assert!(minimal.period(0.0).is_err());
        assert!(
            minimal
                .elements(0.0, ElementRepresentation::ClassicalTrue)
                .is_err()
        );
        assert_eq!(minimal.states(&[0.0, 1.0]).unwrap().len(), 2);
    }

    #[test]
    fn derived_periods_and_every_element_representation_are_available() {
        let elliptic = Minimal {
            state: [1.0, 0.0, 0.0, 0.0, 0.8, 0.1],
            central_mu: Some(1.0),
        };
        assert!(elliptic.period(0.0).unwrap().unwrap().is_finite());
        for representation in [
            ElementRepresentation::ClassicalTrue,
            ElementRepresentation::ClassicalMean,
            ElementRepresentation::ModifiedEquinoctial,
            ElementRepresentation::ModifiedEquinoctialRetrograde,
        ] {
            assert!(
                elliptic
                    .elements(0.0, representation)
                    .unwrap()
                    .iter()
                    .all(|value| value.is_finite())
            );
        }

        let hyperbolic = Minimal {
            state: [1.0, 0.0, 0.0, 0.0, 2.0, 0.1],
            central_mu: Some(1.0),
        };
        assert_eq!(hyperbolic.period(0.0).unwrap(), None);
        assert!(
            hyperbolic
                .elements(0.0, ElementRepresentation::ClassicalMean)
                .is_err()
        );
    }

    #[test]
    fn parallel_derived_batches_match_scalar_order_and_errors() {
        let minimal = Minimal {
            state: [1.0, 0.0, 0.0, 0.0, 0.8, 0.1],
            central_mu: Some(1.0),
        };
        let epochs = [2.0, -1.0, 0.5];
        assert_eq!(
            minimal.states_parallel(&epochs, 2).unwrap(),
            minimal.states(&epochs).unwrap()
        );
        assert_eq!(
            minimal.periods_parallel(&epochs, 2).unwrap(),
            epochs
                .iter()
                .map(|epoch| minimal.period(*epoch))
                .collect::<Result<Vec<_>>>()
                .unwrap()
        );
        assert_eq!(
            minimal
                .elements_parallel(&epochs, ElementRepresentation::ModifiedEquinoctial, 2)
                .unwrap(),
            epochs
                .iter()
                .map(|epoch| {
                    minimal.elements(*epoch, ElementRepresentation::ModifiedEquinoctial)
                })
                .collect::<Result<Vec<_>>>()
                .unwrap()
        );
        assert!(minimal.accelerations_parallel(&epochs, 2).is_err());
    }
}
