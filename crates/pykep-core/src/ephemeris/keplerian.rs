// Copyright (c) 2023-2026 Dario Izzo (dario.izzo@gmail.com)
// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0
//
// Adapted from src/udpla/keplerian.cpp at pykep commit
// 53b1ca3ce5f8c223f96819b2ea9ba16c3719e63e.

use super::{Ephemeris, EphemerisMetadata};
use crate::astro::anomalies::mean_to_true_anomaly;
use crate::astro::elements::{
    ClassicalElements, ModifiedEquinoctialElements, classical_to_cartesian,
    modified_equinoctial_to_cartesian,
};
use crate::astro::propagation::propagate_lagrangian;
use crate::error::ensure_finite;
use crate::time::epoch::Epoch;
use crate::{CartesianState, PykepError, Result};

/// Analytic two-body ephemeris propagated from one reference state.
#[derive(Clone, Debug, PartialEq)]
pub struct KeplerianEphemeris {
    reference_epoch: Epoch,
    reference_state: CartesianState,
    central_mu: f64,
    name: String,
    metadata: EphemerisMetadata,
}

impl KeplerianEphemeris {
    /// Constructs a provider from a Cartesian reference state.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid metadata or a state that cannot be
    /// propagated.
    pub fn from_state(
        reference_epoch: Epoch,
        reference_state: CartesianState,
        central_mu: f64,
        name: impl Into<String>,
        body_mu: Option<f64>,
        radius: Option<f64>,
        safe_radius: Option<f64>,
    ) -> Result<Self> {
        validate_metadata(central_mu, body_mu, radius, safe_radius)?;
        propagate_lagrangian(&reference_state, 0.0, central_mu)?;
        Ok(Self {
            reference_epoch,
            reference_state,
            central_mu,
            name: name.into(),
            metadata: EphemerisMetadata {
                central_mu: Some(central_mu),
                body_mu,
                radius,
                safe_radius,
            },
        })
    }

    /// Constructs a provider from classical true-anomaly elements.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid elements or metadata.
    pub fn from_classical(
        reference_epoch: Epoch,
        elements: ClassicalElements,
        central_mu: f64,
        name: impl Into<String>,
        body_mu: Option<f64>,
        radius: Option<f64>,
        safe_radius: Option<f64>,
    ) -> Result<Self> {
        let state = classical_to_cartesian(elements, central_mu)?;
        Self::from_state(
            reference_epoch,
            state,
            central_mu,
            name,
            body_mu,
            radius,
            safe_radius,
        )
    }

    /// Constructs a provider from classical mean-anomaly elements.
    ///
    /// # Errors
    ///
    /// Returns an error for non-elliptic or otherwise invalid elements.
    pub fn from_classical_mean(
        reference_epoch: Epoch,
        mut elements: ClassicalElements,
        central_mu: f64,
        name: impl Into<String>,
        body_mu: Option<f64>,
        radius: Option<f64>,
        safe_radius: Option<f64>,
    ) -> Result<Self> {
        elements.true_anomaly = mean_to_true_anomaly(elements.true_anomaly, elements.eccentricity)?;
        Self::from_classical(
            reference_epoch,
            elements,
            central_mu,
            name,
            body_mu,
            radius,
            safe_radius,
        )
    }

    /// Constructs from prograde or retrograde modified equinoctial elements.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid elements or metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn from_modified_equinoctial(
        reference_epoch: Epoch,
        elements: ModifiedEquinoctialElements,
        central_mu: f64,
        retrograde: bool,
        name: impl Into<String>,
        body_mu: Option<f64>,
        radius: Option<f64>,
        safe_radius: Option<f64>,
    ) -> Result<Self> {
        let state = modified_equinoctial_to_cartesian(elements, central_mu, retrograde)?;
        Self::from_state(
            reference_epoch,
            state,
            central_mu,
            name,
            body_mu,
            radius,
            safe_radius,
        )
    }

    /// Reference epoch.
    #[must_use]
    pub const fn reference_epoch(&self) -> Epoch {
        self.reference_epoch
    }

    /// Reference Cartesian state.
    #[must_use]
    pub const fn reference_state(&self) -> CartesianState {
        self.reference_state
    }
}

fn validate_metadata(
    central_mu: f64,
    body_mu: Option<f64>,
    radius: Option<f64>,
    safe_radius: Option<f64>,
) -> Result<()> {
    ensure_finite("central_mu", central_mu)?;
    if central_mu <= 0.0 {
        return Err(PykepError::InvalidInput {
            parameter: "central_mu",
            reason: "must be greater than zero".into(),
        });
    }
    for (name, value) in [
        ("body_mu", body_mu),
        ("radius", radius),
        ("safe_radius", safe_radius),
    ] {
        if let Some(value) = value {
            ensure_finite(name, value)?;
            if value <= 0.0 {
                return Err(PykepError::InvalidInput {
                    parameter: name,
                    reason: "must be greater than zero when provided".into(),
                });
            }
        }
    }
    if let (Some(radius), Some(safe_radius)) = (radius, safe_radius)
        && safe_radius < radius
    {
        return Err(PykepError::InvalidInput {
            parameter: "safe_radius",
            reason: "must be at least the physical radius".into(),
        });
    }
    Ok(())
}

impl Ephemeris for KeplerianEphemeris {
    fn state(&self, epoch_mjd2000: f64) -> Result<CartesianState> {
        ensure_finite("epoch_mjd2000", epoch_mjd2000)?;
        let elapsed_seconds =
            (epoch_mjd2000 - self.reference_epoch.mjd2000()) * crate::constants::DAY_TO_SECONDS;
        propagate_lagrangian(&self.reference_state, elapsed_seconds, self.central_mu)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn metadata(&self) -> EphemerisMetadata {
        self.metadata.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{ASTRONOMICAL_UNIT, EARTH_ORBITAL_VELOCITY, MU_SUN, SECONDS_TO_DAY};
    use std::sync::Arc;

    #[test]
    fn circular_state_repeats_after_one_period() {
        let initial = [
            ASTRONOMICAL_UNIT,
            0.0,
            0.0,
            0.0,
            EARTH_ORBITAL_VELOCITY,
            0.0,
        ];
        let provider = KeplerianEphemeris::from_state(
            Epoch::default(),
            initial,
            MU_SUN,
            "Earth-like",
            None,
            None,
            None,
        )
        .unwrap();
        let period = provider.period(0.0).unwrap().unwrap();
        let state = provider.state(period * SECONDS_TO_DAY).unwrap();
        for index in 0..3 {
            assert!((state[index] - initial[index]).abs() / ASTRONOMICAL_UNIT < 2e-13);
            assert!((state[index + 3] - initial[index + 3]).abs() / EARTH_ORBITAL_VELOCITY < 2e-13);
        }
    }

    #[test]
    fn shared_provider_is_thread_safe_and_deterministic() {
        let provider = Arc::new(
            KeplerianEphemeris::from_classical(
                Epoch::default(),
                ClassicalElements::new(3.0, 0.2, 0.4, 0.3, 0.2, 0.1),
                1.0,
                "test",
                Some(0.01),
                Some(0.02),
                Some(0.03),
            )
            .unwrap(),
        );
        let expected = provider.states(&[0.0, 0.1, 0.2]).unwrap();
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let provider = Arc::clone(&provider);
                std::thread::spawn(move || provider.states(&[0.0, 0.1, 0.2]).unwrap())
            })
            .collect();
        for thread in threads {
            assert_eq!(thread.join().unwrap(), expected);
        }
    }
}
