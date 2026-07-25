// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

use crate::error::to_python;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pykep_core::astro::elements::ClassicalElements;
use pykep_core::ephemeris::{
    ElementRepresentation, Ephemeris, JplLowPrecision, KeplerianEphemeris,
};
use pykep_core::time::epoch::Epoch;
use pykep_core::{CartesianState, PykepError};
use pyo3::prelude::*;
use std::sync::Arc;

fn fixed<const N: usize>(values: Vec<f64>) -> Result<[f64; N], PykepError> {
    values
        .try_into()
        .map_err(|values: Vec<f64>| PykepError::DimensionMismatch {
            expected: N,
            actual: values.len(),
        })
}

fn parse_representation(value: &str) -> PyResult<ElementRepresentation> {
    match value {
        "classical_true" => Ok(ElementRepresentation::ClassicalTrue),
        "classical_mean" => Ok(ElementRepresentation::ClassicalMean),
        "modified_equinoctial" => Ok(ElementRepresentation::ModifiedEquinoctial),
        "modified_equinoctial_retrograde" => {
            Ok(ElementRepresentation::ModifiedEquinoctialRetrograde)
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "representation must be classical_true, classical_mean, \
             modified_equinoctial, or modified_equinoctial_retrograde",
        )),
    }
}

/// Thread-safe owner of an ephemeris provider.
#[pyclass(name = "Planet", frozen)]
struct PyPlanet {
    inner: Arc<dyn Ephemeris>,
}

#[pymethods]
impl PyPlanet {
    /// Construct a JPL low-precision heliocentric planetary ephemeris.
    #[staticmethod]
    #[pyo3(signature = (name, safe_radius=None))]
    fn jpl_low_precision(name: &str, safe_radius: Option<f64>) -> PyResult<Self> {
        let mut provider = JplLowPrecision::new(name).map_err(to_python)?;
        if let Some(value) = safe_radius {
            provider.set_safe_radius(value).map_err(to_python)?;
        }
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// Return the canonical names supported by the JPL low-precision model.
    #[staticmethod]
    fn jpl_supported_bodies() -> Vec<&'static str> {
        JplLowPrecision::supported_bodies().into()
    }

    /// Construct a Keplerian planet from a reference Cartesian state.
    #[staticmethod]
    #[pyo3(signature = (reference_epoch_mjd2000, state, central_mu, name="Unknown", body_mu=None, radius=None, safe_radius=None))]
    fn keplerian_from_state(
        reference_epoch_mjd2000: f64,
        state: Vec<f64>,
        central_mu: f64,
        name: &str,
        body_mu: Option<f64>,
        radius: Option<f64>,
        safe_radius: Option<f64>,
    ) -> PyResult<Self> {
        let provider = KeplerianEphemeris::from_state(
            Epoch::from_mjd2000(reference_epoch_mjd2000).map_err(to_python)?,
            fixed(state).map_err(to_python)?,
            central_mu,
            name,
            body_mu,
            radius,
            safe_radius,
        )
        .map_err(to_python)?;
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// Construct a Keplerian planet from classical true-anomaly elements.
    #[staticmethod]
    #[pyo3(signature = (reference_epoch_mjd2000, elements, central_mu, name="Unknown", body_mu=None, radius=None, safe_radius=None))]
    fn keplerian_from_classical(
        reference_epoch_mjd2000: f64,
        elements: Vec<f64>,
        central_mu: f64,
        name: &str,
        body_mu: Option<f64>,
        radius: Option<f64>,
        safe_radius: Option<f64>,
    ) -> PyResult<Self> {
        let provider = KeplerianEphemeris::from_classical(
            Epoch::from_mjd2000(reference_epoch_mjd2000).map_err(to_python)?,
            ClassicalElements::from(fixed(elements).map_err(to_python)?),
            central_mu,
            name,
            body_mu,
            radius,
            safe_radius,
        )
        .map_err(to_python)?;
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// Evaluate `[x,y,z,vx,vy,vz]` at an MJD2000 epoch.
    fn state(&self, epoch_mjd2000: f64) -> PyResult<CartesianState> {
        self.inner.state(epoch_mjd2000).map_err(to_python)
    }

    /// Evaluate an epoch array while releasing the Python GIL.
    fn states<'py>(
        &self,
        python: Python<'py>,
        epochs_mjd2000: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let epochs: Vec<_> = epochs_mjd2000.as_array().iter().copied().collect();
        let provider = Arc::clone(&self.inner);
        let states = python
            .detach(move || provider.states(&epochs))
            .map_err(to_python)?;
        let count = states.len();
        let flat = states.into_iter().flatten().collect();
        let array = Array2::from_shape_vec((count, 6), flat)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        Ok(array.into_pyarray(python))
    }

    /// Evaluate optional Cartesian acceleration.
    fn acceleration(&self, epoch_mjd2000: f64) -> PyResult<[f64; 3]> {
        self.inner.acceleration(epoch_mjd2000).map_err(to_python)
    }

    /// Return osculating elements in the named representation.
    #[pyo3(signature = (epoch_mjd2000, representation="classical_true"))]
    fn elements(&self, epoch_mjd2000: f64, representation: &str) -> PyResult<[f64; 6]> {
        self.inner
            .elements(epoch_mjd2000, parse_representation(representation)?)
            .map_err(to_python)
    }

    /// Return orbital period in seconds, or `None` for a hyperbolic orbit.
    fn period(&self, epoch_mjd2000: f64) -> PyResult<Option<f64>> {
        self.inner.period(epoch_mjd2000).map_err(to_python)
    }

    /// Provider/body name.
    #[getter]
    fn name(&self) -> String {
        self.inner.name().into()
    }

    /// Central-body gravitational parameter when available.
    #[getter]
    fn central_mu(&self) -> Option<f64> {
        self.inner.metadata().central_mu
    }

    /// Provider-body gravitational parameter when available.
    #[getter]
    fn body_mu(&self) -> Option<f64> {
        self.inner.metadata().body_mu
    }

    /// Physical radius when available.
    #[getter]
    fn radius(&self) -> Option<f64> {
        self.inner.metadata().radius
    }

    /// Safe encounter radius when available.
    #[getter]
    fn safe_radius(&self) -> Option<f64> {
        self.inner.metadata().safe_radius
    }

    /// Return whether this provider has native acceleration support.
    fn has_acceleration(&self) -> bool {
        self.inner.acceleration(0.0).is_ok()
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPlanet>()
}
