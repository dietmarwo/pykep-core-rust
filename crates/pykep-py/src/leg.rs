// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

use crate::error::to_python;
use pykep_core::leg::{
    SimsFlanaganAlphaLeg as CoreAlphaLeg, SimsFlanaganLeg as CoreLeg, SimsFlanaganMismatchJacobian,
    SimsFlanaganSettings, SpacecraftEndpoint,
};
use pykep_core::{CartesianState, PykepError, Vector3};
use pyo3::prelude::*;

fn six(values: Vec<f64>) -> Result<CartesianState, PykepError> {
    values
        .try_into()
        .map_err(|values: Vec<f64>| PykepError::DimensionMismatch {
            expected: 6,
            actual: values.len(),
        })
}

fn controls(values: Vec<Vec<f64>>) -> Result<Vec<Vector3>, PykepError> {
    values
        .into_iter()
        .map(|row| {
            row.try_into()
                .map_err(|row: Vec<f64>| PykepError::DimensionMismatch {
                    expected: 3,
                    actual: row.len(),
                })
        })
        .collect()
}

fn endpoint(state: Vec<f64>, mass: f64) -> Result<SpacecraftEndpoint, PykepError> {
    SpacecraftEndpoint::new(six(state)?, mass)
}

fn settings(
    time_of_flight: f64,
    maximum_thrust: f64,
    exhaust_velocity: f64,
    mu: f64,
    cut: f64,
) -> Result<SimsFlanaganSettings, PykepError> {
    SimsFlanaganSettings::new(time_of_flight, maximum_thrust, exhaust_velocity, mu, cut)
}

fn rows<const N: usize>(matrix: [[f64; N]; N]) -> Vec<Vec<f64>> {
    matrix.into_iter().map(Vec::from).collect()
}

type PythonMismatchJacobian = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>);

fn jacobian_rows(jacobian: SimsFlanaganMismatchJacobian) -> PythonMismatchJacobian {
    (
        rows(jacobian.departure),
        rows(jacobian.arrival),
        jacobian.controls_and_time,
    )
}

/// Fixed-duration Sims–Flanagan low-thrust leg.
#[pyclass(name = "SimsFlanaganLeg", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PySimsFlanaganLeg {
    inner: CoreLeg,
}

#[pymethods]
impl PySimsFlanaganLeg {
    /// Construct a validated fixed-duration leg.
    #[new]
    #[pyo3(signature = (
        departure_state,
        departure_mass,
        throttles,
        arrival_state,
        arrival_mass,
        time_of_flight,
        maximum_thrust,
        exhaust_velocity,
        mu,
        cut = 0.5
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        departure_state: Vec<f64>,
        departure_mass: f64,
        throttles: Vec<Vec<f64>>,
        arrival_state: Vec<f64>,
        arrival_mass: f64,
        time_of_flight: f64,
        maximum_thrust: f64,
        exhaust_velocity: f64,
        mu: f64,
        cut: f64,
    ) -> PyResult<Self> {
        CoreLeg::new(
            endpoint(departure_state, departure_mass).map_err(to_python)?,
            controls(throttles).map_err(to_python)?,
            endpoint(arrival_state, arrival_mass).map_err(to_python)?,
            settings(time_of_flight, maximum_thrust, exhaust_velocity, mu, cut)
                .map_err(to_python)?,
        )
        .map(|inner| Self { inner })
        .map_err(to_python)
    }

    /// Evaluate mismatch `[delta_r, delta_v, delta_m]`.
    fn mismatch_constraints(&self) -> PyResult<[f64; 7]> {
        self.inner.mismatch_constraints().map_err(to_python)
    }

    /// Evaluate one `dot(u,u) - 1` constraint per segment.
    fn throttle_constraints(&self) -> Vec<f64> {
        self.inner.throttle_constraints()
    }

    /// Return departure, arrival, and control/time mismatch Jacobian rows.
    fn mismatch_jacobian(&self) -> PyResult<PythonMismatchJacobian> {
        self.inner
            .mismatch_jacobian()
            .map(jacobian_rows)
            .map_err(to_python)
    }

    /// Return the throttle-constraint Jacobian.
    fn throttle_jacobian(&self) -> Vec<Vec<f64>> {
        self.inner.throttle_jacobian()
    }

    /// Total segment count.
    #[getter]
    fn segment_count(&self) -> usize {
        self.inner.segment_count()
    }

    /// Segments propagated from departure.
    #[getter]
    fn forward_segment_count(&self) -> usize {
        self.inner.forward_segment_count()
    }

    /// Segments propagated backward from arrival.
    #[getter]
    fn backward_segment_count(&self) -> usize {
        self.inner.backward_segment_count()
    }

    /// Cut fraction.
    #[getter]
    fn cut(&self) -> f64 {
        self.inner.settings().cut
    }

    /// Complete time of flight.
    #[getter]
    fn time_of_flight(&self) -> f64 {
        self.inner.settings().time_of_flight
    }
}

/// Variable-duration Sims–Flanagan low-thrust leg.
#[pyclass(name = "SimsFlanaganAlphaLeg", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PySimsFlanaganAlphaLeg {
    inner: CoreAlphaLeg,
}

#[pymethods]
impl PySimsFlanaganAlphaLeg {
    /// Construct a validated leg from direct segment durations.
    #[new]
    #[pyo3(signature = (
        departure_state,
        departure_mass,
        throttles,
        segment_durations,
        arrival_state,
        arrival_mass,
        time_of_flight,
        maximum_thrust,
        exhaust_velocity,
        mu,
        cut = 0.5
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        departure_state: Vec<f64>,
        departure_mass: f64,
        throttles: Vec<Vec<f64>>,
        segment_durations: Vec<f64>,
        arrival_state: Vec<f64>,
        arrival_mass: f64,
        time_of_flight: f64,
        maximum_thrust: f64,
        exhaust_velocity: f64,
        mu: f64,
        cut: f64,
    ) -> PyResult<Self> {
        CoreAlphaLeg::new(
            endpoint(departure_state, departure_mass).map_err(to_python)?,
            controls(throttles).map_err(to_python)?,
            segment_durations,
            endpoint(arrival_state, arrival_mass).map_err(to_python)?,
            settings(time_of_flight, maximum_thrust, exhaust_velocity, mu, cut)
                .map_err(to_python)?,
        )
        .map(|inner| Self { inner })
        .map_err(to_python)
    }

    /// Construct a leg after normalizing positive weights to the time of
    /// flight.
    #[staticmethod]
    #[pyo3(signature = (
        departure_state,
        departure_mass,
        throttles,
        time_weights,
        arrival_state,
        arrival_mass,
        time_of_flight,
        maximum_thrust,
        exhaust_velocity,
        mu,
        cut = 0.5
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_time_weights(
        departure_state: Vec<f64>,
        departure_mass: f64,
        throttles: Vec<Vec<f64>>,
        time_weights: Vec<f64>,
        arrival_state: Vec<f64>,
        arrival_mass: f64,
        time_of_flight: f64,
        maximum_thrust: f64,
        exhaust_velocity: f64,
        mu: f64,
        cut: f64,
    ) -> PyResult<Self> {
        CoreAlphaLeg::from_time_weights(
            endpoint(departure_state, departure_mass).map_err(to_python)?,
            controls(throttles).map_err(to_python)?,
            time_weights,
            endpoint(arrival_state, arrival_mass).map_err(to_python)?,
            settings(time_of_flight, maximum_thrust, exhaust_velocity, mu, cut)
                .map_err(to_python)?,
        )
        .map(|inner| Self { inner })
        .map_err(to_python)
    }

    /// Evaluate mismatch `[delta_r, delta_v, delta_m]`.
    fn mismatch_constraints(&self) -> PyResult<[f64; 7]> {
        self.inner.mismatch_constraints().map_err(to_python)
    }

    /// Evaluate one `dot(u,u) - 1` constraint per segment.
    fn throttle_constraints(&self) -> Vec<f64> {
        self.inner.throttle_constraints()
    }

    /// Return the throttle-constraint Jacobian.
    fn throttle_jacobian(&self) -> Vec<Vec<f64>> {
        self.inner.throttle_jacobian()
    }

    /// Direct segment durations.
    #[getter]
    fn segment_durations(&self) -> Vec<f64> {
        self.inner.segment_durations().to_vec()
    }

    /// Total segment count.
    #[getter]
    fn segment_count(&self) -> usize {
        self.inner.segment_count()
    }

    /// Segments propagated from departure.
    #[getter]
    fn forward_segment_count(&self) -> usize {
        self.inner.forward_segment_count()
    }

    /// Segments propagated backward from arrival.
    #[getter]
    fn backward_segment_count(&self) -> usize {
        self.inner.backward_segment_count()
    }

    /// Cut fraction.
    #[getter]
    fn cut(&self) -> f64 {
        self.inner.settings().cut
    }

    /// Complete time of flight.
    #[getter]
    fn time_of_flight(&self) -> f64 {
        self.inner.settings().time_of_flight
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySimsFlanaganLeg>()?;
    module.add_class::<PySimsFlanaganAlphaLeg>()?;
    Ok(())
}
