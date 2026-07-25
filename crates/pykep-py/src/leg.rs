// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

use crate::error::to_python;
use pykep_core::dynamics::zoh::{
    ZohCr3bpDynamics, ZohEquinoctialDynamics, ZohKeplerDynamics, ZohSolarSailDynamics,
};
use pykep_core::integration::IntegratorOptions;
use pykep_core::leg::{
    SimsFlanaganAlphaLeg as CoreAlphaLeg, SimsFlanaganLeg as CoreLeg, SimsFlanaganMismatchJacobian,
    SimsFlanaganSettings, SpacecraftEndpoint, ZohCr3bpLeg, ZohEquinoctialLeg, ZohKeplerLeg,
    ZohLegHistory, ZohLegMismatchJacobian, ZohSolarSailLeg,
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

fn fixed_controls<const C: usize>(values: Vec<Vec<f64>>) -> Result<Vec<[f64; C]>, PykepError> {
    values
        .into_iter()
        .map(|row| {
            row.try_into()
                .map_err(|row: Vec<f64>| PykepError::DimensionMismatch {
                    expected: C,
                    actual: row.len(),
                })
        })
        .collect()
}

fn fixed<const N: usize>(values: Vec<f64>) -> Result<[f64; N], PykepError> {
    values
        .try_into()
        .map_err(|values: Vec<f64>| PykepError::DimensionMismatch {
            expected: N,
            actual: values.len(),
        })
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
    module.add_class::<PyZohModel>()?;
    module.add_class::<PyZohLeg>()?;
    Ok(())
}

/// Built-in dynamics for a generic ZOH leg.
#[pyclass(name = "ZohModel", eq, eq_int, frozen, skip_from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PyZohModel {
    /// Seven-state normalized Kepler low-thrust dynamics.
    Kepler = 0,
    /// Seven-state CR3BP low-thrust dynamics.
    Cr3bp = 1,
    /// Seven-state modified-equinoctial low-thrust dynamics.
    Equinoctial = 2,
    /// Six-state ideal solar-sail dynamics.
    SolarSail = 3,
}

#[derive(Clone)]
enum CoreZohLeg {
    Kepler(ZohKeplerLeg),
    Cr3bp(ZohCr3bpLeg),
    Equinoctial(ZohEquinoctialLeg),
    SolarSail(ZohSolarSailLeg),
}

type PythonZohJacobian = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>);
type PythonZohHistory = (Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>);

impl CoreZohLeg {
    fn mismatch_constraints(&self) -> Result<Vec<f64>, PykepError> {
        match self {
            Self::Kepler(leg) => leg.mismatch_constraints().map(Vec::from),
            Self::Cr3bp(leg) => leg.mismatch_constraints().map(Vec::from),
            Self::Equinoctial(leg) => leg.mismatch_constraints().map(Vec::from),
            Self::SolarSail(leg) => leg.mismatch_constraints().map(Vec::from),
        }
    }

    fn mismatch_jacobian(&self) -> Result<PythonZohJacobian, PykepError> {
        fn convert<const N: usize>(value: ZohLegMismatchJacobian<N>) -> PythonZohJacobian {
            (
                rows(value.initial_state),
                rows(value.final_state),
                value.controls,
                value.time_grid,
            )
        }
        match self {
            Self::Kepler(leg) => leg.mismatch_jacobian().map(convert),
            Self::Cr3bp(leg) => leg.mismatch_jacobian().map(convert),
            Self::Equinoctial(leg) => leg.mismatch_jacobian().map(convert),
            Self::SolarSail(leg) => leg.mismatch_jacobian().map(convert),
        }
    }

    fn state_history(&self, samples: usize) -> Result<PythonZohHistory, PykepError> {
        fn convert<const N: usize>(history: ZohLegHistory<N>) -> PythonZohHistory {
            let side = |segments: Vec<Vec<[f64; N]>>| {
                segments
                    .into_iter()
                    .map(|segment| segment.into_iter().map(Vec::from).collect())
                    .collect()
            };
            (side(history.forward), side(history.backward))
        }
        match self {
            Self::Kepler(leg) => leg.state_history(samples).map(convert),
            Self::Cr3bp(leg) => leg.state_history(samples).map(convert),
            Self::Equinoctial(leg) => leg.state_history(samples).map(convert),
            Self::SolarSail(leg) => leg.state_history(samples).map(convert),
        }
    }

    fn model(&self) -> PyZohModel {
        match self {
            Self::Kepler(_) => PyZohModel::Kepler,
            Self::Cr3bp(_) => PyZohModel::Cr3bp,
            Self::Equinoctial(_) => PyZohModel::Equinoctial,
            Self::SolarSail(_) => PyZohModel::SolarSail,
        }
    }

    fn segment_count(&self) -> usize {
        match self {
            Self::Kepler(leg) => leg.segment_count(),
            Self::Cr3bp(leg) => leg.segment_count(),
            Self::Equinoctial(leg) => leg.segment_count(),
            Self::SolarSail(leg) => leg.segment_count(),
        }
    }

    fn forward_segment_count(&self) -> usize {
        match self {
            Self::Kepler(leg) => leg.forward_segment_count(),
            Self::Cr3bp(leg) => leg.forward_segment_count(),
            Self::Equinoctial(leg) => leg.forward_segment_count(),
            Self::SolarSail(leg) => leg.forward_segment_count(),
        }
    }
}

/// Generic zero-order-hold low-thrust leg using a built-in native model.
#[pyclass(name = "ZohLeg", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyZohLeg {
    inner: CoreZohLeg,
}

#[pymethods]
impl PyZohLeg {
    /// Construct a validated ZOH leg.
    #[new]
    #[pyo3(signature = (
        model,
        initial_state,
        controls,
        final_state,
        time_grid,
        constants,
        cut = 0.5,
        relative_tolerance = 1e-12,
        absolute_tolerance = 1e-12,
        maximum_step = None,
        maximum_steps = 100_000
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: PyRef<'_, PyZohModel>,
        initial_state: Vec<f64>,
        controls: Vec<Vec<f64>>,
        final_state: Vec<f64>,
        time_grid: Vec<f64>,
        constants: Vec<f64>,
        cut: f64,
        relative_tolerance: f64,
        absolute_tolerance: f64,
        maximum_step: Option<f64>,
        maximum_steps: usize,
    ) -> PyResult<Self> {
        let options = IntegratorOptions {
            relative_tolerance,
            absolute_tolerance,
            maximum_step,
            maximum_steps,
            ..IntegratorOptions::default()
        };
        let inner = match *model {
            PyZohModel::Kepler => CoreZohLeg::Kepler(
                ZohKeplerLeg::new(
                    ZohKeplerDynamics,
                    fixed(initial_state).map_err(to_python)?,
                    fixed_controls(controls).map_err(to_python)?,
                    fixed(final_state).map_err(to_python)?,
                    time_grid,
                    fixed(constants).map_err(to_python)?,
                    cut,
                    options,
                )
                .map_err(to_python)?,
            ),
            PyZohModel::Cr3bp => CoreZohLeg::Cr3bp(
                ZohCr3bpLeg::new(
                    ZohCr3bpDynamics,
                    fixed(initial_state).map_err(to_python)?,
                    fixed_controls(controls).map_err(to_python)?,
                    fixed(final_state).map_err(to_python)?,
                    time_grid,
                    fixed(constants).map_err(to_python)?,
                    cut,
                    options,
                )
                .map_err(to_python)?,
            ),
            PyZohModel::Equinoctial => CoreZohLeg::Equinoctial(
                ZohEquinoctialLeg::new(
                    ZohEquinoctialDynamics,
                    fixed(initial_state).map_err(to_python)?,
                    fixed_controls(controls).map_err(to_python)?,
                    fixed(final_state).map_err(to_python)?,
                    time_grid,
                    fixed(constants).map_err(to_python)?,
                    cut,
                    options,
                )
                .map_err(to_python)?,
            ),
            PyZohModel::SolarSail => CoreZohLeg::SolarSail(
                ZohSolarSailLeg::new(
                    ZohSolarSailDynamics,
                    fixed(initial_state).map_err(to_python)?,
                    fixed_controls(controls).map_err(to_python)?,
                    fixed(final_state).map_err(to_python)?,
                    time_grid,
                    fixed(constants).map_err(to_python)?,
                    cut,
                    options,
                )
                .map_err(to_python)?,
            ),
        };
        Ok(Self { inner })
    }

    /// Evaluate `forward_state - backward_state` while releasing the GIL.
    fn mismatch_constraints(&self, python: Python<'_>) -> PyResult<Vec<f64>> {
        let inner = self.inner.clone();
        python
            .detach(move || inner.mismatch_constraints())
            .map_err(to_python)
    }

    /// Return initial, final, control, and time-grid Jacobian rows.
    fn mismatch_jacobian(&self, python: Python<'_>) -> PyResult<PythonZohJacobian> {
        let inner = self.inner.clone();
        python
            .detach(move || inner.mismatch_jacobian())
            .map_err(to_python)
    }

    /// Sample every propagated segment, releasing the GIL.
    #[pyo3(signature = (samples_per_segment = 2))]
    fn state_history(
        &self,
        python: Python<'_>,
        samples_per_segment: usize,
    ) -> PyResult<PythonZohHistory> {
        let inner = self.inner.clone();
        python
            .detach(move || inner.state_history(samples_per_segment))
            .map_err(to_python)
    }

    /// Evaluate multiple legs in input order while releasing the GIL.
    #[staticmethod]
    fn mismatch_constraints_batch(
        python: Python<'_>,
        legs: Vec<PyRef<'_, Self>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let legs = legs
            .into_iter()
            .map(|leg| leg.inner.clone())
            .collect::<Vec<_>>();
        python
            .detach(move || {
                legs.iter()
                    .map(CoreZohLeg::mismatch_constraints)
                    .collect::<Result<Vec<_>, PykepError>>()
            })
            .map_err(to_python)
    }

    /// Selected built-in dynamics.
    #[getter]
    fn model(&self) -> PyZohModel {
        self.inner.model()
    }

    /// State dimension.
    #[getter]
    fn state_dimension(&self) -> usize {
        if self.inner.model() == PyZohModel::SolarSail {
            6
        } else {
            7
        }
    }

    /// Control dimension.
    #[getter]
    fn control_dimension(&self) -> usize {
        if self.inner.model() == PyZohModel::SolarSail {
            2
        } else {
            4
        }
    }

    /// Total number of segments.
    #[getter]
    fn segment_count(&self) -> usize {
        self.inner.segment_count()
    }

    /// Number of forward segments.
    #[getter]
    fn forward_segment_count(&self) -> usize {
        self.inner.forward_segment_count()
    }

    /// Number of backward segments.
    #[getter]
    fn backward_segment_count(&self) -> usize {
        self.inner.segment_count() - self.inner.forward_segment_count()
    }
}
