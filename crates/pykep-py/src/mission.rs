// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

use crate::error::to_python;
use pykep_core::astro::encodings::{
    alpha_to_direct, direct_to_alpha, direct_to_eta, eta_to_direct,
};
use pykep_core::astro::flyby::{
    flyby_constraints, flyby_constraints_jacobian, flyby_delta_v, flyby_outgoing_velocity,
};
use pykep_core::astro::lambert::{
    LambertPath as CoreLambertPath, LambertProblem as CoreLambertProblem,
};
use pykep_core::astro::mima::{mima, mima2};
use pykep_core::astro::transfers::{bielliptic, hohmann};
use pykep_core::{PykepError, Vector3};
use pyo3::prelude::*;

fn fixed<const N: usize>(values: Vec<f64>) -> Result<[f64; N], PykepError> {
    values
        .try_into()
        .map_err(|values: Vec<f64>| PykepError::DimensionMismatch {
            expected: N,
            actual: values.len(),
        })
}

/// One deterministic branch of a Lambert solution family.
#[pyclass(name = "LambertSolution", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyLambertSolution {
    #[pyo3(get)]
    departure_velocity: Vector3,
    #[pyo3(get)]
    arrival_velocity: Vector3,
    #[pyo3(get)]
    x: f64,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    revolutions: usize,
    #[pyo3(get)]
    path: &'static str,
}

/// Solved single- or multi-revolution Lambert boundary-value problem.
#[pyclass(name = "LambertProblem", frozen)]
struct PyLambertProblem {
    inner: CoreLambertProblem,
}

#[pymethods]
impl PyLambertProblem {
    /// Solve a Lambert problem, returning zero/left/right branches in order.
    #[new]
    #[pyo3(signature = (initial_position, final_position, time, mu, clockwise=false, maximum_revolutions=1))]
    fn new(
        initial_position: Vec<f64>,
        final_position: Vec<f64>,
        time: f64,
        mu: f64,
        clockwise: bool,
        maximum_revolutions: usize,
    ) -> PyResult<Self> {
        CoreLambertProblem::new(
            fixed(initial_position).map_err(to_python)?,
            fixed(final_position).map_err(to_python)?,
            time,
            mu,
            clockwise,
            maximum_revolutions,
        )
        .map(|inner| Self { inner })
        .map_err(to_python)
    }

    /// Ordered zero-revolution then left/right multi-revolution solutions.
    #[getter]
    fn solutions(&self) -> Vec<PyLambertSolution> {
        self.inner
            .solutions()
            .iter()
            .map(|solution| PyLambertSolution {
                departure_velocity: solution.departure_velocity,
                arrival_velocity: solution.arrival_velocity,
                x: solution.x,
                iterations: solution.iterations,
                revolutions: solution.revolutions,
                path: match solution.path {
                    CoreLambertPath::ZeroRevolution => "zero",
                    CoreLambertPath::Left => "left",
                    CoreLambertPath::Right => "right",
                },
            })
            .collect()
    }

    /// Maximum revolution count with returned solutions.
    #[getter]
    fn maximum_revolutions(&self) -> usize {
        self.inner.maximum_revolutions()
    }

    /// Initial position.
    #[getter]
    fn initial_position(&self) -> Vector3 {
        self.inner.initial_position()
    }

    /// Final position.
    #[getter]
    fn final_position(&self) -> Vector3 {
        self.inner.final_position()
    }

    /// Time of flight.
    #[getter]
    fn time(&self) -> f64 {
        self.inner.time()
    }

    /// Gravitational parameter.
    #[getter]
    fn mu(&self) -> f64 {
        self.inner.mu()
    }

    /// Whether clockwise/retrograde motion was requested.
    #[getter]
    fn clockwise(&self) -> bool {
        self.inner.clockwise()
    }
}

/// Return `(total_delta_v, transfer_time, impulses)` for a Hohmann transfer.
#[pyfunction(name = "hohmann")]
fn hohmann_py(r1: f64, r2: f64, mu: f64) -> PyResult<(f64, f64, Vec<f64>)> {
    hohmann(r1, r2, mu)
        .map(|result| (result.delta_v, result.time, result.impulses.into()))
        .map_err(to_python)
}

/// Return `(total_delta_v, transfer_time, impulses)` for a bi-elliptic transfer.
#[pyfunction(name = "bielliptic")]
fn bielliptic_py(r1: f64, r2: f64, rb: f64, mu: f64) -> PyResult<(f64, f64, Vec<f64>)> {
    bielliptic(r1, r2, rb, mu)
        .map(|result| (result.delta_v, result.time, result.impulses.into()))
        .map_err(to_python)
}

/// Decode alpha time variables.
#[pyfunction(name = "alpha_to_direct")]
fn alpha_to_direct_py(alphas: Vec<f64>, total_time: f64) -> PyResult<Vec<f64>> {
    alpha_to_direct(&alphas, total_time).map_err(to_python)
}

/// Encode direct leg times, returning `(alphas, total_time)`.
#[pyfunction(name = "direct_to_alpha")]
fn direct_to_alpha_py(times: Vec<f64>) -> PyResult<(Vec<f64>, f64)> {
    direct_to_alpha(&times).map_err(to_python)
}

/// Decode eta time variables.
#[pyfunction(name = "eta_to_direct")]
fn eta_to_direct_py(etas: Vec<f64>, maximum_time: f64) -> PyResult<Vec<f64>> {
    eta_to_direct(&etas, maximum_time).map_err(to_python)
}

/// Encode direct leg times as eta variables.
#[pyfunction(name = "direct_to_eta")]
fn direct_to_eta_py(times: Vec<f64>, maximum_time: f64) -> PyResult<Vec<f64>> {
    direct_to_eta(&times, maximum_time).map_err(to_python)
}

/// Return speed-equality and minimum-turn flyby constraints.
#[pyfunction(name = "flyby_constraints")]
fn flyby_constraints_py(
    incoming: Vec<f64>,
    outgoing: Vec<f64>,
    mu: f64,
    safe_radius: f64,
) -> PyResult<[f64; 2]> {
    flyby_constraints(
        &fixed(incoming).map_err(to_python)?,
        &fixed(outgoing).map_err(to_python)?,
        mu,
        safe_radius,
    )
    .map_err(to_python)
}

/// Return the row-major two-by-six flyby constraint Jacobian.
#[pyfunction(name = "flyby_constraints_jacobian")]
fn flyby_constraints_jacobian_py(
    incoming: Vec<f64>,
    outgoing: Vec<f64>,
    mu: f64,
    safe_radius: f64,
) -> PyResult<Vec<Vec<f64>>> {
    flyby_constraints_jacobian(
        &fixed(incoming).map_err(to_python)?,
        &fixed(outgoing).map_err(to_python)?,
        mu,
        safe_radius,
    )
    .map(|matrix| matrix.into_iter().map(Vec::from).collect())
    .map_err(to_python)
}

/// Return the minimum powered-flyby delta-v.
#[pyfunction(name = "flyby_delta_v")]
fn flyby_delta_v_py(
    incoming: Vec<f64>,
    outgoing: Vec<f64>,
    mu: f64,
    safe_radius: f64,
) -> PyResult<f64> {
    flyby_delta_v(
        &fixed(incoming).map_err(to_python)?,
        &fixed(outgoing).map_err(to_python)?,
        mu,
        safe_radius,
    )
    .map_err(to_python)
}

/// Map an incoming inertial velocity through an unpowered flyby.
#[pyfunction(name = "flyby_outgoing_velocity")]
fn flyby_outgoing_velocity_py(
    incoming: Vec<f64>,
    planet_velocity: Vec<f64>,
    periapsis_radius: f64,
    beta: f64,
    mu: f64,
) -> PyResult<Vector3> {
    flyby_outgoing_velocity(
        &fixed(incoming).map_err(to_python)?,
        &fixed(planet_velocity).map_err(to_python)?,
        periapsis_radius,
        beta,
        mu,
    )
    .map_err(to_python)
}

/// Return `(maximum_mass, characteristic_acceleration)` from MIMA.
#[pyfunction(name = "mima")]
fn mima_py(
    departure_delta_v: Vec<f64>,
    arrival_delta_v: Vec<f64>,
    time: f64,
    maximum_thrust: f64,
    effective_exhaust_velocity: f64,
) -> PyResult<(f64, f64)> {
    mima(
        &fixed(departure_delta_v).map_err(to_python)?,
        &fixed(arrival_delta_v).map_err(to_python)?,
        time,
        maximum_thrust,
        effective_exhaust_velocity,
    )
    .map(|result| (result.mass, result.acceleration))
    .map_err(to_python)
}

/// Return the STM-based `(maximum_mass, acceleration)` MIMA2 estimate.
#[pyfunction(name = "mima2")]
fn mima2_py(
    initial_state: Vec<f64>,
    departure_delta_v: Vec<f64>,
    arrival_delta_v: Vec<f64>,
    time: f64,
    maximum_thrust: f64,
    effective_exhaust_velocity: f64,
    mu: f64,
) -> PyResult<(f64, f64)> {
    mima2(
        &fixed::<6>(initial_state).map_err(to_python)?,
        &fixed(departure_delta_v).map_err(to_python)?,
        &fixed(arrival_delta_v).map_err(to_python)?,
        time,
        maximum_thrust,
        effective_exhaust_velocity,
        mu,
    )
    .map(|result| (result.mass, result.acceleration))
    .map_err(to_python)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLambertSolution>()?;
    module.add_class::<PyLambertProblem>()?;
    module.add_function(wrap_pyfunction!(hohmann_py, module)?)?;
    module.add_function(wrap_pyfunction!(bielliptic_py, module)?)?;
    module.add_function(wrap_pyfunction!(alpha_to_direct_py, module)?)?;
    module.add_function(wrap_pyfunction!(direct_to_alpha_py, module)?)?;
    module.add_function(wrap_pyfunction!(eta_to_direct_py, module)?)?;
    module.add_function(wrap_pyfunction!(direct_to_eta_py, module)?)?;
    module.add_function(wrap_pyfunction!(flyby_constraints_py, module)?)?;
    module.add_function(wrap_pyfunction!(flyby_constraints_jacobian_py, module)?)?;
    module.add_function(wrap_pyfunction!(flyby_delta_v_py, module)?)?;
    module.add_function(wrap_pyfunction!(flyby_outgoing_velocity_py, module)?)?;
    module.add_function(wrap_pyfunction!(mima_py, module)?)?;
    module.add_function(wrap_pyfunction!(mima2_py, module)?)?;
    Ok(())
}
