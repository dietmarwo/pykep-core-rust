// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

use crate::error::to_python;
use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pykep_core::astro::encodings::{
    alpha_to_direct, direct_to_alpha, direct_to_eta, eta_to_direct,
};
use pykep_core::astro::flyby::{
    flyby_constraints, flyby_constraints_jacobian, flyby_delta_v, flyby_outgoing_velocity,
};
use pykep_core::astro::lambert::{
    LambertPath as CoreLambertPath, LambertProblem as CoreLambertProblem, LambertRequest,
    solve_lambert_batch,
};
use pykep_core::astro::mima::{mima, mima2};
use pykep_core::astro::transfers::{bielliptic, hohmann};
use pykep_core::{PykepError, Vector3};
use pyo3::prelude::*;

type PyTransferBatch<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
);
type PyScalarPairBatch<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

fn fixed<const N: usize>(values: Vec<f64>) -> Result<[f64; N], PykepError> {
    values
        .try_into()
        .map_err(|values: Vec<f64>| PykepError::DimensionMismatch {
            expected: N,
            actual: values.len(),
        })
}

fn fixed_rows<const N: usize>(values: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; N]>> {
    let shape = values.shape();
    if shape[1] != N {
        return Err(to_python(PykepError::DimensionMismatch {
            expected: N,
            actual: shape[1],
        }));
    }
    Ok(values
        .as_array()
        .rows()
        .into_iter()
        .map(|row| core::array::from_fn(|index| row[index]))
        .collect())
}

fn matching_length(expected: usize, actual: usize) -> PyResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(to_python(PykepError::DimensionMismatch {
            expected,
            actual,
        }))
    }
}

/// One deterministic branch of a Lambert solution family.
#[pyclass(name = "LambertSolution", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyLambertSolution {
    departure_velocity: Vector3,
    arrival_velocity: Vector3,
    x: f64,
    iterations: usize,
    revolutions: usize,
    path: &'static str,
}

#[pymethods]
impl PyLambertSolution {
    /// Departure velocity at the initial position.
    #[getter]
    fn departure_velocity(&self) -> Vector3 {
        self.departure_velocity
    }

    /// Arrival velocity at the final position.
    #[getter]
    fn arrival_velocity(&self) -> Vector3 {
        self.arrival_velocity
    }

    /// Izzo solver variable for this branch.
    #[getter]
    fn x(&self) -> f64 {
        self.x
    }

    /// Householder iterations used for this branch.
    #[getter]
    fn iterations(&self) -> usize {
        self.iterations
    }

    /// Number of complete revolutions.
    #[getter]
    fn revolutions(&self) -> usize {
        self.revolutions
    }

    /// Deterministic branch name: `zero`, `left`, or `right`.
    #[getter]
    fn path(&self) -> &'static str {
        self.path
    }
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

/// Solve `N` independent Lambert problems while releasing the GIL.
#[pyfunction(signature = (initial_positions, final_positions, times, mu, clockwise=false, maximum_revolutions=1, workers=0))]
#[allow(clippy::too_many_arguments)]
fn lambert_problem_batch(
    python: Python<'_>,
    initial_positions: PyReadonlyArray2<'_, f64>,
    final_positions: PyReadonlyArray2<'_, f64>,
    times: PyReadonlyArray1<'_, f64>,
    mu: f64,
    clockwise: bool,
    maximum_revolutions: usize,
    workers: usize,
) -> PyResult<Vec<Py<PyLambertProblem>>> {
    let initial_shape = initial_positions.shape();
    let final_shape = final_positions.shape();
    if initial_shape[1] != 3 {
        return Err(to_python(PykepError::DimensionMismatch {
            expected: 3,
            actual: initial_shape[1],
        }));
    }
    if final_shape[1] != 3 {
        return Err(to_python(PykepError::DimensionMismatch {
            expected: 3,
            actual: final_shape[1],
        }));
    }
    let count = initial_shape[0];
    if final_shape[0] != count {
        return Err(to_python(PykepError::DimensionMismatch {
            expected: count,
            actual: final_shape[0],
        }));
    }
    if times.len() != count {
        return Err(to_python(PykepError::DimensionMismatch {
            expected: count,
            actual: times.len(),
        }));
    }
    let final_positions = final_positions.as_array();
    let times = times.as_array();
    let requests = initial_positions
        .as_array()
        .rows()
        .into_iter()
        .zip(final_positions.rows())
        .zip(times.iter().copied())
        .map(|((initial, destination), time)| {
            LambertRequest::new(
                [initial[0], initial[1], initial[2]],
                [destination[0], destination[1], destination[2]],
                time,
                mu,
                clockwise,
                maximum_revolutions,
            )
        })
        .collect::<Vec<_>>();
    let problems = python
        .detach(move || solve_lambert_batch(&requests, workers))
        .map_err(to_python)?;
    problems
        .into_iter()
        .map(|inner| Py::new(python, PyLambertProblem { inner }))
        .collect()
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

/// Batch-evaluate Hohmann transfers.
#[pyfunction(signature = (r1, r2, mu, workers=0))]
fn hohmann_batch<'py>(
    python: Python<'py>,
    r1: PyReadonlyArray1<'py, f64>,
    r2: PyReadonlyArray1<'py, f64>,
    mu: f64,
    workers: usize,
) -> PyResult<PyTransferBatch<'py>> {
    matching_length(r1.len(), r2.len())?;
    let inputs = r1
        .as_array()
        .iter()
        .copied()
        .zip(r2.as_array().iter().copied())
        .collect::<Vec<_>>();
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&inputs, workers, |(r1, r2)| hohmann(*r1, *r2, mu))
        })
        .map_err(to_python)?;
    let count = output.len();
    let delta_v =
        Array1::from_vec(output.iter().map(|value| value.delta_v).collect()).into_pyarray(python);
    let times =
        Array1::from_vec(output.iter().map(|value| value.time).collect()).into_pyarray(python);
    let impulses = Array2::from_shape_vec(
        (count, 2),
        output
            .into_iter()
            .flat_map(|value| value.impulses)
            .collect(),
    )
    .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?
    .into_pyarray(python);
    Ok((delta_v, times, impulses))
}

/// Batch-evaluate bi-elliptic transfers.
#[pyfunction(signature = (r1, r2, rb, mu, workers=0))]
fn bielliptic_batch<'py>(
    python: Python<'py>,
    r1: PyReadonlyArray1<'py, f64>,
    r2: PyReadonlyArray1<'py, f64>,
    rb: PyReadonlyArray1<'py, f64>,
    mu: f64,
    workers: usize,
) -> PyResult<PyTransferBatch<'py>> {
    matching_length(r1.len(), r2.len())?;
    matching_length(r1.len(), rb.len())?;
    let inputs = r1
        .as_array()
        .iter()
        .copied()
        .zip(r2.as_array().iter().copied())
        .zip(rb.as_array().iter().copied())
        .collect::<Vec<_>>();
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&inputs, workers, |((r1, r2), rb)| {
                bielliptic(*r1, *r2, *rb, mu)
            })
        })
        .map_err(to_python)?;
    let count = output.len();
    let delta_v =
        Array1::from_vec(output.iter().map(|value| value.delta_v).collect()).into_pyarray(python);
    let times =
        Array1::from_vec(output.iter().map(|value| value.time).collect()).into_pyarray(python);
    let impulses = Array2::from_shape_vec(
        (count, 3),
        output
            .into_iter()
            .flat_map(|value| value.impulses)
            .collect(),
    )
    .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?
    .into_pyarray(python);
    Ok((delta_v, times, impulses))
}

/// Batch-decode alpha time vectors.
#[pyfunction(signature = (alphas, total_time, workers=0))]
fn alpha_to_direct_batch(
    python: Python<'_>,
    alphas: Vec<Vec<f64>>,
    total_time: f64,
    workers: usize,
) -> PyResult<Vec<Vec<f64>>> {
    python
        .detach(move || {
            pykep_core::batch::try_map(&alphas, workers, |values| {
                alpha_to_direct(values, total_time)
            })
        })
        .map_err(to_python)
}

/// Batch-encode direct time vectors as alpha variables.
#[pyfunction(signature = (times, workers=0))]
fn direct_to_alpha_batch(
    python: Python<'_>,
    times: Vec<Vec<f64>>,
    workers: usize,
) -> PyResult<Vec<(Vec<f64>, f64)>> {
    python
        .detach(move || {
            pykep_core::batch::try_map(&times, workers, |values| direct_to_alpha(values))
        })
        .map_err(to_python)
}

/// Batch-decode eta time vectors.
#[pyfunction(signature = (etas, maximum_time, workers=0))]
fn eta_to_direct_batch(
    python: Python<'_>,
    etas: Vec<Vec<f64>>,
    maximum_time: f64,
    workers: usize,
) -> PyResult<Vec<Vec<f64>>> {
    python
        .detach(move || {
            pykep_core::batch::try_map(&etas, workers, |values| eta_to_direct(values, maximum_time))
        })
        .map_err(to_python)
}

/// Batch-encode direct time vectors as eta variables.
#[pyfunction(signature = (times, maximum_time, workers=0))]
fn direct_to_eta_batch(
    python: Python<'_>,
    times: Vec<Vec<f64>>,
    maximum_time: f64,
    workers: usize,
) -> PyResult<Vec<Vec<f64>>> {
    python
        .detach(move || {
            pykep_core::batch::try_map(&times, workers, |values| {
                direct_to_eta(values, maximum_time)
            })
        })
        .map_err(to_python)
}

/// Batch-evaluate flyby constraints.
#[pyfunction(signature = (incoming, outgoing, mu, safe_radius, workers=0))]
fn flyby_constraints_batch<'py>(
    python: Python<'py>,
    incoming: PyReadonlyArray2<'py, f64>,
    outgoing: PyReadonlyArray2<'py, f64>,
    mu: f64,
    safe_radius: f64,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let incoming = fixed_rows::<3>(incoming)?;
    let outgoing = fixed_rows::<3>(outgoing)?;
    matching_length(incoming.len(), outgoing.len())?;
    let inputs = incoming.into_iter().zip(outgoing).collect::<Vec<_>>();
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&inputs, workers, |(incoming, outgoing)| {
                flyby_constraints(incoming, outgoing, mu, safe_radius)
            })
        })
        .map_err(to_python)?;
    let count = output.len();
    let array = Array2::from_shape_vec((count, 2), output.into_iter().flatten().collect())
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(python))
}

/// Batch-evaluate flyby constraint Jacobians.
#[pyfunction(signature = (incoming, outgoing, mu, safe_radius, workers=0))]
fn flyby_constraints_jacobian_batch<'py>(
    python: Python<'py>,
    incoming: PyReadonlyArray2<'py, f64>,
    outgoing: PyReadonlyArray2<'py, f64>,
    mu: f64,
    safe_radius: f64,
    workers: usize,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let incoming = fixed_rows::<3>(incoming)?;
    let outgoing = fixed_rows::<3>(outgoing)?;
    matching_length(incoming.len(), outgoing.len())?;
    let inputs = incoming.into_iter().zip(outgoing).collect::<Vec<_>>();
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&inputs, workers, |(incoming, outgoing)| {
                flyby_constraints_jacobian(incoming, outgoing, mu, safe_radius)
            })
        })
        .map_err(to_python)?;
    let count = output.len();
    let array = Array3::from_shape_vec(
        (count, 2, 6),
        output
            .into_iter()
            .flat_map(|matrix| matrix.into_iter().flatten())
            .collect(),
    )
    .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(python))
}

/// Batch-evaluate powered-flyby delta-v.
#[pyfunction(signature = (incoming, outgoing, mu, safe_radius, workers=0))]
fn flyby_delta_v_batch<'py>(
    python: Python<'py>,
    incoming: PyReadonlyArray2<'py, f64>,
    outgoing: PyReadonlyArray2<'py, f64>,
    mu: f64,
    safe_radius: f64,
    workers: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let incoming = fixed_rows::<3>(incoming)?;
    let outgoing = fixed_rows::<3>(outgoing)?;
    matching_length(incoming.len(), outgoing.len())?;
    let inputs = incoming.into_iter().zip(outgoing).collect::<Vec<_>>();
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&inputs, workers, |(incoming, outgoing)| {
                flyby_delta_v(incoming, outgoing, mu, safe_radius)
            })
        })
        .map_err(to_python)?;
    Ok(Array1::from_vec(output).into_pyarray(python))
}

/// Batch-map incoming velocities through unpowered flybys.
#[pyfunction(signature = (incoming, planet_velocity, periapsis_radius, beta, mu, workers=0))]
fn flyby_outgoing_velocity_batch<'py>(
    python: Python<'py>,
    incoming: PyReadonlyArray2<'py, f64>,
    planet_velocity: PyReadonlyArray2<'py, f64>,
    periapsis_radius: PyReadonlyArray1<'py, f64>,
    beta: PyReadonlyArray1<'py, f64>,
    mu: f64,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let incoming = fixed_rows::<3>(incoming)?;
    let planet_velocity = fixed_rows::<3>(planet_velocity)?;
    let count = incoming.len();
    matching_length(count, planet_velocity.len())?;
    matching_length(count, periapsis_radius.len())?;
    matching_length(count, beta.len())?;
    let inputs = incoming
        .into_iter()
        .zip(planet_velocity)
        .zip(periapsis_radius.as_array().iter().copied())
        .zip(beta.as_array().iter().copied())
        .collect::<Vec<_>>();
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(
                &inputs,
                workers,
                |(((incoming, planet_velocity), periapsis_radius), beta)| {
                    flyby_outgoing_velocity(incoming, planet_velocity, *periapsis_radius, *beta, mu)
                },
            )
        })
        .map_err(to_python)?;
    let array = Array2::from_shape_vec((count, 3), output.into_iter().flatten().collect())
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(python))
}

/// Batch-evaluate MIMA approximations.
#[pyfunction(signature = (departure_delta_v, arrival_delta_v, times, maximum_thrust, effective_exhaust_velocity, workers=0))]
fn mima_batch<'py>(
    python: Python<'py>,
    departure_delta_v: PyReadonlyArray2<'py, f64>,
    arrival_delta_v: PyReadonlyArray2<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
    maximum_thrust: f64,
    effective_exhaust_velocity: f64,
    workers: usize,
) -> PyResult<PyScalarPairBatch<'py>> {
    let departure = fixed_rows::<3>(departure_delta_v)?;
    let arrival = fixed_rows::<3>(arrival_delta_v)?;
    let count = departure.len();
    matching_length(count, arrival.len())?;
    matching_length(count, times.len())?;
    let inputs = departure
        .into_iter()
        .zip(arrival)
        .zip(times.as_array().iter().copied())
        .collect::<Vec<_>>();
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&inputs, workers, |((departure, arrival), time)| {
                mima(
                    departure,
                    arrival,
                    *time,
                    maximum_thrust,
                    effective_exhaust_velocity,
                )
            })
        })
        .map_err(to_python)?;
    let masses =
        Array1::from_vec(output.iter().map(|value| value.mass).collect()).into_pyarray(python);
    let accelerations = Array1::from_vec(output.iter().map(|value| value.acceleration).collect())
        .into_pyarray(python);
    Ok((masses, accelerations))
}

/// Batch-evaluate STM-based MIMA2 approximations.
#[pyfunction(signature = (initial_states, departure_delta_v, arrival_delta_v, times, maximum_thrust, effective_exhaust_velocity, mu, workers=0))]
#[allow(clippy::too_many_arguments)]
fn mima2_batch<'py>(
    python: Python<'py>,
    initial_states: PyReadonlyArray2<'py, f64>,
    departure_delta_v: PyReadonlyArray2<'py, f64>,
    arrival_delta_v: PyReadonlyArray2<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
    maximum_thrust: f64,
    effective_exhaust_velocity: f64,
    mu: f64,
    workers: usize,
) -> PyResult<PyScalarPairBatch<'py>> {
    let states = fixed_rows::<6>(initial_states)?;
    let departure = fixed_rows::<3>(departure_delta_v)?;
    let arrival = fixed_rows::<3>(arrival_delta_v)?;
    let count = states.len();
    matching_length(count, departure.len())?;
    matching_length(count, arrival.len())?;
    matching_length(count, times.len())?;
    let inputs = states
        .into_iter()
        .zip(departure)
        .zip(arrival)
        .zip(times.as_array().iter().copied())
        .collect::<Vec<_>>();
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&inputs, workers, |(((state, departure), arrival), time)| {
                mima2(
                    state,
                    departure,
                    arrival,
                    *time,
                    maximum_thrust,
                    effective_exhaust_velocity,
                    mu,
                )
            })
        })
        .map_err(to_python)?;
    let masses =
        Array1::from_vec(output.iter().map(|value| value.mass).collect()).into_pyarray(python);
    let accelerations = Array1::from_vec(output.iter().map(|value| value.acceleration).collect())
        .into_pyarray(python);
    Ok((masses, accelerations))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLambertSolution>()?;
    module.add_class::<PyLambertProblem>()?;
    module.add_function(wrap_pyfunction!(lambert_problem_batch, module)?)?;
    module.add_function(wrap_pyfunction!(hohmann_py, module)?)?;
    module.add_function(wrap_pyfunction!(hohmann_batch, module)?)?;
    module.add_function(wrap_pyfunction!(bielliptic_py, module)?)?;
    module.add_function(wrap_pyfunction!(bielliptic_batch, module)?)?;
    module.add_function(wrap_pyfunction!(alpha_to_direct_py, module)?)?;
    module.add_function(wrap_pyfunction!(alpha_to_direct_batch, module)?)?;
    module.add_function(wrap_pyfunction!(direct_to_alpha_py, module)?)?;
    module.add_function(wrap_pyfunction!(direct_to_alpha_batch, module)?)?;
    module.add_function(wrap_pyfunction!(eta_to_direct_py, module)?)?;
    module.add_function(wrap_pyfunction!(eta_to_direct_batch, module)?)?;
    module.add_function(wrap_pyfunction!(direct_to_eta_py, module)?)?;
    module.add_function(wrap_pyfunction!(direct_to_eta_batch, module)?)?;
    module.add_function(wrap_pyfunction!(flyby_constraints_py, module)?)?;
    module.add_function(wrap_pyfunction!(flyby_constraints_batch, module)?)?;
    module.add_function(wrap_pyfunction!(flyby_constraints_jacobian_py, module)?)?;
    module.add_function(wrap_pyfunction!(flyby_constraints_jacobian_batch, module)?)?;
    module.add_function(wrap_pyfunction!(flyby_delta_v_py, module)?)?;
    module.add_function(wrap_pyfunction!(flyby_delta_v_batch, module)?)?;
    module.add_function(wrap_pyfunction!(flyby_outgoing_velocity_py, module)?)?;
    module.add_function(wrap_pyfunction!(flyby_outgoing_velocity_batch, module)?)?;
    module.add_function(wrap_pyfunction!(mima_py, module)?)?;
    module.add_function(wrap_pyfunction!(mima_batch, module)?)?;
    module.add_function(wrap_pyfunction!(mima2_py, module)?)?;
    module.add_function(wrap_pyfunction!(mima2_batch, module)?)?;
    Ok(())
}
