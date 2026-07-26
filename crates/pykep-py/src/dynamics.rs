// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

use crate::error::to_python;
use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pykep_core::dynamics::pontryagin::{
    CartesianMassOptimal, CartesianTimeOptimal, EquinoctialMassOptimal, EquinoctialTimeOptimal,
    OptimalControl, cartesian_control_mass, cartesian_control_time, cartesian_hamiltonian_mass,
    cartesian_hamiltonian_time, equinoctial_control_mass, equinoctial_control_time,
    equinoctial_hamiltonian_mass, equinoctial_hamiltonian_time,
};
use pykep_core::dynamics::zoh::{
    ControlSchedule, ZohCr3bpDynamics, ZohEquinoctialDynamics, ZohKeplerDynamics,
    ZohSolarSailDynamics, propagate_schedule, propagate_schedule_backward,
};
use pykep_core::dynamics::{BcpDynamics, Cr3bpDynamics, KeplerDynamics};
use pykep_core::integration::{Dop853, DynamicsModel, InitialValueProblem, IntegratorOptions};
use pykep_core::{CartesianState, Matrix6, PykepError};
use pyo3::prelude::*;

type PyStateStmBatch<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray3<f64>>);

fn six(values: Vec<f64>) -> Result<CartesianState, PykepError> {
    values
        .try_into()
        .map_err(|values: Vec<f64>| PykepError::DimensionMismatch {
            expected: 6,
            actual: values.len(),
        })
}

fn seven(values: Vec<f64>) -> Result<[f64; 7], PykepError> {
    values
        .try_into()
        .map_err(|values: Vec<f64>| PykepError::DimensionMismatch {
            expected: 7,
            actual: values.len(),
        })
}

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

fn state_array<'py, const N: usize>(
    python: Python<'py>,
    states: Vec<[f64; N]>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let count = states.len();
    let array = Array2::from_shape_vec((count, N), states.into_iter().flatten().collect())
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(python))
}

/// Indirect low-thrust optimality criterion.
#[pyclass(name = "Optimality", eq, eq_int, frozen, skip_from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PyOptimality {
    /// Maximize final mass with logarithmic throttle regularization.
    Mass = 0,
    /// Minimize time at full throttle.
    Time = 1,
}

#[derive(Clone, Copy)]
enum PontryaginParameters {
    Mass([f64; 5]),
    Time([f64; 3]),
}

fn pontryagin_parameters(
    optimality: PyOptimality,
    parameters: Vec<f64>,
) -> Result<PontryaginParameters, PykepError> {
    match optimality {
        PyOptimality::Mass => fixed(parameters).map(PontryaginParameters::Mass),
        PyOptimality::Time => fixed(parameters).map(PontryaginParameters::Time),
    }
}

fn control_tuple(control: OptimalControl) -> (f64, [f64; 3], f64) {
    (
        control.throttle,
        control.direction,
        control.switching_function,
    )
}

fn pontryagin_result<M, const P: usize>(
    model: &M,
    state: [f64; 14],
    parameters: [f64; P],
    initial_time: f64,
    final_time: f64,
    integrator_options: IntegratorOptions,
) -> Result<[f64; 14], PykepError>
where
    M: DynamicsModel<14, P>,
{
    Dop853
        .propagate(
            model,
            InitialValueProblem::new(initial_time, state, final_time, parameters),
            integrator_options,
        )
        .map(|result| result.state)
}

/// Evaluate Cartesian Pontryagin state/costate dynamics.
///
/// State order is `[x,y,z,vx,vy,vz,m,lx,ly,lz,lvx,lvy,lvz,lm]`.
/// Mass-optimal parameters are `[mu, thrust, exhaust_velocity, barrier,
/// lambda0]`; time-optimal parameters are `[mu, thrust, exhaust_velocity]`.
#[pyfunction]
fn pontryagin_cartesian_rhs(
    state: Vec<f64>,
    optimality: PyRef<'_, PyOptimality>,
    parameters: Vec<f64>,
) -> PyResult<[f64; 14]> {
    let state = fixed(state).map_err(to_python)?;
    let mut derivative = [0.0; 14];
    match *optimality {
        PyOptimality::Mass => CartesianMassOptimal.rhs(
            0.0,
            &state,
            &fixed(parameters).map_err(to_python)?,
            &mut derivative,
        ),
        PyOptimality::Time => CartesianTimeOptimal.rhs(
            0.0,
            &state,
            &fixed(parameters).map_err(to_python)?,
            &mut derivative,
        ),
    }
    .map_err(to_python)?;
    Ok(derivative)
}

/// Evaluate modified-equinoctial Pontryagin state/costate dynamics.
///
/// State order is `[p,f,g,h,k,L,m,lp,lf,lg,lh,lk,lL,lm]`. Parameter
/// order follows [`pontryagin_cartesian_rhs`].
#[pyfunction]
fn pontryagin_equinoctial_rhs(
    state: Vec<f64>,
    optimality: PyRef<'_, PyOptimality>,
    parameters: Vec<f64>,
) -> PyResult<[f64; 14]> {
    let state = fixed(state).map_err(to_python)?;
    let mut derivative = [0.0; 14];
    match *optimality {
        PyOptimality::Mass => EquinoctialMassOptimal.rhs(
            0.0,
            &state,
            &fixed(parameters).map_err(to_python)?,
            &mut derivative,
        ),
        PyOptimality::Time => EquinoctialTimeOptimal.rhs(
            0.0,
            &state,
            &fixed(parameters).map_err(to_python)?,
            &mut derivative,
        ),
    }
    .map_err(to_python)?;
    Ok(derivative)
}

/// Return `(throttle, direction, switching_function)` for Cartesian
/// Pontryagin dynamics.
#[pyfunction]
fn pontryagin_cartesian_control(
    state: Vec<f64>,
    optimality: PyRef<'_, PyOptimality>,
    parameters: Vec<f64>,
) -> PyResult<(f64, [f64; 3], f64)> {
    let state = fixed(state).map_err(to_python)?;
    match *optimality {
        PyOptimality::Mass => {
            cartesian_control_mass(&state, &fixed(parameters).map_err(to_python)?)
        }
        PyOptimality::Time => {
            cartesian_control_time(&state, &fixed(parameters).map_err(to_python)?)
        }
    }
    .map(control_tuple)
    .map_err(to_python)
}

/// Return `(throttle, RTN direction, switching_function)` for
/// modified-equinoctial Pontryagin dynamics.
#[pyfunction]
fn pontryagin_equinoctial_control(
    state: Vec<f64>,
    optimality: PyRef<'_, PyOptimality>,
    parameters: Vec<f64>,
) -> PyResult<(f64, [f64; 3], f64)> {
    let state = fixed(state).map_err(to_python)?;
    match *optimality {
        PyOptimality::Mass => {
            equinoctial_control_mass(&state, &fixed(parameters).map_err(to_python)?)
        }
        PyOptimality::Time => {
            equinoctial_control_time(&state, &fixed(parameters).map_err(to_python)?)
        }
    }
    .map(control_tuple)
    .map_err(to_python)
}

/// Evaluate the minimized Cartesian Pontryagin Hamiltonian.
#[pyfunction]
fn pontryagin_cartesian_hamiltonian(
    state: Vec<f64>,
    optimality: PyRef<'_, PyOptimality>,
    parameters: Vec<f64>,
) -> PyResult<f64> {
    let state = fixed(state).map_err(to_python)?;
    match *optimality {
        PyOptimality::Mass => {
            cartesian_hamiltonian_mass(&state, &fixed(parameters).map_err(to_python)?)
        }
        PyOptimality::Time => {
            cartesian_hamiltonian_time(&state, &fixed(parameters).map_err(to_python)?)
        }
    }
    .map_err(to_python)
}

/// Evaluate the minimized modified-equinoctial Pontryagin Hamiltonian.
#[pyfunction]
fn pontryagin_equinoctial_hamiltonian(
    state: Vec<f64>,
    optimality: PyRef<'_, PyOptimality>,
    parameters: Vec<f64>,
) -> PyResult<f64> {
    let state = fixed(state).map_err(to_python)?;
    match *optimality {
        PyOptimality::Mass => {
            equinoctial_hamiltonian_mass(&state, &fixed(parameters).map_err(to_python)?)
        }
        PyOptimality::Time => {
            equinoctial_hamiltonian_time(&state, &fixed(parameters).map_err(to_python)?)
        }
    }
    .map_err(to_python)
}

/// Propagate Cartesian Pontryagin state/costate dynamics.
#[pyfunction(signature = (
    state,
    final_time,
    optimality,
    parameters,
    initial_time = 0.0,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None
))]
#[allow(clippy::too_many_arguments)]
fn propagate_pontryagin_cartesian(
    python: Python<'_>,
    state: Vec<f64>,
    final_time: f64,
    optimality: PyRef<'_, PyOptimality>,
    parameters: Vec<f64>,
    initial_time: f64,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
) -> PyResult<[f64; 14]> {
    let state = fixed(state).map_err(to_python)?;
    let optimality = *optimality;
    let integrator_options = options(relative_tolerance, absolute_tolerance, maximum_step);
    match optimality {
        PyOptimality::Mass => {
            let parameters = fixed(parameters).map_err(to_python)?;
            python.detach(move || {
                pontryagin_result(
                    &CartesianMassOptimal,
                    state,
                    parameters,
                    initial_time,
                    final_time,
                    integrator_options,
                )
                .map_err(to_python)
            })
        }
        PyOptimality::Time => {
            let parameters = fixed(parameters).map_err(to_python)?;
            python.detach(move || {
                pontryagin_result(
                    &CartesianTimeOptimal,
                    state,
                    parameters,
                    initial_time,
                    final_time,
                    integrator_options,
                )
                .map_err(to_python)
            })
        }
    }
}

/// Propagate modified-equinoctial Pontryagin state/costate dynamics.
#[pyfunction(signature = (
    state,
    final_time,
    optimality,
    parameters,
    initial_time = 0.0,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None
))]
#[allow(clippy::too_many_arguments)]
fn propagate_pontryagin_equinoctial(
    python: Python<'_>,
    state: Vec<f64>,
    final_time: f64,
    optimality: PyRef<'_, PyOptimality>,
    parameters: Vec<f64>,
    initial_time: f64,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
) -> PyResult<[f64; 14]> {
    let state = fixed(state).map_err(to_python)?;
    let optimality = *optimality;
    let integrator_options = options(relative_tolerance, absolute_tolerance, maximum_step);
    match optimality {
        PyOptimality::Mass => {
            let parameters = fixed(parameters).map_err(to_python)?;
            python.detach(move || {
                pontryagin_result(
                    &EquinoctialMassOptimal,
                    state,
                    parameters,
                    initial_time,
                    final_time,
                    integrator_options,
                )
                .map_err(to_python)
            })
        }
        PyOptimality::Time => {
            let parameters = fixed(parameters).map_err(to_python)?;
            python.detach(move || {
                pontryagin_result(
                    &EquinoctialTimeOptimal,
                    state,
                    parameters,
                    initial_time,
                    final_time,
                    integrator_options,
                )
                .map_err(to_python)
            })
        }
    }
}

/// Batch-evaluate Cartesian Pontryagin dynamics.
#[pyfunction(signature = (states, optimality, parameters, workers=0))]
fn pontryagin_cartesian_rhs_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    optimality: PyRef<'_, PyOptimality>,
    parameters: Vec<f64>,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let states = fixed_rows::<14>(states)?;
    let parameters = pontryagin_parameters(*optimality, parameters).map_err(to_python)?;
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&states, workers, |state| {
                let mut derivative = [0.0; 14];
                match parameters {
                    PontryaginParameters::Mass(parameters) => {
                        CartesianMassOptimal.rhs(0.0, state, &parameters, &mut derivative)
                    }
                    PontryaginParameters::Time(parameters) => {
                        CartesianTimeOptimal.rhs(0.0, state, &parameters, &mut derivative)
                    }
                }?;
                Ok(derivative)
            })
        })
        .map_err(to_python)?;
    state_array(python, output)
}

/// Batch-evaluate modified-equinoctial Pontryagin dynamics.
#[pyfunction(signature = (states, optimality, parameters, workers=0))]
fn pontryagin_equinoctial_rhs_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    optimality: PyRef<'_, PyOptimality>,
    parameters: Vec<f64>,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let states = fixed_rows::<14>(states)?;
    let parameters = pontryagin_parameters(*optimality, parameters).map_err(to_python)?;
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&states, workers, |state| {
                let mut derivative = [0.0; 14];
                match parameters {
                    PontryaginParameters::Mass(parameters) => {
                        EquinoctialMassOptimal.rhs(0.0, state, &parameters, &mut derivative)
                    }
                    PontryaginParameters::Time(parameters) => {
                        EquinoctialTimeOptimal.rhs(0.0, state, &parameters, &mut derivative)
                    }
                }?;
                Ok(derivative)
            })
        })
        .map_err(to_python)?;
    state_array(python, output)
}

/// Batch-evaluate Cartesian Pontryagin controls.
#[pyfunction(signature = (states, optimality, parameters, workers=0))]
fn pontryagin_cartesian_control_batch(
    python: Python<'_>,
    states: PyReadonlyArray2<'_, f64>,
    optimality: PyRef<'_, PyOptimality>,
    parameters: Vec<f64>,
    workers: usize,
) -> PyResult<Vec<(f64, [f64; 3], f64)>> {
    let states = fixed_rows::<14>(states)?;
    let parameters = pontryagin_parameters(*optimality, parameters).map_err(to_python)?;
    python
        .detach(move || {
            pykep_core::batch::try_map(&states, workers, |state| {
                match parameters {
                    PontryaginParameters::Mass(parameters) => {
                        cartesian_control_mass(state, &parameters)
                    }
                    PontryaginParameters::Time(parameters) => {
                        cartesian_control_time(state, &parameters)
                    }
                }
                .map(control_tuple)
            })
        })
        .map_err(to_python)
}

/// Batch-evaluate modified-equinoctial Pontryagin controls.
#[pyfunction(signature = (states, optimality, parameters, workers=0))]
fn pontryagin_equinoctial_control_batch(
    python: Python<'_>,
    states: PyReadonlyArray2<'_, f64>,
    optimality: PyRef<'_, PyOptimality>,
    parameters: Vec<f64>,
    workers: usize,
) -> PyResult<Vec<(f64, [f64; 3], f64)>> {
    let states = fixed_rows::<14>(states)?;
    let parameters = pontryagin_parameters(*optimality, parameters).map_err(to_python)?;
    python
        .detach(move || {
            pykep_core::batch::try_map(&states, workers, |state| {
                match parameters {
                    PontryaginParameters::Mass(parameters) => {
                        equinoctial_control_mass(state, &parameters)
                    }
                    PontryaginParameters::Time(parameters) => {
                        equinoctial_control_time(state, &parameters)
                    }
                }
                .map(control_tuple)
            })
        })
        .map_err(to_python)
}

/// Batch-evaluate minimized Cartesian Pontryagin Hamiltonians.
#[pyfunction(signature = (states, optimality, parameters, workers=0))]
fn pontryagin_cartesian_hamiltonian_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    optimality: PyRef<'_, PyOptimality>,
    parameters: Vec<f64>,
    workers: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let states = fixed_rows::<14>(states)?;
    let parameters = pontryagin_parameters(*optimality, parameters).map_err(to_python)?;
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&states, workers, |state| match parameters {
                PontryaginParameters::Mass(parameters) => {
                    cartesian_hamiltonian_mass(state, &parameters)
                }
                PontryaginParameters::Time(parameters) => {
                    cartesian_hamiltonian_time(state, &parameters)
                }
            })
        })
        .map_err(to_python)?;
    Ok(Array1::from_vec(output).into_pyarray(python))
}

/// Batch-evaluate minimized equinoctial Pontryagin Hamiltonians.
#[pyfunction(signature = (states, optimality, parameters, workers=0))]
fn pontryagin_equinoctial_hamiltonian_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    optimality: PyRef<'_, PyOptimality>,
    parameters: Vec<f64>,
    workers: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let states = fixed_rows::<14>(states)?;
    let parameters = pontryagin_parameters(*optimality, parameters).map_err(to_python)?;
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&states, workers, |state| match parameters {
                PontryaginParameters::Mass(parameters) => {
                    equinoctial_hamiltonian_mass(state, &parameters)
                }
                PontryaginParameters::Time(parameters) => {
                    equinoctial_hamiltonian_time(state, &parameters)
                }
            })
        })
        .map_err(to_python)?;
    Ok(Array1::from_vec(output).into_pyarray(python))
}

/// Batch-propagate Cartesian Pontryagin states.
#[pyfunction(signature = (
    states,
    final_times,
    optimality,
    parameters,
    initial_time = 0.0,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None,
    workers = 0
))]
#[allow(clippy::too_many_arguments)]
fn propagate_pontryagin_cartesian_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    final_times: PyReadonlyArray1<'py, f64>,
    optimality: PyRef<'_, PyOptimality>,
    parameters: Vec<f64>,
    initial_time: f64,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let states = fixed_rows::<14>(states)?;
    matching_length(states.len(), final_times.len())?;
    let parameters = pontryagin_parameters(*optimality, parameters).map_err(to_python)?;
    let integrator_options = options(relative_tolerance, absolute_tolerance, maximum_step);
    let inputs = states
        .into_iter()
        .zip(final_times.as_array().iter().copied())
        .collect::<Vec<_>>();
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&inputs, workers, |(state, final_time)| match parameters {
                PontryaginParameters::Mass(parameters) => pontryagin_result(
                    &CartesianMassOptimal,
                    *state,
                    parameters,
                    initial_time,
                    *final_time,
                    integrator_options,
                ),
                PontryaginParameters::Time(parameters) => pontryagin_result(
                    &CartesianTimeOptimal,
                    *state,
                    parameters,
                    initial_time,
                    *final_time,
                    integrator_options,
                ),
            })
        })
        .map_err(to_python)?;
    state_array(python, output)
}

/// Batch-propagate modified-equinoctial Pontryagin states.
#[pyfunction(signature = (
    states,
    final_times,
    optimality,
    parameters,
    initial_time = 0.0,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None,
    workers = 0
))]
#[allow(clippy::too_many_arguments)]
fn propagate_pontryagin_equinoctial_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    final_times: PyReadonlyArray1<'py, f64>,
    optimality: PyRef<'_, PyOptimality>,
    parameters: Vec<f64>,
    initial_time: f64,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let states = fixed_rows::<14>(states)?;
    matching_length(states.len(), final_times.len())?;
    let parameters = pontryagin_parameters(*optimality, parameters).map_err(to_python)?;
    let integrator_options = options(relative_tolerance, absolute_tolerance, maximum_step);
    let inputs = states
        .into_iter()
        .zip(final_times.as_array().iter().copied())
        .collect::<Vec<_>>();
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&inputs, workers, |(state, final_time)| match parameters {
                PontryaginParameters::Mass(parameters) => pontryagin_result(
                    &EquinoctialMassOptimal,
                    *state,
                    parameters,
                    initial_time,
                    *final_time,
                    integrator_options,
                ),
                PontryaginParameters::Time(parameters) => pontryagin_result(
                    &EquinoctialTimeOptimal,
                    *state,
                    parameters,
                    initial_time,
                    *final_time,
                    integrator_options,
                ),
            })
        })
        .map_err(to_python)?;
    state_array(python, output)
}

fn control_rows<const C: usize>(values: Vec<Vec<f64>>) -> Result<Vec<[f64; C]>, PykepError> {
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

fn options(
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
) -> IntegratorOptions {
    IntegratorOptions {
        relative_tolerance,
        absolute_tolerance,
        maximum_step,
        ..IntegratorOptions::default()
    }
}

fn rows(matrix: Matrix6) -> Vec<Vec<f64>> {
    matrix.into_iter().map(Vec::from).collect()
}

fn cartesian_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    final_times: PyReadonlyArray1<'py, f64>,
    workers: usize,
    operation: impl Fn(CartesianState, f64) -> Result<CartesianState, PykepError>
    + Sync
    + Send
    + 'static,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let states = fixed_rows::<6>(states)?;
    matching_length(states.len(), final_times.len())?;
    let inputs = states
        .into_iter()
        .zip(final_times.as_array().iter().copied())
        .collect::<Vec<_>>();
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&inputs, workers, |(state, final_time)| {
                operation(*state, *final_time)
            })
        })
        .map_err(to_python)?;
    state_array(python, output)
}

fn cartesian_stm_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    final_times: PyReadonlyArray1<'py, f64>,
    workers: usize,
    operation: impl Fn(CartesianState, f64) -> Result<(CartesianState, Matrix6), PykepError>
    + Sync
    + Send
    + 'static,
) -> PyResult<PyStateStmBatch<'py>> {
    let states = fixed_rows::<6>(states)?;
    matching_length(states.len(), final_times.len())?;
    let inputs = states
        .into_iter()
        .zip(final_times.as_array().iter().copied())
        .collect::<Vec<_>>();
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&inputs, workers, |(state, final_time)| {
                operation(*state, *final_time)
            })
        })
        .map_err(to_python)?;
    let count = output.len();
    let states = state_array(
        python,
        output.iter().map(|(state, _)| *state).collect::<Vec<_>>(),
    )?;
    let matrices = Array3::from_shape_vec(
        (count, 6, 6),
        output
            .into_iter()
            .flat_map(|(_, matrix)| matrix.into_iter().flatten())
            .collect(),
    )
    .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?
    .into_pyarray(python);
    Ok((states, matrices))
}

fn zoh_result<M, const N: usize, const C: usize, const K: usize, const P: usize>(
    model: &M,
    boundaries: Vec<f64>,
    controls: Vec<Vec<f64>>,
    state: [f64; N],
    constants: [f64; K],
    backward: bool,
    integrator_options: IntegratorOptions,
) -> PyResult<[f64; N]>
where
    M: pykep_core::dynamics::zoh::ZeroOrderHoldModel<N, C, K, P>,
{
    zoh_result_core(
        model,
        boundaries,
        controls,
        state,
        constants,
        backward,
        integrator_options,
    )
    .map_err(to_python)
}

fn zoh_result_core<M, const N: usize, const C: usize, const K: usize, const P: usize>(
    model: &M,
    boundaries: Vec<f64>,
    controls: Vec<Vec<f64>>,
    state: [f64; N],
    constants: [f64; K],
    backward: bool,
    integrator_options: IntegratorOptions,
) -> Result<[f64; N], PykepError>
where
    M: pykep_core::dynamics::zoh::ZeroOrderHoldModel<N, C, K, P>,
{
    let schedule = ControlSchedule::new(boundaries, control_rows(controls)?)?;
    let result = if backward {
        propagate_schedule_backward(model, &schedule, state, constants, integrator_options)
    } else {
        propagate_schedule(model, &schedule, state, constants, integrator_options)
    };
    result.map(|propagation| propagation.state)
}

#[allow(clippy::too_many_arguments)]
fn zoh_batch<'py, M, const N: usize, const C: usize, const K: usize, const P: usize>(
    python: Python<'py>,
    model: M,
    states: PyReadonlyArray2<'py, f64>,
    boundaries: Vec<Vec<f64>>,
    controls: Vec<Vec<Vec<f64>>>,
    constants: [f64; K],
    backward: bool,
    integrator_options: IntegratorOptions,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>>
where
    M: pykep_core::dynamics::zoh::ZeroOrderHoldModel<N, C, K, P> + Copy + Sync + Send + 'static,
{
    let states = fixed_rows::<N>(states)?;
    let count = states.len();
    matching_length(count, boundaries.len())?;
    matching_length(count, controls.len())?;
    let inputs = states
        .into_iter()
        .zip(boundaries)
        .zip(controls)
        .collect::<Vec<_>>();
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&inputs, workers, |((state, boundaries), controls)| {
                zoh_result_core(
                    &model,
                    boundaries.clone(),
                    controls.clone(),
                    *state,
                    constants,
                    backward,
                    integrator_options,
                )
            })
        })
        .map_err(to_python)?;
    state_array(python, output)
}

/// Evaluate two-body Cartesian dynamics.
#[pyfunction]
fn kepler_rhs(state: Vec<f64>, mu: f64) -> PyResult<CartesianState> {
    KeplerDynamics
        .evaluate(&six(state).map_err(to_python)?, mu)
        .map_err(to_python)
}

/// Evaluate circular restricted three-body dynamics in the synodic frame.
#[pyfunction]
fn cr3bp_rhs(state: Vec<f64>, mu: f64) -> PyResult<CartesianState> {
    Cr3bpDynamics
        .evaluate(&six(state).map_err(to_python)?, mu)
        .map_err(to_python)
}

/// Evaluate bicircular dynamics in the Earth–Moon synodic frame.
#[pyfunction]
fn bcp_rhs(
    time: f64,
    state: Vec<f64>,
    mu: f64,
    mu_sun: f64,
    rho_sun: f64,
    omega_sun: f64,
) -> PyResult<CartesianState> {
    BcpDynamics
        .evaluate(
            time,
            &six(state).map_err(to_python)?,
            [mu, mu_sun, rho_sun, omega_sun],
        )
        .map_err(to_python)
}

/// Evaluate the positive CR3BP effective potential.
#[pyfunction]
fn cr3bp_effective_potential(state: Vec<f64>, mu: f64) -> PyResult<f64> {
    Cr3bpDynamics
        .effective_potential(&six(state).map_err(to_python)?, mu)
        .map_err(to_python)
}

/// Evaluate the CR3BP Jacobi constant.
#[pyfunction]
fn cr3bp_jacobi_constant(state: Vec<f64>, mu: f64) -> PyResult<f64> {
    Cr3bpDynamics
        .jacobi_constant(&six(state).map_err(to_python)?, mu)
        .map_err(to_python)
}

/// Evaluate normalized seven-state low-thrust Kepler dynamics.
#[pyfunction]
fn zoh_kepler_rhs(
    state: Vec<f64>,
    thrust: f64,
    direction: Vec<f64>,
    mass_flow_coefficient: f64,
) -> PyResult<[f64; 7]> {
    let direction: [f64; 3] = direction.try_into().map_err(|values: Vec<f64>| {
        to_python(PykepError::DimensionMismatch {
            expected: 3,
            actual: values.len(),
        })
    })?;
    let mut derivative = [0.0; 7];
    ZohKeplerDynamics
        .rhs(
            0.0,
            &seven(state).map_err(to_python)?,
            &[
                thrust,
                direction[0],
                direction[1],
                direction[2],
                mass_flow_coefficient,
            ],
            &mut derivative,
        )
        .map_err(to_python)?;
    Ok(derivative)
}

/// Evaluate seven-state low-thrust CR3BP dynamics.
#[pyfunction]
fn zoh_cr3bp_rhs(
    state: Vec<f64>,
    thrust: f64,
    direction: Vec<f64>,
    mass_flow_coefficient: f64,
    mu: f64,
) -> PyResult<[f64; 7]> {
    let direction: [f64; 3] = direction.try_into().map_err(|values: Vec<f64>| {
        to_python(PykepError::DimensionMismatch {
            expected: 3,
            actual: values.len(),
        })
    })?;
    let mut derivative = [0.0; 7];
    ZohCr3bpDynamics
        .rhs(
            0.0,
            &seven(state).map_err(to_python)?,
            &[
                thrust,
                direction[0],
                direction[1],
                direction[2],
                mass_flow_coefficient,
                mu,
            ],
            &mut derivative,
        )
        .map_err(to_python)?;
    Ok(derivative)
}

/// Evaluate normalized modified-equinoctial low-thrust dynamics.
#[pyfunction]
fn zoh_equinoctial_rhs(
    state: Vec<f64>,
    thrust: f64,
    rtn_direction: Vec<f64>,
    mass_flow_coefficient: f64,
) -> PyResult<[f64; 7]> {
    let direction: [f64; 3] = rtn_direction.try_into().map_err(|values: Vec<f64>| {
        to_python(PykepError::DimensionMismatch {
            expected: 3,
            actual: values.len(),
        })
    })?;
    let mut derivative = [0.0; 7];
    ZohEquinoctialDynamics
        .rhs(
            0.0,
            &seven(state).map_err(to_python)?,
            &[
                thrust,
                direction[0],
                direction[1],
                direction[2],
                mass_flow_coefficient,
            ],
            &mut derivative,
        )
        .map_err(to_python)?;
    Ok(derivative)
}

/// Evaluate normalized ideal solar-sail dynamics.
#[pyfunction]
fn zoh_solar_sail_rhs(
    state: Vec<f64>,
    alpha: f64,
    beta: f64,
    lightness: f64,
) -> PyResult<CartesianState> {
    let mut derivative = [0.0; 6];
    ZohSolarSailDynamics
        .rhs(
            0.0,
            &six(state).map_err(to_python)?,
            &[alpha, beta, lightness],
            &mut derivative,
        )
        .map_err(to_python)?;
    Ok(derivative)
}

/// Batch-evaluate two-body Cartesian dynamics.
#[pyfunction(signature = (states, mu, workers=0))]
fn kepler_rhs_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    mu: f64,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let states = fixed_rows::<6>(states)?;
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&states, workers, |state| KeplerDynamics.evaluate(state, mu))
        })
        .map_err(to_python)?;
    state_array(python, output)
}

/// Batch-evaluate CR3BP dynamics.
#[pyfunction(signature = (states, mu, workers=0))]
fn cr3bp_rhs_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    mu: f64,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let states = fixed_rows::<6>(states)?;
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&states, workers, |state| Cr3bpDynamics.evaluate(state, mu))
        })
        .map_err(to_python)?;
    state_array(python, output)
}

/// Batch-evaluate bicircular dynamics.
#[pyfunction(signature = (times, states, mu, mu_sun, rho_sun, omega_sun, workers=0))]
#[allow(clippy::too_many_arguments)]
fn bcp_rhs_batch<'py>(
    python: Python<'py>,
    times: PyReadonlyArray1<'py, f64>,
    states: PyReadonlyArray2<'py, f64>,
    mu: f64,
    mu_sun: f64,
    rho_sun: f64,
    omega_sun: f64,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let states = fixed_rows::<6>(states)?;
    matching_length(states.len(), times.len())?;
    let parameters = [mu, mu_sun, rho_sun, omega_sun];
    let inputs = states
        .into_iter()
        .zip(times.as_array().iter().copied())
        .collect::<Vec<_>>();
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&inputs, workers, |(state, time)| {
                BcpDynamics.evaluate(*time, state, parameters)
            })
        })
        .map_err(to_python)?;
    state_array(python, output)
}

/// Batch-evaluate the CR3BP effective potential.
#[pyfunction(signature = (states, mu, workers=0))]
fn cr3bp_effective_potential_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    mu: f64,
    workers: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let states = fixed_rows::<6>(states)?;
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&states, workers, |state| {
                Cr3bpDynamics.effective_potential(state, mu)
            })
        })
        .map_err(to_python)?;
    Ok(Array1::from_vec(output).into_pyarray(python))
}

/// Batch-evaluate the CR3BP Jacobi constant.
#[pyfunction(signature = (states, mu, workers=0))]
fn cr3bp_jacobi_constant_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    mu: f64,
    workers: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let states = fixed_rows::<6>(states)?;
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&states, workers, |state| {
                Cr3bpDynamics.jacobi_constant(state, mu)
            })
        })
        .map_err(to_python)?;
    Ok(Array1::from_vec(output).into_pyarray(python))
}

/// Batch-evaluate normalized low-thrust Kepler dynamics.
#[pyfunction(signature = (states, thrust, direction, mass_flow_coefficient, workers=0))]
fn zoh_kepler_rhs_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    thrust: PyReadonlyArray1<'py, f64>,
    direction: PyReadonlyArray2<'py, f64>,
    mass_flow_coefficient: f64,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let states = fixed_rows::<7>(states)?;
    let direction = fixed_rows::<3>(direction)?;
    let count = states.len();
    matching_length(count, thrust.len())?;
    matching_length(count, direction.len())?;
    let inputs = states
        .into_iter()
        .zip(thrust.as_array().iter().copied())
        .zip(direction)
        .collect::<Vec<_>>();
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&inputs, workers, |((state, thrust), direction)| {
                let mut derivative = [0.0; 7];
                ZohKeplerDynamics.rhs(
                    0.0,
                    state,
                    &[
                        *thrust,
                        direction[0],
                        direction[1],
                        direction[2],
                        mass_flow_coefficient,
                    ],
                    &mut derivative,
                )?;
                Ok(derivative)
            })
        })
        .map_err(to_python)?;
    state_array(python, output)
}

/// Batch-evaluate low-thrust CR3BP dynamics.
#[pyfunction(signature = (states, thrust, direction, mass_flow_coefficient, mu, workers=0))]
#[allow(clippy::too_many_arguments)]
fn zoh_cr3bp_rhs_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    thrust: PyReadonlyArray1<'py, f64>,
    direction: PyReadonlyArray2<'py, f64>,
    mass_flow_coefficient: f64,
    mu: f64,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let states = fixed_rows::<7>(states)?;
    let direction = fixed_rows::<3>(direction)?;
    let count = states.len();
    matching_length(count, thrust.len())?;
    matching_length(count, direction.len())?;
    let inputs = states
        .into_iter()
        .zip(thrust.as_array().iter().copied())
        .zip(direction)
        .collect::<Vec<_>>();
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&inputs, workers, |((state, thrust), direction)| {
                let mut derivative = [0.0; 7];
                ZohCr3bpDynamics.rhs(
                    0.0,
                    state,
                    &[
                        *thrust,
                        direction[0],
                        direction[1],
                        direction[2],
                        mass_flow_coefficient,
                        mu,
                    ],
                    &mut derivative,
                )?;
                Ok(derivative)
            })
        })
        .map_err(to_python)?;
    state_array(python, output)
}

/// Batch-evaluate modified-equinoctial low-thrust dynamics.
#[pyfunction(signature = (states, thrust, rtn_direction, mass_flow_coefficient, workers=0))]
fn zoh_equinoctial_rhs_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    thrust: PyReadonlyArray1<'py, f64>,
    rtn_direction: PyReadonlyArray2<'py, f64>,
    mass_flow_coefficient: f64,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let states = fixed_rows::<7>(states)?;
    let direction = fixed_rows::<3>(rtn_direction)?;
    let count = states.len();
    matching_length(count, thrust.len())?;
    matching_length(count, direction.len())?;
    let inputs = states
        .into_iter()
        .zip(thrust.as_array().iter().copied())
        .zip(direction)
        .collect::<Vec<_>>();
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&inputs, workers, |((state, thrust), direction)| {
                let mut derivative = [0.0; 7];
                ZohEquinoctialDynamics.rhs(
                    0.0,
                    state,
                    &[
                        *thrust,
                        direction[0],
                        direction[1],
                        direction[2],
                        mass_flow_coefficient,
                    ],
                    &mut derivative,
                )?;
                Ok(derivative)
            })
        })
        .map_err(to_python)?;
    state_array(python, output)
}

/// Batch-evaluate ideal solar-sail dynamics.
#[pyfunction(signature = (states, alpha, beta, lightness, workers=0))]
fn zoh_solar_sail_rhs_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    alpha: PyReadonlyArray1<'py, f64>,
    beta: PyReadonlyArray1<'py, f64>,
    lightness: f64,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let states = fixed_rows::<6>(states)?;
    let count = states.len();
    matching_length(count, alpha.len())?;
    matching_length(count, beta.len())?;
    let inputs = states
        .into_iter()
        .zip(alpha.as_array().iter().copied())
        .zip(beta.as_array().iter().copied())
        .collect::<Vec<_>>();
    let output = python
        .detach(move || {
            pykep_core::batch::try_map(&inputs, workers, |((state, alpha), beta)| {
                let mut derivative = [0.0; 6];
                ZohSolarSailDynamics.rhs(
                    0.0,
                    state,
                    &[*alpha, *beta, lightness],
                    &mut derivative,
                )?;
                Ok(derivative)
            })
        })
        .map_err(to_python)?;
    state_array(python, output)
}

/// Propagate a piecewise-constant low-thrust Kepler schedule.
#[pyfunction(signature = (
    state,
    boundaries,
    controls,
    mass_flow_coefficient,
    backward = false,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None
))]
#[allow(clippy::too_many_arguments)]
fn propagate_zoh_kepler(
    python: Python<'_>,
    state: Vec<f64>,
    boundaries: Vec<f64>,
    controls: Vec<Vec<f64>>,
    mass_flow_coefficient: f64,
    backward: bool,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
) -> PyResult<[f64; 7]> {
    let state = seven(state).map_err(to_python)?;
    python.detach(move || {
        zoh_result::<_, 7, 4, 1, 5>(
            &ZohKeplerDynamics,
            boundaries,
            controls,
            state,
            [mass_flow_coefficient],
            backward,
            options(relative_tolerance, absolute_tolerance, maximum_step),
        )
    })
}

/// Propagate a piecewise-constant low-thrust CR3BP schedule.
#[pyfunction(signature = (
    state,
    boundaries,
    controls,
    mass_flow_coefficient,
    mu,
    backward = false,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None
))]
#[allow(clippy::too_many_arguments)]
fn propagate_zoh_cr3bp(
    python: Python<'_>,
    state: Vec<f64>,
    boundaries: Vec<f64>,
    controls: Vec<Vec<f64>>,
    mass_flow_coefficient: f64,
    mu: f64,
    backward: bool,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
) -> PyResult<[f64; 7]> {
    let state = seven(state).map_err(to_python)?;
    python.detach(move || {
        zoh_result::<_, 7, 4, 2, 6>(
            &ZohCr3bpDynamics,
            boundaries,
            controls,
            state,
            [mass_flow_coefficient, mu],
            backward,
            options(relative_tolerance, absolute_tolerance, maximum_step),
        )
    })
}

/// Propagate a piecewise-constant modified-equinoctial low-thrust schedule.
#[pyfunction(signature = (
    state,
    boundaries,
    controls,
    mass_flow_coefficient,
    backward = false,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None
))]
#[allow(clippy::too_many_arguments)]
fn propagate_zoh_equinoctial(
    python: Python<'_>,
    state: Vec<f64>,
    boundaries: Vec<f64>,
    controls: Vec<Vec<f64>>,
    mass_flow_coefficient: f64,
    backward: bool,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
) -> PyResult<[f64; 7]> {
    let state = seven(state).map_err(to_python)?;
    python.detach(move || {
        zoh_result::<_, 7, 4, 1, 5>(
            &ZohEquinoctialDynamics,
            boundaries,
            controls,
            state,
            [mass_flow_coefficient],
            backward,
            options(relative_tolerance, absolute_tolerance, maximum_step),
        )
    })
}

/// Propagate a piecewise-constant ideal solar-sail attitude schedule.
#[pyfunction(signature = (
    state,
    boundaries,
    controls,
    lightness,
    backward = false,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None
))]
#[allow(clippy::too_many_arguments)]
fn propagate_zoh_solar_sail(
    python: Python<'_>,
    state: Vec<f64>,
    boundaries: Vec<f64>,
    controls: Vec<Vec<f64>>,
    lightness: f64,
    backward: bool,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
) -> PyResult<CartesianState> {
    let state = six(state).map_err(to_python)?;
    python.detach(move || {
        zoh_result::<_, 6, 2, 1, 3>(
            &ZohSolarSailDynamics,
            boundaries,
            controls,
            state,
            [lightness],
            backward,
            options(relative_tolerance, absolute_tolerance, maximum_step),
        )
    })
}

/// Batch-propagate piecewise-constant low-thrust Kepler schedules.
#[pyfunction(signature = (
    states,
    boundaries,
    controls,
    mass_flow_coefficient,
    backward = false,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None,
    workers = 0
))]
#[allow(clippy::too_many_arguments)]
fn propagate_zoh_kepler_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    boundaries: Vec<Vec<f64>>,
    controls: Vec<Vec<Vec<f64>>>,
    mass_flow_coefficient: f64,
    backward: bool,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    zoh_batch::<_, 7, 4, 1, 5>(
        python,
        ZohKeplerDynamics,
        states,
        boundaries,
        controls,
        [mass_flow_coefficient],
        backward,
        options(relative_tolerance, absolute_tolerance, maximum_step),
        workers,
    )
}

/// Batch-propagate piecewise-constant low-thrust CR3BP schedules.
#[pyfunction(signature = (
    states,
    boundaries,
    controls,
    mass_flow_coefficient,
    mu,
    backward = false,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None,
    workers = 0
))]
#[allow(clippy::too_many_arguments)]
fn propagate_zoh_cr3bp_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    boundaries: Vec<Vec<f64>>,
    controls: Vec<Vec<Vec<f64>>>,
    mass_flow_coefficient: f64,
    mu: f64,
    backward: bool,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    zoh_batch::<_, 7, 4, 2, 6>(
        python,
        ZohCr3bpDynamics,
        states,
        boundaries,
        controls,
        [mass_flow_coefficient, mu],
        backward,
        options(relative_tolerance, absolute_tolerance, maximum_step),
        workers,
    )
}

/// Batch-propagate piecewise-constant equinoctial schedules.
#[pyfunction(signature = (
    states,
    boundaries,
    controls,
    mass_flow_coefficient,
    backward = false,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None,
    workers = 0
))]
#[allow(clippy::too_many_arguments)]
fn propagate_zoh_equinoctial_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    boundaries: Vec<Vec<f64>>,
    controls: Vec<Vec<Vec<f64>>>,
    mass_flow_coefficient: f64,
    backward: bool,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    zoh_batch::<_, 7, 4, 1, 5>(
        python,
        ZohEquinoctialDynamics,
        states,
        boundaries,
        controls,
        [mass_flow_coefficient],
        backward,
        options(relative_tolerance, absolute_tolerance, maximum_step),
        workers,
    )
}

/// Batch-propagate piecewise-constant ideal solar-sail schedules.
#[pyfunction(signature = (
    states,
    boundaries,
    controls,
    lightness,
    backward = false,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None,
    workers = 0
))]
#[allow(clippy::too_many_arguments)]
fn propagate_zoh_solar_sail_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    boundaries: Vec<Vec<f64>>,
    controls: Vec<Vec<Vec<f64>>>,
    lightness: f64,
    backward: bool,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    zoh_batch::<_, 6, 2, 1, 3>(
        python,
        ZohSolarSailDynamics,
        states,
        boundaries,
        controls,
        [lightness],
        backward,
        options(relative_tolerance, absolute_tolerance, maximum_step),
        workers,
    )
}

/// Adaptively propagate evaluated two-body dynamics.
#[pyfunction(signature = (
    state,
    final_time,
    mu,
    initial_time = 0.0,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None
))]
#[allow(clippy::too_many_arguments)]
fn propagate_kepler_dynamics(
    python: Python<'_>,
    state: Vec<f64>,
    final_time: f64,
    mu: f64,
    initial_time: f64,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
) -> PyResult<CartesianState> {
    let state = six(state).map_err(to_python)?;
    python
        .detach(move || {
            KeplerDynamics.propagate(
                initial_time,
                state,
                final_time,
                mu,
                options(relative_tolerance, absolute_tolerance, maximum_step),
            )
        })
        .map(|result| result.state)
        .map_err(to_python)
}

/// Adaptively propagate CR3BP dynamics.
#[pyfunction(signature = (
    state,
    final_time,
    mu,
    initial_time = 0.0,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None
))]
#[allow(clippy::too_many_arguments)]
fn propagate_cr3bp(
    python: Python<'_>,
    state: Vec<f64>,
    final_time: f64,
    mu: f64,
    initial_time: f64,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
) -> PyResult<CartesianState> {
    let state = six(state).map_err(to_python)?;
    python
        .detach(move || {
            Cr3bpDynamics.propagate(
                initial_time,
                state,
                final_time,
                mu,
                options(relative_tolerance, absolute_tolerance, maximum_step),
            )
        })
        .map(|result| result.state)
        .map_err(to_python)
}

/// Adaptively propagate time-dependent bicircular dynamics.
#[pyfunction(signature = (
    state,
    final_time,
    mu,
    mu_sun,
    rho_sun,
    omega_sun,
    initial_time = 0.0,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None
))]
#[allow(clippy::too_many_arguments)]
fn propagate_bcp(
    python: Python<'_>,
    state: Vec<f64>,
    final_time: f64,
    mu: f64,
    mu_sun: f64,
    rho_sun: f64,
    omega_sun: f64,
    initial_time: f64,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
) -> PyResult<CartesianState> {
    let state = six(state).map_err(to_python)?;
    python
        .detach(move || {
            BcpDynamics.propagate(
                initial_time,
                state,
                final_time,
                [mu, mu_sun, rho_sun, omega_sun],
                options(relative_tolerance, absolute_tolerance, maximum_step),
            )
        })
        .map(|result| result.state)
        .map_err(to_python)
}

/// Propagate evaluated two-body dynamics with a 6 × 6 STM.
#[pyfunction(signature = (
    state,
    final_time,
    mu,
    initial_time = 0.0,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None
))]
#[allow(clippy::too_many_arguments)]
fn propagate_kepler_dynamics_with_stm(
    python: Python<'_>,
    state: Vec<f64>,
    final_time: f64,
    mu: f64,
    initial_time: f64,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
) -> PyResult<(CartesianState, Vec<Vec<f64>>)> {
    let state = six(state).map_err(to_python)?;
    python
        .detach(move || {
            KeplerDynamics.propagate_with_stm(
                initial_time,
                state,
                final_time,
                mu,
                options(relative_tolerance, absolute_tolerance, maximum_step),
            )
        })
        .map(|result| (result.state, rows(result.sensitivities)))
        .map_err(to_python)
}

/// Propagate CR3BP dynamics with a 6 × 6 STM.
#[pyfunction(signature = (
    state,
    final_time,
    mu,
    initial_time = 0.0,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None
))]
#[allow(clippy::too_many_arguments)]
fn propagate_cr3bp_with_stm(
    python: Python<'_>,
    state: Vec<f64>,
    final_time: f64,
    mu: f64,
    initial_time: f64,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
) -> PyResult<(CartesianState, Vec<Vec<f64>>)> {
    let state = six(state).map_err(to_python)?;
    python
        .detach(move || {
            Cr3bpDynamics.propagate_with_stm(
                initial_time,
                state,
                final_time,
                mu,
                options(relative_tolerance, absolute_tolerance, maximum_step),
            )
        })
        .map(|result| (result.state, rows(result.sensitivities)))
        .map_err(to_python)
}

/// Propagate bicircular dynamics with a 6 × 6 STM.
#[pyfunction(signature = (
    state,
    final_time,
    mu,
    mu_sun,
    rho_sun,
    omega_sun,
    initial_time = 0.0,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None
))]
#[allow(clippy::too_many_arguments)]
fn propagate_bcp_with_stm(
    python: Python<'_>,
    state: Vec<f64>,
    final_time: f64,
    mu: f64,
    mu_sun: f64,
    rho_sun: f64,
    omega_sun: f64,
    initial_time: f64,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
) -> PyResult<(CartesianState, Vec<Vec<f64>>)> {
    let state = six(state).map_err(to_python)?;
    python
        .detach(move || {
            BcpDynamics.propagate_with_stm(
                initial_time,
                state,
                final_time,
                [mu, mu_sun, rho_sun, omega_sun],
                options(relative_tolerance, absolute_tolerance, maximum_step),
            )
        })
        .map(|result| (result.state, rows(result.sensitivities)))
        .map_err(to_python)
}

/// Batch-propagate evaluated two-body dynamics.
#[pyfunction(signature = (
    states,
    final_times,
    mu,
    initial_time = 0.0,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None,
    workers = 0
))]
#[allow(clippy::too_many_arguments)]
fn propagate_kepler_dynamics_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    final_times: PyReadonlyArray1<'py, f64>,
    mu: f64,
    initial_time: f64,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let options = options(relative_tolerance, absolute_tolerance, maximum_step);
    cartesian_batch(
        python,
        states,
        final_times,
        workers,
        move |state, final_time| {
            KeplerDynamics
                .propagate(initial_time, state, final_time, mu, options)
                .map(|result| result.state)
        },
    )
}

/// Batch-propagate CR3BP dynamics.
#[pyfunction(signature = (
    states,
    final_times,
    mu,
    initial_time = 0.0,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None,
    workers = 0
))]
#[allow(clippy::too_many_arguments)]
fn propagate_cr3bp_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    final_times: PyReadonlyArray1<'py, f64>,
    mu: f64,
    initial_time: f64,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let options = options(relative_tolerance, absolute_tolerance, maximum_step);
    cartesian_batch(
        python,
        states,
        final_times,
        workers,
        move |state, final_time| {
            Cr3bpDynamics
                .propagate(initial_time, state, final_time, mu, options)
                .map(|result| result.state)
        },
    )
}

/// Batch-propagate bicircular dynamics.
#[pyfunction(signature = (
    states,
    final_times,
    mu,
    mu_sun,
    rho_sun,
    omega_sun,
    initial_time = 0.0,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None,
    workers = 0
))]
#[allow(clippy::too_many_arguments)]
fn propagate_bcp_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    final_times: PyReadonlyArray1<'py, f64>,
    mu: f64,
    mu_sun: f64,
    rho_sun: f64,
    omega_sun: f64,
    initial_time: f64,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let parameters = [mu, mu_sun, rho_sun, omega_sun];
    let options = options(relative_tolerance, absolute_tolerance, maximum_step);
    cartesian_batch(
        python,
        states,
        final_times,
        workers,
        move |state, final_time| {
            BcpDynamics
                .propagate(initial_time, state, final_time, parameters, options)
                .map(|result| result.state)
        },
    )
}

/// Batch-propagate evaluated two-body dynamics and STMs.
#[pyfunction(signature = (
    states,
    final_times,
    mu,
    initial_time = 0.0,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None,
    workers = 0
))]
#[allow(clippy::too_many_arguments)]
fn propagate_kepler_dynamics_with_stm_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    final_times: PyReadonlyArray1<'py, f64>,
    mu: f64,
    initial_time: f64,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
    workers: usize,
) -> PyResult<PyStateStmBatch<'py>> {
    let options = options(relative_tolerance, absolute_tolerance, maximum_step);
    cartesian_stm_batch(
        python,
        states,
        final_times,
        workers,
        move |state, final_time| {
            KeplerDynamics
                .propagate_with_stm(initial_time, state, final_time, mu, options)
                .map(|result| (result.state, result.sensitivities))
        },
    )
}

/// Batch-propagate CR3BP dynamics and STMs.
#[pyfunction(signature = (
    states,
    final_times,
    mu,
    initial_time = 0.0,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None,
    workers = 0
))]
#[allow(clippy::too_many_arguments)]
fn propagate_cr3bp_with_stm_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    final_times: PyReadonlyArray1<'py, f64>,
    mu: f64,
    initial_time: f64,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
    workers: usize,
) -> PyResult<PyStateStmBatch<'py>> {
    let options = options(relative_tolerance, absolute_tolerance, maximum_step);
    cartesian_stm_batch(
        python,
        states,
        final_times,
        workers,
        move |state, final_time| {
            Cr3bpDynamics
                .propagate_with_stm(initial_time, state, final_time, mu, options)
                .map(|result| (result.state, result.sensitivities))
        },
    )
}

/// Batch-propagate bicircular dynamics and STMs.
#[pyfunction(signature = (
    states,
    final_times,
    mu,
    mu_sun,
    rho_sun,
    omega_sun,
    initial_time = 0.0,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-12,
    maximum_step = None,
    workers = 0
))]
#[allow(clippy::too_many_arguments)]
fn propagate_bcp_with_stm_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    final_times: PyReadonlyArray1<'py, f64>,
    mu: f64,
    mu_sun: f64,
    rho_sun: f64,
    omega_sun: f64,
    initial_time: f64,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_step: Option<f64>,
    workers: usize,
) -> PyResult<PyStateStmBatch<'py>> {
    let parameters = [mu, mu_sun, rho_sun, omega_sun];
    let options = options(relative_tolerance, absolute_tolerance, maximum_step);
    cartesian_stm_batch(
        python,
        states,
        final_times,
        workers,
        move |state, final_time| {
            BcpDynamics
                .propagate_with_stm(initial_time, state, final_time, parameters, options)
                .map(|result| (result.state, result.sensitivities))
        },
    )
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyOptimality>()?;
    module.add_function(wrap_pyfunction!(kepler_rhs, module)?)?;
    module.add_function(wrap_pyfunction!(kepler_rhs_batch, module)?)?;
    module.add_function(wrap_pyfunction!(cr3bp_rhs, module)?)?;
    module.add_function(wrap_pyfunction!(cr3bp_rhs_batch, module)?)?;
    module.add_function(wrap_pyfunction!(bcp_rhs, module)?)?;
    module.add_function(wrap_pyfunction!(bcp_rhs_batch, module)?)?;
    module.add_function(wrap_pyfunction!(cr3bp_effective_potential, module)?)?;
    module.add_function(wrap_pyfunction!(cr3bp_effective_potential_batch, module)?)?;
    module.add_function(wrap_pyfunction!(cr3bp_jacobi_constant, module)?)?;
    module.add_function(wrap_pyfunction!(cr3bp_jacobi_constant_batch, module)?)?;
    module.add_function(wrap_pyfunction!(zoh_kepler_rhs, module)?)?;
    module.add_function(wrap_pyfunction!(zoh_kepler_rhs_batch, module)?)?;
    module.add_function(wrap_pyfunction!(zoh_cr3bp_rhs, module)?)?;
    module.add_function(wrap_pyfunction!(zoh_cr3bp_rhs_batch, module)?)?;
    module.add_function(wrap_pyfunction!(zoh_equinoctial_rhs, module)?)?;
    module.add_function(wrap_pyfunction!(zoh_equinoctial_rhs_batch, module)?)?;
    module.add_function(wrap_pyfunction!(zoh_solar_sail_rhs, module)?)?;
    module.add_function(wrap_pyfunction!(zoh_solar_sail_rhs_batch, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_zoh_kepler, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_zoh_kepler_batch, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_zoh_cr3bp, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_zoh_cr3bp_batch, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_zoh_equinoctial, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_zoh_equinoctial_batch, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_zoh_solar_sail, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_zoh_solar_sail_batch, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_kepler_dynamics, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_kepler_dynamics_batch, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_cr3bp, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_cr3bp_batch, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_bcp, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_bcp_batch, module)?)?;
    module.add_function(wrap_pyfunction!(
        propagate_kepler_dynamics_with_stm,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        propagate_kepler_dynamics_with_stm_batch,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(propagate_cr3bp_with_stm, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_cr3bp_with_stm_batch, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_bcp_with_stm, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_bcp_with_stm_batch, module)?)?;
    module.add_function(wrap_pyfunction!(pontryagin_cartesian_rhs, module)?)?;
    module.add_function(wrap_pyfunction!(pontryagin_cartesian_rhs_batch, module)?)?;
    module.add_function(wrap_pyfunction!(pontryagin_equinoctial_rhs, module)?)?;
    module.add_function(wrap_pyfunction!(pontryagin_equinoctial_rhs_batch, module)?)?;
    module.add_function(wrap_pyfunction!(pontryagin_cartesian_control, module)?)?;
    module.add_function(wrap_pyfunction!(
        pontryagin_cartesian_control_batch,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(pontryagin_equinoctial_control, module)?)?;
    module.add_function(wrap_pyfunction!(
        pontryagin_equinoctial_control_batch,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(pontryagin_cartesian_hamiltonian, module)?)?;
    module.add_function(wrap_pyfunction!(
        pontryagin_cartesian_hamiltonian_batch,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        pontryagin_equinoctial_hamiltonian,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        pontryagin_equinoctial_hamiltonian_batch,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(propagate_pontryagin_cartesian, module)?)?;
    module.add_function(wrap_pyfunction!(
        propagate_pontryagin_cartesian_batch,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(propagate_pontryagin_equinoctial, module)?)?;
    module.add_function(wrap_pyfunction!(
        propagate_pontryagin_equinoctial_batch,
        module
    )?)?;
    Ok(())
}
