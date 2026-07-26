// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

use crate::error::to_python;
use numpy::ndarray::{Array2, Array3};
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pykep_core::astro::propagation::{
    propagate_keplerian, propagate_keplerian_batch as core_propagate_keplerian_batch,
    propagate_lagrangian, propagate_lagrangian_batch as core_propagate_lagrangian_batch,
    propagate_lagrangian_grid_parallel, propagate_lagrangian_with_stm,
    propagate_lagrangian_with_stm_batch as core_propagate_lagrangian_with_stm_batch,
    propagate_universal, propagate_universal_batch as core_propagate_universal_batch,
    state_transition_matrix_lagrangian, state_transition_matrix_reynolds,
};
use pykep_core::{CartesianState, Matrix6, PykepError};
use pyo3::prelude::*;

type PropagationBatchFn =
    fn(&[CartesianState], &[f64], f64, usize) -> pykep_core::Result<Vec<CartesianState>>;
type PyStateStmBatch<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray3<f64>>);

fn six(values: Vec<f64>) -> Result<CartesianState, PykepError> {
    values
        .try_into()
        .map_err(|values: Vec<f64>| PykepError::DimensionMismatch {
            expected: 6,
            actual: values.len(),
        })
}

fn rows(matrix: Matrix6) -> Vec<Vec<f64>> {
    matrix.into_iter().map(Vec::from).collect()
}

fn state_rows<'py>(
    python: Python<'py>,
    values: Vec<CartesianState>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let row_count = values.len();
    let flat = values.into_iter().flatten().collect();
    let array = Array2::from_shape_vec((row_count, 6), flat)
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(python))
}

fn batch_inputs(
    states: PyReadonlyArray2<'_, f64>,
    times: PyReadonlyArray1<'_, f64>,
) -> PyResult<(Vec<CartesianState>, Vec<f64>)> {
    let shape = states.shape();
    if shape[1] != 6 {
        return Err(to_python(PykepError::DimensionMismatch {
            expected: 6,
            actual: shape[1],
        }));
    }
    if times.len() != shape[0] {
        return Err(to_python(PykepError::DimensionMismatch {
            expected: shape[0],
            actual: times.len(),
        }));
    }
    let states = states
        .as_array()
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1], row[2], row[3], row[4], row[5]])
        .collect();
    let times = times.as_array().iter().copied().collect();
    Ok((states, times))
}

fn propagate_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
    mu: f64,
    workers: usize,
    operation: PropagationBatchFn,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (states, times) = batch_inputs(states, times)?;
    let output = python
        .detach(move || operation(&states, &times, mu, workers))
        .map_err(to_python)?;
    state_rows(python, output)
}

/// Propagate a Cartesian state with elliptic/hyperbolic Lagrange coefficients.
#[pyfunction(name = "propagate_lagrangian")]
fn propagate_lagrangian_py(state: Vec<f64>, time: f64, mu: f64) -> PyResult<CartesianState> {
    propagate_lagrangian(&six(state).map_err(to_python)?, time, mu).map_err(to_python)
}

/// Propagate a Cartesian state with the universal-variable formulation.
#[pyfunction(name = "propagate_universal")]
fn propagate_universal_py(state: Vec<f64>, time: f64, mu: f64) -> PyResult<CartesianState> {
    propagate_universal(&six(state).map_err(to_python)?, time, mu).map_err(to_python)
}

/// Propagate by advancing the mean anomaly of classical elements.
#[pyfunction(name = "propagate_keplerian")]
fn propagate_keplerian_py(state: Vec<f64>, time: f64, mu: f64) -> PyResult<CartesianState> {
    propagate_keplerian(&six(state).map_err(to_python)?, time, mu).map_err(to_python)
}

/// Propagate and return the row-major Lagrangian state-transition matrix.
#[pyfunction(name = "propagate_lagrangian_with_stm")]
fn propagate_lagrangian_with_stm_py(
    state: Vec<f64>,
    time: f64,
    mu: f64,
) -> PyResult<(CartesianState, Vec<Vec<f64>>)> {
    propagate_lagrangian_with_stm(&six(state).map_err(to_python)?, time, mu)
        .map(|(state, matrix)| (state, rows(matrix)))
        .map_err(to_python)
}

/// Return the row-major Lagrangian state-transition matrix.
#[pyfunction(name = "state_transition_matrix_lagrangian")]
fn state_transition_matrix_lagrangian_py(
    state: Vec<f64>,
    time: f64,
    mu: f64,
) -> PyResult<Vec<Vec<f64>>> {
    state_transition_matrix_lagrangian(&six(state).map_err(to_python)?, time, mu)
        .map(rows)
        .map_err(to_python)
}

/// Return the row-major Reynolds state-transition matrix.
#[pyfunction(name = "state_transition_matrix_reynolds")]
fn state_transition_matrix_reynolds_py(
    initial_state: Vec<f64>,
    final_state: Vec<f64>,
    time: f64,
    mu: f64,
) -> PyResult<Vec<Vec<f64>>> {
    state_transition_matrix_reynolds(
        &six(initial_state).map_err(to_python)?,
        &six(final_state).map_err(to_python)?,
        time,
        mu,
    )
    .map(rows)
    .map_err(to_python)
}

/// Propagate `N x 6` states for `N` durations, releasing the Python GIL.
#[pyfunction(signature = (states, times, mu, workers=0))]
fn propagate_lagrangian_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
    mu: f64,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    propagate_batch(
        python,
        states,
        times,
        mu,
        workers,
        core_propagate_lagrangian_batch,
    )
}

/// Universally propagate `N x 6` states for `N` durations, releasing the GIL.
#[pyfunction(signature = (states, times, mu, workers=0))]
fn propagate_universal_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
    mu: f64,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    propagate_batch(
        python,
        states,
        times,
        mu,
        workers,
        core_propagate_universal_batch,
    )
}

/// Keplerian-propagate `N x 6` states for `N` durations, releasing the GIL.
#[pyfunction(signature = (states, times, mu, workers=0))]
fn propagate_keplerian_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
    mu: f64,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    propagate_batch(
        python,
        states,
        times,
        mu,
        workers,
        core_propagate_keplerian_batch,
    )
}

/// Propagate `N` states and analytic STMs, releasing the GIL.
#[pyfunction(signature = (states, times, mu, workers=0))]
fn propagate_lagrangian_with_stm_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
    mu: f64,
    workers: usize,
) -> PyResult<PyStateStmBatch<'py>> {
    let (states, times) = batch_inputs(states, times)?;
    let output = python
        .detach(move || core_propagate_lagrangian_with_stm_batch(&states, &times, mu, workers))
        .map_err(to_python)?;
    let count = output.len();
    let mut propagated = Vec::with_capacity(count);
    let mut matrices = Vec::with_capacity(count * 36);
    for (state, matrix) in output {
        propagated.push(state);
        matrices.extend(matrix.into_iter().flatten());
    }
    let propagated = state_rows(python, propagated)?;
    let matrices = Array3::from_shape_vec((count, 6, 6), matrices)
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?
        .into_pyarray(python);
    Ok((propagated, matrices))
}

/// Propagate one state over a time grid relative to its first entry.
#[pyfunction(name = "propagate_lagrangian_grid", signature = (state, time_grid, mu, workers=0))]
fn propagate_lagrangian_grid_py<'py>(
    python: Python<'py>,
    state: Vec<f64>,
    time_grid: PyReadonlyArray1<'py, f64>,
    mu: f64,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let state = six(state).map_err(to_python)?;
    let time_grid: Vec<_> = time_grid.as_array().iter().copied().collect();
    let output = python
        .detach(move || propagate_lagrangian_grid_parallel(&state, &time_grid, mu, workers))
        .map_err(to_python)?;
    state_rows(python, output)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(propagate_lagrangian_py, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_universal_py, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_keplerian_py, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_lagrangian_with_stm_py, module)?)?;
    module.add_function(wrap_pyfunction!(
        state_transition_matrix_lagrangian_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        state_transition_matrix_reynolds_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(propagate_lagrangian_batch, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_universal_batch, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_keplerian_batch, module)?)?;
    module.add_function(wrap_pyfunction!(
        propagate_lagrangian_with_stm_batch,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(propagate_lagrangian_grid_py, module)?)?;
    Ok(())
}
