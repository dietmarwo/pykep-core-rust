// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

use crate::error::to_python;
use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyUntypedArrayMethods};
use pykep_core::astro::elements::{
    ClassicalElements, ModifiedEquinoctialElements, cartesian_to_classical,
    cartesian_to_modified_equinoctial, cartesian_to_modified_equinoctial_jacobian,
    classical_to_cartesian, classical_to_modified_equinoctial, modified_equinoctial_to_cartesian,
    modified_equinoctial_to_cartesian_jacobian, modified_equinoctial_to_classical,
};
use pykep_core::{Elements6, Matrix6, PykepError};
use pyo3::prelude::*;

fn six(values: Vec<f64>) -> Result<Elements6, PykepError> {
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

fn batch<'py, F>(
    python: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    workers: usize,
    operation: F,
) -> PyResult<Bound<'py, PyArray2<f64>>>
where
    F: Fn(Elements6) -> pykep_core::Result<Elements6> + Sync + Send + 'static,
{
    let shape = values.shape();
    if shape[1] != 6 {
        return Err(to_python(PykepError::DimensionMismatch {
            expected: 6,
            actual: shape[1],
        }));
    }
    let input = values
        .as_array()
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1], row[2], row[3], row[4], row[5]])
        .collect::<Vec<_>>();
    let output = python
        .detach(move || pykep_core::batch::try_map(&input, workers, |row| operation(*row)))
        .map_err(to_python)?;
    let array = Array2::from_shape_vec((shape[0], 6), output.into_iter().flatten().collect())
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(python))
}

fn matrix_batch<'py, F>(
    python: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    workers: usize,
    operation: F,
) -> PyResult<Bound<'py, PyArray3<f64>>>
where
    F: Fn(Elements6) -> pykep_core::Result<Matrix6> + Sync + Send + 'static,
{
    let shape = values.shape();
    if shape[1] != 6 {
        return Err(to_python(PykepError::DimensionMismatch {
            expected: 6,
            actual: shape[1],
        }));
    }
    let input = values
        .as_array()
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1], row[2], row[3], row[4], row[5]])
        .collect::<Vec<_>>();
    let output = python
        .detach(move || pykep_core::batch::try_map(&input, workers, |row| operation(*row)))
        .map_err(to_python)?;
    let array = Array3::from_shape_vec(
        (shape[0], 6, 6),
        output
            .into_iter()
            .flat_map(|matrix| matrix.into_iter().flatten())
            .collect(),
    )
    .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(python))
}

/// Convert a Cartesian `[x,y,z,vx,vy,vz]` state to `[a,e,i,Omega,omega,nu]`.
#[pyfunction(name = "cartesian_to_classical")]
fn cartesian_to_classical_py(state: Vec<f64>, mu: f64) -> PyResult<Elements6> {
    cartesian_to_classical(&six(state).map_err(to_python)?, mu)
        .map(ClassicalElements::to_array)
        .map_err(to_python)
}

/// Convert classical `[a,e,i,Omega,omega,nu]` elements to a Cartesian state.
#[pyfunction(name = "classical_to_cartesian")]
fn classical_to_cartesian_py(elements: Vec<f64>, mu: f64) -> PyResult<Elements6> {
    classical_to_cartesian(six(elements).map_err(to_python)?.into(), mu).map_err(to_python)
}

/// Convert classical elements to modified equinoctial `[p,f,g,h,k,L]`.
#[pyfunction(name = "classical_to_modified_equinoctial")]
#[pyo3(signature = (elements, retrograde=false))]
fn classical_to_modified_equinoctial_py(
    elements: Vec<f64>,
    retrograde: bool,
) -> PyResult<Elements6> {
    classical_to_modified_equinoctial(six(elements).map_err(to_python)?.into(), retrograde)
        .map(ModifiedEquinoctialElements::to_array)
        .map_err(to_python)
}

/// Convert modified equinoctial elements to classical elements.
#[pyfunction(name = "modified_equinoctial_to_classical")]
#[pyo3(signature = (elements, retrograde=false))]
fn modified_equinoctial_to_classical_py(
    elements: Vec<f64>,
    retrograde: bool,
) -> PyResult<Elements6> {
    modified_equinoctial_to_classical(six(elements).map_err(to_python)?.into(), retrograde)
        .map(ClassicalElements::to_array)
        .map_err(to_python)
}

/// Convert a Cartesian state directly to modified equinoctial elements.
#[pyfunction(name = "cartesian_to_modified_equinoctial")]
#[pyo3(signature = (state, mu, retrograde=false))]
fn cartesian_to_modified_equinoctial_py(
    state: Vec<f64>,
    mu: f64,
    retrograde: bool,
) -> PyResult<Elements6> {
    cartesian_to_modified_equinoctial(&six(state).map_err(to_python)?, mu, retrograde)
        .map(ModifiedEquinoctialElements::to_array)
        .map_err(to_python)
}

/// Convert modified equinoctial elements directly to a Cartesian state.
#[pyfunction(name = "modified_equinoctial_to_cartesian")]
#[pyo3(signature = (elements, mu, retrograde=false))]
fn modified_equinoctial_to_cartesian_py(
    elements: Vec<f64>,
    mu: f64,
    retrograde: bool,
) -> PyResult<Elements6> {
    modified_equinoctial_to_cartesian(six(elements).map_err(to_python)?.into(), mu, retrograde)
        .map_err(to_python)
}

/// Return the row-major Cartesian-to-equinoctial analytic Jacobian.
#[pyfunction(name = "cartesian_to_modified_equinoctial_jacobian")]
#[pyo3(signature = (state, mu, retrograde=false))]
fn cartesian_to_modified_equinoctial_jacobian_py(
    state: Vec<f64>,
    mu: f64,
    retrograde: bool,
) -> PyResult<Vec<Vec<f64>>> {
    cartesian_to_modified_equinoctial_jacobian(&six(state).map_err(to_python)?, mu, retrograde)
        .map(rows)
        .map_err(to_python)
}

/// Return the row-major equinoctial-to-Cartesian analytic Jacobian.
#[pyfunction(name = "modified_equinoctial_to_cartesian_jacobian")]
#[pyo3(signature = (elements, mu, retrograde=false))]
fn modified_equinoctial_to_cartesian_jacobian_py(
    elements: Vec<f64>,
    mu: f64,
    retrograde: bool,
) -> PyResult<Vec<Vec<f64>>> {
    modified_equinoctial_to_cartesian_jacobian(
        six(elements).map_err(to_python)?.into(),
        mu,
        retrograde,
    )
    .map(rows)
    .map_err(to_python)
}

/// Batch-convert an `N x 6` NumPy array of Cartesian states to classical elements.
#[pyfunction(signature = (states, mu, workers=0))]
fn cartesian_to_classical_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    mu: f64,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    batch(python, states, workers, move |state| {
        cartesian_to_classical(&state, mu).map(ClassicalElements::to_array)
    })
}

/// Batch-convert an `N x 6` NumPy array of classical elements to Cartesian states.
#[pyfunction(signature = (elements, mu, workers=0))]
fn classical_to_cartesian_batch<'py>(
    python: Python<'py>,
    elements: PyReadonlyArray2<'py, f64>,
    mu: f64,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    batch(python, elements, workers, move |elements| {
        classical_to_cartesian(elements.into(), mu)
    })
}

/// Batch-convert an `N x 6` NumPy array of classical elements to equinoctial elements.
#[pyfunction]
#[pyo3(signature = (elements, retrograde=false, workers=0))]
fn classical_to_modified_equinoctial_batch<'py>(
    python: Python<'py>,
    elements: PyReadonlyArray2<'py, f64>,
    retrograde: bool,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    batch(python, elements, workers, move |elements| {
        classical_to_modified_equinoctial(elements.into(), retrograde)
            .map(ModifiedEquinoctialElements::to_array)
    })
}

/// Batch-convert an `N x 6` NumPy array of equinoctial elements to classical elements.
#[pyfunction]
#[pyo3(signature = (elements, retrograde=false, workers=0))]
fn modified_equinoctial_to_classical_batch<'py>(
    python: Python<'py>,
    elements: PyReadonlyArray2<'py, f64>,
    retrograde: bool,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    batch(python, elements, workers, move |elements| {
        modified_equinoctial_to_classical(elements.into(), retrograde)
            .map(ClassicalElements::to_array)
    })
}

/// Batch-convert an `N x 6` NumPy array of Cartesian states to equinoctial elements.
#[pyfunction]
#[pyo3(signature = (states, mu, retrograde=false, workers=0))]
fn cartesian_to_modified_equinoctial_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    mu: f64,
    retrograde: bool,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    batch(python, states, workers, move |state| {
        cartesian_to_modified_equinoctial(&state, mu, retrograde)
            .map(ModifiedEquinoctialElements::to_array)
    })
}

/// Batch-convert an `N x 6` NumPy array of equinoctial elements to Cartesian states.
#[pyfunction]
#[pyo3(signature = (elements, mu, retrograde=false, workers=0))]
fn modified_equinoctial_to_cartesian_batch<'py>(
    python: Python<'py>,
    elements: PyReadonlyArray2<'py, f64>,
    mu: f64,
    retrograde: bool,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    batch(python, elements, workers, move |elements| {
        modified_equinoctial_to_cartesian(elements.into(), mu, retrograde)
    })
}

/// Batch-evaluate Cartesian-to-equinoctial analytic Jacobians.
#[pyfunction]
#[pyo3(signature = (states, mu, retrograde=false, workers=0))]
fn cartesian_to_modified_equinoctial_jacobian_batch<'py>(
    python: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    mu: f64,
    retrograde: bool,
    workers: usize,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    matrix_batch(python, states, workers, move |state| {
        cartesian_to_modified_equinoctial_jacobian(&state, mu, retrograde)
    })
}

/// Batch-evaluate equinoctial-to-Cartesian analytic Jacobians.
#[pyfunction]
#[pyo3(signature = (elements, mu, retrograde=false, workers=0))]
fn modified_equinoctial_to_cartesian_jacobian_batch<'py>(
    python: Python<'py>,
    elements: PyReadonlyArray2<'py, f64>,
    mu: f64,
    retrograde: bool,
    workers: usize,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    matrix_batch(python, elements, workers, move |elements| {
        modified_equinoctial_to_cartesian_jacobian(elements.into(), mu, retrograde)
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(cartesian_to_classical_py, module)?)?;
    module.add_function(wrap_pyfunction!(classical_to_cartesian_py, module)?)?;
    module.add_function(wrap_pyfunction!(
        classical_to_modified_equinoctial_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        modified_equinoctial_to_classical_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        cartesian_to_modified_equinoctial_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        modified_equinoctial_to_cartesian_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        cartesian_to_modified_equinoctial_jacobian_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        modified_equinoctial_to_cartesian_jacobian_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(cartesian_to_classical_batch, module)?)?;
    module.add_function(wrap_pyfunction!(classical_to_cartesian_batch, module)?)?;
    module.add_function(wrap_pyfunction!(
        classical_to_modified_equinoctial_batch,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        modified_equinoctial_to_classical_batch,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        cartesian_to_modified_equinoctial_batch,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        modified_equinoctial_to_cartesian_batch,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        cartesian_to_modified_equinoctial_jacobian_batch,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        modified_equinoctial_to_cartesian_jacobian_batch,
        module
    )?)?;
    Ok(())
}
