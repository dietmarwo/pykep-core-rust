// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

use crate::error::to_python;
use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2, PyUntypedArrayMethods};
use pykep_core::constants;
use pykep_core::math::{kepler_equations, linalg, stumpff};
use pykep_core::time::julian;
use pykep_core::{PykepError, Vector3};
use pyo3::prelude::*;

fn vector3(values: Vec<f64>) -> Result<Vector3, PykepError> {
    match values.as_slice() {
        [x, y, z] => Ok([*x, *y, *z]),
        _ => Err(PykepError::DimensionMismatch {
            expected: 3,
            actual: values.len(),
        }),
    }
}

fn vector3_rows(values: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<Vector3>> {
    let shape = values.shape();
    if shape[1] != 3 {
        return Err(to_python(PykepError::DimensionMismatch {
            expected: 3,
            actual: shape[1],
        }));
    }
    Ok(values
        .as_array()
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect())
}

/// Convert Julian date to modified Julian date, in days.
#[pyfunction]
fn jd_to_mjd(value: f64) -> PyResult<f64> {
    julian::jd_to_mjd(value).map_err(to_python)
}

/// Convert Julian date to MJD2000, in days.
#[pyfunction]
fn jd_to_mjd2000(value: f64) -> PyResult<f64> {
    julian::jd_to_mjd2000(value).map_err(to_python)
}

/// Convert modified Julian date to Julian date, in days.
#[pyfunction]
fn mjd_to_jd(value: f64) -> PyResult<f64> {
    julian::mjd_to_jd(value).map_err(to_python)
}

/// Convert modified Julian date to MJD2000, in days.
#[pyfunction]
fn mjd_to_mjd2000(value: f64) -> PyResult<f64> {
    julian::mjd_to_mjd2000(value).map_err(to_python)
}

/// Convert MJD2000 to Julian date, in days.
#[pyfunction]
fn mjd2000_to_jd(value: f64) -> PyResult<f64> {
    julian::mjd2000_to_jd(value).map_err(to_python)
}

/// Convert MJD2000 to modified Julian date, in days.
#[pyfunction]
fn mjd2000_to_mjd(value: f64) -> PyResult<f64> {
    julian::mjd2000_to_mjd(value).map_err(to_python)
}

/// Evaluate the dimensionless Stumpff C function.
#[pyfunction]
fn stumpff_c(value: f64) -> PyResult<f64> {
    stumpff::stumpff_c(value).map_err(to_python)
}

/// Evaluate the dimensionless Stumpff S function.
#[pyfunction]
fn stumpff_s(value: f64) -> PyResult<f64> {
    stumpff::stumpff_s(value).map_err(to_python)
}

/// Evaluate the Stumpff C function for a sequence, preserving input order.
#[pyfunction(signature = (values, workers=0))]
fn stumpff_c_batch(python: Python<'_>, values: Vec<f64>, workers: usize) -> PyResult<Vec<f64>> {
    python
        .detach(move || {
            pykep_core::batch::try_map(&values, workers, |value| stumpff::stumpff_c(*value))
        })
        .map_err(to_python)
}

/// Evaluate the Stumpff S function for a sequence, preserving input order.
#[pyfunction(signature = (values, workers=0))]
fn stumpff_s_batch(python: Python<'_>, values: Vec<f64>, workers: usize) -> PyResult<Vec<f64>> {
    python
        .detach(move || {
            pykep_core::batch::try_map(&values, workers, |value| stumpff::stumpff_s(*value))
        })
        .map_err(to_python)
}

/// Evaluate the elliptic Kepler residual in radians.
#[pyfunction]
fn elliptic_kepler_residual(
    eccentric_anomaly: f64,
    mean_anomaly: f64,
    eccentricity: f64,
) -> PyResult<f64> {
    kepler_equations::elliptic_kepler_residual(eccentric_anomaly, mean_anomaly, eccentricity)
        .map_err(to_python)
}

/// Evaluate the first derivative of the elliptic Kepler residual.
#[pyfunction]
fn elliptic_kepler_derivative(eccentric_anomaly: f64, eccentricity: f64) -> PyResult<f64> {
    kepler_equations::elliptic_kepler_derivative(eccentric_anomaly, eccentricity).map_err(to_python)
}

/// Evaluate the second derivative of the elliptic Kepler residual.
#[pyfunction]
fn elliptic_kepler_second_derivative(eccentric_anomaly: f64, eccentricity: f64) -> PyResult<f64> {
    kepler_equations::elliptic_kepler_second_derivative(eccentric_anomaly, eccentricity)
        .map_err(to_python)
}

/// Evaluate the hyperbolic Kepler residual.
#[pyfunction]
fn hyperbolic_kepler_residual(
    hyperbolic_anomaly: f64,
    mean_anomaly: f64,
    eccentricity: f64,
) -> PyResult<f64> {
    kepler_equations::hyperbolic_kepler_residual(hyperbolic_anomaly, mean_anomaly, eccentricity)
        .map_err(to_python)
}

/// Evaluate the first derivative of the hyperbolic Kepler residual.
#[pyfunction]
fn hyperbolic_kepler_derivative(hyperbolic_anomaly: f64, eccentricity: f64) -> PyResult<f64> {
    kepler_equations::hyperbolic_kepler_derivative(hyperbolic_anomaly, eccentricity)
        .map_err(to_python)
}

/// Evaluate the second derivative of the hyperbolic Kepler residual.
#[pyfunction]
fn hyperbolic_kepler_second_derivative(
    hyperbolic_anomaly: f64,
    eccentricity: f64,
) -> PyResult<f64> {
    kepler_equations::hyperbolic_kepler_second_derivative(hyperbolic_anomaly, eccentricity)
        .map_err(to_python)
}

/// Evaluate Kepler's equation in elliptic anomaly difference.
#[pyfunction]
fn elliptic_difference_residual(
    delta_eccentric_anomaly: f64,
    delta_mean_anomaly: f64,
    sigma0: f64,
    sqrt_semi_major_axis: f64,
    semi_major_axis: f64,
    initial_radius: f64,
) -> PyResult<f64> {
    kepler_equations::elliptic_difference_residual(
        delta_eccentric_anomaly,
        delta_mean_anomaly,
        sigma0,
        sqrt_semi_major_axis,
        semi_major_axis,
        initial_radius,
    )
    .map_err(to_python)
}

/// Evaluate the first derivative of the elliptic difference residual.
#[pyfunction]
fn elliptic_difference_derivative(
    delta_eccentric_anomaly: f64,
    sigma0: f64,
    sqrt_semi_major_axis: f64,
    semi_major_axis: f64,
    initial_radius: f64,
) -> PyResult<f64> {
    kepler_equations::elliptic_difference_derivative(
        delta_eccentric_anomaly,
        sigma0,
        sqrt_semi_major_axis,
        semi_major_axis,
        initial_radius,
    )
    .map_err(to_python)
}

/// Evaluate the second derivative of the elliptic difference residual.
#[pyfunction]
fn elliptic_difference_second_derivative(
    delta_eccentric_anomaly: f64,
    sigma0: f64,
    sqrt_semi_major_axis: f64,
    semi_major_axis: f64,
    initial_radius: f64,
) -> PyResult<f64> {
    kepler_equations::elliptic_difference_second_derivative(
        delta_eccentric_anomaly,
        sigma0,
        sqrt_semi_major_axis,
        semi_major_axis,
        initial_radius,
    )
    .map_err(to_python)
}

/// Evaluate Kepler's equation in hyperbolic anomaly difference.
#[pyfunction]
fn hyperbolic_difference_residual(
    delta_hyperbolic_anomaly: f64,
    delta_mean_anomaly: f64,
    sigma0: f64,
    sqrt_abs_semi_major_axis: f64,
    semi_major_axis: f64,
    initial_radius: f64,
) -> PyResult<f64> {
    kepler_equations::hyperbolic_difference_residual(
        delta_hyperbolic_anomaly,
        delta_mean_anomaly,
        sigma0,
        sqrt_abs_semi_major_axis,
        semi_major_axis,
        initial_radius,
    )
    .map_err(to_python)
}

/// Evaluate the first derivative of the hyperbolic difference residual.
#[pyfunction]
fn hyperbolic_difference_derivative(
    delta_hyperbolic_anomaly: f64,
    sigma0: f64,
    sqrt_abs_semi_major_axis: f64,
    semi_major_axis: f64,
    initial_radius: f64,
) -> PyResult<f64> {
    kepler_equations::hyperbolic_difference_derivative(
        delta_hyperbolic_anomaly,
        sigma0,
        sqrt_abs_semi_major_axis,
        semi_major_axis,
        initial_radius,
    )
    .map_err(to_python)
}

/// Evaluate the second derivative of the hyperbolic difference residual.
#[pyfunction]
fn hyperbolic_difference_second_derivative(
    delta_hyperbolic_anomaly: f64,
    sigma0: f64,
    sqrt_abs_semi_major_axis: f64,
    semi_major_axis: f64,
    initial_radius: f64,
) -> PyResult<f64> {
    kepler_equations::hyperbolic_difference_second_derivative(
        delta_hyperbolic_anomaly,
        sigma0,
        sqrt_abs_semi_major_axis,
        semi_major_axis,
        initial_radius,
    )
    .map_err(to_python)
}

/// Evaluate universal-variable Kepler's equation.
#[pyfunction]
fn universal_kepler_residual(
    delta_s: f64,
    delta_time: f64,
    initial_radius: f64,
    initial_radial_velocity: f64,
    alpha: f64,
    mu: f64,
) -> PyResult<f64> {
    kepler_equations::universal_kepler_residual(
        delta_s,
        delta_time,
        initial_radius,
        initial_radial_velocity,
        alpha,
        mu,
    )
    .map_err(to_python)
}

/// Evaluate the first derivative of universal-variable Kepler's equation.
#[pyfunction]
fn universal_kepler_derivative(
    delta_s: f64,
    initial_radius: f64,
    initial_radial_velocity: f64,
    alpha: f64,
    mu: f64,
) -> PyResult<f64> {
    kepler_equations::universal_kepler_derivative(
        delta_s,
        initial_radius,
        initial_radial_velocity,
        alpha,
        mu,
    )
    .map_err(to_python)
}

/// Evaluate the second derivative of universal-variable Kepler's equation.
#[pyfunction]
fn universal_kepler_second_derivative(
    delta_s: f64,
    initial_radius: f64,
    initial_radial_velocity: f64,
    alpha: f64,
    mu: f64,
) -> PyResult<f64> {
    kepler_equations::universal_kepler_second_derivative(
        delta_s,
        initial_radius,
        initial_radial_velocity,
        alpha,
        mu,
    )
    .map_err(to_python)
}

/// Compute the Euclidean dot product of two three-vectors.
#[pyfunction]
fn dot(left: Vec<f64>, right: Vec<f64>) -> PyResult<f64> {
    linalg::dot(
        &vector3(left).map_err(to_python)?,
        &vector3(right).map_err(to_python)?,
    )
    .map_err(to_python)
}

/// Compute the Euclidean norm of a three-vector.
#[pyfunction]
fn norm(vector: Vec<f64>) -> PyResult<f64> {
    linalg::norm(&vector3(vector).map_err(to_python)?).map_err(to_python)
}

/// Return a normalized three-vector.
#[pyfunction]
fn normalize(vector: Vec<f64>) -> PyResult<Vector3> {
    linalg::normalize(&vector3(vector).map_err(to_python)?).map_err(to_python)
}

/// Compute the right-handed cross product of two three-vectors.
#[pyfunction]
fn cross(left: Vec<f64>, right: Vec<f64>) -> PyResult<Vector3> {
    linalg::cross(
        &vector3(left).map_err(to_python)?,
        &vector3(right).map_err(to_python)?,
    )
    .map_err(to_python)
}

/// Return the row-major 3 × 3 skew-symmetric matrix of a three-vector.
#[pyfunction]
fn skew(vector: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
    let matrix = linalg::skew(&vector3(vector).map_err(to_python)?).map_err(to_python)?;
    Ok(matrix.into_iter().map(Vec::from).collect())
}

/// Compute ordered dot products for two `N x 3` arrays.
#[pyfunction(signature = (left, right, workers=0))]
fn dot_batch<'py>(
    python: Python<'py>,
    left: PyReadonlyArray2<'py, f64>,
    right: PyReadonlyArray2<'py, f64>,
    workers: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let left = vector3_rows(left)?;
    let right = vector3_rows(right)?;
    let values = python
        .detach(move || linalg::dot_batch(&left, &right, workers))
        .map_err(to_python)?;
    Ok(Array1::from_vec(values).into_pyarray(python))
}

/// Compute ordered norms for an `N x 3` array.
#[pyfunction(signature = (vectors, workers=0))]
fn norm_batch<'py>(
    python: Python<'py>,
    vectors: PyReadonlyArray2<'py, f64>,
    workers: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let vectors = vector3_rows(vectors)?;
    let values = python
        .detach(move || linalg::norm_batch(&vectors, workers))
        .map_err(to_python)?;
    Ok(Array1::from_vec(values).into_pyarray(python))
}

/// Normalize an ordered `N x 3` vector array.
#[pyfunction(signature = (vectors, workers=0))]
fn normalize_batch<'py>(
    python: Python<'py>,
    vectors: PyReadonlyArray2<'py, f64>,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let vectors = vector3_rows(vectors)?;
    let values = python
        .detach(move || linalg::normalize_batch(&vectors, workers))
        .map_err(to_python)?;
    let count = values.len();
    let array = Array2::from_shape_vec((count, 3), values.into_iter().flatten().collect())
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(python))
}

/// Compute ordered cross products for two `N x 3` arrays.
#[pyfunction(signature = (left, right, workers=0))]
fn cross_batch<'py>(
    python: Python<'py>,
    left: PyReadonlyArray2<'py, f64>,
    right: PyReadonlyArray2<'py, f64>,
    workers: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let left = vector3_rows(left)?;
    let right = vector3_rows(right)?;
    let values = python
        .detach(move || linalg::cross_batch(&left, &right, workers))
        .map_err(to_python)?;
    let count = values.len();
    let array = Array2::from_shape_vec((count, 3), values.into_iter().flatten().collect())
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(python))
}

/// Build ordered skew matrices for an `N x 3` vector array.
#[pyfunction(signature = (vectors, workers=0))]
fn skew_batch<'py>(
    python: Python<'py>,
    vectors: PyReadonlyArray2<'py, f64>,
    workers: usize,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let vectors = vector3_rows(vectors)?;
    let values = python
        .detach(move || linalg::skew_batch(&vectors, workers))
        .map_err(to_python)?;
    let count = values.len();
    let array = Array3::from_shape_vec(
        (count, 3, 3),
        values
            .into_iter()
            .flat_map(|matrix| matrix.into_iter().flatten())
            .collect(),
    )
    .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(python))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("PI", constants::PI)?;
    module.add("HALF_PI", constants::HALF_PI)?;
    module.add("ASTRONOMICAL_UNIT", constants::ASTRONOMICAL_UNIT)?;
    module.add("CAVENDISH_CONSTANT", constants::CAVENDISH_CONSTANT)?;
    module.add("MU_SUN", constants::MU_SUN)?;
    module.add("MU_EARTH", constants::MU_EARTH)?;
    module.add("MU_MOON", constants::MU_MOON)?;
    module.add("EARTH_ORBITAL_VELOCITY", constants::EARTH_ORBITAL_VELOCITY)?;
    module.add("EARTH_J2", constants::EARTH_J2)?;
    module.add("EARTH_RADIUS", constants::EARTH_RADIUS)?;
    module.add("DEGREES_TO_RADIANS", constants::DEGREES_TO_RADIANS)?;
    module.add("RADIANS_TO_DEGREES", constants::RADIANS_TO_DEGREES)?;
    module.add("DAY_TO_SECONDS", constants::DAY_TO_SECONDS)?;
    module.add("SECONDS_TO_DAY", constants::SECONDS_TO_DAY)?;
    module.add("JULIAN_YEAR_DAYS", constants::JULIAN_YEAR_DAYS)?;
    module.add("DAYS_TO_JULIAN_YEAR", constants::DAYS_TO_JULIAN_YEAR)?;
    module.add("STANDARD_GRAVITY", constants::STANDARD_GRAVITY)?;
    module.add("CR3BP_MU_EARTH_MOON", constants::CR3BP_MU_EARTH_MOON)?;
    module.add("BCP_MU_EARTH_MOON", constants::BCP_MU_EARTH_MOON)?;
    module.add("BCP_MU_SUN", constants::BCP_MU_SUN)?;
    module.add("BCP_SUN_DISTANCE", constants::BCP_SUN_DISTANCE)?;
    module.add(
        "BCP_SUN_ANGULAR_VELOCITY",
        constants::BCP_SUN_ANGULAR_VELOCITY,
    )?;

    module.add_function(wrap_pyfunction!(jd_to_mjd, module)?)?;
    module.add_function(wrap_pyfunction!(jd_to_mjd2000, module)?)?;
    module.add_function(wrap_pyfunction!(mjd_to_jd, module)?)?;
    module.add_function(wrap_pyfunction!(mjd_to_mjd2000, module)?)?;
    module.add_function(wrap_pyfunction!(mjd2000_to_jd, module)?)?;
    module.add_function(wrap_pyfunction!(mjd2000_to_mjd, module)?)?;
    module.add_function(wrap_pyfunction!(stumpff_c, module)?)?;
    module.add_function(wrap_pyfunction!(stumpff_s, module)?)?;
    module.add_function(wrap_pyfunction!(stumpff_c_batch, module)?)?;
    module.add_function(wrap_pyfunction!(stumpff_s_batch, module)?)?;
    module.add_function(wrap_pyfunction!(elliptic_kepler_residual, module)?)?;
    module.add_function(wrap_pyfunction!(elliptic_kepler_derivative, module)?)?;
    module.add_function(wrap_pyfunction!(elliptic_kepler_second_derivative, module)?)?;
    module.add_function(wrap_pyfunction!(hyperbolic_kepler_residual, module)?)?;
    module.add_function(wrap_pyfunction!(hyperbolic_kepler_derivative, module)?)?;
    module.add_function(wrap_pyfunction!(
        hyperbolic_kepler_second_derivative,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(elliptic_difference_residual, module)?)?;
    module.add_function(wrap_pyfunction!(elliptic_difference_derivative, module)?)?;
    module.add_function(wrap_pyfunction!(
        elliptic_difference_second_derivative,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(hyperbolic_difference_residual, module)?)?;
    module.add_function(wrap_pyfunction!(hyperbolic_difference_derivative, module)?)?;
    module.add_function(wrap_pyfunction!(
        hyperbolic_difference_second_derivative,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(universal_kepler_residual, module)?)?;
    module.add_function(wrap_pyfunction!(universal_kepler_derivative, module)?)?;
    module.add_function(wrap_pyfunction!(
        universal_kepler_second_derivative,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(dot, module)?)?;
    module.add_function(wrap_pyfunction!(dot_batch, module)?)?;
    module.add_function(wrap_pyfunction!(norm, module)?)?;
    module.add_function(wrap_pyfunction!(norm_batch, module)?)?;
    module.add_function(wrap_pyfunction!(normalize, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_batch, module)?)?;
    module.add_function(wrap_pyfunction!(cross, module)?)?;
    module.add_function(wrap_pyfunction!(cross_batch, module)?)?;
    module.add_function(wrap_pyfunction!(skew, module)?)?;
    module.add_function(wrap_pyfunction!(skew_batch, module)?)?;
    Ok(())
}
