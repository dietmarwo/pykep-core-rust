// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

use crate::error::to_python;
use pykep_core::dynamics::{BcpDynamics, Cr3bpDynamics, KeplerDynamics};
use pykep_core::integration::IntegratorOptions;
use pykep_core::{CartesianState, Matrix6, PykepError};
use pyo3::prelude::*;

fn six(values: Vec<f64>) -> Result<CartesianState, PykepError> {
    values
        .try_into()
        .map_err(|values: Vec<f64>| PykepError::DimensionMismatch {
            expected: 6,
            actual: values.len(),
        })
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

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(kepler_rhs, module)?)?;
    module.add_function(wrap_pyfunction!(cr3bp_rhs, module)?)?;
    module.add_function(wrap_pyfunction!(bcp_rhs, module)?)?;
    module.add_function(wrap_pyfunction!(cr3bp_effective_potential, module)?)?;
    module.add_function(wrap_pyfunction!(cr3bp_jacobi_constant, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_kepler_dynamics, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_cr3bp, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_bcp, module)?)?;
    module.add_function(wrap_pyfunction!(
        propagate_kepler_dynamics_with_stm,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(propagate_cr3bp_with_stm, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_bcp_with_stm, module)?)?;
    Ok(())
}
