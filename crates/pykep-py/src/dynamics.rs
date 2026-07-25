// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

use crate::error::to_python;
use pykep_core::dynamics::zoh::{
    ControlSchedule, ZohCr3bpDynamics, ZohEquinoctialDynamics, ZohKeplerDynamics,
    ZohSolarSailDynamics, propagate_schedule, propagate_schedule_backward,
};
use pykep_core::dynamics::{BcpDynamics, Cr3bpDynamics, KeplerDynamics};
use pykep_core::integration::{DynamicsModel, IntegratorOptions};
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

fn seven(values: Vec<f64>) -> Result<[f64; 7], PykepError> {
    values
        .try_into()
        .map_err(|values: Vec<f64>| PykepError::DimensionMismatch {
            expected: 7,
            actual: values.len(),
        })
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
    let schedule = ControlSchedule::new(boundaries, control_rows(controls).map_err(to_python)?)
        .map_err(to_python)?;
    let result = if backward {
        propagate_schedule_backward(model, &schedule, state, constants, integrator_options)
    } else {
        propagate_schedule(model, &schedule, state, constants, integrator_options)
    };
    result
        .map(|propagation| propagation.state)
        .map_err(to_python)
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
    module.add_function(wrap_pyfunction!(zoh_kepler_rhs, module)?)?;
    module.add_function(wrap_pyfunction!(zoh_cr3bp_rhs, module)?)?;
    module.add_function(wrap_pyfunction!(zoh_equinoctial_rhs, module)?)?;
    module.add_function(wrap_pyfunction!(zoh_solar_sail_rhs, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_zoh_kepler, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_zoh_cr3bp, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_zoh_equinoctial, module)?)?;
    module.add_function(wrap_pyfunction!(propagate_zoh_solar_sail, module)?)?;
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
