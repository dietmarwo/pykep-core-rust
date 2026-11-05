// SPDX-License-Identifier: MPL-2.0

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, ndarray::Array2};
use pykep_core::integration::IntegrationMethod;
use pykep_trajopt::{GymProblem, Launcher, Problem, TopsOptions};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::error::to_python;

#[pyclass(module = "pykep_rust._pykep_rust")]
struct NativeGym {
    problem: GymProblem,
}

fn json<T: serde::Serialize>(value: &T) -> PyResult<String> {
    serde_json::to_string(value).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pymethods]
impl NativeGym {
    fn spec_json(&self) -> PyResult<String> {
        json(self.problem.spec())
    }

    fn fitness(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<Vec<f64>> {
        py.detach(|| self.problem.evaluate(&x).map(|value| value.fitness()))
            .map_err(to_python)
    }

    fn gradient(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<Vec<f64>> {
        py.detach(|| {
            self.problem
                .gradient(&x)
                .map(|rows| rows.into_iter().flatten().collect())
        })
        .map_err(to_python)
    }

    fn evaluate_json(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<String> {
        json(&py.detach(|| self.problem.evaluate(&x)).map_err(to_python)?)
    }

    fn decode_json(&self, py: Python<'_>, x: Vec<f64>, samples: usize) -> PyResult<String> {
        json(
            &py.detach(|| self.problem.decode(&x, samples))
                .map_err(to_python)?,
        )
    }

    fn forward_residual(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<Vec<f64>> {
        py.detach(|| self.problem.forward_residual(&x))
            .map_err(to_python)
    }

    fn fitness_batch<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        workers: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let view = x.as_array();
        let (n, d) = view.dim();
        let spec = self.problem.spec();
        if d != spec.bounds.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} columns, got {d}",
                spec.bounds.len()
            )));
        }
        let m = spec.objectives.len() + spec.equalities.len() + spec.inequalities.len();
        let items: Vec<_> = view.rows().into_iter().map(|row| row.to_vec()).collect();
        let values = py
            .detach(|| {
                self.problem
                    .evaluate_batch(&items, workers)
                    .map(|v| v.into_iter().flat_map(|r| r.fitness()).collect())
            })
            .map_err(to_python)?;
        let matrix = Array2::from_shape_vec((n, m), values)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(matrix.into_pyarray(py))
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature=(id, nseg=None, cut=0.5, time_encoding="uniform", direction_inequalities=None, backend="taylor",
    tolerance=1e-13, derivative_tolerance=1e-13, maximum_steps=None))]
fn _gym_problem(
    id: &str,
    nseg: Option<usize>,
    cut: f64,
    time_encoding: &str,
    direction_inequalities: Option<bool>,
    backend: &str,
    tolerance: f64,
    derivative_tolerance: f64,
    maximum_steps: Option<usize>,
) -> PyResult<NativeGym> {
    let method = match backend {
        "taylor" => IntegrationMethod::Taylor,
        "dop853" => IntegrationMethod::Dop853,
        _ => return Err(PyValueError::new_err("expected taylor or dop853")),
    };
    let problem = GymProblem::new(
        id,
        TopsOptions {
            segments: nseg,
            cut,
            time_encoding: time_encoding.parse().map_err(to_python)?,
            direction_inequalities,
            method,
            tolerance,
            derivative_tolerance,
            maximum_steps,
        },
    )
    .map_err(to_python)?;
    Ok(NativeGym { problem })
}

#[pyfunction]
fn _gym_ids() -> PyResult<Vec<String>> {
    GymProblem::ids().map_err(to_python)
}

#[pyfunction]
fn _launcher_mass(name: &str, speed_kms: f64, declination_deg: f64) -> PyResult<f64> {
    Launcher::ALL
        .into_iter()
        .find(|table| table.id() == name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown launcher {name}")))?
        .mass(speed_kms, declination_deg)
        .map_err(to_python)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeGym>()?;
    module.add_function(wrap_pyfunction!(_gym_problem, module)?)?;
    module.add_function(wrap_pyfunction!(_gym_ids, module)?)?;
    module.add_function(wrap_pyfunction!(_launcher_mass, module)?)?;
    Ok(())
}
