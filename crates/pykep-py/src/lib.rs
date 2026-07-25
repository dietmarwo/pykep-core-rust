//! Thin PyO3 bindings for `pykep-core`.

use pyo3::prelude::*;

mod elements;
mod ephemeris;
mod error;
mod foundations;
mod mission;
mod propagation;
mod time_anomalies;

/// Return the implementation status of the native core.
#[pyfunction]
fn port_status() -> &'static str {
    pykep_core::PORT_STATUS
}

/// Native extension module for the `pykep_rust` Python package.
#[pymodule]
fn _pykep_rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(port_status, module)?)?;
    error::register(module)?;
    foundations::register(module)?;
    elements::register(module)?;
    ephemeris::register(module)?;
    mission::register(module)?;
    propagation::register(module)?;
    time_anomalies::register(module)?;
    Ok(())
}
