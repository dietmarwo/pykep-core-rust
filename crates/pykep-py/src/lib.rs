//! Thin PyO3 bindings for `pykep-core`.

use pyo3::prelude::*;

/// Return the implementation status of the native core.
#[pyfunction]
fn port_status() -> &'static str {
    pykep_core::PORT_STATUS
}

/// Native extension module for the `pykep_rust` Python package.
#[pymodule]
fn _pykep_rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(port_status, module)?)?;
    Ok(())
}
