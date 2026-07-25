// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

use pykep_core::PykepError as CoreError;
use pyo3::create_exception;
use pyo3::exceptions::{PyOverflowError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

create_exception!(
    _pykep_rust,
    PykepError,
    PyRuntimeError,
    "Base exception for pykep-rust numerical failures."
);
create_exception!(
    _pykep_rust,
    ConvergenceError,
    PykepError,
    "An iterative numerical algorithm did not converge."
);
create_exception!(
    _pykep_rust,
    SingularGeometryError,
    PykepError,
    "The supplied geometry is singular for the requested operation."
);
create_exception!(
    _pykep_rust,
    UnsupportedCapabilityError,
    PykepError,
    "A provider or backend does not implement a requested capability."
);
create_exception!(
    _pykep_rust,
    IntegrationError,
    PykepError,
    "A numerical integration could not be completed."
);

pub(crate) fn to_python(error: CoreError) -> PyErr {
    match error {
        CoreError::InvalidInput { .. }
        | CoreError::NonFiniteInput { .. }
        | CoreError::DimensionMismatch { .. } => PyValueError::new_err(error.to_string()),
        CoreError::SingularGeometry { .. } => SingularGeometryError::new_err(error.to_string()),
        CoreError::ConvergenceFailure { .. } => ConvergenceError::new_err(error.to_string()),
        CoreError::UnsupportedCapability { .. } => {
            UnsupportedCapabilityError::new_err(error.to_string())
        }
        CoreError::NumericalOverflow { .. } => PyOverflowError::new_err(error.to_string()),
        CoreError::IntegrationFailure { .. } => IntegrationError::new_err(error.to_string()),
        _ => PykepError::new_err(error.to_string()),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let python = module.py();
    module.add("PykepError", python.get_type::<PykepError>())?;
    module.add("ConvergenceError", python.get_type::<ConvergenceError>())?;
    module.add(
        "SingularGeometryError",
        python.get_type::<SingularGeometryError>(),
    )?;
    module.add(
        "UnsupportedCapabilityError",
        python.get_type::<UnsupportedCapabilityError>(),
    )?;
    module.add("IntegrationError", python.get_type::<IntegrationError>())?;
    Ok(())
}
