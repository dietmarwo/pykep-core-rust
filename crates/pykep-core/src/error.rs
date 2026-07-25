// Copyright (c) 2026 pykep-rust contributors
// SPDX-License-Identifier: MPL-2.0

//! Error types shared by the numerical core.

use core::fmt;

/// Error returned by a pykep numerical operation.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PykepError {
    /// A finite input lies outside the mathematical domain.
    InvalidInput {
        /// Public parameter name.
        parameter: &'static str,
        /// Human-readable domain requirement.
        reason: String,
    },
    /// A public input is NaN or infinite.
    NonFiniteInput {
        /// Public parameter name.
        parameter: &'static str,
    },
    /// The requested operation is undefined for the supplied geometry.
    SingularGeometry {
        /// Operation that detected the singularity.
        operation: &'static str,
    },
    /// An iterative algorithm exhausted its iteration limit.
    ConvergenceFailure {
        /// Iterative operation that failed.
        operation: &'static str,
        /// Number of iterations attempted.
        iterations: usize,
    },
    /// A dynamically sized value has an invalid length.
    DimensionMismatch {
        /// Required number of scalar values.
        expected: usize,
        /// Supplied number of scalar values.
        actual: usize,
    },
    /// An ephemeris or backend does not implement a requested capability.
    UnsupportedCapability {
        /// Provider or backend name.
        provider: String,
        /// Unsupported capability name.
        capability: &'static str,
    },
    /// A finite input produced a value outside the binary64 range.
    NumericalOverflow {
        /// Numerical operation that overflowed.
        operation: &'static str,
    },
    /// A numerical integrator could not complete a propagation.
    IntegrationFailure {
        /// Dynamics model being integrated.
        model: &'static str,
        /// Human-readable failure context.
        reason: String,
    },
    /// A required embedded or external dataset is unavailable or corrupt.
    DataUnavailable {
        /// Stable dataset name.
        dataset: &'static str,
    },
}

impl fmt::Display for PykepError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { parameter, reason } => {
                write!(formatter, "invalid {parameter}: {reason}")
            }
            Self::NonFiniteInput { parameter } => {
                write!(formatter, "{parameter} must be finite")
            }
            Self::SingularGeometry { operation } => {
                write!(formatter, "singular geometry in {operation}")
            }
            Self::ConvergenceFailure {
                operation,
                iterations,
            } => write!(
                formatter,
                "{operation} did not converge after {iterations} iterations"
            ),
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    formatter,
                    "dimension mismatch: expected {expected}, got {actual}"
                )
            }
            Self::UnsupportedCapability {
                provider,
                capability,
            } => write!(formatter, "{provider} does not support {capability}"),
            Self::NumericalOverflow { operation } => {
                write!(formatter, "floating-point overflow in {operation}")
            }
            Self::IntegrationFailure { model, reason } => {
                write!(formatter, "integration of {model} failed: {reason}")
            }
            Self::DataUnavailable { dataset } => {
                write!(formatter, "required dataset is unavailable: {dataset}")
            }
        }
    }
}

impl std::error::Error for PykepError {}

/// Result returned by fallible pykep operations.
pub type Result<T> = core::result::Result<T, PykepError>;

pub(crate) fn ensure_finite(parameter: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(PykepError::NonFiniteInput { parameter })
    }
}

pub(crate) fn ensure_finite_values(names_and_values: &[(&'static str, f64)]) -> Result<()> {
    for &(name, value) in names_and_values {
        ensure_finite(name, value)?;
    }
    Ok(())
}

pub(crate) fn ensure_finite_output(operation: &'static str, value: f64) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PykepError::NumericalOverflow { operation })
    }
}

#[cfg(test)]
mod tests {
    use super::PykepError;

    #[test]
    fn errors_have_stable_useful_messages() {
        let error = PykepError::DimensionMismatch {
            expected: 3,
            actual: 2,
        };
        assert_eq!(error.to_string(), "dimension mismatch: expected 3, got 2");
    }
}
