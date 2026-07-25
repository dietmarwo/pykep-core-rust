#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]

/// Physical and astrodynamical constants.
pub mod constants;
/// Stable error categories returned by numerical APIs.
pub mod error;
/// Small, dependency-free numerical helpers.
pub mod math;
/// Time representations and conversions.
pub mod time;
/// Fixed-shape numerical types used throughout the crate.
pub mod types;

pub use error::{PykepError, Result};
pub use types::{CartesianState, Elements6, Matrix3, Matrix6, Vector3};

/// Current implementation status exposed by both the Rust and Python smoke
/// tests.
pub const PORT_STATUS: &str = "phase 2: numerical foundations implemented";

#[cfg(test)]
mod tests {
    use super::PORT_STATUS;

    #[test]
    fn status_reports_foundations() {
        assert!(PORT_STATUS.contains("foundations"));
    }
}
